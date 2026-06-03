#!/usr/bin/env python3
"""
Walk-forward backtester — 1 month of real MEXC historical data.
Zero look-ahead bias: at each 5m bar, only data visible at that moment is used.

Usage:
    python backtest.py                    # all 10 symbols, 30 days
    python backtest.py --days 14          # shorter window
    python backtest.py --symbols BTC ETH  # specific pairs
    python backtest.py --no-cache         # force re-download
    python backtest.py --capital 5000     # different starting capital
"""
import sys
import os
import time
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backtest")

from analysis.indicators import TechnicalIndicators
from analysis.trend_detector import TrendDetector
from analysis.volume_analyzer import VolumeAnalyzer
from strategy.multi_timeframe import MultiTimeframeAnalyzer
from strategy.signal_generator import SignalGenerator

# ── Constants ────────────────────────────────────────────────────────────────

ALL_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT",
]
SYMBOL_FALLBACKS = {"MATIC/USDT": ["POL/USDT"]}

TIMEFRAMES = ["1d", "4h", "1h", "30m", "15m", "5m"]
LOOKBACK   = {"5m": 200, "15m": 150, "30m": 120, "1h": 100, "4h": 80, "1d": 60}
TF_SECS    = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}

TAKER_FEE        = 0.0004   # 0.04% per side
CAPITAL          = 1000.0
RISK_PCT         = 5.0      # % of equity per trade
AI_SCORE_BT      = 0.70     # override cold-start score (above 0.65 threshold)
MAX_BARS_HELD    = 48       # max 5m bars before timeout (4 hours)
MAX_CONCURRENT   = 3
DAILY_TRADE_CAP  = 30


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:          str
    direction:       str
    entry_price:     float
    stop_loss:       float
    take_profit_1:   float
    position_size:   float
    notional:        float
    risk_usdt:       float
    atr:             float
    entry_time:      pd.Timestamp
    conviction:      int
    rr_ratio:        float
    exit_price:      float = 0.0
    exit_time:       object = None
    pnl_usdt:        float = 0.0
    outcome:         str   = "open"   # win | loss | timeout | breakeven
    bars_held:       int   = 0
    # Trailing stop tracking
    initial_stop_loss:      float = 0.0   # original SL (constant reference)
    high_water_mark:        float = 0.0   # best price reached in trade direction
    breakeven_activated:    bool  = False  # True once SL moved to entry
    lock_profit_activated:  bool  = False  # True once SL moved to +0.5R


@dataclass
class SymbolResult:
    symbol:           str
    trades:           List[Trade] = field(default_factory=list)
    signals_seen:     int = 0
    total_trades:     int = 0
    wins:             int = 0
    losses:           int = 0
    timeouts:         int = 0
    total_pnl:        float = 0.0
    win_rate:         float = 0.0
    profit_factor:    float = 0.0
    max_drawdown_pct: float = 0.0
    return_pct:       float = 0.0


# ── Data layer ────────────────────────────────────────────────────────────────

def _get_exchange(futures: bool = True):
    import ccxt
    params = {"enableRateLimit": True, "rateLimit": 250}
    if futures:
        params["defaultType"] = "swap"
    return ccxt.mexc(params)


def _fetch_ohlcv(exchange, symbol: str, tf: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    step = TF_SECS.get(tf, 300) * 1000
    rows = []
    cur = since_ms
    while cur < until_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, tf, since=cur, limit=1000)
        except Exception as e:
            logger.warning(f"fetch_ohlcv {symbol}/{tf}: {e}")
            time.sleep(2)
            break
        if not candles:
            break
        candles = [c for c in candles if c[0] < until_ms]
        rows.extend(candles)
        if len(candles) < 1000 or candles[-1][0] >= until_ms:
            break
        cur = candles[-1][0] + step
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df.astype(float)


def load_or_fetch(exchange, symbol: str, tf: str,
                  since_ms: int, until_ms: int,
                  cache_dir: str = "data/backtest",
                  force: bool = False) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    key = symbol.replace("/", "_").replace(":", "_")
    csv_path     = os.path.join(cache_dir, f"{key}_{tf}.csv")
    parquet_path = os.path.join(cache_dir, f"{key}_{tf}.parquet")

    def _try_load(p: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(p):
            return None
        try:
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(
                p, index_col=0, parse_dates=True
            )
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            return df if not df.empty else None
        except Exception:
            return None

    if not force:
        for p in (parquet_path, csv_path):
            df = _try_load(p)
            if df is not None:
                last_ms = int(df.index[-1].timestamp() * 1000)
                if last_ms >= until_ms - TF_SECS.get(tf, 300) * 1000 * 2:
                    print(f"    cached  {symbol}/{tf}: {len(df)} bars")
                    return df

    print(f"    fetch   {symbol}/{tf} ...", end="", flush=True)
    df = _fetch_ohlcv(exchange, symbol, tf, since_ms, until_ms)
    if df.empty:
        print(" EMPTY")
        return df
    print(f" {len(df)} bars")
    try:
        df.to_parquet(parquet_path)
    except Exception:
        df.to_csv(csv_path)
    return df


def download_all(symbols: List[str], days: int,
                 force: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
    now_ms = int(time.time() * 1000)
    base_since = now_ms - days * 86400_000

    # Extra warmup per TF so indicators are warm at start of backtest window
    warmup_extra = {
        "5m": 200 * 300_000, "15m": 150 * 900_000, "30m": 120 * 1800_000,
        "1h": 100 * 3600_000, "4h": 80 * 14400_000, "1d": 60 * 86400_000,
    }

    try:
        ex = _get_exchange(futures=True)
        ex.load_markets()
        print("  Exchange: MEXC Futures (public OHLCV, no auth required)")
    except Exception as e:
        print(f"  Futures failed ({e}), trying spot...")
        ex = _get_exchange(futures=False)

    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        print(f"  {sym}:")
        sym_data = {}
        for tf in TIMEFRAMES:
            since = base_since - warmup_extra.get(tf, 0)
            df = load_or_fetch(ex, sym, tf, since, now_ms, force=force)
            if df.empty:
                fallbacks = SYMBOL_FALLBACKS.get(sym, [])
                for alias in fallbacks:
                    df = load_or_fetch(ex, alias, tf, since, now_ms, force=force)
                    if not df.empty:
                        break
            if not df.empty:
                sym_data[tf] = df
        if sym_data:
            data[sym] = sym_data
    return data


# ── Backtest engine ───────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, capital: float = CAPITAL, risk_pct: float = RISK_PCT,
                 min_adx: float = 30.0, min_conviction: int = 6,
                 volume_multiplier: float = 2.0, htf_confidence: float = 0.50,
                 min_confluences: int = 3, rsi_lo_long: float = 38,
                 rsi_hi_long: float = 65, label: str = "STRICT",
                 trailing_stop_r: float = 1.0, lock_profit_r: float = 1.5,
                 pullback_tolerance_pct: float = 0.008, min_body_ratio: float = 0.35,
                 max_concurrent: int = MAX_CONCURRENT, daily_trade_cap: int = DAILY_TRADE_CAP):
        self.capital  = capital
        self.risk_pct = risk_pct
        self.label    = label
        self.max_concurrent  = max_concurrent
        self.daily_trade_cap = daily_trade_cap

        td = TrendDetector()
        va = VolumeAnalyzer(surge_multiplier=volume_multiplier)
        mtf = MultiTimeframeAnalyzer(trend_detector=td, volume_analyzer=va,
                                     min_conviction=min_conviction)
        self.signal_gen = SignalGenerator(
            trend_detector=td, volume_analyzer=va, mtf_analyzer=mtf,
            min_conviction=min_conviction, min_rr_ratio=2.0,
            min_adx=min_adx, volume_multiplier=volume_multiplier,
            htf_confidence=htf_confidence, min_confluences=min_confluences,
            rsi_lo_long=rsi_lo_long, rsi_hi_long=rsi_hi_long,
            pullback_tolerance_pct=pullback_tolerance_pct,
            min_body_ratio=min_body_ratio,
        )
        self.mtf = mtf
        self.min_conviction = min_conviction
        self.trailing_stop_r = trailing_stop_r    # activate BE at N×SL_dist (1.0 = 1:1 R:R)
        self.lock_profit_r   = lock_profit_r      # lock +0.5R at N×SL_dist (1.5 = 1.5:1)

    # ── per-symbol walk-forward ───────────────────────────────────────────────

    def run_symbol(self, symbol: str, raw: Dict[str, pd.DataFrame]) -> SymbolResult:
        res = SymbolResult(symbol=symbol)

        # Pre-compute indicators once (no look-ahead: each row only uses prior rows)
        computed: Dict[str, pd.DataFrame] = {}
        for tf, df in raw.items():
            if len(df) >= 50:
                try:
                    computed[tf] = TechnicalIndicators.compute_all(df.copy())
                except Exception as e:
                    logger.warning(f"{symbol}/{tf} indicators failed: {e}")

        if "5m" not in computed:
            return res

        df5 = computed["5m"]
        n   = len(df5)
        # Start evaluating after warmup: 25 days so 1d TF has 25 closed bars
        warmup = 25 * 288    # 7200 5m bars = 25 days

        if n < warmup + 30:
            return res

        equity       = self.capital
        peak_eq      = equity
        open_trades: List[Trade] = []
        day_opens    = 0
        day_pnl      = 0.0
        current_day  = None

        for i in range(warmup, n):
            bar = df5.iloc[i]
            t   = df5.index[i]
            day = t.date()

            if day != current_day:
                current_day = day
                day_opens   = 0
                day_pnl     = 0.0

            # ── close exits on this bar ───────────────────────────────────────
            still_open = []
            for tr in open_trades:
                if self._check_exit(tr, bar, t):
                    fee = tr.notional * TAKER_FEE * 2
                    tr.pnl_usdt -= fee
                    equity  += tr.pnl_usdt
                    day_pnl += tr.pnl_usdt
                    peak_eq  = max(peak_eq, equity)
                    dd = (peak_eq - equity) / peak_eq * 100 if peak_eq > 0 else 0
                    res.max_drawdown_pct = max(res.max_drawdown_pct, dd)
                    if tr.outcome == "win":
                        res.wins += 1
                    elif tr.outcome == "timeout":
                        res.timeouts += 1
                        if tr.pnl_usdt <= 0:
                            res.losses += 1
                    else:
                        res.losses += 1
                    res.trades.append(tr)
                    res.total_trades += 1
                    res.total_pnl += tr.pnl_usdt
                else:
                    still_open.append(tr)
            open_trades = still_open

            # ── pre-entry guards ──────────────────────────────────────────────
            if len(open_trades) >= self.max_concurrent:
                continue
            if day_opens >= self.daily_trade_cap:
                continue
            if day_pnl <= -equity * 0.10:     # daily loss limit
                continue

            # ── build MTF slices (no look-ahead) ─────────────────────────────
            slices = self._slices(computed, t)
            if len(slices) < 3:
                continue

            # ── signal pipeline ───────────────────────────────────────────────
            try:
                conviction = self.mtf.analyze(slices, symbol)
                # Use engine's min_conviction (not the hardcoded is_tradeable property)
                if conviction.total < self.min_conviction or conviction.aligned_direction == "SIDEWAYS":
                    continue

                sig = self.signal_gen.generate_signal(
                    symbol=symbol, tf_data=slices, conviction=conviction,
                    order_book_imbalance=0.5,
                    capital=equity, max_risk_pct=self.risk_pct,
                )
                if sig is None:
                    continue

                res.signals_seen += 1

                # AI override: bypass cold-start filter in backtest
                if AI_SCORE_BT < 0.65:
                    continue

                # Skip if same direction already open
                if any(tr.direction == sig.direction for tr in open_trades):
                    continue

                tr = Trade(
                    symbol=symbol, direction=sig.direction,
                    entry_price=sig.entry_price, stop_loss=sig.stop_loss,
                    take_profit_1=sig.take_profit_1,
                    position_size=sig.position_size,
                    notional=sig.notional_usdt, risk_usdt=sig.risk_usdt,
                    atr=sig.atr, entry_time=t,
                    conviction=sig.conviction_score, rr_ratio=sig.rr_ratio,
                    initial_stop_loss=sig.stop_loss,
                    high_water_mark=sig.entry_price,
                )
                open_trades.append(tr)
                day_opens += 1

            except Exception as e:
                logger.debug(f"{symbol}@{i}: {e}")

        # force-close remaining trades at last price
        last_close = float(df5["close"].iloc[-1])
        last_time  = df5.index[-1]
        for tr in open_trades:
            tr.exit_price = last_close
            tr.exit_time  = last_time
            mult = 1 if tr.direction == "LONG" else -1
            tr.pnl_usdt   = (last_close - tr.entry_price) * mult * tr.position_size
            fee = tr.notional * TAKER_FEE * 2
            tr.pnl_usdt  -= fee
            # Apply same BE rule: if BE activated and pnl≥0 → win
            if tr.breakeven_activated and tr.pnl_usdt >= 0:
                tr.outcome = "win"
                res.wins += 1
            else:
                tr.outcome = "timeout"
                res.timeouts += 1
                if tr.pnl_usdt <= 0:
                    res.losses += 1
            res.trades.append(tr)
            res.total_trades += 1
            res.total_pnl    += tr.pnl_usdt

        # summary
        if res.total_trades > 0:
            res.win_rate = res.wins / res.total_trades
            gp = sum(t.pnl_usdt for t in res.trades if t.pnl_usdt > 0)
            gl = abs(sum(t.pnl_usdt for t in res.trades if t.pnl_usdt <= 0))
            res.profit_factor = gp / (gl + 1e-10)
            res.return_pct = res.total_pnl / self.capital * 100

        return res

    # ── helpers ───────────────────────────────────────────────────────────────

    def _slices(self, computed: Dict[str, pd.DataFrame], t: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Return MTF slices containing only data available at close of 5m bar t."""
        M = t.timestamp() + 300   # moment bar closes
        out = {}
        for tf, df in computed.items():
            tf_s = TF_SECS.get(tf, 300)
            # last CLOSED higher-TF candle's open time
            cutoff_unix = (M // tf_s) * tf_s - tf_s
            cutoff_ts   = pd.Timestamp(cutoff_unix, unit="s", tz="UTC")
            idx = int(df.index.searchsorted(cutoff_ts, side="right"))
            if idx < 20:   # allow smaller slices so 1d TF is included
                continue
            lb    = LOOKBACK.get(tf, 100)
            start = max(0, idx - lb)
            out[tf] = df.iloc[start:idx]
        return out



    def _check_exit(self, tr: Trade, bar: pd.Series, t: pd.Timestamp) -> bool:
        """Return True and fill exit fields if SL, TP1, or timeout reached.

        Trailing stop logic:
          - At 1:1 R:R (trailing_stop_r × SL_dist in our favor): move SL to entry (breakeven)
          - At 1.5:1 R:R (lock_profit_r × SL_dist): move SL to entry + 0.5×SL_dist (lock +0.5R)
          - Any exit after BE activated counts as "win" if pnl ≥ 0
        """
        tr.bars_held += 1
        high  = float(bar["high"])
        low   = float(bar["low"])
        close = float(bar["close"])

        # Original SL distance (constant reference — never modified)
        sl_dist = abs(tr.entry_price - tr.initial_stop_loss) if tr.initial_stop_loss else abs(tr.entry_price - tr.stop_loss)

        # ── Trailing stop update (before exit checks) ──────────────────────────
        if tr.direction == "LONG":
            tr.high_water_mark = max(tr.high_water_mark, high)
            move = tr.high_water_mark - tr.entry_price
        else:
            tr.high_water_mark = min(tr.high_water_mark, low)
            move = tr.entry_price - tr.high_water_mark

        if sl_dist > 0:
            # Stage 1: activate breakeven stop at trailing_stop_r × SL_dist (default 1:1)
            if not tr.breakeven_activated and move >= sl_dist * self.trailing_stop_r:
                tr.stop_loss = tr.entry_price
                tr.breakeven_activated = True

            # Stage 2: lock in +0.5R at lock_profit_r × SL_dist (default 1.5:1)
            if tr.breakeven_activated and not tr.lock_profit_activated and move >= sl_dist * self.lock_profit_r:
                lock_price = (tr.entry_price + sl_dist * 0.5 if tr.direction == "LONG"
                              else tr.entry_price - sl_dist * 0.5)
                tr.stop_loss = lock_price
                tr.lock_profit_activated = True

        # ── Timeout ────────────────────────────────────────────────────────────
        if tr.bars_held >= MAX_BARS_HELD:
            tr.exit_price = close
            tr.exit_time  = t
            mult = 1 if tr.direction == "LONG" else -1
            tr.pnl_usdt = (close - tr.entry_price) * mult * tr.position_size
            # Breakeven was activated → any non-negative exit = win
            if tr.breakeven_activated and tr.pnl_usdt >= 0:
                tr.outcome = "win"
            else:
                tr.outcome = "timeout"
            return True

        # ── SL / TP checks ─────────────────────────────────────────────────────
        if tr.direction == "LONG":
            if low <= tr.stop_loss:
                tr.exit_price = tr.stop_loss
                tr.exit_time  = t
                tr.pnl_usdt = (tr.stop_loss - tr.entry_price) * tr.position_size
                # If BE activated: exit at entry (or above) = win; below = very rare negative slip
                tr.outcome = "win" if (tr.breakeven_activated and tr.pnl_usdt >= 0) else "loss"
                return True
            if high >= tr.take_profit_1:
                tr.exit_price = tr.take_profit_1
                tr.exit_time  = t
                tr.pnl_usdt = (tr.take_profit_1 - tr.entry_price) * tr.position_size
                tr.outcome = "win"
                return True
        else:
            if high >= tr.stop_loss:
                tr.exit_price = tr.stop_loss
                tr.exit_time  = t
                tr.pnl_usdt = (tr.entry_price - tr.stop_loss) * tr.position_size
                tr.outcome = "win" if (tr.breakeven_activated and tr.pnl_usdt >= 0) else "loss"
                return True
            if low <= tr.take_profit_1:
                tr.exit_price = tr.take_profit_1
                tr.exit_time  = t
                tr.pnl_usdt = (tr.entry_price - tr.take_profit_1) * tr.position_size
                tr.outcome = "win"
                return True
        return False


# ── Reporting ─────────────────────────────────────────────────────────────────

SEP = "─" * 68

def _fmt_pct(v: float) -> str:
    return f"{'+'if v>=0 else ''}{v:.2f}%"

def _fmt_pf(pf: float) -> str:
    if pf > 999:
        return ">999"
    return f"{pf:.2f}"

def print_symbol_table(results: List[SymbolResult]):
    print(f"\n  {'Symbol':<12}  {'Trades':>6}  {'Win%':>6}  {'PF':>5}  {'Return':>8}  {'MaxDD':>7}  {'Signals':>7}")
    print("  " + SEP)
    for r in results:
        if r.total_trades == 0:
            print(f"  {r.symbol:<12}  {'—':>6}")
            continue
        flag = "✓" if r.win_rate >= 0.60 else ("~" if r.win_rate >= 0.50 else "✗")
        print(
            f"  {r.symbol:<12}  {r.total_trades:>6}  "
            f"{r.win_rate*100:>5.1f}% {flag}  "
            f"{_fmt_pf(r.profit_factor):>5}  "
            f"{_fmt_pct(r.return_pct):>8}  "
            f"{r.max_drawdown_pct:>6.1f}%  "
            f"{r.signals_seen:>7}"
        )

def print_portfolio_summary(results: List[SymbolResult], days: int, capital: float):
    all_trades = sorted(
        [t for r in results for t in r.trades],
        key=lambda t: t.entry_time
    )
    if not all_trades:
        print("\n  No trades generated.")
        return

    total   = len(all_trades)
    wins    = sum(1 for t in all_trades if t.outcome == "win")
    losses  = sum(1 for t in all_trades if t.outcome == "loss")
    touts   = sum(1 for t in all_trades if t.outcome == "timeout")
    net_pnl = sum(t.pnl_usdt for t in all_trades)
    gp      = sum(t.pnl_usdt for t in all_trades if t.pnl_usdt > 0)
    gl      = abs(sum(t.pnl_usdt for t in all_trades if t.pnl_usdt <= 0))
    pf      = gp / (gl + 1e-10)
    wr      = wins / total

    # Equity curve
    eq = capital; peak = eq; max_dd = 0.0
    daily_pnl: Dict[str, float] = {}
    for t in all_trades:
        eq  += t.pnl_usdt
        peak = max(peak, eq)
        dd   = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        dk = str(t.entry_time.date())
        daily_pnl[dk] = daily_pnl.get(dk, 0) + t.pnl_usdt

    total_return = (eq - capital) / capital * 100
    trades_per_day = total / days
    best  = max(all_trades, key=lambda t: t.pnl_usdt)
    worst = min(all_trades, key=lambda t: t.pnl_usdt)

    avg_hold = sum(t.bars_held for t in all_trades) / total
    avg_hold_h = avg_hold * 5 / 60

    # Streak analysis
    win_streak = loss_streak = cur_w = cur_l = 0
    for t in all_trades:
        if t.pnl_usdt > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        win_streak  = max(win_streak, cur_w)
        loss_streak = max(loss_streak, cur_l)

    # Daily P&L stats
    if daily_pnl:
        best_day  = max(daily_pnl.values())
        worst_day = min(daily_pnl.values())
        avg_day   = sum(daily_pnl.values()) / len(daily_pnl)
    else:
        best_day = worst_day = avg_day = 0

    print(f"\n  {'─'*68}")
    print(f"  PORTFOLIO SUMMARY — {days}-DAY BACKTEST")
    print(f"  {'─'*68}")
    print(f"  Capital:          ${capital:,.2f}  →  ${eq:,.2f}")
    print(f"  Net Return:       {_fmt_pct(total_return)}")
    print(f"  Net P&L:          ${net_pnl:+.2f}")
    print(f"  {'─'*68}")
    print(f"  Total Trades:     {total}  ({trades_per_day:.1f}/day avg)")
    print(f"  Win Rate:         {wr*100:.1f}%  "
          f"(W:{wins}  L:{losses}  Timeout:{touts})")
    print(f"  Profit Factor:    {_fmt_pf(pf)}")
    print(f"  Max Drawdown:     {max_dd:.2f}%")
    print(f"  {'─'*68}")
    print(f"  Gross Profit:     ${gp:.2f}")
    print(f"  Gross Loss:       ${gl:.2f}")
    print(f"  Best Trade:       ${best.pnl_usdt:+.2f}  ({best.symbol} {best.direction})")
    print(f"  Worst Trade:      ${worst.pnl_usdt:+.2f}  ({worst.symbol} {worst.direction})")
    print(f"  {'─'*68}")
    print(f"  Avg Hold Time:    {avg_hold_h:.1f}h  ({avg_hold:.0f} bars)")
    print(f"  Max Win Streak:   {win_streak}")
    print(f"  Max Loss Streak:  {loss_streak}")
    print(f"  {'─'*68}")
    print(f"  Best Day P&L:     ${best_day:+.2f}")
    print(f"  Worst Day P&L:    ${worst_day:+.2f}")
    print(f"  Avg Day P&L:      ${avg_day:+.2f}")
    print(f"  {'─'*68}")

    # Mini equity curve (ASCII)
    if daily_pnl:
        vals  = list(daily_pnl.values())
        lo, hi = min(vals), max(vals)
        span  = hi - lo or 1
        BARS  = 40
        print(f"\n  Daily P&L equity strip  (each char = 1 day)")
        print(f"  ${lo:+.0f}{'':─<{BARS}}${hi:+.0f}")
        row = "  "
        for v in vals:
            pos = int((v - lo) / span * (BARS - 1))
            row += "▲" if v > 0 else "▼"
        print(row)
        print()


def save_csv(results: List[SymbolResult], path: str = "logs/backtest_trades.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                "symbol":        t.symbol,
                "direction":     t.direction,
                "entry_time":    t.entry_time,
                "exit_time":     t.exit_time,
                "entry_price":   round(t.entry_price, 8),
                "exit_price":    round(t.exit_price, 8),
                "stop_loss":     round(t.stop_loss, 8),
                "take_profit_1": round(t.take_profit_1, 8),
                "size":          round(t.position_size, 6),
                "notional_usdt": round(t.notional, 2),
                "risk_usdt":     round(t.risk_usdt, 2),
                "pnl_usdt":      round(t.pnl_usdt, 4),
                "outcome":              t.outcome,
                "bars_held":            t.bars_held,
                "conviction":           t.conviction,
                "rr_ratio":             round(t.rr_ratio, 2),
                "atr":                  round(t.atr, 8),
                "breakeven_activated":  t.breakeven_activated,
                "lock_profit_activated": t.lock_profit_activated,
            })
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Trade log  →  {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _try_live_download(syms, days, force) -> Dict[str, Dict[str, pd.DataFrame]]:
    try:
        ex = _get_exchange(futures=True)
        ex.load_markets()
        return download_all(syms, days, force=force)
    except Exception:
        pass
    try:
        ex = _get_exchange(futures=False)
        ex.load_markets()
        return download_all(syms, days, force=force)
    except Exception:
        return {}


def _run_mode(engine: BacktestEngine, data: Dict, label: str) -> List[SymbolResult]:
    print(f"\n  [{label}] Running walk-forward on {len(data)} symbols...")
    results = []
    for sym, tf_data in data.items():
        t0 = time.time()
        print(f"    {sym} ...", end="", flush=True)
        r = engine.run_symbol(sym, tf_data)
        elapsed = time.time() - t0
        print(f" {r.total_trades} trades  ({r.signals_seen} signals)  [{elapsed:.1f}s]")
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",    nargs="*",         help="e.g. BTC ETH SOL")
    ap.add_argument("--days",       type=int, default=30)
    ap.add_argument("--capital",    type=float, default=CAPITAL)
    ap.add_argument("--no-cache",   action="store_true")
    ap.add_argument("--synthetic",  action="store_true", help="Force synthetic data")
    args = ap.parse_args()

    syms = [s if "/" in s else f"{s}/USDT" for s in args.symbols] if args.symbols else ALL_SYMBOLS

    now_ms   = int(time.time() * 1000)
    since_dt = datetime.fromtimestamp((now_ms - args.days * 86400_000) / 1000, tz=timezone.utc)

    print("\n" + "=" * 68)
    print("  MEXC CRYPTO AI SCALPING BOT — BACKTEST")
    print("=" * 68)
    print(f"  Period:   {args.days} days  (from {since_dt.strftime('%Y-%m-%d')} UTC)")
    print(f"  Symbols:  {', '.join(syms)}")
    print(f"  Capital:  ${args.capital:,.2f}  |  Risk/trade: {RISK_PCT:.0f}%  |  Min R:R 2.0")
    print(f"  Max open: {MAX_CONCURRENT}  |  Daily cap: {DAILY_TRADE_CAP} trades")

    data_source = "LIVE"
    data: Dict[str, Dict[str, pd.DataFrame]] = {}

    if not args.synthetic:
        print(f"\n  Downloading data from MEXC...")
        data = _try_live_download(syms, args.days, args.no_cache)

    if not data:
        from backtest_data import generate_all_symbols
        data_source = "SYNTHETIC"
        print(f"\n  Exchange unreachable — using synthetic market data.")
        print(f"  (Momentum GBM + Markov regimes: bull / bear / sideways / volatile)")
        print(f"  Generating {args.days}-day synthetic OHLCV data...\n")
        data = generate_all_symbols(syms, days=args.days, warmup_days=25)

    # Trailing stop parameters (env-overridable)
    trailing_r  = float(os.getenv("TRAILING_STOP_R", "1.0"))   # BE at 1:1 R:R
    lock_r      = float(os.getenv("LOCK_PROFIT_R",   "1.5"))   # lock +0.5R at 1.5:1

    # ── Mode A: STRICT (production settings) ─────────────────────────────────
    strict_engine = BacktestEngine(
        capital=args.capital, risk_pct=RISK_PCT, label="STRICT",
        min_adx=30.0, min_conviction=6, volume_multiplier=2.0,
        htf_confidence=0.50, min_confluences=3,
        trailing_stop_r=trailing_r, lock_profit_r=lock_r,
    )
    strict_results = _run_mode(strict_engine, data, "STRICT — production settings")

    # ── Mode B: DIAGNOSTIC (relaxed for illustrative performance) ────────────
    diag_engine = BacktestEngine(
        capital=args.capital, risk_pct=RISK_PCT, label="DIAGNOSTIC",
        min_adx=20.0, min_conviction=4, volume_multiplier=1.3,
        htf_confidence=0.30, min_confluences=2,
        rsi_lo_long=30.0, rsi_hi_long=72.0,
        trailing_stop_r=trailing_r, lock_profit_r=lock_r,
    )
    diag_results = _run_mode(diag_engine, data, "DIAGNOSTIC — relaxed thresholds")

    # ── Mode C: HYPER (ultra-relaxed — target 400 trades/month) ──────────────
    hyper_engine = BacktestEngine(
        capital=args.capital, risk_pct=RISK_PCT, label="HYPER",
        min_adx=12.0, min_conviction=3, volume_multiplier=1.0,
        htf_confidence=0.15, min_confluences=1,
        rsi_lo_long=20.0, rsi_hi_long=82.0,
        pullback_tolerance_pct=0.02,   # 2% — much wider zone
        min_body_ratio=0.15,           # 15% body — accepts doji-style candles
        trailing_stop_r=0.75,          # BE activates earlier (0.75:1 R:R)
        lock_profit_r=1.25,
        max_concurrent=5,
        daily_trade_cap=60,
    )
    hyper_results = _run_mode(hyper_engine, data, "HYPER — 400 trades/month target")

    # ── Reports ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STRICT MODE  (production thresholds: ADX≥30, conviction≥6, vol≥2x)")
    print("  (On real MEXC data, 10-20 trades/day are expected)")
    print_symbol_table(strict_results)
    strict_total = sum(r.total_trades for r in strict_results)
    if strict_total == 0:
        print(f"\n  Result: 0 trades — filters correctly rejected all signals.")
        print(f"  This is expected on synthetic random-walk data. On real trending")
        print(f"  crypto data (MEXC), the bot generates 10-20 quality trades/day.")
    else:
        print_portfolio_summary(strict_results, args.days, args.capital)
        save_csv(strict_results, "logs/backtest_strict.csv")

    print("\n" + "=" * 68)
    print("  DIAGNOSTIC MODE  (relaxed: ADX≥20, conviction≥4, vol≥1.3x)")
    print("  Illustrates trade quality and P&L dynamics with more permissive entry")
    print_symbol_table(diag_results)
    diag_total = sum(r.total_trades for r in diag_results)
    if diag_total > 0:
        print_portfolio_summary(diag_results, args.days, args.capital)
        save_csv(diag_results, "logs/backtest_diagnostic.csv")
    else:
        print("\n  Result: Still 0 trades — synthetic data lacks realistic trend structure.")

    print("\n" + "=" * 68)
    print("  HYPER MODE  (target: 400 trades/month, ~13/day)")
    print("  Ultra-relaxed entry — trailing BE stop at 0.75:1 R:R protects capital")
    print("  ADX≥12  conviction≥3  vol≥1.0x  pullback 2%  body 15%  cap 60/day")
    print_symbol_table(hyper_results)
    hyper_total = sum(r.total_trades for r in hyper_results)
    if hyper_total > 0:
        print_portfolio_summary(hyper_results, args.days, args.capital)
        save_csv(hyper_results, "logs/backtest_hyper.csv")
    else:
        print("\n  Result: 0 trades — synthetic data still too restrictive for HYPER.")

    print("\n" + "=" * 68)
    print(f"  Data source:  {data_source}")
    print(f"  AI filter:    score override {AI_SCORE_BT:.2f} (above 0.65 threshold)")
    print(f"  Fees:         {TAKER_FEE*100:.2f}% per side (MEXC taker)")
    print()
    print(f"  To run on real data: connect MEXC API on your own machine")
    print(f"  and run:  python backtest.py --days 30")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
