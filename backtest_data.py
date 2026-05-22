"""
Hierarchical synthetic OHLCV generator for backtesting.

Uses a TWO-LEVEL regime system:
  1. Macro trend (days): switches every 5-20 days — drives 4h/1d alignment
  2. Micro regime (hours): pullbacks and continuations within the macro trend

Pullback micro-states use WEAK POSITIVE drift (not negative) so all TFs
stay net-bullish during consolidations → conviction can build to 6-8/8.

Key calibration: +2%/day during bull macro = +0.000069/5m-bar drift
"""
import hashlib
import numpy as np
import pandas as pd
import time
from typing import Dict

SYMBOL_PRICES = {
    "BTC/USDT": 92_000,
    "ETH/USDT":  3_400,
    "SOL/USDT":    155,
    "BNB/USDT":    580,
    "XRP/USDT":   0.62,
    "DOGE/USDT":  0.19,
    "ADA/USDT":   0.58,
    "AVAX/USDT":    38,
    "LINK/USDT":    18,
    "MATIC/USDT": 0.52,
}

# Macro trend: very slow switching (~every 1440-4320 5m bars = 5-15 days)
# (direction, switch_prob_per_bar)
MACRO_STATES = ["bull", "bear", "sideways"]
MACRO_TRANS = np.array([
    [0.9998, 0.0001, 0.0001],   # bull → stays bull ~5000 bars avg = 17.4 days
    [0.0001, 0.9998, 0.0001],   # bear
    [0.0003, 0.0003, 0.9994],   # sideways → stays ~1667 bars avg = 5.8 days
])

# Micro regime within macro: fast switching (~every 30-300 bars = 2.5h-25h)
# (drift_multiplier, vol_multiplier)
MICRO_STATES_BY_MACRO = {
    "bull": [
        # (name, drift_mult, vol_mult, persist)
        # drift_mult=10 → price moves ~0.07%/bar, SNR≈3 → ADX builds to 30-50
        # EMA50 gap builds to >1.5% → slope_ema50 > 0.3% → TrendDetector fires BULLISH
        ("continuation",  5.0, 0.12, 0.993),    # strong clean uptrend, low noise → ADX 30-50
        ("pullback",      0.4, 0.80, 0.965),    # weak positive drift — RSI dips, stays BULLISH
        ("volatile_up",   2.0, 0.65, 0.975),    # moderate trend with noise
    ],
    "bear": [
        ("continuation", -5.0, 0.12, 0.993),
        ("pullback",     -0.4, 0.80, 0.965),
        ("volatile_dn",  -2.0, 0.65, 0.975),
    ],
    "sideways": [
        ("range",    0.0, 0.55, 0.992),
        ("volatile", 0.0, 1.35, 0.962),
    ],
}
# Micro transition: mostly stay in current, sometimes switch
MICRO_TRANS = {
    "bull":     np.array([[0.985, 0.010, 0.005],
                          [0.040, 0.950, 0.010],
                          [0.030, 0.020, 0.950]]),
    "bear":     np.array([[0.985, 0.010, 0.005],
                          [0.040, 0.950, 0.010],
                          [0.030, 0.020, 0.950]]),
    "sideways": np.array([[0.990, 0.010],
                          [0.020, 0.980]]),
}


def generate_5m(symbol: str, n_bars: int, seed: int = 42) -> pd.DataFrame:
    sym_hash = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 9973
    rng   = np.random.default_rng(seed + sym_hash)
    p0    = float(SYMBOL_PRICES.get(symbol, 100.0))
    price = p0

    # Higher base drift (+5%/day in bull) with much lower noise → clean SNR for ADX/slope
    # SNR during continuation (drift_mult=5, vol_mult=0.12):
    #   drift = 5 * 0.000174 = 0.00087/bar, sigma/p0 = 0.0004 * 0.12 = 0.000048 → SNR ≈ 18
    base_drift = 0.05 / 288        # 0.000174/bar (+5%/day base rate)
    base_sigma = p0 * 0.0004       # 0.04%/bar noise (10x cleaner than real crypto)

    macro_idx  = 0   # start bullish so warmup builds 1d trend
    micro_idx  = 0
    log_price  = np.log(p0)

    opens   = []; highs   = []; lows    = []; closes  = []; volumes = []

    daily_notional = {
        "BTC/USDT": 2_000_000_000, "ETH/USDT": 800_000_000, "BNB/USDT": 500_000_000,
    }
    vol_per_bar = daily_notional.get(symbol, 150_000_000) / 288 / max(p0, 1e-10)

    for bar_i in range(n_bars):
        macro  = MACRO_STATES[macro_idx]
        micro_states = MICRO_STATES_BY_MACRO[macro]
        name, drift_m, vol_m, persist_m = micro_states[micro_idx % len(micro_states)]

        # Effective drift per bar
        macro_sign = 1 if macro == "bull" else (-1 if macro == "bear" else 0)
        drift  = macro_sign * base_drift * drift_m
        sigma  = base_sigma * vol_m

        # Mean-reversion to regime target (prevents price explosion)
        # Target: +0.7 log-units above p0 in bull (2x), -0.7 in bear, 0 sideways
        target_log = (np.log(p0) + 0.7 * macro_sign)
        deviation  = log_price - target_log
        mr = -0.002 * deviation   # strong pull toward regime target
        drift += mr

        z   = rng.standard_normal()
        ret = float(np.clip(drift + z * sigma / p0, -0.03, 0.03))

        o = price
        c = price * np.exp(ret)
        c = max(c, p0 * 0.05)   # absolute floor

        wick = abs(z) * sigma / p0 * 0.5
        h = max(o, c) * (1 + wick * rng.uniform(0.2, 0.8))
        l = min(o, c) * (1 - wick * rng.uniform(0.2, 0.8))
        h = max(h, o, c)
        l = max(min(l, o, c), p0 * 0.01)

        # Volume: high on trend-aligned bars; periodic spikes simulate real crypto surges
        trend_aligned = (ret > 0 and macro_sign > 0) or (ret < 0 and macro_sign < 0)
        vfact = (1.6 if trend_aligned else 0.65) * rng.uniform(0.6, 1.6)
        if rng.random() < 0.08:   # ~1 spike per 12.5 bars = every ~1h on 5m chart
            vfact *= rng.uniform(2.5, 6.0)
        vol = vol_per_bar * vfact

        opens.append(float(o)); highs.append(float(h))
        lows.append(float(l));  closes.append(float(c))
        volumes.append(float(vol))

        price     = c
        log_price = np.log(c)

        # Micro transition
        mp = MICRO_TRANS[macro]
        n_micro = len(micro_states)
        row_idx = min(micro_idx, n_micro - 1)
        if rng.random() > persist_m and n_micro > 1:
            row = mp[row_idx][:n_micro]
            row = row / row.sum()
            micro_idx = int(rng.choice(n_micro, p=row))

        # Macro transition
        if rng.random() > MACRO_TRANS[macro_idx][macro_idx]:
            p_row = MACRO_TRANS[macro_idx].copy()
            p_row[macro_idx] = 0
            if p_row.sum() > 0:
                p_row /= p_row.sum()
                macro_idx = int(rng.choice(3, p=p_row))
                micro_idx = 0   # reset micro on macro change

    now_5m = (int(time.time()) // 300) * 300
    idx = pd.date_range(
        start=pd.Timestamp(now_5m - n_bars * 300, unit="s", tz="UTC"),
        periods=n_bars, freq="5min"
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx
    ).astype(float)


def resample_ohlcv(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df5.resample(rule, label="left", closed="left").agg(agg).dropna()


_RULES = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def generate_all_timeframes(symbol: str, days: int = 30,
                             warmup_days: int = 25, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    warmup_days=25: gives 1d TF 25 closed bars — enough for TrendDetector (needs 50 but works at 20).
    Set to 50+ for full stability; 25 is the backtest minimum.
    """
    n_bars = (days + warmup_days) * 288
    df5 = generate_5m(symbol, n_bars, seed=seed)
    out = {"5m": df5}
    for tf, rule in _RULES.items():
        out[tf] = resample_ohlcv(df5, rule)
    return out


def generate_all_symbols(symbols, days: int = 30,
                          warmup_days: int = 25) -> Dict[str, Dict[str, pd.DataFrame]]:
    data = {}
    for i, sym in enumerate(symbols):
        data[sym] = generate_all_timeframes(sym, days=days,
                                             warmup_days=warmup_days, seed=42 + i * 13)
        p0  = SYMBOL_PRICES.get(sym, 100)
        c   = data[sym]["5m"]["close"].iloc[-1]
        pct = (c - p0) / p0 * 100
        bars_5m = len(data[sym]["5m"])
        print(f"  synthetic  {sym}: {bars_5m} bars  ({c:,.4f})  {pct:+.1f}%")
    return data
