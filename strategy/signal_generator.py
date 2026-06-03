import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

from analysis.trend_detector import TrendDetector, TrendDirection
from analysis.volume_analyzer import VolumeAnalyzer, VolumeState
from strategy.multi_timeframe import ConvictionScore, MultiTimeframeAnalyzer

logger = logging.getLogger("bot.signal")


@dataclass
class TradingSignal:
    signal_id: str
    symbol: str
    direction: str            # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    take_profit_1: float      # 1:2 R:R
    take_profit_2: float      # 1:3 R:R (for trailing portion)
    position_size: float      # in base currency
    notional_usdt: float
    risk_usdt: float
    atr: float
    conviction_score: int
    conviction_max: int
    ai_score: float           # set by AISignalFilter later
    rr_ratio: float
    checklist: Dict[str, bool]
    entry_reason: List[str]
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    timeframe: str = "5m"
    trailing_activation_price: float = 0.0  # price that triggers breakeven stop (entry ± 1×SL_dist)


class SignalGenerator:
    """
    7-criteria entry checklist — ALL must pass, zero soft checks.
    Built to match a manual 60%+ win-rate trader's discipline:
    - Only trades with strong trend alignment across all timeframes
    - Requires confirmed volume (not just average)
    - Needs price at a meaningful key level (structure matters)
    - Entry candle must show directional intent
    - 3 of 4 momentum/volume confirmations required
    """

    def __init__(self, trend_detector: TrendDetector = None,
                 volume_analyzer: VolumeAnalyzer = None,
                 mtf_analyzer: MultiTimeframeAnalyzer = None,
                 min_conviction: int = 6,
                 min_rr_ratio: float = 2.0,
                 min_adx: float = 30.0,
                 volume_multiplier: float = 2.0,
                 htf_confidence: float = 0.50,
                 min_confluences: int = 3,
                 rsi_lo_long: float = 38.0,
                 rsi_hi_long: float = 65.0,
                 pullback_tolerance_pct: float = 0.008,
                 min_body_ratio: float = 0.35):
        self.trend = trend_detector or TrendDetector()
        self.volume = volume_analyzer or VolumeAnalyzer()
        self.mtf = mtf_analyzer or MultiTimeframeAnalyzer()
        self.min_conviction = min_conviction
        self.min_rr_ratio = min_rr_ratio
        self.min_adx = min_adx
        self.volume_multiplier = volume_multiplier
        self.htf_confidence = htf_confidence
        self.min_confluences = min_confluences
        self.rsi_lo_long = rsi_lo_long
        self.rsi_hi_long = rsi_hi_long
        self.pullback_tolerance_pct = pullback_tolerance_pct
        self.min_body_ratio = min_body_ratio

    def generate_signal(self, symbol: str, tf_data: Dict[str, pd.DataFrame],
                        conviction: ConvictionScore,
                        order_book_imbalance: float = 0.5,
                        capital: float = 1000.0,
                        max_risk_pct: float = 5.0,
                        kelly_sizer=None,
                        quantum_conviction: float = 0.625) -> Optional[TradingSignal]:

        df_5m = tf_data.get("5m")

        if df_5m is None or len(df_5m) < 30:
            return None

        checklist = {}
        reasons = []
        last_5m = df_5m.iloc[-1]
        direction = conviction.aligned_direction

        # === CHECK 1: HTF Trend Alignment ===
        htf_ok = (conviction.htf_bias != "SIDEWAYS" and
                  conviction.htf_bias_confidence > self.htf_confidence)
        checklist["htf_trend_aligned"] = htf_ok
        if htf_ok:
            reasons.append(f"HTF {conviction.htf_bias} ({conviction.htf_bias_confidence:.0%} conf)")

        # === CHECK 2: MTF Conviction Score (6/8 minimum) ===
        conviction_ok = conviction.total >= self.min_conviction
        checklist["conviction_score"] = conviction_ok
        if conviction_ok:
            reasons.append(f"Conviction {conviction.total}/{conviction.max_score}")

        # === CHECK 3: ADX >= 30 (strong trending market, not ranging) ===
        adx = float(last_5m.get("adx", 0))
        adx_ok = adx >= self.min_adx
        checklist["adx_trending"] = adx_ok
        if adx_ok:
            reasons.append(f"ADX {adx:.1f}")

        # === CHECK 4: Momentum in safe zone (not overbought/oversold) ===
        rsi = float(last_5m.get("rsi", 50))
        stoch_k = float(last_5m.get("stochrsi_k", 50))

        rsi_lo_short = max(25.0, self.rsi_lo_long - 3)
        rsi_hi_short = max(55.0, self.rsi_hi_long - 3)
        if direction == "BULLISH":
            momentum_ok = self.rsi_lo_long < rsi < self.rsi_hi_long and stoch_k < 75
        else:
            momentum_ok = rsi_lo_short < rsi < rsi_hi_short and stoch_k > 25
        checklist["momentum_not_extreme"] = momentum_ok
        if momentum_ok:
            reasons.append(f"RSI {rsi:.1f}, StochRSI {stoch_k:.1f}")

        # === CHECK 5: Volume surge >= 2x average ===
        vol_state = self.volume.analyze(df_5m, order_book_imbalance)
        volume_ok = vol_state.surge_ratio >= self.volume_multiplier
        checklist["volume_confirmed"] = volume_ok
        if volume_ok:
            reasons.append(f"Volume {vol_state.surge_ratio:.1f}x avg")

        # === CHECK 6: Price near key level (always required — no override) ===
        near_level = self.mtf.is_near_key_level(df_5m, tolerance_pct=0.003)
        checklist["near_key_level"] = near_level
        if near_level:
            reasons.append("Price at key level (VWAP/EMA/BB/S-R/Fib)")

        # === CHECK 7: Entry candle confirms direction ===
        candle_ok = self._check_entry_candle(df_5m, direction)
        checklist["entry_candle_confirmed"] = candle_ok
        if candle_ok:
            reasons.append("Entry candle shows directional intent")

        # === ALL 7 CHECKS MUST PASS — zero exceptions ===
        failed = [k for k, v in checklist.items() if not v]
        if failed:
            logger.debug(f"{symbol}: Signal rejected — failed checks: {failed}")
            return None

        # === DIRECTION CONFLUENCE: require 3 of 4 confirmations ===
        macd_hist = float(last_5m.get("macd_hist", 0))
        cvd_bullish = vol_state.cvd_bullish
        obv_up = vol_state.obv_trend == "UP"

        if direction == "BULLISH":
            confirmations = sum([
                macd_hist > 0,
                cvd_bullish,
                obv_up,
                order_book_imbalance > 0.52,
            ])
            if confirmations < self.min_confluences:
                logger.debug(f"{symbol}: Long rejected — {confirmations}/4 confluence ({macd_hist:.4f}, cvd={cvd_bullish}, obv={vol_state.obv_trend}, ob={order_book_imbalance:.2f})")
                return None
        else:
            confirmations = sum([
                macd_hist < 0,
                not cvd_bullish,
                not obv_up,
                order_book_imbalance < 0.48,
            ])
            if confirmations < self.min_confluences:
                logger.debug(f"{symbol}: Short rejected — {confirmations}/4 confluence")
                return None

        reasons.append(f"Confluence {confirmations}/4")

        # === PULLBACK CONFIRMATION: entry near EMA21 or BB midline ===
        pullback_ok = self._check_pullback_quality(df_5m, direction)
        if not pullback_ok:
            logger.debug(f"{symbol}: Rejected — price not in pullback zone (chasing entry)")
            return None
        reasons.append("Pullback to key EMA/midline")

        # === CALCULATE ENTRY, SL, TP ===
        price = float(df_5m["close"].iloc[-1])
        atr = float(last_5m.get("atr", price * 0.01))
        if atr <= 0:
            atr = price * 0.01

        if direction == "BULLISH":
            entry = price
            stop_loss = entry - 1.5 * atr
            support = float(last_5m.get("support", entry - 2 * atr))
            stop_loss = min(stop_loss, support - 0.05 * atr)
        else:
            entry = price
            stop_loss = entry + 1.5 * atr
            resistance = float(last_5m.get("resistance", entry + 2 * atr))
            stop_loss = max(stop_loss, resistance + 0.05 * atr)

        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            logger.warning(f"{symbol}: Invalid SL distance")
            return None

        take_profit_1 = entry + sl_distance * self.min_rr_ratio * (1 if direction == "BULLISH" else -1)
        take_profit_2 = entry + sl_distance * (self.min_rr_ratio + 1) * (1 if direction == "BULLISH" else -1)

        rr = abs(take_profit_1 - entry) / sl_distance
        if rr < self.min_rr_ratio:
            return None

        # === POSITION SIZING: Kelly or fixed risk ===
        if kelly_sizer is not None:
            effective_risk_pct = kelly_sizer.calculate_risk_pct(quantum_conviction)
        else:
            effective_risk_pct = max_risk_pct
        risk_amount = capital * (effective_risk_pct / 100)
        position_size = risk_amount / sl_distance
        notional = position_size * entry

        # Cap notional at 50% of capital (allows 5% risk to work at typical SL distances)
        max_notional = capital * 0.50
        if notional > max_notional:
            position_size = max_notional / entry
            notional = max_notional

        if position_size <= 0:
            return None

        return TradingSignal(
            signal_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            direction="LONG" if direction == "BULLISH" else "SHORT",
            entry_price=round(entry, 8),
            stop_loss=round(stop_loss, 8),
            take_profit_1=round(take_profit_1, 8),
            take_profit_2=round(take_profit_2, 8),
            position_size=round(position_size, 6),
            notional_usdt=round(notional, 2),
            risk_usdt=round(risk_amount, 2),
            atr=round(atr, 8),
            conviction_score=conviction.total,
            conviction_max=conviction.max_score,
            ai_score=0.0,
            rr_ratio=round(rr, 2),
            checklist=checklist,
            entry_reason=reasons,
            trailing_activation_price=round(
                entry + sl_distance if direction == "BULLISH" else entry - sl_distance, 8
            ),
        )

    def _check_entry_candle(self, df: pd.DataFrame, direction: TrendDirection) -> bool:
        """
        Entry candle must show clear directional intent.
        LONG: bullish candle (close > open), close in top 50% of candle range,
              body >= 35% of total candle range
        SHORT: bearish candle (close < open), close in bottom 50% of range,
               body >= 35% of range
        Exception: high-volume absorption (volume >= 2.5x avg) forgives small body (doji = indecision resolved by volume)
        """
        if len(df) < 2:
            return True
        last = df.iloc[-1]
        open_ = float(last["open"])
        close = float(last["close"])
        high = float(last["high"])
        low = float(last["low"])
        candle_range = high - low
        if candle_range <= 0:
            return False

        body = abs(close - open_)
        body_ratio = body / candle_range
        close_position = (close - low) / candle_range  # 0 = at low, 1 = at high

        # High-volume absorption exception
        avg_vol = float(df["volume"].tail(20).mean())
        current_vol = float(last["volume"])
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        absorption = vol_ratio >= 2.5

        if direction == "BULLISH":
            is_bullish_candle = close > open_
            close_in_upper_half = close_position >= 0.50
            strong_body = body_ratio >= self.min_body_ratio
            return (is_bullish_candle and close_in_upper_half) and (strong_body or absorption)
        else:
            is_bearish_candle = close < open_
            close_in_lower_half = close_position <= 0.50
            strong_body = body_ratio >= self.min_body_ratio
            return (is_bearish_candle and close_in_lower_half) and (strong_body or absorption)

    def _check_pullback_quality(self, df: pd.DataFrame, direction: TrendDirection) -> bool:
        """
        Avoids chasing breakouts. Requires price to be in a pullback zone:
        - Near EMA21 (within 0.8% for longs = pulled back to trend)
        - OR near BB midline (mean reversion sweet spot)
        - OR near VWAP (institutional equilibrium)
        - OR Supertrend is bullish/bearish and price is near it
        """
        if len(df) < 2:
            return True
        last = df.iloc[-1]
        close = float(last["close"])

        levels_to_check = []
        for col in ["ema_21", "ema_50", "bb_mid", "vwap"]:
            val = float(last.get(col, 0))
            if val > 0:
                levels_to_check.append(val)

        # Supertrend proximity
        supertrend = float(last.get("supertrend", 0))
        if supertrend > 0:
            levels_to_check.append(supertrend)

        if not levels_to_check:
            return True  # can't check, allow

        for level in levels_to_check:
            dist_pct = abs(close - level) / close
            if dist_pct <= self.pullback_tolerance_pct:
                return True

        return False

    def validate_spread(self, symbol: str, ticker: Dict,
                         max_spread_pct: float = 0.001) -> bool:
        """Returns False if bid-ask spread is too wide (low liquidity)."""
        try:
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            mid = (bid + ask) / 2
            if mid <= 0:
                return True
            spread = (ask - bid) / mid
            return spread <= max_spread_pct
        except Exception:
            return True
