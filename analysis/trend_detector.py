import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple
import logging

logger = logging.getLogger("bot.trend")

TrendDirection = Literal["BULLISH", "BEARISH", "SIDEWAYS"]


@dataclass
class TrendState:
    direction: TrendDirection
    strength: float           # ADX value
    ema_stack_aligned: bool
    price_above_vwap: bool
    supertrend_bullish: bool
    slope_ema50: float
    higher_highs: bool
    higher_lows: bool
    confidence: float         # 0.0 to 1.0


class TrendDetector:
    def detect_trend(self, df: pd.DataFrame, timeframe: str = "1h") -> TrendState:
        if df is None or len(df) < 50:
            return TrendState("SIDEWAYS", 20, False, False, False, 0.0, False, False, 0.0)

        last = df.iloc[-1]
        prev5 = df.iloc[-6:-1] if len(df) >= 6 else df.iloc[:-1]

        # ADX strength
        adx = float(last.get("adx", 20))
        dmp = float(last.get("dmp", 25))
        dmn = float(last.get("dmn", 25))

        # EMA stack
        ema9 = float(last.get("ema_9", last["close"]))
        ema21 = float(last.get("ema_21", last["close"]))
        ema50 = float(last.get("ema_50", last["close"]))
        ema200 = float(last.get("ema_200", last["close"]))
        close = float(last["close"])

        bull_stack = ema9 > ema21 > ema50
        bear_stack = ema9 < ema21 < ema50
        price_above_vwap = close > float(last.get("vwap", close))
        supertrend_dir = int(last.get("supertrend_dir", 1))
        supertrend_bullish = supertrend_dir == 1

        # EMA50 slope
        if len(df) >= 6 and "ema_50" in df.columns:
            ema50_now = float(df["ema_50"].iloc[-1])
            ema50_5ago = float(df["ema_50"].iloc[-6])
            slope_ema50 = (ema50_now - ema50_5ago) / (ema50_5ago + 1e-10) * 100
        else:
            slope_ema50 = 0.0

        # Higher highs / higher lows detection (last 3 swing highs/lows)
        higher_highs = False
        higher_lows = False
        if len(df) >= 20:
            highs = df["high"].rolling(5).max().dropna()
            lows = df["low"].rolling(5).min().dropna()
            if len(highs) >= 3:
                higher_highs = float(highs.iloc[-1]) > float(highs.iloc[-2]) > float(highs.iloc[-3])
                higher_lows = float(lows.iloc[-1]) > float(lows.iloc[-2]) > float(lows.iloc[-3])

        # Determine direction
        bullish_signals = sum([
            bull_stack,
            price_above_vwap,
            supertrend_bullish,
            dmp > dmn,
            slope_ema50 > 0,
            higher_highs,
            higher_lows,
        ])
        bearish_signals = sum([
            bear_stack,
            not price_above_vwap,
            not supertrend_bullish,
            dmn > dmp,
            slope_ema50 < 0,
            not higher_highs,
            not higher_lows,
        ])

        if adx < 20:
            direction: TrendDirection = "SIDEWAYS"
        elif bullish_signals >= 5:
            direction = "BULLISH"
        elif bearish_signals >= 5:
            direction = "BEARISH"
        elif bullish_signals >= 4:
            direction = "BULLISH"
        elif bearish_signals >= 4:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"

        confidence = min(1.0, (adx / 50) * (max(bullish_signals, bearish_signals) / 7))

        return TrendState(
            direction=direction,
            strength=adx,
            ema_stack_aligned=bull_stack if direction == "BULLISH" else bear_stack,
            price_above_vwap=price_above_vwap,
            supertrend_bullish=supertrend_bullish,
            slope_ema50=slope_ema50,
            higher_highs=higher_highs,
            higher_lows=higher_lows,
            confidence=confidence,
        )

    def get_htf_bias(self, tf_data: Dict[str, pd.DataFrame]) -> Tuple[TrendDirection, float]:
        """Combined 4h + 1d macro bias."""
        votes = []
        confidences = []

        for tf in ["1d", "4h"]:
            if tf in tf_data and len(tf_data[tf]) > 10:
                state = self.detect_trend(tf_data[tf], tf)
                votes.append(state.direction)
                confidences.append(state.confidence)

        if not votes:
            return "SIDEWAYS", 0.0

        bull_count = votes.count("BULLISH")
        bear_count = votes.count("BEARISH")
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        if bull_count > bear_count:
            return "BULLISH", avg_conf
        elif bear_count > bull_count:
            return "BEARISH", avg_conf
        else:
            return "SIDEWAYS", avg_conf * 0.5

    def detect_ema_crossover(self, df: pd.DataFrame,
                              fast: int = 9, slow: int = 21) -> str:
        if df is None or len(df) < 3:
            return "NONE"
        fast_col = f"ema_{fast}"
        slow_col = f"ema_{slow}"
        if fast_col not in df.columns or slow_col not in df.columns:
            return "NONE"

        fast_now = float(df[fast_col].iloc[-1])
        fast_prev = float(df[fast_col].iloc[-2])
        slow_now = float(df[slow_col].iloc[-1])
        slow_prev = float(df[slow_col].iloc[-2])

        if fast_prev <= slow_prev and fast_now > slow_now:
            return "CROSS_UP"
        elif fast_prev >= slow_prev and fast_now < slow_now:
            return "CROSS_DOWN"
        return "NONE"

    def detect_market_structure(self, df: pd.DataFrame, lookback: int = 15) -> Dict:
        result = {
            "bos_bullish": False,
            "bos_bearish": False,
            "choch": False,
            "swing_high": 0.0,
            "swing_low": 0.0,
        }
        if df is None or len(df) < lookback + 5:
            return result

        recent = df.tail(lookback + 5)
        swing_high = float(recent["high"].max())
        swing_low = float(recent["low"].min())
        result["swing_high"] = swing_high
        result["swing_low"] = swing_low

        last_close = float(df["close"].iloc[-1])
        prev_swing_high = float(df["high"].iloc[-lookback:-5].max()) if len(df) >= lookback + 5 else swing_high
        prev_swing_low = float(df["low"].iloc[-lookback:-5].min()) if len(df) >= lookback + 5 else swing_low

        # Break of structure: price closes beyond previous swing
        if last_close > prev_swing_high:
            result["bos_bullish"] = True
        if last_close < prev_swing_low:
            result["bos_bearish"] = True

        # Change of character: BOS in opposing direction of prior trend
        if result["bos_bullish"] and self.detect_trend(df.iloc[:-5]).direction == "BEARISH":
            result["choch"] = True
        if result["bos_bearish"] and self.detect_trend(df.iloc[:-5]).direction == "BULLISH":
            result["choch"] = True

        return result

    def is_near_level(self, price: float, level: float, tolerance_pct: float = 0.003) -> bool:
        return abs(price - level) / (price + 1e-10) <= tolerance_pct
