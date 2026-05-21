import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("bot.patterns")


@dataclass
class CandlePattern:
    is_bullish_candle: bool
    is_bearish_candle: bool
    body_ratio: float              # body / range, 0–1
    close_position: float          # 0=at low, 1=at high
    is_hammer: bool                # bullish reversal
    is_shooting_star: bool         # bearish reversal
    is_bullish_engulfing: bool
    is_bearish_engulfing: bool
    is_inside_bar: bool            # compression before breakout
    is_doji: bool                  # indecision
    pattern_strength: float        # 0–1 composite score


class CandlePatternAnalyzer:
    def analyze(self, df: pd.DataFrame) -> CandlePattern:
        if df is None or len(df) < 2:
            return self._neutral()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        o = float(last["open"])
        c = float(last["close"])
        h = float(last["high"])
        lo = float(last["low"])

        o_p = float(prev["open"])
        c_p = float(prev["close"])
        h_p = float(prev["high"])
        lo_p = float(prev["low"])

        candle_range = h - lo
        if candle_range <= 0:
            return self._neutral()

        body = abs(c - o)
        body_ratio = body / candle_range
        close_position = (c - lo) / candle_range

        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - lo

        is_bullish = c > o
        is_bearish = c < o

        # Doji: tiny body (< 10% of range)
        is_doji = body_ratio < 0.10

        # Hammer: small body in upper 40%, lower wick >= 2x body, minimal upper wick
        is_hammer = (
            close_position >= 0.60 and
            lower_wick >= 2 * max(body, 0.001) and
            upper_wick <= 0.3 * candle_range and
            body_ratio < 0.40
        )

        # Shooting star: small body in lower 40%, upper wick >= 2x body
        is_shooting_star = (
            close_position <= 0.40 and
            upper_wick >= 2 * max(body, 0.001) and
            lower_wick <= 0.3 * candle_range and
            body_ratio < 0.40
        )

        # Bullish engulfing: current bullish candle body fully covers previous bearish body
        prev_bearish = c_p < o_p
        is_bullish_engulfing = (
            is_bullish and prev_bearish and
            c >= o_p and o <= c_p and
            body >= abs(c_p - o_p)
        )

        # Bearish engulfing
        prev_bullish = c_p > o_p
        is_bearish_engulfing = (
            is_bearish and prev_bullish and
            c <= o_p and o >= c_p and
            body >= abs(c_p - o_p)
        )

        # Inside bar: current candle inside previous (compression = breakout pending)
        is_inside_bar = h < h_p and lo > lo_p

        # Pattern strength (0–1 composite)
        strength = 0.0
        if is_bullish_engulfing or is_bearish_engulfing:
            strength = 0.85
        elif is_hammer or is_shooting_star:
            strength = 0.75
        elif not is_doji and body_ratio >= 0.60:  # strong directional candle
            strength = 0.65
        elif is_inside_bar:
            strength = 0.50
        elif not is_doji and body_ratio >= 0.40:
            strength = 0.45

        return CandlePattern(
            is_bullish_candle=is_bullish,
            is_bearish_candle=is_bearish,
            body_ratio=round(body_ratio, 3),
            close_position=round(close_position, 3),
            is_hammer=is_hammer,
            is_shooting_star=is_shooting_star,
            is_bullish_engulfing=is_bullish_engulfing,
            is_bearish_engulfing=is_bearish_engulfing,
            is_inside_bar=is_inside_bar,
            is_doji=is_doji,
            pattern_strength=round(strength, 3),
        )

    def add_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candle pattern columns to a DataFrame."""
        if len(df) < 2:
            return df
        df = df.copy()

        body_ratios = []
        close_positions = []
        hammers = []
        shooting_stars = []
        bull_engulfings = []
        bear_engulfings = []
        inside_bars = []
        dozis = []
        strengths = []

        for i in range(len(df)):
            if i == 0:
                body_ratios.append(0.5)
                close_positions.append(0.5)
                hammers.append(False)
                shooting_stars.append(False)
                bull_engulfings.append(False)
                bear_engulfings.append(False)
                inside_bars.append(False)
                dozis.append(False)
                strengths.append(0.0)
                continue

            pattern = self.analyze(df.iloc[max(0, i-1):i+1])
            body_ratios.append(pattern.body_ratio)
            close_positions.append(pattern.close_position)
            hammers.append(pattern.is_hammer)
            shooting_stars.append(pattern.is_shooting_star)
            bull_engulfings.append(pattern.is_bullish_engulfing)
            bear_engulfings.append(pattern.is_bearish_engulfing)
            inside_bars.append(pattern.is_inside_bar)
            dozis.append(pattern.is_doji)
            strengths.append(pattern.pattern_strength)

        df["candle_body_ratio"] = body_ratios
        df["candle_close_position"] = close_positions
        df["candle_is_hammer"] = hammers
        df["candle_is_shooting_star"] = shooting_stars
        df["candle_bull_engulfing"] = bull_engulfings
        df["candle_bear_engulfing"] = bear_engulfings
        df["candle_inside_bar"] = inside_bars
        df["candle_is_doji"] = dozis
        df["candle_strength"] = strengths

        return df

    def _neutral(self) -> CandlePattern:
        return CandlePattern(
            is_bullish_candle=False, is_bearish_candle=False,
            body_ratio=0.5, close_position=0.5,
            is_hammer=False, is_shooting_star=False,
            is_bullish_engulfing=False, is_bearish_engulfing=False,
            is_inside_bar=False, is_doji=False, pattern_strength=0.0,
        )
