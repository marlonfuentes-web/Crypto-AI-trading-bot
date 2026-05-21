import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

from analysis.trend_detector import TrendDetector, TrendDirection, TrendState
from analysis.volume_analyzer import VolumeAnalyzer, VolumeState

logger = logging.getLogger("bot.mtf")

# Timeframe weights: higher timeframes carry more weight
TIMEFRAME_WEIGHTS = {
    "1d": 2,
    "4h": 2,
    "1h": 1,
    "30m": 1,
    "15m": 1,
    "5m": 1,
}
MAX_WEIGHT = sum(TIMEFRAME_WEIGHTS.values())  # = 8


@dataclass
class TimeframeScore:
    timeframe: str
    direction: TrendDirection
    weight: int
    score: int                # 0 or weight
    trend_state: Optional[TrendState] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class ConvictionScore:
    total: int                 # 0-8 (weighted)
    max_score: int             # = 8
    breakdown: Dict[str, int]  # {"5m": 1, "15m": 0, ...}
    aligned_direction: TrendDirection
    htf_bias: TrendDirection
    htf_bias_confidence: float
    tf_scores: Dict[str, TimeframeScore] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        return self.total / self.max_score if self.max_score > 0 else 0.0

    @property
    def is_tradeable(self) -> bool:
        return self.total >= 5 and self.aligned_direction != "SIDEWAYS"


class MultiTimeframeAnalyzer:
    def __init__(self, trend_detector: TrendDetector = None,
                 volume_analyzer: VolumeAnalyzer = None,
                 min_conviction: int = 5):
        self.trend = trend_detector or TrendDetector()
        self.volume = volume_analyzer or VolumeAnalyzer()
        self.min_conviction = min_conviction

    def analyze(self, tf_data: Dict[str, pd.DataFrame], symbol: str = "") -> ConvictionScore:
        if not tf_data:
            return self._empty_score()

        # Step 1: Get macro HTF bias from 1d + 4h
        htf_bias, htf_confidence = self.trend.get_htf_bias(tf_data)

        if htf_bias == "SIDEWAYS":
            return ConvictionScore(
                total=0, max_score=MAX_WEIGHT, breakdown={},
                aligned_direction="SIDEWAYS", htf_bias="SIDEWAYS",
                htf_bias_confidence=htf_confidence,
                notes=["HTF bias is SIDEWAYS — no trade"],
            )

        # Step 2: Score each timeframe
        tf_scores: Dict[str, TimeframeScore] = {}
        total_score = 0
        breakdown = {}

        available_tfs = [tf for tf in TIMEFRAME_WEIGHTS.keys() if tf in tf_data]

        for tf in available_tfs:
            df = tf_data[tf]
            if df is None or len(df) < 30:
                continue

            weight = TIMEFRAME_WEIGHTS[tf]
            tf_score = self._score_timeframe(df, tf, htf_bias)
            tf_scores[tf] = tf_score
            total_score += tf_score.score
            breakdown[tf] = tf_score.score

        # Step 3: Check if there are any opposing signals (conflict detection)
        opposing = [tf for tf, sc in tf_scores.items()
                    if sc.direction not in (htf_bias, "SIDEWAYS") and sc.score == 0]
        notes = []
        if len(opposing) >= 2:
            notes.append(f"Conflicting signals on {opposing}")

        return ConvictionScore(
            total=total_score,
            max_score=MAX_WEIGHT,
            breakdown=breakdown,
            aligned_direction=htf_bias,
            htf_bias=htf_bias,
            htf_bias_confidence=htf_confidence,
            tf_scores=tf_scores,
            notes=notes,
        )

    def _score_timeframe(self, df: pd.DataFrame, timeframe: str,
                          htf_bias: TrendDirection) -> TimeframeScore:
        weight = TIMEFRAME_WEIGHTS.get(timeframe, 1)
        state = self.trend.detect_trend(df, timeframe)
        reasons = []

        aligned = state.direction == htf_bias
        if aligned:
            reasons.append(f"Trend {state.direction} matches HTF {htf_bias}")
        else:
            reasons.append(f"Trend {state.direction} conflicts with HTF {htf_bias}")

        # For higher timeframes (1d, 4h) — direction match is enough
        if timeframe in ("1d", "4h"):
            score = weight if aligned else 0
        else:
            # For lower timeframes, also check momentum
            momentum_ok = self._check_momentum_ok(df, htf_bias)
            if aligned and momentum_ok:
                score = weight
            elif aligned:
                score = max(0, weight - 1)  # partial credit
                reasons.append("Momentum not confirmed")
            else:
                score = 0

        return TimeframeScore(
            timeframe=timeframe,
            direction=state.direction,
            weight=weight,
            score=score,
            trend_state=state,
            reasons=reasons,
        )

    def _check_momentum_ok(self, df: pd.DataFrame, direction: TrendDirection) -> bool:
        if len(df) < 2:
            return True
        last = df.iloc[-1]
        rsi = float(last.get("rsi", 50))
        stoch_k = float(last.get("stochrsi_k", 50))

        if direction == "BULLISH":
            # Not overbought — room to run
            rsi_ok = 30 < rsi < 75
            stoch_ok = stoch_k < 85
        else:
            # Not oversold — room to drop
            rsi_ok = 25 < rsi < 70
            stoch_ok = stoch_k > 15

        return rsi_ok and stoch_ok

    def is_near_key_level(self, df: pd.DataFrame, tolerance_pct: float = 0.003) -> bool:
        if df is None or len(df) < 2:
            return False
        price = float(df["close"].iloc[-1])
        checks = []

        for col in ["vwap", "ema_21", "ema_50", "bb_mid", "support", "resistance",
                    "fib_382", "fib_500", "fib_618"]:
            if col in df.columns:
                level = float(df[col].iloc[-1])
                if level > 0:
                    checks.append(abs(price - level) / price <= tolerance_pct)

        return any(checks)

    def get_entry_zone(self, tf_data: Dict[str, pd.DataFrame],
                       direction: TrendDirection) -> Dict:
        """Returns entry price, SL, and TP1 based on 5m + 15m candles."""
        result = {"valid": False, "price": 0.0, "sl": 0.0, "tp": 0.0, "atr": 0.0}

        df = tf_data.get("5m")
        if df is None or len(df) < 20:
            return result

        last = df.iloc[-1]
        price = float(last["close"])
        atr = float(last.get("atr", price * 0.01))
        if atr <= 0:
            atr = price * 0.01

        if direction == "BULLISH":
            entry = price
            sl = entry - 1.5 * atr
            # Ensure SL is below support
            support = float(last.get("support", entry - 2 * atr))
            sl = min(sl, support - 0.1 * atr)
            tp = entry + 3.0 * atr
        else:
            entry = price
            sl = entry + 1.5 * atr
            resistance = float(last.get("resistance", entry + 2 * atr))
            sl = max(sl, resistance + 0.1 * atr)
            tp = entry - 3.0 * atr

        result.update({"valid": True, "price": entry, "sl": sl, "tp": tp, "atr": atr})
        return result

    def _empty_score(self) -> ConvictionScore:
        return ConvictionScore(
            total=0, max_score=MAX_WEIGHT, breakdown={},
            aligned_direction="SIDEWAYS", htf_bias="SIDEWAYS",
            htf_bias_confidence=0.0, notes=["No data available"],
        )
