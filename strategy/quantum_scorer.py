"""
Quantum-inspired conviction scorer.

Each timeframe is modeled as a quantum particle in superposition over
[BULLISH, BEARISH]. Timeframes aligned with the HTF bias interfere
constructively (phase=+1); opposing timeframes interfere destructively
(phase=−1). The final probability is measured via the Born rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("bot.quantum")

TIMEFRAME_WEIGHTS = {
    "1d": 2,
    "4h": 2,
    "1h": 1,
    "30m": 1,
    "15m": 1,
    "5m": 1,
}
MAX_WEIGHT = sum(TIMEFRAME_WEIGHTS.values())  # 8


@dataclass
class QuantumState:
    p_bull: float                   # Born-rule probability for BULLISH
    p_bear: float                   # Born-rule probability for BEARISH
    entanglement_score: float       # [0, 1] cross-TF synchronisation
    superposition_strength: float   # |p_bull - p_bear|, 0=ambiguous, 1=pure
    dominant_direction: str         # "BULLISH" | "BEARISH" | "SIDEWAYS"
    conviction_float: float         # [0, 1] continuous conviction
    legacy_conviction_int: int      # round(conviction_float * 8) for backward compat


class QuantumConvictionScorer:
    """
    Replaces the integer conviction score with a quantum amplitude model.
    Backward-compatible: returns `legacy_conviction_int` (0-8) alongside
    continuous fields so existing signal gates work unchanged.
    """

    def score(
        self,
        tf_scores: Dict,  # Dict[str, TimeframeScore]
        htf_bias: str,
        htf_confidence: float,
    ) -> QuantumState:
        """
        Compute quantum conviction from per-timeframe TrendState signals.

        Parameters
        ----------
        tf_scores : dict of TimeframeScore objects (from MultiTimeframeAnalyzer)
        htf_bias  : "BULLISH" | "BEARISH" | "SIDEWAYS"
        htf_confidence : float [0, 1]
        """
        if not tf_scores or htf_bias == "SIDEWAYS":
            return self._neutral()

        A_bull_total = 0.0
        A_bear_total = 0.0
        directions: List[str] = []

        for tf, tf_score in tf_scores.items():
            weight = TIMEFRAME_WEIGHTS.get(tf, 1)
            htf_weight = weight / MAX_WEIGHT

            state = tf_score.trend_state
            if state is None:
                continue

            # Count bullish signals from TrendState booleans
            bull_sigs = sum([
                bool(getattr(state, "bull_stack", False)),
                bool(getattr(state, "price_above_vwap", False)),
                bool(getattr(state, "supertrend_bullish", False)),
                bool(getattr(state, "higher_highs", False)),
                bool(getattr(state, "higher_lows", False)),
                bool(getattr(state, "slope_ema50", 0.0) > 0),
                bool(getattr(state, "dmi_bull", False)),
            ])
            total_sigs = 7
            p_bull = bull_sigs / total_sigs
            p_bear = (total_sigs - bull_sigs) / total_sigs

            # ADX modulates amplitude (0 → no signal, high ADX → strong signal)
            adx = float(getattr(state, "adx", 25.0))
            adx_scale = min(1.0, adx / 50.0)

            # Phase: +1 if aligned with HTF bias, -1 if opposing
            tf_direction = str(state.direction) if hasattr(state, "direction") else "SIDEWAYS"
            directions.append(tf_direction)
            phase = +1.0 if tf_direction == htf_bias else -1.0

            A_bull_total += phase * math.sqrt(max(p_bull, 0.0)) * adx_scale * htf_weight
            A_bear_total += phase * math.sqrt(max(p_bear, 0.0)) * adx_scale * htf_weight

        # Born rule: probability = amplitude²
        norm = math.sqrt(A_bull_total ** 2 + A_bear_total ** 2 + 1e-12)
        p_bull_final = (A_bull_total / norm) ** 2
        p_bear_final = (A_bear_total / norm) ** 2

        # Entanglement: cross-TF pairwise alignment
        entanglement = self._entanglement(directions, htf_bias)

        # Superposition strength: how pure the state is
        superposition_strength = abs(p_bull_final - p_bear_final)

        # Dominant direction
        if htf_bias == "BULLISH":
            dominant = "BULLISH" if p_bull_final > 0.4 else "SIDEWAYS"
        elif htf_bias == "BEARISH":
            dominant = "BEARISH" if p_bear_final > 0.4 else "SIDEWAYS"
        else:
            dominant = "SIDEWAYS"

        # Conviction float: scale by HTF confidence and superposition
        conviction_float = superposition_strength * htf_confidence
        conviction_int = min(8, round(conviction_float * 8))

        return QuantumState(
            p_bull=round(p_bull_final, 4),
            p_bear=round(p_bear_final, 4),
            entanglement_score=round(entanglement, 4),
            superposition_strength=round(superposition_strength, 4),
            dominant_direction=dominant,
            conviction_float=round(conviction_float, 4),
            legacy_conviction_int=conviction_int,
        )

    def _entanglement(self, directions: List[str], htf_bias: str) -> float:
        """Pairwise cross-TF synchronisation mapped to [0, 1]."""
        if len(directions) < 2:
            return 0.5
        pairs = []
        for i in range(len(directions)):
            for j in range(i + 1, len(directions)):
                a, b = directions[i], directions[j]
                if "SIDEWAYS" in (a, b):
                    pairs.append(0.0)
                elif a == b:
                    pairs.append(+1.0)
                else:
                    pairs.append(-1.0)
        mean_pair = sum(pairs) / len(pairs)
        return (mean_pair + 1.0) / 2.0  # map [-1, 1] → [0, 1]

    def _neutral(self) -> QuantumState:
        return QuantumState(
            p_bull=0.5, p_bear=0.5,
            entanglement_score=0.5,
            superposition_strength=0.0,
            dominant_direction="SIDEWAYS",
            conviction_float=0.0,
            legacy_conviction_int=0,
        )
