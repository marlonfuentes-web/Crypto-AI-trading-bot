"""
Adaptive Aggression Controller.

Reads entanglement from QuantumState and adjusts the strategy's
signal interval, cooldown, concurrent cap, and daily cap in place.

Entanglement >= 0.8 → HIGH_CLARITY  → aggressive (fast scan, big cap)
Entanglement 0.5–0.8 → MID_CLARITY → moderate
Entanglement < 0.5  → CONSERVATIVE → cautious (slow scan, small cap)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("bot.aggression")


@dataclass
class AggressionMode:
    name: str
    signal_interval_seconds: int
    cooldown_minutes: int
    max_concurrent: int
    daily_cap: int


_MODES = {
    "HIGH_CLARITY":  AggressionMode("HIGH_CLARITY",  120, 15, 5, 50),
    "MID_CLARITY":   AggressionMode("MID_CLARITY",   300, 45, 4, 30),
    "CONSERVATIVE":  AggressionMode("CONSERVATIVE",  600, 45, 3, 30),
}

# Default (no quantum state)
_DEFAULT = _MODES["CONSERVATIVE"]


class AdaptiveAggressionController:
    """
    Mutates strategy + risk_manager settings based on market clarity.
    Call `apply(quantum_state)` once per `run_cycle()`.
    """

    def __init__(self,
                 high_clarity_threshold: float = 0.8,
                 mid_clarity_threshold: float = 0.5):
        self.high_clarity_threshold = high_clarity_threshold
        self.mid_clarity_threshold = mid_clarity_threshold
        self._current_mode: Optional[str] = None

    def apply(self, quantum_state, strategy, risk_manager) -> str:
        """
        Determine aggression mode from entanglement score and apply.

        Parameters
        ----------
        quantum_state : QuantumState or None
        strategy      : ScalpingStrategy instance
        risk_manager  : RiskManager instance (must have daily_trade_cap attr)

        Returns
        -------
        str : mode name applied
        """
        if quantum_state is None:
            mode = _DEFAULT
        else:
            e = float(quantum_state.entanglement_score)
            if e >= self.high_clarity_threshold:
                mode = _MODES["HIGH_CLARITY"]
            elif e >= self.mid_clarity_threshold:
                mode = _MODES["MID_CLARITY"]
            else:
                mode = _MODES["CONSERVATIVE"]

        if mode.name != self._current_mode:
            logger.info(
                f"Aggression mode → {mode.name} "
                f"(interval={mode.signal_interval_seconds}s "
                f"cooldown={mode.cooldown_minutes}m "
                f"concurrent={mode.max_concurrent} "
                f"daily_cap={mode.daily_cap})"
            )
            self._current_mode = mode.name

        # Mutate strategy
        strategy.min_signal_interval_seconds = mode.signal_interval_seconds
        strategy.cooldown_minutes = mode.cooldown_minutes

        # Mutate risk manager
        risk_manager.max_concurrent = mode.max_concurrent
        if hasattr(risk_manager, "daily_trade_cap"):
            risk_manager.daily_trade_cap = mode.daily_cap

        return mode.name
