"""
Kelly Criterion position sizer.

f* = (b·p − q) / b
  p = win_rate (rolling last 20 trades, default 0.50 if < 10 trades)
  q = 1 − p
  b = avg_win_pct / avg_loss_pct (default 1.5 if < 5 trades)

Kelly is fractional-scaled by quantum conviction [0.10, 0.50].
"""
from __future__ import annotations

import collections
import logging
from typing import Deque, Tuple

logger = logging.getLogger("bot.kelly")

_MIN_TRADES_FOR_KELLY = 10
_MIN_TRADES_FOR_B = 5


class TradeOutcomeBuffer:
    """Ring buffer of (win: bool, pnl_pct: float) tuples."""

    def __init__(self, maxlen: int = 20):
        self._buf: Deque[Tuple[bool, float]] = collections.deque(maxlen=maxlen)

    def record(self, win: bool, pnl_pct: float):
        self._buf.append((win, abs(pnl_pct)))

    @property
    def n(self) -> int:
        return len(self._buf)

    @property
    def win_rate(self) -> float:
        if not self._buf:
            return 0.50
        return sum(1 for w, _ in self._buf if w) / len(self._buf)

    @property
    def avg_win_pct(self) -> float:
        wins = [p for w, p in self._buf if w]
        return sum(wins) / len(wins) if wins else 1.5

    @property
    def avg_loss_pct(self) -> float:
        losses = [p for w, p in self._buf if not w]
        return sum(losses) / len(losses) if losses else 1.0


class KellyPositionSizer:
    """
    Computes risk_pct using fractional Kelly Criterion.
    Falls back to `fallback_risk_pct` if Kelly fraction is negative (negative EV).
    """

    def __init__(self,
                 min_risk_pct: float = 0.5,
                 max_risk_pct: float = 10.0,
                 fallback_risk_pct: float = 5.0):
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        self.fallback_risk_pct = fallback_risk_pct
        self._buffer = TradeOutcomeBuffer(maxlen=20)

    def record_outcome(self, win: bool, pnl_pct: float):
        self._buffer.record(win, pnl_pct)

    def calculate_risk_pct(self, quantum_conviction_float: float = 0.625) -> float:
        """
        Returns risk % of capital for the next trade.

        Parameters
        ----------
        quantum_conviction_float : [0, 1] from QuantumState.conviction_float
                                   (0.625 default ≈ 5/8 legacy score)
        """
        p = self._buffer.win_rate if self._buffer.n >= _MIN_TRADES_FOR_KELLY else 0.50
        q = 1.0 - p
        b = (self._buffer.avg_win_pct / (self._buffer.avg_loss_pct + 1e-9)
             if self._buffer.n >= _MIN_TRADES_FOR_B else 1.5)

        f_star = (b * p - q) / (b + 1e-9)

        if f_star <= 0:
            logger.debug(f"Kelly f*={f_star:.4f} <= 0 — negative EV, using fallback {self.fallback_risk_pct}%")
            return self.fallback_risk_pct

        # Scale Kelly by conviction: [0.10, 0.50]
        kelly_scale = 0.10 + 0.40 * float(quantum_conviction_float)
        risk_pct = f_star * kelly_scale * 100

        clamped = max(self.min_risk_pct, min(self.max_risk_pct, risk_pct))
        logger.debug(f"Kelly f*={f_star:.4f} scale={kelly_scale:.2f} → risk={clamped:.2f}%")
        return round(clamped, 2)

    @property
    def sample_count(self) -> int:
        return self._buffer.n
