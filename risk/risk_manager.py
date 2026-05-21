import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("bot.risk")

STATE_FILE = "data/cache/risk_state.json"


@dataclass
class DailyStats:
    date: str                    # ISO date string
    starting_capital: float
    realized_pnl: float = 0.0
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    max_drawdown_pct: float = 0.0
    peak_capital: float = 0.0
    is_trading_halted: bool = False
    halt_reason: str = ""

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return self.wins / self.trades_taken

    @property
    def daily_pnl_pct(self) -> float:
        if self.starting_capital <= 0:
            return 0.0
        return self.realized_pnl / self.starting_capital * 100


@dataclass
class PositionSizeResult:
    units: float
    notional_usdt: float
    risk_amount_usdt: float
    risk_pct: float
    sl_distance_pct: float
    approved: bool
    rejection_reason: Optional[str] = None


class RiskManager:
    def __init__(self, capital: float, max_risk_pct: float = 5.0,
                 max_daily_loss_pct: float = 10.0, max_drawdown_pct: float = 20.0,
                 max_concurrent: int = 3, min_rr: float = 2.0,
                 sl_atr_multiplier: float = 1.5):
        self.capital = capital
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_concurrent = max_concurrent
        self.min_rr = min_rr
        self.sl_atr_multiplier = sl_atr_multiplier

        self.daily_stats = self._load_or_create_daily_stats()
        self._open_positions: Dict[str, Dict] = {}

    def _load_or_create_daily_stats(self) -> DailyStats:
        today = date.today().isoformat()
        try:
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE) as f:
                    data = json.load(f)
                if data.get("date") == today:
                    return DailyStats(**data)
        except Exception as e:
            logger.warning(f"Could not load risk state: {e}")
        return DailyStats(date=today, starting_capital=self.capital,
                          peak_capital=self.capital)

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(asdict(self.daily_stats), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")

    def calculate_position_size(self, entry: float, stop_loss: float,
                                  capital: float, atr: float = None) -> PositionSizeResult:
        if entry <= 0 or stop_loss <= 0:
            return PositionSizeResult(0, 0, 0, 0, 0, False, "Invalid prices")

        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            return PositionSizeResult(0, 0, 0, 0, 0, False, "Zero SL distance")

        # ATR-based SL sanity check
        if atr and atr > 0:
            min_sl = 0.5 * atr
            max_sl = 3.0 * atr
            if sl_distance < min_sl:
                sl_distance = min_sl
                stop_loss = entry - sl_distance  # recalculate
            elif sl_distance > max_sl:
                return PositionSizeResult(0, 0, 0, 0, 0, False,
                                          f"SL too wide: {sl_distance:.6f} > {max_sl:.6f}")

        risk_amount = capital * (self.max_risk_pct / 100)
        units = risk_amount / sl_distance
        notional = units * entry

        # Cap at 50% of capital per position (allows 5% risk to work at typical SL distances)
        max_notional = capital * 0.50
        if notional > max_notional:
            units = max_notional / entry
            notional = max_notional

        sl_distance_pct = sl_distance / entry * 100

        return PositionSizeResult(
            units=round(units, 6),
            notional_usdt=round(notional, 2),
            risk_amount_usdt=round(risk_amount, 2),
            risk_pct=self.max_risk_pct,
            sl_distance_pct=round(sl_distance_pct, 4),
            approved=True,
        )

    def can_open_trade(self, symbol: str = None) -> Tuple[bool, str]:
        """Final gate check before placing an order."""
        if self.daily_stats.is_trading_halted:
            return False, f"Trading halted: {self.daily_stats.halt_reason}"

        if self.daily_stats.trades_taken >= 30:
            return False, "Daily trade limit (30) reached"

        if len(self._open_positions) >= self.max_concurrent:
            return False, f"Max concurrent trades ({self.max_concurrent}) reached"

        daily_loss_pct = abs(min(self.daily_stats.daily_pnl_pct, 0))
        if daily_loss_pct >= self.max_daily_loss_pct:
            self._halt_trading(f"Daily loss limit {self.max_daily_loss_pct}% reached")
            return False, f"Daily loss limit {self.max_daily_loss_pct}% reached"

        # Hard halt if win rate below 35% after 10 trades — the strategy is not working today
        if self.daily_stats.trades_taken >= 10 and self.daily_stats.win_rate < 0.35:
            self._halt_trading(f"Win rate {self.daily_stats.win_rate:.0%} below 35% after {self.daily_stats.trades_taken} trades")
            return False, f"Win rate circuit breaker: {self.daily_stats.win_rate:.0%}"

        return True, "OK"

    def get_portfolio_heat(self) -> float:
        """Sum of all open position risk as % of capital."""
        total_risk = 0.0
        for pos in self._open_positions.values():
            entry = pos.get("entry_price", 0)
            sl = pos.get("stop_loss", 0)
            size = pos.get("size", 0)
            if entry > 0 and sl > 0:
                risk = abs(entry - sl) * size
                total_risk += risk
        return total_risk / self.capital * 100 if self.capital > 0 else 0.0

    def register_open_trade(self, trade_id: str, trade_info: Dict):
        self._open_positions[trade_id] = trade_info

    def close_trade(self, trade_id: str, exit_price: float):
        if trade_id not in self._open_positions:
            return
        pos = self._open_positions.pop(trade_id)
        entry = pos.get("entry_price", exit_price)
        size = pos.get("size", 0)
        side = pos.get("side", "BUY")

        if side.upper() in ("BUY", "LONG"):
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size

        self.daily_stats.realized_pnl += pnl
        self.daily_stats.trades_taken += 1
        if pnl > 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

        # Update drawdown
        current_capital = self.capital + self.daily_stats.realized_pnl
        if current_capital > self.daily_stats.peak_capital:
            self.daily_stats.peak_capital = current_capital

        if self.daily_stats.peak_capital > 0:
            dd = (self.daily_stats.peak_capital - current_capital) / self.daily_stats.peak_capital * 100
            self.daily_stats.max_drawdown_pct = max(self.daily_stats.max_drawdown_pct, dd)
            if dd >= self.max_drawdown_pct:
                self._halt_trading(f"Max drawdown {self.max_drawdown_pct}% hit")

        self._check_daily_loss_limit()
        self._save_state()
        return pnl

    def _check_daily_loss_limit(self):
        if self.daily_stats.daily_pnl_pct <= -self.max_daily_loss_pct:
            self._halt_trading(f"Daily loss {abs(self.daily_stats.daily_pnl_pct):.2f}% >= {self.max_daily_loss_pct}%")

    def _halt_trading(self, reason: str):
        if not self.daily_stats.is_trading_halted:
            self.daily_stats.is_trading_halted = True
            self.daily_stats.halt_reason = reason
            logger.warning(f"TRADING HALTED: {reason}")
            self._save_state()

    def reset_daily_stats(self):
        """Called at UTC midnight."""
        today = date.today().isoformat()
        current_capital = self.capital + self.daily_stats.realized_pnl
        self.daily_stats = DailyStats(
            date=today,
            starting_capital=current_capital,
            peak_capital=current_capital,
        )
        self.capital = current_capital
        self._save_state()
        logger.info(f"Daily stats reset. New capital: ${current_capital:.2f}")

    def get_summary(self) -> Dict:
        return {
            "date": self.daily_stats.date,
            "capital": self.capital,
            "realized_pnl": round(self.daily_stats.realized_pnl, 2),
            "pnl_pct": round(self.daily_stats.daily_pnl_pct, 2),
            "trades": self.daily_stats.trades_taken,
            "wins": self.daily_stats.wins,
            "losses": self.daily_stats.losses,
            "win_rate": round(self.daily_stats.win_rate * 100, 1),
            "open_positions": len(self._open_positions),
            "portfolio_heat_pct": round(self.get_portfolio_heat(), 2),
            "halted": self.daily_stats.is_trading_halted,
            "halt_reason": self.daily_stats.halt_reason,
        }
