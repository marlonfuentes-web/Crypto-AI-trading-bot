import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional
import logging

from strategy.signal_generator import TradingSignal

logger = logging.getLogger("bot.trades")

TRADE_STATE_FILE = "data/cache/open_trades.json"


class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class ManagedTrade:
    trade_id: str
    symbol: str
    side: str                     # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    initial_size: float
    remaining_size: float
    status: str                   # TradeStatus value
    atr_at_entry: float
    entry_order_id: str = ""
    sl_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    trailing_stop_active: bool = False
    trailing_stop_price: Optional[float] = None
    partial_closed: bool = False
    breakeven_set: bool = False
    unrealized_pnl: float = 0.0
    opened_at: int = field(default_factory=lambda: int(time.time() * 1000))
    closed_at: Optional[int] = None
    close_price: Optional[float] = None
    close_reason: str = ""
    ai_score: float = 0.0
    conviction_score: int = 0
    signal_id: str = ""

    def update_pnl(self, current_price: float):
        self.current_price = current_price
        if self.side == "LONG":
            self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_size

    def profit_in_atr(self, current_price: float) -> float:
        if self.atr_at_entry <= 0:
            return 0.0
        if self.side == "LONG":
            return (current_price - self.entry_price) / self.atr_at_entry
        else:
            return (self.entry_price - current_price) / self.atr_at_entry

    def is_tp1_hit(self, price: float) -> bool:
        if self.side == "LONG":
            return price >= self.take_profit_1
        return price <= self.take_profit_1

    def is_tp2_hit(self, price: float) -> bool:
        if self.side == "LONG":
            return price >= self.take_profit_2
        return price <= self.take_profit_2

    def is_sl_hit(self, price: float) -> bool:
        if self.side == "LONG":
            return price <= self.stop_loss
        return price >= self.stop_loss


class TradeManager:
    def __init__(self, exchange_client, risk_manager,
                 mode: str = "paper", notifier=None):
        self.client = exchange_client
        self.risk = risk_manager
        self.mode = mode
        self.notifier = notifier
        self._trades: Dict[str, ManagedTrade] = {}
        self._trade_history: List[ManagedTrade] = []
        self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(TRADE_STATE_FILE):
                with open(TRADE_STATE_FILE) as f:
                    data = json.load(f)
                for t_data in data:
                    if t_data.get("status") == TradeStatus.OPEN.value:
                        trade = ManagedTrade(**t_data)
                        self._trades[trade.trade_id] = trade
                logger.info(f"Loaded {len(self._trades)} open trades from state file")
        except Exception as e:
            logger.warning(f"Could not load trade state: {e}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(TRADE_STATE_FILE), exist_ok=True)
            open_trades = [asdict(t) for t in self._trades.values()]
            with open(TRADE_STATE_FILE, "w") as f:
                json.dump(open_trades, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save trade state: {e}")

    async def open_trade(self, signal: TradingSignal) -> Optional[ManagedTrade]:
        trade_id = str(uuid.uuid4())[:12]

        if self.mode == "paper":
            return await self._open_paper_trade(trade_id, signal)
        else:
            return await self._open_live_trade(trade_id, signal)

    async def _open_paper_trade(self, trade_id: str,
                                  signal: TradingSignal) -> ManagedTrade:
        trade = ManagedTrade(
            trade_id=trade_id,
            symbol=signal.symbol,
            side=signal.direction,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            initial_size=signal.position_size,
            remaining_size=signal.position_size,
            status=TradeStatus.OPEN.value,
            atr_at_entry=signal.atr,
            entry_order_id=f"PAPER_{trade_id}",
            sl_order_id=f"PAPER_SL_{trade_id}",
            tp1_order_id=f"PAPER_TP1_{trade_id}",
            ai_score=signal.ai_score,
            conviction_score=signal.conviction_score,
            signal_id=signal.signal_id,
        )

        self._trades[trade_id] = trade
        self.risk.register_open_trade(trade_id, {
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "size": signal.position_size,
            "side": signal.direction,
        })
        self._save_state()

        logger.info(f"[PAPER] Trade opened: {trade_id} {signal.symbol} "
                    f"{signal.direction} @ {signal.entry_price} "
                    f"SL={signal.stop_loss} TP1={signal.take_profit_1}")

        if self.notifier:
            await self.notifier.send_trade_open(
                symbol=signal.symbol, side=signal.direction,
                entry=signal.entry_price, sl=signal.stop_loss,
                tp1=signal.take_profit_1, size=signal.position_size,
                conviction=signal.conviction_score, ai_score=signal.ai_score,
            )

        return trade

    async def _open_live_trade(self, trade_id: str,
                                signal: TradingSignal) -> Optional[ManagedTrade]:
        try:
            order_result = await self.client.place_bracket_order(
                symbol=signal.symbol,
                side="buy" if signal.direction == "LONG" else "sell",
                amount=signal.position_size,
                entry_price=signal.entry_price,
                sl_price=signal.stop_loss,
                tp_price=signal.take_profit_1,
            )

            trade = ManagedTrade(
                trade_id=trade_id,
                symbol=signal.symbol,
                side=signal.direction,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                initial_size=signal.position_size,
                remaining_size=signal.position_size,
                status=TradeStatus.OPEN.value,
                atr_at_entry=signal.atr,
                entry_order_id=order_result.get("entry_order_id", ""),
                sl_order_id=order_result.get("sl_order_id", ""),
                tp1_order_id=order_result.get("tp_order_id", ""),
                ai_score=signal.ai_score,
                conviction_score=signal.conviction_score,
                signal_id=signal.signal_id,
            )

            self._trades[trade_id] = trade
            self.risk.register_open_trade(trade_id, {
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "size": signal.position_size,
                "side": signal.direction,
            })
            self._save_state()

            if self.notifier:
                await self.notifier.send_trade_open(
                    symbol=signal.symbol, side=signal.direction,
                    entry=signal.entry_price, sl=signal.stop_loss,
                    tp1=signal.take_profit_1, size=signal.position_size,
                    conviction=signal.conviction_score, ai_score=signal.ai_score,
                )

            return trade

        except Exception as e:
            logger.error(f"Failed to open live trade for {signal.symbol}: {e}")
            return None

    async def monitor_trades(self, market_data_service=None):
        """Poll all open trades and manage SL/TP/trailing stops."""
        trades_to_close = []

        for trade_id, trade in list(self._trades.items()):
            if trade.status != TradeStatus.OPEN.value:
                continue

            # Get current price
            current_price = 0.0
            if market_data_service:
                current_price = await market_data_service.get_current_price(trade.symbol)
            else:
                # Fallback: estimate from cached candle
                pass

            if current_price <= 0:
                continue

            trade.update_pnl(current_price)
            profit_atr = trade.profit_in_atr(current_price)

            # Check SL hit (backup — exchange should handle this)
            if trade.is_sl_hit(current_price):
                trades_to_close.append((trade_id, current_price, "stop_loss"))
                continue

            # Check TP2 hit
            if trade.is_tp2_hit(current_price):
                trades_to_close.append((trade_id, current_price, "take_profit_2"))
                continue

            # Partial close at TP1 (50% position)
            if trade.is_tp1_hit(current_price) and not trade.partial_closed:
                await self._handle_partial_close(trade, current_price)

            # Activate trailing stop when profit >= 1.5x ATR
            if profit_atr >= 1.5 and not trade.trailing_stop_active:
                await self._activate_trailing_stop(trade, current_price)

            # Update trailing stop
            if trade.trailing_stop_active:
                await self._update_trailing_stop(trade, current_price)

        # Close finished trades
        for trade_id, price, reason in trades_to_close:
            await self._close_trade(trade_id, price, reason)

        self._save_state()

    async def _handle_partial_close(self, trade: ManagedTrade, price: float):
        close_size = trade.remaining_size * 0.5
        trade.partial_closed = True
        trade.remaining_size -= close_size
        trade.status = TradeStatus.PARTIAL_CLOSE.value

        # Move SL to breakeven
        trade.stop_loss = trade.entry_price
        trade.breakeven_set = True

        if self.mode == "live" and trade.sl_order_id:
            try:
                new_sl = await self.client.modify_sl_order(
                    trade.symbol, trade.sl_order_id,
                    trade.remaining_size, trade.entry_price, trade.side,
                )
                trade.sl_order_id = new_sl.get("id", trade.sl_order_id)
            except Exception as e:
                logger.warning(f"Could not move SL to BE for {trade.trade_id}: {e}")

        pnl = abs(price - trade.entry_price) * close_size
        logger.info(f"Partial close {trade.trade_id}: 50% at {price}, pnl={pnl:.2f}, SL moved to BE")

    async def _activate_trailing_stop(self, trade: ManagedTrade, price: float):
        trade.trailing_stop_active = True
        atr = trade.atr_at_entry
        if trade.side == "LONG":
            trade.trailing_stop_price = price - atr
            # Only update if better than current SL
            if trade.trailing_stop_price > trade.stop_loss:
                trade.stop_loss = trade.trailing_stop_price
        else:
            trade.trailing_stop_price = price + atr
            if trade.trailing_stop_price < trade.stop_loss:
                trade.stop_loss = trade.trailing_stop_price

        logger.info(f"Trailing stop activated for {trade.trade_id} at {trade.trailing_stop_price:.6f}")

    async def _update_trailing_stop(self, trade: ManagedTrade, price: float):
        atr = trade.atr_at_entry
        if trade.side == "LONG":
            new_trail = price - atr
            if new_trail > trade.trailing_stop_price:
                old_stop = trade.trailing_stop_price
                trade.trailing_stop_price = new_trail
                trade.stop_loss = new_trail

                if self.mode == "live" and trade.sl_order_id:
                    try:
                        new_sl_order = await self.client.modify_sl_order(
                            trade.symbol, trade.sl_order_id,
                            trade.remaining_size, new_trail, trade.side,
                        )
                        trade.sl_order_id = new_sl_order.get("id", trade.sl_order_id)
                    except Exception as e:
                        logger.warning(f"Trailing stop update failed {trade.trade_id}: {e}")
        else:
            new_trail = price + atr
            if new_trail < trade.trailing_stop_price:
                trade.trailing_stop_price = new_trail
                trade.stop_loss = new_trail

                if self.mode == "live" and trade.sl_order_id:
                    try:
                        new_sl_order = await self.client.modify_sl_order(
                            trade.symbol, trade.sl_order_id,
                            trade.remaining_size, new_trail, trade.side,
                        )
                        trade.sl_order_id = new_sl_order.get("id", trade.sl_order_id)
                    except Exception as e:
                        logger.warning(f"Trailing stop update failed {trade.trade_id}: {e}")

    async def _close_trade(self, trade_id: str, price: float, reason: str):
        trade = self._trades.get(trade_id)
        if not trade:
            return

        if self.mode == "live":
            try:
                await self.client.create_market_close(
                    trade.symbol, trade.side, trade.remaining_size
                )
                # Cancel any remaining SL/TP orders
                for oid in [trade.sl_order_id, trade.tp1_order_id, trade.tp2_order_id]:
                    if oid and not oid.startswith("PAPER"):
                        await self.client.cancel_order(oid, trade.symbol)
            except Exception as e:
                logger.error(f"Failed to close live trade {trade_id}: {e}")

        trade.status = TradeStatus.CLOSED.value
        trade.close_price = price
        trade.close_reason = reason
        trade.closed_at = int(time.time() * 1000)

        # Calculate final PnL
        if trade.side == "LONG":
            pnl = (price - trade.entry_price) * trade.remaining_size
        else:
            pnl = (trade.entry_price - price) * trade.remaining_size

        # Add TP1 partial close PnL
        partial_pnl = 0.0
        if trade.partial_closed:
            partial_size = trade.initial_size - trade.remaining_size
            if trade.side == "LONG":
                partial_pnl = (trade.take_profit_1 - trade.entry_price) * partial_size
            else:
                partial_pnl = (trade.entry_price - trade.take_profit_1) * partial_size
        total_pnl = pnl + partial_pnl

        self.risk.close_trade(trade_id, price)
        self._trade_history.append(trade)
        del self._trades[trade_id]

        logger.info(f"Trade CLOSED {trade_id} {trade.symbol}: {reason} "
                    f"@ {price:.6f}, PnL={total_pnl:.4f}")

        if self.notifier:
            pnl_pct = (total_pnl / (trade.entry_price * trade.initial_size)) * 100
            await self.notifier.send_trade_close(
                symbol=trade.symbol, side=trade.side,
                entry=trade.entry_price, exit_price=price,
                pnl_usdt=total_pnl, pnl_pct=pnl_pct, reason=reason,
            )

    async def close_all_trades(self, reason: str = "manual_close"):
        """Emergency: close all open positions."""
        for trade_id in list(self._trades.keys()):
            trade = self._trades[trade_id]
            price = trade.current_price or trade.entry_price
            await self._close_trade(trade_id, price, reason)

    @property
    def active_trades(self) -> Dict[str, ManagedTrade]:
        return self._trades

    @property
    def trade_history(self) -> List[ManagedTrade]:
        return self._trade_history

    def has_position_in(self, symbol: str) -> bool:
        return any(t.symbol == symbol for t in self._trades.values())

    def get_open_count(self) -> int:
        return len(self._trades)
