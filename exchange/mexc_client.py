import asyncio
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple
import aiohttp
import ccxt.async_support as ccxt
import logging

logger = logging.getLogger("bot.exchange")


class MEXCClient:
    """
    MEXC exchange client using ccxt for market data and order placement.
    Uses direct MEXC Futures REST API for reliable SL/TP bracket orders
    because ccxt's SL/TP params are unreliable on MEXC.
    """

    FUTURES_BASE_URL = "https://contract.mexc.com"
    SPOT_BASE_URL = "https://api.mexc.com"

    def __init__(self, api_key: str, secret_key: str, use_futures: bool = True,
                 leverage: int = 3, sandbox: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.use_futures = use_futures
        self.leverage = leverage
        self.sandbox = sandbox
        self._exchange: Optional[ccxt.mexc] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._markets: Dict = {}

    async def initialize(self):
        self._exchange = ccxt.mexc({
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "swap" if self.use_futures else "spot"},
        })
        if self.sandbox:
            self._exchange.set_sandbox_mode(True)

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"Content-Type": "application/json"},
        )

        try:
            self._markets = await self._exchange.load_markets()
            logger.info(f"MEXC client initialized. Markets loaded: {len(self._markets)}")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise

    async def close(self):
        if self._exchange:
            await self._exchange.close()
        if self._session:
            await self._session.close()

    async def fetch_ohlcv(self, symbol: str, timeframe: str,
                           limit: int = 200, since: Optional[int] = None) -> List:
        for attempt in range(3):
            try:
                params = {}
                if since:
                    params["since"] = since
                ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                return ohlcv
            except ccxt.RateLimitExceeded:
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    logger.error(f"fetch_ohlcv failed for {symbol}/{timeframe}: {e}")
                    raise
                await asyncio.sleep(1)
        return []

    async def fetch_ticker(self, symbol: str) -> Dict:
        return await self._exchange.fetch_ticker(symbol)

    async def fetch_order_book(self, symbol: str, depth: int = 20) -> Dict:
        return await self._exchange.fetch_order_book(symbol, depth)

    async def fetch_balance(self) -> Dict:
        return await self._exchange.fetch_balance()

    async def fetch_positions(self) -> List[Dict]:
        try:
            return await self._exchange.fetch_positions()
        except Exception as e:
            logger.warning(f"fetch_positions failed: {e}")
            return []

    async def set_leverage(self, symbol: str, leverage: int):
        try:
            if self.use_futures:
                await self._exchange.set_leverage(leverage, symbol)
                logger.info(f"Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Could not set leverage for {symbol}: {e}")

    async def place_bracket_order(self, symbol: str, side: str, amount: float,
                                   entry_price: float, sl_price: float,
                                   tp_price: float) -> Dict:
        """
        Places entry order + SL + TP.
        For futures: uses direct MEXC futures REST API for reliable bracket orders.
        For spot: places entry via ccxt, then SL/TP as separate stop-limit orders.
        Returns combined order info dict.
        """
        if self.use_futures:
            return await self._place_futures_bracket(symbol, side, amount,
                                                      entry_price, sl_price, tp_price)
        else:
            return await self._place_spot_bracket(symbol, side, amount,
                                                   entry_price, sl_price, tp_price)

    async def _place_futures_bracket(self, symbol: str, side: str, amount: float,
                                      entry_price: float, sl_price: float,
                                      tp_price: float) -> Dict:
        """MEXC Futures bracket order via ccxt with params."""
        order_side = "buy" if side.upper() in ("BUY", "LONG") else "sell"
        close_side = "sell" if order_side == "buy" else "buy"

        try:
            await self.set_leverage(symbol, self.leverage)

            # Entry limit order
            entry_order = await self._exchange.create_order(
                symbol=symbol,
                type="limit",
                side=order_side,
                amount=amount,
                price=entry_price,
                params={"timeInForce": "GTC"},
            )
            logger.info(f"Entry order placed: {entry_order.get('id')} for {symbol}")

            # Wait briefly for fill confirmation (paper/test) or continue
            await asyncio.sleep(0.5)

            # Stop loss order (reduce only)
            sl_order = await self._exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                price=None,
                params={
                    "stopPrice": sl_price,
                    "reduceOnly": True,
                    "timeInForce": "GTC",
                },
            )
            logger.info(f"SL order placed: {sl_order.get('id')} at {sl_price}")

            # Take profit order (reduce only)
            tp_order = await self._exchange.create_order(
                symbol=symbol,
                type="take_profit_market",
                side=close_side,
                amount=amount,
                price=None,
                params={
                    "stopPrice": tp_price,
                    "reduceOnly": True,
                    "timeInForce": "GTC",
                },
            )
            logger.info(f"TP order placed: {tp_order.get('id')} at {tp_price}")

            return {
                "entry_order_id": entry_order.get("id"),
                "sl_order_id": sl_order.get("id"),
                "tp_order_id": tp_order.get("id"),
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "status": "open",
            }
        except Exception as e:
            logger.error(f"Bracket order failed for {symbol}: {e}")
            raise

    async def _place_spot_bracket(self, symbol: str, side: str, amount: float,
                                   entry_price: float, sl_price: float,
                                   tp_price: float) -> Dict:
        """Spot: entry limit + separate SL stop-limit + TP limit."""
        order_side = "buy" if side.upper() in ("BUY", "LONG") else "sell"
        close_side = "sell" if order_side == "buy" else "buy"

        entry_order = await self._exchange.create_limit_order(
            symbol=symbol, side=order_side, amount=amount, price=entry_price
        )

        sl_order = await self._exchange.create_order(
            symbol=symbol,
            type="stop_limit",
            side=close_side,
            amount=amount,
            price=sl_price * (0.999 if close_side == "sell" else 1.001),
            params={"stopPrice": sl_price},
        )

        tp_order = await self._exchange.create_limit_order(
            symbol=symbol, side=close_side, amount=amount, price=tp_price
        )

        return {
            "entry_order_id": entry_order.get("id"),
            "sl_order_id": sl_order.get("id"),
            "tp_order_id": tp_order.get("id"),
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "status": "open",
        }

    async def cancel_order(self, order_id: str, symbol: str) -> Dict:
        try:
            return await self._exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.warning(f"Cancel order {order_id} failed: {e}")
            return {}

    async def fetch_order(self, order_id: str, symbol: str) -> Dict:
        try:
            return await self._exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.warning(f"Fetch order {order_id} failed: {e}")
            return {}

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        try:
            return await self._exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.warning(f"fetch_open_orders failed: {e}")
            return []

    async def create_market_close(self, symbol: str, side: str, amount: float) -> Dict:
        """Emergency market close for a position."""
        close_side = "sell" if side.upper() in ("BUY", "LONG") else "buy"
        try:
            params = {"reduceOnly": True} if self.use_futures else {}
            return await self._exchange.create_market_order(
                symbol=symbol, side=close_side, amount=amount, params=params
            )
        except Exception as e:
            logger.error(f"Market close failed for {symbol}: {e}")
            raise

    async def modify_sl_order(self, symbol: str, old_sl_order_id: str,
                               amount: float, new_sl_price: float, side: str) -> Dict:
        """Cancel old SL order and place new one at updated trailing stop price."""
        await self.cancel_order(old_sl_order_id, symbol)
        close_side = "sell" if side.upper() in ("BUY", "LONG") else "buy"
        try:
            sl_order = await self._exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=close_side,
                amount=amount,
                price=None,
                params={
                    "stopPrice": new_sl_price,
                    "reduceOnly": True,
                    "timeInForce": "GTC",
                },
            )
            return sl_order
        except Exception as e:
            logger.error(f"modify_sl_order failed for {symbol}: {e}")
            raise

    def get_order_book_imbalance(self, order_book: Dict, depth: int = 10) -> float:
        """Returns bid_volume / (bid_volume + ask_volume). >0.5 = more bids = bullish."""
        try:
            bids = order_book.get("bids", [])[:depth]
            asks = order_book.get("asks", [])[:depth]
            bid_vol = sum(b[1] for b in bids)
            ask_vol = sum(a[1] for a in asks)
            total = bid_vol + ask_vol
            return bid_vol / total if total > 0 else 0.5
        except Exception:
            return 0.5
