import asyncio
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logger = logging.getLogger("bot.data")

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400, "1w": 604800,
}

CANDLE_LIMITS = {
    "5m": 200, "15m": 200, "30m": 150, "1h": 100, "4h": 100, "1d": 100,
}


class CandleCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}

    def get(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        key = (symbol, timeframe)
        if key in self._cache:
            return self._cache[key][0]
        return None

    def set(self, symbol: str, timeframe: str, df: pd.DataFrame):
        self._cache[(symbol, timeframe)] = (df, time.time())

    def is_stale(self, symbol: str, timeframe: str, max_age_factor: float = 0.8) -> bool:
        key = (symbol, timeframe)
        if key not in self._cache:
            return True
        _, cached_at = self._cache[key]
        tf_seconds = TIMEFRAME_SECONDS.get(timeframe, 300)
        max_age = tf_seconds * max_age_factor
        return (time.time() - cached_at) > max_age

    def invalidate(self, symbol: str, timeframe: str):
        self._cache.pop((symbol, timeframe), None)

    def size(self) -> int:
        return len(self._cache)


class MarketDataService:
    def __init__(self, exchange_client, config):
        self.client = exchange_client
        self.config = config
        self.cache = CandleCache()
        self._last_candle_timestamps: Dict[Tuple[str, str], int] = {}

    async def get_candles(self, symbol: str, timeframe: str,
                           limit: Optional[int] = None,
                           force_refresh: bool = False) -> pd.DataFrame:
        if not force_refresh and not self.cache.is_stale(symbol, timeframe):
            cached = self.cache.get(symbol, timeframe)
            if cached is not None and len(cached) > 10:
                return cached

        candle_limit = limit or CANDLE_LIMITS.get(timeframe, 150)

        try:
            raw = await self.client.fetch_ohlcv(symbol, timeframe, limit=candle_limit)
            if not raw or len(raw) < 10:
                logger.warning(f"Insufficient candle data for {symbol}/{timeframe}: {len(raw) if raw else 0}")
                cached = self.cache.get(symbol, timeframe)
                return cached if cached is not None else pd.DataFrame()

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df.astype(float)

            # Remove the last (incomplete/forming) candle
            if len(df) > 1:
                df = df.iloc[:-1]

            self.cache.set(symbol, timeframe, df)
            return df

        except Exception as e:
            logger.error(f"get_candles error {symbol}/{timeframe}: {e}")
            cached = self.cache.get(symbol, timeframe)
            return cached if cached is not None else pd.DataFrame()

    async def get_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        timeframes = self.config.trading.timeframes
        tasks = [self.get_candles(symbol, tf) for tf in timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tf_data = {}
        for tf, result in zip(timeframes, results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to fetch {symbol}/{tf}: {result}")
                cached = self.cache.get(symbol, tf)
                if cached is not None:
                    tf_data[tf] = cached
            elif isinstance(result, pd.DataFrame) and len(result) > 10:
                tf_data[tf] = result

        return tf_data

    async def get_current_price(self, symbol: str) -> float:
        try:
            ticker = await self.client.fetch_ticker(symbol)
            return float(ticker.get("last", 0))
        except Exception as e:
            logger.warning(f"get_current_price failed for {symbol}: {e}")
            df = self.cache.get(symbol, "5m")
            if df is not None and len(df) > 0:
                return float(df["close"].iloc[-1])
            return 0.0

    async def get_order_book_imbalance(self, symbol: str, depth: int = 10) -> float:
        try:
            ob = await self.client.fetch_order_book(symbol, depth)
            return self.client.get_order_book_imbalance(ob, depth)
        except Exception as e:
            logger.warning(f"Order book fetch failed for {symbol}: {e}")
            return 0.5

    def has_new_candle(self, symbol: str, timeframe: str) -> bool:
        df = self.cache.get(symbol, timeframe)
        if df is None or len(df) == 0:
            return True
        key = (symbol, timeframe)
        last_ts = int(df.index[-1].timestamp() * 1000)
        prev_ts = self._last_candle_timestamps.get(key, 0)
        if last_ts > prev_ts:
            self._last_candle_timestamps[key] = last_ts
            return True
        return False

    async def warm_cache(self, symbols: List[str]):
        logger.info(f"Warming cache for {len(symbols)} symbols × {len(self.config.trading.timeframes)} timeframes...")
        tasks = []
        for symbol in symbols:
            for tf in self.config.trading.timeframes:
                tasks.append(self.get_candles(symbol, tf, force_refresh=True))

        # Batch to avoid rate limits
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch, return_exceptions=True)
            await asyncio.sleep(0.5)

        logger.info(f"Cache warmed. Total entries: {self.cache.size()}")
