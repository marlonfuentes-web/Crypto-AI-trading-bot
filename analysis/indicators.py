import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger("bot.indicators")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available, using fallback calculations")


class TechnicalIndicators:
    """
    Stateless computation class. All methods append indicator columns to a DataFrame.
    Uses iloc[-2] pattern awareness — callers should pass closed-candle DataFrames.
    """

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [9, 21, 50, 200]) -> pd.DataFrame:
        for p in periods:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            tp_vol = typical_price * df["volume"]
            df["vwap"] = tp_vol.cumsum() / df["volume"].cumsum()
        except Exception:
            df["vwap"] = df["close"]
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        try:
            if PANDAS_TA_AVAILABLE:
                adx_df = ta.adx(df["high"], df["low"], df["close"], length=period)
                if adx_df is not None and not adx_df.empty:
                    df["adx"] = adx_df.get(f"ADX_{period}", adx_df.iloc[:, 0])
                    df["dmp"] = adx_df.get(f"DMP_{period}", adx_df.iloc[:, 1] if adx_df.shape[1] > 1 else 25)
                    df["dmn"] = adx_df.get(f"DMN_{period}", adx_df.iloc[:, 2] if adx_df.shape[1] > 2 else 25)
                    return df

            # Fallback: manual ADX
            high = df["high"]
            low = df["low"]
            close = df["close"]

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm < plus_dm] = 0

            tr = pd.concat([high - low, (high - close.shift()).abs(),
                            (low - close.shift()).abs()], axis=1).max(axis=1)

            atr_n = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr_n)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr_n)
            dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)

            df["adx"] = dx.rolling(period).mean()
            df["dmp"] = plus_di
            df["dmn"] = minus_di
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")
            df["adx"] = 25.0
            df["dmp"] = 25.0
            df["dmn"] = 25.0
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        try:
            delta = df["close"].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50)
        except Exception:
            df["rsi"] = 50.0
        return df

    @staticmethod
    def add_stoch_rsi(df: pd.DataFrame, rsi_period: int = 14,
                      stoch_period: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
        try:
            if "rsi" not in df.columns:
                TechnicalIndicators.add_rsi(df, rsi_period)

            rsi = df["rsi"]
            rsi_low = rsi.rolling(stoch_period).min()
            rsi_high = rsi.rolling(stoch_period).max()
            stoch = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-10) * 100
            df["stochrsi_k"] = stoch.rolling(k).mean().fillna(50)
            df["stochrsi_d"] = df["stochrsi_k"].rolling(d).mean().fillna(50)
        except Exception:
            df["stochrsi_k"] = 50.0
            df["stochrsi_d"] = 50.0
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        try:
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        except Exception:
            df["macd"] = 0.0
            df["macd_signal"] = 0.0
            df["macd_hist"] = 0.0
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        try:
            rolling_mean = df["close"].rolling(period).mean()
            rolling_std = df["close"].rolling(period).std()
            df["bb_upper"] = rolling_mean + std * rolling_std
            df["bb_mid"] = rolling_mean
            df["bb_lower"] = rolling_mean - std * rolling_std
            width = df["bb_upper"] - df["bb_lower"]
            df["bb_width"] = width
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (width + 1e-10)
        except Exception:
            df["bb_upper"] = df["close"] * 1.02
            df["bb_mid"] = df["close"]
            df["bb_lower"] = df["close"] * 0.98
            df["bb_width"] = df["close"] * 0.04
            df["bb_pct"] = 0.5
        return df

    @staticmethod
    def add_keltner_channels(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> pd.DataFrame:
        try:
            if "atr" not in df.columns:
                TechnicalIndicators.add_atr(df, 14)
            ema_mid = df["close"].ewm(span=period, adjust=False).mean()
            df["kc_upper"] = ema_mid + atr_mult * df["atr"]
            df["kc_mid"] = ema_mid
            df["kc_lower"] = ema_mid - atr_mult * df["atr"]
        except Exception:
            df["kc_upper"] = df["close"] * 1.02
            df["kc_mid"] = df["close"]
            df["kc_lower"] = df["close"] * 0.98
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            df["atr"] = tr.rolling(period).mean()
            df["atr"] = df["atr"].fillna(df["close"] * 0.01)
        except Exception:
            df["atr"] = df["close"] * 0.01
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        try:
            direction = np.sign(df["close"].diff()).fillna(0)
            obv = (direction * df["volume"]).cumsum()
            df["obv"] = obv
            df["obv_ema"] = obv.ewm(span=14, adjust=False).mean()
        except Exception:
            df["obv"] = 0.0
            df["obv_ema"] = 0.0
        return df

    @staticmethod
    def add_cvd(df: pd.DataFrame) -> pd.DataFrame:
        try:
            bullish = (df["close"] > df["open"]).astype(float)
            bearish = (df["close"] < df["open"]).astype(float)
            df["cvd_delta"] = bullish * df["volume"] - bearish * df["volume"]
            df["cvd"] = df["cvd_delta"].cumsum()
        except Exception:
            df["cvd_delta"] = 0.0
            df["cvd"] = 0.0
        return df

    @staticmethod
    def add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        df["momentum"] = df["close"].diff(period).fillna(0)
        df["roc"] = df["close"].pct_change(period).fillna(0) * 100
        return df

    @staticmethod
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        try:
            rolling_high = df["high"].rolling(window).max()
            rolling_low = df["low"].rolling(window).min()
            df["resistance"] = rolling_high
            df["support"] = rolling_low
            price = df["close"]
            df["sr_dist_resistance"] = (rolling_high - price).abs() / price
            df["sr_dist_support"] = (price - rolling_low).abs() / price
        except Exception:
            df["resistance"] = df["high"]
            df["support"] = df["low"]
            df["sr_dist_resistance"] = 0.02
            df["sr_dist_support"] = 0.02
        return df

    @staticmethod
    def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        try:
            if "atr" not in df.columns:
                TechnicalIndicators.add_atr(df, period)

            hl2 = (df["high"] + df["low"]) / 2
            upper_band = hl2 + multiplier * df["atr"]
            lower_band = hl2 - multiplier * df["atr"]

            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)

            for i in range(1, len(df)):
                prev_upper = upper_band.iloc[i - 1]
                prev_lower = lower_band.iloc[i - 1]
                curr_close = df["close"].iloc[i]
                prev_close = df["close"].iloc[i - 1]

                # Adjust bands
                if lower_band.iloc[i] < prev_lower or prev_close < prev_lower:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = prev_lower

                if upper_band.iloc[i] > prev_upper or prev_close > prev_upper:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = prev_upper

                prev_dir = direction.iloc[i - 1] if i > 1 else 1
                if curr_close <= lower_band.iloc[i]:
                    direction.iloc[i] = -1
                elif curr_close >= upper_band.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = prev_dir

                supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

            df["supertrend"] = supertrend
            df["supertrend_dir"] = direction.fillna(1)
        except Exception:
            df["supertrend"] = df["close"]
            df["supertrend_dir"] = 1
        return df

    @staticmethod
    def add_squeeze_momentum(df: pd.DataFrame) -> pd.DataFrame:
        try:
            if "bb_upper" not in df.columns:
                TechnicalIndicators.add_bollinger_bands(df)
            if "kc_upper" not in df.columns:
                TechnicalIndicators.add_keltner_channels(df)
            df["squeeze_on"] = (
                (df["bb_upper"] < df["kc_upper"]) &
                (df["bb_lower"] > df["kc_lower"])
            ).astype(int)
        except Exception:
            df["squeeze_on"] = 0
        return df

    @staticmethod
    def add_fibonacci_levels(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        try:
            recent = df.tail(lookback)
            swing_high = recent["high"].max()
            swing_low = recent["low"].min()
            diff = swing_high - swing_low
            df["fib_236"] = swing_high - 0.236 * diff
            df["fib_382"] = swing_high - 0.382 * diff
            df["fib_500"] = swing_high - 0.500 * diff
            df["fib_618"] = swing_high - 0.618 * diff
            df["fib_786"] = swing_high - 0.786 * diff
            df["swing_high"] = swing_high
            df["swing_low"] = swing_low
        except Exception:
            for level in ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]:
                df[level] = df["close"]
            df["swing_high"] = df["high"].max()
            df["swing_low"] = df["low"].min()
        return df

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) < 30:
            return df

        df = df.copy()
        TechnicalIndicators.add_ema(df)
        TechnicalIndicators.add_vwap(df)
        TechnicalIndicators.add_atr(df)
        TechnicalIndicators.add_adx(df)
        TechnicalIndicators.add_rsi(df)
        TechnicalIndicators.add_stoch_rsi(df)
        TechnicalIndicators.add_macd(df)
        TechnicalIndicators.add_bollinger_bands(df)
        TechnicalIndicators.add_keltner_channels(df)
        TechnicalIndicators.add_obv(df)
        TechnicalIndicators.add_cvd(df)
        TechnicalIndicators.add_momentum(df)
        TechnicalIndicators.add_support_resistance(df)
        TechnicalIndicators.add_supertrend(df)
        TechnicalIndicators.add_squeeze_momentum(df)
        TechnicalIndicators.add_fibonacci_levels(df)

        from analysis.candle_patterns import CandlePatternAnalyzer
        cp = CandlePatternAnalyzer()
        df = cp.add_to_dataframe(df)

        return df
