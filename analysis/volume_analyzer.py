import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger("bot.volume")


@dataclass
class VolumeState:
    is_surge: bool
    surge_ratio: float
    obv_trend: str             # "UP", "DOWN", "FLAT"
    cvd_bullish: bool
    cvd_divergence: bool       # price down but CVD up = hidden bull
    order_book_imbalance: float
    institutional_candle: bool
    anomaly_score: float       # z-score


class VolumeAnalyzer:
    def __init__(self, surge_multiplier: float = 1.5, window: int = 20):
        self.surge_multiplier = surge_multiplier
        self.window = window

    def analyze(self, df: pd.DataFrame,
                order_book_imbalance: Optional[float] = None) -> VolumeState:
        if df is None or len(df) < self.window + 5:
            return VolumeState(
                is_surge=False, surge_ratio=1.0, obv_trend="FLAT",
                cvd_bullish=True, cvd_divergence=False,
                order_book_imbalance=0.5, institutional_candle=False,
                anomaly_score=0.0,
            )

        is_surge, surge_ratio = self.detect_volume_surge(df)
        obv_trend = self.analyze_obv_trend(df)
        cvd_bullish = self.is_cvd_bullish(df)
        cvd_divergence = self.detect_cvd_divergence(df)
        anomaly = self.compute_volume_anomaly_score(df)
        institutional = self.is_institutional_candle(df)

        return VolumeState(
            is_surge=is_surge,
            surge_ratio=surge_ratio,
            obv_trend=obv_trend,
            cvd_bullish=cvd_bullish,
            cvd_divergence=cvd_divergence,
            order_book_imbalance=order_book_imbalance if order_book_imbalance is not None else 0.5,
            institutional_candle=institutional,
            anomaly_score=anomaly,
        )

    def detect_volume_surge(self, df: pd.DataFrame) -> tuple:
        try:
            avg_vol = df["volume"].rolling(self.window).mean().iloc[-1]
            current_vol = float(df["volume"].iloc[-1])
            if avg_vol <= 0:
                return False, 1.0
            ratio = current_vol / avg_vol
            return ratio >= self.surge_multiplier, round(ratio, 3)
        except Exception:
            return False, 1.0

    def analyze_obv_trend(self, df: pd.DataFrame) -> str:
        if "obv" not in df.columns or len(df) < 10:
            return "FLAT"
        try:
            obv_recent = df["obv"].tail(5)
            slope = np.polyfit(range(len(obv_recent)), obv_recent.values, 1)[0]
            avg_obv = float(df["obv"].tail(20).mean())
            threshold = abs(avg_obv) * 0.01 if avg_obv != 0 else 1.0
            if slope > threshold:
                return "UP"
            elif slope < -threshold:
                return "DOWN"
            return "FLAT"
        except Exception:
            return "FLAT"

    def is_cvd_bullish(self, df: pd.DataFrame) -> bool:
        if "cvd" not in df.columns or len(df) < 5:
            return True
        try:
            cvd_recent = df["cvd"].tail(5)
            return float(cvd_recent.iloc[-1]) > float(cvd_recent.iloc[0])
        except Exception:
            return True

    def detect_cvd_divergence(self, df: pd.DataFrame) -> bool:
        """Bullish divergence: price lower but CVD higher (smart money buying)."""
        if "cvd" not in df.columns or len(df) < 10:
            return False
        try:
            price_now = float(df["close"].iloc[-1])
            price_prev = float(df["close"].iloc[-5])
            cvd_now = float(df["cvd"].iloc[-1])
            cvd_prev = float(df["cvd"].iloc[-5])
            return price_now < price_prev and cvd_now > cvd_prev
        except Exception:
            return False

    def compute_volume_anomaly_score(self, df: pd.DataFrame, window: int = 50) -> float:
        try:
            recent = df["volume"].tail(window)
            mean = float(recent.mean())
            std = float(recent.std())
            if std == 0:
                return 0.0
            current = float(df["volume"].iloc[-1])
            return round((current - mean) / std, 3)
        except Exception:
            return 0.0

    def is_institutional_candle(self, df: pd.DataFrame,
                                  body_ratio_min: float = 0.6) -> bool:
        """High volume + large candle body = institutional activity."""
        try:
            last = df.iloc[-1]
            candle_range = float(last["high"]) - float(last["low"])
            body = abs(float(last["close"]) - float(last["open"]))
            if candle_range == 0:
                return False
            body_ratio = body / candle_range

            avg_vol = float(df["volume"].tail(20).mean())
            current_vol = float(last["volume"])
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            return body_ratio >= body_ratio_min and vol_ratio >= 2.0
        except Exception:
            return False

    def get_volume_confirmation(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Confirm entry direction with volume:
        LONG: OBV trending up + CVD bullish + no bearish surge
        SHORT: OBV trending down + CVD bearish
        """
        state = self.analyze(df)
        if direction.upper() in ("LONG", "BUY", "BULLISH"):
            return (
                state.obv_trend in ("UP", "FLAT") and
                state.cvd_bullish and
                state.surge_ratio >= 1.0
            )
        else:
            return (
                state.obv_trend in ("DOWN", "FLAT") and
                not state.cvd_bullish and
                state.surge_ratio >= 1.0
            )
