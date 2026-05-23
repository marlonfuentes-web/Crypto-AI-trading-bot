import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from strategy.multi_timeframe import ConvictionScore
from analysis.volume_analyzer import VolumeState

logger = logging.getLogger("bot.features")

FEATURE_NAMES = [
    # Trend (7)
    "ema9_21_spread_pct",
    "ema21_50_spread_pct",
    "price_vwap_spread_pct",
    "adx_value",
    "dmi_directional_bias",
    "ema50_slope",
    "htf_bias_encoded",
    # Momentum (6)
    "rsi_normalized",
    "stochrsi_k",
    "stochrsi_divergence",
    "macd_hist_normalized",
    "momentum_10_normalized",
    "roc_5",
    # Volatility (4)
    "atr_pct",
    "bb_pct_b",
    "bb_width_normalized",
    "squeeze_on",
    # Volume (5)
    "volume_surge_ratio",
    "obv_slope",
    "cvd_delta_normalized",
    "order_book_imbalance",
    "volume_anomaly_zscore",
    # Structure (4)
    "distance_nearest_sr",
    "price_vs_fib_level",
    "near_key_level",
    "market_structure_score",
    # MTF (2)
    "conviction_score_normalized",
    "timeframe_alignment_ratio",
    # Quantum (4)
    "quantum_bull_amplitude",
    "quantum_bear_amplitude",
    "entanglement_score",
    "superposition_strength",
]


class FeatureEngineer:
    def extract_features(self, df_5m: pd.DataFrame, conviction: ConvictionScore,
                          volume_state: VolumeState,
                          order_book_imbalance: float = 0.5,
                          quantum_state=None) -> Dict[str, float]:
        features = {}
        last = df_5m.iloc[-1] if len(df_5m) > 0 else pd.Series(dtype=float)
        close = float(last.get("close", 1.0))

        # --- Trend Features ---
        ema9 = float(last.get("ema_9", close))
        ema21 = float(last.get("ema_21", close))
        ema50 = float(last.get("ema_50", close))
        vwap = float(last.get("vwap", close))
        dmp = float(last.get("dmp", 25))
        dmn = float(last.get("dmn", 25))

        features["ema9_21_spread_pct"] = (ema9 - ema21) / (ema21 + 1e-10) * 100
        features["ema21_50_spread_pct"] = (ema21 - ema50) / (ema50 + 1e-10) * 100
        features["price_vwap_spread_pct"] = (close - vwap) / (vwap + 1e-10) * 100
        features["adx_value"] = float(last.get("adx", 20))
        dmi_total = dmp + dmn + 1e-10
        features["dmi_directional_bias"] = (dmp - dmn) / dmi_total

        # EMA50 slope (5-period)
        if "ema_50" in df_5m.columns and len(df_5m) >= 6:
            ema50_now = float(df_5m["ema_50"].iloc[-1])
            ema50_5ago = float(df_5m["ema_50"].iloc[-6])
            features["ema50_slope"] = (ema50_now - ema50_5ago) / (ema50_5ago + 1e-10) * 100
        else:
            features["ema50_slope"] = 0.0

        htf_map = {"BULLISH": 1.0, "BEARISH": -1.0, "SIDEWAYS": 0.0}
        features["htf_bias_encoded"] = htf_map.get(conviction.htf_bias, 0.0)

        # --- Momentum Features ---
        rsi = float(last.get("rsi", 50))
        features["rsi_normalized"] = (rsi - 50) / 50
        features["stochrsi_k"] = float(last.get("stochrsi_k", 50))

        # RSI divergence: price lower but RSI higher (or vice versa) over 5 candles
        if len(df_5m) >= 6:
            price_diff = float(df_5m["close"].iloc[-1]) - float(df_5m["close"].iloc[-6])
            rsi_diff = float(df_5m["rsi"].iloc[-1]) - float(df_5m["rsi"].iloc[-6]) if "rsi" in df_5m.columns else 0.0
            features["stochrsi_divergence"] = 1.0 if (price_diff < 0 and rsi_diff > 0) else 0.0
        else:
            features["stochrsi_divergence"] = 0.0

        atr = float(last.get("atr", close * 0.01))
        macd_hist = float(last.get("macd_hist", 0))
        features["macd_hist_normalized"] = macd_hist / (atr + 1e-10)

        momentum = float(last.get("momentum", 0))
        features["momentum_10_normalized"] = momentum / (close + 1e-10)
        features["roc_5"] = float(last.get("roc", 0))

        # --- Volatility Features ---
        features["atr_pct"] = atr / (close + 1e-10) * 100
        features["bb_pct_b"] = float(last.get("bb_pct", 0.5))
        bb_width = float(last.get("bb_width", atr * 2))
        features["bb_width_normalized"] = bb_width / (close + 1e-10) * 100
        features["squeeze_on"] = float(last.get("squeeze_on", 0))

        # --- Volume Features ---
        features["volume_surge_ratio"] = volume_state.surge_ratio
        # OBV slope
        if "obv" in df_5m.columns and len(df_5m) >= 6:
            obv_now = float(df_5m["obv"].iloc[-1])
            obv_5ago = float(df_5m["obv"].iloc[-6])
            avg_vol = float(df_5m["volume"].tail(20).mean())
            features["obv_slope"] = (obv_now - obv_5ago) / (avg_vol * 5 + 1e-10)
        else:
            features["obv_slope"] = 0.0

        avg_vol = float(df_5m["volume"].tail(20).mean()) if "volume" in df_5m.columns else 1.0
        cvd_delta = float(last.get("cvd_delta", 0))
        features["cvd_delta_normalized"] = cvd_delta / (avg_vol + 1e-10)
        features["order_book_imbalance"] = order_book_imbalance
        features["volume_anomaly_zscore"] = min(volume_state.anomaly_score, 5.0)

        # --- Structure Features ---
        dist_res = float(last.get("sr_dist_resistance", 0.02))
        dist_sup = float(last.get("sr_dist_support", 0.02))
        features["distance_nearest_sr"] = min(dist_res, dist_sup) * 100

        # Fibonacci level encoding: which zone is price in?
        fib_levels = ["fib_236", "fib_382", "fib_500", "fib_618", "fib_786"]
        fib_values = [(float(last.get(f, close)), i) for i, f in enumerate(fib_levels)]
        fib_distances = [(abs(close - fv), idx) for fv, idx in fib_values if fv > 0]
        if fib_distances:
            closest_idx = min(fib_distances, key=lambda x: x[0])[1]
            features["price_vs_fib_level"] = (closest_idx - 2) / 2.0
        else:
            features["price_vs_fib_level"] = 0.0

        # Near key level (any of VWAP, EMA21, EMA50, BB mid, S/R)
        near_checks = []
        for col in ["vwap", "ema_21", "ema_50", "bb_mid", "support", "resistance"]:
            level = float(last.get(col, 0))
            if level > 0:
                near_checks.append(abs(close - level) / close <= 0.004)
        features["near_key_level"] = 1.0 if any(near_checks) else 0.0

        # Market structure: BOS encoded
        supertrend_dir = float(last.get("supertrend_dir", 1))
        features["market_structure_score"] = supertrend_dir  # 1=bull, -1=bear

        # --- MTF Features ---
        features["conviction_score_normalized"] = conviction.total / (conviction.max_score + 1e-10)
        total_tfs = len(conviction.breakdown)
        if total_tfs > 0:
            aligned = sum(1 for v in conviction.breakdown.values() if v > 0)
            features["timeframe_alignment_ratio"] = aligned / total_tfs
        else:
            features["timeframe_alignment_ratio"] = 0.5

        # --- Quantum Features (zeros when quantum disabled — backward safe) ---
        if quantum_state is not None:
            features["quantum_bull_amplitude"] = float(quantum_state.p_bull)
            features["quantum_bear_amplitude"] = float(quantum_state.p_bear)
            features["entanglement_score"] = float(quantum_state.entanglement_score)
            features["superposition_strength"] = float(quantum_state.superposition_strength)
        else:
            features["quantum_bull_amplitude"] = 0.0
            features["quantum_bear_amplitude"] = 0.0
            features["entanglement_score"] = 0.0
            features["superposition_strength"] = 0.0

        return features

    def features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)

    def build_training_row(self, features: Dict[str, float], outcome: int) -> Dict:
        row = dict(features)
        row["label"] = outcome
        return row

    def get_feature_names(self) -> List[str]:
        return FEATURE_NAMES
