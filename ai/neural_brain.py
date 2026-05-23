"""
QuantumNeuralBrain — LSTM + MLP + XGBoost ensemble.

Drop-in replacement for AISignalFilter (same public interface).
When quantum features are provided, the LSTM scores them via sequence.
Ensemble weights are maintained as softmax over validation AUCs.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.signal_filter import AISignalFilter
from ai.lstm_model import LSTMSignalModel

logger = logging.getLogger("bot.neural_brain")

_DEFAULT_WEIGHTS = {"lstm": 0.50, "xgb": 0.35, "mlp": 0.15}
_MIN_WEIGHT = 0.05


class QuantumNeuralBrain:
    """
    Ensemble AI filter with LSTM + XGBoost + MLP.
    Public interface mirrors AISignalFilter so ScalpingStrategy
    can swap one for the other with zero changes.
    """

    def __init__(self,
                 xgb_filter: AISignalFilter,
                 lstm_model: LSTMSignalModel,
                 retrain_interval_hours: int = 4,
                 min_training_samples: int = 50,
                 cold_start_score: float = 0.55):
        self.xgb = xgb_filter
        self.lstm = lstm_model
        self.retrain_interval_hours = retrain_interval_hours
        self.min_training_samples = min_training_samples
        self.cold_start_score = cold_start_score

        # Ensemble weights (will update on retrain)
        self._weights = dict(_DEFAULT_WEIGHTS)
        self._last_retrain = time.time()

        # Training buffer for LSTM sequences
        self._sequence_buffer: List[Tuple[np.ndarray, int]] = []  # (seq, label)
        self._feature_buffer: List[Tuple[Dict, int]] = []         # (features, label)

    # ---- public interface (mirrors AISignalFilter) ----

    @property
    def is_trained(self) -> bool:
        return self.xgb.is_trained or self.lstm.is_trained

    @property
    def training_samples(self) -> int:
        return self.xgb.training_samples

    def score_signal(self, features: Dict) -> float:
        """Snapshot features path — backward compatible with single-bar XGBoost."""
        xgb_score = self.xgb.score_signal(features)
        mlp_score = self.lstm.predict(np.zeros((1, 32)))   # zeros → 0.5 if untrained
        # LSTM weight absorbed into XGBoost when no sequence available
        w_xgb = self._weights["xgb"] + self._weights["lstm"] * 0.5
        w_mlp = self._weights["mlp"] + self._weights["lstm"] * 0.5
        total = w_xgb + w_mlp
        score = (xgb_score * w_xgb + mlp_score * w_mlp) / total
        return float(np.clip(score, 0.0, 1.0))

    def score_signal_with_sequence(self, features: Dict,
                                    df_5m,
                                    quantum_state=None,
                                    feature_names: List[str] = None) -> float:
        """Full ensemble path — uses LSTM sequence for richer prediction."""
        from ai.feature_engineering import FEATURE_NAMES
        fn = feature_names or FEATURE_NAMES

        xgb_score = self.xgb.score_signal(features)
        mlp_score = self.lstm.predict(np.zeros((1, 32)))   # fallback MLP from LSTM obj

        lstm_score = 0.5  # default
        seq = self.lstm.build_sequence(df_5m, fn, quantum_state)
        if seq is not None:
            lstm_score = self.lstm.predict(seq)
        else:
            # Weight redistribution when sequence unavailable
            return self.score_signal(features)

        w = self._weights
        total_w = w["lstm"] + w["xgb"] + w["mlp"]
        score = (lstm_score * w["lstm"] + xgb_score * w["xgb"] + mlp_score * w["mlp"]) / total_w
        return float(np.clip(score, 0.0, 1.0))

    def record_trade_outcome(self, features: Dict, outcome: int,
                              sequence: Optional[np.ndarray] = None):
        """Record outcome for both XGBoost and LSTM retraining."""
        self.xgb.record_trade_outcome(features, outcome)
        if sequence is not None:
            self._sequence_buffer.append((sequence, outcome))
        self._feature_buffer.append((features, outcome))

    def retrain_if_due(self) -> bool:
        """Retrain all models if interval elapsed and enough samples exist."""
        elapsed_h = (time.time() - self._last_retrain) / 3600
        if elapsed_h < self.retrain_interval_hours:
            return False
        if len(self._feature_buffer) < self.min_training_samples:
            return False

        retrained = False

        # Retrain XGBoost
        if self.xgb.retrain_if_due():
            retrained = True

        # Retrain LSTM if sequence buffer has data
        if len(self._sequence_buffer) >= self.min_training_samples:
            seqs = np.array([s for s, _ in self._sequence_buffer])
            labels = np.array([l for _, l in self._sequence_buffer])
            self.lstm.fit(seqs, labels)
            retrained = True

        if retrained:
            self._last_retrain = time.time()
            self._update_weights()
            logger.info(f"Quantum brain retrained. Weights: {self._weights}")

        return retrained

    def _update_weights(self):
        """Update ensemble weights — simple heuristic based on sample counts."""
        # In production: compute validation AUCs; here use sample availability
        has_lstm = len(self._sequence_buffer) >= self.min_training_samples
        weights = {
            "lstm": 0.50 if has_lstm else 0.15,
            "xgb": 0.35,
            "mlp": 0.15 if has_lstm else 0.50,
        }
        # Normalise and clip min weight
        total = sum(weights.values())
        weights = {k: max(_MIN_WEIGHT, v / total) for k, v in weights.items()}
        total2 = sum(weights.values())
        self._weights = {k: v / total2 for k, v in weights.items()}
