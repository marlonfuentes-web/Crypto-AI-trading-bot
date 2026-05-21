import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from ai.feature_engineering import FeatureEngineer, FEATURE_NAMES

logger = logging.getLogger("bot.ai")

MODEL_FILE = "models/signal_filter.pkl"
TRAINING_DATA_FILE = "data/cache/training_data.json"


class AISignalFilter:
    """
    XGBoost-based signal quality classifier.
    Cold start: if insufficient data, returns cold_start_score (permissive).
    After min_samples trades, retrains on actual outcomes.
    """

    def __init__(self, model_path: str = MODEL_FILE,
                 min_training_samples: int = 100,
                 cold_start_score: float = 0.55,
                 retrain_interval_hours: int = 8):
        self.model_path = model_path
        self.min_training_samples = min_training_samples
        self.cold_start_score = cold_start_score
        self.retrain_interval_hours = retrain_interval_hours
        self.feature_engineer = FeatureEngineer()

        self._model = None
        self._scaler = None
        self._training_data: List[Dict] = []
        self._last_retrain_time: float = 0.0
        self._is_trained = False

        self._load_training_data()
        self._try_load_model()

    def _try_load_model(self):
        try:
            import joblib
            if os.path.exists(self.model_path):
                saved = joblib.load(self.model_path)
                self._model = saved.get("model")
                self._scaler = saved.get("scaler")
                self._last_retrain_time = saved.get("trained_at", 0)
                self._is_trained = True
                logger.info(f"AI model loaded from {self.model_path} "
                            f"(trained on {saved.get('n_samples', '?')} samples)")
        except Exception as e:
            logger.info(f"No existing model found, using cold-start mode: {e}")
            self._is_trained = False

    def _load_training_data(self):
        try:
            if os.path.exists(TRAINING_DATA_FILE):
                with open(TRAINING_DATA_FILE) as f:
                    self._training_data = json.load(f)
                logger.info(f"Loaded {len(self._training_data)} training samples")
        except Exception as e:
            logger.debug(f"No training data found: {e}")
            self._training_data = []

    def _save_training_data(self):
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_FILE), exist_ok=True)
            with open(TRAINING_DATA_FILE, "w") as f:
                json.dump(self._training_data[-2000:], f)  # Keep last 2000 samples
        except Exception as e:
            logger.warning(f"Could not save training data: {e}")

    def score_signal(self, features: Dict[str, float]) -> float:
        """
        Returns win probability 0.0-1.0.
        Returns cold_start_score if model not yet trained.
        """
        if not self._is_trained or self._model is None:
            logger.debug(f"Cold-start mode: returning score {self.cold_start_score} "
                         f"(need {self.min_training_samples} samples, have {len(self._training_data)})")
            return self.cold_start_score

        try:
            import numpy as np
            x = self.feature_engineer.features_to_array(features).reshape(1, -1)

            if self._scaler is not None:
                x = self._scaler.transform(x)

            prob = float(self._model.predict_proba(x)[0][1])
            return round(prob, 4)
        except Exception as e:
            logger.warning(f"AI scoring failed: {e}, returning cold_start_score")
            return self.cold_start_score

    def record_trade_outcome(self, features: Dict[str, float], outcome: int):
        """
        Record a completed trade for future training.
        outcome: 1 = win (hit TP), 0 = loss (hit SL)
        """
        row = dict(features)
        row["label"] = outcome
        row["recorded_at"] = int(time.time())
        self._training_data.append(row)
        self._save_training_data()

    def retrain_if_due(self) -> bool:
        """
        Retrain if:
        - Enough samples collected
        - retrain_interval_hours has elapsed since last retrain
        Returns True if retrain happened.
        """
        n_samples = len(self._training_data)
        if n_samples < self.min_training_samples:
            logger.info(f"Not enough training data: {n_samples}/{self.min_training_samples}")
            return False

        hours_since_retrain = (time.time() - self._last_retrain_time) / 3600
        if self._is_trained and hours_since_retrain < self.retrain_interval_hours:
            return False

        return self._train()

    def _train(self) -> bool:
        try:
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import roc_auc_score

            try:
                import xgboost as xgb
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier

            logger.info(f"Training AI model on {len(self._training_data)} samples...")

            df = pd.DataFrame(self._training_data)
            df = df.dropna()

            if "label" not in df.columns or len(df) < self.min_training_samples:
                return False

            X = df[[f for f in FEATURE_NAMES if f in df.columns]].fillna(0)
            y = df["label"].astype(int)

            if len(y.unique()) < 2:
                logger.warning("Training data has only one class — skipping retrain")
                return False

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_positive = int(y.sum())
            n_negative = len(y) - n_positive
            scale_pos_weight = n_negative / (n_positive + 1e-10)

            try:
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="auc",
                    use_label_encoder=False,
                    verbosity=0,
                    random_state=42,
                )
            except Exception:
                from sklearn.ensemble import GradientBoostingClassifier
                model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42,
                )

            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=min(5, len(y) // 20))
            auc_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_tr, y_tr)
                if len(y_val.unique()) > 1:
                    prob = model.predict_proba(X_val)[:, 1]
                    auc_scores.append(roc_auc_score(y_val, prob))

            # Final fit on all data
            model.fit(X_scaled, y)

            avg_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0.5
            logger.info(f"Model trained. Avg AUC: {avg_auc:.4f}, Samples: {len(y)}")

            self._model = model
            self._scaler = scaler
            self._is_trained = True
            self._last_retrain_time = time.time()

            import joblib
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                "model": model,
                "scaler": scaler,
                "trained_at": self._last_retrain_time,
                "n_samples": len(y),
                "avg_auc": avg_auc,
            }, self.model_path)

            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            return False

    def get_feature_importance(self) -> Dict[str, float]:
        if not self._is_trained or self._model is None:
            return {}
        try:
            importances = self._model.feature_importances_
            return {name: round(float(imp), 4)
                    for name, imp in zip(FEATURE_NAMES, importances)}
        except Exception:
            return {}

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def training_samples(self) -> int:
        return len(self._training_data)
