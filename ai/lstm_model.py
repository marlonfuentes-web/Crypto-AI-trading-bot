"""
LSTM signal model with graceful fallback to MLP when PyTorch is unavailable.

Architecture (PyTorch path):
  Input: (batch=1, seq_len=50, n_features=32)
  LSTM(input=32, hidden=128, num_layers=2, dropout=0.2, batch_first=True)
  → LayerNorm(128) → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2)
  → Linear(64→1) → Sigmoid   →  win probability [0, 1]

Fallback (numpy MLP):
  2-layer MLP with tanh activations, trained via gradient descent.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("bot.lstm")

SEQ_LEN = 50
N_FEATURES = 32   # 28 indicator features + 4 quantum features


# ---------------------------------------------------------------------------
# Numpy MLP fallback
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """Tiny 2-layer MLP trained with mini-batch SGD on numpy arrays."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64,
                 lr: float = 0.001, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.standard_normal((n_features, hidden)) * scale1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * scale2
        self.b2 = np.zeros(1)
        self.lr = lr
        self.is_trained = False

    def _forward(self, X: np.ndarray):
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.tanh(self._z1)
        self._z2 = self._a1 @ self.W2 + self.b2
        out = 1.0 / (1.0 + np.exp(-self._z2))
        return out.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 20,
            batch_size: int = 32):
        N = len(X)
        for _ in range(epochs):
            idx = np.random.permutation(N)
            for start in range(0, N, batch_size):
                b_idx = idx[start:start + batch_size]
                Xb, yb = X[b_idx], y[b_idx].reshape(-1, 1)
                p = self._forward(Xb).reshape(-1, 1)
                eps = 1e-9
                p = np.clip(p, eps, 1 - eps)
                dL = (p - yb) / N
                dW2 = self._a1[b_idx].T @ dL
                db2 = dL.sum(axis=0)
                dA1 = dL @ self.W2.T
                dZ1 = dA1 * (1 - self._a1[b_idx] ** 2)
                dW1 = Xb.T @ dZ1
                db1 = dZ1.sum(axis=0)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
        self.is_trained = True


# ---------------------------------------------------------------------------
# PyTorch LSTM (optional)
# ---------------------------------------------------------------------------

def _try_build_lstm(n_features: int, hidden: int, num_layers: int):
    """Returns (model, True) if torch available, else (None, False)."""
    try:
        import torch
        import torch.nn as nn

        class _LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features, hidden_size=hidden,
                    num_layers=num_layers, dropout=0.2 if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.norm = nn.LayerNorm(hidden)
                self.drop1 = nn.Dropout(0.3)
                self.fc1 = nn.Linear(hidden, 64)
                self.relu = nn.ReLU()
                self.drop2 = nn.Dropout(0.2)
                self.fc2 = nn.Linear(64, 1)
                self.sig = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                h = out[:, -1, :]   # last time-step
                h = self.drop1(self.norm(h))
                h = self.drop2(self.relu(self.fc1(h)))
                return self.sig(self.fc2(h)).squeeze(-1)

        return _LSTMNet(), True
    except ImportError:
        return None, False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LSTMSignalModel:
    """
    Wraps either a PyTorch LSTM or numpy MLP behind a common interface.
    Call `predict(sequence)` to get a win-probability scalar.
    """

    def __init__(self, model_path: str = "models/lstm_model.pt",
                 mlp_path: str = "models/mlp_brain.pkl",
                 n_features: int = N_FEATURES, seq_len: int = SEQ_LEN):
        self.model_path = model_path
        self.mlp_path = mlp_path
        self.n_features = n_features
        self.seq_len = seq_len
        self._torch_available = False
        self._lstm = None
        self._mlp: Optional[_NumpyMLP] = None
        self.is_trained = False

        # Try torch first
        net, ok = _try_build_lstm(n_features, hidden=128, num_layers=2)
        if ok:
            self._torch_available = True
            self._lstm = net
            self._load_lstm()
        else:
            logger.info("PyTorch not available — using numpy MLP fallback")
            self._mlp = _NumpyMLP(n_features=n_features)
            self._load_mlp()

    # ---- persistence ----

    def _load_lstm(self):
        try:
            import torch
            if os.path.exists(self.model_path):
                state = torch.load(self.model_path, map_location="cpu")
                self._lstm.load_state_dict(state)
                self._lstm.eval()
                self.is_trained = True
                logger.info(f"LSTM model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")

    def _save_lstm(self):
        try:
            import torch
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            torch.save(self._lstm.state_dict(), self.model_path)
            logger.info(f"LSTM model saved → {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not save LSTM: {e}")

    def _load_mlp(self):
        try:
            if os.path.exists(self.mlp_path):
                with open(self.mlp_path, "rb") as f:
                    self._mlp = pickle.load(f)
                self.is_trained = self._mlp.is_trained
                logger.info(f"MLP model loaded from {self.mlp_path}")
        except Exception as e:
            logger.warning(f"Could not load MLP model: {e}")

    def _save_mlp(self):
        try:
            os.makedirs(os.path.dirname(self.mlp_path) or ".", exist_ok=True)
            with open(self.mlp_path, "wb") as f:
                pickle.dump(self._mlp, f)
            logger.info(f"MLP model saved → {self.mlp_path}")
        except Exception as e:
            logger.warning(f"Could not save MLP: {e}")

    # ---- inference ----

    def predict(self, sequence: np.ndarray) -> float:
        """
        Parameters
        ----------
        sequence : np.ndarray of shape (seq_len, n_features)
        Returns
        -------
        float : win probability [0, 1]
        """
        if not self.is_trained:
            return 0.5  # untrained → neutral

        if self._torch_available and self._lstm is not None:
            return self._predict_lstm(sequence)
        else:
            return self._predict_mlp(sequence[-1:])   # MLP uses last bar only

    def _predict_lstm(self, sequence: np.ndarray) -> float:
        try:
            import torch
            x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                return float(self._lstm(x).item())
        except Exception as e:
            logger.warning(f"LSTM predict error: {e}")
            return 0.5

    def _predict_mlp(self, X: np.ndarray) -> float:
        try:
            return float(self._mlp.predict_proba(X)[0])
        except Exception as e:
            logger.warning(f"MLP predict error: {e}")
            return 0.5

    # ---- training ----

    def fit(self, sequences: np.ndarray, labels: np.ndarray):
        """
        sequences : (N, seq_len, n_features)
        labels    : (N,) int 0/1
        """
        if self._torch_available and self._lstm is not None:
            self._fit_lstm(sequences, labels)
        else:
            # MLP uses last bar of each sequence
            X = sequences[:, -1, :]
            self._mlp.fit(X, labels.astype(float))
            self._save_mlp()
        self.is_trained = True

    def _fit_lstm(self, sequences: np.ndarray, labels: np.ndarray):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            X = torch.tensor(sequences, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.float32)

            optimizer = optim.Adam(self._lstm.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCELoss()
            self._lstm.train()

            batch_size = 32
            N = len(X)
            for epoch in range(30):
                idx = torch.randperm(N)
                for start in range(0, N, batch_size):
                    b = idx[start:start + batch_size]
                    optimizer.zero_grad()
                    loss = criterion(self._lstm(X[b]), y[b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._lstm.parameters(), 1.0)
                    optimizer.step()

            self._lstm.eval()
            self._save_lstm()
        except Exception as e:
            logger.warning(f"LSTM training error: {e}")

    def build_sequence(self, df_5m, feature_names: List[str],
                        quantum_state=None) -> Optional[np.ndarray]:
        """
        Build a (seq_len, n_features) array from the 5m OHLCV+indicator DataFrame.
        Last `seq_len` bars are used. Quantum features appended as constant column.
        """
        try:
            if df_5m is None or len(df_5m) < self.seq_len:
                return None

            available = [c for c in feature_names[:28] if c in df_5m.columns]
            seq = df_5m[available].iloc[-self.seq_len:].values.astype(float)

            # Pad / truncate to 28 base features
            if seq.shape[1] < 28:
                pad = np.zeros((self.seq_len, 28 - seq.shape[1]))
                seq = np.hstack([seq, pad])
            else:
                seq = seq[:, :28]

            # Append 4 quantum features (constant across time steps)
            if quantum_state is not None:
                q_cols = np.array([
                    quantum_state.p_bull,
                    quantum_state.p_bear,
                    quantum_state.entanglement_score,
                    quantum_state.superposition_strength,
                ], dtype=float)
            else:
                q_cols = np.zeros(4, dtype=float)

            q_block = np.tile(q_cols, (self.seq_len, 1))
            seq = np.hstack([seq, q_block])   # (seq_len, 32)

            # Z-score normalise per feature
            mean = seq.mean(axis=0)
            std = seq.std(axis=0) + 1e-8
            seq = (seq - mean) / std

            return seq.astype(np.float32)
        except Exception as e:
            logger.warning(f"build_sequence error: {e}")
            return None
