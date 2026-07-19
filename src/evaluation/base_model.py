"""
src/evaluation/base_model.py
Abstract base class for all Prompt Firewall models.
Every concrete model must inherit from BaseModel and implement
the three abstract methods: train, predict, evaluate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract interface that every model in the pipeline must satisfy."""

    # ── Must-implement contract ──────────────────────────────────────

    @abstractmethod
    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Fit the model on training data.

        Args:
            X_train: Series of raw text prompts.
            y_train: Series of integer labels (0 = safe, 1 = unsafe).
        """

    @abstractmethod
    def predict(self, X: pd.Series) -> np.ndarray:
        """Return hard predictions (0 or 1) for each prompt in X.

        Args:
            X: Series of raw text prompts.

        Returns:
            1-D numpy array of integer predictions.
        """

    @abstractmethod
    def evaluate(self, X: pd.Series, y_true: pd.Series) -> dict[str, Any]:
        """Evaluate the model and return a metrics dictionary.

        The returned dict MUST contain at least:
            - "accuracy"  (float)
            - "f1_score"  (float, macro average)

        Args:
            X:      Series of raw text prompts.
            y_true: Series of ground-truth integer labels.

        Returns:
            Dictionary of metric names → values.
        """

    # ── Concrete helpers (available to all subclasses) ───────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
