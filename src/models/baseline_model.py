from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

# ── resolve project root so this script works both as module and as __main__ ──
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.evaluation.base_model import BaseModel  # noqa: E402

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(_PROJECT_ROOT, 'data', 'raw')
TRAIN_PATH = os.path.join(RAW_DIR, 'train_master.csv')
VAL_PATH   = os.path.join(RAW_DIR, 'val_master.csv')


class TFIDFBaselineModel(BaseModel):
    """
    Baseline Prompt-Firewall classifier:
        TF-IDF (word n-grams 1-2) + Logistic Regression.

    Inherits from BaseModel and implements the full train / predict / evaluate
    contract required by the evaluation framework.
    """

    def __init__(
        self,
        max_features: int = 100_000,
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1_000,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            max_features:  Maximum number of TF-IDF features.
            ngram_range:   Word n-gram range for TF-IDF.
            sublinear_tf:  Apply sublinear TF scaling (log(1+tf)).
            C:             Inverse regularization strength for LR.
            max_iter:      Maximum iterations for LR solver.
            random_state:  Random seed for reproducibility.
        """
        self.max_features  = max_features
        self.ngram_range   = ngram_range
        self.sublinear_tf  = sublinear_tf
        self.C             = C
        self.max_iter      = max_iter
        self.random_state  = random_state
        self._pipeline: Pipeline | None = None

    # ── BaseModel interface ────────────────────────────────────────────────────

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Fit TF-IDF + Logistic Regression on training data.

        Args:
            X_train: Series of raw prompt strings.
            y_train: Series of integer labels (0 = safe, 1 = unsafe).
        """
        print("[TFIDFBaselineModel] Building pipeline ...")
        self._pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                sublinear_tf=self.sublinear_tf,
                strip_accents='unicode',
                lowercase=True,
                min_df=2,
            )),
            ('clf', LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                class_weight='balanced',
                solver='lbfgs',
                random_state=self.random_state,
                n_jobs=-1,
            )),
        ])

        t0 = time.time()
        print(f"[TFIDFBaselineModel] Training on {len(X_train):,} samples ...")
        self._pipeline.fit(X_train.astype(str), y_train)
        elapsed = time.time() - t0
        print(f"[TFIDFBaselineModel] Training done in {elapsed:.1f}s")

    def predict(self, X: pd.Series) -> np.ndarray:
        """Return hard predictions (0 or 1).

        Args:
            X: Series of raw prompt strings.

        Returns:
            numpy array of shape (n_samples,) with values in {0, 1}.
        """
        self._check_fitted()
        return self._pipeline.predict(X.astype(str))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Return probability estimates for class 1 (unsafe).

        Args:
            X: Series of raw prompt strings.

        Returns:
            numpy array of shape (n_samples, 2) with probabilities.
        """
        self._check_fitted()
        return self._pipeline.predict_proba(X.astype(str))

    def evaluate(self, X: pd.Series, y_true: pd.Series) -> dict[str, Any]:
        """Evaluate on a held-out set and print a full report.

        Args:
            X:      Series of raw prompt strings.
            y_true: Ground-truth integer labels.

        Returns:
            dict with keys: accuracy, f1_score, f1_safe, f1_unsafe,
                            precision_unsafe, recall_unsafe.
        """
        self._check_fitted()
        y_pred = self.predict(X)

        acc       = accuracy_score(y_true, y_pred)
        f1_macro  = f1_score(y_true, y_pred, average='macro')
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1])

        metrics: dict[str, Any] = {
            'accuracy':         round(acc, 4),
            'f1_score':         round(f1_macro, 4),       # macro F1 (required by BaseModel)
            'f1_safe':          round(float(f1_per_class[0]), 4),
            'f1_unsafe':        round(float(f1_per_class[1]), 4),
        }

        # ── Console output ────────────────────────────────────────────
        print()
        print("=" * 55)
        print("  Validation Results — TF-IDF + Logistic Regression")
        print("=" * 55)
        print(f"  Accuracy    : {acc * 100:.2f}%")
        print(f"  F1-Score (macro) : {f1_macro:.4f}")
        print(f"  F1 Safe     (0)  : {float(f1_per_class[0]):.4f}")
        print(f"  F1 Unsafe   (1)  : {float(f1_per_class[1]):.4f}")
        print()
        print("  Classification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=['Safe (0)', 'Unsafe (1)'],
            digits=4,
        ))
        print("=" * 55)

        return metrics

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model is not trained yet. Call .train(X_train, y_train) first."
            )

    def __repr__(self) -> str:
        return (
            f"TFIDFBaselineModel("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"C={self.C})"
        )


# ── Standalone entry-point ─────────────────────────────────────────────────────

def main() -> None:
    """Load train_master & val_master, train and evaluate the baseline model."""

    print("=" * 55)
    print("  Baseline Model: TF-IDF + Logistic Regression")
    print("=" * 55)

    # ── 1. Load data ──────────────────────────────────────────────────
    print(f"\n[Data] Loading train  : {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"       Rows: {len(train_df):,}  |  label dist: {train_df['label_unsafe'].value_counts().to_dict()}")

    print(f"[Data] Loading val    : {VAL_PATH}")
    val_df = pd.read_csv(VAL_PATH)
    print(f"       Rows: {len(val_df):,}  |  label dist: {val_df['label_unsafe'].value_counts().to_dict()}")

    X_train = train_df['prompt']
    y_train = train_df['label_unsafe']
    X_val   = val_df['prompt']
    y_val   = val_df['label_unsafe']

    # ── 2. Train ──────────────────────────────────────────────────────
    model = TFIDFBaselineModel(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        C=1.0,
        max_iter=1_000,
        random_state=42,
    )
    model.train(X_train, y_train)

    # ── 3. Evaluate on validation set ────────────────────────────────
    metrics = model.evaluate(X_val, y_val)
    print(f"\n[Summary] Accuracy: {metrics['accuracy']:.4f}  |  F1-Score (macro): {metrics['f1_score']:.4f}")


if __name__ == '__main__':
    main()
