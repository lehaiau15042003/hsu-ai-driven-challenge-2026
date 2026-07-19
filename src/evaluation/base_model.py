from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        
    @abstractmethod
    def predict(self, X: pd.Series) -> np.ndarray:

    @abstractmethod
    def evaluate(self, X: pd.Series, y_true: pd.Series) -> dict[str, Any]:

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, prompt: str):
        pass

    @abstractmethod
    def unload(self):
        pass