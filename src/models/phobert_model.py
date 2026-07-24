import torch
import numpy as np
import pandas as pd
from typing import Any
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.evaluation.base_model import BaseModel

class PhoBERTModel(BaseModel):
    def __init__(self, model_path: str = "models/phobert_model.pt", model_name: str = "vinai/phobert-base-v2", max_len: int = 256):
        self.model_path = model_path
        self.model_name = model_name
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"Loading tokenizer {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        print(f"Loading model weights from {self.model_path} onto {self.device}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
        # Load local weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("PhoBERT model loaded successfully.")

    def unload(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        raise NotImplementedError("Training should be done via notebook 5.0")

    def evaluate(self, X: pd.Series, y_true: pd.Series) -> dict[str, Any]:
        pass

    def predict(self, X: pd.Series) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        texts = X.tolist()
        batch_size = 32
        all_preds = []
        
        print(f"Predicting on {len(texts)} samples using {self.device}...")
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = [str(t) for t in texts[i:i+batch_size]]
                
                encoding = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                
        return np.array(all_preds)
