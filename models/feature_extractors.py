"""
Unified feature extraction utilities for different model backbones.

Provides a consistent interface to obtain item/user embeddings from:
- BLAIR (RoBERTa)
- DeepSeek API
- Voyage AI API
- Sentence-BERT
"""
import numpy as np
import torch
from typing import List, Optional, Union

from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class BLAIRFeatureExtractor:
    """Extract embeddings using a BLAIR checkpoint."""
    def __init__(self, model_name: str = "hyp1231/blair-roberta-base", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 64, max_length: int = 512) -> np.ndarray:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                    max_length=max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                cls_emb = cls_emb / cls_emb.norm(dim=1, keepdim=True)
            embeddings.append(cls_emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)


class DeepSeekFeatureExtractor:
    """Extract embeddings via DeepSeek API (wrapper around DeepSeekRecommender)."""
    def __init__(self, api_key: Optional[str] = None):
        from models.deepseek_model import DeepSeekRecommender
        self.client = DeepSeekRecommender(api_key=api_key)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.client.encode(texts, batch_size=batch_size)


class VoyageAIFeatureExtractor:
    """Extract embeddings via Voyage AI."""
    def __init__(self, api_key: Optional[str] = None, model: str = "voyage-3"):
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def encode(self, texts: List[str], batch_size: int = 128, input_type: str = "document") -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            result = self.client.embed(batch, model=self.model, input_type=input_type)
            all_embs.extend(result.embeddings)
        return np.array(all_embs, dtype=np.float32)