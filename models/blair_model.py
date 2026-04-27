"""
BLAIR model wrapper for baseline evaluation.
Uses the official BLAIR checkpoints from HuggingFace.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional


class BLAIRBaseline(nn.Module):
    """
    BLAIR baseline model for recommendation and retrieval.

    Uses the pretrained BLAIR checkpoints:
    - hyp1231/blair-roberta-base (125M params, 768-dim embeddings)
    - hyp1231/blair-roberta-large (355M params, 1024-dim embeddings)
    """

    def __init__(
            self,
            model_name: str = "hyp1231/blair-roberta-base",
            pooler_type: str = "cls",
            device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.pooler_type = pooler_type
        self.device = device

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = self.encoder.config.hidden_size

        # Freeze encoder for evaluation (standard BLAIR usage)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode(
            self,
            texts: List[str],
            max_length: int = 512,
            batch_size: int = 64,
    ) -> torch.Tensor:
        """
        Encode texts into BLAIR embeddings.

        Args:
            texts: List of text strings to encode
            max_length: Maximum token length
            batch_size: Batch size for encoding

        Returns:
            Tensor of shape (len(texts), embedding_dim) with normalized embeddings.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.encoder(**inputs, return_dict=True)
                # [CLS] token embedding (BLAIR-style pooling)
                embeddings = outputs.last_hidden_state[:, 0, :]
                # Normalize (BLAIR uses normalized embeddings)
                embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def encode_query(self, query: str, max_length: int = 512) -> torch.Tensor:
        """Encode a single query."""
        return self.encode([query], max_length=max_length)

    def encode_items(
            self,
            item_texts: List[str],
            max_length: int = 512,
            batch_size: int = 128,
    ) -> torch.Tensor:
        """Encode item metadata texts."""
        return self.encode(item_texts, max_length=max_length, batch_size=batch_size)

    def similarity(self, query_emb: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and items."""
        return query_emb @ item_embs.T

    def retrieve(
            self,
            query: str,
            item_texts: List[str],
            item_embeddings: Optional[torch.Tensor] = None,
            top_k: int = 100,
    ) -> List[int]:
        """
        Retrieve top-k items for a query.

        Returns:
            List of item indices sorted by similarity (descending).
        """
        query_emb = self.encode_query(query)

        if item_embeddings is None:
            item_embeddings = self.encode_items(item_texts)

        scores = self.similarity(query_emb, item_embeddings).squeeze(0)
        top_indices = scores.argsort(descending=True)[:top_k]

        return top_indices.cpu().tolist()

    def get_item_representations(
            self,
            item_texts: List[str],
            batch_size: int = 128,
            cache_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get item representations, optionally with caching.
        """
        if cache_path is not None:
            try:
                cached = torch.load(cache_path)
                if cached.shape[0] == len(item_texts):
                    return cached
            except FileNotFoundError:
                pass

        embeddings = self.encode_items(item_texts, batch_size=batch_size)

        if cache_path is not None:
            torch.save(embeddings, cache_path)

        return embeddings