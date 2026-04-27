"""
Custom recommendation model with multi-objective loss and custom prompts.

Architecture: Two-tower model with MoE adapter, following RecXplore's best configuration.
- User tower: User ID -> Embedding -> MLP -> user embedding
- Item tower: Item ID -> Embedding -> MoE adapter (fusing ID + LLM embeddings) -> item embedding
- Optional PCA for dimensionality reduction
- LoRA for efficient fine-tuning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .loss_functions import MultiObjectiveLoss


class MoEAdapter(nn.Module):
    """
    Mixture-of-Experts adapter for fusing ID embeddings with LLM-derived embeddings.
    Based on RecXplore's best configuration.
    """

    def __init__(
            self,
            id_dim: int,
            llm_dim: int,
            output_dim: int,
            num_experts: int = 8,
            top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(id_dim + llm_dim, output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
            )
            for _ in range(num_experts)
        ])

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(id_dim + llm_dim, num_experts * 2),
            nn.ReLU(),
            nn.Linear(num_experts * 2, num_experts),
        )

    def forward(self, id_emb: torch.Tensor, llm_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            id_emb: (batch_size, id_dim) or (num_items, id_dim)
            llm_emb: (batch_size, llm_dim) or (num_items, llm_dim)
        Returns:
            fused_emb: (batch_size, output_dim)
        """
        combined = torch.cat([id_emb, llm_emb], dim=-1)

        # Gate logits
        gate_logits = self.gate(combined)  # (batch_size, num_experts)

        # Top-k gating
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Weighted sum of expert outputs
        output = torch.zeros(combined.shape[0], self.experts[0][-1].out_features, device=combined.device)
        for i, expert in enumerate(self.experts):
            expert_out = expert(combined)
            # Find positions where this expert is selected
            mask = (top_k_indices == i).float().sum(dim=-1, keepdim=True)  # (batch_size, 1)
            weight = (top_k_indices == i).float() * top_k_weights
            weight = weight.sum(dim=-1, keepdim=True)
            output += weight * expert_out

        return output


class CustomRecommendationModel(nn.Module):
    """
    Custom two-tower recommendation model with:
    - BLAIR-initialized backbone for text encoding
    - MoE adapter for fusing LLM embeddings
    - Multi-objective loss (BPR + fairness + diversity + debiasing + augmentation)
    - LoRA for parameter-efficient fine-tuning
    """

    def __init__(
            self,
            blair_model_name: str = "hyp1231/blair-roberta-base",
            embedding_dim: int = 768,
            adapter_type: str = "moe",
            num_experts: int = 8,
            lora_rank: int = 8,
            lora_alpha: int = 16,
            use_pca: bool = True,
            pca_dim: int = 256,
            # Loss parameters
            loss_config: Optional[Dict] = None,
            device: str = 'cuda',
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adapter_type = adapter_type
        self.use_pca = use_pca
        self.device = device

        # Text encoder (BLAIR backbone)
        self.text_encoder = AutoModel.from_pretrained(blair_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(blair_model_name)

        # Apply LoRA for efficient fine-tuning
        if lora_rank > 0:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=0.1,
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        # User tower
        self.user_embedding = nn.Embedding(100000, embedding_dim)  # Placeholder size
        self.user_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Item tower
        self.item_id_embedding = nn.Embedding(500000, embedding_dim)  # Placeholder size

        # PCA for LLM embeddings
        if use_pca:
            self.pca_projection = nn.Linear(embedding_dim, pca_dim)
            adapter_input_dim = pca_dim
        else:
            self.pca_projection = None
            adapter_input_dim = embedding_dim

        # Adapter for fusing ID + LLM embeddings
        if adapter_type == "moe":
            self.adapter = MoEAdapter(
                id_dim=embedding_dim,
                llm_dim=adapter_input_dim,
                output_dim=embedding_dim,
                num_experts=num_experts,
            )
        else:
            # Simple linear adapter
            self.adapter = nn.Sequential(
                nn.Linear(embedding_dim + adapter_input_dim, embedding_dim * 2),
                nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim),
            )

        # Loss function
        loss_config = loss_config or {}
        self.loss_fn = MultiObjectiveLoss(
            bpr_weight=loss_config.get('bpr_weight', 1.0),
            adv_weight=loss_config.get('adv_weight', 0.1),
            div_weight=loss_config.get('div_weight', 0.05),
            pop_weight=loss_config.get('pop_weight', 0.1),
            aug_weight=loss_config.get('aug_weight', 0.2),
            aug_temperature=loss_config.get('aug_temperature', 0.07),
            div_topk=loss_config.get('div_topk', 10),
            adv_hidden_dim=loss_config.get('adv_hidden_dim', 128),
            device=device,
        )

    def encode_text(self, texts: List[str], max_length: int = 64) -> torch.Tensor:
        """Encode text using BLAIR backbone."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.text_encoder(**inputs, return_dict=True)
            # Use [CLS] token embedding (BLAIR-style)
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize (BLAIR-style)
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings."""
        user_emb = self.user_embedding(user_ids)
        user_emb = self.user_mlp(user_emb)
        return F.normalize(user_emb, p=2, dim=-1)

    def get_item_embedding(
            self,
            item_ids: torch.Tensor,
            item_texts: Optional[List[str]] = None,
            precomputed_llm_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get item embeddings by fusing ID and text embeddings."""
        # ID embedding
        id_emb = self.item_id_embedding(item_ids)

        # LLM text embedding
        if precomputed_llm_emb is not None:
            llm_emb = precomputed_llm_emb
        elif item_texts is not None:
            llm_emb = self.encode_text(item_texts)
        else:
            llm_emb = torch.zeros_like(id_emb)

        # Apply PCA if enabled
        if self.pca_projection is not None:
            llm_emb = self.pca_projection(llm_emb)

        # Fuse embeddings via adapter
        fused_emb = self.adapter(id_emb, llm_emb)

        return F.normalize(fused_emb, p=2, dim=-1)

    def forward(
            self,
            user_ids: torch.Tensor,
            pos_item_ids: torch.Tensor,
            neg_item_ids: torch.Tensor,
            pos_item_texts: Optional[List[str]] = None,
            neg_item_texts: Optional[List[str]] = None,
            orig_item_emb: Optional[torch.Tensor] = None,
            aug_item_emb: Optional[torch.Tensor] = None,
            item_popularities: Optional[torch.Tensor] = None,
            protected_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass with multi-objective loss computation.
        """
        # Get embeddings
        user_emb = self.get_user_embedding(user_ids)
        pos_item_emb = self.get_item_embedding(pos_item_ids, pos_item_texts)
        neg_item_emb = self.get_item_embedding(neg_item_ids, neg_item_texts)

        # Compute loss
        total_loss, loss_dict = self.loss_fn(
            user_embeddings=user_emb,
            pos_item_embeddings=pos_item_emb,
            neg_item_embeddings=neg_item_emb,
            all_item_embeddings=self.item_id_embedding.weight,  # All item embeddings
            orig_item_embeddings=orig_item_emb,
            aug_item_embeddings=aug_item_emb,
            item_popularities=item_popularities,
            protected_labels=protected_labels,
        )

        return total_loss, loss_dict

    def predict(
            self,
            user_ids: torch.Tensor,
            item_ids: torch.Tensor,
            item_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Predict scores for user-item pairs."""
        user_emb = self.get_user_embedding(user_ids)
        item_emb = self.get_item_embedding(item_ids, item_texts)
        scores = (user_emb * item_emb).sum(dim=-1)
        return scores