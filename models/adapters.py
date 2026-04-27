"""
Adapter modules for fusing ID embeddings with LLM-based item embeddings.

Includes:
- Simple linear fusion
- Mixture-of-Experts (MoE) adapter (as in RecXplore)
- PCA reduction wrapper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    """Basic linear fusion of ID and LLM embeddings."""
    def __init__(self, id_dim: int, llm_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(id_dim + llm_dim, output_dim)

    def forward(self, id_emb: torch.Tensor, llm_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([id_emb, llm_emb], dim=-1)
        return self.fc(combined)


class MoEAdapter(nn.Module):
    """
    Mixture-of-Experts adapter. Routes through a subset of expert networks
    for each input.
    """
    def __init__(self, id_dim: int, llm_dim: int, output_dim: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        input_dim = id_dim + llm_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim * 2),
                nn.ReLU(),
                nn.Linear(output_dim * 2, output_dim),
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts * 2),
            nn.ReLU(),
            nn.Linear(num_experts * 2, num_experts),
        )

    def forward(self, id_emb: torch.Tensor, llm_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([id_emb, llm_emb], dim=-1)
        gate_logits = self.gate(combined)
        topk_weights, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        output = torch.zeros(combined.size(0), self.experts[0][-1].out_features,
                             device=combined.device)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).float().sum(dim=-1, keepdim=True)
            weight = (topk_indices == i).float() * topk_weights
            weight = weight.sum(dim=-1, keepdim=True)
            output += weight * expert(combined)
        return output


class PCAAdapter(nn.Module):
    """Dimensionality reduction wrapper."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)