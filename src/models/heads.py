"""
Task-Specific Prediction Heads

Heads for PF, OPF, and Cascade prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class PowerFlowHead(nn.Module):
    """
    Predict voltage magnitude and angle (as sin/cos).

    CRITICAL: Uses sin/cos representation for angles to avoid discontinuity at ±π.
    Includes normalization to enforce sin²θ + cos²θ = 1.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Separate outputs for V_mag, sin(θ), cos(θ)
        self.v_mag_head = nn.Linear(hidden_dim // 2, 1)
        self.sin_head = nn.Linear(hidden_dim // 2, 1)
        self.cos_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, node_embeddings):
        h = self.mlp(node_embeddings)

        v_mag = self.v_mag_head(h).squeeze(-1)
        sin_theta = self.sin_head(h).squeeze(-1)
        cos_theta = self.cos_head(h).squeeze(-1)

        # Normalize sin/cos to unit circle (enforce sin²+cos²=1)
        norm = torch.sqrt(sin_theta**2 + cos_theta**2 + 1e-8)
        sin_theta = sin_theta / norm
        cos_theta = cos_theta / norm

        return {
            "v_mag": v_mag,
            "sin_theta": sin_theta,
            "cos_theta": cos_theta,
        }


class OPFHead(nn.Module):
    """
    Predict generator setpoints and total cost.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        # Per-node generation prediction
        self.gen_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Graph-level cost prediction (via pooling)
        self.cost_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, gen_mask=None, batch=None):
        # Generator setpoints
        pg_pred = self.gen_mlp(node_embeddings).squeeze(-1)

        # Zero non-generator buses if mask provided
        if gen_mask is not None:
            pg_pred = pg_pred * gen_mask

        # Graph-level cost
        if batch is None:
            # Single graph
            graph_emb = node_embeddings.mean(dim=0, keepdim=True)
        else:
            # Batched graphs
            graph_emb = global_mean_pool(node_embeddings, batch)

        cost_pred = self.cost_mlp(graph_emb).squeeze(-1)

        return {
            "pg": pg_pred,
            "cost": cost_pred,
        }


class CascadeHead(nn.Module):
    """
    Graph-level cascade severity prediction.

    Uses attention-based graph pooling for interpretability.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 2,  # Binary: cascade or not
    ):
        super().__init__()

        # Attention-based graph pooling
        self.gate = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, node_embeddings, batch=None):
        # Attention weights
        att = torch.sigmoid(self.gate(node_embeddings))

        # Weighted pooling
        if batch is None:
            # Single graph
            graph_emb = (att * node_embeddings).sum(dim=0, keepdim=True)
        else:
            # Batched graphs - use scatter_add
            num_graphs = batch.max().item() + 1
            graph_emb = torch.zeros(
                num_graphs, node_embeddings.size(-1),
                device=node_embeddings.device, dtype=node_embeddings.dtype
            )
            graph_emb.scatter_add_(
                0, batch.unsqueeze(-1).expand_as(att * node_embeddings),
                att * node_embeddings
            )

        logits = self.classifier(graph_emb)

        return {
            "logits": logits,
            "attention": att,
        }


class CascadeBinaryHead(nn.Module):
    """
    Binary cascade prediction (cascade vs no cascade).

    Simpler head for binary classification.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, node_embeddings, batch=None):
        # Global mean pooling
        if batch is None:
            graph_emb = node_embeddings.mean(dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(node_embeddings, batch)

        h = self.pool_mlp(graph_emb)
        logits = self.classifier(h).squeeze(-1)

        return {
            "logits": logits,
        }
