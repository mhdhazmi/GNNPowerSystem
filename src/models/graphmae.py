"""
GraphMAE Baseline for SSL Comparison

Implements GraphMAE (Hou et al., 2022) as a baseline for comparing against
physics-guided SSL. Key differences from our approach:
- Standard GIN encoder (no physics-guided edge weighting)
- Scaled cosine error loss instead of MSE
- Random node feature masking without domain knowledge
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class GINEncoder(nn.Module):
    """
    Standard Graph Isomorphism Network encoder.

    No physics-guided edge weighting - uses uniform message passing.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Linear(node_in_dim, hidden_dim)

        # GIN layers (standard, no edge features used in aggregation)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,  # Ignored in GIN
    ) -> torch.Tensor:
        """Forward pass returning node embeddings."""
        x = self.node_embedding(x)

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        return x


def scaled_cosine_error(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Scaled cosine error loss from GraphMAE.

    L = (1 - cos_sim(pred, target))^gamma

    Args:
        pred: Predicted features [N, D]
        target: Target features [N, D]
        gamma: Scaling factor (default 2.0 per GraphMAE paper)

    Returns:
        Scalar loss
    """
    # Normalize
    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)

    # Cosine similarity
    cos_sim = (pred_norm * target_norm).sum(dim=-1)

    # Scaled error
    loss = (1 - cos_sim).pow(gamma).mean()

    return loss


class GraphMAE(nn.Module):
    """
    GraphMAE: Self-Supervised Masked Graph Autoencoders (Hou et al., 2022)

    Key design choices:
    - GIN encoder (no physics-guided edge weighting)
    - Scaled cosine error loss
    - Learnable mask token
    - Re-mask decoding (optional, not implemented here for simplicity)

    This serves as a baseline to show that physics-guided SSL outperforms
    generic graph SSL methods on power system tasks.
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
        gamma: float = 2.0,  # Scaling factor for cosine error
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.mask_ratio = mask_ratio
        self.gamma = gamma

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(node_in_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Standard GIN encoder (NOT physics-guided)
        self.encoder = GINEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_in_dim),
        )

    def create_mask(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Create random node masking."""
        num_nodes = x.size(0)
        device = x.device

        # Random mask selection
        mask_indices = torch.rand(num_nodes, device=device) < self.mask_ratio

        # Store original
        original_x = x.clone()

        # Apply mask token to selected nodes
        masked_x = x.clone()
        if mask_indices.sum() > 0:
            masked_x[mask_indices] = self.mask_token

        return masked_x, mask_indices, original_x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with masking and reconstruction."""
        # Create mask
        masked_x, mask_indices, original_x = self.create_mask(x)

        # Encode masked input
        node_emb = self.encoder(masked_x, edge_index, edge_attr)

        # Decode
        reconstructed = self.decoder(node_emb)

        # Compute scaled cosine error on masked nodes
        if mask_indices.sum() > 0:
            loss = scaled_cosine_error(
                reconstructed[mask_indices],
                original_x[mask_indices],
                gamma=self.gamma,
            )
        else:
            loss = torch.tensor(0.0, device=x.device)

        return {
            "loss": loss,
            "reconstructed": reconstructed,
            "mask": mask_indices,
            "original": original_x,
            "num_masked": mask_indices.sum(),
        }

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings without masking (for downstream tasks)."""
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer to downstream models."""
        return self.encoder.state_dict()
