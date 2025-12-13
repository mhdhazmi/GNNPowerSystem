"""
Self-Supervised Learning Models for PowerGraph

Implements masked node feature reconstruction for pretraining the
physics-guided encoder on unlabeled power grid data.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PhysicsGuidedEncoder


class MaskedNodeReconstruction(nn.Module):
    """
    SSL model for masked node feature reconstruction.

    Randomly masks node features and trains the encoder to reconstruct
    the original values from the graph context.

    Similar to BERT-style masking but for graph nodes:
    - 15% of nodes are selected for prediction
    - Of those: 80% masked, 10% random, 10% unchanged
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.mask_ratio = mask_ratio

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(node_in_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Encoder (same as cascade model)
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_in_dim),
        )

    def create_mask(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masking for nodes following BERT-style strategy.

        Args:
            x: Node features [num_nodes, node_in_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            masked_x: Node features with masking applied
            mask_indices: Boolean mask of which nodes to predict
            original_x: Original features for loss computation
        """
        num_nodes = x.size(0)
        device = x.device

        # Select nodes to mask (15% by default)
        mask_indices = torch.rand(num_nodes, device=device) < self.mask_ratio

        # Store original values
        original_x = x.clone()

        # Apply masking strategy to selected nodes:
        # 80% -> mask token, 10% -> random, 10% -> unchanged
        masked_x = x.clone()

        if mask_indices.sum() > 0:
            mask_positions = mask_indices.nonzero(as_tuple=True)[0]
            num_masked = len(mask_positions)

            # Random assignment for each masked node
            rand_vals = torch.rand(num_masked, device=device)

            # 80% get mask token
            mask_token_mask = rand_vals < 0.8
            mask_token_positions = mask_positions[mask_token_mask]
            masked_x[mask_token_positions] = self.mask_token

            # 10% get random values
            random_mask = (rand_vals >= 0.8) & (rand_vals < 0.9)
            random_positions = mask_positions[random_mask]
            if len(random_positions) > 0:
                # Sample random values from the same distribution
                random_indices = torch.randint(0, num_nodes, (len(random_positions),), device=device)
                masked_x[random_positions] = x[random_indices]

            # 10% remain unchanged (no action needed)

        return masked_x, mask_indices, original_x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masking and reconstruction.

        Returns:
            Dictionary with 'reconstructed', 'mask', 'original', 'loss'
        """
        # Create mask
        masked_x, mask_indices, original_x = self.create_mask(x, batch)

        # Encode masked input
        node_emb = self.encoder(masked_x, edge_index, edge_attr)

        # Reconstruct node features
        reconstructed = self.reconstruction_head(node_emb)

        # Compute loss only on masked nodes
        if mask_indices.sum() > 0:
            loss = F.mse_loss(
                reconstructed[mask_indices],
                original_x[mask_indices],
            )
        else:
            loss = torch.tensor(0.0, device=x.device)

        return {
            "reconstructed": reconstructed,
            "mask": mask_indices,
            "original": original_x,
            "loss": loss,
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


class MaskedEdgeReconstruction(nn.Module):
    """
    SSL model for masked edge feature reconstruction.

    Randomly masks edge features and trains to reconstruct them.
    Useful for learning edge-centric representations.
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()

        self.edge_in_dim = edge_in_dim
        self.mask_ratio = mask_ratio

        # Learnable mask token for edges
        self.mask_token = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Encoder
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Edge reconstruction head (from endpoint node embeddings)
        self.edge_reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_in_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with edge masking and reconstruction."""
        num_edges = edge_attr.size(0)
        device = edge_attr.device

        # Create edge mask
        mask_indices = torch.rand(num_edges, device=device) < self.mask_ratio
        original_edge_attr = edge_attr.clone()

        # Apply masking
        masked_edge_attr = edge_attr.clone()
        if mask_indices.sum() > 0:
            mask_positions = mask_indices.nonzero(as_tuple=True)[0]
            num_masked = len(mask_positions)
            rand_vals = torch.rand(num_masked, device=device)

            # 80% mask token
            mask_token_mask = rand_vals < 0.8
            mask_token_positions = mask_positions[mask_token_mask]
            masked_edge_attr[mask_token_positions] = self.mask_token

            # 10% random
            random_mask = (rand_vals >= 0.8) & (rand_vals < 0.9)
            random_positions = mask_positions[random_mask]
            if len(random_positions) > 0:
                random_indices = torch.randint(0, num_edges, (len(random_positions),), device=device)
                masked_edge_attr[random_positions] = edge_attr[random_indices]

        # Encode with masked edges
        node_emb = self.encoder(x, edge_index, masked_edge_attr)

        # Reconstruct edges from endpoint embeddings
        src, dst = edge_index
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)
        reconstructed = self.edge_reconstruction_head(edge_emb)

        # Loss on masked edges
        if mask_indices.sum() > 0:
            loss = F.mse_loss(
                reconstructed[mask_indices],
                original_edge_attr[mask_indices],
            )
        else:
            loss = torch.tensor(0.0, device=device)

        return {
            "reconstructed": reconstructed,
            "mask": mask_indices,
            "original": original_edge_attr,
            "loss": loss,
            "num_masked": mask_indices.sum(),
        }

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings without masking."""
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer."""
        return self.encoder.state_dict()


class CombinedSSL(nn.Module):
    """
    Combined SSL model with both node and edge reconstruction.

    Jointly optimizes both objectives for richer representations.
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.15,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
    ):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.node_weight = node_weight
        self.edge_weight = edge_weight

        # Learnable mask tokens
        self.node_mask_token = nn.Parameter(torch.zeros(node_in_dim))
        self.edge_mask_token = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.node_mask_token, std=0.02)
        nn.init.normal_(self.edge_mask_token, std=0.02)

        # Shared encoder
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Reconstruction heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_in_dim),
        )

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_in_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with both node and edge masking."""
        device = x.device
        num_nodes = x.size(0)
        num_edges = edge_attr.size(0)

        # Store originals
        original_x = x.clone()
        original_edge_attr = edge_attr.clone()

        # Create masks
        node_mask = torch.rand(num_nodes, device=device) < self.node_mask_ratio
        edge_mask = torch.rand(num_edges, device=device) < self.edge_mask_ratio

        # Apply node masking (simplified: just mask token)
        masked_x = x.clone()
        if node_mask.sum() > 0:
            masked_x[node_mask] = self.node_mask_token

        # Apply edge masking
        masked_edge_attr = edge_attr.clone()
        if edge_mask.sum() > 0:
            masked_edge_attr[edge_mask] = self.edge_mask_token

        # Encode
        node_emb = self.encoder(masked_x, edge_index, masked_edge_attr)

        # Reconstruct nodes
        node_reconstructed = self.node_head(node_emb)

        # Reconstruct edges
        src, dst = edge_index
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=1)
        edge_reconstructed = self.edge_head(edge_emb)

        # Compute losses
        node_loss = torch.tensor(0.0, device=device)
        edge_loss = torch.tensor(0.0, device=device)

        if node_mask.sum() > 0:
            node_loss = F.mse_loss(node_reconstructed[node_mask], original_x[node_mask])

        if edge_mask.sum() > 0:
            edge_loss = F.mse_loss(edge_reconstructed[edge_mask], original_edge_attr[edge_mask])

        total_loss = self.node_weight * node_loss + self.edge_weight * edge_loss

        return {
            "loss": total_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "num_node_masked": node_mask.sum(),
            "num_edge_masked": edge_mask.sum(),
        }

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings without masking."""
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer."""
        return self.encoder.state_dict()
