"""
Contrastive Self-Supervised Learning Models for PowerGraph

Implements graph contrastive learning methods including:
- GraphCL: Graph-level contrastive learning with augmentations
- GRACE: Node-level contrastive representation learning

These methods complement the existing masking-based SSL approaches
(MaskedNodeReconstruction, MaskedEdgeReconstruction, CombinedSSL).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

from .encoder import PhysicsGuidedEncoder
from .ssl import ProjectionHead
from .losses import NTXentLoss
from .augmentations import (
    GraphAugmentation,
    Compose,
    EdgeDropping,
    NodeFeatureMasking,
    FeaturePerturbation,
    create_augmentation,
)


class GraphCL(nn.Module):
    """
    Graph Contrastive Learning (GraphCL).

    Graph-level contrastive learning using augmentations:
    1. Generate two augmented views of each input graph
    2. Encode both views using shared encoder
    3. Pool node embeddings to graph-level representations
    4. Apply projection head
    5. Maximize agreement between views via NT-Xent loss

    Reference: "Graph Contrastive Learning with Augmentations" (NeurIPS 2020)

    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hidden_dim: Hidden dimension of encoder
        proj_dim: Output dimension of projection head
        num_layers: Number of encoder layers
        dropout: Dropout rate
        temperature: Temperature for NT-Xent loss
        augmentation_1: First view augmentation (default: edge_drop + node_mask)
        augmentation_2: Second view augmentation (default: edge_drop + feature_noise)
        pooling: Graph pooling method ("mean" or "sum")
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.5,
        augmentation_1: Optional[GraphAugmentation] = None,
        augmentation_2: Optional[GraphAugmentation] = None,
        pooling: str = "mean",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Physics-guided encoder (shared with masking methods)
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Projection head (discarded after pretraining)
        self.projection = ProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=proj_dim,
        )

        # Graph pooling
        self.pool = global_mean_pool if pooling == "mean" else global_add_pool

        # Contrastive loss
        self.loss_fn = NTXentLoss(temperature=temperature)

        # Default augmentations if not provided
        self.aug1 = augmentation_1 or Compose([
            EdgeDropping(drop_ratio=0.2),
            NodeFeatureMasking(mask_ratio=0.1),
        ])
        self.aug2 = augmentation_2 or Compose([
            EdgeDropping(drop_ratio=0.2),
            FeaturePerturbation(noise_std=0.1),
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.

        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D]
            batch: Batch assignment [N]

        Returns:
            Dict with:
                - loss: Contrastive loss (scalar)
                - z1: Projected embeddings from view 1 [B, proj_dim]
                - z2: Projected embeddings from view 2 [B, proj_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Create two augmented views
        x1, ei1, ea1 = self.aug1(x, edge_index, edge_attr, batch)
        x2, ei2, ea2 = self.aug2(x, edge_index, edge_attr, batch)

        # Handle empty graphs after augmentation
        if ei1.size(1) == 0 or ei2.size(1) == 0:
            # Fallback to original graph if augmentation removed all edges
            x1, ei1, ea1 = x, edge_index, edge_attr
            x2, ei2, ea2 = x, edge_index, edge_attr

        # Encode both views
        h1 = self.encoder(x1, ei1, ea1)
        h2 = self.encoder(x2, ei2, ea2)

        # Pool to graph-level representations
        # Note: batch assignment may change with subgraph sampling
        g1 = self.pool(h1, batch)
        g2 = self.pool(h2, batch)

        # Project for contrastive loss
        z1 = self.projection(g1)
        z2 = self.projection(g2)

        # Compute contrastive loss
        loss = self.loss_fn(z1, z2)

        return {
            "loss": loss,
            "z1": z1,
            "z2": z2,
        }

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get node embeddings without augmentation (for downstream tasks).

        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D]

        Returns:
            Node embeddings [N, hidden_dim]
        """
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer to downstream tasks."""
        return self.encoder.state_dict()


class GRACE(nn.Module):
    """
    Graph Contrastive Representation Learning (GRACE).

    Node-level contrastive learning:
    1. Generate two augmented views of the input graph
    2. Encode both views using shared encoder
    3. For each node, contrast its embedding across views (positive)
       against other nodes' embeddings (negatives)

    Reference: "Deep Graph Contrastive Representation Learning" (ICML Workshop 2020)

    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hidden_dim: Hidden dimension of encoder
        proj_dim: Output dimension of projection head
        num_layers: Number of encoder layers
        dropout: Dropout rate
        temperature: Temperature for contrastive loss
        augmentation_1: First view augmentation
        augmentation_2: Second view augmentation
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
        temperature: float = 0.5,
        augmentation_1: Optional[GraphAugmentation] = None,
        augmentation_2: Optional[GraphAugmentation] = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Physics-guided encoder
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Projection head
        self.projection = ProjectionHead(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=proj_dim,
        )

        # Contrastive loss
        self.loss_fn = NTXentLoss(temperature=temperature)

        # GRACE typically uses stronger augmentations than GraphCL
        self.aug1 = augmentation_1 or Compose([
            EdgeDropping(drop_ratio=0.3),
            NodeFeatureMasking(mask_ratio=0.3),
        ])
        self.aug2 = augmentation_2 or Compose([
            EdgeDropping(drop_ratio=0.4),
            NodeFeatureMasking(mask_ratio=0.4),
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with node-level contrastive learning.

        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D]
            batch: Batch assignment [N]

        Returns:
            Dict with:
                - loss: Contrastive loss (scalar)
                - z1: Projected node embeddings from view 1 [N, proj_dim]
                - z2: Projected node embeddings from view 2 [N, proj_dim]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Create two augmented views
        x1, ei1, ea1 = self.aug1(x, edge_index, edge_attr, batch)
        x2, ei2, ea2 = self.aug2(x, edge_index, edge_attr, batch)

        # Handle empty graphs
        if ei1.size(1) == 0:
            x1, ei1, ea1 = x, edge_index, edge_attr
        if ei2.size(1) == 0:
            x2, ei2, ea2 = x, edge_index, edge_attr

        # Encode both views (node-level embeddings)
        h1 = self.encoder(x1, ei1, ea1)
        h2 = self.encoder(x2, ei2, ea2)

        # Project for contrastive loss
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        # Compute node-level contrastive loss
        loss = self._compute_node_loss(z1, z2, batch)

        return {
            "loss": loss,
            "z1": z1,
            "z2": z2,
        }

    def _compute_node_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute node-level InfoNCE loss.

        For each node i:
            - Positive: same node in other view (z1[i] <-> z2[i])
            - Negatives: other nodes within the same graph

        Using within-graph negatives only to avoid O(N^2) complexity
        for large batches.
        """
        num_nodes = z1.size(0)
        device = z1.device

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        total_loss = 0.0
        num_graphs = batch.max().item() + 1

        for graph_id in range(num_graphs):
            # Get nodes in this graph
            mask = batch == graph_id
            z1_g = z1[mask]
            z2_g = z2[mask]
            n = z1_g.size(0)

            if n <= 1:
                continue

            # Compute similarities
            # z1 -> z2: positive pairs on diagonal
            sim_12 = torch.mm(z1_g, z2_g.t()) / self.temperature  # [n, n]
            # z1 -> z1: for additional negatives
            sim_11 = torch.mm(z1_g, z1_g.t()) / self.temperature  # [n, n]

            # Diagonal mask
            diag_mask = torch.eye(n, dtype=torch.bool, device=device)

            # Positive similarities (diagonal of sim_12)
            pos_sim = torch.diag(sim_12)  # [n]

            # Negatives from z2 (off-diagonal)
            neg_12 = sim_12.masked_fill(diag_mask, float('-inf'))
            # Negatives from z1 (off-diagonal, same view)
            neg_11 = sim_11.masked_fill(diag_mask, float('-inf'))

            # InfoNCE: log softmax with positive at index 0
            logits = torch.cat([pos_sim.unsqueeze(1), neg_12, neg_11], dim=1)
            labels = torch.zeros(n, dtype=torch.long, device=device)

            loss_g = F.cross_entropy(logits, labels)
            total_loss = total_loss + loss_g * n

        return total_loss / num_nodes

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get node embeddings without augmentation (for downstream tasks).
        """
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer to downstream tasks."""
        return self.encoder.state_dict()


class InfoGraph(nn.Module):
    """
    InfoGraph: Mutual information maximization between node and graph representations.

    Maximizes mutual information between:
    - Node embeddings and their containing graph's embedding (positive)
    - Node embeddings and other graphs' embeddings (negative)

    Useful for graph-level downstream tasks.

    Reference: "InfoGraph: Unsupervised and Semi-supervised Graph-Level
                Representation Learning via Mutual Information Maximization"

    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hidden_dim: Hidden dimension of encoder
        num_layers: Number of encoder layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Physics-guided encoder
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Discriminator for MI estimation (bilinear)
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Graph pooling
        self.pool = global_mean_pool

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with mutual information maximization.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Encode nodes
        node_emb = self.encoder(x, edge_index, edge_attr)

        # Pool to graph-level
        graph_emb = self.pool(node_emb, batch)

        # Compute MI loss
        loss = self._compute_mi_loss(node_emb, graph_emb, batch)

        return {
            "loss": loss,
            "node_emb": node_emb,
            "graph_emb": graph_emb,
        }

    def _compute_mi_loss(
        self,
        node_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon MI estimator loss.

        Positive pairs: (node_i, graph containing node_i)
        Negative pairs: (node_i, graph NOT containing node_i)
        """
        num_nodes = node_emb.size(0)
        num_graphs = graph_emb.size(0)
        device = node_emb.device

        # Get graph embedding for each node's containing graph
        pos_graph_emb = graph_emb[batch]  # [N, hidden_dim]

        # Positive scores
        pos_scores = self.discriminator(node_emb, pos_graph_emb).squeeze()  # [N]

        # Negative scores: pair each node with random different graph
        neg_batch = (batch + torch.randint(1, num_graphs, (num_nodes,), device=device)) % num_graphs
        neg_graph_emb = graph_emb[neg_batch]
        neg_scores = self.discriminator(node_emb, neg_graph_emb).squeeze()  # [N]

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        return pos_loss + neg_loss

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings for downstream tasks."""
        return self.encoder(x, edge_index, edge_attr)

    def get_encoder_state_dict(self) -> dict:
        """Get encoder weights for transfer."""
        return self.encoder.state_dict()
