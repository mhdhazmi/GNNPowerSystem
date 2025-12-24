"""
Contrastive Loss Functions for Self-Supervised Learning

Implements InfoNCE/NT-Xent loss and related contrastive objectives
for graph representation learning.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).

    Also known as InfoNCE loss. Used by SimCLR, GraphCL, GRACE, etc.

    For each anchor z_i with positive z_j:
        loss = -log(exp(sim(z_i, z_j)/tau) / sum_k(exp(sim(z_i, z_k)/tau)))

    where sim() is cosine similarity and tau is temperature.

    Args:
        temperature: Scaling factor for similarity scores (default: 0.5)
        normalize: Whether to L2-normalize embeddings (default: True)
    """

    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two views.

        For graph-level contrast (batch=None or all same):
            z1, z2: [B, D] where B is batch size
            Positive pairs: (z1[i], z2[i])
            Negative pairs: all other combinations

        For node-level contrast (batch provided):
            z1, z2: [N, D] where N is total nodes
            Positive pairs: same node across views
            Negatives: other nodes within same graph

        Args:
            z1: Embeddings from first view [B, D] or [N, D]
            z2: Embeddings from second view [B, D] or [N, D]
            batch: Optional batch assignment for node-level contrast

        Returns:
            Scalar loss value
        """
        if batch is None:
            return self._graph_level_loss(z1, z2)
        else:
            return self._node_level_loss(z1, z2, batch)

    def _graph_level_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Graph-level contrastive loss.

        Each graph in z1 is paired with corresponding graph in z2.
        All other graphs in the batch serve as negatives.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Normalize embeddings
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        # Concatenate both views: [2B, D]
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix: [2B, 2B]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Create mask for positive pairs
        # Positives: (i, i+B) and (i+B, i) for i in [0, B)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def _node_level_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Node-level contrastive loss (GRACE-style).

        For each node, its embedding in z1 should be similar to its
        embedding in z2 (positive pair), and dissimilar to other nodes
        within the same graph (negatives).
        """
        num_nodes = z1.size(0)
        device = z1.device

        # Normalize embeddings
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

        # Compute within-graph negative masks
        # Only nodes in same graph are valid negatives
        batch_expanded = batch.unsqueeze(0) == batch.unsqueeze(1)  # [N, N]

        # Total loss accumulator
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

            # Similarity matrices for this graph
            # Positive: sim(z1_g[i], z2_g[i])
            # Negatives: sim(z1_g[i], z1_g[j]) and sim(z1_g[i], z2_g[j]) for j != i

            # z1 -> z2 similarities
            sim_12 = torch.mm(z1_g, z2_g.t()) / self.temperature  # [n, n]
            # z1 -> z1 similarities (excluding diagonal)
            sim_11 = torch.mm(z1_g, z1_g.t()) / self.temperature

            # Positive similarities are on the diagonal of sim_12
            pos_sim = torch.diag(sim_12)  # [n]

            # Negatives: off-diagonal of sim_12 + off-diagonal of sim_11
            # Create mask to exclude diagonal
            diag_mask = torch.eye(n, dtype=torch.bool, device=device)

            # For each anchor in z1, compute log-sum-exp over negatives
            # Negatives from z2 (same graph, different node)
            neg_12 = sim_12.masked_fill(diag_mask, float('-inf'))
            # Negatives from z1 (same view, different node)
            neg_11 = sim_11.masked_fill(diag_mask, float('-inf'))

            # Combine all negatives
            all_neg = torch.cat([neg_12, neg_11], dim=1)  # [n, 2n]

            # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            # = -pos + log(exp(pos) + sum(exp(neg)))
            # = -pos + logsumexp([pos, neg1, neg2, ...])

            logits = torch.cat([pos_sim.unsqueeze(1), all_neg], dim=1)  # [n, 1+2n]
            labels = torch.zeros(n, dtype=torch.long, device=device)  # positive is at index 0

            loss_g = F.cross_entropy(logits, labels)
            total_loss = total_loss + loss_g * n

        # Average over all nodes
        return total_loss / num_nodes


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss - no explicit negatives needed.

    Maximizes correlation between embedding dimensions of positive pairs
    while decorrelating different dimensions (prevents collapse).

    Args:
        lambda_coeff: Weight for off-diagonal (decorrelation) terms
    """

    def __init__(self, lambda_coeff: float = 0.005):
        super().__init__()
        self.lambda_coeff = lambda_coeff

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute Barlow Twins loss.

        Args:
            z1: Embeddings from first view [B, D]
            z2: Embeddings from second view [B, D]

        Returns:
            Scalar loss value
        """
        batch_size = z1.size(0)

        # Normalize along batch dimension
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)

        # Cross-correlation matrix: [D, D]
        c = torch.mm(z1_norm.t(), z2_norm) / batch_size

        # Loss: diagonal should be 1, off-diagonal should be 0
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()
        off_diag = self._off_diagonal(c).pow(2).sum()

        loss = on_diag + self.lambda_coeff * off_diag

        return loss

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Extract off-diagonal elements of a square matrix."""
        n = x.size(0)
        # Flatten, skip diagonal elements, reshape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
