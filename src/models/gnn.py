"""
PowerGraph GNN Models

Complete multi-task model combining encoder and task-specific heads.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PhysicsGuidedEncoder, SimpleGNNEncoder
from .heads import CascadeBinaryHead, CascadeHead, OPFHead, PowerFlowHead


class PowerGraphGNN(nn.Module):
    """
    Complete multi-task GNN model for PF/OPF/Cascade.

    Architecture:
        Input Graph → Physics-Guided Encoder → Task-Specific Heads
                                               ├── PF Head (node-level)
                                               ├── OPF Head (node-level)
                                               └── Cascade Head (graph-level)
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_cascade_classes: int = 2,
        dropout: float = 0.1,
        encoder_type: str = "physics_guided",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        # Encoder
        if encoder_type == "physics_guided":
            self.encoder = PhysicsGuidedEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            self.encoder = SimpleGNNEncoder(
                node_in_dim=node_in_dim,
                edge_in_dim=edge_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )

        # Task heads
        self.pf_head = PowerFlowHead(hidden_dim)
        self.opf_head = OPFHead(hidden_dim)
        self.cascade_head = CascadeBinaryHead(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        gen_mask: Optional[torch.Tensor] = None,
        tasks: Tuple[str, ...] = ("pf",),
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through encoder and selected task heads.

        Args:
            x: Node features [num_nodes, node_in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_in_dim]
            batch: Batch assignment for nodes [num_nodes]
            gen_mask: Generator mask for OPF [num_nodes]
            tasks: Which task heads to use

        Returns:
            Dictionary of task outputs
        """
        # Shared encoding
        node_emb = self.encoder(x, edge_index, edge_attr)

        outputs = {}

        if "pf" in tasks:
            outputs["pf"] = self.pf_head(node_emb)

        if "opf" in tasks:
            outputs["opf"] = self.opf_head(node_emb, gen_mask, batch)

        if "cascade" in tasks:
            outputs["cascade"] = self.cascade_head(node_emb, batch)

        return outputs

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Extract node embeddings for analysis/visualization."""
        return self.encoder(x, edge_index, edge_attr)


class PFBaselineModel(nn.Module):
    """
    Simple baseline model for Power Flow prediction only.

    Use this for initial experiments before multi-task training.
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

        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.head = PowerFlowHead(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict voltage magnitude and angle.

        Returns:
            Dictionary with 'v_mag', 'sin_theta', 'cos_theta'
        """
        node_emb = self.encoder(x, edge_index, edge_attr)
        return self.head(node_emb)


class CascadeBaselineModel(nn.Module):
    """
    Simple baseline model for Cascade prediction only.

    Includes explanation extraction methods for edge importance.
    """

    def __init__(
        self,
        node_in_dim: int = 3,
        edge_in_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()

        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.head = CascadeBinaryHead(hidden_dim) if num_classes == 2 else CascadeHead(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict cascade severity.

        Returns:
            Dictionary with 'logits'
        """
        node_emb = self.encoder(x, edge_index, edge_attr)
        return self.head(node_emb, batch)

    def get_edge_importance_gradient(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge importance via gradient-based attribution.

        Uses gradient of prediction w.r.t. edge features as importance score.

        Returns:
            Edge importance scores [num_edges] (higher = more important)
        """
        # Enable gradients for edge features
        edge_attr_grad = edge_attr.clone().requires_grad_(True)

        # Forward pass
        node_emb = self.encoder(x, edge_index, edge_attr_grad)
        outputs = self.head(node_emb, batch)
        logits = outputs["logits"]

        # For binary, take sum of positive class scores
        if logits.dim() == 1 or logits.size(-1) == 1:
            score = logits.sum()
        else:
            score = logits[:, 1].sum()  # Positive class

        # Backward pass
        score.backward()

        # Edge importance = L2 norm of gradient across feature dimensions
        edge_importance = edge_attr_grad.grad.abs().mean(dim=1)

        return edge_importance.detach()

    def get_edge_importance_attention(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge importance via attention-like scores from encoder.

        Uses the physics-guided weights (admittance magnitudes) as base,
        combined with learned edge representations.

        Returns:
            Edge importance scores [num_edges]
        """
        with torch.no_grad():
            # Get node embeddings
            node_emb = self.encoder(x, edge_index, edge_attr)

            # Edge importance from admittance (reactance is 3rd feature)
            # High admittance (1/X) = high importance
            reactance = edge_attr[:, 2].abs() + 1e-8
            admittance = 1.0 / reactance

            # Combine with edge attribute magnitude
            edge_magnitude = edge_attr.abs().mean(dim=1)

            # Also consider endpoint node embedding similarity
            src, dst = edge_index
            src_emb = node_emb[src]
            dst_emb = node_emb[dst]
            emb_similarity = (src_emb * dst_emb).sum(dim=1)

            # Normalize each component
            admittance_norm = admittance / (admittance.max() + 1e-8)
            edge_mag_norm = edge_magnitude / (edge_magnitude.max() + 1e-8)
            sim_norm = (emb_similarity - emb_similarity.min()) / (
                emb_similarity.max() - emb_similarity.min() + 1e-8
            )

            # Combine scores
            importance = admittance_norm * 0.4 + edge_mag_norm * 0.3 + sim_norm * 0.3

        return importance

    def get_edge_importance_integrated_gradients(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        steps: int = 20,
    ) -> torch.Tensor:
        """
        Compute edge importance via integrated gradients.

        More robust than simple gradients - integrates gradient along
        path from baseline (zero) to actual edge features.

        Returns:
            Edge importance scores [num_edges]
        """
        baseline = torch.zeros_like(edge_attr)

        # Accumulate gradients along interpolation path
        accumulated_grads = torch.zeros_like(edge_attr)

        for step in range(1, steps + 1):
            alpha = step / steps
            interpolated = baseline + alpha * (edge_attr - baseline)
            interpolated = interpolated.clone().requires_grad_(True)

            # Forward pass
            node_emb = self.encoder(x, edge_index, interpolated)
            outputs = self.head(node_emb, batch)
            logits = outputs["logits"]

            if logits.dim() == 1 or logits.size(-1) == 1:
                score = logits.sum()
            else:
                score = logits[:, 1].sum()

            score.backward()
            accumulated_grads += interpolated.grad

        # Average gradients and multiply by input difference
        avg_grads = accumulated_grads / steps
        integrated_grads = (edge_attr - baseline) * avg_grads

        # Edge importance = L2 norm across features
        edge_importance = integrated_grads.abs().mean(dim=1)

        return edge_importance.detach()


# Loss functions

def pf_loss(
    pred: Dict[str, torch.Tensor],
    target_v_mag: torch.Tensor,
    target_sin: torch.Tensor,
    target_cos: torch.Tensor,
    lambda_norm: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Power flow loss.

    Args:
        pred: Model predictions with 'v_mag', 'sin_theta', 'cos_theta'
        target_v_mag: Ground truth voltage magnitudes
        target_sin: Ground truth sin(θ)
        target_cos: Ground truth cos(θ)
        lambda_norm: Weight for sin²+cos² normalization loss

    Returns:
        Total loss and component breakdown
    """
    # Voltage magnitude
    loss_v = F.mse_loss(pred["v_mag"], target_v_mag)

    # Angle (via sin/cos)
    loss_sin = F.mse_loss(pred["sin_theta"], target_sin)
    loss_cos = F.mse_loss(pred["cos_theta"], target_cos)

    # Normalization constraint: sin²+cos² should be 1
    # This is already enforced in the head, but we can add soft regularization
    pred_norm = pred["sin_theta"] ** 2 + pred["cos_theta"] ** 2
    loss_norm = F.mse_loss(pred_norm, torch.ones_like(pred_norm))

    total = loss_v + loss_sin + loss_cos + lambda_norm * loss_norm

    breakdown = {
        "loss_v_mag": loss_v.item(),
        "loss_sin": loss_sin.item(),
        "loss_cos": loss_cos.item(),
        "loss_norm": loss_norm.item(),
        "loss_total": total.item(),
    }

    return total, breakdown


def cascade_loss(
    pred: Dict[str, torch.Tensor],
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Cascade classification loss.

    Args:
        pred: Model predictions with 'logits'
        target: Ground truth labels
        class_weights: Optional class weights for imbalanced data
        pos_weight: Optional positive class weight for binary classification (for class imbalance)

    Returns:
        Total loss and breakdown
    """
    logits = pred["logits"]

    # Handle both binary and multi-class
    if logits.dim() == 1 or logits.size(-1) == 1:
        # Binary classification
        logits = logits.view(-1)
        target = target.view(-1).float()
        loss = F.binary_cross_entropy_with_logits(logits, target, weight=class_weights, pos_weight=pos_weight)
    else:
        # Multi-class
        loss = F.cross_entropy(logits, target.long(), weight=class_weights)

    return loss, {"loss_cascade": loss.item()}
