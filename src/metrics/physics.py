"""
Physics Consistency Metrics for Power Grid GNNs

Validates that model representations align with physical principles:
1. Power balance (Kirchhoff's Current Law)
2. Edge importance correlation with physical quantities
3. Embedding consistency with electrical distance
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def compute_power_balance_residual(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute power balance residual for each node.

    In a balanced power system:
    - Net injection at node i = Sum of flows on incident edges

    Node features: [P_net, S_net, V]
    Edge features: [P_flow, Q_flow, X, rating]

    Args:
        x: Node features [num_nodes, 3]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, 4]

    Returns:
        Dictionary with residual metrics
    """
    num_nodes = x.size(0)
    device = x.device

    # Extract features
    p_injection = x[:, 0]  # Net active power at each node
    p_flow = edge_attr[:, 0]  # Active power flow on each edge

    # Compute net flow into each node
    # For edge (i, j), flow is from i to j, so it's negative for i, positive for j
    src, dst = edge_index

    # Sum of incoming flows (positive direction)
    flow_in = torch.zeros(num_nodes, device=device)
    flow_in.scatter_add_(0, dst, p_flow)

    # Sum of outgoing flows (negative direction)
    flow_out = torch.zeros(num_nodes, device=device)
    flow_out.scatter_add_(0, src, p_flow)

    # Net flow = incoming - outgoing (for bidirectional edges, this should balance)
    net_flow = flow_in - flow_out

    # Residual = injection - net_flow (should be ~0 for balanced system)
    # Note: Due to normalization in dataset, this is approximate
    residual = p_injection - net_flow

    return {
        "residual_mean": residual.abs().mean(),
        "residual_std": residual.std(),
        "residual_max": residual.abs().max(),
        "residual_per_node": residual,
    }


def compute_edge_importance_physics_correlation(
    edge_importance: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute correlation between learned edge importance and physical quantities.

    Physical importance indicators:
    - High power flow (P, Q) = more important
    - Low reactance (X) = stronger connection = more important
    - High loading (flow/rating) = critical edge

    Args:
        edge_importance: Learned importance scores [num_edges]
        edge_attr: Edge features [num_edges, 4] - [P_flow, Q_flow, X, rating]

    Returns:
        Correlation metrics
    """
    # Normalize importance to [0, 1]
    imp = edge_importance.detach()
    imp = (imp - imp.min()) / (imp.max() - imp.min() + 1e-8)

    # Physical quantities
    p_flow = edge_attr[:, 0].abs()
    q_flow = edge_attr[:, 1].abs()
    reactance = edge_attr[:, 2].abs()
    rating = edge_attr[:, 3]

    # Admittance (inverse of reactance) - higher = stronger connection
    admittance = 1.0 / (reactance + 1e-8)

    # Loading ratio (flow / rating) - higher = more stressed
    apparent_flow = torch.sqrt(p_flow**2 + q_flow**2)
    # Rating might be normalized, use relative loading
    loading = apparent_flow / (apparent_flow.max() + 1e-8)

    def pearson_corr(x, y):
        """Compute Pearson correlation coefficient."""
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (torch.sqrt((x**2).sum() * (y**2).sum()) + 1e-8)

    return {
        "corr_p_flow": pearson_corr(imp, p_flow).item(),
        "corr_q_flow": pearson_corr(imp, q_flow).item(),
        "corr_admittance": pearson_corr(imp, admittance).item(),
        "corr_loading": pearson_corr(imp, loading).item(),
    }


def compute_embedding_electrical_consistency(
    node_emb: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[str, float]:
    """
    Check if node embeddings respect electrical distance.

    Nodes connected by low-impedance lines should have similar embeddings.

    Args:
        node_emb: Node embeddings [num_nodes, hidden_dim]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, 4]

    Returns:
        Consistency metrics
    """
    src, dst = edge_index

    # Embedding similarity for connected nodes
    src_emb = node_emb[src]
    dst_emb = node_emb[dst]

    # Cosine similarity
    cos_sim = F.cosine_similarity(src_emb, dst_emb, dim=1)

    # Euclidean distance
    euclidean_dist = torch.norm(src_emb - dst_emb, dim=1)

    # Physical: admittance (inverse reactance)
    reactance = edge_attr[:, 2].abs() + 1e-8
    admittance = 1.0 / reactance

    # Expectation: high admittance edges should have high similarity
    def pearson_corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        return (x * y).sum() / (torch.sqrt((x**2).sum() * (y**2).sum()) + 1e-8)

    return {
        "emb_similarity_mean": cos_sim.mean().item(),
        "emb_similarity_std": cos_sim.std().item(),
        "emb_distance_mean": euclidean_dist.mean().item(),
        "corr_similarity_admittance": pearson_corr(cos_sim, admittance).item(),
        "corr_distance_reactance": pearson_corr(euclidean_dist, reactance).item(),
    }


def evaluate_physics_consistency(
    model: torch.nn.Module,
    data,
    device: torch.device,
) -> Dict[str, float]:
    """
    Comprehensive physics consistency evaluation.

    Args:
        model: Trained model with encoder
        data: PyG Data object
        device: Torch device

    Returns:
        Dictionary of all physics metrics
    """
    model.eval()
    data = data.to(device)

    metrics = {}

    with torch.no_grad():
        # Get embeddings
        if hasattr(model, "encoder"):
            node_emb = model.encoder(data.x, data.edge_index, data.edge_attr)
        elif hasattr(model, "encode"):
            node_emb = model.encode(data.x, data.edge_index, data.edge_attr)
        else:
            # Try forward pass
            node_emb = model(data.x, data.edge_index, data.edge_attr)
            if isinstance(node_emb, dict):
                return {"error": "Model returns dict, not embeddings"}

        # Power balance
        balance = compute_power_balance_residual(data.x, data.edge_index, data.edge_attr)
        metrics["power_balance_residual_mean"] = balance["residual_mean"].item()
        metrics["power_balance_residual_max"] = balance["residual_max"].item()

        # Embedding consistency
        consistency = compute_embedding_electrical_consistency(
            node_emb, data.edge_index, data.edge_attr
        )
        metrics.update(consistency)

    # Edge importance correlation (if model supports it)
    if hasattr(model, "get_edge_importance_attention"):
        importance = model.get_edge_importance_attention(
            data.x, data.edge_index, data.edge_attr
        )
        corr = compute_edge_importance_physics_correlation(importance, data.edge_attr)
        metrics.update({f"importance_{k}": v for k, v in corr.items()})

    return metrics
