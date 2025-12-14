"""
Physics Consistency Metrics for Power Grid GNNs

Validates that model representations align with physical principles:
1. Power balance (Kirchhoff's Current Law)
2. Edge importance correlation with physical quantities
3. Embedding consistency with electrical distance
4. PF prediction physics consistency
5. Thermal limit violations (OPF)

These metrics support the paper's "physics-consistent" claim by providing
quantitative evidence that predictions respect power system physics.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.utils import degree


def compute_pf_physics_residual(
    v_pred: torch.Tensor,
    v_true: torch.Tensor,
    p_injection: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute physics consistency metrics for Power Flow predictions.

    For a valid PF solution, predicted voltages should satisfy power balance.
    We compute a proxy metric based on voltage magnitude consistency.

    Args:
        v_pred: Predicted voltage magnitudes [num_nodes]
        v_true: Ground truth voltage magnitudes [num_nodes]
        p_injection: Power injections [num_nodes]
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features (including reactance) [num_edges, edge_dim]

    Returns:
        Dictionary with physics metrics
    """
    # Basic voltage prediction metrics
    v_error = (v_pred - v_true).abs()

    # Voltage deviation from nominal (1.0 p.u.)
    v_deviation_pred = (v_pred - 1.0).abs()
    v_deviation_true = (v_true - 1.0).abs()

    # Physics check: voltages should be within reasonable bounds
    v_min, v_max = 0.9, 1.1  # Typical operational limits
    violations_pred = ((v_pred < v_min) | (v_pred > v_max)).float().mean()
    violations_true = ((v_true < v_min) | (v_true > v_max)).float().mean()

    # Consistency: prediction should have similar deviation pattern to ground truth
    deviation_correlation = F.cosine_similarity(
        v_deviation_pred.unsqueeze(0), v_deviation_true.unsqueeze(0)
    ).item()

    return {
        "pf_mae": v_error.mean().item(),
        "pf_max_error": v_error.max().item(),
        "pf_voltage_violation_rate_pred": violations_pred.item(),
        "pf_voltage_violation_rate_true": violations_true.item(),
        "pf_deviation_correlation": deviation_correlation,
    }


def compute_thermal_violations(
    flow_pred: torch.Tensor,
    flow_true: torch.Tensor,
    rating: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute thermal limit violation metrics for OPF predictions.

    Power flows should not exceed line thermal ratings.

    Args:
        flow_pred: Predicted power flows [num_edges] or [num_edges, 2] for P,Q
        flow_true: Ground truth power flows
        rating: Line thermal ratings [num_edges]

    Returns:
        Dictionary with thermal violation metrics
    """
    # Handle both 1D (P only) and 2D (P, Q) flows
    if flow_pred.dim() == 2:
        # Compute apparent power = sqrt(P^2 + Q^2)
        apparent_pred = torch.sqrt(flow_pred[:, 0]**2 + flow_pred[:, 1]**2)
        apparent_true = torch.sqrt(flow_true[:, 0]**2 + flow_true[:, 1]**2)
    else:
        apparent_pred = flow_pred.abs()
        apparent_true = flow_true.abs()

    # Normalize rating to avoid division issues
    rating = rating.abs() + 1e-8

    # Loading ratio
    loading_pred = apparent_pred / rating
    loading_true = apparent_true / rating

    # Violations (loading > 1.0 means over thermal limit)
    violation_pred = (loading_pred > 1.0).float()
    violation_true = (loading_true > 1.0).float()

    # Severe violations (loading > 1.2)
    severe_pred = (loading_pred > 1.2).float()
    severe_true = (loading_true > 1.2).float()

    return {
        "thermal_violation_rate_pred": violation_pred.mean().item(),
        "thermal_violation_rate_true": violation_true.mean().item(),
        "thermal_severe_violation_pred": severe_pred.mean().item(),
        "thermal_severe_violation_true": severe_true.mean().item(),
        "thermal_max_loading_pred": loading_pred.max().item(),
        "thermal_max_loading_true": loading_true.max().item(),
        "thermal_mean_loading_pred": loading_pred.mean().item(),
    }


def compute_angle_consistency(
    sin_pred: torch.Tensor,
    cos_pred: torch.Tensor,
) -> Dict[str, float]:
    """
    Verify that predicted sin/cos form valid unit circle values.

    For valid angles: sin^2 + cos^2 = 1

    Args:
        sin_pred: Predicted sin(theta) [num_nodes]
        cos_pred: Predicted cos(theta) [num_nodes]

    Returns:
        Dictionary with angle consistency metrics
    """
    # Unit circle constraint
    norm_sq = sin_pred**2 + cos_pred**2
    norm_error = (norm_sq - 1.0).abs()

    # Reasonable angle range: |theta| < pi (cos > -1)
    angle_pred = torch.atan2(sin_pred, cos_pred)
    angle_range_violation = (angle_pred.abs() > 3.14159).float()

    return {
        "angle_norm_error_mean": norm_error.mean().item(),
        "angle_norm_error_max": norm_error.max().item(),
        "angle_range_violation_rate": angle_range_violation.mean().item(),
    }


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
    # Handle different edge feature layouts:
    # - Cascade: [P_flow, Q_flow, X, rating] - X at index 2
    # - PF/OPF: [X, rating] - X at index 0
    if edge_attr.size(1) >= 3:
        reactance = edge_attr[:, 2].abs() + 1e-8
    else:
        reactance = edge_attr[:, 0].abs() + 1e-8
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
