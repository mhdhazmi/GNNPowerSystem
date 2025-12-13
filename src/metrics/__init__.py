"""Metrics for PowerGraph GNN evaluation."""

from .physics import (
    compute_edge_importance_physics_correlation,
    compute_embedding_electrical_consistency,
    compute_power_balance_residual,
    evaluate_physics_consistency,
)

__all__ = [
    "compute_power_balance_residual",
    "compute_edge_importance_physics_correlation",
    "compute_embedding_electrical_consistency",
    "evaluate_physics_consistency",
]
