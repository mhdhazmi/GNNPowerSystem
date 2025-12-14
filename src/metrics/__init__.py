"""Metrics for PowerGraph GNN evaluation."""

from .physics import (
    compute_angle_consistency,
    compute_edge_importance_physics_correlation,
    compute_embedding_electrical_consistency,
    compute_pf_physics_residual,
    compute_power_balance_residual,
    compute_thermal_violations,
    evaluate_physics_consistency,
)

__all__ = [
    "compute_angle_consistency",
    "compute_edge_importance_physics_correlation",
    "compute_embedding_electrical_consistency",
    "compute_pf_physics_residual",
    "compute_power_balance_residual",
    "compute_thermal_violations",
    "evaluate_physics_consistency",
]
