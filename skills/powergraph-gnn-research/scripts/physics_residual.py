#!/usr/bin/env python3
"""
Physics Residual Metrics

Compute and track Kirchhoff's Current Law (KCL) mismatch as a physics consistency metric.
This is a key "soundness anchor" for publishable GNN surrogate models.

The physics residual should be:
- Near numerical tolerance for ground-truth solver outputs
- Low for a well-trained model (indicates physical consistency)
- High for random predictions (sanity check)

Usage:
    python physics_residual.py --checkpoint ./checkpoints/best.pt --data_dir ./data
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import argparse


def compute_power_mismatch_dc(
    v_mag: torch.Tensor,
    v_angle: torch.Tensor,
    edge_index: torch.Tensor,
    susceptance: torch.Tensor,
    p_inject: torch.Tensor,
) -> torch.Tensor:
    """
    Compute DC power flow mismatch (simplified).
    
    DC approximation: P_ij = B_ij * (θ_i - θ_j)
    KCL: P_inject_i = sum_j P_ij for all lines connected to bus i
    
    Args:
        v_mag: Voltage magnitudes [num_nodes] (ignored in DC)
        v_angle: Voltage angles in radians [num_nodes]
        edge_index: Edge connectivity [2, num_edges]
        susceptance: Line susceptances [num_edges]
        p_inject: Net power injection at each bus [num_nodes]
    
    Returns:
        Power mismatch at each bus [num_nodes]
    """
    num_nodes = v_angle.shape[0]
    
    src, dst = edge_index[0], edge_index[1]
    
    # Line flows (DC approximation)
    angle_diff = v_angle[src] - v_angle[dst]
    p_flow = susceptance * angle_diff  # B * (θ_i - θ_j)
    
    # Sum of outgoing flows per bus
    p_out = torch.zeros(num_nodes, device=v_angle.device)
    p_out.scatter_add_(0, src, p_flow)
    p_out.scatter_add_(0, dst, -p_flow)  # Opposite direction
    
    # Mismatch = injection - outflow
    mismatch = p_inject - p_out
    
    return mismatch


def compute_power_mismatch_ac(
    v_mag: torch.Tensor,
    v_angle: torch.Tensor,
    edge_index: torch.Tensor,
    conductance: torch.Tensor,
    susceptance: torch.Tensor,
    p_inject: torch.Tensor,
    q_inject: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute AC power flow mismatch (more accurate).
    
    P_ij = V_i * V_j * (G_ij * cos(θ_i - θ_j) + B_ij * sin(θ_i - θ_j))
    Q_ij = V_i * V_j * (G_ij * sin(θ_i - θ_j) - B_ij * cos(θ_i - θ_j))
    
    Args:
        v_mag: Voltage magnitudes [num_nodes]
        v_angle: Voltage angles in radians [num_nodes]
        edge_index: Edge connectivity [2, num_edges]
        conductance: Line conductances G [num_edges]
        susceptance: Line susceptances B [num_edges]
        p_inject: Net active power injection [num_nodes]
        q_inject: Net reactive power injection [num_nodes]
    
    Returns:
        (p_mismatch, q_mismatch) at each bus
    """
    num_nodes = v_mag.shape[0]
    
    src, dst = edge_index[0], edge_index[1]
    
    # Voltage products
    v_src = v_mag[src]
    v_dst = v_mag[dst]
    v_prod = v_src * v_dst
    
    # Angle differences
    angle_diff = v_angle[src] - v_angle[dst]
    cos_diff = torch.cos(angle_diff)
    sin_diff = torch.sin(angle_diff)
    
    # Line flows (AC)
    p_flow = v_prod * (conductance * cos_diff + susceptance * sin_diff)
    q_flow = v_prod * (conductance * sin_diff - susceptance * cos_diff)
    
    # Sum outgoing flows per bus
    p_out = torch.zeros(num_nodes, device=v_mag.device)
    q_out = torch.zeros(num_nodes, device=v_mag.device)
    
    p_out.scatter_add_(0, src, p_flow)
    p_out.scatter_add_(0, dst, -p_flow)
    
    q_out.scatter_add_(0, src, q_flow)
    q_out.scatter_add_(0, dst, -q_flow)
    
    # Mismatch
    p_mismatch = p_inject - p_out
    q_mismatch = q_inject - q_out
    
    return p_mismatch, q_mismatch


class PhysicsResidualMetric:
    """
    Track physics residual as a metric during training/evaluation.
    """
    
    def __init__(self, mode: str = 'dc'):
        """
        Args:
            mode: 'dc' for DC approximation, 'ac' for full AC equations
        """
        self.mode = mode
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.p_mismatches = []
        self.q_mismatches = []
    
    def update(
        self,
        pred_v_mag: torch.Tensor,
        pred_v_angle: torch.Tensor,
        data,  # PyG Data object with edge info and injections
    ):
        """
        Update metric with batch predictions.
        
        Args:
            pred_v_mag: Predicted voltage magnitudes
            pred_v_angle: Predicted voltage angles (radians)
            data: PyG Data object containing edge_index, edge_attr, and injection info
        """
        # Extract edge parameters (assumes edge_attr contains G, B)
        # Adapt indices based on your actual data format
        edge_index = data.edge_index
        
        if data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
            conductance = data.edge_attr[:, 0]
            susceptance = data.edge_attr[:, 1]
        else:
            # Default if not available
            susceptance = torch.ones(edge_index.shape[1], device=pred_v_mag.device)
            conductance = torch.zeros_like(susceptance)
        
        # Extract injections (assumes x contains P, Q)
        if data.x.shape[1] >= 4:
            p_load = data.x[:, 0]
            q_load = data.x[:, 1]
            p_gen = data.x[:, 2]
            q_gen = data.x[:, 3]
            p_inject = p_gen - p_load
            q_inject = q_gen - q_load
        else:
            p_inject = torch.zeros(pred_v_mag.shape[0], device=pred_v_mag.device)
            q_inject = torch.zeros(pred_v_mag.shape[0], device=pred_v_mag.device)
        
        if self.mode == 'dc':
            p_mismatch = compute_power_mismatch_dc(
                pred_v_mag, pred_v_angle, edge_index, susceptance, p_inject
            )
            self.p_mismatches.append(p_mismatch.abs().detach())
        else:
            p_mismatch, q_mismatch = compute_power_mismatch_ac(
                pred_v_mag, pred_v_angle, edge_index,
                conductance, susceptance, p_inject, q_inject
            )
            self.p_mismatches.append(p_mismatch.abs().detach())
            self.q_mismatches.append(q_mismatch.abs().detach())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute aggregate physics residual metrics.
        """
        all_p = torch.cat(self.p_mismatches)
        
        metrics = {
            'p_residual_mean': all_p.mean().item(),
            'p_residual_max': all_p.max().item(),
            'p_residual_std': all_p.std().item(),
            'p_residual_median': all_p.median().item(),
        }
        
        if self.q_mismatches:
            all_q = torch.cat(self.q_mismatches)
            metrics.update({
                'q_residual_mean': all_q.mean().item(),
                'q_residual_max': all_q.max().item(),
                'q_residual_std': all_q.std().item(),
            })
        
        return metrics


def physics_consistency_loss(
    pred_v_mag: torch.Tensor,
    pred_v_angle: torch.Tensor,
    data,
    lambda_p: float = 0.1,
    lambda_q: float = 0.1,
) -> torch.Tensor:
    """
    Physics consistency loss for regularization.
    
    Add to main loss: total_loss = task_loss + physics_consistency_loss(...)
    """
    edge_index = data.edge_index
    
    # Extract parameters (adapt to your data format)
    if data.edge_attr is not None and data.edge_attr.shape[1] >= 2:
        conductance = data.edge_attr[:, 0]
        susceptance = data.edge_attr[:, 1]
    else:
        susceptance = torch.ones(edge_index.shape[1], device=pred_v_mag.device)
        conductance = torch.zeros_like(susceptance)
    
    if data.x.shape[1] >= 4:
        p_inject = data.x[:, 2] - data.x[:, 0]
        q_inject = data.x[:, 3] - data.x[:, 1]
    else:
        return torch.tensor(0.0, device=pred_v_mag.device)
    
    p_mismatch, q_mismatch = compute_power_mismatch_ac(
        pred_v_mag, pred_v_angle, edge_index,
        conductance, susceptance, p_inject, q_inject
    )
    
    loss = lambda_p * p_mismatch.abs().mean() + lambda_q * q_mismatch.abs().mean()
    
    return loss


def evaluate_physics_consistency(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate model's physics consistency on a dataset.
    """
    model.eval()
    metric = PhysicsResidualMetric(mode='ac')
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            
            # Recover angle from sin/cos
            pred_angle = torch.atan2(pred['sin_theta'], pred['cos_theta'])
            
            metric.update(pred['v_mag'], pred_angle, batch)
    
    return metric.compute()


def compare_physics_residuals(
    model_pred: Dict,
    ground_truth: Dict,
    random_pred: Dict,
):
    """
    Compare physics residuals across predictions.
    
    Expectation:
    - Ground truth: near zero (numerical tolerance)
    - Model: low (good physics consistency)
    - Random: high (sanity check)
    """
    print("\n" + "="*60)
    print("Physics Residual Comparison")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Ground Truth':<15} {'Model':<15} {'Random':<15}")
    print("-"*70)
    
    for key in ['p_residual_mean', 'p_residual_max']:
        gt = ground_truth.get(key, 'N/A')
        model = model_pred.get(key, 'N/A')
        rand = random_pred.get(key, 'N/A')
        
        if isinstance(gt, float):
            gt = f"{gt:.6f}"
        if isinstance(model, float):
            model = f"{model:.6f}"
        if isinstance(rand, float):
            rand = f"{rand:.6f}"
        
        print(f"{key:<25} {gt:<15} {model:<15} {rand:<15}")
    
    print("\n✓ Ground truth should have near-zero residuals")
    print("✓ Model residual should be << Random residual")
    print("✓ Lower is better (more physically consistent)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate physics residuals')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Demo with dummy data
    print("Physics Residual Metric Demo")
    print("="*50)
    
    # Create dummy scenario
    num_nodes = 24
    num_edges = 34
    
    v_mag = torch.ones(num_nodes)  # 1.0 p.u.
    v_angle = torch.zeros(num_nodes)  # All at 0 degrees
    
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    susceptance = torch.rand(num_edges) * 10
    p_inject = torch.randn(num_nodes)
    
    # Compute DC mismatch
    mismatch = compute_power_mismatch_dc(
        v_mag, v_angle, edge_index, susceptance, p_inject
    )
    
    print(f"Sample DC power mismatch:")
    print(f"  Mean: {mismatch.abs().mean():.6f}")
    print(f"  Max:  {mismatch.abs().max():.6f}")
    print(f"  Std:  {mismatch.std():.6f}")
    
    # The mismatch won't be zero because we have random injections
    # In a real scenario with consistent solver output, it should be ~0


if __name__ == '__main__':
    main()
