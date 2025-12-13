#!/usr/bin/env python3
"""
Cascade explanation fidelity evaluation against PowerGraph ground-truth masks.

Usage:
    python scripts/eval_cascade_explanation.py --model_path outputs/best_model.pt

Evaluates:
- Explanation AUC: edge importance scores vs binary ground-truth mask
- Precision@K: K = number of truly important edges
- Coverage@K: fraction of true important edges in top-K predictions

This is a key publication metric - demonstrates learned representations 
capture meaningful propagation patterns, not just classification accuracy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve
)
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj

# ============================================================================
# Edge Importance Extraction Methods
# ============================================================================

class EdgeImportanceExtractor:
    """Extract edge importance scores via different methods."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def attention_based(self, data: Data) -> torch.Tensor:
        """Extract importance from attention weights (if model uses GAT)."""
        self.model.eval()
        data = data.to(self.device)
        
        # Get attention weights from cascade head
        outputs = self.model(data, tasks=['cascade'])
        _, att_weights = outputs['cascade']
        
        # Convert node attention to edge importance
        # Edges connecting high-attention nodes are more important
        edge_index = data.edge_index
        src_att = att_weights[edge_index[0]].squeeze()
        tgt_att = att_weights[edge_index[1]].squeeze()
        edge_importance = (src_att + tgt_att) / 2
        
        return edge_importance.cpu()
    
    def gradient_based(self, data: Data, target_class: Optional[int] = None) -> torch.Tensor:
        """Integrated gradients for edge importance."""
        self.model.train()  # Need gradients
        data = data.to(self.device)
        
        # Get baseline (zero edge features)
        baseline_edge_attr = torch.zeros_like(data.edge_attr)
        
        # Interpolation steps
        n_steps = 50
        edge_importance = torch.zeros(data.edge_index.size(1), device=self.device)
        
        for alpha in torch.linspace(0, 1, n_steps):
            # Interpolate edge attributes
            interp_edge_attr = baseline_edge_attr + alpha * (data.edge_attr - baseline_edge_attr)
            interp_edge_attr.requires_grad_(True)
            
            # Forward pass
            interp_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=interp_edge_attr,
                batch=data.batch
            )
            outputs = self.model(interp_data, tasks=['cascade'])
            logits, _ = outputs['cascade']
            
            # Target: predicted class or specified class
            if target_class is None:
                target = logits.argmax(dim=1)
            else:
                target = torch.tensor([target_class], device=self.device)
            
            # Compute gradients w.r.t. edge attributes
            loss = logits[0, target]
            grad = torch.autograd.grad(loss, interp_edge_attr, create_graph=False)[0]
            
            # Accumulate
            edge_importance += grad.abs().sum(dim=1)
        
        # Scale by (x - baseline)
        diff = (data.edge_attr - baseline_edge_attr).abs().sum(dim=1)
        edge_importance = edge_importance * diff / n_steps
        
        return edge_importance.detach().cpu()
    
    @torch.no_grad()
    def embedding_similarity(self, data: Data) -> torch.Tensor:
        """Edge importance via endpoint embedding similarity."""
        self.model.eval()
        data = data.to(self.device)
        
        # Get node embeddings
        node_emb = self.model.get_encoder_embeddings(data)
        
        # Compute edge scores as similarity between endpoints
        edge_index = data.edge_index
        src_emb = node_emb[edge_index[0]]
        tgt_emb = node_emb[edge_index[1]]
        
        # Cosine similarity
        edge_importance = F.cosine_similarity(src_emb, tgt_emb, dim=1)
        
        return edge_importance.cpu()
    
    @torch.no_grad()
    def perturbation_based(self, data: Data, n_samples: int = 10) -> torch.Tensor:
        """Importance via edge masking perturbation."""
        self.model.eval()
        data = data.to(self.device)
        n_edges = data.edge_index.size(1)
        
        # Baseline prediction
        outputs = self.model(data, tasks=['cascade'])
        baseline_logits, _ = outputs['cascade']
        baseline_prob = F.softmax(baseline_logits, dim=1)
        
        # Measure importance by masking each edge
        edge_importance = torch.zeros(n_edges)
        
        for i in range(n_edges):
            # Create mask (all ones except edge i)
            mask = torch.ones(n_edges, dtype=torch.bool)
            mask[i] = False
            
            # Perturbed graph
            perturbed_edge_index = data.edge_index[:, mask]
            perturbed_edge_attr = data.edge_attr[mask]
            
            perturbed_data = Data(
                x=data.x,
                edge_index=perturbed_edge_index,
                edge_attr=perturbed_edge_attr,
                batch=data.batch
            )
            
            # Measure change
            outputs = self.model(perturbed_data, tasks=['cascade'])
            perturbed_logits, _ = outputs['cascade']
            perturbed_prob = F.softmax(perturbed_logits, dim=1)
            
            # KL divergence as importance
            importance = F.kl_div(
                perturbed_prob.log(), baseline_prob, reduction='sum'
            ).item()
            edge_importance[i] = importance
        
        return edge_importance


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_explanation_metrics(
    edge_scores: np.ndarray,
    ground_truth_mask: np.ndarray,
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute explanation fidelity metrics.
    
    Args:
        edge_scores: Model's edge importance scores (higher = more important)
        ground_truth_mask: Binary mask (1 = important edge, 0 = not important)
        k_values: Values of K for Precision@K / Recall@K
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # ROC AUC
    try:
        metrics['auc'] = roc_auc_score(ground_truth_mask, edge_scores)
    except ValueError:
        metrics['auc'] = 0.5  # All same class
    
    # Average Precision (area under PR curve)
    metrics['avg_precision'] = average_precision_score(ground_truth_mask, edge_scores)
    
    # Number of truly important edges
    n_important = int(ground_truth_mask.sum())
    metrics['n_important_edges'] = n_important
    
    # Top-K metrics
    if k_values is None:
        k_values = [n_important, n_important * 2, 10, 20]
    
    sorted_indices = np.argsort(edge_scores)[::-1]  # Descending
    
    for k in k_values:
        k = min(k, len(edge_scores))
        top_k_indices = sorted_indices[:k]
        top_k_mask = np.zeros_like(ground_truth_mask)
        top_k_mask[top_k_indices] = 1
        
        # Precision@K: fraction of top-K that are truly important
        precision_at_k = ground_truth_mask[top_k_indices].mean()
        metrics[f'precision@{k}'] = precision_at_k
        
        # Recall@K: fraction of truly important found in top-K
        if n_important > 0:
            recall_at_k = ground_truth_mask[top_k_indices].sum() / n_important
        else:
            recall_at_k = 1.0
        metrics[f'recall@{k}'] = recall_at_k
    
    return metrics


def bootstrap_confidence_interval(
    edge_scores: np.ndarray,
    ground_truth_mask: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """Bootstrap CI for explanation metric."""
    n_samples = len(edge_scores)
    bootstrap_values = []
    
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        sampled_scores = edge_scores[indices]
        sampled_mask = ground_truth_mask[indices]
        
        try:
            value = metric_fn(sampled_mask, sampled_scores)
            bootstrap_values.append(value)
        except ValueError:
            continue
    
    bootstrap_values = np.array(bootstrap_values)
    mean_val = bootstrap_values.mean()
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - alpha / 2))
    
    return mean_val, lower, upper


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_explanation_masks(exp_mat_path: Path) -> Dict[str, np.ndarray]:
    """
    Load ground-truth explanation masks from PowerGraph exp.mat files.
    
    PowerGraph provides explanation masks indicating which edges are
    critical for cascade propagation.
    """
    mat_data = loadmat(str(exp_mat_path))
    
    # Adapt based on actual PowerGraph exp.mat structure
    # This is a placeholder - check actual file format
    masks = {}
    
    if 'exp_mask' in mat_data:
        masks['default'] = mat_data['exp_mask'].flatten().astype(np.int32)
    elif 'explanation' in mat_data:
        masks['default'] = mat_data['explanation'].flatten().astype(np.int32)
    else:
        # Try to find mask-like arrays
        for key, value in mat_data.items():
            if not key.startswith('_') and isinstance(value, np.ndarray):
                if value.ndim <= 2 and np.unique(value).size <= 10:
                    masks[key] = value.flatten().astype(np.int32)
    
    return masks


# ============================================================================
# Evaluation Pipeline
# ============================================================================

class CascadeExplanationEvaluator:
    """Complete evaluation pipeline for cascade explanations."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        importance_method: str = 'gradient'
    ):
        self.model = model
        self.device = device
        self.extractor = EdgeImportanceExtractor(model, device)
        self.importance_method = importance_method
        
    def evaluate_single(
        self,
        data: Data,
        ground_truth_mask: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on a single cascade scenario."""
        # Extract edge importance
        if self.importance_method == 'attention':
            scores = self.extractor.attention_based(data)
        elif self.importance_method == 'gradient':
            scores = self.extractor.gradient_based(data)
        elif self.importance_method == 'embedding':
            scores = self.extractor.embedding_similarity(data)
        elif self.importance_method == 'perturbation':
            scores = self.extractor.perturbation_based(data)
        else:
            raise ValueError(f"Unknown method: {self.importance_method}")
        
        scores = scores.numpy()
        
        # Compute metrics
        metrics = compute_explanation_metrics(scores, ground_truth_mask)
        return metrics
    
    def evaluate_dataset(
        self,
        loader: DataLoader,
        exp_masks: Dict[int, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate across full dataset."""
        all_metrics = []
        
        for i, data in enumerate(loader):
            if i not in exp_masks:
                continue
                
            metrics = self.evaluate_single(data, exp_masks[i])
            all_metrics.append(metrics)
        
        # Aggregate
        agg = {}
        if all_metrics:
            keys = all_metrics[0].keys()
            for key in keys:
                values = [m[key] for m in all_metrics if key in m]
                agg[f'{key}_mean'] = np.mean(values)
                agg[f'{key}_std'] = np.std(values)
        
        return agg
    
    def compare_methods(
        self,
        data: Data,
        ground_truth_mask: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compare all importance extraction methods."""
        methods = ['attention', 'gradient', 'embedding']
        results = {}
        
        original_method = self.importance_method
        for method in methods:
            self.importance_method = method
            try:
                results[method] = self.evaluate_single(data, ground_truth_mask)
            except Exception as e:
                results[method] = {'error': str(e)}
        
        self.importance_method = original_method
        return results


# ============================================================================
# Random Baseline (Sanity Check)
# ============================================================================

def random_baseline_auc(ground_truth_mask: np.ndarray, n_trials: int = 100) -> Dict:
    """Verify random edge scores give ~0.5 AUC (sanity check)."""
    aucs = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_trials):
        random_scores = rng.random(len(ground_truth_mask))
        try:
            auc = roc_auc_score(ground_truth_mask, random_scores)
            aucs.append(auc)
        except ValueError:
            continue
    
    return {
        'random_auc_mean': np.mean(aucs),
        'random_auc_std': np.std(aucs),
        'expected': 0.5
    }


# ============================================================================
# Main Evaluation Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cascade explanation evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory with processed PowerGraph data')
    parser.add_argument('--exp_mask_path', type=str, default='data/raw/exp.mat',
                        help='Path to explanation ground truth')
    parser.add_argument('--method', type=str, default='gradient',
                        choices=['attention', 'gradient', 'embedding', 'perturbation'])
    parser.add_argument('--output_dir', type=str, default='outputs/explanation_eval')
    parser.add_argument('--compare_methods', action='store_true',
                        help='Compare all importance extraction methods')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model_path}")
    # model = load_model(args.model_path)
    # model.to(device)
    
    print(f"Loading explanation ground truth from {args.exp_mask_path}")
    if Path(args.exp_mask_path).exists():
        exp_masks = load_explanation_masks(Path(args.exp_mask_path))
        print(f"Loaded {len(exp_masks)} explanation masks")
    else:
        print(f"WARNING: exp.mat not found at {args.exp_mask_path}")
        exp_masks = {}
    
    print(f"\nRunning evaluation with method: {args.method}")
    
    # Placeholder results (replace with actual evaluation)
    results = {
        'method': args.method,
        'model_path': args.model_path,
        'metrics': {
            'auc_mean': 0.0,
            'auc_std': 0.0,
            'avg_precision_mean': 0.0,
            'precision@K_mean': 0.0,
            'recall@K_mean': 0.0,
        }
    }
    
    # Random baseline sanity check
    if exp_masks:
        sample_mask = list(exp_masks.values())[0]
        random_baseline = random_baseline_auc(sample_mask)
        results['random_baseline'] = random_baseline
        print(f"\nRandom baseline AUC: {random_baseline['random_auc_mean']:.3f} "
              f"± {random_baseline['random_auc_std']:.3f} (expected ~0.5)")
    
    # Save results
    results_path = output_dir / 'explanation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*60)
    print("PUBLICATION CHECKLIST:")
    print("="*60)
    print("☐ Random baseline AUC ≈ 0.5 (sanity check)")
    print("☐ Report AUC with 95% CI via bootstrap")
    print("☐ Report Precision@K where K = n_important_edges")
    print("☐ Compare: scratch vs multi-task vs SSL-pretrained")
    print("☐ Ablate: physics regularization impact on explanation")


if __name__ == '__main__':
    main()
