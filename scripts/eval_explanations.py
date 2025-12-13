#!/usr/bin/env python3
"""
Explanation Evaluation Script

Evaluates model explanations (edge importance) against ground-truth masks
from the PowerGraph exp.mat files.

Metrics:
- AUC-ROC: How well importance ranks match ground truth
- Precision@K: Precision of top-K predicted edges
- Recall@K: Recall of ground truth edges in top-K predictions
- Fidelity+: Performance drop when removing important edges
- Fidelity-: Performance drop when keeping only important edges

Usage:
    python scripts/eval_explanations.py --checkpoint outputs/cascade_ieee24_*/best_model.pt
    python scripts/eval_explanations.py --grid ieee24 --method gradient
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models import CascadeBaselineModel
from src.utils import get_device


def compute_explanation_metrics(
    importance: np.ndarray,
    ground_truth: np.ndarray,
    k_values: list = [5, 10, 20],
) -> dict:
    """
    Compute explanation quality metrics.

    Args:
        importance: Predicted edge importance scores [num_edges]
        ground_truth: Binary ground truth mask [num_edges]
        k_values: Values of K for Precision@K and Recall@K

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Skip if no positive edges in ground truth
    if ground_truth.sum() == 0:
        return {"valid": False, "reason": "no_positive_edges"}

    # AUC-ROC
    try:
        metrics["auc_roc"] = roc_auc_score(ground_truth, importance)
    except ValueError:
        metrics["auc_roc"] = 0.5  # Random baseline

    # Precision-Recall curve and AUC-PR
    precision, recall, _ = precision_recall_curve(ground_truth, importance)
    metrics["auc_pr"] = auc(recall, precision)

    # Sort edges by importance (descending)
    sorted_indices = np.argsort(importance)[::-1]

    # Precision@K and Recall@K
    num_positive = ground_truth.sum()

    for k in k_values:
        if k <= len(sorted_indices):
            top_k = sorted_indices[:k]
            hits = ground_truth[top_k].sum()

            metrics[f"precision@{k}"] = hits / k
            metrics[f"recall@{k}"] = hits / num_positive if num_positive > 0 else 0.0

    # Hit@K (at least one correct in top-K)
    for k in k_values:
        if k <= len(sorted_indices):
            top_k = sorted_indices[:k]
            metrics[f"hit@{k}"] = float(ground_truth[top_k].sum() > 0)

    metrics["valid"] = True
    metrics["num_edges"] = len(importance)
    metrics["num_positive"] = int(num_positive)

    return metrics


def evaluate_explanations(
    model: CascadeBaselineModel,
    dataset: PowerGraphDataset,
    method: str = "gradient",
    device: torch.device = None,
) -> dict:
    """
    Evaluate model explanations on a dataset.

    Args:
        model: Trained cascade model
        dataset: PowerGraph dataset with ground-truth edge masks
        method: Explanation method (gradient, attention, integrated_gradients)
        device: Torch device

    Returns:
        Dictionary with aggregate metrics
    """
    model.eval()

    all_metrics = []

    for data in tqdm(dataset, desc=f"Evaluating ({method})"):
        data = data.to(device)

        # Skip if no ground truth explanation
        if not hasattr(data, "edge_mask") or data.edge_mask is None:
            continue

        ground_truth = data.edge_mask.cpu().numpy()

        # Skip if no positive edges
        if ground_truth.sum() == 0:
            continue

        # Get edge importance
        if method == "gradient":
            importance = model.get_edge_importance_gradient(
                data.x, data.edge_index, data.edge_attr, batch=None
            )
        elif method == "attention":
            importance = model.get_edge_importance_attention(
                data.x, data.edge_index, data.edge_attr, batch=None
            )
        elif method == "integrated_gradients":
            importance = model.get_edge_importance_integrated_gradients(
                data.x, data.edge_index, data.edge_attr, batch=None, steps=10
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        importance = importance.cpu().numpy()

        # Compute metrics
        metrics = compute_explanation_metrics(importance, ground_truth)
        if metrics.get("valid", False):
            all_metrics.append(metrics)

    # Aggregate metrics
    if not all_metrics:
        return {"error": "No valid samples with ground truth explanations"}

    aggregate = {
        "method": method,
        "num_samples": len(all_metrics),
    }

    # Average each metric
    metric_keys = [k for k in all_metrics[0].keys() if k not in ["valid", "reason"]]
    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))

    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Explanations")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--grid", type=str, default="ieee24", help="Grid name")
    parser.add_argument("--split", type=str, default="test", help="Data split")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["gradient", "attention", "integrated_gradients", "all"],
        help="Explanation method"
    )
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()

    device = get_device()

    print("=" * 60)
    print("EXPLANATION EVALUATION")
    print("=" * 60)
    print(f"Grid: {args.grid}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = PowerGraphDataset(
        root="./data",
        name=args.grid,
        task="cascade",
        label_type="binary",
        split=args.split,
    )
    print(f"Loaded {len(dataset)} samples")

    # Count samples with explanations
    samples_with_exp = sum(1 for d in dataset if hasattr(d, "edge_mask") and d.edge_mask.sum() > 0)
    print(f"Samples with ground-truth explanations: {samples_with_exp}")

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Find most recent checkpoint
        output_dir = Path(args.output_dir)
        candidates = list(output_dir.glob(f"cascade_{args.grid}_*/best_model.pt"))
        if not candidates:
            print("No checkpoint found. Run training first.")
            return
        checkpoint_path = max(candidates, key=lambda p: p.parent.stat().st_mtime)

    print(f"\nLoading model from: {checkpoint_path}")

    # Load model
    sample = dataset[0]
    model = CascadeBaselineModel(
        node_in_dim=sample.x.size(-1),
        edge_in_dim=sample.edge_attr.size(-1),
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Evaluate explanations
    methods = ["gradient", "attention", "integrated_gradients"] if args.method == "all" else [args.method]

    all_results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method.upper()}")
        print("=" * 60)

        results = evaluate_explanations(model, dataset, method, device)
        all_results[method] = results

        # Print results
        print(f"\nResults ({method}):")
        print(f"  Samples evaluated: {results.get('num_samples', 0)}")

        if "auc_roc_mean" in results:
            print(f"\n  AUC-ROC:  {results['auc_roc_mean']:.4f} ± {results['auc_roc_std']:.4f}")
            print(f"  AUC-PR:   {results['auc_pr_mean']:.4f} ± {results['auc_pr_std']:.4f}")

            for k in [5, 10, 20]:
                p_key = f"precision@{k}_mean"
                r_key = f"recall@{k}_mean"
                h_key = f"hit@{k}_mean"
                if p_key in results:
                    print(f"\n  @{k}:")
                    print(f"    Precision: {results[p_key]:.4f}")
                    print(f"    Recall:    {results[r_key]:.4f}")
                    print(f"    Hit Rate:  {results[h_key]:.4f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    print(f"\n{'Method':<25} {'AUC-ROC':>10} {'AUC-PR':>10} {'P@10':>10} {'R@10':>10}")
    print("-" * 65)

    for method, results in all_results.items():
        if "auc_roc_mean" in results:
            print(
                f"{method:<25} "
                f"{results['auc_roc_mean']:>10.4f} "
                f"{results['auc_pr_mean']:>10.4f} "
                f"{results.get('precision@10_mean', 0):>10.4f} "
                f"{results.get('recall@10_mean', 0):>10.4f}"
            )

    # Save results
    results_path = checkpoint_path.parent / "explanation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
