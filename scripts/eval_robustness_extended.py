#!/usr/bin/env python3
"""
Extended OOD Robustness Evaluation

Evaluates model robustness under multiple perturbation types:
1. Load scaling (existing)
2. Topology dropout - random edge removal simulating line outages
3. Measurement noise - Gaussian noise on node features simulating sensor errors

Usage:
    python scripts/eval_robustness_extended.py --grid ieee24
    python scripts/eval_robustness_extended.py --grid ieee118 --perturbations all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models import CascadeBaselineModel
from src.utils import get_device, set_seed


def perturb_load_scaling(data, scale_factor: float):
    """Scale load (power injection) features."""
    data = data.clone()
    data.x = data.x.clone()
    data.x[:, 0] *= scale_factor  # P_net
    data.x[:, 1] *= scale_factor  # S_net
    return data


def perturb_topology_dropout(data, dropout_rate: float):
    """
    Randomly remove edges to simulate line outages.

    This tests robustness to N-k contingencies where k edges fail.
    """
    data = data.clone()
    num_edges = data.edge_index.size(1)

    # Generate keep mask
    keep_mask = torch.rand(num_edges) > dropout_rate

    # Ensure we keep at least some edges
    if keep_mask.sum() < 2:
        keep_mask[:2] = True

    # Apply mask to edge_index and edge_attr
    data.edge_index = data.edge_index[:, keep_mask]
    data.edge_attr = data.edge_attr[keep_mask]

    return data


def perturb_measurement_noise(data, noise_std: float):
    """
    Add Gaussian noise to node features simulating sensor/measurement errors.

    This tests robustness to SCADA measurement uncertainty.
    """
    data = data.clone()
    data.x = data.x.clone()

    # Add relative noise (proportional to feature magnitude)
    noise = torch.randn_like(data.x) * noise_std * (data.x.abs() + 0.1)
    data.x = data.x + noise

    return data


def compute_metrics(logits, targets):
    """Compute classification metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (preds == targets).float().mean().item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


@torch.no_grad()
def evaluate_with_perturbation(
    model, loader, device, perturb_fn=None, perturb_param=None, num_trials: int = 5
):
    """
    Evaluate model with perturbations, averaging over multiple trials for stochastic perturbations.
    """
    model.eval()

    if perturb_fn is None:
        # No perturbation - single evaluation
        all_logits, all_targets = [], []
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_logits.append(outputs["logits"].cpu())
            all_targets.append(batch.y.cpu())
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        return compute_metrics(all_logits, all_targets), 0.0

    # Stochastic perturbation - average over trials
    trial_f1s = []
    all_metrics = None

    for trial in range(num_trials):
        all_logits, all_targets = [], []
        for batch in loader:
            # Apply perturbation (stochastic for dropout/noise)
            perturbed_batch = perturb_fn(batch, perturb_param)
            perturbed_batch = perturbed_batch.to(device)
            outputs = model(
                perturbed_batch.x,
                perturbed_batch.edge_index,
                perturbed_batch.edge_attr,
                perturbed_batch.batch,
            )
            all_logits.append(outputs["logits"].cpu())
            all_targets.append(batch.y.cpu())

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_logits, all_targets)
        trial_f1s.append(metrics["f1"])

        if all_metrics is None:
            all_metrics = metrics
        else:
            for k in all_metrics:
                all_metrics[k] += metrics[k]

    # Average metrics
    for k in all_metrics:
        all_metrics[k] /= num_trials

    return all_metrics, np.std(trial_f1s)


def find_model_checkpoints(output_dir: Path, grid: str, seeds: List[int]) -> Dict[str, Dict[int, Path]]:
    """Find model checkpoints for each seed."""
    models = {"ssl": {}, "scratch": {}}

    # Look for multiseed experiment directories
    multiseed_dirs = list(output_dir.glob(f"multiseed_{grid}_*"))
    if multiseed_dirs:
        multiseed_dir = max(multiseed_dirs, key=lambda p: p.stat().st_mtime)
        for seed in seeds:
            ssl_path = multiseed_dir / f"ssl_frac1.0_seed{seed}" / "best_model.pt"
            scratch_path = multiseed_dir / f"scratch_frac1.0_seed{seed}" / "best_model.pt"
            if ssl_path.exists():
                models["ssl"][seed] = ssl_path
            if scratch_path.exists():
                models["scratch"][seed] = scratch_path

    # Fallback to comparison directories
    if not models["ssl"] or not models["scratch"]:
        comparison_dirs = list(output_dir.glob(f"comparison_{grid}_*"))
        if comparison_dirs:
            comparison_dir = max(comparison_dirs, key=lambda p: p.stat().st_mtime)
            ssl_path = comparison_dir / "ssl_frac1.0" / "best_model.pt"
            scratch_path = comparison_dir / "scratch_frac1.0" / "best_model.pt"
            if ssl_path.exists() and 42 in seeds:
                models["ssl"][42] = ssl_path
            if scratch_path.exists() and 42 in seeds:
                models["scratch"][42] = scratch_path

    return models


def run_extended_robustness(args):
    """Run extended robustness evaluation."""
    device = get_device()
    seeds = args.seeds

    print("=" * 70)
    print("EXTENDED OOD ROBUSTNESS EVALUATION")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Seeds: {seeds}")
    print(f"Perturbations: {args.perturbations}")
    print("=" * 70)

    # Load test data
    test_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    sample = test_dataset[0]

    print(f"\nTest set: {len(test_dataset)} samples")
    print(f"Node features: {sample.x.size(-1)}, Edge features: {sample.edge_attr.size(-1)}")

    # Find model checkpoints
    output_dir = Path(args.output_dir)
    model_paths = find_model_checkpoints(output_dir, args.grid, seeds)

    print(f"\nFound models:")
    for model_type, seed_paths in model_paths.items():
        print(f"  {model_type}: seeds {list(seed_paths.keys())}")

    if not model_paths["ssl"] and not model_paths["scratch"]:
        print("ERROR: No model checkpoints found!")
        return None

    # Define perturbation configurations
    perturbation_configs = {
        "load_scaling": {
            "fn": perturb_load_scaling,
            "params": [1.0, 1.1, 1.2, 1.3],
            "param_name": "scale",
        },
        "topology_dropout": {
            "fn": perturb_topology_dropout,
            "params": [0.0, 0.05, 0.10, 0.15],
            "param_name": "dropout_rate",
        },
        "measurement_noise": {
            "fn": perturb_measurement_noise,
            "params": [0.0, 0.01, 0.05, 0.10],
            "param_name": "noise_std",
        },
    }

    # Filter perturbations
    if args.perturbations != "all":
        perturbation_configs = {
            k: v for k, v in perturbation_configs.items()
            if k in args.perturbations.split(",")
        }

    # Results storage
    results = {}

    for perturb_name, config in perturbation_configs.items():
        print(f"\n{'='*70}")
        print(f"PERTURBATION: {perturb_name}")
        print("=" * 70)

        results[perturb_name] = {
            "ssl": {p: [] for p in config["params"]},
            "scratch": {p: [] for p in config["params"]},
        }

        for model_type in ["ssl", "scratch"]:
            seed_paths = model_paths.get(model_type, {})
            if not seed_paths:
                print(f"\n  No {model_type} models found, skipping...")
                continue

            print(f"\n  {model_type.upper()}:")

            for seed, checkpoint_path in seed_paths.items():
                print(f"\n    Seed {seed}:")
                set_seed(seed)

                # Load model
                model = CascadeBaselineModel(
                    node_in_dim=sample.x.size(-1),
                    edge_in_dim=sample.edge_attr.size(-1),
                    hidden_dim=128,
                    num_layers=4,
                ).to(device)

                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    print(f"      WARNING: Unexpected checkpoint format")
                    continue

                model.eval()

                # Evaluate at each perturbation level
                for param in config["params"]:
                    if param == 0.0 or param == 1.0:
                        # Baseline - no perturbation
                        metrics, std = evaluate_with_perturbation(model, test_loader, device)
                    else:
                        metrics, std = evaluate_with_perturbation(
                            model, test_loader, device,
                            config["fn"], param, num_trials=5
                        )

                    results[perturb_name][model_type][param].append(metrics["f1"])
                    print(f"      {config['param_name']}={param}: F1={metrics['f1']:.4f} (±{std:.4f})")

    # Compute and display summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = {}

    for perturb_name, config in perturbation_configs.items():
        print(f"\n{perturb_name}:")
        print(f"{'Param':<12} {'SSL Mean':>10} {'SSL Std':>10} {'Scratch Mean':>12} {'Scratch Std':>12} {'Δ F1':>10}")
        print("-" * 68)

        summary[perturb_name] = {}

        for param in config["params"]:
            ssl_f1s = results[perturb_name]["ssl"].get(param, [])
            scratch_f1s = results[perturb_name]["scratch"].get(param, [])

            ssl_mean = np.mean(ssl_f1s) if ssl_f1s else float('nan')
            ssl_std = np.std(ssl_f1s) if ssl_f1s else float('nan')
            scratch_mean = np.mean(scratch_f1s) if scratch_f1s else float('nan')
            scratch_std = np.std(scratch_f1s) if scratch_f1s else float('nan')

            delta = ssl_mean - scratch_mean if ssl_f1s and scratch_f1s else float('nan')

            summary[perturb_name][str(param)] = {
                "ssl_mean": float(ssl_mean) if not np.isnan(ssl_mean) else None,
                "ssl_std": float(ssl_std) if not np.isnan(ssl_std) else None,
                "ssl_values": ssl_f1s,
                "scratch_mean": float(scratch_mean) if not np.isnan(scratch_mean) else None,
                "scratch_std": float(scratch_std) if not np.isnan(scratch_std) else None,
                "scratch_values": scratch_f1s,
                "delta": float(delta) if not np.isnan(delta) else None,
            }

            print(f"{param:<12} {ssl_mean:>10.4f} {ssl_std:>10.4f} {scratch_mean:>12.4f} {scratch_std:>12.4f} {delta:>+10.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extended OOD Robustness Evaluation")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--perturbations",
        type=str,
        default="all",
        help="Comma-separated list of perturbations: load_scaling,topology_dropout,measurement_noise or 'all'",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"robustness_extended_{args.grid}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = run_extended_robustness(args)

    if summary:
        # Save results
        with open(results_dir / "robustness_extended_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {results_dir}")

    return summary


if __name__ == "__main__":
    main()
