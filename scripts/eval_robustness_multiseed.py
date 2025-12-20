#!/usr/bin/env python3
"""
Multi-Seed Robustness Evaluation

Runs robustness tests (load scaling 1.0-1.3x) with multiple random seeds
to compute variance estimates, addressing reviewer concern about single-seed
robustness results.

Usage:
    python scripts/eval_robustness_multiseed.py --grid ieee24
    python scripts/eval_robustness_multiseed.py --grid ieee24 --seeds 42 123 456 789 1337
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


def compute_metrics(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (preds == targets).float().mean().item()

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate_model(model, loader, device, perturb_fn=None, perturb_param=None):
    """Evaluate model, optionally with perturbations."""
    model.eval()
    all_logits, all_targets = [], []

    for batch in loader:
        if perturb_fn is not None:
            batch = perturb_fn(batch, perturb_param)
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(outputs["logits"].cpu())
        all_targets.append(batch.y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    return compute_metrics(all_logits, all_targets)


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
            # These are single-seed, so only use seed 42
            ssl_path = comparison_dir / "ssl_frac1.0" / "best_model.pt"
            scratch_path = comparison_dir / "scratch_frac1.0" / "best_model.pt"
            if ssl_path.exists() and 42 in seeds:
                models["ssl"][42] = ssl_path
            if scratch_path.exists() and 42 in seeds:
                models["scratch"][42] = scratch_path

    return models


def run_multiseed_robustness(args):
    """Run robustness evaluation with multiple seeds."""
    device = get_device()
    seeds = args.seeds

    print("=" * 70)
    print("MULTI-SEED ROBUSTNESS EVALUATION")
    print(f"Grid: {args.grid}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    # Load test data
    test_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=64)
    sample = test_dataset[0]

    # Find model checkpoints
    output_dir = Path(args.output_dir)
    model_paths = find_model_checkpoints(output_dir, args.grid, seeds)

    print(f"\nFound models:")
    for model_type, seed_paths in model_paths.items():
        print(f"  {model_type}: {list(seed_paths.keys())}")

    if not model_paths["ssl"] and not model_paths["scratch"]:
        print("ERROR: No model checkpoints found!")
        return None

    # Load scaling factors to test
    load_scales = [1.0, 1.1, 1.2, 1.3]

    # Results storage
    results = {
        "ssl": {scale: [] for scale in load_scales},
        "scratch": {scale: [] for scale in load_scales},
    }

    # Evaluate each model type and seed
    for model_type in ["ssl", "scratch"]:
        seed_paths = model_paths.get(model_type, {})
        if not seed_paths:
            print(f"\nNo {model_type} models found, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_type.upper()}")
        print("=" * 70)

        for seed, checkpoint_path in seed_paths.items():
            print(f"\n  Seed {seed}:")
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
                print(f"    WARNING: Unexpected checkpoint format")
                continue

            model.eval()

            # Evaluate at each load scale
            for scale in load_scales:
                if scale == 1.0:
                    metrics = evaluate_model(model, test_loader, device)
                else:
                    metrics = evaluate_model(
                        model, test_loader, device,
                        perturb_load_scaling, scale
                    )
                results[model_type][scale].append(metrics["f1"])
                print(f"    Load {scale}x: F1 = {metrics['f1']:.4f}")

    # Compute statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    summary = {}

    print(f"\n{'Load Scale':<12} {'SSL Mean':>10} {'SSL Std':>10} {'Scratch Mean':>12} {'Scratch Std':>12} {'Δ F1':>10}")
    print("-" * 68)

    for scale in load_scales:
        ssl_f1s = results["ssl"][scale]
        scratch_f1s = results["scratch"][scale]

        ssl_mean = np.mean(ssl_f1s) if ssl_f1s else float('nan')
        ssl_std = np.std(ssl_f1s) if ssl_f1s else float('nan')
        scratch_mean = np.mean(scratch_f1s) if scratch_f1s else float('nan')
        scratch_std = np.std(scratch_f1s) if scratch_f1s else float('nan')

        delta = ssl_mean - scratch_mean if ssl_f1s and scratch_f1s else float('nan')

        summary[f"load_{scale}x"] = {
            "ssl_mean": ssl_mean,
            "ssl_std": ssl_std,
            "ssl_values": ssl_f1s,
            "scratch_mean": scratch_mean,
            "scratch_std": scratch_std,
            "scratch_values": scratch_f1s,
            "delta": delta,
        }

        print(f"{scale}x{'':<10} {ssl_mean:>10.4f} {ssl_std:>10.4f} {scratch_mean:>12.4f} {scratch_std:>12.4f} {delta:>+10.4f}")

    # Compute relative improvement at 1.3x
    if summary["load_1.3x"]["scratch_mean"] > 0:
        rel_improvement = (summary["load_1.3x"]["delta"] / summary["load_1.3x"]["scratch_mean"]) * 100
        print(f"\nRelative SSL improvement at 1.3x load: {rel_improvement:+.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Robustness Evaluation")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1337])
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"robustness_multiseed_{args.grid}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = run_multiseed_robustness(args)

    if summary:
        # Save results
        with open(results_dir / "robustness_summary.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_summary = {}
            for k, v in summary.items():
                json_summary[k] = {
                    "ssl_mean": float(v["ssl_mean"]) if not np.isnan(v["ssl_mean"]) else None,
                    "ssl_std": float(v["ssl_std"]) if not np.isnan(v["ssl_std"]) else None,
                    "ssl_values": v["ssl_values"],
                    "scratch_mean": float(v["scratch_mean"]) if not np.isnan(v["scratch_mean"]) else None,
                    "scratch_std": float(v["scratch_std"]) if not np.isnan(v["scratch_std"]) else None,
                    "scratch_values": v["scratch_values"],
                    "delta": float(v["delta"]) if not np.isnan(v["delta"]) else None,
                }
            json.dump(json_summary, f, indent=2)

        print(f"\nResults saved to: {results_dir}")

    return summary


if __name__ == "__main__":
    main()
