#!/usr/bin/env python3
"""
Physics Consistency and Robustness Evaluation

WP3: Compare physics-guided vs vanilla GNN
WP7: Test robustness under perturbations

Usage:
    python scripts/eval_physics_robustness.py --checkpoint outputs/cascade_ieee24_*/best_model.pt
    python scripts/eval_physics_robustness.py --run_comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.metrics import evaluate_physics_consistency
from src.models import CascadeBaselineModel, cascade_loss
from src.models.encoder import PhysicsGuidedEncoder, SimpleGNNEncoder
from src.utils import get_device, set_seed


# ============================================================================
# Model with swappable encoder
# ============================================================================

class CascadeModel(torch.nn.Module):
    """Cascade model with configurable encoder type."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "physics_guided",
    ):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "physics_guided":
            self.encoder = PhysicsGuidedEncoder(
                node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
            )
        else:
            self.encoder = SimpleGNNEncoder(
                node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
            )

        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        node_emb = self.encoder(x, edge_index, edge_attr)

        # Global mean pooling
        if batch is None:
            graph_emb = node_emb.mean(dim=0, keepdim=True)
        else:
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(node_emb, batch)

        logits = self.classifier(graph_emb).squeeze(-1)
        return {"logits": logits}


# ============================================================================
# Perturbation functions for robustness testing
# ============================================================================

def perturb_load_scaling(data, scale_factor: float):
    """Scale load (power injection) features."""
    data = data.clone()
    # Node features: [P_net, S_net, V]
    data.x = data.x.clone()
    data.x[:, 0] *= scale_factor  # P_net
    data.x[:, 1] *= scale_factor  # S_net
    return data


def perturb_feature_noise(data, noise_std: float):
    """Add Gaussian noise to node features."""
    data = data.clone()
    data.x = data.x.clone()
    noise = torch.randn_like(data.x) * noise_std
    data.x = data.x + noise
    return data


def perturb_edge_drop(data, drop_ratio: float):
    """Randomly drop edges (simulate line outages)."""
    if drop_ratio <= 0:
        return data

    data = data.clone()
    num_edges = data.edge_index.size(1)
    num_keep = int(num_edges * (1 - drop_ratio))

    if num_keep < 2:
        return data

    # Random permutation
    perm = torch.randperm(num_edges)[:num_keep]

    data.edge_index = data.edge_index[:, perm]
    data.edge_attr = data.edge_attr[perm]

    return data


# ============================================================================
# Training and evaluation
# ============================================================================

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


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train model and return best validation F1."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss, _ = cascade_loss(outputs, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                all_logits.append(outputs["logits"].cpu())
                all_targets.append(batch.y.cpu())

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_logits, all_targets)

        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]

        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Val F1 = {metrics['f1']:.4f}")

    return best_val_f1


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


# ============================================================================
# Main experiments
# ============================================================================

def run_architecture_comparison(args):
    """Compare physics-guided vs vanilla GNN."""
    device = get_device()
    set_seed(args.seed)

    print("=" * 70)
    print("ARCHITECTURE COMPARISON: Physics-Guided vs Vanilla GNN")
    print("=" * 70)

    # Load data
    train_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="train"
    )
    val_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="val"
    )
    test_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    sample = train_dataset[0]
    results = {}

    for encoder_type in ["physics_guided", "vanilla"]:
        print(f"\n{'='*70}")
        print(f"Training: {encoder_type.upper()}")
        print("=" * 70)

        model = CascadeModel(
            node_in_dim=sample.x.size(-1),
            edge_in_dim=sample.edge_attr.size(-1),
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            encoder_type=encoder_type,
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        best_val_f1 = train_model(model, train_loader, val_loader, device, epochs=args.epochs)
        test_metrics = evaluate_model(model, test_loader, device)

        results[encoder_type] = {
            "val_f1": best_val_f1,
            "test_f1": test_metrics["f1"],
            "test_accuracy": test_metrics["accuracy"],
            "num_params": num_params,
        }

        print(f"\n  Best Val F1: {best_val_f1:.4f}")
        print(f"  Test F1: {test_metrics['f1']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")

        # Physics consistency metrics
        print("\n  Physics Consistency Metrics:")
        physics_metrics = evaluate_physics_consistency(model, sample.to(device), device)
        for k, v in physics_metrics.items():
            print(f"    {k}: {v:.4f}")
            results[encoder_type][f"physics_{k}"] = v

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Encoder':<20} {'Val F1':>10} {'Test F1':>10} {'Accuracy':>10}")
    print("-" * 50)
    for enc, r in results.items():
        print(f"{enc:<20} {r['val_f1']:>10.4f} {r['test_f1']:>10.4f} {r['test_accuracy']:>10.4f}")

    improvement = results["physics_guided"]["test_f1"] - results["vanilla"]["test_f1"]
    print(f"\nPhysics-guided improvement: {improvement:+.4f} ({improvement/results['vanilla']['test_f1']*100:+.1f}%)")

    return results


def run_robustness_tests(args):
    """Test model robustness under perturbations."""
    device = get_device()
    set_seed(args.seed)

    print("=" * 70)
    print("ROBUSTNESS TESTS")
    print("=" * 70)

    # Load test data
    test_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Find models to compare
    models_to_test = {}

    # First, look for fine-tuned models from comparison experiments
    comparison_dirs = list(Path(args.output_dir).glob(f"comparison_{args.grid}_*"))
    if comparison_dirs:
        comparison_dir = max(comparison_dirs, key=lambda p: p.stat().st_mtime)
        # SSL fine-tuned at 100% labels
        ssl_finetuned = comparison_dir / "ssl_frac1.0" / "best_model.pt"
        if ssl_finetuned.exists():
            models_to_test["ssl_finetuned"] = ssl_finetuned
        # Scratch at 100% labels
        scratch_finetuned = comparison_dir / "scratch_frac1.0" / "best_model.pt"
        if scratch_finetuned.exists():
            models_to_test["scratch"] = scratch_finetuned

    # Fallback: Find scratch trained model from cascade training
    if "scratch" not in models_to_test:
        cascade_dirs = list(Path(args.output_dir).glob(f"cascade_{args.grid}_*"))
        if cascade_dirs:
            cascade_dir = max(cascade_dirs, key=lambda p: p.stat().st_mtime)
            cascade_checkpoint = cascade_dir / "best_model.pt"
            if cascade_checkpoint.exists():
                models_to_test["scratch"] = cascade_checkpoint

    if not models_to_test:
        print("No trained models found. Run training first.")
        return {}

    sample = test_dataset[0]
    results = {}

    # Perturbation configurations
    perturbations = [
        ("none", None, None),
        ("load_scale_1.1", perturb_load_scaling, 1.1),
        ("load_scale_1.2", perturb_load_scaling, 1.2),
        ("load_scale_1.3", perturb_load_scaling, 1.3),
        ("noise_0.05", perturb_feature_noise, 0.05),
        ("noise_0.10", perturb_feature_noise, 0.10),
        ("noise_0.20", perturb_feature_noise, 0.20),
        ("edge_drop_0.05", perturb_edge_drop, 0.05),
        ("edge_drop_0.10", perturb_edge_drop, 0.10),
        ("edge_drop_0.15", perturb_edge_drop, 0.15),
    ]

    for model_name, checkpoint_path in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_name.upper()}")
        print("=" * 70)

        # Load model
        model = CascadeBaselineModel(
            node_in_dim=sample.x.size(-1),
            edge_in_dim=sample.edge_attr.size(-1),
            hidden_dim=128,
            num_layers=4,
        ).to(device)

        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
            except RuntimeError:
                # Checkpoint might be from SSL model - try encoder only
                print("  (Full model load failed, trying encoder-only)")
                if "encoder_state_dict" in checkpoint:
                    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                    print("  (Loaded encoder from SSL, head is random)")
                else:
                    print(f"  WARNING: Could not load checkpoint {checkpoint_path}")
                    continue
        elif "encoder_state_dict" in checkpoint:
            # SSL model - need to load encoder separately
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            print("  (Loaded encoder from SSL, head is random)")

        model.eval()
        results[model_name] = {}

        print(f"\n  {'Perturbation':<20} {'F1':>10} {'Accuracy':>10} {'Drop':>10}")
        print("  " + "-" * 50)

        baseline_f1 = None
        for perturb_name, perturb_fn, perturb_param in perturbations:
            metrics = evaluate_model(model, test_loader, device, perturb_fn, perturb_param)
            results[model_name][perturb_name] = metrics

            if baseline_f1 is None:
                baseline_f1 = metrics["f1"]
                drop = 0.0
            else:
                drop = baseline_f1 - metrics["f1"]

            print(f"  {perturb_name:<20} {metrics['f1']:>10.4f} {metrics['accuracy']:>10.4f} {drop:>+10.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    if len(models_to_test) >= 2:
        print("\nF1 degradation under perturbations:")
        print(f"\n{'Perturbation':<20}", end="")
        for model_name in models_to_test:
            print(f" {model_name:>15}", end="")
        print()
        print("-" * (20 + 16 * len(models_to_test)))

        for perturb_name, _, _ in perturbations:
            print(f"{perturb_name:<20}", end="")
            for model_name in models_to_test:
                if perturb_name in results.get(model_name, {}):
                    f1 = results[model_name][perturb_name]["f1"]
                    print(f" {f1:>15.4f}", end="")
            print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Physics & Robustness Evaluation")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--run_comparison", action="store_true", help="Compare architectures")
    parser.add_argument("--run_robustness", action="store_true", help="Run robustness tests")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(args.output_dir) / f"eval_physics_robustness_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.run_comparison or (not args.run_comparison and not args.run_robustness):
        all_results["architecture_comparison"] = run_architecture_comparison(args)

    if args.run_robustness or (not args.run_comparison and not args.run_robustness):
        all_results["robustness"] = run_robustness_tests(args)

    # Save results
    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
