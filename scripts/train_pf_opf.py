#!/usr/bin/env python3
"""
Power Flow (PF) and Optimal Power Flow (OPF) Training Script

Trains models for node-level voltage prediction (PF) and edge-level flow prediction (OPF).
Demonstrates SSL transfer benefits for these tasks.

Usage:
    # Train PF from scratch
    python scripts/train_pf_opf.py --task pf --from_scratch

    # Train PF with SSL pretrained encoder
    python scripts/train_pf_opf.py --task pf --pretrained outputs/ssl_combined_ieee24_*/best_model.pt

    # Run comparison (SSL vs scratch at different label fractions)
    python scripts/train_pf_opf.py --task pf --run_comparison

    # Train OPF
    python scripts/train_pf_opf.py --task opf --run_comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.metrics import (
    compute_pf_physics_residual,
    compute_thermal_violations,
    compute_embedding_electrical_consistency,
)
from src.models.encoder import PhysicsGuidedEncoder
from src.utils import get_device, set_seed


# ============================================================================
# Models for PF and OPF tasks
# ============================================================================

class PFModel(nn.Module):
    """Power Flow model: predicts voltage magnitude at each node."""

    def __init__(
        self,
        node_in_dim: int = 2,  # P_net, S_net
        edge_in_dim: int = 2,  # X, rating
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        node_emb = self.encoder(x, edge_index, edge_attr)
        v_pred = self.head(node_emb).squeeze(-1)
        return {"v_pred": v_pred, "node_emb": node_emb}


class OPFModel(nn.Module):
    """OPF model: predicts edge flows (P_flow, Q_flow) from node features."""

    def __init__(
        self,
        node_in_dim: int = 3,  # P_net, S_net, V
        edge_in_dim: int = 2,  # X, rating
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
        )
        # Edge prediction head: concat src and dst embeddings
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # P_flow, Q_flow
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        node_emb = self.encoder(x, edge_index, edge_attr)

        # Get source and destination embeddings for each edge
        src, dst = edge_index
        src_emb = node_emb[src]
        dst_emb = node_emb[dst]

        # Concatenate and predict
        edge_emb = torch.cat([src_emb, dst_emb], dim=-1)
        flow_pred = self.edge_head(edge_emb)

        return {"flow_pred": flow_pred, "node_emb": node_emb}


# ============================================================================
# Training and evaluation
# ============================================================================

def compute_pf_metrics(v_pred, v_target):
    """Compute metrics for voltage prediction."""
    mse = F.mse_loss(v_pred, v_target).item()
    mae = F.l1_loss(v_pred, v_target).item()
    # R^2 score
    ss_res = ((v_target - v_pred) ** 2).sum()
    ss_tot = ((v_target - v_target.mean()) ** 2).sum()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return {"mse": mse, "mae": mae, "r2": r2.item()}


def compute_opf_metrics(flow_pred, flow_target):
    """Compute metrics for flow prediction."""
    mse = F.mse_loss(flow_pred, flow_target).item()
    mae = F.l1_loss(flow_pred, flow_target).item()
    # Per-component metrics
    p_mae = F.l1_loss(flow_pred[:, 0], flow_target[:, 0]).item()
    q_mae = F.l1_loss(flow_pred[:, 1], flow_target[:, 1]).item()
    return {"mse": mse, "mae": mae, "p_mae": p_mae, "q_mae": q_mae}


@torch.no_grad()
def compute_physics_metrics(model, loader, device, task: str):
    """
    Compute physics consistency metrics for model predictions.

    Args:
        model: Trained model
        loader: DataLoader
        device: Torch device
        task: "pf" or "opf"

    Returns:
        Dictionary of physics consistency metrics
    """
    model.eval()
    all_physics = []

    for batch in tqdm(loader, desc="Computing physics metrics", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if task == "pf":
            # Voltage physics consistency
            v_pred = outputs["v_pred"]
            v_true = batch.y

            # Extract power injection from node features (P_net is index 0)
            p_injection = batch.x[:, 0]

            metrics = compute_pf_physics_residual(
                v_pred, v_true, p_injection, batch.edge_index, batch.edge_attr
            )
            all_physics.append(metrics)

        else:  # OPF
            # Thermal violation metrics
            flow_pred = outputs["flow_pred"]
            flow_true = batch.y

            # Rating is typically edge_attr[:, -1] or computed
            # Use relative loading based on max flow for now
            rating = flow_true.abs().max(dim=1).values + 1e-8

            metrics = compute_thermal_violations(flow_pred, flow_true, rating)
            all_physics.append(metrics)

        # Embedding consistency (for both tasks)
        node_emb = outputs["node_emb"]
        emb_metrics = compute_embedding_electrical_consistency(
            node_emb, batch.edge_index, batch.edge_attr
        )
        all_physics[-1].update({f"emb_{k}": v for k, v in emb_metrics.items()})

    # Average metrics across batches
    avg_metrics = {}
    if all_physics:
        for key in all_physics[0].keys():
            values = [m[key] for m in all_physics if key in m]
            avg_metrics[key] = sum(values) / len(values)

    return avg_metrics


def train_epoch_pf(model, loader, optimizer, device):
    """Train PF model for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(outputs["v_pred"], batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.append(outputs["v_pred"].detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_pf_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_epoch_opf(model, loader, optimizer, device):
    """Train OPF model for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(outputs["flow_pred"], batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_preds.append(outputs["flow_pred"].detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    metrics = compute_opf_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate_pf(model, loader, device):
    """Evaluate PF model."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.append(outputs["v_pred"].cpu())
        all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return compute_pf_metrics(all_preds, all_targets)


@torch.no_grad()
def evaluate_opf(model, loader, device):
    """Evaluate OPF model."""
    model.eval()
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.append(outputs["flow_pred"].cpu())
        all_targets.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return compute_opf_metrics(all_preds, all_targets)


def create_subset_dataset(dataset, fraction, seed=42):
    """Create a subset of the dataset."""
    if fraction >= 1.0:
        return dataset
    torch.manual_seed(seed)
    n = len(dataset)
    n_subset = max(1, int(n * fraction))
    indices = torch.randperm(n)[:n_subset].tolist()
    return torch.utils.data.Subset(dataset, indices)


# ============================================================================
# Main training loop
# ============================================================================

def run_single_experiment(
    task: str,
    grid: str,
    label_fraction: float,
    pretrained_path: str = None,
    hidden_dim: int = 128,
    num_layers: int = 4,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    output_dir: Path = None,
    device: torch.device = None,
):
    """Run a single training experiment."""
    set_seed(seed)

    # Load datasets
    train_dataset_full = PowerGraphDataset(
        root="./data", name=grid, task=task, split="train"
    )
    val_dataset = PowerGraphDataset(
        root="./data", name=grid, task=task, split="val"
    )
    test_dataset = PowerGraphDataset(
        root="./data", name=grid, task=task, split="test"
    )

    train_dataset = create_subset_dataset(train_dataset_full, label_fraction, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get dimensions from sample
    sample = train_dataset_full[0]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)

    # Create model
    if task == "pf":
        model = PFModel(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        train_fn = train_epoch_pf
        eval_fn = evaluate_pf
        metric_key = "mae"  # Primary metric for PF
    else:  # opf
        model = OPFModel(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        train_fn = train_epoch_opf
        eval_fn = evaluate_opf
        metric_key = "mae"  # Primary metric for OPF

    # Load pretrained encoder if provided
    init_type = "scratch"
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location=device)
        if "encoder_state_dict" in checkpoint:
            # Need to handle dimension mismatch for PF task
            # SSL was trained with 3 node features, PF has 2
            try:
                model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                init_type = "ssl_pretrained"
                print(f"  Loaded pretrained encoder from: {pretrained_path}")
            except RuntimeError as e:
                print(f"  WARNING: Could not load pretrained encoder (dimension mismatch): {e}")
                print("  Training from scratch instead.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_metric = float("inf")  # Lower is better for MAE
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_fn(model, train_loader, optimizer, device)
        val_metrics = eval_fn(model, val_loader, device)
        scheduler.step()

        if val_metrics[metric_key] < best_val_metric:
            best_val_metric = val_metrics[metric_key]
            best_epoch = epoch
            if output_dir:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_metric": best_val_metric,
                    },
                    output_dir / "best_model.pt",
                )

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"    Epoch {epoch:3d} | Train MAE: {train_metrics['mae']:.6f} | Val MAE: {val_metrics['mae']:.6f}"
            )

    # Load best model and test
    if output_dir and (output_dir / "best_model.pt").exists():
        checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = eval_fn(model, test_loader, device)

    # Compute physics consistency metrics
    physics_metrics = compute_physics_metrics(model, test_loader, device, task)

    result = {
        "task": task,
        "init_type": init_type,
        "label_fraction": label_fraction,
        "train_samples": len(train_dataset),
        "best_epoch": best_epoch,
        "best_val_mae": best_val_metric,
        "test_mae": test_metrics["mae"],
        "test_mse": test_metrics["mse"],
        "test_r2": test_metrics.get("r2", None),
    }

    # Add physics metrics to result
    result["physics"] = physics_metrics

    return result


def run_comparison(args):
    """Run comparison: SSL vs scratch at different label fractions."""
    device = get_device()
    label_fractions = [0.1, 0.2, 0.5, 1.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.task}_comparison_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{args.task.upper()} TASK: SSL-PRETRAINED vs SCRATCH COMPARISON")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Task: {args.task}")
    print(f"Label fractions: {label_fractions}")
    print(f"Device: {device}")
    print("=" * 70)

    # Find pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        ssl_dirs = list(Path(args.output_dir).glob(f"ssl_*_{args.grid}_*"))
        if ssl_dirs:
            latest = max(ssl_dirs, key=lambda p: p.stat().st_mtime)
            pretrained_path = latest / "best_model.pt"
            if pretrained_path.exists():
                print(f"Found pretrained model: {pretrained_path}")

    if not pretrained_path or not Path(pretrained_path).exists():
        print("WARNING: No pretrained model found. Will only run scratch experiments.")
        pretrained_path = None

    all_results = []

    for fraction in label_fractions:
        print(f"\n{'='*70}")
        print(f"LABEL FRACTION: {fraction*100:.0f}%")
        print("=" * 70)

        # Scratch
        print("\n  Training from SCRATCH...")
        exp_dir = output_dir / f"scratch_frac{fraction}"
        exp_dir.mkdir(exist_ok=True)

        result_scratch = run_single_experiment(
            task=args.task,
            grid=args.grid,
            label_fraction=fraction,
            pretrained_path=None,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            output_dir=exp_dir,
            device=device,
        )
        all_results.append(result_scratch)
        print(f"  Scratch Test MAE: {result_scratch['test_mae']:.6f}")

        # SSL-pretrained (if available and compatible)
        if pretrained_path:
            print("\n  Training from SSL-PRETRAINED...")
            exp_dir = output_dir / f"ssl_frac{fraction}"
            exp_dir.mkdir(exist_ok=True)

            result_ssl = run_single_experiment(
                task=args.task,
                grid=args.grid,
                label_fraction=fraction,
                pretrained_path=str(pretrained_path),
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                output_dir=exp_dir,
                device=device,
            )
            all_results.append(result_ssl)
            print(f"  SSL Test MAE: {result_ssl['test_mae']:.6f}")

            if result_ssl["init_type"] == "ssl_pretrained":
                improvement = result_scratch["test_mae"] - result_ssl["test_mae"]
                pct = improvement / result_scratch["test_mae"] * 100
                print(f"  Improvement: {improvement:+.6f} ({pct:+.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Fraction':<10} {'Init':<15} {'Train N':<10} {'Val MAE':<12} {'Test MAE':<12}")
    print("-" * 60)

    for r in all_results:
        print(
            f"{r['label_fraction']*100:>6.0f}%   "
            f"{r['init_type']:<15} "
            f"{r['train_samples']:<10} "
            f"{r['best_val_mae']:<12.6f} "
            f"{r['test_mae']:<12.6f}"
        )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return all_results


def run_multi_seed_comparison(args):
    """Run comparison across multiple seeds and compute statistics."""
    import numpy as np

    device = get_device()
    label_fractions = [0.1, 0.2, 0.5, 1.0]
    seeds = args.seeds if args.seeds else [42, 123, 456, 789, 1337]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.task}_multiseed_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{args.task.upper()} MULTI-SEED COMPARISON: SSL-PRETRAINED vs SCRATCH")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Task: {args.task}")
    print(f"Seeds: {seeds}")
    print(f"Label fractions: {label_fractions}")
    print(f"Device: {device}")
    print("=" * 70)

    # Find pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        # Look for task-specific SSL model
        ssl_dirs = list(Path(args.output_dir).glob(f"ssl_*_{args.grid}_*"))
        if ssl_dirs:
            latest = max(ssl_dirs, key=lambda p: p.stat().st_mtime)
            pretrained_path = latest / "best_model.pt"
            if pretrained_path.exists():
                print(f"Found pretrained model: {pretrained_path}")

    all_results = []
    aggregated = {}

    for fraction in label_fractions:
        aggregated[fraction] = {"scratch": [], "ssl": []}

        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"FRACTION: {fraction*100:.0f}% | SEED: {seed}")
            print("=" * 70)

            # Scratch
            exp_dir = output_dir / f"scratch_frac{fraction}_seed{seed}"
            exp_dir.mkdir(exist_ok=True)

            result_scratch = run_single_experiment(
                task=args.task,
                grid=args.grid,
                label_fraction=fraction,
                pretrained_path=None,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
                output_dir=exp_dir,
                device=device,
            )
            result_scratch["seed"] = seed
            all_results.append(result_scratch)
            aggregated[fraction]["scratch"].append(result_scratch["test_mae"])
            print(f"  Scratch MAE: {result_scratch['test_mae']:.6f}")

            # SSL (if available)
            if pretrained_path:
                exp_dir = output_dir / f"ssl_frac{fraction}_seed{seed}"
                exp_dir.mkdir(exist_ok=True)

                result_ssl = run_single_experiment(
                    task=args.task,
                    grid=args.grid,
                    label_fraction=fraction,
                    pretrained_path=str(pretrained_path),
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=seed,
                    output_dir=exp_dir,
                    device=device,
                )
                result_ssl["seed"] = seed
                all_results.append(result_ssl)
                aggregated[fraction]["ssl"].append(result_ssl["test_mae"])
                print(f"  SSL MAE: {result_ssl['test_mae']:.6f}")

    # Summary with statistics
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS SUMMARY (mean ± std)")
    print("=" * 70)
    print(f"\n{'Fraction':<10} {'Scratch MAE':<20} {'SSL MAE':<20} {'Improvement':<15}")
    print("-" * 65)

    summary_stats = []
    for fraction in label_fractions:
        scratch_vals = aggregated[fraction]["scratch"]
        ssl_vals = aggregated[fraction]["ssl"]

        scratch_mean = np.mean(scratch_vals)
        scratch_std = np.std(scratch_vals)
        ssl_mean = np.mean(ssl_vals) if ssl_vals else 0
        ssl_std = np.std(ssl_vals) if ssl_vals else 0

        # For MAE, lower is better, so improvement = (scratch - ssl) / scratch * 100
        improvement = (scratch_mean - ssl_mean) / scratch_mean * 100 if ssl_vals else 0

        print(
            f"{fraction*100:>6.0f}%   "
            f"{scratch_mean:.6f}±{scratch_std:.6f}   "
            f"{ssl_mean:.6f}±{ssl_std:.6f}   "
            f"{improvement:+.1f}%"
        )

        summary_stats.append({
            "label_fraction": fraction,
            "scratch_mean": scratch_mean,
            "scratch_std": scratch_std,
            "ssl_mean": ssl_mean,
            "ssl_std": ssl_std,
            "improvement_pct": improvement,
            "n_seeds": len(seeds),
        })

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return all_results, summary_stats


def main():
    parser = argparse.ArgumentParser(description="PF/OPF Training")
    parser.add_argument("--task", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained SSL model")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--label_fraction", type=float, default=1.0)
    parser.add_argument("--run_comparison", action="store_true")
    parser.add_argument("--run_multi_seed", action="store_true", help="Run multi-seed comparison (5 seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to use for multi-seed experiments")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    if args.run_multi_seed:
        run_multi_seed_comparison(args)
    elif args.run_comparison:
        run_comparison(args)
    else:
        device = get_device()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        init_type = "ssl" if args.pretrained else "scratch"
        output_dir = Path(args.output_dir) / f"{args.task}_{init_type}_{args.grid}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"{args.task.upper()} TRAINING")
        print("=" * 60)
        print(f"Grid: {args.grid}")
        print(f"Task: {args.task}")
        print(f"Label fraction: {args.label_fraction*100:.0f}%")
        print(f"Init: {init_type}")
        print(f"Device: {device}")
        print("=" * 60)

        result = run_single_experiment(
            task=args.task,
            grid=args.grid,
            label_fraction=args.label_fraction,
            pretrained_path=args.pretrained if not args.from_scratch else None,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            output_dir=output_dir,
            device=device,
        )

        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Train samples: {result['train_samples']}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Best Val MAE: {result['best_val_mae']:.6f}")
        print(f"Test MAE: {result['test_mae']:.6f}")
        print(f"Test MSE: {result['test_mse']:.6f}")

        # Print physics consistency metrics
        if result.get("physics"):
            print("\n" + "-" * 40)
            print("PHYSICS CONSISTENCY METRICS")
            print("-" * 40)
            for key, value in result["physics"].items():
                print(f"  {key}: {value:.6f}")

        with open(output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
