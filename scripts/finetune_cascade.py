#!/usr/bin/env python3
"""
Cascade Fine-tuning Script with Low-Label Experiments

Fine-tune cascade prediction from SSL-pretrained or scratch initialization.
Supports training with different label fractions to demonstrate SSL benefits.

Usage:
    # Fine-tune from SSL pretrained
    python scripts/finetune_cascade.py --pretrained outputs/ssl_combined_ieee24_*/best_model.pt

    # Train from scratch for comparison
    python scripts/finetune_cascade.py --from_scratch

    # Low-label experiment (10% of training data)
    python scripts/finetune_cascade.py --label_fraction 0.1 --pretrained outputs/ssl_*/best_model.pt

    # Run full comparison
    python scripts/finetune_cascade.py --run_comparison
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models import CascadeBaselineModel, cascade_loss
from src.utils import get_device, set_seed


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute classification metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

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


def train_epoch(model, loader, optimizer, device, pos_weight=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _ = cascade_loss(outputs, batch.y, pos_weight=pos_weight)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(outputs["logits"].detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(model, loader, device, pos_weight=None):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _ = cascade_loss(outputs, batch.y, pos_weight=pos_weight)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(outputs["logits"].cpu())
        all_targets.append(batch.y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def compute_pos_weight(dataset):
    """Compute positive class weight for imbalanced binary classification."""
    pos = sum(1 for d in dataset if d.y.item() == 1)
    neg = len(dataset) - pos
    if pos == 0:
        return None
    return torch.tensor([neg / pos])


def create_subset_dataset(dataset, fraction, seed=42):
    """Create a subset of the dataset with the given fraction."""
    if fraction >= 1.0:
        return dataset

    torch.manual_seed(seed)
    n = len(dataset)
    n_subset = max(1, int(n * fraction))
    indices = torch.randperm(n)[:n_subset].tolist()

    return torch.utils.data.Subset(dataset, indices)


def run_single_experiment(
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
    """Run a single fine-tuning experiment."""
    set_seed(seed)

    # Load datasets
    train_dataset_full = PowerGraphDataset(
        root="./data", name=grid, task="cascade", label_type="binary", split="train"
    )
    val_dataset = PowerGraphDataset(
        root="./data", name=grid, task="cascade", label_type="binary", split="val"
    )
    test_dataset = PowerGraphDataset(
        root="./data", name=grid, task="cascade", label_type="binary", split="test"
    )

    # Create subset for low-label setting
    train_dataset = create_subset_dataset(train_dataset_full, label_fraction, seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Compute pos_weight for class imbalance
    pos_weight = compute_pos_weight(train_dataset)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    # Model
    sample = train_dataset_full[0]
    model = CascadeBaselineModel(
        node_in_dim=sample.x.size(-1),
        edge_in_dim=sample.edge_attr.size(-1),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    ).to(device)

    # Load pretrained weights if provided
    init_type = "scratch"
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, weights_only=False, map_location=device)
        encoder_state = checkpoint["encoder_state_dict"]
        model.encoder.load_state_dict(encoder_state)
        init_type = "ssl_pretrained"
        print(f"  Loaded pretrained encoder from: {pretrained_path}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_f1 = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight)
        val_metrics = evaluate(model, val_loader, device, pos_weight=pos_weight)
        scheduler.step()

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            if output_dir:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_f1": best_val_f1,
                    },
                    output_dir / "best_model.pt",
                )

        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"    Epoch {epoch:3d} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

    # Load best model and test
    if output_dir and (output_dir / "best_model.pt").exists():
        checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, pos_weight=pos_weight)

    return {
        "init_type": init_type,
        "label_fraction": label_fraction,
        "train_samples": len(train_dataset),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
    }


def run_comparison(args):
    """Run full comparison: SSL-pretrained vs scratch at different label fractions."""
    device = get_device()
    label_fractions = [0.1, 0.2, 0.5, 1.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"comparison_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOW-LABEL COMPARISON: SSL-PRETRAINED vs SCRATCH")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Label fractions: {label_fractions}")
    print(f"Device: {device}")
    print("=" * 70)

    # Find pretrained model
    pretrained_path = None
    if args.pretrained:
        pretrained_path = args.pretrained
    else:
        # Find most recent SSL checkpoint
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
        print(f"  Scratch Test F1: {result_scratch['test_f1']:.4f}")

        # SSL-pretrained (if available)
        if pretrained_path:
            print("\n  Training from SSL-PRETRAINED...")
            exp_dir = output_dir / f"ssl_frac{fraction}"
            exp_dir.mkdir(exist_ok=True)

            result_ssl = run_single_experiment(
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
            print(f"  SSL Test F1: {result_ssl['test_f1']:.4f}")

            improvement = result_ssl["test_f1"] - result_scratch["test_f1"]
            if result_scratch["test_f1"] > 0:
                pct = improvement / result_scratch["test_f1"] * 100
                print(f"  Improvement: {improvement:+.4f} ({pct:+.1f}%)")
            else:
                print(f"  Improvement: {improvement:+.4f} (scratch F1=0)")

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Fraction':<10} {'Init':<15} {'Train N':<10} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 55)

    for r in all_results:
        print(
            f"{r['label_fraction']*100:>6.0f}%   "
            f"{r['init_type']:<15} "
            f"{r['train_samples']:<10} "
            f"{r['best_val_f1']:<10.4f} "
            f"{r['test_f1']:<10.4f}"
        )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Cascade Fine-tuning")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained SSL model")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--label_fraction", type=float, default=1.0, help="Fraction of training labels")
    parser.add_argument("--run_comparison", action="store_true", help="Run full comparison")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    if args.run_comparison:
        run_comparison(args)
    else:
        device = get_device()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        init_type = "ssl" if args.pretrained else "scratch"
        output_dir = Path(args.output_dir) / f"finetune_{init_type}_{args.grid}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("CASCADE FINE-TUNING")
        print("=" * 60)
        print(f"Grid: {args.grid}")
        print(f"Label fraction: {args.label_fraction*100:.0f}%")
        print(f"Init: {init_type}")
        print(f"Device: {device}")
        print("=" * 60)

        result = run_single_experiment(
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
        print(f"Best Val F1: {result['best_val_f1']:.4f}")
        print(f"Test F1: {result['test_f1']:.4f}")
        print(f"Test Accuracy: {result['test_accuracy']:.4f}")

        with open(output_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
