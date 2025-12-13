#!/usr/bin/env python3
"""
Cascade Classification Training Script

Train a GNN to predict cascading failures in power grids.

Usage:
    python scripts/train_cascade.py --config configs/base.yaml
    python scripts/train_cascade.py --grid ieee24 --epochs 50
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
from src.utils import get_device, load_config, set_seed


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute classification metrics."""
    # Binary classification
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    # Accuracy
    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total

    # True positives, false positives, etc.
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    # Precision, Recall, F1
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


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Loss
        loss, _ = cascade_loss(outputs, batch.y, class_weights)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

        # Collect predictions
        all_logits.append(outputs["logits"].detach().cpu())
        all_targets.append(batch.y.detach().cpu())

    # Compute metrics
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor = None,
) -> dict:
    """Evaluate model on a dataset."""
    model.eval()

    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss, _ = cascade_loss(outputs, batch.y, class_weights)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(outputs["logits"].cpu())
        all_targets.append(batch.y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Cascade GNN")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cascade_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CASCADE CLASSIFICATION TRAINING")
    print("=" * 60)
    print(f"Grid: {args.grid}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PowerGraphDataset(
        root="./data",
        name=args.grid,
        task="cascade",
        label_type="binary",
        split="train",
    )
    val_dataset = PowerGraphDataset(
        root="./data",
        name=args.grid,
        task="cascade",
        label_type="binary",
        split="val",
    )
    test_dataset = PowerGraphDataset(
        root="./data",
        name=args.grid,
        task="cascade",
        label_type="binary",
        split="test",
    )

    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val: {len(val_dataset)} graphs")
    print(f"Test: {len(test_dataset)} graphs")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Class weights for imbalanced data
    pos_samples = sum(1 for d in train_dataset if d.y.item() == 1)
    neg_samples = len(train_dataset) - pos_samples
    pos_weight = neg_samples / (pos_samples + 1e-8)
    print(f"\nClass distribution: {neg_samples} negative, {pos_samples} positive")
    print(f"Positive weight: {pos_weight:.2f}")

    # Model
    sample = train_dataset[0]
    model = CascadeBaselineModel(
        node_in_dim=sample.x.size(-1),
        edge_in_dim=sample.edge_attr.size(-1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # TensorBoard
    writer = SummaryWriter(output_dir / "tensorboard")

    # Training loop
    best_val_f1 = 0
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Update learning rate
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # TensorBoard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("F1/train", train_metrics["f1"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": best_val_f1,
                },
                output_dir / "best_model.pt",
            )

    print(f"\nBest Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print("=" * 60)

    # Save results
    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "num_params": num_params,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
