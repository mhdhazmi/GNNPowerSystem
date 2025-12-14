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
from src.metrics import compute_embedding_electrical_consistency
from src.models import CascadeBaselineModel, cascade_loss
from src.utils import get_device, set_seed


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for class imbalance - focuses on hard examples.

    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Ground truth labels (0 or 1)
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        Focal loss value
    """
    probs = torch.sigmoid(logits)
    targets = targets.float().view(-1)
    probs = probs.view(-1)

    # Binary cross entropy
    bce = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')

    # Focal weighting
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * bce
    return loss.mean()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute classification metrics including PR-AUC and confusion matrix.

    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Ground truth labels (0 or 1)
        threshold: Classification threshold (default 0.5)

    Returns:
        Dictionary with all metrics including confusion matrix and PR-AUC
    """
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix as sk_confusion_matrix

    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy().astype(int)
    preds = (probs > threshold).astype(int)

    # Basic metrics
    tp = ((preds == 1) & (targets_np == 1)).sum()
    fp = ((preds == 1) & (targets_np == 0)).sum()
    fn = ((preds == 0) & (targets_np == 1)).sum()
    tn = ((preds == 0) & (targets_np == 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)

    # PR-AUC (better for imbalanced datasets)
    precision_curve, recall_curve, _ = precision_recall_curve(targets_np, probs)
    pr_auc = auc(recall_curve, precision_curve)

    # Confusion matrix as list for JSON serialization
    cm = sk_confusion_matrix(targets_np, preds, labels=[0, 1])

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def train_epoch(model, loader, optimizer, device, pos_weight=None, use_focal_loss=False):
    """Train for one epoch.

    Args:
        model: The cascade model
        loader: DataLoader
        optimizer: Optimizer
        device: Device to use
        pos_weight: Positive class weight for BCE loss
        use_focal_loss: If True, use focal loss instead of BCE
    """
    model.train()
    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if use_focal_loss:
            loss = focal_loss(outputs["logits"], batch.y)
        else:
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
def evaluate(model, loader, device, pos_weight=None, use_focal_loss=False, threshold=0.5):
    """Evaluate model.

    Args:
        model: The cascade model
        loader: DataLoader
        device: Device to use
        pos_weight: Positive class weight for BCE loss
        use_focal_loss: If True, use focal loss instead of BCE
        threshold: Classification threshold
    """
    model.eval()
    total_loss = 0
    all_logits = []
    all_targets = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if use_focal_loss:
            loss = focal_loss(outputs["logits"], batch.y)
        else:
            loss, _ = cascade_loss(outputs, batch.y, pos_weight=pos_weight)

        total_loss += loss.item() * batch.num_graphs
        all_logits.append(outputs["logits"].cpu())
        all_targets.append(batch.y.cpu())

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics = compute_metrics(all_logits, all_targets, threshold=threshold)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, all_logits, all_targets


def tune_threshold(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Find optimal threshold on validation set by maximizing F1.

    Args:
        logits: Model logits
        targets: Ground truth labels

    Returns:
        Optimal threshold
    """
    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = compute_metrics(logits, targets, threshold=threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold


@torch.no_grad()
def compute_cascade_physics_metrics(model, loader, device):
    """Compute physics consistency metrics for cascade model embeddings.

    For cascade prediction, we verify that embeddings respect electrical distance:
    - Nodes connected by low-impedance lines should have similar embeddings
    - This shows the model learns physically meaningful representations

    Args:
        model: Trained cascade model
        loader: DataLoader
        device: Torch device

    Returns:
        Dictionary of embedding physics consistency metrics
    """
    model.eval()
    all_metrics = []

    for batch in tqdm(loader, desc="Computing physics metrics", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Get node embeddings from encoder
        if hasattr(model, "encoder"):
            node_emb = model.encoder(batch.x, batch.edge_index, batch.edge_attr)
        else:
            node_emb = outputs.get("node_emb")
            if node_emb is None:
                continue

        metrics = compute_embedding_electrical_consistency(
            node_emb, batch.edge_index, batch.edge_attr
        )
        all_metrics.append(metrics)

    # Average across batches
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = sum(values) / len(values)

    return avg_metrics


def compute_pos_weight(dataset):
    """Compute positive class weight for imbalanced binary classification."""
    pos = sum(1 for d in dataset if d.y.item() == 1)
    neg = len(dataset) - pos
    if pos == 0:
        return None
    return torch.tensor([neg / pos])


def create_subset_dataset(dataset, fraction, seed=42, stratified=True):
    """Create a subset of the dataset with the given fraction.

    Args:
        dataset: Full dataset
        fraction: Fraction to sample (0.0 to 1.0)
        seed: Random seed for reproducibility
        stratified: If True, use stratified sampling to preserve class ratio

    Returns:
        Subset of the dataset
    """
    if fraction >= 1.0:
        return dataset

    n = len(dataset)
    n_subset = max(1, int(n * fraction))

    if stratified:
        # Stratified sampling to preserve class ratio
        from sklearn.model_selection import train_test_split
        import numpy as np

        # Get labels
        labels = np.array([int(dataset[i].y.item()) for i in range(n)])
        indices = np.arange(n)

        # Check if we have both classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            # Only one class, fall back to random sampling
            torch.manual_seed(seed)
            indices = torch.randperm(n)[:n_subset].tolist()
        else:
            # Stratified split: keep n_subset samples, discard the rest
            try:
                subset_indices, _ = train_test_split(
                    indices,
                    train_size=n_subset,
                    stratify=labels,
                    random_state=seed
                )
                indices = subset_indices.tolist()
            except ValueError:
                # Fallback if stratification fails (e.g., too few samples in minority class)
                print(f"  Warning: Stratified sampling failed, using random sampling")
                torch.manual_seed(seed)
                indices = torch.randperm(n)[:n_subset].tolist()

        # Report class distribution
        subset_labels = labels[indices]
        pos_count = sum(subset_labels == 1)
        neg_count = sum(subset_labels == 0)
        print(f"  Subset class distribution: {neg_count} negative, {pos_count} positive ({pos_count/(pos_count+neg_count)*100:.1f}% positive)")
    else:
        torch.manual_seed(seed)
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
    use_focal_loss: bool = False,
    min_epochs: int = 20,
):
    """Run a single fine-tuning experiment.

    Args:
        grid: Grid name (ieee24, ieee118, etc.)
        label_fraction: Fraction of training labels to use
        pretrained_path: Path to pretrained SSL model
        hidden_dim: Hidden dimension size
        num_layers: Number of GNN layers
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        seed: Random seed
        output_dir: Output directory for checkpoints
        device: Device to use
        use_focal_loss: If True, use focal loss instead of BCE
        min_epochs: Minimum epochs before early stopping can save (burn-in period)
    """
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

    # Compute pos_weight for class imbalance (only used if not using focal loss)
    pos_weight = None
    if not use_focal_loss:
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

    # Training with burn-in period
    # Don't save models during burn-in to avoid best_epoch=0 artifacts
    best_val_f1 = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device,
                                     pos_weight=pos_weight, use_focal_loss=use_focal_loss)
        val_metrics, _, _ = evaluate(model, val_loader, device,
                                      pos_weight=pos_weight, use_focal_loss=use_focal_loss)
        scheduler.step()

        # Only consider saving after burn-in period to avoid degenerate early stopping
        if epoch >= min_epochs and val_metrics["f1"] > best_val_f1:
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
    elif best_epoch == 0:
        # No model saved during training - use final model
        # This can happen if model never improved after burn-in
        best_epoch = epochs
        best_val_f1 = val_metrics["f1"]
        print(f"  Warning: No checkpoint saved during training, using final model")

    # Tune threshold on validation set
    _, val_logits, val_targets = evaluate(model, val_loader, device,
                                           pos_weight=pos_weight, use_focal_loss=use_focal_loss)
    optimal_threshold = tune_threshold(val_logits, val_targets)
    print(f"  Optimal threshold (tuned on val): {optimal_threshold}")

    # Evaluate on test set with tuned threshold
    test_metrics, _, _ = evaluate(model, test_loader, device,
                                   pos_weight=pos_weight, use_focal_loss=use_focal_loss,
                                   threshold=optimal_threshold)

    # Compute physics consistency metrics
    physics_metrics = compute_cascade_physics_metrics(model, test_loader, device)

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
        "test_pr_auc": test_metrics["pr_auc"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "optimal_threshold": optimal_threshold,
        "use_focal_loss": use_focal_loss,
        "physics": physics_metrics,
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
            use_focal_loss=args.focal_loss,
            min_epochs=args.min_epochs,
        )
        all_results.append(result_scratch)
        print(f"  Scratch Test F1: {result_scratch['test_f1']:.4f} | PR-AUC: {result_scratch['test_pr_auc']:.4f}")

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
                use_focal_loss=args.focal_loss,
                min_epochs=args.min_epochs,
            )
            all_results.append(result_ssl)
            print(f"  SSL Test F1: {result_ssl['test_f1']:.4f} | PR-AUC: {result_ssl['test_pr_auc']:.4f}")

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
    print(f"\n{'Fraction':<10} {'Init':<15} {'Train N':<10} {'Val F1':<10} {'Test F1':<10} {'PR-AUC':<10}")
    print("-" * 75)

    for r in all_results:
        print(
            f"{r['label_fraction']*100:>6.0f}%   "
            f"{r['init_type']:<15} "
            f"{r['train_samples']:<10} "
            f"{r['best_val_f1']:<10.4f} "
            f"{r['test_f1']:<10.4f} "
            f"{r['test_pr_auc']:<10.4f}"
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
    output_dir = Path(args.output_dir) / f"multiseed_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MULTI-SEED COMPARISON: SSL-PRETRAINED vs SCRATCH")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Seeds: {seeds}")
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
                use_focal_loss=args.focal_loss,
                min_epochs=args.min_epochs,
            )
            result_scratch["seed"] = seed
            all_results.append(result_scratch)
            aggregated[fraction]["scratch"].append(result_scratch["test_f1"])
            print(f"  Scratch F1: {result_scratch['test_f1']:.4f}")

            # SSL
            if pretrained_path:
                exp_dir = output_dir / f"ssl_frac{fraction}_seed{seed}"
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
                    seed=seed,
                    output_dir=exp_dir,
                    device=device,
                    use_focal_loss=args.focal_loss,
                    min_epochs=args.min_epochs,
                )
                result_ssl["seed"] = seed
                all_results.append(result_ssl)
                aggregated[fraction]["ssl"].append(result_ssl["test_f1"])
                print(f"  SSL F1: {result_ssl['test_f1']:.4f}")

    # Summary with statistics
    print("\n" + "=" * 70)
    print("MULTI-SEED RESULTS SUMMARY (mean ± std)")
    print("=" * 70)
    print(f"\n{'Fraction':<10} {'Scratch F1':<20} {'SSL F1':<20} {'Improvement':<15}")
    print("-" * 65)

    summary_stats = []
    for fraction in label_fractions:
        scratch_vals = aggregated[fraction]["scratch"]
        ssl_vals = aggregated[fraction]["ssl"]

        scratch_mean = np.mean(scratch_vals)
        scratch_std = np.std(scratch_vals)
        ssl_mean = np.mean(ssl_vals) if ssl_vals else 0
        ssl_std = np.std(ssl_vals) if ssl_vals else 0

        improvement = (ssl_mean - scratch_mean) / scratch_mean * 100 if ssl_vals else 0

        print(
            f"{fraction*100:>6.0f}%   "
            f"{scratch_mean:.4f}±{scratch_std:.4f}      "
            f"{ssl_mean:.4f}±{ssl_std:.4f}      "
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
    parser = argparse.ArgumentParser(description="Cascade Fine-tuning")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained SSL model")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    parser.add_argument("--label_fraction", type=float, default=1.0, help="Fraction of training labels")
    parser.add_argument("--run_comparison", action="store_true", help="Run full comparison")
    parser.add_argument("--run_multi_seed", action="store_true", help="Run multi-seed comparison (5 seeds)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Seeds to use for multi-seed experiments")
    parser.add_argument("--focal_loss", action="store_true", help="Use focal loss instead of BCE (better for class imbalance)")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min_epochs", type=int, default=20, help="Minimum epochs before early stopping (burn-in)")
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
            use_focal_loss=args.focal_loss,
            min_epochs=args.min_epochs,
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
