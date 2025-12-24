#!/usr/bin/env python3
"""
SSL Pretraining Script

Pretrain a physics-guided encoder using self-supervised learning objectives.
Supports both masking-based (node/edge/combined) and contrastive (graphcl/grace/infograph) methods.

The pretrained encoder can then be transferred to downstream tasks
(cascade prediction, PF, OPF) for improved performance, especially
in low-label settings.

Usage:
    # Masking-based SSL
    python scripts/pretrain_ssl.py --grid ieee24 --epochs 50
    python scripts/pretrain_ssl.py --ssl_type combined --epochs 100

    # Contrastive SSL
    python scripts/pretrain_ssl.py --ssl_type graphcl --grid ieee24 --epochs 100
    python scripts/pretrain_ssl.py --ssl_type grace --temperature 0.5 --aug_strength 0.2
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
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models import (
    CombinedSSL,
    MaskedEdgeReconstruction,
    MaskedNodeReconstruction,
    GraphCL,
    GRACE,
    InfoGraph,
    create_augmentation,
    Compose,
)
from src.utils import get_device, set_seed


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train SSL for one epoch."""
    model.train()

    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_samples = 0
    total_masked = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        # Track component losses if available
        if "node_loss" in outputs:
            total_node_loss += outputs["node_loss"].item() * batch.num_graphs
        if "edge_loss" in outputs:
            total_edge_loss += outputs["edge_loss"].item() * batch.num_graphs
        if "num_masked" in outputs:
            total_masked += outputs["num_masked"].item()
        elif "num_node_masked" in outputs:
            total_masked += outputs["num_node_masked"].item()

    metrics = {
        "loss": total_loss / total_samples,
        "avg_masked_per_batch": total_masked / len(loader),
    }

    if total_node_loss > 0:
        metrics["node_loss"] = total_node_loss / total_samples
    if total_edge_loss > 0:
        metrics["edge_loss"] = total_edge_loss / total_samples

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate SSL model."""
    model.eval()

    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_samples = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        total_loss += outputs["loss"].item() * batch.num_graphs
        total_samples += batch.num_graphs

        if "node_loss" in outputs:
            total_node_loss += outputs["node_loss"].item() * batch.num_graphs
        if "edge_loss" in outputs:
            total_edge_loss += outputs["edge_loss"].item() * batch.num_graphs

    metrics = {"loss": total_loss / total_samples}

    if total_node_loss > 0:
        metrics["node_loss"] = total_node_loss / total_samples
    if total_edge_loss > 0:
        metrics["edge_loss"] = total_edge_loss / total_samples

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining")
    parser.add_argument("--grid", type=str, default="ieee24", help="Grid name")
    parser.add_argument(
        "--ssl_type",
        type=str,
        default="combined",
        choices=["node", "edge", "combined", "graphcl", "grace", "infograph"],
        help="SSL objective type: masking (node/edge/combined) or contrastive (graphcl/grace/infograph)",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--use_projection_head",
        action="store_true",
        help="Use projection head (discarded during transfer) to mitigate representation lock-in",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=128,
        help="Output dimension of projection head",
    )

    # Contrastive-specific arguments
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for contrastive loss (NT-Xent)",
    )
    parser.add_argument(
        "--aug1",
        type=str,
        default="edge_drop",
        choices=["edge_drop", "node_mask", "feature_noise", "subgraph",
                 "physics_edge_drop", "physics_node_mask", "identity"],
        help="First augmentation for contrastive learning",
    )
    parser.add_argument(
        "--aug2",
        type=str,
        default="node_mask",
        choices=["edge_drop", "node_mask", "feature_noise", "subgraph",
                 "physics_edge_drop", "physics_node_mask", "identity"],
        help="Second augmentation for contrastive learning",
    )
    parser.add_argument(
        "--aug_strength",
        type=float,
        default=0.2,
        help="Augmentation strength (drop/mask ratio or noise std)",
    )

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proj_suffix = "_proj" if args.use_projection_head else ""
    output_dir = Path(args.output_dir) / f"ssl_{args.ssl_type}{proj_suffix}_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    is_contrastive = args.ssl_type in ["graphcl", "grace", "infograph"]

    print("=" * 60)
    print("SSL PRETRAINING")
    print("=" * 60)
    print(f"Grid: {args.grid}")
    print(f"SSL Type: {args.ssl_type}")
    if is_contrastive:
        print(f"Method: Contrastive Learning")
        print(f"Temperature: {args.temperature}")
        print(f"Augmentations: {args.aug1} + {args.aug2} (strength={args.aug_strength})")
    else:
        print(f"Method: Masking-based")
        print(f"Mask Ratio: {args.mask_ratio}")
        print(f"Projection Head: {args.use_projection_head}")
        if args.use_projection_head:
            print(f"Projection Dim: {args.projection_dim}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load datasets (use all data for SSL - no labels needed)
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

    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val: {len(val_dataset)} graphs")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    sample = train_dataset[0]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)

    if args.ssl_type == "node":
        model = MaskedNodeReconstruction(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            mask_ratio=args.mask_ratio,
        ).to(device)
    elif args.ssl_type == "edge":
        model = MaskedEdgeReconstruction(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            mask_ratio=args.mask_ratio,
        ).to(device)
    elif args.ssl_type == "combined":
        model = CombinedSSL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            node_mask_ratio=args.mask_ratio,
            edge_mask_ratio=args.mask_ratio,
            use_projection_head=args.use_projection_head,
            projection_dim=args.projection_dim,
        ).to(device)
    elif args.ssl_type == "graphcl":
        # Create augmentations for GraphCL
        aug1 = create_augmentation(args.aug1, args.aug_strength)
        aug2 = create_augmentation(args.aug2, args.aug_strength)
        model = GraphCL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            proj_dim=args.projection_dim,
            num_layers=args.num_layers,
            temperature=args.temperature,
            augmentation_1=aug1,
            augmentation_2=aug2,
        ).to(device)
    elif args.ssl_type == "grace":
        # Create augmentations for GRACE
        aug1 = create_augmentation(args.aug1, args.aug_strength)
        aug2 = create_augmentation(args.aug2, args.aug_strength)
        model = GRACE(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            proj_dim=args.projection_dim,
            num_layers=args.num_layers,
            temperature=args.temperature,
            augmentation_1=aug1,
            augmentation_2=aug2,
        ).to(device)
    elif args.ssl_type == "infograph":
        model = InfoGraph(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
    else:
        raise ValueError(f"Unknown ssl_type: {args.ssl_type}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    writer = SummaryWriter(output_dir / "tensorboard")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    print("\nStarting pretraining...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        # Update learning rate
        scheduler.step()

        # Logging
        log_str = f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        if "node_loss" in train_metrics:
            log_str += f" | Node: {train_metrics['node_loss']:.4f}"
        if "edge_loss" in train_metrics:
            log_str += f" | Edge: {train_metrics['edge_loss']:.4f}"
        print(log_str)

        # TensorBoard
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        if "node_loss" in train_metrics:
            writer.add_scalar("Loss/train_node", train_metrics["node_loss"], epoch)
            writer.add_scalar("Loss/val_node", val_metrics.get("node_loss", 0), epoch)
        if "edge_loss" in train_metrics:
            writer.add_scalar("Loss/train_edge", train_metrics["edge_loss"], epoch)
            writer.add_scalar("Loss/val_edge", val_metrics.get("edge_loss", 0), epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.get_encoder_state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )

    print(f"\nBest Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Save final model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.get_encoder_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
            "args": vars(args),
        },
        output_dir / "final_model.pt",
    )

    # Save results
    results = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_metrics["loss"],
        "num_params": num_params,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    writer.close()

    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {output_dir / 'best_model.pt'}")
    print(f"Encoder weights key: 'encoder_state_dict'")
    print("\nTo use for downstream tasks:")
    print("  checkpoint = torch.load('best_model.pt')")
    print("  model.encoder.load_state_dict(checkpoint['encoder_state_dict'])")
    print("=" * 60)


if __name__ == "__main__":
    main()
