#!/usr/bin/env python3
"""
SSL Masking Strategy Ablation: Node-only vs Edge-only vs Combined

This script addresses the reviewer concern that edge-parameter reconstruction
may be trivial on fixed-topology grids where line parameters (X, rating) are
constant across samples.

The ablation compares:
1. Node-only masking: Only reconstructs P_net, S_net (varying per sample)
2. Edge-only masking: Only reconstructs X, rating (constant - potentially trivial)
3. Combined masking: Reconstructs both (current approach)

Usage:
    python scripts/run_ssl_masking_ablation.py --grid ieee24
    python scripts/run_ssl_masking_ablation.py --grid ieee24 --seeds 42 123 456
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
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models import CombinedSSL, MaskedEdgeReconstruction, MaskedNodeReconstruction
from src.models import CascadeBaselineModel
from src.utils import get_device, set_seed


def compute_trivial_edge_baseline(dataset):
    """
    Compute MSE for predicting mean edge parameters.

    This establishes a trivial baseline: if the learned edge reconstruction
    does not beat simply predicting the mean, then edge masking is not learning
    anything beyond memorization.
    """
    # Collect all edge attributes
    all_edge_attr = []
    for data in dataset:
        all_edge_attr.append(data.edge_attr)
    all_edge_attr = torch.cat(all_edge_attr, dim=0)

    # For cascade task: edge_attr = [P_flow, Q_flow, X, rating]
    # X and rating are at indices 2 and 3 (if 4 features)
    # For PF/LineFlow: edge_attr = [X, rating] at indices 0 and 1

    num_features = all_edge_attr.size(1)

    if num_features == 4:
        # Cascade task
        x_vals = all_edge_attr[:, 2]
        rating_vals = all_edge_attr[:, 3]

        mean_x = x_vals.mean()
        mean_rating = rating_vals.mean()

        # Trivial MSE: predict mean for all
        trivial_x_mse = ((x_vals - mean_x) ** 2).mean().item()
        trivial_rating_mse = ((rating_vals - mean_rating) ** 2).mean().item()

        # Also compute for P_flow, Q_flow (these should vary more)
        p_flow = all_edge_attr[:, 0]
        q_flow = all_edge_attr[:, 1]
        trivial_p_mse = ((p_flow - p_flow.mean()) ** 2).mean().item()
        trivial_q_mse = ((q_flow - q_flow.mean()) ** 2).mean().item()

        return {
            "trivial_x_mse": trivial_x_mse,
            "trivial_rating_mse": trivial_rating_mse,
            "trivial_p_flow_mse": trivial_p_mse,
            "trivial_q_flow_mse": trivial_q_mse,
            "mean_x": mean_x.item(),
            "mean_rating": mean_rating.item(),
            "x_std": x_vals.std().item(),
            "rating_std": rating_vals.std().item(),
            "p_flow_std": p_flow.std().item(),
            "q_flow_std": q_flow.std().item(),
        }
    else:
        # PF/LineFlow task: edge_attr = [X, rating]
        x_vals = all_edge_attr[:, 0]
        rating_vals = all_edge_attr[:, 1]

        mean_x = x_vals.mean()
        mean_rating = rating_vals.mean()

        trivial_x_mse = ((x_vals - mean_x) ** 2).mean().item()
        trivial_rating_mse = ((rating_vals - mean_rating) ** 2).mean().item()

        return {
            "trivial_x_mse": trivial_x_mse,
            "trivial_rating_mse": trivial_rating_mse,
            "mean_x": mean_x.item(),
            "mean_rating": mean_rating.item(),
            "x_std": x_vals.std().item(),
            "rating_std": rating_vals.std().item(),
        }


def compute_trivial_node_baseline(dataset):
    """Compute MSE for predicting mean node features."""
    all_node_attr = []
    for data in dataset:
        all_node_attr.append(data.x)
    all_node_attr = torch.cat(all_node_attr, dim=0)

    num_features = all_node_attr.size(1)
    results = {}

    feature_names = ["P_net", "S_net", "V"] if num_features >= 3 else ["P_net", "S_net"]

    for i, name in enumerate(feature_names):
        if i < num_features:
            vals = all_node_attr[:, i]
            mean_val = vals.mean()
            trivial_mse = ((vals - mean_val) ** 2).mean().item()
            results[f"trivial_{name}_mse"] = trivial_mse
            results[f"{name}_mean"] = mean_val.item()
            results[f"{name}_std"] = vals.std().item()

    return results


def train_ssl_epoch(model, loader, optimizer, device):
    """Train SSL for one epoch, returning component losses."""
    model.train()

    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_samples += batch.num_graphs

        if "node_loss" in outputs:
            total_node_loss += outputs["node_loss"].item() * batch.num_graphs
        if "edge_loss" in outputs:
            total_edge_loss += outputs["edge_loss"].item() * batch.num_graphs

    return {
        "loss": total_loss / total_samples,
        "node_loss": total_node_loss / total_samples if total_node_loss > 0 else None,
        "edge_loss": total_edge_loss / total_samples if total_edge_loss > 0 else None,
    }


@torch.no_grad()
def eval_ssl(model, loader, device):
    """Evaluate SSL model, returning component losses."""
    model.eval()

    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        total_loss += outputs["loss"].item() * batch.num_graphs
        total_samples += batch.num_graphs

        if "node_loss" in outputs:
            total_node_loss += outputs["node_loss"].item() * batch.num_graphs
        if "edge_loss" in outputs:
            total_edge_loss += outputs["edge_loss"].item() * batch.num_graphs

    return {
        "loss": total_loss / total_samples,
        "node_loss": total_node_loss / total_samples if total_node_loss > 0 else None,
        "edge_loss": total_edge_loss / total_samples if total_edge_loss > 0 else None,
    }


def pretrain_ssl(ssl_type, train_loader, val_loader, node_in_dim, edge_in_dim,
                 device, epochs=50, hidden_dim=128):
    """Pretrain SSL model with specified masking strategy."""

    if ssl_type == "node":
        model = MaskedNodeReconstruction(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            mask_ratio=0.15,
        ).to(device)
    elif ssl_type == "edge":
        model = MaskedEdgeReconstruction(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            mask_ratio=0.15,
        ).to(device)
    else:  # combined
        model = CombinedSSL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            node_mask_ratio=0.15,
            edge_mask_ratio=0.15,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    ssl_losses = {"train": [], "val": [], "node": [], "edge": []}

    for epoch in range(epochs):
        train_metrics = train_ssl_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_ssl(model, val_loader, device)
        scheduler.step()

        ssl_losses["train"].append(train_metrics["loss"])
        ssl_losses["val"].append(val_metrics["loss"])
        if val_metrics["node_loss"] is not None:
            ssl_losses["node"].append(val_metrics["node_loss"])
        if val_metrics["edge_loss"] is not None:
            ssl_losses["edge"].append(val_metrics["edge_loss"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={train_metrics['loss']:.6f}, val={val_metrics['loss']:.6f}")

    model.load_state_dict(best_state)

    return model, best_val_loss, ssl_losses


def finetune_cascade(pretrained_encoder_state, train_loader, val_loader, test_loader,
                     node_in_dim, edge_in_dim, device, label_frac=1.0, epochs=100):
    """Fine-tune on cascade task with pretrained encoder."""

    model = CascadeBaselineModel(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_dim=128,
        num_layers=4,
    ).to(device)

    # Load pretrained encoder weights
    if pretrained_encoder_state is not None:
        # Handle dimension mismatches by loading only compatible weights
        current_state = model.encoder.state_dict()
        filtered_state = {}
        for k, v in pretrained_encoder_state.items():
            if k in current_state and v.shape == current_state[k].shape:
                filtered_state[k] = v
        model.encoder.load_state_dict(filtered_state, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(outputs["logits"], batch.y.float())
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = torch.sigmoid(outputs["logits"])
                val_preds.extend((probs > 0.5).cpu().numpy())
                val_targets.extend(batch.y.cpu().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        tp = ((val_preds == 1) & (val_targets == 1)).sum()
        fp = ((val_preds == 1) & (val_targets == 0)).sum()
        fn = ((val_preds == 0) & (val_targets == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        val_f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = torch.sigmoid(outputs["logits"])
            test_preds.extend((probs > 0.5).cpu().numpy())
            test_targets.extend(batch.y.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    tp = ((test_preds == 1) & (test_targets == 1)).sum()
    fp = ((test_preds == 1) & (test_targets == 0)).sum()
    fn = ((test_preds == 0) & (test_targets == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    test_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return test_f1


def run_ablation(args):
    """Run the full SSL masking ablation study."""
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ssl_masking_ablation_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SSL MASKING STRATEGY ABLATION")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Seeds: {args.seeds}")
    print(f"SSL Types: {args.ssl_types}")
    print(f"Label Fractions: {args.label_fracs}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="train"
    )
    val_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="val"
    )
    test_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task="cascade", split="test"
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    sample = train_dataset[0]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)
    print(f"Node features: {node_in_dim}, Edge features: {edge_in_dim}")

    # Compute trivial baselines
    print("\n" + "=" * 70)
    print("TRIVIAL BASELINES (predict mean)")
    print("=" * 70)

    node_baseline = compute_trivial_node_baseline(train_dataset)
    edge_baseline = compute_trivial_edge_baseline(train_dataset)

    print("\nNode feature statistics:")
    for k, v in node_baseline.items():
        print(f"  {k}: {v:.6f}")

    print("\nEdge feature statistics:")
    for k, v in edge_baseline.items():
        print(f"  {k}: {v:.6f}")

    # Save baselines
    with open(output_dir / "trivial_baselines.json", "w") as f:
        json.dump({"node": node_baseline, "edge": edge_baseline}, f, indent=2)

    # Results storage
    results = {ssl_type: {frac: [] for frac in args.label_fracs} for ssl_type in args.ssl_types}
    ssl_losses_all = {ssl_type: [] for ssl_type in args.ssl_types}

    # Run ablation
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print("=" * 70)
        set_seed(seed)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

        for ssl_type in args.ssl_types:
            print(f"\n--- SSL Type: {ssl_type.upper()} ---")

            # Pretrain
            print(f"  Pretraining {ssl_type}...")
            ssl_model, best_loss, ssl_losses = pretrain_ssl(
                ssl_type, train_loader, val_loader,
                node_in_dim, edge_in_dim, device,
                epochs=args.ssl_epochs
            )
            print(f"  Best SSL validation loss: {best_loss:.6f}")

            ssl_losses_all[ssl_type].append({
                "seed": seed,
                "best_val_loss": best_loss,
                "final_node_loss": ssl_losses["node"][-1] if ssl_losses["node"] else None,
                "final_edge_loss": ssl_losses["edge"][-1] if ssl_losses["edge"] else None,
            })

            # Extract encoder state
            encoder_state = ssl_model.encoder.state_dict()

            # Fine-tune at each label fraction
            for frac in args.label_fracs:
                print(f"  Fine-tuning at {frac*100:.0f}% labels...")

                # Subsample training data
                n_samples = int(len(train_dataset) * frac)
                indices = torch.randperm(len(train_dataset))[:n_samples]
                subset = torch.utils.data.Subset(train_dataset, indices.tolist())
                subset_loader = DataLoader(subset, batch_size=64, shuffle=True)

                test_f1 = finetune_cascade(
                    encoder_state, subset_loader, val_loader, test_loader,
                    node_in_dim, edge_in_dim, device,
                    label_frac=frac, epochs=args.finetune_epochs
                )

                results[ssl_type][frac].append(test_f1)
                print(f"    Test F1: {test_f1:.4f}")

    # Compute summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS")
    print("=" * 70)

    summary = {}

    print(f"\n{'SSL Type':<12}", end="")
    for frac in args.label_fracs:
        print(f"{frac*100:.0f}% Labels".center(18), end="")
    print()
    print("-" * (12 + 18 * len(args.label_fracs)))

    for ssl_type in args.ssl_types:
        summary[ssl_type] = {}
        print(f"{ssl_type:<12}", end="")
        for frac in args.label_fracs:
            f1_values = results[ssl_type][frac]
            mean_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)
            summary[ssl_type][str(frac)] = {
                "mean": mean_f1,
                "std": std_f1,
                "values": f1_values
            }
            print(f"{mean_f1:.3f} +/- {std_f1:.3f}".center(18), end="")
        print()

    # Print SSL loss comparison
    print("\n" + "=" * 70)
    print("SSL PRETRAINING LOSSES (Final)")
    print("=" * 70)

    for ssl_type in args.ssl_types:
        losses = ssl_losses_all[ssl_type]
        val_losses = [l["best_val_loss"] for l in losses]
        node_losses = [l["final_node_loss"] for l in losses if l["final_node_loss"] is not None]
        edge_losses = [l["final_edge_loss"] for l in losses if l["final_edge_loss"] is not None]

        print(f"\n{ssl_type.upper()}:")
        print(f"  Total val loss: {np.mean(val_losses):.6f} +/- {np.std(val_losses):.6f}")
        if node_losses:
            print(f"  Node loss: {np.mean(node_losses):.6f} +/- {np.std(node_losses):.6f}")
        if edge_losses:
            print(f"  Edge loss: {np.mean(edge_losses):.6f} +/- {np.std(edge_losses):.6f}")

    # Save results
    full_results = {
        "config": {
            "grid": args.grid,
            "seeds": args.seeds,
            "ssl_types": args.ssl_types,
            "label_fracs": args.label_fracs,
            "ssl_epochs": args.ssl_epochs,
            "finetune_epochs": args.finetune_epochs,
        },
        "trivial_baselines": {
            "node": node_baseline,
            "edge": edge_baseline,
        },
        "ssl_losses": ssl_losses_all,
        "downstream_f1": summary,
    }

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="SSL Masking Strategy Ablation")
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--ssl_types", type=str, nargs="+", default=["node", "edge", "combined"])
    parser.add_argument("--label_fracs", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--ssl_epochs", type=int, default=50)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
