#!/usr/bin/env python3
"""
Ablation Studies for Physics-Guided GNN

Compares:
1. PhysicsGuided vs Vanilla GNN (no admittance weighting)
2. Edge-aware message passing vs standard GCN
3. SSL pretraining vs no pretraining

Usage:
    python scripts/run_ablations.py --grid ieee24 --task cascade
    python scripts/run_ablations.py --grid ieee24 --task pf
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
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models.encoder import PhysicsGuidedEncoder
from src.utils import get_device, set_seed


# ============================================================================
# Ablation Model Variants
# ============================================================================

class VanillaGNNEncoder(nn.Module):
    """GNN encoder WITHOUT physics-guided admittance weighting."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_in_dim, hidden_dim)

        # Standard message passing without admittance weighting
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),  # node + edge
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_emb = self.edge_proj(edge_attr)

        for layer, norm in zip(self.layers, self.norms):
            src, dst = edge_index
            # Simple aggregation without admittance weighting
            msg = torch.cat([x[src], edge_emb], dim=-1)
            msg = layer(msg)
            # Sum aggregation
            out = torch.zeros_like(x)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            # Residual + norm
            x = norm(x + self.dropout(out))

        return x


class GCNEncoder(nn.Module):
    """Standard GCN encoder (no edge features)."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,  # Unused but kept for interface compatibility
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(F.relu(conv(x, edge_index))))
        return x


class CascadeModel(nn.Module):
    """Cascade prediction model with pluggable encoder."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = global_mean_pool(node_emb, batch)
        logits = self.classifier(graph_emb).squeeze(-1)
        return {"logits": logits, "node_emb": node_emb}


class PFModel(nn.Module):
    """PF model with pluggable encoder."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
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


# ============================================================================
# Training Functions
# ============================================================================

def train_cascade(model, loader, optimizer, device, pos_weight=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.binary_cross_entropy_with_logits(
            outputs["logits"], batch.y.float(), pos_weight=pos_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_cascade(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(outputs["logits"].cpu())
        all_targets.append(batch.y.cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = (torch.sigmoid(logits) > 0.5).float()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"f1": f1, "precision": precision, "recall": recall}


def train_pf(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.mse_loss(outputs["v_pred"], batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_pf(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_preds.append(outputs["v_pred"].cpu())
        all_targets.append(batch.y.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    mae = F.l1_loss(preds, targets).item()
    mse = F.mse_loss(preds, targets).item()
    return {"mae": mae, "mse": mse}


# ============================================================================
# Ablation Runner
# ============================================================================

def run_ablation(
    encoder_type: str,
    task: str,
    grid: str,
    label_fraction: float,
    hidden_dim: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    output_dir: Path,
    device: torch.device,
):
    """Run single ablation experiment."""
    set_seed(seed)

    # Load data
    train_dataset = PowerGraphDataset(root="./data", name=grid, task=task, split="train")
    val_dataset = PowerGraphDataset(root="./data", name=grid, task=task, split="val")
    test_dataset = PowerGraphDataset(root="./data", name=grid, task=task, split="test")

    # Subset if needed
    if label_fraction < 1.0:
        n = len(train_dataset)
        n_subset = max(1, int(n * label_fraction))
        indices = torch.randperm(n)[:n_subset].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get dimensions
    sample = train_dataset[0] if hasattr(train_dataset, '__getitem__') else train_dataset.dataset[train_dataset.indices[0]]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)

    # Create encoder based on type
    if encoder_type == "physics_guided":
        encoder = PhysicsGuidedEncoder(node_in_dim, edge_in_dim, hidden_dim, num_layers)
    elif encoder_type == "vanilla":
        encoder = VanillaGNNEncoder(node_in_dim, edge_in_dim, hidden_dim, num_layers)
    elif encoder_type == "gcn":
        encoder = GCNEncoder(node_in_dim, edge_in_dim, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Create model
    if task == "cascade":
        model = CascadeModel(encoder, hidden_dim).to(device)
        train_fn = train_cascade
        eval_fn = eval_cascade
        metric_key = "f1"
        higher_is_better = True

        # Compute pos_weight
        pos = sum(1 for d in train_dataset for _ in [d] if d.y.item() == 1)
        neg = len(train_dataset) - pos
        pos_weight = torch.tensor([neg / pos]).to(device) if pos > 0 else None
    else:  # pf
        model = PFModel(encoder, hidden_dim).to(device)
        train_fn = train_pf
        eval_fn = eval_pf
        metric_key = "mae"
        higher_is_better = False
        pos_weight = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float("-inf") if higher_is_better else float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        if task == "cascade":
            train_fn(model, train_loader, optimizer, device, pos_weight)
        else:
            train_fn(model, train_loader, optimizer, device)

        val_metrics = eval_fn(model, val_loader, device)
        scheduler.step()

        current = val_metrics[metric_key]
        is_better = current > best_val if higher_is_better else current < best_val

        if is_better:
            best_val = current
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | Val {metric_key}: {current:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    test_metrics = eval_fn(model, test_loader, device)

    return {
        "encoder_type": encoder_type,
        "task": task,
        "label_fraction": label_fraction,
        "best_epoch": best_epoch,
        f"best_val_{metric_key}": best_val,
        f"test_{metric_key}": test_metrics[metric_key],
    }


def run_all_ablations(args):
    """Run all ablation experiments."""
    device = get_device()
    encoder_types = ["physics_guided", "vanilla", "gcn"]
    label_fractions = [0.1, 0.5, 1.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ablations_{args.task}_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"ENCODER ABLATION STUDY: {args.task.upper()}")
    print("=" * 70)
    print(f"Grid: {args.grid}")
    print(f"Encoder types: {encoder_types}")
    print(f"Label fractions: {label_fractions}")
    print("=" * 70)

    all_results = []

    for encoder_type in encoder_types:
        for fraction in label_fractions:
            print(f"\n{'='*60}")
            print(f"ENCODER: {encoder_type} | LABELS: {fraction*100:.0f}%")
            print("=" * 60)

            exp_dir = output_dir / f"{encoder_type}_frac{fraction}"
            exp_dir.mkdir(exist_ok=True)

            result = run_ablation(
                encoder_type=encoder_type,
                task=args.task,
                grid=args.grid,
                label_fraction=fraction,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                output_dir=exp_dir,
                device=device,
            )
            all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    metric = "f1" if args.task == "cascade" else "mae"
    print(f"\n{'Encoder':<20} {'Labels':<10} {'Test {}':<15}".format(metric.upper()))
    print("-" * 50)

    for r in all_results:
        print(f"{r['encoder_type']:<20} {r['label_fraction']*100:>6.0f}%    {r[f'test_{metric}']:.4f}")

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Encoder Ablation Studies")
    parser.add_argument("--task", type=str, default="cascade", choices=["cascade", "pf"])
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()
    run_all_ablations(args)


if __name__ == "__main__":
    main()
