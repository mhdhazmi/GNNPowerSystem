#!/usr/bin/env python3
"""
SSL Pretraining for PF/OPF Tasks

Pretrains encoder on power flow data using masked voltage reconstruction.
This creates a pretrained encoder compatible with PF/OPF task dimensions.

Usage:
    python scripts/pretrain_ssl_pf.py --task pf
    python scripts/pretrain_ssl_pf.py --task opf
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
from tqdm import tqdm

from src.data import PowerGraphDataset
from src.models.encoder import PhysicsGuidedEncoder
from src.utils import get_device, set_seed


class MaskedInjectionSSL(nn.Module):
    """
    SSL model for PF task: masks power injections and learns to reconstruct them.

    This is a physics-meaningful pretext task: learning to predict power injections
    from graph structure teaches the model about power flow relationships.

    Note: Voltage (V) is the PF TARGET and is NOT included in the input features.
    This avoids label leakage - we only mask/reconstruct P_net and S_net.
    """

    def __init__(
        self,
        node_in_dim: int = 2,  # P_net, S_net (V is the target, not input)
        edge_in_dim: int = 2,  # X, rating
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.node_in_dim = node_in_dim

        self.encoder = PhysicsGuidedEncoder(
            node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
        )

        # Reconstruction head for masked node features
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_in_dim),
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(node_in_dim))
        nn.init.normal_(self.mask_token)

    def create_mask(self, x, batch):
        """Create BERT-style mask for node features."""
        num_nodes = x.size(0)
        device = x.device

        # Determine nodes to mask
        num_mask = max(1, int(num_nodes * self.mask_ratio))
        mask_indices = torch.randperm(num_nodes, device=device)[:num_mask]

        # Create masked input
        masked_x = x.clone()
        original_x = x.clone()

        # BERT-style: 80% mask token, 10% random, 10% unchanged
        for idx in mask_indices:
            rand = torch.rand(1).item()
            if rand < 0.8:
                masked_x[idx] = self.mask_token
            elif rand < 0.9:
                masked_x[idx] = torch.randn(self.node_in_dim, device=device)
            # else: keep original (10%)

        return masked_x, mask_indices, original_x

    def forward(self, x, edge_index, edge_attr, batch=None):
        masked_x, mask_indices, original_x = self.create_mask(x, batch)

        # Encode
        node_emb = self.encoder(masked_x, edge_index, edge_attr)

        # Reconstruct
        reconstructed = self.reconstruction_head(node_emb)

        # Loss only on masked positions
        loss = F.mse_loss(reconstructed[mask_indices], original_x[mask_indices])

        return {
            "loss": loss,
            "reconstructed": reconstructed,
            "mask_indices": mask_indices,
            "node_emb": node_emb,
        }

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


class MaskedLineParamSSL(nn.Module):
    """
    SSL model for OPF task: masks line parameters and learns to reconstruct them.

    This learns the relationship between node embeddings and line characteristics.

    Note: Power flows (P_flow, Q_flow) are the OPF TARGET and are NOT included
    in the edge input features. This avoids label leakage - we only mask/reconstruct
    line parameters (X, rating).
    """

    def __init__(
        self,
        node_in_dim: int = 3,  # P_net, S_net, V (all available as inputs)
        edge_in_dim: int = 2,  # X, rating (flows are the target, not input)
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.edge_in_dim = edge_in_dim

        self.encoder = PhysicsGuidedEncoder(
            node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
        )

        # Edge reconstruction head
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_in_dim),
        )

        # Learnable mask token for edges
        self.edge_mask_token = nn.Parameter(torch.zeros(edge_in_dim))
        nn.init.normal_(self.edge_mask_token)

    def create_edge_mask(self, edge_attr, edge_index):
        """Create mask for edge features."""
        num_edges = edge_attr.size(0)
        device = edge_attr.device

        num_mask = max(1, int(num_edges * self.mask_ratio))
        mask_indices = torch.randperm(num_edges, device=device)[:num_mask]

        masked_edge_attr = edge_attr.clone()
        original_edge_attr = edge_attr.clone()

        for idx in mask_indices:
            rand = torch.rand(1).item()
            if rand < 0.8:
                masked_edge_attr[idx] = self.edge_mask_token
            elif rand < 0.9:
                masked_edge_attr[idx] = torch.randn(self.edge_in_dim, device=device)

        return masked_edge_attr, mask_indices, original_edge_attr

    def forward(self, x, edge_index, edge_attr, batch=None):
        masked_edge_attr, mask_indices, original_edge_attr = self.create_edge_mask(
            edge_attr, edge_index
        )

        # Encode with masked edges
        node_emb = self.encoder(x, edge_index, masked_edge_attr)

        # Reconstruct edges from node embeddings
        src, dst = edge_index
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        reconstructed = self.edge_head(edge_emb)

        # Loss on masked edges
        loss = F.mse_loss(reconstructed[mask_indices], original_edge_attr[mask_indices])

        return {
            "loss": loss,
            "reconstructed": reconstructed,
            "mask_indices": mask_indices,
            "node_emb": node_emb,
        }

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


def train_ssl(model, loader, optimizer, device):
    """Train SSL model for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Pretraining", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()

        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_ssl(model, loader, device):
    """Evaluate SSL model."""
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        total_loss += outputs["loss"].item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining for PF/OPF")
    parser.add_argument("--task", type=str, default="pf", choices=["pf", "opf"])
    parser.add_argument("--grid", type=str, default="ieee24")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")

    args = parser.parse_args()

    device = get_device()
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ssl_{args.task}_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"SSL PRETRAINING FOR {args.task.upper()} TASK")
    print("=" * 60)
    print(f"Grid: {args.grid}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load dataset
    train_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task=args.task, split="train"
    )
    val_dataset = PowerGraphDataset(
        root="./data", name=args.grid, task=args.task, split="val"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    sample = train_dataset[0]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)

    print(f"Node features: {node_in_dim}, Edge features: {edge_in_dim}")

    # Create model
    if args.task == "pf":
        model = MaskedInjectionSSL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)
    else:
        model = MaskedLineParamSSL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_ssl(model, train_loader, optimizer, device)
        val_loss = evaluate_ssl(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": model.get_encoder_state_dict(),
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "task": args.task,
                    "node_in_dim": node_in_dim,
                    "edge_in_dim": edge_in_dim,
                },
                output_dir / "best_model.pt",
            )

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    print("\n" + "=" * 60)
    print("PRETRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")

    # Save config
    config = {
        "task": args.task,
        "grid": args.grid,
        "node_in_dim": node_in_dim,
        "edge_in_dim": edge_in_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
