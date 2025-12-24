#!/usr/bin/env python3
"""
Multi-Seed GraphMAE vs Physics-SSL Comparison

Runs consistent 5-seed experiments for both GraphMAE baseline and Physics-guided SSL
to ensure Table XV uses the same experimental setup as main results tables.

Usage:
    python scripts/run_graphmae_comparison.py --grid ieee118 --seeds 42,123,456,789,1337
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from src.data import PowerGraphDataset
from src.models import CombinedSSL, GraphMAE
from src.models.encoder import PhysicsGuidedEncoder
from src.models.graphmae import GINEncoder
from src.utils import get_device, set_seed


def pretrain_ssl(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    """Pretrain SSL model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                val_loss += outputs['loss'].item() * batch.num_graphs

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Load best model
    model.load_state_dict(best_state)
    return best_val_loss


def finetune_cascade(encoder, train_loader, val_loader, test_loader, device,
                     hidden_dim=128, epochs=100, lr=1e-3, label_fraction=0.1):
    """Finetune for cascade prediction."""
    from torch_geometric.nn import global_mean_pool

    # Simple classifier head (takes pooled graph embeddings)
    head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    # Freeze encoder initially for 10 epochs, then unfreeze
    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    # Subsample training data based on label_fraction
    train_indices = list(range(len(train_loader.dataset)))
    np.random.shuffle(train_indices)
    num_train = int(len(train_indices) * label_fraction)
    train_subset = torch.utils.data.Subset(train_loader.dataset, train_indices[:num_train])
    train_loader_subset = DataLoader(train_subset, batch_size=64, shuffle=True)

    best_val_f1 = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after 10 epochs
        if epoch == 11:
            for param in encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(head.parameters()),
                lr=lr * 0.1, weight_decay=1e-4
            )

        # Train
        encoder.train()
        head.train()
        for batch in train_loader_subset:
            batch = batch.to(device)
            optimizer.zero_grad()

            node_emb = encoder(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb = global_mean_pool(node_emb, batch.batch)
            logits = head(graph_emb).squeeze(-1)

            # Focal loss for class imbalance
            pos_weight = torch.tensor([10.0], device=device)
            loss = F.binary_cross_entropy_with_logits(logits, batch.y.float(), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

        # Validate
        encoder.eval()
        head.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_emb = encoder(batch.x, batch.edge_index, batch.edge_attr)
                graph_emb = global_mean_pool(node_emb, batch.batch)
                logits = head(graph_emb).squeeze(-1)
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(batch.y.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                'encoder': {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }

    # Load best model and evaluate on test
    encoder.load_state_dict(best_state['encoder'])
    head.load_state_dict(best_state['head'])

    encoder.eval()
    head.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            node_emb = encoder(batch.x, batch.edge_index, batch.edge_attr)
            graph_emb = global_mean_pool(node_emb, batch.batch)
            logits = head(graph_emb).squeeze(-1)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(batch.y.cpu().numpy())

    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_acc = accuracy_score(test_labels, test_preds)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec = recall_score(test_labels, test_preds, zero_division=0)

    return {
        'test_f1': test_f1,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'best_val_f1': best_val_f1,
    }


def run_experiment(grid, ssl_type, seed, label_fraction, device, output_dir):
    """Run a single experiment with given config."""
    set_seed(seed)

    # Load data
    train_dataset = PowerGraphDataset(root="./data", name=grid, task="cascade",
                                       label_type="binary", split="train")
    val_dataset = PowerGraphDataset(root="./data", name=grid, task="cascade",
                                     label_type="binary", split="val")
    test_dataset = PowerGraphDataset(root="./data", name=grid, task="cascade",
                                      label_type="binary", split="test")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    sample = train_dataset[0]
    node_in_dim = sample.x.size(-1)
    edge_in_dim = sample.edge_attr.size(-1)
    hidden_dim = 128

    # Create SSL model
    if ssl_type == 'graphmae':
        ssl_model = GraphMAE(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            mask_ratio=0.15,
            gamma=2.0,
        ).to(device)
    else:  # physics_ssl
        ssl_model = CombinedSSL(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            node_mask_ratio=0.15,
            edge_mask_ratio=0.15,
        ).to(device)

    # Pretrain
    print(f"  Pretraining {ssl_type}...")
    pretrain_ssl(ssl_model, train_loader, val_loader, device, epochs=100)

    # Get encoder for downstream task
    if ssl_type == 'graphmae':
        encoder = GINEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
        ).to(device)
        encoder.load_state_dict(ssl_model.get_encoder_state_dict())
    else:
        encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
        ).to(device)
        encoder.load_state_dict(ssl_model.get_encoder_state_dict())

    # Finetune
    print(f"  Finetuning with {int(label_fraction*100)}% labels...")
    results = finetune_cascade(
        encoder, train_loader, val_loader, test_loader, device,
        hidden_dim=hidden_dim, epochs=100, label_fraction=label_fraction
    )

    results['ssl_type'] = ssl_type
    results['seed'] = seed
    results['label_fraction'] = label_fraction
    results['grid'] = grid

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', type=str, default='ieee118')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,1337')
    parser.add_argument('--label_fractions', type=str, default='0.1,1.0')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    label_fractions = [float(f) for f in args.label_fractions.split(',')]
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"graphmae_comparison_{args.grid}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GRAPHMAE VS PHYSICS-SSL MULTI-SEED COMPARISON")
    print("=" * 60)
    print(f"Grid: {args.grid}")
    print(f"Seeds: {seeds}")
    print(f"Label fractions: {label_fractions}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_results = []

    for label_fraction in label_fractions:
        for ssl_type in ['graphmae', 'physics_ssl']:
            print(f"\n=== {ssl_type.upper()} @ {int(label_fraction*100)}% labels ===")

            for seed in seeds:
                print(f"\nSeed {seed}:")
                result = run_experiment(
                    args.grid, ssl_type, seed, label_fraction, device, output_dir
                )
                all_results.append(result)
                print(f"  Test F1: {result['test_f1']:.4f}")

    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Compute summary statistics
    summary = []
    for label_fraction in label_fractions:
        for ssl_type in ['graphmae', 'physics_ssl']:
            f1_scores = [r['test_f1'] for r in all_results
                        if r['ssl_type'] == ssl_type and r['label_fraction'] == label_fraction]
            summary.append({
                'ssl_type': ssl_type,
                'label_fraction': label_fraction,
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'n_seeds': len(f1_scores),
            })

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summary:
        print(f"{s['ssl_type']:12} @ {int(s['label_fraction']*100):3}% labels: "
              f"F1 = {s['mean_f1']:.4f} ± {s['std_f1']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
