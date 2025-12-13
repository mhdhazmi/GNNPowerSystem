#!/usr/bin/env python3
"""
Self-Supervised Pretraining for Power Grid GNNs

Implements grid-specific SSL tasks:
1. Masked Injection Reconstruction - mask Pd/Qd, reconstruct from topology
2. Masked Edge Reconstruction - mask line features, reconstruct

These tasks force the model to learn Kirchhoff's laws implicitly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import dropout_edge
import json
import argparse
from pathlib import Path
from datetime import datetime


# =============================================================================
# SSL Pretext Tasks
# =============================================================================

class MaskedInjectionSSL(nn.Module):
    """
    Mask bus power injections (Pd/Qd) and reconstruct from neighbors.
    Forces learning of power flow distribution patterns.
    """
    
    def __init__(self, encoder, hidden_dim, injection_dim=2, mask_ratio=0.15):
        """
        Args:
            encoder: GNN encoder module
            hidden_dim: encoder output dimension
            injection_dim: number of injection features to mask (typically 2: Pd, Qd)
            mask_ratio: fraction of buses to mask
        """
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.injection_dim = injection_dim
        
        # Reconstruction head
        self.reconstructor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, injection_dim)
        )
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(injection_dim))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data with x containing [Pd, Qd, ...] in first injection_dim columns
        
        Returns:
            loss: reconstruction loss on masked buses
            metrics: dict with additional info
        """
        x = data.x.clone()
        num_nodes = x.size(0)
        
        # Select buses to mask (avoid masking all of one type)
        num_mask = max(1, int(num_nodes * self.mask_ratio))
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        
        # Store original values for loss computation
        original_injections = x[mask_indices, :self.injection_dim].clone()
        
        # Replace masked values with mask token
        x[mask_indices, :self.injection_dim] = self.mask_token
        
        # Encode with masked inputs
        data_masked = data.clone()
        data_masked.x = x
        embeddings = self.encoder(data_masked)
        
        # Reconstruct masked injections
        pred_injections = self.reconstructor(embeddings[mask_indices])
        
        # L1 loss (more robust to outliers than MSE)
        loss = F.l1_loss(pred_injections, original_injections)
        
        metrics = {
            'num_masked': num_mask,
            'pred_mean': pred_injections.mean().item(),
            'true_mean': original_injections.mean().item(),
        }
        
        return loss, metrics


class MaskedEdgeSSL(nn.Module):
    """
    Mask edge features (line G, B) and reconstruct from endpoint embeddings.
    Forces learning of electrical relationships between buses.
    """
    
    def __init__(self, encoder, hidden_dim, edge_dim=2, mask_ratio=0.15):
        """
        Args:
            encoder: GNN encoder module
            hidden_dim: encoder output dimension
            edge_dim: number of edge features to reconstruct (G, B)
            mask_ratio: fraction of edges to mask
        """
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.edge_dim = edge_dim
        
        # Edge reconstruction from endpoint embeddings
        self.edge_reconstructor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Mask token for edges
        self.edge_mask_token = nn.Parameter(torch.zeros(edge_dim))
        nn.init.normal_(self.edge_mask_token, std=0.02)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data with edge_attr containing [G, B, ...] in first edge_dim columns
        
        Returns:
            loss: reconstruction loss on masked edges
            metrics: dict with additional info
        """
        edge_attr = data.edge_attr.clone()
        num_edges = edge_attr.size(0)
        
        # Select edges to mask
        num_mask = max(1, int(num_edges * self.mask_ratio))
        mask_indices = torch.randperm(num_edges)[:num_mask]
        
        # Store original values
        original_edge_features = edge_attr[mask_indices, :self.edge_dim].clone()
        
        # Replace with mask token
        edge_attr[mask_indices, :self.edge_dim] = self.edge_mask_token
        
        # Encode with masked edge features
        data_masked = data.clone()
        data_masked.edge_attr = edge_attr
        embeddings = self.encoder(data_masked)
        
        # Get endpoint embeddings for masked edges
        src = data.edge_index[0, mask_indices]
        dst = data.edge_index[1, mask_indices]
        edge_embeddings = torch.cat([embeddings[src], embeddings[dst]], dim=-1)
        
        # Reconstruct edge features
        pred_edge_features = self.edge_reconstructor(edge_embeddings)
        
        loss = F.l1_loss(pred_edge_features, original_edge_features)
        
        metrics = {
            'num_masked': num_mask,
            'pred_mean': pred_edge_features.mean().item(),
            'true_mean': original_edge_features.mean().item(),
        }
        
        return loss, metrics


class CombinedSSL(nn.Module):
    """
    Combined node (injection) and edge masking SSL.
    """
    
    def __init__(self, encoder, hidden_dim, injection_dim=2, edge_dim=2,
                 mask_ratio=0.15, node_weight=0.5, edge_weight=0.5):
        super().__init__()
        self.node_ssl = MaskedInjectionSSL(encoder, hidden_dim, injection_dim, mask_ratio)
        self.edge_ssl = MaskedEdgeSSL(encoder, hidden_dim, edge_dim, mask_ratio)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
    
    def forward(self, data):
        node_loss, node_metrics = self.node_ssl(data)
        edge_loss, edge_metrics = self.edge_ssl(data)
        
        total_loss = self.node_weight * node_loss + self.edge_weight * edge_loss
        
        metrics = {
            'node_loss': node_loss.item(),
            'edge_loss': edge_loss.item(),
            'total_loss': total_loss.item(),
            **{f'node_{k}': v for k, v in node_metrics.items()},
            **{f'edge_{k}': v for k, v in edge_metrics.items()},
        }
        
        return total_loss, metrics


# =============================================================================
# Encoder Architecture (shared with downstream tasks)
# =============================================================================

class PhysicsGuidedConv(MessagePassing):
    """Message passing weighted by line admittance."""
    
    def __init__(self, in_channels, out_channels, edge_dim=4):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_msg = nn.Linear(out_channels * 2, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        edge_weight = self.lin_edge(edge_attr)
        msg = torch.cat([x_j, edge_weight], dim=-1)
        return self.lin_msg(msg)


class SSLEncoder(nn.Module):
    """
    Encoder for SSL pretraining. Will be reused for downstream tasks.
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            PhysicsGuidedConv(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x = self.input_proj(data.x)
        
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, data.edge_index, data.edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual connection
        
        return x


# =============================================================================
# Pretraining Loop
# =============================================================================

class SSLPretrainer:
    """
    Self-supervised pretraining manager.
    """
    
    def __init__(self, model, optimizer, device='cuda', log_dir='outputs/ssl'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            loss, metrics = self.model(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                loss, metrics = self.model(batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def fit(self, train_loader, val_loader, epochs=100, patience=20):
        """
        Full pretraining loop with early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_ssl.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Save final model and history
        self.save_checkpoint('final_ssl.pt')
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, filename):
        path = self.log_dir / filename
        # Save only encoder weights (what we transfer to downstream)
        if hasattr(self.model, 'encoder'):
            torch.save(self.model.encoder.state_dict(), path)
        elif hasattr(self.model, 'node_ssl'):  # CombinedSSL
            torch.save(self.model.node_ssl.encoder.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
        print(f"Saved checkpoint to {path}")
    
    def save_history(self):
        path = self.log_dir / 'ssl_history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# Low-Label Fine-tuning Utilities
# =============================================================================

def create_low_label_subset(dataset, fraction, seed=42):
    """
    Create a subset of the training data for low-label experiments.
    
    Args:
        dataset: full training dataset
        fraction: fraction to keep (0.1, 0.2, 0.5, etc.)
        seed: random seed for reproducibility
    
    Returns:
        Subset of dataset
    """
    torch.manual_seed(seed)
    n = len(dataset)
    n_keep = max(1, int(n * fraction))
    indices = torch.randperm(n)[:n_keep].tolist()
    return torch.utils.data.Subset(dataset, indices)


def load_pretrained_encoder(encoder, checkpoint_path, strict=False):
    """
    Load pretrained weights into encoder.
    
    Args:
        encoder: SSLEncoder instance
        checkpoint_path: path to saved weights
        strict: whether to require exact match
    
    Returns:
        encoder with loaded weights
    """
    state_dict = torch.load(checkpoint_path)
    encoder.load_state_dict(state_dict, strict=strict)
    print(f"Loaded pretrained weights from {checkpoint_path}")
    return encoder


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SSL Pretraining for Power Grid GNN')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='outputs/ssl')
    parser.add_argument('--ssl_type', type=str, default='combined',
                        choices=['injection', 'edge', 'combined'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load dataset (placeholder - replace with actual PowerGraph loader)
    print("Loading dataset...")
    # from load_powergraph import PowerGraphDataset
    # dataset = PowerGraphDataset(args.data_dir, task='pf')
    # train_dataset, val_dataset = dataset.get_splits()
    
    # Placeholder dataset for demonstration
    print("NOTE: Using placeholder data. Replace with PowerGraphDataset.")
    from torch_geometric.datasets import FakeDataset
    train_dataset = FakeDataset(num_graphs=1000, avg_num_nodes=50, avg_degree=3,
                                 num_channels=6, edge_dim=4)
    val_dataset = FakeDataset(num_graphs=200, avg_num_nodes=50, avg_degree=3,
                               num_channels=6, edge_dim=4)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Get dimensions from data
    sample = train_dataset[0]
    node_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1) if sample.edge_attr is not None else 4
    
    print(f"Node dim: {node_dim}, Edge dim: {edge_dim}")
    
    # Create encoder and SSL model
    encoder = SSLEncoder(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    if args.ssl_type == 'injection':
        ssl_model = MaskedInjectionSSL(encoder, args.hidden_dim, 
                                       injection_dim=2, mask_ratio=args.mask_ratio)
    elif args.ssl_type == 'edge':
        ssl_model = MaskedEdgeSSL(encoder, args.hidden_dim,
                                  edge_dim=2, mask_ratio=args.mask_ratio)
    else:  # combined
        ssl_model = CombinedSSL(encoder, args.hidden_dim,
                                injection_dim=2, edge_dim=2, mask_ratio=args.mask_ratio)
    
    # Optimizer
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Train
    print(f"\nStarting SSL pretraining ({args.ssl_type})...")
    print(f"Epochs: {args.epochs}, Mask ratio: {args.mask_ratio}")
    
    trainer = SSLPretrainer(ssl_model, optimizer, args.device, args.output_dir)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
    
    print("\nPretraining complete!")
    print(f"Best val loss: {min(h['val_loss'] for h in history):.4f}")
    print(f"Encoder saved to {args.output_dir}/best_ssl.pt")


if __name__ == '__main__':
    main()
