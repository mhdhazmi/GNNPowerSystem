#!/usr/bin/env python3
"""
Power Flow Baseline Training Script

Trains a GNN model for power flow prediction with physics consistency metrics.

Usage:
    python train_pf_baseline.py --config configs/pf_baseline.yaml
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional
import json
from datetime import datetime


# ============================================================================
# Model Architecture
# ============================================================================

class PhysicsGuidedConv(MessagePassing):
    """Physics-guided message passing layer."""
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr='add')
        
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.admittance_scale = nn.Linear(edge_dim, 1)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        y_mag = self.admittance_scale(edge_attr).sigmoid()
        edge_emb = self.lin_edge(edge_attr)
        
        return self.propagate(edge_index, x=x, edge_attr=edge_emb, y_mag=y_mag)
    
    def message(self, x_j, edge_attr, y_mag):
        return y_mag * (x_j + edge_attr)


class PhysicsGuidedEncoder(nn.Module):
    """Multi-layer physics-guided GNN encoder."""
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(PhysicsGuidedConv(hidden_dim, hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual
        
        return x


class PowerFlowHead(nn.Module):
    """Predict voltage magnitude and angle (as sin/cos)."""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.v_mag_head = nn.Linear(hidden_dim // 2, 1)
        self.sin_head = nn.Linear(hidden_dim // 2, 1)
        self.cos_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, node_embeddings):
        h = self.mlp(node_embeddings)
        
        v_mag = self.v_mag_head(h).squeeze(-1)
        sin_theta = self.sin_head(h).squeeze(-1)
        cos_theta = self.cos_head(h).squeeze(-1)
        
        # Normalize to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2 + 1e-8)
        sin_theta = sin_theta / norm
        cos_theta = cos_theta / norm
        
        return {
            'v_mag': v_mag,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta,
        }


class PowerFlowGNN(nn.Module):
    """Complete GNN model for power flow prediction."""
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim, edge_in_dim, hidden_dim, num_layers, dropout
        )
        self.pf_head = PowerFlowHead(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        node_emb = self.encoder(x, edge_index, edge_attr)
        return self.pf_head(node_emb)
    
    def get_embeddings(self, x, edge_index, edge_attr):
        return self.encoder(x, edge_index, edge_attr)


# ============================================================================
# Loss Functions
# ============================================================================

def pf_loss(pred: Dict, target: Dict, lambda_physics: float = 0.0) -> torch.Tensor:
    """Power flow loss with optional physics regularization."""
    
    # Voltage magnitude MSE
    loss_v = F.mse_loss(pred['v_mag'], target['v_mag'])
    
    # Angle loss via sin/cos
    loss_sin = F.mse_loss(pred['sin_theta'], target['sin_theta'])
    loss_cos = F.mse_loss(pred['cos_theta'], target['cos_theta'])
    
    total = loss_v + loss_sin + loss_cos
    
    return total


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Training loop for power flow model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    def train_epoch(self) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            target = {
                'v_mag': batch.y_v_mag,
                'sin_theta': batch.y_v_ang_sin,
                'cos_theta': batch.y_v_ang_cos,
            }
            
            loss = pf_loss(pred, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict]:
        """Validation pass with metrics."""
        self.model.eval()
        total_loss = 0
        
        all_v_mae, all_angle_mae = [], []
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr)
            
            target = {
                'v_mag': batch.y_v_mag,
                'sin_theta': batch.y_v_ang_sin,
                'cos_theta': batch.y_v_ang_cos,
            }
            
            loss = pf_loss(pred, target)
            total_loss += loss.item()
            
            # Metrics
            v_mae = (pred['v_mag'] - target['v_mag']).abs().mean()
            all_v_mae.append(v_mae.item())
            
            # Angle MAE (via atan2)
            pred_angle = torch.atan2(pred['sin_theta'], pred['cos_theta'])
            target_angle = torch.atan2(target['sin_theta'], target['cos_theta'])
            angle_diff = torch.abs(pred_angle - target_angle)
            angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
            all_angle_mae.append(angle_diff.mean().item())
        
        metrics = {
            'v_mae': np.mean(all_v_mae),
            'angle_mae_rad': np.mean(all_angle_mae),
            'angle_mae_deg': np.degrees(np.mean(all_angle_mae)),
        }
        
        return total_loss / len(self.val_loader), metrics
    
    def fit(self, num_epochs: int, patience: int = 15):
        """Full training loop."""
        
        no_improve = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            if self.scheduler:
                self.scheduler.step()
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best.pt')
                no_improve = 0
            else:
                no_improve += 1
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"V_MAE: {val_metrics['v_mae']:.4f} | "
                  f"Angle_MAE: {val_metrics['angle_mae_deg']:.2f}°")
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        self.save_checkpoint('last.pt')
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, self.checkpoint_dir / filename)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PF baseline')
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--grid', type=str, default='ieee24')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data (placeholder - replace with actual dataset)
    # train_dataset = PowerGraphDataset(args.data_dir, args.grid, 'pf', 'train')
    # val_dataset = PowerGraphDataset(args.data_dir, args.grid, 'pf', 'val')
    
    # For testing: create dummy data
    from torch_geometric.data import Data
    
    def create_dummy_data(num_samples=100, num_nodes=24, num_edges=34):
        data_list = []
        for _ in range(num_samples):
            x = torch.randn(num_nodes, 8)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.randn(num_edges, 4)
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y_v_mag=torch.rand(num_nodes),
                y_v_ang_sin=torch.randn(num_nodes),
                y_v_ang_cos=torch.randn(num_nodes),
            )
            data_list.append(data)
        return data_list
    
    train_data = create_dummy_data(500)
    val_data = create_dummy_data(100)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Model
    model = PowerFlowGNN(
        node_in_dim=8,
        edge_in_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Train
    output_dir = Path(args.output_dir) / f'pf_baseline_{args.grid}_{datetime.now():%Y%m%d_%H%M}'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        checkpoint_dir=str(output_dir / 'checkpoints'),
    )
    
    history = trainer.fit(args.epochs)
    
    # Save results
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
