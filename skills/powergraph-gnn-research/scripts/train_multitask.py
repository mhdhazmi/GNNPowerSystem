#!/usr/bin/env python3
"""
Multi-task training: PF + OPF + Cascade with shared encoder.

Usage:
    python scripts/train_multitask.py --config configs/multitask.yaml

Key features:
- Shared physics-guided encoder
- Task-specific heads (PF, OPF, Cascade)
- Dynamic loss weighting (uncertainty weighting or fixed)
- Negative transfer monitoring via per-task gradients
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import softmax

# ============================================================================
# Model Components
# ============================================================================

class PhysicsGuidedConv(MessagePassing):
    """Message passing weighted by line admittance."""
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 2):
        super().__init__(aggr='add')
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.lin_msg = nn.Linear(out_channels * 2, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.lin_node(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        edge_feat = self.lin_edge(edge_attr)  # edge_attr = [G_ij, B_ij]
        msg = torch.cat([x_j, edge_feat], dim=-1)
        return F.relu(self.lin_msg(msg))


class PhysicsGuidedEncoder(nn.Module):
    """Shared encoder for all tasks."""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        self.convs = nn.ModuleList([
            PhysicsGuidedConv(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x = norm(x + x_new)  # Residual connection
        return x


class PowerFlowHead(nn.Module):
    """Node-level predictions: voltage magnitude and angle (sin/cos)."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # V_mag, sin(θ), cos(θ)
        )
        
    def forward(self, node_embeddings):
        out = self.mlp(node_embeddings)
        v_mag = F.softplus(out[:, 0:1])  # Positive voltage
        sin_theta = out[:, 1:2]
        cos_theta = out[:, 2:3]
        # Normalize to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2 + 1e-8)
        sin_theta = sin_theta / norm
        cos_theta = cos_theta / norm
        return v_mag, sin_theta, cos_theta


class OPFHead(nn.Module):
    """Node-level generator setpoints + graph-level cost."""
    
    def __init__(self, hidden_dim: int, num_gen_features: int = 2):
        super().__init__()
        # Per-generator outputs (P_g, Q_g)
        self.gen_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gen_features)
        )
        # Graph-level cost prediction
        self.cost_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, node_embeddings, batch, gen_mask=None):
        # Generator-level predictions
        gen_out = self.gen_mlp(node_embeddings)
        if gen_mask is not None:
            gen_out = gen_out * gen_mask.unsqueeze(-1)
        
        # Graph-level cost via pooling
        graph_emb = global_mean_pool(node_embeddings, batch)
        cost = self.cost_mlp(graph_emb)
        return gen_out, cost


class CascadeHead(nn.Module):
    """Graph-level cascade severity classification with attention pooling."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        # Attention-based pooling
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, node_embeddings, edge_index, batch):
        # Attention scores
        att_scores = self.att_mlp(node_embeddings)
        att_weights = softmax(att_scores, batch, dim=0)
        
        # Weighted sum pooling
        graph_emb = global_add_pool(node_embeddings * att_weights, batch)
        
        # Classification
        logits = self.classifier(graph_emb)
        return logits, att_weights


class MultiTaskGNN(nn.Module):
    """Complete multi-task model."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_cascade_classes: int = 3
    ):
        super().__init__()
        self.encoder = PhysicsGuidedEncoder(node_dim, edge_dim, hidden_dim, num_layers)
        self.pf_head = PowerFlowHead(hidden_dim)
        self.opf_head = OPFHead(hidden_dim)
        self.cascade_head = CascadeHead(hidden_dim, num_cascade_classes)
        
    def forward(self, data: Data, tasks: list = ['pf', 'opf', 'cascade']):
        """Forward pass returning outputs for specified tasks."""
        node_emb = self.encoder(data.x, data.edge_index, data.edge_attr)
        
        outputs = {}
        if 'pf' in tasks:
            outputs['pf'] = self.pf_head(node_emb)
        if 'opf' in tasks:
            gen_mask = getattr(data, 'gen_mask', None)
            outputs['opf'] = self.opf_head(node_emb, data.batch, gen_mask)
        if 'cascade' in tasks:
            outputs['cascade'] = self.cascade_head(node_emb, data.edge_index, data.batch)
            
        outputs['node_embeddings'] = node_emb
        return outputs
    
    def get_encoder_embeddings(self, data: Data):
        """Get node embeddings only (for transfer/analysis)."""
        return self.encoder(data.x, data.edge_index, data.edge_attr)


# ============================================================================
# Loss Functions
# ============================================================================

def pf_loss(pred: Tuple, target: Data, v_weight: float = 1.0, angle_weight: float = 1.0):
    """Power flow loss with sin/cos angle handling."""
    v_pred, sin_pred, cos_pred = pred
    
    # Voltage magnitude loss
    v_loss = F.mse_loss(v_pred.squeeze(), target.v_mag)
    
    # Angle loss via sin/cos (handles wrap-around)
    sin_loss = F.mse_loss(sin_pred.squeeze(), torch.sin(target.theta))
    cos_loss = F.mse_loss(cos_pred.squeeze(), torch.cos(target.theta))
    angle_loss = sin_loss + cos_loss
    
    return v_weight * v_loss + angle_weight * angle_loss


def opf_loss(pred: Tuple, target: Data, gen_weight: float = 1.0, cost_weight: float = 0.1):
    """OPF loss: generator setpoints + total cost."""
    gen_pred, cost_pred = pred
    
    # Generator setpoint loss (masked to actual generators)
    if hasattr(target, 'gen_mask') and target.gen_mask is not None:
        mask = target.gen_mask.bool()
        gen_loss = F.mse_loss(gen_pred[mask], target.gen_setpoint[mask])
    else:
        gen_loss = F.mse_loss(gen_pred, target.gen_setpoint)
    
    # Cost prediction loss
    cost_loss = F.mse_loss(cost_pred.squeeze(), target.total_cost)
    
    return gen_weight * gen_loss + cost_weight * cost_loss


def cascade_loss(pred: Tuple, target: Data, class_weights: Optional[torch.Tensor] = None):
    """Cascade classification loss with optional class weighting."""
    logits, _ = pred
    if class_weights is not None:
        return F.cross_entropy(logits, target.cascade_label, weight=class_weights)
    return F.cross_entropy(logits, target.cascade_label)


class UncertaintyWeighting(nn.Module):
    """Learned task uncertainty weighting (Kendall et al., 2018)."""
    
    def __init__(self, num_tasks: int = 3):
        super().__init__()
        # Log variance parameters (initialized to 0 = equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses with learned uncertainty weighting."""
        total = 0
        for i, (name, loss) in enumerate(losses.items()):
            precision = torch.exp(-self.log_vars[i])
            total += precision * loss + self.log_vars[i]
        return total
    
    def get_weights(self) -> Dict[str, float]:
        """Get effective task weights."""
        weights = torch.exp(-self.log_vars).detach().cpu().numpy()
        return {f'task_{i}': float(w) for i, w in enumerate(weights)}


# ============================================================================
# Training Loop
# ============================================================================

class MultiTaskTrainer:
    def __init__(
        self,
        model: MultiTaskGNN,
        device: torch.device,
        tasks: list = ['pf', 'opf', 'cascade'],
        loss_weighting: str = 'uncertainty',  # 'fixed' or 'uncertainty'
        fixed_weights: Dict[str, float] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        cascade_class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.tasks = tasks
        self.cascade_class_weights = cascade_class_weights
        
        # Loss weighting strategy
        self.loss_weighting = loss_weighting
        if loss_weighting == 'uncertainty':
            self.uncertainty_weighting = UncertaintyWeighting(len(tasks)).to(device)
            params = list(model.parameters()) + list(self.uncertainty_weighting.parameters())
        else:
            self.uncertainty_weighting = None
            self.fixed_weights = fixed_weights or {'pf': 1.0, 'opf': 1.0, 'cascade': 1.0}
            params = model.parameters()
            
        self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        
    def compute_losses(self, outputs: Dict, data: Data) -> Dict[str, torch.Tensor]:
        """Compute per-task losses."""
        losses = {}
        if 'pf' in self.tasks and 'pf' in outputs:
            losses['pf'] = pf_loss(outputs['pf'], data)
        if 'opf' in self.tasks and 'opf' in outputs:
            losses['opf'] = opf_loss(outputs['opf'], data)
        if 'cascade' in self.tasks and 'cascade' in outputs:
            losses['cascade'] = cascade_loss(
                outputs['cascade'], data, self.cascade_class_weights
            )
        return losses
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {task: 0.0 for task in self.tasks}
        epoch_losses['total'] = 0.0
        
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(batch, self.tasks)
            losses = self.compute_losses(outputs, batch)
            
            # Combine losses
            if self.loss_weighting == 'uncertainty':
                total_loss = self.uncertainty_weighting(losses)
            else:
                total_loss = sum(self.fixed_weights.get(k, 1.0) * v for k, v in losses.items())
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            for task, loss in losses.items():
                epoch_losses[task] += loss.item()
            epoch_losses['total'] += total_loss.item()
            
        n_batches = len(loader)
        return {k: v / n_batches for k, v in epoch_losses.items()}
    
    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validation pass."""
        self.model.eval()
        val_losses = {task: 0.0 for task in self.tasks}
        val_losses['total'] = 0.0
        
        # Task-specific metrics
        pf_metrics = {'v_mae': 0.0, 'angle_mae': 0.0}
        cascade_metrics = {'correct': 0, 'total': 0}
        
        for batch in loader:
            batch = batch.to(self.device)
            outputs = self.model(batch, self.tasks)
            losses = self.compute_losses(outputs, batch)
            
            for task, loss in losses.items():
                val_losses[task] += loss.item()
            val_losses['total'] += sum(losses.values()).item()
            
            # PF metrics
            if 'pf' in outputs:
                v_pred, sin_pred, cos_pred = outputs['pf']
                pf_metrics['v_mae'] += (v_pred.squeeze() - batch.v_mag).abs().mean().item()
                angle_pred = torch.atan2(sin_pred, cos_pred).squeeze()
                angle_diff = torch.abs(angle_pred - batch.theta)
                angle_diff = torch.minimum(angle_diff, 2 * math.pi - angle_diff)
                pf_metrics['angle_mae'] += torch.rad2deg(angle_diff).mean().item()
            
            # Cascade accuracy
            if 'cascade' in outputs:
                logits, _ = outputs['cascade']
                preds = logits.argmax(dim=1)
                cascade_metrics['correct'] += (preds == batch.cascade_label).sum().item()
                cascade_metrics['total'] += batch.cascade_label.size(0)
        
        n_batches = len(loader)
        metrics = {k: v / n_batches for k, v in val_losses.items()}
        metrics['v_mae'] = pf_metrics['v_mae'] / n_batches
        metrics['angle_mae_deg'] = pf_metrics['angle_mae'] / n_batches
        if cascade_metrics['total'] > 0:
            metrics['cascade_acc'] = cascade_metrics['correct'] / cascade_metrics['total']
            
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_dir: Optional[Path] = None
    ) -> Dict:
        """Full training loop with early stopping."""
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            train_losses = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            
            history['train'].append(train_losses)
            history['val'].append(val_metrics)
            
            # Log progress
            log_str = f"Epoch {epoch+1:3d} | "
            log_str += f"Train: {train_losses['total']:.4f} | "
            log_str += f"Val: {val_metrics['total']:.4f} | "
            if 'pf' in self.tasks:
                log_str += f"V_MAE: {val_metrics['v_mae']:.4f} | "
            if 'cascade' in self.tasks and 'cascade_acc' in val_metrics:
                log_str += f"Casc_Acc: {val_metrics['cascade_acc']:.3f}"
            print(log_str)
            
            # Early stopping
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'uncertainty_weighting': (
                self.uncertainty_weighting.state_dict() 
                if self.uncertainty_weighting else None
            )
        }, path)
        
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        if ckpt['uncertainty_weighting'] and self.uncertainty_weighting:
            self.uncertainty_weighting.load_state_dict(ckpt['uncertainty_weighting'])


# ============================================================================
# Negative Transfer Diagnostics
# ============================================================================

def compute_gradient_similarity(model: MultiTaskGNN, data: Data, tasks: list) -> Dict:
    """Compute cosine similarity between task gradients (negative transfer diagnostic)."""
    model.train()
    
    task_grads = {}
    for task in tasks:
        model.zero_grad()
        outputs = model(data, [task])
        
        if task == 'pf':
            loss = pf_loss(outputs['pf'], data)
        elif task == 'opf':
            loss = opf_loss(outputs['opf'], data)
        elif task == 'cascade':
            loss = cascade_loss(outputs['cascade'], data)
        else:
            continue
            
        loss.backward()
        
        # Collect encoder gradients
        grads = []
        for param in model.encoder.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())
        task_grads[task] = torch.cat(grads)
    
    # Compute pairwise cosine similarity
    similarities = {}
    task_list = list(task_grads.keys())
    for i, t1 in enumerate(task_list):
        for t2 in task_list[i+1:]:
            g1, g2 = task_grads[t1], task_grads[t2]
            sim = F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()
            similarities[f'{t1}_vs_{t2}'] = sim
    
    return similarities


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--output_dir', type=str, default='outputs/multitask')
    parser.add_argument('--tasks', nargs='+', default=['pf', 'opf', 'cascade'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss_weighting', type=str, default='uncertainty',
                        choices=['fixed', 'uncertainty'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (placeholder - replace with actual PowerGraph loader)
    print(f"Loading data from {args.data_dir}...")
    # train_loader, val_loader, test_loader = load_powergraph_dataloaders(...)
    
    # Placeholder data info
    node_dim = 6  # P_load, Q_load, P_gen, Q_gen, V_setpoint, bus_type
    edge_dim = 2  # G_ij, B_ij
    
    # Initialize model
    model = MultiTaskGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = MultiTaskTrainer(
        model=model,
        device=device,
        tasks=args.tasks,
        loss_weighting=args.loss_weighting,
        lr=args.lr
    )
    
    print(f"\nTraining multi-task model with tasks: {args.tasks}")
    print(f"Loss weighting: {args.loss_weighting}")
    
    # Training would happen here
    # history = trainer.fit(train_loader, val_loader, epochs=args.epochs, checkpoint_dir=output_dir)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
