# Self-Supervised Pretraining Reference

## Overview

Grid-specific SSL objectives that capture power system physics:

1. **Masked Injection Reconstruction** - Predict hidden bus loads/generation
2. **Masked Edge Feature Reconstruction** - Predict hidden line parameters
3. **Contrastive Learning** - Learn topology-invariant representations

## Why Grid-Specific SSL?

Generic graph SSL (like GraphMAE) works, but grid-specific tasks are more effective because:
- They force the model to learn power flow redistribution (Kirchhoff's laws)
- Masked injections require understanding how power balances across the network
- Better sample efficiency and transfer to downstream tasks

## Masked Injection Reconstruction

### Concept

Randomly mask bus injections (P_load, Q_load) and predict them from neighbors + topology.

```
Before: Bus with P=100MW, Q=50MVAr visible
After:  P=?, Q=? masked; model must infer from neighbors
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MaskedInjectionSSL(nn.Module):
    """
    Self-supervised task: Reconstruct masked bus injections.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        mask_ratio: float = 0.15,
        injection_features: tuple = (0, 1),  # Indices of P, Q in node features
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.injection_features = injection_features
        
        # Decoder: predict masked injections from embeddings
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(injection_features)),
        )
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(len(injection_features)))
    
    def forward(self, data):
        x = data.x.clone()
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        num_nodes = x.shape[0]
        
        # Select nodes to mask
        num_mask = int(num_nodes * self.mask_ratio)
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        
        # Store original values for loss
        original_injections = x[mask_indices][:, self.injection_features].clone()
        
        # Replace with mask token
        x[mask_indices][:, self.injection_features] = self.mask_token
        
        # Encode
        node_emb = self.encoder(x, edge_index, edge_attr)
        
        # Decode masked nodes
        pred_injections = self.decoder(node_emb[mask_indices])
        
        # Reconstruction loss
        loss = F.mse_loss(pred_injections, original_injections)
        
        return loss, {
            'mask_indices': mask_indices,
            'pred': pred_injections,
            'target': original_injections,
        }


class MaskedInjectionTrainer:
    """Training loop for masked injection SSL."""
    
    def __init__(
        self,
        model: MaskedInjectionSSL,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, _ = self.model(batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
```

## Masked Edge Feature Reconstruction

### Concept

Mask line parameters (conductance, susceptance) and predict from endpoints + topology.

```
Before: Line with G=0.1, B=0.5 visible
After:  G=?, B=? masked; model predicts from connected buses
```

### Implementation

```python
class MaskedEdgeSSL(nn.Module):
    """
    Self-supervised task: Reconstruct masked edge features.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        edge_dim: int = 4,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        
        # Edge decoder: concat endpoint embeddings → predict edge features
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        
        # Learnable mask token for edges
        self.edge_mask_token = nn.Parameter(torch.randn(edge_dim))
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr.clone()
        
        num_edges = edge_attr.shape[0]
        
        # Select edges to mask
        num_mask = int(num_edges * self.mask_ratio)
        mask_indices = torch.randperm(num_edges)[:num_mask]
        
        # Store originals
        original_edge_attr = edge_attr[mask_indices].clone()
        
        # Replace with mask token
        edge_attr[mask_indices] = self.edge_mask_token
        
        # Encode (with masked edge features)
        node_emb = self.encoder(x, edge_index, edge_attr)
        
        # Predict masked edges from endpoint embeddings
        src_idx = edge_index[0, mask_indices]
        dst_idx = edge_index[1, mask_indices]
        
        edge_emb = torch.cat([node_emb[src_idx], node_emb[dst_idx]], dim=-1)
        pred_edge_attr = self.edge_decoder(edge_emb)
        
        # Loss
        loss = F.mse_loss(pred_edge_attr, original_edge_attr)
        
        return loss, {
            'mask_indices': mask_indices,
            'pred': pred_edge_attr,
            'target': original_edge_attr,
        }
```

## Combined SSL Objective

```python
class CombinedSSL(nn.Module):
    """
    Joint masked injection + masked edge SSL.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        edge_dim: int = 4,
        injection_features: tuple = (0, 1),
        node_mask_ratio: float = 0.15,
        edge_mask_ratio: float = 0.10,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.node_mask_ratio = node_mask_ratio
        self.edge_mask_ratio = edge_mask_ratio
        self.injection_features = injection_features
        
        # Decoders
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(injection_features)),
        )
        
        self.edge_decoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        
        # Mask tokens
        self.node_mask = nn.Parameter(torch.randn(len(injection_features)))
        self.edge_mask = nn.Parameter(torch.randn(edge_dim))
    
    def forward(self, data, alpha=0.5):
        """
        alpha: weight between node and edge SSL losses
        """
        x = data.x.clone()
        edge_attr = data.edge_attr.clone()
        edge_index = data.edge_index
        
        # Mask nodes
        num_nodes = x.shape[0]
        num_node_mask = int(num_nodes * self.node_mask_ratio)
        node_mask_idx = torch.randperm(num_nodes)[:num_node_mask]
        
        orig_node = x[node_mask_idx][:, self.injection_features].clone()
        x[node_mask_idx][:, self.injection_features] = self.node_mask
        
        # Mask edges
        num_edges = edge_attr.shape[0]
        num_edge_mask = int(num_edges * self.edge_mask_ratio)
        edge_mask_idx = torch.randperm(num_edges)[:num_edge_mask]
        
        orig_edge = edge_attr[edge_mask_idx].clone()
        edge_attr[edge_mask_idx] = self.edge_mask
        
        # Encode
        node_emb = self.encoder(x, edge_index, edge_attr)
        
        # Decode nodes
        pred_node = self.node_decoder(node_emb[node_mask_idx])
        node_loss = F.mse_loss(pred_node, orig_node)
        
        # Decode edges
        src = edge_index[0, edge_mask_idx]
        dst = edge_index[1, edge_mask_idx]
        edge_emb = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        pred_edge = self.edge_decoder(edge_emb)
        edge_loss = F.mse_loss(pred_edge, orig_edge)
        
        total_loss = alpha * node_loss + (1 - alpha) * edge_loss
        
        return total_loss, {
            'node_loss': node_loss.item(),
            'edge_loss': edge_loss.item(),
        }
```

## Contrastive Learning (Alternative)

```python
class ContrastiveSSL(nn.Module):
    """
    Contrastive learning: similar grid states → similar embeddings.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # EMA encoder for stability
        self.ema_encoder = deepcopy(encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        
        self.ema_decay = 0.99
    
    def augment(self, data):
        """Create augmented view via small perturbations."""
        aug_data = data.clone()
        
        # Add noise to injections
        noise = torch.randn_like(aug_data.x) * 0.05
        aug_data.x = aug_data.x + noise
        
        return aug_data
    
    def forward(self, data):
        # Two augmented views
        view1 = self.augment(data)
        view2 = self.augment(data)
        
        # Online encoder (view1)
        emb1 = self.encoder(view1.x, view1.edge_index, view1.edge_attr)
        z1 = self.projector(emb1.mean(dim=0))  # Graph-level
        
        # EMA encoder (view2)
        with torch.no_grad():
            emb2 = self.ema_encoder(view2.x, view2.edge_index, view2.edge_attr)
            z2 = emb2.mean(dim=0)  # No projection for target
        
        # Cosine similarity loss
        loss = 1 - F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0))
        
        return loss.mean(), {}
    
    @torch.no_grad()
    def update_ema(self):
        """Update EMA encoder parameters."""
        for p_online, p_ema in zip(
            self.encoder.parameters(),
            self.ema_encoder.parameters()
        ):
            p_ema.data = self.ema_decay * p_ema.data + (1 - self.ema_decay) * p_online.data
```

## Pretraining Workflow

```python
def pretrain_ssl(
    encoder: nn.Module,
    train_loader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda',
):
    """Complete SSL pretraining loop."""
    
    ssl_model = CombinedSSL(encoder, hidden_dim=128, edge_dim=4).to(device)
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    for epoch in range(num_epochs):
        ssl_model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss, info = ssl_model(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | SSL Loss: {avg_loss:.4f}")
    
    # Return pretrained encoder
    return encoder
```

## Fine-Tuning from SSL

```python
def finetune_from_ssl(
    pretrained_encoder: nn.Module,
    task: str,  # 'pf', 'opf', 'cascade'
    train_loader,
    val_loader,
    num_epochs: int = 50,
    lr: float = 1e-4,  # Lower LR for fine-tuning
    device: str = 'cuda',
):
    """Fine-tune pretrained encoder on downstream task."""
    
    # Create full model with pretrained encoder
    model = PowerGraphGNN(
        node_in_dim=...,  # From data
        edge_in_dim=...,
        hidden_dim=128,
    ).to(device)
    
    # Load pretrained weights
    model.encoder.load_state_dict(pretrained_encoder.state_dict())
    
    # Optionally freeze encoder initially
    for p in model.encoder.parameters():
        p.requires_grad = False
    
    # Train heads only first
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10  # Higher LR for heads
    )
    
    for epoch in range(num_epochs // 2):
        # Train heads only
        train_epoch(model, train_loader, optimizer, task, device)
    
    # Unfreeze and train all
    for p in model.encoder.parameters():
        p.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs // 2):
        train_epoch(model, train_loader, optimizer, task, device)
    
    return model
```

## Low-Label Experiments

```python
def low_label_experiment(
    pretrained_encoder,
    full_train_loader,
    val_loader,
    label_fractions: list = [0.1, 0.2, 0.5, 1.0],
):
    """Evaluate SSL benefit under different label budgets."""
    
    results = {}
    
    for frac in label_fractions:
        # Subsample training data
        subset_loader = subsample_loader(full_train_loader, frac)
        
        # Train from scratch
        scratch_model = train_from_scratch(subset_loader, val_loader)
        scratch_metric = evaluate(scratch_model, val_loader)
        
        # Fine-tune from SSL
        ssl_model = finetune_from_ssl(pretrained_encoder, subset_loader, val_loader)
        ssl_metric = evaluate(ssl_model, val_loader)
        
        results[frac] = {
            'scratch': scratch_metric,
            'ssl': ssl_metric,
            'improvement': (ssl_metric - scratch_metric) / scratch_metric * 100,
        }
        
        print(f"{frac*100:.0f}% labels: Scratch={scratch_metric:.4f}, SSL={ssl_metric:.4f}")
    
    return results
```

## Avoiding SSL Collapse

1. **Use stop-gradient** on one branch (asymmetric design)
2. **Batch normalization** in projector
3. **EMA encoder** for stability
4. **Diverse augmentations** (not just noise)
5. **Monitor embedding variance** during training

```python
def check_collapse(embeddings):
    """Check if SSL has collapsed (all embeddings similar)."""
    std = embeddings.std(dim=0).mean()
    if std < 0.1:
        print("WARNING: Possible SSL collapse detected!")
    return std.item()
```
