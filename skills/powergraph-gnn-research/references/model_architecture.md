# Model Architecture Reference

## Overview

Physics-guided GNN with multi-task heads for PF/OPF/Cascade prediction.

```
Input Graph → Physics-Guided Encoder → Task-Specific Heads
                                       ├── PF Head (node-level)
                                       ├── OPF Head (node-level)
                                       └── Cascade Head (graph-level)
```

## Physics-Guided Encoder

### Base Architecture

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class PhysicsGuidedConv(MessagePassing):
    """
    Message passing weighted by line admittance.
    Embeds Kirchhoff's laws into message aggregation.
    """
    
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__(aggr='add')
        
        # Node transform
        self.lin_node = nn.Linear(in_channels, out_channels)
        
        # Edge-conditioned message weighting
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        
        # Learnable admittance scaling
        self.admittance_scale = nn.Linear(edge_dim, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        self.admittance_scale.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        # Node transformation
        x = self.lin_node(x)
        
        # Compute admittance-based weights
        # G_ij, B_ij → |Y_ij| for weighting
        y_mag = self.admittance_scale(edge_attr).sigmoid()
        
        # Edge features for message modulation
        edge_emb = self.lin_edge(edge_attr)
        
        # Message passing
        return self.propagate(
            edge_index, x=x, edge_attr=edge_emb, y_mag=y_mag
        )
    
    def message(self, x_j, edge_attr, y_mag):
        # Physics-weighted message: neighbor features scaled by admittance
        return y_mag * (x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out


class PhysicsGuidedEncoder(nn.Module):
    """
    Multi-layer physics-guided GNN encoder.
    """
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        
        # Physics-guided convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(
                PhysicsGuidedConv(hidden_dim, hidden_dim, hidden_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr):
        # Initial embeddings
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        # Message passing with residual connections
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr)
            x_new = norm(x_new)
            x_new = torch.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual
        
        return x
```

## Task-Specific Heads

### Power Flow Head

```python
class PowerFlowHead(nn.Module):
    """
    Predict voltage magnitude and angle (as sin/cos).
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Separate outputs for V_mag, sin(θ), cos(θ)
        self.v_mag_head = nn.Linear(hidden_dim // 2, 1)
        self.sin_head = nn.Linear(hidden_dim // 2, 1)
        self.cos_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, node_embeddings):
        h = self.mlp(node_embeddings)
        
        v_mag = self.v_mag_head(h).squeeze(-1)
        sin_theta = self.sin_head(h).squeeze(-1)
        cos_theta = self.cos_head(h).squeeze(-1)
        
        # Normalize sin/cos to unit circle
        norm = torch.sqrt(sin_theta**2 + cos_theta**2 + 1e-8)
        sin_theta = sin_theta / norm
        cos_theta = cos_theta / norm
        
        return {
            'v_mag': v_mag,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta,
        }
```

### OPF Head

```python
class OPFHead(nn.Module):
    """
    Predict generator setpoints and total cost.
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Per-node generation prediction
        self.gen_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Graph-level cost prediction (via pooling)
        self.cost_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, node_embeddings, gen_mask, batch=None):
        # Generator setpoints (only at generator buses)
        pg_pred = self.gen_mlp(node_embeddings).squeeze(-1)
        pg_pred = pg_pred * gen_mask  # Zero non-generator buses
        
        # Graph-level cost
        if batch is None:
            # Single graph
            graph_emb = node_embeddings.mean(dim=0, keepdim=True)
        else:
            # Batched graphs
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(node_embeddings, batch)
        
        cost_pred = self.cost_mlp(graph_emb).squeeze(-1)
        
        return {
            'pg': pg_pred,
            'cost': cost_pred,
        }
```

### Cascade Classification Head

```python
class CascadeHead(nn.Module):
    """
    Graph-level cascade severity prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 3,  # No cascade, small, large
    ):
        super().__init__()
        
        # Attention-based graph pooling
        self.gate = nn.Linear(hidden_dim, 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, node_embeddings, batch=None):
        # Attention weights
        att = torch.sigmoid(self.gate(node_embeddings))
        
        # Weighted pooling
        if batch is None:
            graph_emb = (att * node_embeddings).sum(dim=0, keepdim=True)
        else:
            from torch_scatter import scatter_sum
            graph_emb = scatter_sum(att * node_embeddings, batch, dim=0)
        
        logits = self.classifier(graph_emb)
        
        return {'logits': logits}
```

## Full Multi-Task Model

```python
class PowerGraphGNN(nn.Module):
    """
    Complete multi-task model for PF/OPF/Cascade.
    """
    
    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_cascade_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = PhysicsGuidedEncoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.pf_head = PowerFlowHead(hidden_dim)
        self.opf_head = OPFHead(hidden_dim)
        self.cascade_head = CascadeHead(hidden_dim, num_cascade_classes)
    
    def forward(
        self,
        x, edge_index, edge_attr,
        gen_mask=None,
        batch=None,
        tasks=('pf', 'opf', 'cascade'),
    ):
        # Shared encoding
        node_emb = self.encoder(x, edge_index, edge_attr)
        
        outputs = {}
        
        if 'pf' in tasks:
            outputs['pf'] = self.pf_head(node_emb)
        
        if 'opf' in tasks and gen_mask is not None:
            outputs['opf'] = self.opf_head(node_emb, gen_mask, batch)
        
        if 'cascade' in tasks:
            outputs['cascade'] = self.cascade_head(node_emb, batch)
        
        return outputs
    
    def get_embeddings(self, x, edge_index, edge_attr):
        """Extract node embeddings for analysis."""
        return self.encoder(x, edge_index, edge_attr)
```

## Loss Functions

```python
def pf_loss(pred, target, lambda_physics=0.1):
    """
    Power flow loss with physics regularization.
    """
    # Voltage magnitude
    loss_v = F.mse_loss(pred['v_mag'], target['v_mag'])
    
    # Angle (via sin/cos)
    loss_sin = F.mse_loss(pred['sin_theta'], target['sin_theta'])
    loss_cos = F.mse_loss(pred['cos_theta'], target['cos_theta'])
    
    total = loss_v + loss_sin + loss_cos
    
    return total


def opf_loss(pred, target):
    """OPF loss for generation and cost."""
    loss_pg = F.mse_loss(pred['pg'], target['pg'])
    loss_cost = F.mse_loss(pred['cost'], target['cost'])
    
    return loss_pg + 0.1 * loss_cost


def cascade_loss(pred, target, class_weights=None):
    """Cascade classification loss with optional class weighting."""
    return F.cross_entropy(
        pred['logits'], target['label'],
        weight=class_weights
    )


def multitask_loss(
    outputs, targets,
    task_weights={'pf': 1.0, 'opf': 1.0, 'cascade': 1.0},
):
    """Combined multi-task loss."""
    total = 0
    
    if 'pf' in outputs:
        total += task_weights['pf'] * pf_loss(outputs['pf'], targets['pf'])
    
    if 'opf' in outputs:
        total += task_weights['opf'] * opf_loss(outputs['opf'], targets['opf'])
    
    if 'cascade' in outputs:
        total += task_weights['cascade'] * cascade_loss(
            outputs['cascade'], targets['cascade']
        )
    
    return total
```

## Model Variants

### Attention-Based (GAT-style)

```python
from torch_geometric.nn import GATv2Conv

class GATEncoder(nn.Module):
    """Alternative: Graph Attention Network encoder."""
    
    def __init__(self, node_in, hidden, num_layers=4, heads=4):
        super().__init__()
        
        self.embed = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden if i == 0 else hidden * heads
            self.convs.append(
                GATv2Conv(in_dim, hidden, heads=heads, concat=(i < num_layers-1))
            )
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
        return x
```

### Edge-Aware (EGNN-style)

```python
from torch_geometric.nn import NNConv

class EdgeAwareEncoder(nn.Module):
    """Alternative: NNConv for edge-conditioned convolutions."""
    
    def __init__(self, node_in, edge_in, hidden, num_layers=4):
        super().__init__()
        
        self.embed = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            nn_module = nn.Sequential(
                nn.Linear(edge_in, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, nn_module))
    
    def forward(self, x, edge_index, edge_attr):
        x = self.embed(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        return x
```

## Gradient Checkpointing (Memory Optimization)

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedEncoder(PhysicsGuidedEncoder):
    """Memory-efficient encoder using gradient checkpointing."""
    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            # Checkpoint each layer
            x = checkpoint(
                self._layer_forward,
                x, edge_index, edge_attr, conv, norm,
                use_reentrant=False
            )
        
        return x
    
    def _layer_forward(self, x, edge_index, edge_attr, conv, norm):
        x_new = conv(x, edge_index, edge_attr)
        x_new = norm(x_new)
        x_new = torch.relu(x_new)
        return x + x_new
```
