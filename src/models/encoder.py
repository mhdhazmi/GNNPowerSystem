"""
Physics-Guided GNN Encoder

Message passing weighted by line admittance, embedding Kirchhoff's laws.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class PhysicsGuidedConv(MessagePassing):
    """
    Message passing weighted by line admittance.
    Embeds Kirchhoff's laws into message aggregation.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="add")

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
        # Edge features → |Y_ij| for weighting
        y_mag = self.admittance_scale(edge_attr).sigmoid()

        # Edge features for message modulation
        edge_emb = self.lin_edge(edge_attr)

        # Message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_emb, y_mag=y_mag)

    def message(self, x_j, edge_attr, y_mag):
        # Physics-weighted message: neighbor features scaled by admittance
        return y_mag * (x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class PhysicsGuidedEncoder(nn.Module):
    """
    Multi-layer physics-guided GNN encoder.

    Uses residual connections and layer normalization for stable training.
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

        self.hidden_dim = hidden_dim

        # Input projection
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        # Physics-guided convolutions
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(PhysicsGuidedConv(hidden_dim, hidden_dim, hidden_dim))
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
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new  # Residual

        return x


class SimpleGNNEncoder(nn.Module):
    """
    Simple GNN encoder without physics-guided weighting.
    Useful as a baseline comparison.
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

        from torch_geometric.nn import GINEConv

        self.hidden_dim = hidden_dim

        # Input projection
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)

        # GIN convolutions with edge features
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
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
            x = x + x_new

        return x
