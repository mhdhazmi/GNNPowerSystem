"""Model architectures for PowerGraph GNN."""

from .encoder import PhysicsGuidedConv, PhysicsGuidedEncoder, SimpleGNNEncoder
from .gnn import (
    CascadeBaselineModel,
    PFBaselineModel,
    PowerGraphGNN,
    cascade_loss,
    pf_loss,
)
from .heads import CascadeBinaryHead, CascadeHead, OPFHead, PowerFlowHead

__all__ = [
    # Encoders
    "PhysicsGuidedConv",
    "PhysicsGuidedEncoder",
    "SimpleGNNEncoder",
    # Heads
    "PowerFlowHead",
    "OPFHead",
    "CascadeHead",
    "CascadeBinaryHead",
    # Full models
    "PowerGraphGNN",
    "PFBaselineModel",
    "CascadeBaselineModel",
    # Loss functions
    "pf_loss",
    "cascade_loss",
]
