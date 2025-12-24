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
from .ssl import CombinedSSL, MaskedEdgeReconstruction, MaskedNodeReconstruction, ProjectionHead
from .graphmae import GraphMAE, GINEncoder, scaled_cosine_error

# Contrastive SSL
from .ssl_contrastive import GraphCL, GRACE, InfoGraph
from .losses import NTXentLoss, BarlowTwinsLoss
from .augmentations import (
    GraphAugmentation,
    Identity,
    NodeFeatureMasking,
    EdgeDropping,
    FeaturePerturbation,
    EdgeFeaturePerturbation,
    SubgraphSampling,
    PhysicsAwareEdgeDropping,
    PhysicsAwareNodeMasking,
    Compose,
    RandomChoice,
    create_augmentation,
)

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
    # SSL models (masking-based)
    "MaskedNodeReconstruction",
    "MaskedEdgeReconstruction",
    "CombinedSSL",
    "ProjectionHead",
    # GraphMAE baseline
    "GraphMAE",
    "GINEncoder",
    "scaled_cosine_error",
    # SSL models (contrastive)
    "GraphCL",
    "GRACE",
    "InfoGraph",
    # Contrastive losses
    "NTXentLoss",
    "BarlowTwinsLoss",
    # Graph augmentations
    "GraphAugmentation",
    "Identity",
    "NodeFeatureMasking",
    "EdgeDropping",
    "FeaturePerturbation",
    "EdgeFeaturePerturbation",
    "SubgraphSampling",
    "PhysicsAwareEdgeDropping",
    "PhysicsAwareNodeMasking",
    "Compose",
    "RandomChoice",
    "create_augmentation",
    # Loss functions
    "pf_loss",
    "cascade_loss",
]
