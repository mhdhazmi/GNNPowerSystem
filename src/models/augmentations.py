"""
Graph Augmentation Transforms for Contrastive Learning

Implements various graph augmentation strategies used by GraphCL, GRACE,
and other contrastive self-supervised learning methods.

Augmentations are designed to create diverse views of the same graph
while preserving its essential structure and semantics.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class GraphAugmentation(ABC):
    """
    Base class for graph augmentations.

    All augmentations take (x, edge_index, edge_attr) and return
    the augmented versions of these tensors.
    """

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentation to graph.

        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, D]
            batch: Batch assignment [N] (optional)

        Returns:
            Tuple of (augmented_x, augmented_edge_index, augmented_edge_attr)
        """
        pass


class Identity(GraphAugmentation):
    """No-op augmentation, returns input unchanged."""

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x, edge_index, edge_attr


class NodeFeatureMasking(GraphAugmentation):
    """
    Randomly mask node features.

    Args:
        mask_ratio: Fraction of nodes to mask (default: 0.15)
        mask_type: Type of masking:
            - "zero": Replace with zeros
            - "noise": Replace with Gaussian noise
            - "mean": Replace with feature mean
    """

    def __init__(self, mask_ratio: float = 0.15, mask_type: str = "zero"):
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        device = x.device

        # Select nodes to mask
        mask = torch.rand(num_nodes, device=device) < self.mask_ratio

        # Clone to avoid modifying original
        x_aug = x.clone()

        if mask.sum() > 0:
            if self.mask_type == "zero":
                x_aug[mask] = 0.0
            elif self.mask_type == "noise":
                x_aug[mask] = torch.randn_like(x_aug[mask])
            elif self.mask_type == "mean":
                feature_mean = x.mean(dim=0, keepdim=True)
                x_aug[mask] = feature_mean.expand(mask.sum(), -1)
            else:
                raise ValueError(f"Unknown mask_type: {self.mask_type}")

        return x_aug, edge_index, edge_attr


class EdgeDropping(GraphAugmentation):
    """
    Randomly drop edges from the graph.

    Args:
        drop_ratio: Fraction of edges to drop (default: 0.2)
    """

    def __init__(self, drop_ratio: float = 0.2):
        self.drop_ratio = drop_ratio

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        device = edge_index.device

        # Select edges to keep
        keep_mask = torch.rand(num_edges, device=device) >= self.drop_ratio

        # Apply mask
        edge_index_aug = edge_index[:, keep_mask]
        edge_attr_aug = edge_attr[keep_mask]

        return x, edge_index_aug, edge_attr_aug


class FeaturePerturbation(GraphAugmentation):
    """
    Add Gaussian noise to node features.

    Args:
        noise_std: Standard deviation of noise (default: 0.1)
        feature_indices: Optional list of feature indices to perturb.
                        If None, perturb all features.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        feature_indices: Optional[List[int]] = None,
    ):
        self.noise_std = noise_std
        self.feature_indices = feature_indices

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_aug = x.clone()

        if self.feature_indices is None:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        else:
            for idx in self.feature_indices:
                noise = torch.randn(x_aug.size(0), device=x_aug.device) * self.noise_std
                x_aug[:, idx] = x_aug[:, idx] + noise

        return x_aug, edge_index, edge_attr


class EdgeFeaturePerturbation(GraphAugmentation):
    """
    Add Gaussian noise to edge features.

    Args:
        noise_std: Standard deviation of noise (default: 0.1)
        feature_indices: Optional list of feature indices to perturb.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        feature_indices: Optional[List[int]] = None,
    ):
        self.noise_std = noise_std
        self.feature_indices = feature_indices

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_attr_aug = edge_attr.clone()

        if self.feature_indices is None:
            noise = torch.randn_like(edge_attr_aug) * self.noise_std
            edge_attr_aug = edge_attr_aug + noise
        else:
            for idx in self.feature_indices:
                noise = torch.randn(edge_attr_aug.size(0), device=edge_attr_aug.device) * self.noise_std
                edge_attr_aug[:, idx] = edge_attr_aug[:, idx] + noise

        return x, edge_index, edge_attr_aug


class SubgraphSampling(GraphAugmentation):
    """
    Sample a subgraph via random walk.

    Args:
        ratio: Fraction of nodes to keep (default: 0.8)
        walk_length: Number of random walk steps (default: 3)
    """

    def __init__(self, ratio: float = 0.8, walk_length: int = 3):
        self.ratio = ratio
        self.walk_length = walk_length

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        num_keep = max(1, int(num_nodes * self.ratio))
        device = x.device

        # Start from random nodes
        start_nodes = torch.randperm(num_nodes, device=device)[:max(1, num_keep // 2)]

        # Build adjacency for random walk
        adj = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj:
                adj[src] = []
            adj[src].append(dst)

        # Random walk to collect nodes
        visited = set(start_nodes.tolist())
        frontier = list(start_nodes.tolist())

        for _ in range(self.walk_length):
            if len(visited) >= num_keep:
                break
            new_frontier = []
            for node in frontier:
                if node in adj:
                    neighbors = adj[node]
                    if neighbors:
                        next_node = neighbors[torch.randint(len(neighbors), (1,)).item()]
                        if next_node not in visited:
                            visited.add(next_node)
                            new_frontier.append(next_node)
            frontier = new_frontier

        # Ensure we have enough nodes
        if len(visited) < num_keep:
            remaining = list(set(range(num_nodes)) - visited)
            extra = min(num_keep - len(visited), len(remaining))
            visited.update(remaining[:extra])

        keep_nodes = torch.tensor(list(visited), device=device, dtype=torch.long)

        # Create node mapping
        node_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        node_map[keep_nodes] = torch.arange(len(keep_nodes), device=device)

        # Filter edges
        src, dst = edge_index
        edge_mask = (node_map[src] >= 0) & (node_map[dst] >= 0)

        edge_index_aug = torch.stack([node_map[src[edge_mask]], node_map[dst[edge_mask]]])
        edge_attr_aug = edge_attr[edge_mask]
        x_aug = x[keep_nodes]

        return x_aug, edge_index_aug, edge_attr_aug


class PhysicsAwareEdgeDropping(GraphAugmentation):
    """
    Power grid-specific edge dropping.

    Drops edges based on physical importance:
    - Lower probability for high-loading lines (important for power transfer)
    - Higher probability for low-loading lines (less critical)

    Assumes edge_attr contains [P_flow, Q_flow, X, rating] format.

    Args:
        drop_ratio: Base fraction of edges to drop (default: 0.2)
        importance_bias: How much to bias by loading (0=uniform, 1=full bias)
    """

    def __init__(self, drop_ratio: float = 0.2, importance_bias: float = 0.5):
        self.drop_ratio = drop_ratio
        self.importance_bias = importance_bias

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_edges = edge_index.size(1)
        device = edge_index.device

        if num_edges == 0:
            return x, edge_index, edge_attr

        # Compute loading from edge features
        # Assuming edge_attr = [P_flow, Q_flow, X, rating]
        if edge_attr.size(1) >= 4:
            p_flow = edge_attr[:, 0]
            q_flow = edge_attr[:, 1]
            rating = edge_attr[:, 3]

            # Loading = sqrt(P^2 + Q^2) / rating
            apparent_power = torch.sqrt(p_flow**2 + q_flow**2)
            loading = apparent_power / (rating.abs() + 1e-8)
            loading = loading.clamp(0, 1)  # Normalize to [0, 1]
        else:
            # Fallback to uniform if format doesn't match
            loading = torch.zeros(num_edges, device=device)

        # Drop probability: lower for high-loading edges
        # drop_prob = base_ratio * (1 - importance_bias * loading)
        drop_prob = self.drop_ratio * (1 - self.importance_bias * loading)

        # Sample which edges to drop
        keep_mask = torch.rand(num_edges, device=device) >= drop_prob

        # Apply mask
        edge_index_aug = edge_index[:, keep_mask]
        edge_attr_aug = edge_attr[keep_mask]

        return x, edge_index_aug, edge_attr_aug


class PhysicsAwareNodeMasking(GraphAugmentation):
    """
    Power grid-specific node masking.

    Masks nodes based on their type/importance:
    - Lower probability for generators (node_type > 0 or based on features)
    - Higher probability for load buses

    Assumes x contains [P_net, Q_net, node_type] or similar.

    Args:
        mask_ratio: Base fraction of nodes to mask (default: 0.15)
        generator_ratio: Reduced mask ratio for generators (default: 0.05)
    """

    def __init__(self, mask_ratio: float = 0.15, generator_ratio: float = 0.05):
        self.mask_ratio = mask_ratio
        self.generator_ratio = generator_ratio

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        device = x.device

        # Identify generators based on positive net power injection
        # (Generators inject power, loads consume)
        if x.size(1) >= 1:
            p_net = x[:, 0]
            is_generator = p_net > 0.1  # Threshold for generation
        else:
            is_generator = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Compute per-node mask probability
        mask_prob = torch.full((num_nodes,), self.mask_ratio, device=device)
        mask_prob[is_generator] = self.generator_ratio

        # Sample which nodes to mask
        mask = torch.rand(num_nodes, device=device) < mask_prob

        # Clone and apply masking
        x_aug = x.clone()
        x_aug[mask] = 0.0

        return x_aug, edge_index, edge_attr


class Compose(GraphAugmentation):
    """
    Apply multiple augmentations sequentially.

    Args:
        augmentations: List of augmentations to apply in order
    """

    def __init__(self, augmentations: List[GraphAugmentation]):
        self.augmentations = augmentations

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for aug in self.augmentations:
            x, edge_index, edge_attr = aug(x, edge_index, edge_attr, batch)
        return x, edge_index, edge_attr


class RandomChoice(GraphAugmentation):
    """
    Randomly select one augmentation to apply.

    Args:
        augmentations: List of augmentations to choose from
        weights: Optional weights for each augmentation (default: uniform)
    """

    def __init__(
        self,
        augmentations: List[GraphAugmentation],
        weights: Optional[List[float]] = None,
    ):
        self.augmentations = augmentations
        if weights is None:
            weights = [1.0] * len(augmentations)
        total = sum(weights)
        self.probs = [w / total for w in weights]

    def __call__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.multinomial(torch.tensor(self.probs), 1).item()
        return self.augmentations[idx](x, edge_index, edge_attr, batch)


def create_augmentation(aug_type: str, strength: float = 0.2) -> GraphAugmentation:
    """
    Factory function to create augmentations by name.

    Args:
        aug_type: Type of augmentation:
            - "edge_drop": EdgeDropping
            - "node_mask": NodeFeatureMasking
            - "feature_noise": FeaturePerturbation
            - "subgraph": SubgraphSampling
            - "physics_edge_drop": PhysicsAwareEdgeDropping
            - "physics_node_mask": PhysicsAwareNodeMasking
            - "identity": Identity (no-op)
        strength: Augmentation strength (interpretation depends on type)

    Returns:
        GraphAugmentation instance
    """
    if aug_type == "edge_drop":
        return EdgeDropping(drop_ratio=strength)
    elif aug_type == "node_mask":
        return NodeFeatureMasking(mask_ratio=strength)
    elif aug_type == "feature_noise":
        return FeaturePerturbation(noise_std=strength)
    elif aug_type == "subgraph":
        return SubgraphSampling(ratio=1.0 - strength)
    elif aug_type == "physics_edge_drop":
        return PhysicsAwareEdgeDropping(drop_ratio=strength)
    elif aug_type == "physics_node_mask":
        return PhysicsAwareNodeMasking(mask_ratio=strength)
    elif aug_type == "identity":
        return Identity()
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
