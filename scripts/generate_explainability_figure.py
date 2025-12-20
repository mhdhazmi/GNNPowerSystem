#!/usr/bin/env python3
"""
Generate Explainability Visualization Figure

Creates a visualization showing Integrated Gradients edge importance on
the IEEE 24-bus network for a cascade prediction example.

Output: Paper/Final Version For Review/figures/explainability_example.pdf
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from src.data import PowerGraphDataset
from src.models import CascadeBaselineModel
from src.utils import get_device


# IEEE 24-bus node positions (approximate geographic layout)
IEEE24_POSITIONS = {
    0: (0.1, 0.9),    # Bus 1
    1: (0.1, 0.7),    # Bus 2
    2: (0.25, 0.85),  # Bus 3
    3: (0.35, 0.75),  # Bus 4
    4: (0.35, 0.6),   # Bus 5
    5: (0.5, 0.8),    # Bus 6
    6: (0.5, 0.65),   # Bus 7
    7: (0.65, 0.75),  # Bus 8
    8: (0.65, 0.6),   # Bus 9
    9: (0.8, 0.7),    # Bus 10
    10: (0.25, 0.5),  # Bus 11
    11: (0.25, 0.35), # Bus 12
    12: (0.4, 0.45),  # Bus 13
    13: (0.55, 0.5),  # Bus 14
    14: (0.7, 0.45),  # Bus 15
    15: (0.55, 0.35), # Bus 16
    16: (0.7, 0.3),   # Bus 17
    17: (0.85, 0.5),  # Bus 18
    18: (0.1, 0.2),   # Bus 19
    19: (0.25, 0.15), # Bus 20
    20: (0.4, 0.2),   # Bus 21
    21: (0.55, 0.15), # Bus 22
    22: (0.7, 0.1),   # Bus 23
    23: (0.85, 0.25), # Bus 24
}


def find_cascade_sample_with_explanation(dataset, min_critical_edges=2):
    """Find a sample with cascade label and ground-truth explanation."""
    for i, data in enumerate(dataset):
        # Check if it's a cascade (positive label)
        if data.y.item() == 1:
            # Check if it has ground truth explanation
            if hasattr(data, "edge_mask") and data.edge_mask is not None:
                num_critical = data.edge_mask.sum().item()
                if num_critical >= min_critical_edges:
                    return i, data
    return None, None


def create_network_visualization(
    edge_index: torch.Tensor,
    edge_importance: np.ndarray,
    ground_truth_mask: np.ndarray = None,
    num_nodes: int = 24,
    output_path: str = "explainability_example.pdf",
):
    """
    Create a network visualization with edge importance coloring.

    Args:
        edge_index: [2, num_edges] edge connectivity
        edge_importance: [num_edges] importance scores (0-1 normalized)
        ground_truth_mask: [num_edges] binary ground truth mask (optional)
        num_nodes: Number of nodes in the graph
        output_path: Path to save the figure
    """
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # Add edges (handling undirected by keeping unique pairs)
    edge_list = []
    edge_importance_dict = {}
    ground_truth_dict = {}

    edge_index_np = edge_index.cpu().numpy()

    for idx in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, idx], edge_index_np[1, idx]
        edge_key = (min(src, dst), max(src, dst))

        if edge_key not in edge_importance_dict:
            edge_list.append(edge_key)
            edge_importance_dict[edge_key] = edge_importance[idx]
            if ground_truth_mask is not None:
                ground_truth_dict[edge_key] = ground_truth_mask[idx]

    G.add_edges_from(edge_list)

    # Get positions
    pos = {i: IEEE24_POSITIONS.get(i, (np.random.rand(), np.random.rand()))
           for i in range(num_nodes)}

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Create colormap: blue (low) -> yellow (medium) -> red (high)
    colors = ["#2166ac", "#67a9cf", "#fddbc7", "#ef8a62", "#b2182b"]
    cmap = LinearSegmentedColormap.from_list("importance", colors)

    # Normalize importance scores
    importance_values = [edge_importance_dict[e] for e in edge_list]
    vmin, vmax = min(importance_values), max(importance_values)
    if vmax - vmin > 1e-6:
        norm_importance = [(v - vmin) / (vmax - vmin) for v in importance_values]
    else:
        norm_importance = [0.5] * len(importance_values)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="#333333",
        node_size=300,
        alpha=0.9,
    )

    # Draw node labels
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # Draw edges with importance coloring
    edge_colors = [cmap(ni) for ni in norm_importance]
    edge_widths = [1.5 + 3.0 * ni for ni in norm_importance]  # Width varies with importance

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edge_list,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.8,
    )

    # Mark ground truth critical edges with a marker
    if ground_truth_mask is not None:
        critical_edges = [e for e in edge_list if ground_truth_dict.get(e, 0) > 0.5]
        if critical_edges:
            # Draw circles around critical edge midpoints
            for edge in critical_edges:
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                circle = plt.Circle((mid_x, mid_y), 0.03, fill=False,
                                   color='black', linewidth=2, linestyle='--')
                ax.add_patch(circle)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label("Edge Importance (Integrated Gradients)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Add legend for ground truth marker
    if ground_truth_mask is not None and any(ground_truth_dict.values()):
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='black',
                          linestyle='--', linewidth=2,
                          label='Ground-truth critical edges')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title and formatting
    ax.set_title("Integrated Gradients Edge Attribution\nIEEE 24-Bus Cascade Prediction",
                fontsize=13, fontweight='bold')
    ax.axis('off')

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to: {output_path}")
    return output_path


def main():
    device = get_device()

    print("=" * 60)
    print("GENERATING EXPLAINABILITY FIGURE")
    print("=" * 60)

    # Load dataset
    print("\nLoading IEEE 24-bus cascade dataset...")
    dataset = PowerGraphDataset(
        root="./data",
        name="ieee24",
        task="cascade",
        label_type="binary",
        split="test",
    )
    print(f"Loaded {len(dataset)} test samples")

    # Find a good cascade sample
    print("\nFinding cascade sample with ground-truth explanation...")
    idx, sample = find_cascade_sample_with_explanation(dataset)

    if sample is None:
        print("No suitable sample found. Using first cascade sample.")
        for i, data in enumerate(dataset):
            if data.y.item() == 1:
                idx, sample = i, data
                break

    if sample is None:
        print("ERROR: No cascade samples found in test set!")
        return

    print(f"Using sample {idx}")
    print(f"  Nodes: {sample.x.size(0)}")
    print(f"  Edges: {sample.edge_index.size(1)}")
    if hasattr(sample, "edge_mask") and sample.edge_mask is not None:
        print(f"  Critical edges (ground truth): {sample.edge_mask.sum().item()}")

    # Find and load model checkpoint
    print("\nLoading trained model...")
    output_dir = Path("outputs")
    candidates = list(output_dir.glob("cascade_ieee24_*/best_model.pt"))
    candidates.extend(output_dir.glob("multiseed_ieee24_*/ssl_frac1.0_seed42/best_model.pt"))

    if not candidates:
        print("ERROR: No trained model found. Run training first.")
        return

    checkpoint_path = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Using checkpoint: {checkpoint_path}")

    # Create model
    model = CascadeBaselineModel(
        node_in_dim=sample.x.size(-1),
        edge_in_dim=sample.edge_attr.size(-1),
        hidden_dim=128,
        num_layers=4,
        dropout=0.1,
    ).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model loaded successfully")

    # Move sample to device
    sample = sample.to(device)

    # Compute edge importance using Integrated Gradients
    print("\nComputing Integrated Gradients edge importance...")
    importance = model.get_edge_importance_integrated_gradients(
        sample.x, sample.edge_index, sample.edge_attr, batch=None, steps=50
    )
    importance = importance.cpu().numpy()
    print(f"  Min importance: {importance.min():.4f}")
    print(f"  Max importance: {importance.max():.4f}")
    print(f"  Mean importance: {importance.mean():.4f}")

    # Get ground truth if available
    ground_truth = None
    if hasattr(sample, "edge_mask") and sample.edge_mask is not None:
        ground_truth = sample.edge_mask.cpu().numpy()
        print(f"  Ground truth critical edges: {ground_truth.sum()}")

    # Create visualization
    print("\nGenerating visualization...")
    output_path = str(project_root / "Paper" / "Final Version For Review" / "figures" / "explainability_example.pdf")

    create_network_visualization(
        edge_index=sample.edge_index,
        edge_importance=importance,
        ground_truth_mask=ground_truth,
        num_nodes=sample.x.size(0),
        output_path=output_path,
    )

    print("\nDone!")
    print("=" * 60)


if __name__ == "__main__":
    main()
