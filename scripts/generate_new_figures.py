#!/usr/bin/env python3
"""
Generate figures for new experimental results:
1. GraphMAE vs Physics-SSL comparison
2. Extended robustness (topology dropout, measurement noise)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("Paper/Final Version For Review/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings for IEEE papers
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (3.5, 2.5),  # Single column width
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Colors
COLORS = {
    'scratch': '#E74C3C',  # Red
    'graphmae': '#F39C12',  # Orange
    'physics_ssl': '#3498DB',  # Blue
    'ssl': '#3498DB',  # Blue
}


def create_graphmae_comparison():
    """Create bar chart comparing GraphMAE vs Physics-SSL."""

    # Data from experiments
    data = {
        'IEEE-24': {
            '10%': {'GraphMAE': 0.667, 'Physics-SSL': 0.903},
            '100%': {'GraphMAE': 0.964, 'Physics-SSL': 0.984},
        },
        'IEEE-118': {
            '10%': {'GraphMAE': 0.000, 'Physics-SSL': 0.715},
            '100%': {'GraphMAE': 0.998, 'Physics-SSL': 0.996},
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    x = np.arange(2)  # 10%, 100%
    width = 0.35

    for idx, (grid, grid_data) in enumerate(data.items()):
        ax = axes[idx]

        graphmae_vals = [grid_data['10%']['GraphMAE'], grid_data['100%']['GraphMAE']]
        physics_vals = [grid_data['10%']['Physics-SSL'], grid_data['100%']['Physics-SSL']]

        bars1 = ax.bar(x - width/2, graphmae_vals, width, label='GraphMAE',
                       color=COLORS['graphmae'], edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, physics_vals, width, label='Physics-SSL (Ours)',
                       color=COLORS['physics_ssl'], edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Label Fraction')
        ax.set_ylabel('F1-Score')
        ax.set_title(grid)
        ax.set_xticks(x)
        ax.set_xticklabels(['10%', '100%'])
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

        # Add improvement annotation for 10% labels
        improvement = (physics_vals[0] - graphmae_vals[0]) / max(graphmae_vals[0], 0.001) * 100
        if graphmae_vals[0] == 0:
            ax.annotate(f'GraphMAE\nfails', xy=(x[0] - width/2, 0.05),
                       ha='center', va='bottom', fontsize=7, color='red')
        elif improvement > 10:
            ax.annotate(f'+{improvement:.0f}%',
                       xy=(x[0], max(physics_vals[0], graphmae_vals[0]) + 0.12),
                       ha='center', fontsize=9, fontweight='bold', color='green')

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_DIR / 'graphmae_comparison.pdf')
    fig.savefig(OUTPUT_DIR / 'graphmae_comparison.png')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'graphmae_comparison.pdf'}")


def create_extended_robustness():
    """Create figure showing extended robustness results."""

    # IEEE-118 extended robustness data
    perturbations = ['Nominal', 'Load 1.3×', 'Noise σ=0.1', 'Topology -10%']
    scratch_f1 = [0.987, 0.958, 0.969, 0.180]
    ssl_f1 = [0.994, 0.985, 0.989, 0.160]

    fig, ax = plt.subplots(figsize=(4.5, 3))

    x = np.arange(len(perturbations))
    width = 0.35

    bars1 = ax.bar(x - width/2, scratch_f1, width, label='Scratch',
                   color=COLORS['scratch'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ssl_f1, width, label='SSL',
                   color=COLORS['ssl'], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Perturbation Type')
    ax.set_ylabel('F1-Score')
    ax.set_title('IEEE 118-bus Extended Robustness')
    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=15, ha='right')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')

    # Add delta annotations
    for i, (s, ssl) in enumerate(zip(scratch_f1, ssl_f1)):
        delta = ssl - s
        color = 'green' if delta > 0 else 'red'
        sign = '+' if delta > 0 else ''
        ax.annotate(f'{sign}{delta:.3f}',
                   xy=(x[i], max(s, ssl) + 0.03),
                   ha='center', fontsize=8, color=color)

    # Add warning box for topology dropout
    ax.axvspan(2.5, 3.5, alpha=0.2, color='red')
    ax.annotate('Both fail\nunder topology\nperturbation',
               xy=(3, 0.35), ha='center', fontsize=7,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'extended_robustness.pdf')
    fig.savefig(OUTPUT_DIR / 'extended_robustness.png')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'extended_robustness.pdf'}")


def create_robustness_degradation_curves():
    """Create line plots showing degradation under each perturbation type."""

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))

    # Load scaling data (IEEE-118)
    load_factors = [1.0, 1.1, 1.2, 1.3]
    ssl_load = [0.994, 0.994, 0.990, 0.985]
    scratch_load = [0.987, 0.987, 0.976, 0.958]

    ax = axes[0]
    ax.plot(load_factors, ssl_load, 'o-', color=COLORS['ssl'], label='SSL', linewidth=2, markersize=6)
    ax.plot(load_factors, scratch_load, 's--', color=COLORS['scratch'], label='Scratch', linewidth=2, markersize=6)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('F1-Score')
    ax.set_title('Load Stress')
    ax.set_ylim(0.9, 1.01)
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xticks(load_factors)

    # Measurement noise data (IEEE-118)
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    ssl_noise = [0.994, 0.994, 0.992, 0.989]
    scratch_noise = [0.987, 0.987, 0.983, 0.969]

    ax = axes[1]
    ax.plot(noise_levels, ssl_noise, 'o-', color=COLORS['ssl'], label='SSL', linewidth=2, markersize=6)
    ax.plot(noise_levels, scratch_noise, 's--', color=COLORS['scratch'], label='Scratch', linewidth=2, markersize=6)
    ax.set_xlabel('Noise Std (σ)')
    ax.set_ylabel('F1-Score')
    ax.set_title('Measurement Noise')
    ax.set_ylim(0.9, 1.01)
    ax.legend(loc='lower left', fontsize=8)

    # Topology dropout data (IEEE-118)
    dropout_rates = [0.0, 0.05, 0.1, 0.15]
    ssl_dropout = [0.994, 0.266, 0.160, 0.121]
    scratch_dropout = [0.987, 0.318, 0.180, 0.137]

    ax = axes[2]
    ax.plot(dropout_rates, ssl_dropout, 'o-', color=COLORS['ssl'], label='SSL', linewidth=2, markersize=6)
    ax.plot(dropout_rates, scratch_dropout, 's--', color=COLORS['scratch'], label='Scratch', linewidth=2, markersize=6)
    ax.set_xlabel('Edge Dropout Rate')
    ax.set_ylabel('F1-Score')
    ax.set_title('Topology Dropout')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=0.057, color='gray', linestyle=':', alpha=0.7)  # Class rate
    ax.annotate('class rate', xy=(0.12, 0.09), fontsize=7, color='gray')

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'robustness_degradation_curves.pdf')
    fig.savefig(OUTPUT_DIR / 'robustness_degradation_curves.png')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'robustness_degradation_curves.pdf'}")


def create_projection_head_comparison():
    """Create figure comparing projection head vs baseline SSL."""

    # Data from experiments
    label_fractions = ['10%', '50%', '100%']
    baseline_ssl = [0.715, 0.996, 0.996]
    projection_ssl = [0.546, 0.997, 0.999]

    fig, ax = plt.subplots(figsize=(4, 3))

    x = np.arange(len(label_fractions))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_ssl, width, label='Baseline SSL',
                   color=COLORS['ssl'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, projection_ssl, width, label='+ Projection Head',
                   color='#9B59B6', edgecolor='black', linewidth=0.5)  # Purple

    ax.set_xlabel('Label Fraction')
    ax.set_ylabel('F1-Score')
    ax.set_title('Projection Head Ablation (IEEE 118-bus)')
    ax.set_xticks(x)
    ax.set_xticklabels(label_fractions)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='lower right')

    # Add delta annotations
    for i, (base, proj) in enumerate(zip(baseline_ssl, projection_ssl)):
        delta = proj - base
        color = 'green' if delta > 0 else 'red'
        sign = '+' if delta > 0 else ''
        y_pos = max(base, proj) + 0.03
        ax.annotate(f'{sign}{delta:.3f}',
                   xy=(x[i], y_pos),
                   ha='center', fontsize=8, color=color, fontweight='bold')

    # Highlight the negative impact at 10%
    ax.annotate('Projection head\nhurts at low labels!',
               xy=(0, 0.45), ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.8))

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / 'projection_head_ablation.pdf')
    fig.savefig(OUTPUT_DIR / 'projection_head_ablation.png')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'projection_head_ablation.pdf'}")


if __name__ == "__main__":
    print("Generating new figures...")
    create_graphmae_comparison()
    create_extended_robustness()
    create_robustness_degradation_curves()
    create_projection_head_comparison()
    print("Done!")
