#!/usr/bin/env python3
"""
Generate publication-quality figures for ACTIVSg500 multi-seed validation results.
Outputs figures to Paper/Final Version For Review/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Multi-seed validation results
SEEDS = [42, 123, 456, 789, 1024]
SCRATCH_F1 = [0.3103, 0.3499, 0.3103, 0.3103, 0.3103]
SSL_F1 = [0.6957, 0.6986, 0.9444, 0.6591, 0.6199]

# Statistics
SCRATCH_MEAN = np.mean(SCRATCH_F1)
SCRATCH_STD = np.std(SCRATCH_F1)
SSL_MEAN = np.mean(SSL_F1)
SSL_STD = np.std(SSL_F1)
IMPROVEMENT = (SSL_MEAN - SCRATCH_MEAN) / SCRATCH_MEAN * 100


def create_multiseed_comparison():
    """Create bar chart comparing SSL vs Scratch across seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(SEEDS))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, SCRATCH_F1, width, label='Scratch (Random Init)',
                   color='#E57373', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, SSL_F1, width, label='SSL Pretrained',
                   color='#64B5F6', edgecolor='black', linewidth=0.5)

    # Add horizontal lines for means
    ax.axhline(y=SCRATCH_MEAN, color='#C62828', linestyle='--', linewidth=1.5,
               label=f'Scratch Mean ({SCRATCH_MEAN:.3f})')
    ax.axhline(y=SSL_MEAN, color='#1565C0', linestyle='--', linewidth=1.5,
               label=f'SSL Mean ({SSL_MEAN:.3f})')

    # Labels and formatting
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('F1 Score')
    ax.set_title('ACTIVSg500 Cascade Prediction: Multi-Seed Validation')
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.9)

    # Add improvement annotation
    ax.annotate(f'+{IMPROVEMENT:.1f}%\nimprovement',
                xy=(0.02, 0.85), xycoords='axes fraction',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Add p-value annotation
    ax.annotate('p = 0.002',
                xy=(0.02, 0.75), xycoords='axes fraction',
                fontsize=9, fontstyle='italic',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    return fig


def create_summary_comparison():
    """Create summary bar chart with error bars."""
    fig, ax = plt.subplots(figsize=(5, 4))

    methods = ['Scratch\n(Random Init)', 'SSL\nPretrained']
    means = [SCRATCH_MEAN, SSL_MEAN]
    stds = [SCRATCH_STD, SSL_STD]
    colors = ['#E57373', '#64B5F6']

    bars = ax.bar(methods, means, yerr=stds, capsize=8,
                  color=colors, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f} ± {std:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.02),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('F1 Score')
    ax.set_title('ACTIVSg500: SSL vs Scratch (n=5 seeds)')
    ax.set_ylim(0, 1.1)

    # Add improvement arrow
    ax.annotate('', xy=(1, SSL_MEAN), xytext=(0, SCRATCH_MEAN),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate(f'+{IMPROVEMENT:.1f}%', xy=(0.5, (SSL_MEAN + SCRATCH_MEAN)/2 + 0.05),
                ha='center', fontsize=11, fontweight='bold', color='green')

    plt.tight_layout()
    return fig


def create_grid_scalability():
    """Create bar chart showing SSL advantage across grid sizes."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    grids = ['IEEE 24-bus\n(24 nodes)', 'IEEE 118-bus\n(118 nodes)', 'ACTIVSg500\n(500 nodes)']
    scratch_f1 = [0.85, 0.91, 0.32]
    ssl_f1 = [0.92, 0.99, 0.72]
    improvements = [(s - sc) / sc * 100 for s, sc in zip(ssl_f1, scratch_f1)]

    x = np.arange(len(grids))
    width = 0.35

    bars1 = ax.bar(x - width/2, scratch_f1, width, label='Scratch',
                   color='#E57373', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ssl_f1, width, label='SSL Pretrained',
                   color='#64B5F6', edgecolor='black', linewidth=0.5)

    # Add improvement labels
    for i, (imp, y) in enumerate(zip(improvements, ssl_f1)):
        ax.annotate(f'+{imp:.1f}%', xy=(x[i] + width/2, y + 0.03),
                    ha='center', fontsize=9, fontweight='bold', color='green')

    ax.set_xlabel('Power Grid')
    ax.set_ylabel('F1 Score')
    ax.set_title('Cascade Prediction: SSL Advantage Increases with Grid Scale')
    ax.set_xticks(x)
    ax.set_xticklabels(grids)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left')

    plt.tight_layout()
    return fig


def main():
    # Output directory
    output_dir = Path(__file__).parent.parent / "Paper" / "Final Version For Review" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ACTIVSg500 figures...")

    # Generate multi-seed comparison
    fig1 = create_multiseed_comparison()
    fig1.savefig(output_dir / "activsg500_multiseed_comparison.pdf", format='pdf')
    fig1.savefig(output_dir / "activsg500_multiseed_comparison.png", format='png')
    print(f"  Saved: activsg500_multiseed_comparison.pdf/png")

    # Generate summary with error bars
    fig2 = create_summary_comparison()
    fig2.savefig(output_dir / "activsg500_summary.pdf", format='pdf')
    fig2.savefig(output_dir / "activsg500_summary.png", format='png')
    print(f"  Saved: activsg500_summary.pdf/png")

    # Generate grid scalability comparison
    fig3 = create_grid_scalability()
    fig3.savefig(output_dir / "grid_scalability_ssl.pdf", format='pdf')
    fig3.savefig(output_dir / "grid_scalability_ssl.png", format='png')
    print(f"  Saved: grid_scalability_ssl.pdf/png")

    plt.close('all')
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
