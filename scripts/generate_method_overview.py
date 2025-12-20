#!/usr/bin/env python3
"""
Generate the method overview figure (Figure 1) for the paper.
This script creates the SSL pretraining and fine-tuning workflow diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.05)

# Colors
colors = {
    'data_cylinder': '#F5DEB3',  # Wheat/tan for data cylinders
    'process_box': '#E8F4E8',    # Light green for process boxes
    'encoder_box': '#E8E8F4',    # Light blue/purple for encoder
    'head_box': '#E8F4E8',       # Light green for heads
    'output_cylinder': '#D4EDDA', # Light green for outputs
    'loss_box': '#E8F4E8',       # Light green for loss
    'annotation_box': '#FFF3CD', # Light yellow for annotations
    'arrow': '#CC0000',          # Red for arrows
    'text': '#000000',
}

def draw_cylinder(ax, x, y, width, height, color, label, fontsize=9):
    """Draw a cylinder shape (for data)"""
    # Bottom ellipse
    ellipse_height = height * 0.15
    ellipse = mpatches.Ellipse((x + width/2, y), width, ellipse_height,
                                facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(ellipse)

    # Rectangle body
    rect = plt.Rectangle((x, y), width, height, facecolor=color,
                         edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    # Top ellipse
    ellipse_top = mpatches.Ellipse((x + width/2, y + height), width, ellipse_height,
                                   facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(ellipse_top)

    # Label
    ax.text(x + width/2, y + height/2, label, ha='center', va='center',
            fontsize=fontsize, wrap=True)

def draw_box(ax, x, y, width, height, color, label, fontsize=9, bold=False):
    """Draw a rectangular box"""
    rect = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.1",
                          facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, label, ha='center', va='center',
            fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, start, end, color='black', style='->', linestyle='-'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color,
                               linestyle=linestyle, lw=1.5))

# ============ Left Panel: SSL Pretraining ============
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 14)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a) SSL Pretraining', fontsize=14, fontweight='bold', pad=20)

# Draw dashed border
border = plt.Rectangle((0.3, 0.3), 9.4, 13.4, fill=False,
                       edgecolor='gray', linestyle='--', linewidth=1.5)
ax1.add_patch(border)

# Unlabeled Training Graphs (cylinder at top)
draw_cylinder(ax1, 3, 11.5, 4, 1.2, colors['data_cylinder'],
              'Unlabeled\nTraining Graphs', fontsize=9)

# Arrow down
draw_arrow(ax1, (5, 11.5), (5, 10.5), colors['arrow'], '->')

# Mask box
draw_box(ax1, 2.5, 9.5, 5, 1, 'white', 'Mask 15% of Features', fontsize=9)

# Feature annotation
ax1.text(8.5, 10, 'Node: $P_{net}$, $S_{net}$\nEdge: X, rating',
         fontsize=8, ha='left', va='center', style='italic')

# Arrow down
draw_arrow(ax1, (5, 9.5), (5, 8.5), colors['arrow'], '->')

# Masked Graph Input (cylinder)
draw_cylinder(ax1, 3, 7, 4, 1.2, '#ADD8E6', 'Masked Graph Input', fontsize=9)

# Arrow down
draw_arrow(ax1, (5, 7), (5, 6), colors['arrow'], '->')

# Physics Guided Encoder (larger box)
encoder_box = FancyBboxPatch((1.5, 3.5), 7, 2.3,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=colors['encoder_box'],
                              edgecolor='black', linewidth=1.5)
ax1.add_patch(encoder_box)
ax1.text(5, 5.2, 'PhysicsGuidedEncoder', ha='center', va='center',
         fontsize=10, fontweight='bold')
ax1.text(5, 4.5, '4 layers', ha='center', va='center', fontsize=9)
ax1.text(5, 3.9, 'Electrically-parameterized\nmessage passing',
         ha='center', va='center', fontsize=8)

# Arrow down
draw_arrow(ax1, (5, 3.5), (5, 2.8), colors['arrow'], '->')

# Reconstruction Head
draw_box(ax1, 3, 2, 4, 0.7, colors['head_box'], 'Reconstruction Head', fontsize=9)

# Arrow down
draw_arrow(ax1, (5, 2), (5, 1.3), colors['arrow'], '->')

# Reconstructed Features (cylinder)
draw_cylinder(ax1, 3, 0.3, 4, 0.8, colors['output_cylinder'],
              'Reconstructed\nFeatures', fontsize=8)

# MSE Loss box (to the left)
draw_box(ax1, 0.5, 0.5, 2, 0.8, colors['loss_box'], 'MSE Loss\n(masked positions)', fontsize=7)

# Gradient arrow (curved, going up)
ax1.annotate('', xy=(1.5, 3.5), xytext=(1.5, 1.3),
             arrowprops=dict(arrowstyle='->', color=colors['arrow'],
                            linestyle='--', lw=1.5,
                            connectionstyle='arc3,rad=0.3'))
ax1.text(0.3, 2.5, 'gradient', fontsize=8, color=colors['arrow'], rotation=90)

# "No labels needed" annotation box
annot_box = FancyBboxPatch((0.5, 5.5), 2.5, 1.2,
                           boxstyle="round,pad=0.02,rounding_size=0.1",
                           facecolor=colors['annotation_box'],
                           edgecolor='black', linestyle='--', linewidth=1)
ax1.add_patch(annot_box)
ax1.text(1.75, 6.1, 'No labels needed!', ha='center', va='center',
         fontsize=8, fontweight='bold')
ax1.text(1.75, 5.7, 'Learn from structure\nand physics', ha='center', va='center',
         fontsize=7)

# ============ Right Panel: Fine-tuning ============
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 14)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('(b) Task-Specific Fine-tuning', fontsize=14, fontweight='bold', pad=20)

# Draw dashed border
border2 = plt.Rectangle((0.3, 0.3), 9.4, 13.4, fill=False,
                        edgecolor='gray', linestyle='--', linewidth=1.5)
ax2.add_patch(border2)

# Labeled Training Graphs (cylinder at top)
draw_cylinder(ax2, 3, 12.2, 4, 1.2, colors['data_cylinder'],
              'Labeled Training\nGraphs (10-100%)', fontsize=9)

# Arrow down
draw_arrow(ax2, (5, 12.2), (5, 10.5), colors['arrow'], '->')

# Graph with Labels (cylinder)
draw_cylinder(ax2, 3, 9, 4, 1.2, '#ADD8E6', 'Graph with Labels', fontsize=9)

# Arrow down
draw_arrow(ax2, (5, 9), (5, 8), colors['arrow'], '->')

# Transfer annotation
ax2.text(8.5, 6.5, 'Weights from SSL', fontsize=8, color='blue', ha='center')
draw_arrow(ax2, (8, 6.3), (7, 5.5), 'blue', '->', linestyle='--')

# Physics Guided Encoder (larger box with blue border)
encoder_box2 = FancyBboxPatch((1.5, 5), 7, 2.3,
                               boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=colors['encoder_box'],
                               edgecolor='blue', linewidth=2)
ax2.add_patch(encoder_box2)
ax2.text(5, 6.7, 'PhysicsGuidedEncoder', ha='center', va='center',
         fontsize=10, fontweight='bold')
ax2.text(5, 6.1, '(initialized from SSL)', ha='center', va='center',
         fontsize=8, color='blue')

# Three task heads
# Power Flow Head
draw_box(ax2, 0.8, 3, 2.4, 1.2, colors['head_box'],
         'Power Flow Head\n(node-level)', fontsize=7)
draw_cylinder(ax2, 1.3, 1.5, 1.4, 0.8, colors['output_cylinder'], '$\\hat{V}$', fontsize=10)
draw_box(ax2, 1.3, 0.5, 1.4, 0.6, colors['loss_box'], 'MAE', fontsize=8)

# Line Flow Head
draw_box(ax2, 3.8, 3, 2.4, 1.2, colors['head_box'],
         'Line Flow Head\n(edge-level)', fontsize=7)
draw_cylinder(ax2, 4.3, 1.5, 1.4, 0.8, colors['output_cylinder'],
              '$\\hat{P}_{ij}, \\hat{Q}_{ij}$', fontsize=9)
draw_box(ax2, 4.3, 0.5, 1.4, 0.6, colors['loss_box'], 'MAE', fontsize=8)

# Cascade Head
draw_box(ax2, 6.8, 3, 2.4, 1.2, colors['head_box'],
         'Cascade Head\n(graph-level)', fontsize=7)
draw_cylinder(ax2, 7.3, 1.5, 1.4, 0.8, colors['output_cylinder'], 'Cascade?', fontsize=8)
draw_box(ax2, 7.3, 0.5, 1.4, 0.6, colors['loss_box'], 'BCE', fontsize=8)

# Arrows from encoder to heads
draw_arrow(ax2, (3, 5), (2, 4.2), colors['arrow'], '->')
draw_arrow(ax2, (5, 5), (5, 4.2), colors['arrow'], '->')
draw_arrow(ax2, (7, 5), (8, 4.2), colors['arrow'], '->')

# Arrows from heads to outputs
draw_arrow(ax2, (2, 3), (2, 2.3), colors['arrow'], '->')
draw_arrow(ax2, (5, 3), (5, 2.3), colors['arrow'], '->')
draw_arrow(ax2, (8, 3), (8, 2.3), colors['arrow'], '->')

# Arrows from outputs to losses
draw_arrow(ax2, (2, 1.5), (2, 1.1), colors['arrow'], '->')
draw_arrow(ax2, (5, 1.5), (5, 1.1), colors['arrow'], '->')
draw_arrow(ax2, (8, 1.5), (8, 1.1), colors['arrow'], '->')

# Physics-Guided Architecture annotation box (CORRECTED TEXT) - positioned at bottom
annot_box2 = FancyBboxPatch((1.5, 10.5), 7, 1.3,
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=colors['annotation_box'],
                            edgecolor='black', linewidth=1)
ax2.add_patch(annot_box2)
ax2.text(5, 11.4, 'Physics-Guided Architecture:', ha='center', va='center',
         fontsize=9, fontweight='bold')
ax2.text(5, 10.85, 'Message passing weighted by learned edge importance\nfrom electrical features (g, b, x, rating)',
         ha='center', va='center', fontsize=8)

# Transfer pretrained weights annotation
ax2.text(8.5, 4.5, 'Transfer\npretrained\nweights', fontsize=7, color=colors['arrow'],
         ha='center', va='center')

# Save figure
output_path = '/mnt/c/Users/hasty/OneDrive/Desktop/Code/PowerResearch/GNN/Paper/Final Version For Review/figures/method_overview.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', dpi=300)
print(f"Figure saved to {output_path}")

plt.close()
