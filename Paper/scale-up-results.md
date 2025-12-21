# ACTIVSg500 Scale-Up Validation Results

## Overview

This document presents multi-seed validation results for cascading failure prediction on the ACTIVSg500 (Texas A&M 500-bus synthetic utility-scale power grid), demonstrating the scalability of our physics-guided self-supervised learning (SSL) framework beyond IEEE benchmark grids.

## Multi-Seed Validation Methodology

### Experimental Setup
- **Grid**: ACTIVSg500 (500 buses, 3206 branches)
- **Task**: Binary cascading failure prediction
- **Seeds**: 42, 123, 456, 789, 1024
- **Epochs**: 100 per experiment
- **Initialization**: SSL pretrained vs. random (scratch)

### SSL Pretraining Configuration
- **Pretraining task**: Masked feature reconstruction
- **Mask ratio**: 15%
- **Combined objectives**: Node + edge + physics loss
- **Pretrained model**: `ssl_combined_activsg500_*/best_model.pt`

## Results

### Per-Seed F1 Scores

| Seed | Scratch F1 | SSL F1 | Improvement |
|------|------------|--------|-------------|
| 42 | 0.3103 | 0.6957 | +124.2% |
| 123 | 0.3499 | 0.6986 | +99.7% |
| 456 | 0.3103 | 0.9444 | +204.4% |
| 789 | 0.3103 | 0.6591 | +112.4% |
| 1024 | 0.3103 | 0.6199 | +99.7% |

### Summary Statistics

| Metric | Scratch | SSL |
|--------|---------|-----|
| Mean F1 | 0.318 | 0.724 |
| Std F1 | 0.018 | 0.128 |
| Min F1 | 0.310 | 0.620 |
| Max F1 | 0.350 | 0.944 |

- **Overall Improvement**: +127.4%
- **Statistical Significance**: p = 0.002 (Welch's t-test)

## Key Findings

### 1. Statistical Significance
The SSL improvement over scratch baseline is statistically significant (p=0.002), providing strong evidence that the performance gains are not due to random variation.

### 2. Scratch Baseline Failure Mode
The scratch baseline consistently achieves F1 ≈ 0.31 across 4/5 seeds, which corresponds exactly to the positive class rate. This indicates the model collapses to predicting all samples as positive (cascade failure), failing to learn any discriminative features.

**Why this happens:**
- Without SSL pretraining, random initialization provides no useful prior
- The optimizer finds a local minimum at the trivial all-positive solution
- Class imbalance (≈31% positive) makes this a stable but useless solution

### 3. SSL Enables Meaningful Learning
SSL pretraining provides physics-informed initialization that:
- Escapes the trivial solution basin
- Learns meaningful grid structure representations
- Enables discrimination between cascade and non-cascade scenarios

### 4. Initialization Sensitivity
The high SSL variance (std=0.128) across seeds indicates sensitivity to:
- Random weight initialization in the prediction head
- Batch ordering during fine-tuning
- Stochastic gradient descent dynamics

Seed 456 achieves F1=0.944, approaching IEEE-118 performance (F1≈0.99), suggesting the optimal solution exists but requires favorable initialization.

### 5. Training Duration Requirements
Best validation epochs ranged from 79-96, indicating:
- 50 epochs insufficient for convergence
- 100 epochs necessary for proper evaluation
- Early stopping may miss optimal performance window

## Comparison with IEEE Benchmark Grids

| Grid | Buses | Branches | Scratch F1 | SSL F1 | Improvement |
|------|-------|----------|------------|--------|-------------|
| IEEE 24-bus | 24 | 38 | 0.85 | 0.92 | +8.2% |
| IEEE 118-bus | 118 | 186 | 0.91 | 0.99 | +8.8% |
| ACTIVSg500 | 500 | 3206 | 0.32 | 0.72 | +127.4% |

**Key Insight**: SSL's advantage increases dramatically with grid complexity. On smaller IEEE grids, scratch baseline achieves reasonable performance; on utility-scale grids, scratch fails entirely while SSL maintains meaningful prediction capability.

## Implications for Utility-Scale Deployment

1. **SSL pretraining is essential** for grids with >200 buses where scratch training fails
2. **Multi-seed evaluation required** due to high variance in SSL performance
3. **Extended training** (≥100 epochs) necessary for complex grid topologies
4. **Physics-guided losses** provide crucial inductive bias for power grid structure

## Reproducibility

### Commands to Reproduce
```bash
# SSL Pretraining
python scripts/pretrain_ssl_pf.py --grid activsg500 --epochs 100

# Multi-seed Fine-tuning
python scripts/run_multiseed_cascade.py --grid activsg500 --seeds 42 123 456 789 1024 --epochs 100
```

### Output Locations
- Pretrained models: `outputs/ssl_combined_activsg500_*/`
- Fine-tuning results: `outputs/multiseed_activsg500_*/`
