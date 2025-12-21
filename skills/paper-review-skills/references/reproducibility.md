# Reproducibility Reference

## Priority Levels

| Level | Definition | Impact |
|-------|------------|--------|
| P0 | Cannot reproduce | Paper rejection risk |
| P1 | Likely result drift | Reviewers will question |
| P2 | Polish issues | Minor reviewer concerns |

## Minimum Reproduction Package

### Required Files
```
project/
├── README.md           # Setup + run instructions
├── requirements.txt    # Python dependencies with versions
├── configs/
│   ├── base.yaml       # Default hyperparameters
│   └── splits.yaml     # Train/val/test splits
├── scripts/
│   ├── pretrain_ssl.py
│   ├── train_*.py
│   └── eval_*.py
└── data/
    └── README.md       # Data download/generation instructions
```

### Required Documentation

1. **Environment**:
   - Python version
   - PyTorch version
   - PyG (PyTorch Geometric) version
   - GPU requirements

2. **Data**:
   - IEEE 24-bus and 118-bus source
   - Preprocessing steps
   - Train/val/test split ratios
   - Random seed for splits

3. **Training**:
   - Exact commands to reproduce
   - Hyperparameters (learning rate, epochs, batch size)
   - Seeds used (e.g., 42, 123, 456, 789, 1011)
   - Expected runtime

4. **Evaluation**:
   - Metrics computation code
   - Expected output values (for verification)

## Data Leakage Checklist

| Risk | Check | Fix |
|------|-------|-----|
| Future leakage | Temporal data splits respect time order | Sort by timestamp before split |
| Graph leakage | Test edges not in train graph | Verify edge disjoint |
| Feature leakage | No test labels in train features | Audit feature pipeline |
| Normalization leakage | Normalize with train stats only | fit_transform on train, transform on test |

## Reporting Standards

### Required Statistics
- Mean ± std across N seeds (N ≥ 3, prefer 5)
- Seed values explicitly stated
- Statistical significance test (paired t-test or Wilcoxon)
- Effect size when claiming improvement

### Table Format
```
| Method | Metric (↓/↑) | IEEE-24 | IEEE-118 |
|--------|--------------|---------|----------|
| Scratch | F1 (↑) | 0.82±0.03 | 0.26±0.24 |
| SSL | F1 (↑) | 0.88±0.02 | 0.87±0.05 |
| Δ (p-value) | | +0.06 (p<0.05) | +0.61 (p<0.01) |
```

## Common Reproducibility Failures

1. **Unstated preprocessing**: "Standard normalization" without specifics
2. **Missing seeds**: Results not reproducible due to random init
3. **Ambiguous splits**: "80/10/10 split" without stating stratification
4. **Version drift**: Works on PyTorch 1.x, fails on 2.x
5. **Hardware dependence**: GPU-specific numerics differ
