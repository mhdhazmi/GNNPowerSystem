# VI. Results

---

## VI-A. Main Transfer Summary Across Tasks and Grids

### P1: Headline Takeaways

Self-supervised pretraining consistently improves performance across all tasks and label fractions. The benefit is most pronounced in **low-label regimes** (10-20% labels), where SSL provides 6-29% improvement. On the larger IEEE-118 grid, SSL **stabilizes training** that otherwise fails: Scratch achieves F1 = 0.262 ± 0.243 at 10% labels (high variance indicating some seeds fail entirely), while SSL achieves F1 = 0.874 ± 0.051 (consistent performance).

### P2: Main Results Table

**Table 1: SSL Transfer Benefits Across Tasks and Grid Scales**

*All results evaluated on held-out test set. Improvement defined as relative gain (positive = SSL better).*

| Task | Grid | Metric | Label % | Scratch | SSL | Improvement | Seeds |
|------|------|--------|---------|---------|-----|-------------|-------|
| **Cascade Prediction** | IEEE-24 | F1 ↑ | 10% | 0.773 ± 0.015 | **0.826 ± 0.016** | +6.8% | 5 |
| | | | 100% | 0.955 ± 0.007 | **0.958 ± 0.005** | +0.3% | 5 |
| | IEEE-118 | F1 ↑ | 10% | 0.262 ± 0.243 | **0.874 ± 0.051** | +234%* | 5 |
| | | | 100% | 0.987 ± 0.005 | **0.994 ± 0.002** | +0.7% | 5 |
| **Power Flow** | IEEE-24 | MAE ↓ | 10% | 0.0149 ± 0.0004 | **0.0106 ± 0.0003** | +29.1% | 5 |
| | | | 100% | 0.0040 ± 0.0002 | **0.0035 ± 0.0001** | +13.0% | 5 |
| **Line Flow** | IEEE-24 | MAE ↓ | 10% | 0.0084 ± 0.0003 | **0.0062 ± 0.0002** | +26.4% | 5 |
| | | | 100% | 0.0022 ± 0.00002 | **0.0021 ± 0.0005** | +2.3% | 5 |

*\*IEEE-118 at 10%: Large percentage improvement reflects F1 increase from 0.262 to 0.874 (ΔF1 = +0.612). Absolute ΔF1 is the more interpretable metric for this case.*

---

## VI-B. Cascade Prediction on IEEE-24

### P1: Setup Recap

Graph-level binary classification predicting whether a cascade occurs. Positive class defined as Demand Not Served (DNS) > 0 MW. F1 score computed at graph level.

### P2: Low-Label Effect and Learning Curve

**Table 2: Cascade Prediction Results (IEEE-24, 5-seed)**

| Label % | Scratch F1 | SSL F1 | Improvement |
|---------|------------|--------|-------------|
| 10% | 0.773 ± 0.015 | **0.826 ± 0.016** | +6.8% |
| 20% | 0.818 ± 0.019 | **0.895 ± 0.016** | +9.4% |
| 50% | 0.920 ± 0.005 | **0.940 ± 0.008** | +2.1% |
| 100% | 0.955 ± 0.007 | **0.958 ± 0.005** | +0.3% |

SSL provides largest improvement (+9.4%) at 20% labels, with gains diminishing as labeled data increases. At 100% labels, both methods achieve >95% F1, with SSL providing marginal +0.3% improvement.

### P3: Baseline Context

GNN (SSL, 10% labels) achieves F1 = 0.826, substantially outperforming:
- XGBoost with full labels (0.72)
- Heuristic Max Loading Threshold (0.41)
- Top-K Loading Check (0.52)

This demonstrates that (1) graph structure provides substantial value, and (2) SSL enables strong performance with minimal labels.

**Figures:**
- **Figure 2a**: `cascade_ssl_comparison.png` (bar chart)
- **Figure 2b**: `cascade_improvement_curve.png` (improvement vs label fraction)

---

## VI-C. Scalability: Cascade Prediction on IEEE-118

### P1: Why IEEE-118 is Harder

The IEEE-118 grid is 5× larger (118 buses, 370 lines) with more complex topology. Additionally, cascade events are rarer (~5% positive rate vs ~20% on IEEE-24), creating severe class imbalance. We address this with:
- **Focal Loss** (γ=2.0): Down-weights easy negatives
- **Threshold Tuning**: Optimal threshold via validation F1
- **Stratified Sampling**: Maintains class distribution across splits

### P2: Quantitative Results and Stability Story

**Table 3: Cascade Prediction Results (IEEE-118, 5-seed)**

| Label % | Scratch F1 | SSL F1 | ΔF1 | Improvement |
|---------|------------|--------|-----|-------------|
| 10% | 0.262 ± 0.243 | **0.874 ± 0.051** | +0.612 | +234% |
| 20% | 0.837 ± 0.020 | **0.977 ± 0.006** | +0.140 | +16.7% |
| 50% | 0.966 ± 0.004 | **0.992 ± 0.003** | +0.026 | +2.7% |
| 100% | 0.987 ± 0.005 | **0.994 ± 0.002** | +0.007 | +0.7% |

**Critical Finding**: At 10% labels, Scratch exhibits σ=0.243 (extremely high variance), indicating training instability—some seeds learn while others fail completely. SSL reduces variance to σ=0.051 (5× reduction), providing reliable performance across all seeds.

### P3: Practical Implication

The "reliability over lucky seeds" argument: with SSL, practitioners can deploy models without worrying about seed selection. Scratch training requires multiple runs hoping for good initialization, which is operationally impractical.

**Figures:**
- **Figure 3a**: `cascade_118_ssl_comparison.png`
- **Figure 3b**: `cascade_118_improvement_curve.png` (with ΔF1)

---

## VI-D. Power Flow Prediction (IEEE-24)

### P1: Task Definition and Units

Predict bus voltage magnitudes ($V_{mag}$) given load injections ($P$, $Q$). Reported as Mean Absolute Error in per-unit (p.u.) system.

### P2: Results and Interpretation

**Table 4: Power Flow Prediction Results (5-seed)**

| Label % | Scratch MAE | SSL MAE | Improvement |
|---------|-------------|---------|-------------|
| 10% | 0.0149 ± 0.0004 | **0.0106 ± 0.0003** | +29.1% |
| 20% | 0.0101 ± 0.0004 | **0.0078 ± 0.0001** | +23.1% |
| 50% | 0.0056 ± 0.0001 | **0.0048 ± 0.0001** | +13.7% |
| 100% | 0.0040 ± 0.0002 | **0.0035 ± 0.0001** | +13.0% |

SSL achieves **29.1% MAE reduction** at 10% labels—the largest improvement among all tasks. Even at 100% labels, SSL maintains 13% advantage, suggesting the pretrained representations provide lasting benefit.

**Operational Context**: MAE of 0.0106 p.u. translates to ~1% voltage error, which is within acceptable operational tolerance for most applications.

**Figures:**
- **Figure 4a**: `pf_ssl_comparison.png`
- **Figure 4b**: `pf_improvement_curve.png`

---

## VI-E. Line Flow Prediction (IEEE-24)

### P1: Task Definition and Units

Predict active and reactive power flows ($P_{ij}$, $Q_{ij}$) on each transmission line. Reported as Mean Absolute Error in per-unit system.

### P2: Results and Low-Label Advantage

**Table 5: Line Flow Prediction Results (5-seed)**

| Label % | Scratch MAE | SSL MAE | Improvement |
|---------|-------------|---------|-------------|
| 10% | 0.0084 ± 0.0003 | **0.0062 ± 0.0002** | +26.4% |
| 20% | 0.0056 ± 0.0001 | **0.0044 ± 0.0001** | +20.5% |
| 50% | 0.0031 ± 0.0001 | **0.0026 ± 0.0001** | +16.6% |
| 100% | 0.0022 ± 0.00002 | **0.0021 ± 0.0005** | +2.3% |

SSL provides consistent improvement across all label fractions, with largest gains (+26.4%) at 10% labels.

### P3: 100% Variance Note

The elevated SSL std at 100% labels (0.0005) is due to one outlier seed; median SSL MAE (0.0019) confirms typical performance is better than scratch. Per-seed breakdown available in Appendix.

**Figures:**
- **Figure 5a**: `lineflow_ssl_comparison.png`
- **Figure 5b**: `lineflow_improvement_curve.png`

---

## VI-F. Cross-Task Synthesis

### P1: Unified Story

Across all tasks, a consistent pattern emerges:
1. **Largest gains at 10% labels**: SSL improvements range from +6.8% to +29.1%
2. **Diminishing returns at 100%**: Improvements reduce to +0.3% to +13%
3. **Variance reduction**: SSL stabilizes training, especially on challenging settings

### P2: Scalability Narrative

The IEEE-24 → IEEE-118 comparison demonstrates SSL robustness to scale:
- Both grids benefit from SSL
- Larger grid sees more dramatic stabilization effect
- Physics-guided representations transfer across topologies

**Table 6: Cross-Task Summary at 10% Labels**

| Task | Grid | Metric | Scratch | SSL | Improvement |
|------|------|--------|---------|-----|-------------|
| Cascade | IEEE-24 | F1 | 0.773 | **0.826** | +6.8% |
| Cascade | IEEE-118 | F1 | 0.262 | **0.874** | +234% (ΔF1=+0.61) |
| Power Flow | IEEE-24 | MAE | 0.0149 | **0.0106** | +29.1% |
| Line Flow | IEEE-24 | MAE | 0.0084 | **0.0062** | +26.4% |

**Figures:**
- **Figure 6a**: `grid_scalability_comparison.png`
- **Figure 6b**: `multi_task_comparison.png`

---

## VI-G. Explainability Fidelity

### P1: Protocol

We evaluate whether the model's learned edge importance aligns with ground-truth failure edges from simulation. For each test graph:
1. Compute edge attribution scores via Integrated Gradients
2. Rank edges by attribution
3. Compute AUC-ROC against ground-truth failure mask

### P2: Results and Implication

**Table 7: Edge Attribution Fidelity (AUC-ROC)**

| Method | AUC-ROC | Description |
|--------|---------|-------------|
| Random (baseline) | 0.50 | Random edge ordering |
| Heuristic (line loading) | 0.72 | Rank by power flow / capacity |
| Basic Gradient | 0.62 | ∂output/∂edge_features |
| Attention-based | 0.84 | Learned attention weights |
| **Integrated Gradients** | **0.93** | Path-integrated attribution |

Integrated Gradients achieves 0.93 AUC-ROC, significantly outperforming the line loading heuristic (0.72). This demonstrates the model learns physically meaningful edge importance beyond simple loading rules.

---

## VI-H. Robustness Under Load Scaling (Preliminary)

### P1: Stress Test Framing

**Important Caveat**: This section presents **preliminary single-seed results** (seed=42) to demonstrate the trend of SSL advantage under distribution shift. Multi-seed robustness evaluation is planned for extended publication.

**Table 8: Out-of-Distribution Performance (Load Scaling)**

| Load Multiplier | Scratch F1 | SSL F1 | SSL Advantage |
|-----------------|------------|--------|---------------|
| 1.0× (In-Distribution) | 0.936 | 0.956 | +2.1% |
| 1.1× | 0.875 | 0.912 | +4.2% |
| 1.2× | 0.756 | 0.867 | +14.7% |
| **1.3× (OOD)** | 0.673 | **0.821** | **+22.0%** |

**Key Finding**: SSL advantage increases with distribution shift. At 1.3× load (out-of-distribution), SSL achieves +22% improvement, suggesting physics-grounded representations generalize better to unseen conditions.

---

## VI-I. Encoder Ablation (Appendix)

### P1: Ablation Setup

- **Training**: From scratch (no SSL) — isolates encoder architecture effect
- **Seeds**: Single seed (seed=42) — for ablation comparison only
- **Note**: Numbers differ from main results (multi-seed SSL)

**Table 9: Encoder Architecture Comparison (Scratch, Single-Seed)**

| Encoder | 10% Labels | 50% Labels | 100% Labels |
|---------|------------|------------|-------------|
| **PhysicsGuided** | **0.774** | 0.876 | 0.919 |
| Vanilla GNN | 0.767 | 0.859 | **0.946** |
| Standard GCN | 0.598 | 0.861 | 0.938 |

PhysicsGuidedEncoder outperforms alternatives in low-data regimes (10-50% labels), where physics inductive biases compensate for limited supervision. At 100% labels, Vanilla GNN slightly outperforms due to greater flexibility.

---

## LaTeX Tables Ready for Paper

### cascade_table.tex
```latex
\begin{table}[t]
\caption{Cascade Prediction SSL Transfer Results (IEEE-24, 5-seed)}
\label{tab:cascade_ssl}
\centering
\begin{tabular}{lccc}
\toprule
Label \% & Scratch & SSL & Improvement \\
\midrule
10\% & 0.773±0.015 & \textbf{0.826±0.016} & +6.8\% \\
20\% & 0.818±0.019 & \textbf{0.895±0.016} & +9.4\% \\
50\% & 0.920±0.005 & \textbf{0.940±0.008} & +2.1\% \\
100\% & 0.955±0.007 & \textbf{0.958±0.005} & +0.3\% \\
\bottomrule
\end{tabular}
\end{table}
```

### pf_table.tex
```latex
\begin{table}[t]
\caption{Power Flow Prediction SSL Transfer Results (5-seed)}
\label{tab:pf_ssl}
\centering
\begin{tabular}{lccc}
\toprule
Label \% & Scratch MAE & SSL MAE & Improvement \\
\midrule
10\% & 0.0149±0.0004 & \textbf{0.0106±0.0003} & +29.1\% \\
20\% & 0.0101±0.0004 & \textbf{0.0078±0.0001} & +23.1\% \\
50\% & 0.0056±0.0001 & \textbf{0.0048±0.0001} & +13.7\% \\
100\% & 0.0040±0.0002 & \textbf{0.0035±0.0001} & +13.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

### lineflow_table.tex
```latex
\begin{table}[t]
\caption{Line Flow Prediction SSL Transfer Results (5-seed)}
\label{tab:lineflow_ssl}
\centering
\begin{tabular}{lccc}
\toprule
Label \% & Scratch MAE & SSL MAE & Improvement \\
\midrule
10\% & 0.0084±0.0003 & \textbf{0.0062±0.0002} & +26.4\% \\
20\% & 0.0056±0.0001 & \textbf{0.0044±0.0001} & +20.5\% \\
50\% & 0.0031±0.0001 & \textbf{0.0026±0.0001} & +16.6\% \\
100\% & 0.0022±0.00002 & \textbf{0.0021±0.0005} & +2.3\% \\
\bottomrule
\end{tabular}
\end{table}
```

### cross_task_table.tex
```latex
\begin{table}[t]
\caption{Cross-Task SSL Transfer Summary (10\% Labels)}
\label{tab:crosstask}
\centering
\begin{tabular}{llccc}
\toprule
Task & Grid & Scratch & SSL & Improvement \\
\midrule
Cascade & IEEE-24 & 0.773 & \textbf{0.826} & +6.8\% \\
Cascade & IEEE-118 & 0.262 & \textbf{0.874} & +234\% \\
Power Flow & IEEE-24 & 0.0149 & \textbf{0.0106} & +29.1\% \\
Line Flow & IEEE-24 & 0.0084 & \textbf{0.0062} & +26.4\% \\
\bottomrule
\end{tabular}
\end{table}
```
