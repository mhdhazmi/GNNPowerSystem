# VII. Discussion

---

## P1: Why SSL Helps Most at Extreme Low-Label Regimes

The consistent pattern of largest gains at 10-20% labels suggests SSL addresses a fundamental challenge in supervised learning: **representation initialization**. With random initialization, GNNs must simultaneously learn (1) how to extract useful features from raw inputs and (2) the task-specific mapping from features to outputs. When labels are scarce, this joint optimization often fails to converge to good solutions.

SSL pretraining decouples these objectives: the encoder first learns general-purpose grid representations through self-supervised reconstruction, then the downstream task only needs to learn the (simpler) mapping from pretrained features to task outputs. This is particularly valuable for power systems where:
- Labeled data requires expensive simulations
- Rare events (cascades) are underrepresented
- Grid topology contains rich structural information exploitable without labels

The IEEE-118 results dramatically illustrate this: scratch training with 10% labels produces F1 = 0.262 ± 0.243 (essentially random for half the seeds), while SSL achieves F1 = 0.874 ± 0.051. The pretrained encoder provides a consistent starting point from which any seed can learn.

---

## P2: Operational Implications and Deployment Considerations

Our observability analysis (Table C) confirms all model inputs are available from standard SCADA/PMU infrastructure. This enables practical deployment without requiring oracle information:

- **Cascade prediction**: Uses pre-outage measurements to estimate post-outage risk
- **Power flow**: Predicts voltage from injection measurements, enabling state estimation surrogates
- **Line flow**: Predicts unmeasured flows from measured bus states, supporting observability enhancement

The per-task feature schema (Section V) explicitly prevents target leakage: power flow models cannot access voltage as input, line flow models cannot access edge flows, ensuring the learned representations are truly predictive rather than merely copying inputs.

**Computational Benefits**: Once trained, GNN inference is orders of magnitude faster than Newton-Raphson power flow iteration, enabling real-time contingency screening at scale.

---

## P3: Scalability and Grid Generalization

The IEEE-24 → IEEE-118 comparison provides evidence of scalability, but with important caveats:

**Positive Findings:**
- SSL benefits transfer to larger grid (IEEE-118)
- Stabilization effect is more pronounced on the harder setting
- Same architecture/hyperparameters work across scales

**Limitations:**
- Both grids are from the same benchmark suite with similar data generation
- We do not evaluate transfer across grid topologies (e.g., train on IEEE-24, test on IEEE-118)
- Real transmission networks may have different characteristics

Future work should evaluate **cross-grid transfer**: pretraining on one grid family and fine-tuning on another, testing whether physics-guided representations truly capture universal power system principles.

---

## P4: Limitations and Future Directions

**Current Limitations:**

1. **Single benchmark**: All experiments use PowerGraph benchmark; validation on real utility data would strengthen claims
2. **Single-seed robustness**: OOD load scaling results are preliminary (single seed); multi-seed evaluation needed before claiming robustness as a headline contribution
3. **Limited task diversity**: Three tasks demonstrated; extension to protection coordination, topology optimization, and market-aware dispatch would broaden impact
4. **No cross-grid transfer**: Models are trained and tested on same grid topology; cross-topology generalization is unexplored

**Future Directions:**

1. **Multi-grid pretraining**: SSL on diverse grid topologies to learn universal representations
2. **Temporal modeling**: Extend to time-series prediction (load forecasting, state evolution)
3. **Physics constraints**: Combine SSL with hard physics constraints (power balance, Kirchhoff's laws) as loss regularizers
4. **Uncertainty quantification**: Augment predictions with calibrated uncertainty estimates for risk-aware decision support
5. **Real-world validation**: Partner with utilities for deployment on actual transmission networks

---

## LaTeX Draft

```latex
\section{Discussion}

\paragraph{Why SSL Helps at Low Labels}
SSL decouples representation learning from task-specific mapping. With random initialization, GNNs must jointly learn both, which fails when labels are scarce. SSL pretraining provides consistent initialization from which any seed can learn. This is dramatically illustrated on IEEE-118: scratch achieves F1 = $0.262 \pm 0.243$ at 10\% labels (random for half the seeds), while SSL achieves $0.874 \pm 0.051$.

\paragraph{Operational Implications}
All model inputs are available from standard SCADA/PMU infrastructure. GNN inference is orders of magnitude faster than Newton-Raphson iteration, enabling real-time contingency screening.

\paragraph{Scalability}
SSL benefits transfer from IEEE-24 to IEEE-118, with stabilization more pronounced on the harder setting. However, we do not evaluate cross-grid transfer (train on one topology, test on another).

\paragraph{Limitations}
(1) Single benchmark—validation on real utility data needed; (2) single-seed robustness results—multi-seed evaluation required; (3) no cross-grid transfer evaluation. Future work should explore multi-grid pretraining, temporal modeling, and real-world deployment.
```
