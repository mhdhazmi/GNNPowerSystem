# Title, Abstract, and Index Terms

---

## Title

**Physics-Guided Self-Supervised Graph Neural Networks for Power Grid Analysis: Transfer Across Cascade Prediction, Power Flow, and Line Flow**

---

## Abstract

Modern power grid operations require fast, accurate solutions to computationally expensive problems—power flow analysis, optimal dispatch, and rare-event prediction—while facing chronic labeled data scarcity and distribution shift challenges. We propose a physics-guided Graph Neural Network (GNN) encoder with self-supervised learning (SSL) pretraining that learns transferable grid representations from unlabeled topology data. Our PhysicsGuidedConv layer incorporates admittance-weighted message passing, embedding power flow physics directly into the architecture. SSL pretraining via masked reconstruction on the training set alone enables effective transfer to three downstream tasks: cascade failure prediction, power flow estimation, and line flow prediction.

Extensive multi-seed experiments on IEEE 24-bus and 118-bus benchmarks demonstrate consistent SSL benefits: **+29.1% MAE improvement** for power flow and **+26.4%** for line flow prediction at 10% labeled data. On cascade prediction, SSL stabilizes training on the larger IEEE-118 grid, reducing variance from σ=0.243 (Scratch) to σ=0.051 (SSL) at 10% labels, with absolute F1 improvement ΔF1=+0.61. Explainability evaluation via Integrated Gradients achieves 0.93 AUC-ROC edge attribution fidelity, and preliminary robustness tests show SSL advantage increases to +22% under 1.3× out-of-distribution load conditions.

---

## Index Terms

Power systems, graph neural networks, self-supervised learning, cascading failures, power flow, explainability, robustness.

---

## Key Results Summary (for reference)

| Task | Grid | Improvement at 10% Labels |
|------|------|---------------------------|
| Power Flow | IEEE-24 | +29.1% MAE reduction |
| Line Flow | IEEE-24 | +26.4% MAE reduction |
| Cascade Prediction | IEEE-24 | +6.8% F1 improvement |
| Cascade Prediction | IEEE-118 | ΔF1=+0.61 (0.26→0.87), σ reduction 5× |

*All results are 5-seed validated with mean ± std reporting.*
