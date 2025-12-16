# I. Introduction

---

## P1: Motivation and Stakes

Power grid reliability is critical infrastructure: cascading failures can propagate within seconds, causing widespread blackouts affecting millions. Grid operators require fast screening tools for contingency analysis—traditional power flow solvers and Monte Carlo simulations are computationally expensive, limiting real-time decision support. Simultaneously, obtaining labeled training data for rare failure events is costly: each labeled scenario requires expensive physics simulations or post-hoc analysis of real outages. This creates a fundamental tension between the need for data-driven approaches and the scarcity of labeled examples.

---

## P2: Why GNNs, and the Gap

Power grids are naturally represented as graphs: buses (substations) are nodes, and transmission lines are edges. Graph Neural Networks (GNNs) can exploit this topology, learning representations that respect the grid's physical structure. However, purely supervised GNNs face critical limitations: (1) they are label-hungry, requiring thousands of labeled scenarios for robust training; (2) they exhibit training instability in rare-event regimes where positive examples (cascades) are scarce; and (3) they fail to generalize when operating conditions shift from training distributions. Our experiments on IEEE-118 cascade prediction demonstrate this directly: scratch-trained GNNs achieve F1 = 0.262 ± 0.243 at 10% labels—high variance indicating some seeds learn while others fail entirely.

---

## P3: Proposed Approach

We propose a **physics-guided self-supervised learning** framework addressing these challenges. The core innovation is a **PhysicsGuidedEncoder** incorporating admittance-weighted message passing: messages between nodes are weighted by learned line importance, mimicking how power actually flows according to Kirchhoff's laws. This physics-informed architecture is pretrained via **masked reconstruction** on unlabeled grid topologies, learning representations that capture power system structure without any task-specific labels. The pretrained encoder then transfers to three downstream tasks through task-specific heads: graph-level cascade prediction, node-level power flow estimation (V_mag), and edge-level line flow prediction (P_ij, Q_ij).

---

## P4: Contributions

Our main contributions are:

- **Physics-guided message passing mechanism**: An admittance-inspired weighting scheme that embeds power flow physics into GNN aggregation, improving low-data performance.

- **Grid-specific SSL pretraining objective**: Masked reconstruction on node injections and edge parameters, using training data only (no validation/test exposure), enabling label-free representation learning.

- **Multi-task transfer with consistent gains**: Demonstrated SSL improvements across three distinct tasks (cascade, power flow, line flow) in low-label regimes (+6.8% to +29.1% at 10% labels).

- **Scalability and stability**: SSL stabilizes training on larger grids (IEEE-118), reducing variance 5× and enabling learning where scratch training fails.

- **Explainability fidelity**: Integrated Gradients edge attribution achieves 0.93 AUC-ROC against ground-truth failure edges, demonstrating physically meaningful learned representations.

- **Robustness under distribution shift**: Preliminary single-seed results show SSL advantage increases to +22% at 1.3× out-of-distribution load conditions.

---

## P5: Paper Roadmap

The remainder of this paper is organized as follows. Section II reviews related work on learning-based power system analysis, physics-informed machine learning, and self-supervised learning on graphs. Section III formalizes the problem setup, defining the graph representation and three downstream tasks. Section IV presents our method: the PhysicsGuidedConv layer, encoder architecture, SSL pretraining objective, and explainability approach. Section V details the experimental setup including datasets, evaluation protocol, and baselines. Section VI presents comprehensive results with multi-seed validation, cross-task analysis, and supporting evidence from robustness and explainability evaluations. Section VII discusses implications and limitations. Section VIII concludes.

---

## LaTeX Draft

```latex
\section{Introduction}

% P1: Motivation
Power grid reliability is critical infrastructure: cascading failures can propagate within seconds, causing widespread blackouts affecting millions. Grid operators require fast screening tools for contingency analysis—traditional power flow solvers and Monte Carlo simulations are computationally expensive, limiting real-time decision support. Simultaneously, obtaining labeled training data for rare failure events is costly: each labeled scenario requires expensive physics simulations or post-hoc analysis of real outages.

% P2: Gap
Power grids are naturally represented as graphs, making Graph Neural Networks (GNNs) an appealing approach. However, purely supervised GNNs face critical limitations: they are label-hungry, exhibit training instability in rare-event regimes, and fail to generalize under distribution shift. Our experiments on IEEE-118 cascade prediction demonstrate this directly: scratch-trained GNNs achieve F1 = $0.262 \pm 0.243$ at 10\% labels—high variance indicating some seeds learn while others fail entirely.

% P3: Approach
We propose a physics-guided self-supervised learning framework. Our \textbf{PhysicsGuidedEncoder} incorporates admittance-weighted message passing, and SSL pretraining via masked reconstruction enables transfer to cascade prediction, power flow, and line flow tasks.

% P4: Contributions
Our contributions include: (1) physics-guided message passing embedding power flow physics; (2) grid-specific SSL pretraining using training data only; (3) multi-task transfer with +6.8\% to +29.1\% improvement at 10\% labels; (4) 5$\times$ variance reduction on IEEE-118; (5) 0.93 AUC-ROC explainability fidelity.

% P5: Roadmap
Section~II reviews related work. Section~III formalizes the problem. Section~IV presents our method. Section~V details experimental setup. Section~VI presents results. Section~VII discusses implications. Section~VIII concludes.
```
