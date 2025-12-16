# II. Related Work

---

## P1: Learning-Based Power Flow and Optimal Power Flow Surrogates

Machine learning approaches for power system analysis have gained significant attention as fast alternatives to iterative solvers. Neural network surrogates for power flow [Donnot et al., 2017] and optimal power flow [Fioretto et al., 2020] can achieve order-of-magnitude speedups over Newton-Raphson methods. Recent work applies GNNs to exploit grid topology: PowerFlowNet [Donon et al., 2019] uses message passing for power flow regression, while topology-aware approaches [Owerko et al., 2020] demonstrate improved generalization to unseen grid configurations. However, these methods typically require large labeled datasets and focus on single tasks, limiting practical applicability.

---

## P2: Physics-Informed Machine Learning for Power Systems

Physics-informed neural networks (PINNs) incorporate domain knowledge as inductive biases or loss regularizers [Raissi et al., 2019]. For power systems, this includes power balance constraints [Huang et al., 2021], Kirchhoff's laws as loss terms [Pagnier & Chertkov, 2021], and admittance matrix structure [Baker et al., 2022]. Our PhysicsGuidedConv layer differs by embedding physics directly into the message-passing mechanism: learned admittance-like weights modulate information flow, allowing the model to discover relevant physics rather than enforcing hard constraints. This provides flexibility while maintaining physical intuition.

---

## P3: Self-Supervised Learning on Graphs

Self-supervised learning has revolutionized representation learning across domains. For graphs, contrastive methods like GraphCL [You et al., 2020] and GRACE [Zhu et al., 2020] learn by maximizing agreement between augmented views. Masked reconstruction approaches, inspired by BERT [Devlin et al., 2019], mask and predict node/edge features: GraphMAE [Hou et al., 2022] demonstrates strong performance across benchmarks. We adapt masked reconstruction for power grids, masking node injections (P, S) and edge parameters (X, rating) while predicting them from topology and unmasked features. Critically, we use training data only for SSL, avoiding potential label leakage through validation/test exposure.

---

## P4: Explainability for Graph Neural Networks

Explaining GNN predictions is essential for safety-critical power system applications. Gradient-based methods [Simonyan et al., 2014] and attention mechanisms [Veličković et al., 2018] provide feature importance, but can be noisy or uninformative [Jain & Wallace, 2019]. Integrated Gradients [Sundararajan et al., 2017] addresses this by accumulating gradients along a path from baseline to input, satisfying completeness and sensitivity axioms. For cascade prediction, we adapt Integrated Gradients for edge attribution and evaluate against ground-truth failure masks—a unique opportunity in power systems where simulation provides oracle explanations.

---

## LaTeX Draft

```latex
\section{Related Work}

\subsection{Learning-Based Power System Surrogates}
Machine learning approaches for power flow~\cite{donnot2017introducing,donon2019graph} and optimal power flow~\cite{fioretto2020predicting} achieve order-of-magnitude speedups over iterative solvers. Recent GNN methods~\cite{owerko2020optimal} exploit grid topology but require large labeled datasets.

\subsection{Physics-Informed Machine Learning}
Physics-informed neural networks incorporate domain knowledge as inductive biases~\cite{raissi2019physics}. For power systems, this includes power balance constraints~\cite{huang2021power} and Kirchhoff's laws~\cite{pagnier2021physics}. Our PhysicsGuidedConv embeds physics into message passing rather than loss functions.

\subsection{Self-Supervised Learning on Graphs}
Contrastive methods~\cite{you2020graph,zhu2020grace} and masked reconstruction~\cite{hou2022graphmae} enable label-free representation learning. We adapt masked reconstruction for power grids, using training data only to avoid label leakage.

\subsection{GNN Explainability}
Integrated Gradients~\cite{sundararajan2017axiomatic} provides reliable feature attribution. We evaluate edge attribution against ground-truth failure masks from simulation.
```

---

## References to Include

1. Donnot, B., et al. (2017). Introducing machine learning for power system operation support. *arXiv preprint arXiv:1709.09527*.
2. Donon, B., et al. (2019). Graph neural solver for power systems. *IJCNN*.
3. Fioretto, F., et al. (2020). Predicting AC optimal power flows: Combining deep learning and Lagrangian dual methods. *AAAI*.
4. Owerko, D., et al. (2020). Optimal power flow using graph neural networks. *ICASSP*.
5. Raissi, M., et al. (2019). Physics-informed neural networks. *Journal of Computational Physics*.
6. Huang, G., et al. (2021). Physics-informed deep learning for power flow. *IEEE Transactions on Power Systems*.
7. Pagnier, L., & Chertkov, M. (2021). Physics-informed graphical neural network for parameter & state estimations in power systems. *arXiv*.
8. You, Y., et al. (2020). Graph contrastive learning with augmentations. *NeurIPS*.
9. Zhu, Y., et al. (2020). Deep graph contrastive representation learning. *ICML Workshop*.
10. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. *NAACL*.
11. Hou, Z., et al. (2022). GraphMAE: Self-supervised masked graph autoencoders. *KDD*.
12. Sundararajan, M., et al. (2017). Axiomatic attribution for deep networks. *ICML*.
