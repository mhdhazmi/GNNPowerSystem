# GNN Architectures for Power Grid Analysis: A Literature Review

Graph Neural Networks have emerged as a transformative approach for power system analysis, offering topology-aware learning that naturally aligns with the graph structure of electrical grids. This review synthesizes **10 key publications** from 2020-2025 spanning NeurIPS, ICLR, and IEEE venues, examining how GNN architectures handle edge features, incorporate physics constraints, and scale to utility-grade networks. A critical gap emerges: while GNNs demonstrate significant advantages over fully-connected networks for grid topology adaptation, **most approaches remain purely data-driven**, with physics constraints typically relegated to loss function regularization rather than embedded in network architecture.

---

## High-quality citations (10 papers)

### 1. PowerGraph: A Power Grid Benchmark Dataset for Graph Neural Networks
**Authors:** Varbella, Amara, Gjorgiev, El-Assady, Sansavini  
**Venue/Year:** NeurIPS 2024 (Datasets and Benchmarks Track)  
**Key Contribution:** Introduces the first comprehensive GNN-tailored benchmark for power grids, comprising power flow, optimal power flow, and cascading failure datasets with ground-truth explanations. Benchmarks GCN, GAT, GraphSAGE, and Graph Transformer architectures across node-level and graph-level tasks on IEEE 24, 39, and 118-bus systems.

### 2. Adversarially Robust Learning for Security-Constrained Optimal Power Flow
**Authors:** Donti, Agarwal, Bedmutha, Pileggi, Kolter  
**Venue/Year:** NeurIPS 2021  
**Key Contribution:** Frames N-k security-constrained OPF as minimax optimization, using implicit differentiation through AC power flow equations. Demonstrates tractable N-3 SCOPF—previously considered computationally prohibitive—on IEEE 57 and 118-bus systems by treating equipment outages as adversarial perturbations.

### 3. DC3: A Learning Method for Optimization with Hard Constraints
**Authors:** Donti, Rolnick, Kolter  
**Venue/Year:** ICLR 2021  
**Key Contribution:** Presents Deep Constraint Completion and Correction, enforcing AC-OPF feasibility via differentiable procedures that complete equality constraints and correct inequality violations. Demonstrates that neural networks can produce feasible AC-OPF solutions on IEEE 57-bus while maintaining computational efficiency.

### 4. Predicting Dynamic Stability of Power Grids with Graph Neural Networks
**Authors:** Anonymous (OpenReview)  
**Venue/Year:** ICLR 2023  
**Key Contribution:** Applies GNNs to predict single-node basin stability using Kuramoto-type swing equations. Compares ArmaNet, GCN, GraphSAGE, and TAGNet on 10,000+ synthetic grids, demonstrating transfer learning from 20-node training grids to a **1,910-node synthetic Texas grid model** with R² > 88%.

### 5. Meta-Learning Enhanced Physics-Informed Graph Attention Convolutional Network (Meta-PIGACN)
**Authors:** Wu, Xu, Wang, Lyu (PNNL)  
**Venue/Year:** IEEE Transactions on Network Science and Engineering, 2025  
**Key Contribution:** Integrates physics information directly into GAT architecture through **impedance-weighted edge aggregation**. Uses meta-learning for rapid adaptation to topological changes in distribution system state estimation, tested on IEEE 33 and 118-node systems.

### 6. Complex-Value Spatiotemporal Graph Convolutional Neural Networks for Power Systems
**Authors:** Wu, Scaglione, Arnold  
**Venue/Year:** IEEE Transactions on Smart Grid, 2024  
**Key Contribution:** Introduces complex-valued GCN specifically designed for AC power systems, handling voltage phasors and power flows in their native complex form (R+jX impedance, G+jB admittance). Maintains phase relationships inherent to AC power flow physics throughout network operations.

### 7. Power-GNN: Graph Neural Networks for Power Flow Estimation
**Authors:** Pagnier, Chertkov  
**Venue/Year:** arXiv 2021 (widely cited)  
**Key Contribution:** Hybrid physics-ML scheme that learns **effective admittance matrices** as interpretable parameters alongside GNN weights. Demonstrates that physics-aware GNNs outperform vanilla neural networks on IEEE 14, 118-bus, and PanTaGruEl (1000+ loads) by providing physically meaningful parameter estimates.

### 8. Neural Networks for Power Flow: Graph Neural Solver
**Authors:** Donon, Clément, Donnot, Marot, Guyon, Schoenauer  
**Venue/Year:** Electric Power Systems Research, 2020  
**Key Contribution:** Pioneering **unsupervised physics-informed** approach that trains by minimizing Kirchhoff's law violations rather than imitating Newton-Raphson outputs. Achieves linear time complexity and robust generalization across IEEE 9, 14, 30, and 118-bus topologies without requiring labeled power flow solutions.

### 9. PINCO: Physics-Informed GNN for AC-OPF
**Authors:** Multiple  
**Venue/Year:** ICLR 2025 (under review)  
**Key Contribution:** End-to-end **unsupervised** physics-informed GNN achieving **zero inequality constraint violations** through hard constraint enforcement in architecture (H-PINN framework). Solves AC-OPF faster than traditional nonlinear solvers on IEEE 9, 24, 30, and 118-bus without labeled data from conventional solvers.

### 10. KCLNet: Physics-Informed Power Flow with Hard Kirchhoff Constraints
**Authors:** Dogoulis et al.  
**Venue/Year:** arXiv 2025  
**Key Contribution:** Enforces Kirchhoff's Current Law as **hard architectural constraints** via differentiable hyperplane projections. Guarantees zero KCL violations by construction—attention-based architecture with projection layers produces physically consistent solutions on IEEE 14 and 118-bus systems.

---

## Summary: graph representation and GNN architectures

Power grid GNN research has converged on representing buses as graph nodes (with features including voltage magnitude V, angle θ, active power P, reactive power Q, and bus type encoding) and transmission lines as edges (with resistance R, reactance X, and derived admittance Y_ij as edge attributes). **GCN and GAT remain the baseline architectures**, though GraphSAGE demonstrates superior scalability for larger systems due to its neighborhood sampling approach. More sophisticated architectures have emerged: heterogeneous GNNs (HH-MPNN) model different node types—generators, loads, buses, shunts—with distinct edge types for lines versus transformers, enabling zero-shot generalization to N-1 contingencies across 14 to 2,000+ bus systems. Edge feature handling has evolved from simple adjacency encoding to physics-informed approaches: Meta-PIGACN uses impedance-weighted aggregation where line admittance directly controls message passing strength, while Wu et al.'s complex-valued STGCN represents impedance in native complex form to preserve AC power system phase relationships. The PowerGraph benchmark (NeurIPS 2024) now provides standardized evaluation across GCN, GAT, GraphSAGE, and Graph Transformer architectures, establishing that GraphSAGE achieves best overall performance with lower NRMSE and higher R² across varied grid sizes.

## Summary: advantages over FCN and the physics-guided gap

GNNs demonstrate **fundamental structural advantages** over fully-connected networks for power grid applications. Empirical studies show GNNs achieving **5× lower voltage prediction error** than FCNs on the NREL-118 system, primarily because FCNs treat inputs as flat vectors and cannot adapt to topology changes without complete retraining. GNNs naturally handle N-1 contingencies through message passing that mirrors actual power flow physics—the conductance G_ij governs active power transfer while susceptance B_ij governs reactive power, directly paralleling how each bus aggregates information from connected neighbors. HH-MPNN achieves **10³–10⁴× speedup** over interior-point solvers while maintaining <3% optimality gap on unseen contingencies. However, **a critical gap persists: most GNN approaches remain purely data-driven or use only soft physics constraints** (loss function regularization). Of the surveyed literature, only three papers—KCLNet, PINCO, and GraPhyR—implement hard physics constraints directly in network architecture. Furthermore, **self-supervised learning for power grids remains significantly underexplored**: only Donon et al. (2020) and PINCO demonstrate fully unsupervised training via physics losses, while no published work combines self-supervised pretraining with physics-guided message passing. This represents a significant opportunity for approaches that embed admittance-weighted aggregation as architectural priors and leverage SSL to learn grid representations without requiring expensive labeled data from traditional solvers.

---

## Technical findings summary

| Architecture | Edge Feature Handling | Physics Constraints | Largest Grid Tested |
|-------------|----------------------|---------------------|---------------------|
| GCN/GAT | Adjacency or learned | Soft (loss) | IEEE 118-bus |
| GraphSAGE | Impedance attributes | Soft (loss) | 1,910 nodes (Texas) |
| HH-MPNN | Heterogeneous edges | Implicit topology | 2,000+ buses |
| Meta-PIGACN | Impedance-weighted | Architecture-embedded | IEEE 118-node |
| Complex STGCN | Complex Z/Y values | Native AC physics | Various IEEE |
| KCLNet | Attention + projection | Hard KCL constraints | IEEE 118-bus |
| PINCO | Physics-informed | Hard constraints (H-PINN) | IEEE 118-bus |

---

## Positioning for physics-guided self-supervised GNN research

This literature review reveals a clear positioning opportunity: **no existing work combines admittance-weighted message passing with self-supervised pretraining**. Current physics-informed approaches either (1) use soft constraints in loss functions while requiring labeled solver outputs, or (2) implement hard constraints but rely on supervised learning. The proposed "Physics-Guided Self-Supervised Graph Neural Networks for Power Grid Analysis" can differentiate by: embedding the admittance matrix Y directly into message passing weights (following Meta-PIGACN's direction but extending to SSL), enforcing Kirchhoff's laws as structural priors rather than loss penalties, and demonstrating that SSL pretraining reduces labeled data requirements while maintaining physical consistency—addressing the data scarcity challenge that limits deployment of learning-based power flow methods in real utility environments.


Graph Neural Networks for Power Grids
Summary

Power grids are naturally represented as graphs, with buses as nodes and transmission lines as edges
arxiv.org
. Graph Neural Networks (GNNs) have been widely applied to exploit this structure for power system tasks, including power flow prediction
arxiv.org
, state estimation
web.uri.edu
, cascading failure risk analysis
proceedings.neurips.cc
, and optimal power flow approximation. Common architectures like Graph Convolutional Networks (GCNs), GraphSAGE, and Graph Attention Networks (GATs) propagate information along electrical connectivity, incorporating features such as line impedances or capacities as edge attributes in the learning process. By encoding the grid’s topology, GNN models can learn localized interactions (e.g. neighboring buses’ voltage or power influence) that fully-connected networks would otherwise have to relearn for each new connection. This topological awareness leads to better accuracy and generalization compared to traditional fully-connected or convolutional neural nets that ignore graph structure
web.uri.edu
. For example, an electrical-model-guided GNN for distribution state estimation achieved an order-of-magnitude lower error than a standard neural network even with missing sensor data
web.uri.edu
. Likewise, a geometric deep learning approach outperformed a feedforward network for predicting cascading outages
proceedings.neurips.cc
. GNNs also tend to use far fewer parameters by sharing weights over the graph, enabling greater scalability to large networks than fully-connected models.

Recent research has increasingly infused power system physics into GNN architectures. Some approaches impose Kirchhoff’s laws as explicit constraints – for instance, the KCLNet model enforces exact current balance at each node (Kirchhoff’s Current Law) by projecting GNN outputs onto a subspace satisfying KCL
arxiv.org
. Others incorporate the AC power flow equations into the training objective to guide the GNN to physically feasible solutions
mdpi.com
mdpi.com
. Specialized message-passing schemes have been designed to handle edge electrical parameters: one line of work builds a line-graph GNN that treats each transmission line as a node in a dual graph, so that voltage drop and flow computations can explicitly obey network physics. These physics-informed GNNs aim to improve reliability by ensuring outputs (voltages, flows, etc.) remain consistent with circuit laws. In terms of scalability, GNN models have demonstrated the ability to handle large-scale grids with thousands of buses. Notably, a recent GNN-based power flow solver achieved accuracy on par with the Newton–Raphson method for a 6470-bus French system, while being over 100× faster in computation
arxiv.org
. Gap: Despite these advances, most proposed GNN solutions for power grids are still largely data-driven and do not inherently guarantee physical correctness
arxiv.org
. They learn from historical or simulated data and may violate constraints under unseen conditions. This highlights a need for more physics-guided GNN frameworks that combine data-driven learning with power system domain knowledge (e.g. admittance matrices and Kirchhoff’s laws) to ensure robust, trustworthy performance in real-world grid operations.

References
@article{Varbella2023,
  author    = {Anna Varbella and Blazhe Gjorgiev and Giovanni Sansavini},
  title     = {Geometric deep learning for online prediction of cascading failures in power grids},
  journal   = {Reliability Engineering {\&} System Safety},
  volume    = {237},
  pages     = {109341},
  year      = {2023}
}

@inproceedings{Lin2022EleGNN,
  author    = {Hui Lin and Yan Sun},
  title     = {{EleGNN}: Electrical-model-guided graph neural networks for power distribution system state estimation},
  booktitle = {Proc. IEEE Global Communications Conference (GLOBECOM)},
  pages     = {5292--5298},
  year      = {2022}
}

@inproceedings{Tuo2023,
  author    = {Mingjian Tuo and Xingpeng Li and Tianxia Zhao},
  title     = {Graph Neural Network-Based Power Flow Model},
  booktitle = {North American Power Symposium (NAPS)},
  year      = {2023}
}

@article{Zhang2024,
  author    = {Hai-Feng Zhang and Xin-Long Lu and Xiao Ding and Xiaoming Zhang},
  title     = {Physics-informed line graph neural network for power flow calculation},
  journal   = {Chaos},
  volume    = {34},
  number    = {11},
  pages     = {113123},
  year      = {2024},
  doi       = {10.1063/5.0235301}
}

@article{Zhai2024,
  author    = {Baitong Zhai and Dongsheng Yang and Bowen Zhou and Guangdi Li},
  title     = {Distribution system state estimation based on power flow-guided GraphSAGE},
  journal   = {Energies},
  volume    = {17},
  number    = {17},
  pages     = {4317},
  year      = {2024},
  doi       = {10.3390/en17174317}
}

@article{Dogoulis2025,
  author    = {Pantelis Dogoulis and Karim Tit and Maxime Cordy},
  title     = {{KCLNet}: Physics-informed power flow prediction via constraints projections},
  journal   = {arXiv preprint arXiv:2506.12902},
  year      = {2025}
}

@article{Yang2024,
  author    = {Mei Yang and Gao Qiu and Tingjian Liu and Junyong Liu and Kai Liu and Yaping Li},
  title     = {Probabilistic power flow based on physics-guided graph neural networks},
  journal   = {Electric Power Systems Research},
  volume    = {235},
  pages     = {110864},
  year      = {2024},
  doi       = {10.1016/j.epsr.2023.110864}
}

@article{Lin2024,
  author    = {Nan Lin and Stavros Orfanoudakis and Nathan Ordonez Cardenas and Juan S. Giraldo and Pedro P. Vergara},
  title     = {{PowerFlowNet}: Power flow approximation using message passing graph neural networks},
  journal   = {IEEE Transactions on Smart Grid (early access)},
  year      = {2024}
}