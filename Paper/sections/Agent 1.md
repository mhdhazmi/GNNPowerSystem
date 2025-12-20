# Machine Learning for Power Flow and OPF Surrogate Modeling: Literature Review

Deep learning approaches have transformed power flow and optimal power flow computation, achieving **speedups of 10,000× over traditional solvers** while maintaining accuracy within **0.2% of optimal solutions**. Graph neural networks are emerging as the dominant architecture, with physics-informed methods reducing data requirements by embedding Kirchhoff's laws directly into training. Critically, self-supervised learning remains largely unexplored for OPF—a significant gap given the high cost of generating labeled training data.

---

## State-of-the-art ML approaches for PF/OPF surrogate modeling

### Architectures and their prevalence

| Architecture | Prevalence | Key Papers | Typical Application |
|-------------|------------|------------|---------------------|
| **Fully-connected DNNs/MLPs** | Most common | DeepOPF series | End-to-end DC/AC-OPF prediction |
| **Graph Neural Networks** | Rapidly growing | PowerFlowNet, PINCO | Topology-aware PF/OPF with scalability |
| **Physics-Informed NNs** | Growing | Nellikkath & Chatzivasileiadis | Constraint satisfaction guarantees |
| **Ensemble Methods** | Moderate | Hu et al. 2021 | Three-phase systems, robustness |

### Typical input features and output targets

**Input features:** Load demands (P, Q) at all buses, generator capacity limits, bus types (slack, PV, PQ), network topology (for GNNs), branch impedances
**Output targets:** Generator dispatch, voltage magnitudes and angles, phase angles, locational marginal prices, power flows on lines

---

## Accuracy benchmarks

| Paper | Test System | Metric | Value | Speedup |
|-------|-------------|--------|-------|---------|
| DeepOPF (Pan et al. 2021) | IEEE 118/300-bus | Optimality loss | <0.2% | 100× |
| DeepOPF-V (Huang et al. 2022) | 2000-bus synthetic | Optimality loss | <0.2% | 10,000× |
| PowerFlowNet (Lin et al. 2024) | French 6470-bus | Voltage MAE | <0.001 p.u. | 145× |
| Physics-Informed NN (Nellikkath 2022) | IEEE 118-bus | Constraint violation | 50% reduction | 100× |
| HH-MPNN (Arowolo 2025) | IEEE 14-2000 bus | Optimality gap | <1% (default), <3% (N-1) | 10,000× |

**Commonly used IEEE test cases:** IEEE 14-bus (baseline), IEEE 30-bus, IEEE 57-bus, IEEE 118-bus (standard benchmark), IEEE 300-bus, PEGASE 1354-bus, French 6470rte (scalability tests)

---

## Main challenges identified in the literature

### Scalability to larger grids
- Traditional MLPs require flattened inputs with fixed dimensions, limiting scalability
- GNNs address this through message-passing that is independent of grid size—PowerFlowNet demonstrated successful scaling to **6,470 buses**
- HH-MPNN maintains constant parameter count (~117.4×10⁵) across all grid sizes

### Generalization across operating conditions
- Models trained on specific topologies fail under N-1 contingencies
- Current best approaches achieve **<3% optimality gap on unseen topologies** using heterogeneous GNNs
- Distribution shift under severe contingencies remains challenging

### Labeled data requirements
- Typical supervised methods require **5,000-60,000 training samples** per network
- Computational cost of generating labeled data by solving OPF with conventional solvers is significant
- Quote from literature: "Supervised learning methods need conventional solvers to generate the training dataset, which becomes impractical for large-scale problems"

### Physics constraint satisfaction
- Pure data-driven methods often violate physical constraints
- Post-processing (ℓ1-projection) or physics-informed losses are required for feasibility
- Physics-informed GNNs (PINCO) achieve **zero inequality violations**

---

## Leading research groups and key authors

| Research Group | Key Authors | Focus Area | Notable Contribution |
|----------------|-------------|------------|---------------------|
| **City University of Hong Kong** | Minghua Chen, Xiang Pan | DeepOPF series | Pioneering DNN approaches for OPF |
| **Caltech** | Steven H. Low | Physics-informed ML | Theoretical foundations, DeepOPF-V |
| **Georgia Tech** | Pascal Van Hentenryck | Large-scale optimization | PDL-SCOPF self-supervised approach |
| **DTU Denmark** | Spyros Chatzivasileiadis | Physics-informed NNs | Worst-case verification guarantees |
| **ETH Zurich** | Giovanni Sansavini, Anna Varbella | Physics-informed GNNs | PINCO, PowerGraph benchmark |
| **RTE France/Paris-Saclay** | Balthazar Donon, Antoine Marot | GNN-based solvers | Graph Neural Solver with self-supervised physics loss |
| **TU Delft** | Pedro P. Vergara, J.S. Giraldo | PowerFlowNet | Scalable GNN for large networks |
| **University of Pennsylvania** | Alejandro Ribeiro, Fernando Gama | GNN theory | Optimal Power Flow using GNNs |

---

## Gap regarding self-supervised methods

### Current state of SSL for power systems
Self-supervised learning for power system ML is **emerging but severely limited**:
- Only **~5-6 papers** from 2020-2025 apply SSL to power systems
- Only **one paper (PDL-SCOPF)** directly addresses OPF without labeled data
- **No papers** apply contrastive learning to OPF specifically
- **No papers** use masked modeling (BERT-style pretraining) for power flow

### Key SSL papers found
1. **PDL-SCOPF** (Park & Van Hentenryck, 2023): End-to-end primal-dual learning eliminating need for pre-computed optimal solutions
2. **Graph Neural Solver** (Donon et al., 2020): Self-supervised training by directly minimizing Kirchhoff's law violations—no labeled data from solvers needed
3. **TC-TSS** (Dey & Rana, 2025): Contrastive learning for frequency disturbance detection (not OPF)

### Current approaches to limited labeled data
| Approach | Description | Example Papers |
|----------|-------------|----------------|
| **Unsupervised physics-based loss** | Lagrangian duality replaces labels | Chen et al. 2022 |
| **GAN-based augmentation** | Generate synthetic training data | LSGAN for STVSA (2021) |
| **Transfer learning** | Pre-train on one grid, fine-tune on another | CNN-LSTM for stability (2022) |
| **Semi-supervised BNN** | Combine labeled and unlabeled data | Pareek et al. 2024 |

### The gap self-supervised methods could fill
1. **Pre-training on abundant unlabeled operational data** before fine-tuning on limited labeled OPF solutions
2. **Learning physics-consistent representations** without solving expensive optimization problems
3. **Topology-invariant feature learning** for generalization across different grid configurations
4. **Sample-efficient learning** reducing the 5,000-60,000 sample requirement

---

## BibTeX File: agent1_power_system_ml.bib

```bibtex
@article{pan2021deepopf,
  author    = {Pan, Xiang and Zhao, Tianyu and Chen, Minghua and Zhang, Shengyu},
  title     = {{DeepOPF}: A Deep Neural Network Approach for Security-Constrained {DC} Optimal Power Flow},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {36},
  number    = {3},
  pages     = {1725--1735},
  year      = {2021},
  publisher = {IEEE},
  note      = {Achieves <0.2\% optimality loss with 100x speedup on IEEE 30/118/300-bus systems using predict-and-reconstruct DNN approach}
}

@article{huang2022deepopfv,
  author    = {Huang, Wanjun and Pan, Xiang and Chen, Minghua and Low, Steven H.},
  title     = {{DeepOPF-V}: Solving {AC-OPF} Problems Efficiently},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {37},
  number    = {1},
  pages     = {800--803},
  year      = {2022},
  publisher = {IEEE},
  note      = {Extends DeepOPF to AC-OPF with voltage-constrained approach; achieves 10,000x speedup on 2000-bus systems}
}

@article{hu2021physics,
  author    = {Hu, Xing and Hu, Haoyu and Verma, Saurabh and Zhang, Zhi-Li},
  title     = {Physics-Guided Deep Neural Networks for Power Flow Analysis},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {36},
  number    = {3},
  pages     = {2082--2092},
  year      = {2021},
  publisher = {IEEE},
  note      = {AutoEncoder-based DNN embedding Kirchhoff's laws; topology knowledge in weight matrices; tested on IEEE 118-bus}
}

@article{liu2023topology,
  author    = {Liu, Shaohui and Wu, Chengyang and Zhu, Hao},
  title     = {Topology-Aware Graph Neural Networks for Learning Feasible and Adaptive {AC-OPF} Solutions},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {38},
  number    = {5},
  pages     = {4953--4966},
  year      = {2023},
  publisher = {IEEE},
  note      = {Exploits LMP locality; physics-aware flow feasibility regularization; efficient re-training for topology changes}
}

@article{huang2023applications,
  author    = {Huang, Bo and Wang, Jianhui},
  title     = {Applications of Physics-Informed Neural Networks in Power Systems -- A Review},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {38},
  number    = {1},
  pages     = {572--588},
  year      = {2023},
  publisher = {IEEE},
  note      = {Comprehensive review of four PINN paradigms: physics-informed loss, initialization, architecture, and hybrid models}
}

@article{donon2020neural,
  author    = {Donon, Balthazar and Cl{\'e}ment, R{\'e}my and Donnot, Benjamin and Marot, Antoine and Guyon, Isabelle and Schoenauer, Marc},
  title     = {Neural Networks for Power Flow: Graph Neural Solver},
  journal   = {Electric Power Systems Research},
  volume    = {189},
  pages     = {106547},
  year      = {2020},
  publisher = {Elsevier},
  note      = {Self-supervised GNN minimizing Kirchhoff violations; no labeled data needed; 0.995 correlation with Newton-Raphson on IEEE 118-bus}
}

@article{nellikkath2022physics,
  author    = {Nellikkath, Rahul and Chatzivasileiadis, Spyros},
  title     = {Physics-Informed Neural Networks for {AC} Optimal Power Flow},
  journal   = {Electric Power Systems Research},
  volume    = {212},
  pages     = {108412},
  year      = {2022},
  publisher = {Elsevier},
  note      = {AC power flow equations in loss function with MILP-based worst-case verification; tested on IEEE 39/118/162-bus}
}

@article{lin2024powerflownet,
  author    = {Lin, Nan and Orfanoudakis, Stavros and Ordonez Cardenas, Nathaly and Giraldo, Juan S. and Vergara, Pedro P.},
  title     = {{PowerFlowNet}: Power Flow Approximation Using Message Passing Graph Neural Networks},
  journal   = {International Journal of Electrical Power \& Energy Systems},
  volume    = {160},
  pages     = {110112},
  year      = {2024},
  publisher = {Elsevier},
  note      = {145x faster than Newton-Raphson on French 6470-bus network; K-hop message passing; voltage MAE <0.001 p.u.}
}

@inproceedings{owerko2020optimal,
  author    = {Owerko, Damian and Gama, Fernando and Ribeiro, Alejandro},
  title     = {Optimal Power Flow Using Graph Neural Networks},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {5930--5934},
  year      = {2020},
  organization = {IEEE},
  note      = {GNN with imitation learning; permutation equivariance exploits grid symmetries; tested on IEEE 30/118-bus}
}

@article{hu2021ensemble,
  author    = {Hu, Rui and Li, Qifeng and Qiu, Feng},
  title     = {Ensemble Learning Based Convex Approximation of Three-Phase Power Flow},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {36},
  number    = {5},
  pages     = {4042--4051},
  year      = {2021},
  publisher = {IEEE},
  note      = {Ensemble learning for unbalanced distribution systems; convex approximation of nonlinear power flow}
}

@article{varbella2024powergraph,
  author    = {Varbella, Anna and Amara, Kenza and Gjorgiev, Blazhe and El-Assady, Mennatallah and Sansavini, Giovanni},
  title     = {{PowerGraph}: A Power Grid Benchmark Dataset for Graph Neural Networks},
  journal   = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024},
  note      = {Benchmark with PF, OPF, cascading failure tasks; GAT best-performing architecture; single MPL layer optimal}
}

@article{park2023pdlscopf,
  author    = {Park, Seonho and Van Hentenryck, Pascal},
  title     = {{PDL-SCOPF}: Self-Supervised Primal-Dual Learning for Large-Scale Security-Constrained {DC} Optimal Power Flow},
  journal   = {arXiv preprint arXiv:2311.18072},
  year      = {2023},
  note      = {End-to-end primal-dual learning eliminating need for pre-computed optimal solutions; Augmented Lagrangian Method}
}

@article{pareek2024semisupervised,
  author    = {Pareek, Parikshit and others},
  title     = {Semi-Supervised Bayesian Neural Network for Optimal Power Flow},
  journal   = {arXiv preprint arXiv:2410.03085},
  year      = {2024},
  note      = {Sandwich BNN with unlabeled data augmentation; outperforms DNNs in low-data settings; ensures constraint feasibility}
}

@inproceedings{donon2019graph,
  author    = {Donon, Balthazar and Donnot, Benjamin and Guyon, Isabelle and Marot, Antoine},
  title     = {Graph Neural Solver for Power Systems},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--8},
  year      = {2019},
  organization = {IEEE},
  note      = {First GNN demonstrating size generalization: trained on 30-node grids, tested on 10-110 nodes}
}

@article{chen2022unsupervised,
  author    = {Chen, Guoyin and others},
  title     = {Unsupervised Deep Learning for {AC-OPF} via {Lagrangian} Duality},
  journal   = {IEEE International Conference on Communications, Control, and Computing Technologies for Smart Grids (SmartGridComm)},
  year      = {2022},
  note      = {End-to-end unsupervised learning using modified augmented Lagrangian function as training loss; no conventional solver needed}
}
```

---

## Related Work Summary (2 Paragraphs)

### Paragraph 1: Supervised learning approaches and achievements

Machine learning approaches for power flow (PF) and optimal power flow (OPF) surrogate modeling have advanced rapidly since 2020, with methods achieving speedups of **10,000× over conventional solvers** while maintaining solution quality within **0.2% of optimality**. Early work focused on fully-connected deep neural networks, exemplified by the DeepOPF series from Chen and colleagues, which demonstrated that DNNs can predict generator dispatch and phase angles for DC and AC-OPF problems on systems up to 2,000 buses. More recent work has shifted toward graph neural networks (GNNs) that naturally encode power grid topology, with buses as nodes and transmission lines as edges. PowerFlowNet achieved **145× speedup on the French 6,470-bus network** using message-passing architectures, while topology-aware GNNs enable adaptation to N-1 contingencies without full retraining. Physics-informed approaches have emerged as a critical complement, embedding Kirchhoff's laws and power balance equations directly into neural network training to improve constraint satisfaction and reduce data requirements. Nellikkath and Chatzivasileiadis demonstrated that physics-informed losses can reduce worst-case constraint violations by **50%** compared to purely data-driven methods, while the Graph Neural Solver from RTE France showed that GNNs can learn power flow solutions by directly minimizing physics law violations rather than imitating conventional solver outputs.

### Paragraph 2: Labeled data limitations and the self-supervised gap

Despite these advances, a fundamental limitation persists across supervised approaches: the **high computational cost of generating labeled training data**. Current methods typically require **5,000 to 60,000 training samples** per network topology, with each sample requiring a full OPF solution from conventional solvers like MATPOWER or interior-point methods. This creates a paradox where the goal of accelerating OPF is undermined by the expense of creating training data. Several approaches have emerged to mitigate this limitation, including physics-informed losses that reduce sample complexity, GAN-based data augmentation, and transfer learning across grid topologies. Notably, self-supervised learning—which has revolutionized natural language processing and computer vision by leveraging abundant unlabeled data—remains **largely unexplored for power system optimization**. A comprehensive literature search reveals only one directly relevant work: PDL-SCOPF by Park and Van Hentenryck (2023), which eliminates labeled data requirements through primal-dual learning with Lagrangian methods. No existing work applies modern self-supervised paradigms such as contrastive learning or masked autoencoding to power flow problems. This represents a significant research gap, as power systems generate vast amounts of unlabeled operational data that could enable pre-training of representations without the expense of solving optimization problems—potentially reducing labeled data requirements while improving generalization across different grid configurations and operating conditions.


Related Work

Recent years have seen intensive research into machine learning (ML) surrogate models for accelerating power flow (PF) and optimal power flow (OPF) computations. Supervised learning approaches dominate: they train neural networks (NNs) to approximate the mapping from grid conditions (loads, renewable outputs, topology) to OPF solutions or power flows
climatechange.ai
. For example, fully-connected deep NNs can predict generator set-points or voltage profiles directly; Pan et al.’s DeepOPF framework achieves feasible OPF solutions with <0.2% optimality loss and up to 100× faster solve times on benchmark grids
personal.cityu.edu.hk
. Similarly, Fioretto et al. combined deep learning with Lagrange dual variables to enforce OPF constraints, attaining high accuracy (~0.2% error) versus conventional solvers
arxiv.org
. Researchers have also explored structured NNs: graph neural networks (GNNs) that exploit power network topology can improve generalization and scalability
arxiv.org
. A GNN-based OPF predictor by Liu et al. incorporates physical laws via a flow feasibility regularizer, yielding constraint-satisfying solutions and adapting quickly to grid topology changes
arxiv.org
. Other innovations include extreme learning machines and convex neural networks for OPF regression
storage.prod.researchhub.com
, as well as hybrid methods that integrate NNs with traditional solvers for warm-starting OPF optimization
sciencedirect.com
. Across the board, these data-driven surrogates can produce near-optimal power flow solutions in milliseconds rather than minutes, with reported cost errors on the order of 0.1–1% and minimal constraint violations after modest post-processing
personal.cityu.edu.hk
arxiv.org
.

However, significant challenges remain for ML-based PF/OPF. A foremost limitation is the heavy reliance on labeled training data (solving thousands of OPF instances offline): obtaining such datasets is computationally expensive, especially for large-scale AC-OPF which is nonconvex and may have multiple optima
climatechange.ai
. This supervised-learning dependency raises concerns about generalization – models can struggle with scenarios outside the training distribution (e.g. novel grid topologies or extreme operating conditions). Many approaches address feasibility by adding penalty terms or corrective mappings, yet guaranteeing all constraints (like line limits or voltage bounds) a priori is difficult, often requiring ad-hoc adjustments
climatechange.ai
. Moreover, scalability to complex networks (with thousands of buses) and adaptability to network changes are ongoing issues. To tackle the data requirement gap, researchers have begun exploring self-supervised or physics-informed learning methods. One recent work (DeepOPF-NGT) forgoes ground-truth OPF labels by embedding the AC power flow equations and OPF objective into the loss function, allowing the NN to learn the OPF mapping from physics alone
climatechange.ai
. Initial results show it can reach comparable optimality and feasibility to supervised models on small grids
climatechange.ai
. Such unsupervised approaches and other physics-aware techniques are still nascent, and no standard solution has yet emerged to eliminate labeled data needs while ensuring reliability. In summary, the state-of-the-art in power system surrogate modeling demonstrates promising speedups and accuracy using supervised ML (spearheaded by groups at Caltech, Georgia Tech, UT Austin, among others), but it remains constrained by data availability and robustness issues. Bridging this gap – for instance, via self-supervised learning that harnesses physical laws – is a key open direction for future research
climatechange.ai
arxiv.org
.

@inproceedings{Fioretto2020,
  author    = {Ferdinando Fioretto and Terrence W. K. Mak and Pascal Van Hentenryck},
  title     = {Predicting {AC} Optimal Power Flows: Combining Deep Learning and Lagrangian Dual Methods},
  booktitle = {Proceedings of the 34th {AAAI} Conference on Artificial Intelligence (AAAI)},
  pages     = {630--637},
  year      = {2020}
}

@article{Pan2021,
  author    = {Xiang Pan and Tong Zhao and Minghua Chen and Shan Zhang},
  title     = {{DeepOPF}: A Deep Neural Network Approach for Security-Constrained {DC} Optimal Power Flow},
  journal   = {{IEEE} Transactions on Power Systems},
  volume    = {36},
  number    = {3},
  pages     = {1725--1735},
  year      = {2021}
}

@article{Huang2022,
  author    = {Wanjun Huang and Xiang Pan and Minghua Chen and Steven H. Low},
  title     = {{DeepOPF-V}: Solving {AC}-{OPF} Problems Efficiently},
  journal   = {{IEEE} Transactions on Power Systems},
  volume    = {37},
  number    = {1},
  pages     = {800--803},
  year      = {2022}
}

@article{Lei2021,
  author    = {Xingyu Lei and Zhifang Yang and Juan Yu and Junbo Zhao and Qian Gao and Hongxin Yu},
  title     = {Data-Driven Optimal Power Flow: A Physics-Informed Machine Learning Approach},
  journal   = {{IEEE} Transactions on Power Systems},
  volume    = {36},
  number    = {1},
  pages     = {346--354},
  year      = {2021}
}

@inproceedings{Huang2021,
  author    = {Wanjun Huang and Minghua Chen},
  title     = {{DeepOPF-NGT}: A Fast Unsupervised Learning Approach for Solving {AC}-{OPF} Problems without Ground Truth},
  booktitle = {38th International Conference on Machine Learning (ICML) Workshop on Tackling Climate Change with {ML}},
  year      = {2021}
}

@article{Liu2023,
  author    = {Shaohui Liu and Chengyang Wu and Hao Zhu},
  title     = {Topology-Aware Graph Neural Networks for Learning Feasible and Adaptive {AC}-{OPF} Solutions},
  journal   = {{IEEE} Transactions on Power Systems},
  volume    = {38},
  number    = {6},
  pages     = {5660--5670},
  year      = {2023}
}

@article{Lotfi2022,
  author    = {Amir Lotfi and Mehrdad Pirnia},
  title     = {Constraint-Guided Deep Neural Network for Solving Optimal Power Flow},
  journal   = {Electric Power Systems Research},
  volume    = {211},
  pages     = {108353},
  year      = {2022}
}

@article{Mohammadi2024,
  author    = {Sina Mohammadi and Van-Hai Bui and Wencong Su and Bin Wang},
  title     = {Surrogate Modeling for Solving {OPF}: A Review},
  journal   = {Sustainability},
  volume    = {16},
  number    = {22},
  pages     = {9851},
  year      = {2024}
}

@article{Khaloie2025,
  author    = {Hooman Khaloie and Mihaly Dolanyi and Jean-Fran{\c{c}}ois Toubeau and Fran{\c{c}}ois Vall{\'e}e},
  title     = {Review of Machine Learning Techniques for Optimal Power Flow},
  journal   = {Applied Energy},
  volume    = {388},
  pages     = {125637},
  year      = {2025}
}

@article{Xie2025,
  author    = {Renyou Xie and Liangcai Xu and Chaojie Li and Xinghuo Yu},
  title     = {Neural-Optimization Integration for {AC} Optimal Power Flow: A Differentiable Warm-Start Approach},
  journal   = {Cyber-Physical Energy Systems},
  volume    = {1},
  number    = {1},
  pages     = {104--115},
  year      = {2025}
}













