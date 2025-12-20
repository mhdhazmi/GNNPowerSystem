# Physics-Informed Neural Networks for Power Systems: Related Work References

**Bottom line:** The physics-informed ML literature offers three distinct strategies for encoding domain knowledge—soft regularization, hard constraints, and architectural inductive bias—with strong evidence for improved generalization. However, most work targets continuous PDE-governed systems rather than discrete graph-structured networks like power grids, creating a clear positioning opportunity for physics-guided GNN approaches.

---

## Recommended References (8 High-Quality Citations)

### Foundational PINN Work

**1. Raissi, Perdikaris, & Karniadakis (2019)** — *Canonical PINN Framework*
- **Title:** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations"
- **Venue:** Journal of Computational Physics, Vol. 378, pp. 686–707
- **Key Contribution:** Introduced the PINN paradigm encoding PDEs as soft constraints via automatic differentiation. The composite loss **L = L_data + λ·L_physics** minimizes both data mismatch and PDE residuals at collocation points. Demonstrated on Navier-Stokes, Schrödinger, and Burgers equations for both forward solutions and inverse parameter discovery.

**2. Cuomo et al. (2022)** — *Comprehensive Survey*
- **Title:** "Scientific Machine Learning Through Physics–Informed Neural Networks: Where We Are and What's Next"
- **Venue:** Journal of Scientific Computing, Vol. 92, Article 88
- **Key Contribution:** Comprehensive review characterizing PINN variants including vanilla PINNs, variational hp-VPINNs, and conservative cPINNs. Provides taxonomy of soft vs. hard constraint enforcement and identifies that most research focuses on activation functions, optimization techniques, and loss formulations for continuous PDE domains.

### Physics Incorporation Methods

**3. Beucler et al. (2021)** — *Hard Constraints via Architecture*
- **Title:** "Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems"
- **Venue:** Physical Review Letters, Vol. 126, 098302
- **Key Contribution:** Systematic framework comparing unconstrained (UCnet), loss-constrained (LCnet), and architecture-constrained (ACnet) networks for climate modeling. Hard constraints achieve conservation of energy, mass, and radiation **to machine precision** via custom layers that analytically solve conservation equations, without degrading predictive performance.

**4. Greydanus, Dzamba, & Yosinski (2019)** — *Architectural Inductive Bias*
- **Title:** "Hamiltonian Neural Networks"
- **Venue:** NeurIPS 2019, pp. 15353–15363
- **Key Contribution:** Parameterizes the Hamiltonian function H(q,p) with a neural network, deriving dynamics via Hamilton's equations. Energy conservation is **exact by construction** through symplectic gradient structure. Demonstrates faster training, superior generalization, and perfect time-reversibility compared to standard dynamics learning.

### Power Systems Applications

**5. Misyris, Venzke, & Chatzivasileiadis (2020)** — *PINNs for Power System Dynamics*
- **Title:** "Physics-Informed Neural Networks for Power Systems"
- **Venue:** IEEE Power & Energy Society General Meeting (PESGM)
- **Key Contribution:** First paper introducing PINNs for power systems, embedding **swing equations** governing rotor angle and frequency dynamics into the loss function. Achieves **87× speedup** over numerical methods while requiring substantially less training data. Enables estimation of uncertain parameters (inertia, damping) at a fraction of computational cost.

**6. Zamzam & Sidiropoulos (2020)** — *Physics-Aware Architecture Design*
- **Title:** "Physics-Aware Neural Networks for Distribution System State Estimation"
- **Venue:** IEEE Transactions on Power Systems, Vol. 35, No. 6, pp. 4347–4356
- **Key Contribution:** Exploits power grid topology to design neural network architecture, incorporating the **separability of state estimation** derived from power flow equations. Physics-based architecture reduces parameterization and prevents overfitting. Validated on IEEE-37 and IEEE-123 bus feeders, outperforming Gauss-Newton approaches.

**7. Authier et al. (2024)** — *Physics-Informed GNN for Power Grids*
- **Title:** "Physics-Informed Graph Neural Network for Dynamic Reconfiguration of Power Systems (GraPhyR)"
- **Venue:** Applied Energy (also arXiv:2310.00728)
- **Key Contribution:** End-to-end physics-informed GNN with **gated message passing** modeling switches as physical gates controlling power flow. Physics-informed rounding layer embeds discrete switch decisions respecting operational constraints. Local predictors provide scale-free predictions generalizable to any grid topology.

### Generalization Benefits and Gap

**8. Thangamuthu et al. (2022)** — *Physics-Informed GNN Benchmarking*
- **Title:** "Unravelling the Performance of Physics-informed Graph Neural Networks for Dynamical Systems"
- **Venue:** NeurIPS 2022 Datasets and Benchmarks Track
- **Key Contribution:** Benchmarked 13 physics-informed GNNs (Hamiltonian, Lagrangian, graph neural ODE variants) on spring, pendulum, and gravitational systems. Key finding: **all physics-informed GNNs exhibit zero-shot generalizability to system sizes an order of magnitude larger than training**. GNNs with explicit constraints showed significantly enhanced OOD performance.

---

## Two-Paragraph Summary for Related Work Section

Physics-informed machine learning has emerged as a powerful paradigm for incorporating domain knowledge into neural networks, with three primary integration strategies. **Soft constraint** approaches, exemplified by Raissi et al.'s foundational PINN framework, encode governing equations as regularization terms in the loss function, enabling flexible training but without guaranteed constraint satisfaction. **Hard constraint** methods, such as Beucler et al.'s architecture-constrained networks, embed conservation laws directly into network layers that analytically solve physical equations, achieving satisfaction to machine precision. **Architectural inductive bias** approaches, including Hamiltonian Neural Networks and equivariant GNNs, design network structure to inherently respect physical symmetries and conservation principles. In power systems specifically, researchers have applied PINNs to encode swing equations for transient dynamics and exploited grid topology to inform architecture design for state estimation, demonstrating significant speedups and reduced data requirements compared to purely data-driven methods.

The physics-informed paradigm offers compelling benefits for out-of-distribution generalization—physics-informed GNNs have demonstrated zero-shot generalization to systems an order of magnitude larger than training configurations, while physics constraints act as regularization that limits the solution space and improves extrapolation to unseen operating conditions. However, a significant gap remains: the vast majority of PINN research targets **continuous PDE-governed domains** such as fluid dynamics, solid mechanics, and heat transfer, where automatic differentiation naturally computes spatial and temporal derivatives. Discrete, graph-structured infrastructure networks like power grids present fundamentally different challenges—physical laws (Kirchhoff's laws, power balance) operate directly on graph edges and nodes rather than as discretized continuous fields, and topological changes require handling discrete switching dynamics. While recent work has begun addressing physics-informed GNNs for power systems, these approaches remain nascent compared to the mature PINN literature for PDEs, creating a clear opportunity for physics-guided graph-based methods that encode power system physics through message passing and self-supervised learning rather than traditional collocation-based formulations.

---

## Quick Reference Table

| Reference | Category | Domain | Key Physics |
|-----------|----------|--------|-------------|
| Raissi et al. (2019) | Soft constraints | General PDEs | Loss-function regularization |
| Cuomo et al. (2022) | Survey | PINNs overview | Taxonomy of methods |
| Beucler et al. (2021) | Hard constraints | Climate | Conservation laws (exact) |
| Greydanus et al. (2019) | Architecture | Dynamics | Hamiltonian mechanics |
| Misyris et al. (2020) | Soft constraints | Power systems | Swing equations |
| Zamzam & Sidiropoulos (2020) | Architecture | Power systems | Grid topology |
| Authier et al. (2024) | Architecture + GNN | Power systems | Message passing = power flow |
| Thangamuthu et al. (2022) | Benchmark | GNNs | OOD generalization evidence |

This reference set provides balanced coverage of foundational methods, incorporation strategies, power systems applications, and the generalization benefits that position your physics-guided GNN encoder contribution against the broader literature.


Self-Supervised Learning on Graphs (Graph SSL)

Graph self-supervised learning (SSL) methods fall into two broad categories: contrastive vs. generative approaches
arxiv.org
. Contrastive methods have dominated in recent years by maximizing agreement between different augmented views of a graph (as in pioneering frameworks like DGI, InfoGraph, GraphCL)
arxiv.org
. However, these methods often rely on complex training strategies – e.g. bi-encoder momentum updates in GCC and BGRL, large negative sample sets, and carefully tuned graph augmentations
arxiv.org
. To reduce this burden, simplified contrastive frameworks have emerged: SimGRACE dispenses with manual data augmentation by perturbing the GNN encoder itself to generate two views
arxiv.org
, while BGRL uses a bootstrapping approach (predicting one encoder’s outputs from another) that avoids negative pairs and still achieves state-of-the-art results with far lower memory use
arxiv.org
. On the other hand, generative (reconstruction-based) graph SSL has seen slower progress historically
arxiv.org
. Recent advances like GraphMAE show that a carefully designed generative strategy can close this gap
arxiv.org
. GraphMAE masks a subset of node features and trains a GNN to reconstruct them, using techniques like a dedicated decoder and scaled cosine error for robustness; this simple autoencoder outperforms both contrastive and previous generative models across numerous benchmark tasks
arxiv.org
.

Graph SSL methods have demonstrated significant benefits in low-label and transfer learning regimes. Unsupervised pre-training of GNNs on large unlabeled graph data can markedly improve downstream performance when labels are scarce – yielding up to 9.4% higher ROC-AUC and achieving state-of-the-art results in molecular property and protein function prediction benchmarks
cs.stanford.edu
. Likewise, the bootstrapped BGRL approach outperforms purely supervised baselines on huge graphs while using 2–10× less memory
arxiv.org
. Contrastive pre-training can also aid out-of-distribution transfer: for example, GCC learns universal structural patterns from diverse networks and delivers comparable or better performance on new graph tasks than models trained from scratch
keg.cs.tsinghua.edu.cn
. These techniques have so far been applied mainly to typical graph domains (social/information networks, biochemical molecule graphs) with strong results
arxiv.org
. In contrast, their application to physical infrastructure networks (e.g. power grids) remains largely unexplored, as such domains carry hard physical constraints not captured by standard SSL objectives. Only very recent work has begun integrating physics knowledge into graph SSL for power systems
github.com
, highlighting a promising open direction for future research.

@inproceedings{Sun2020InfoGraph,
  title = {InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization},
  author = {Sun, Fan-Yun and Hoffmann, Jordan and Verma, Vikas and Tang, Jian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@inproceedings{You2020GraphCL,
  title = {Graph Contrastive Learning with Augmentations},
  author = {You, Yuning and Chen, Tianlong and Sui, Yongduo and Chen, Ting and Wang, Zhangyang and Shen, Yang},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020}
}

@inproceedings{Qiu2020GCC,
  title = {GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training},
  author = {Qiu, Jiezhong and Chen, Qibin and Dong, Yuxiao and Zhang, Jing and Yang, Hongxia and Ding, Ming and Wang, Kuansan and Tang, Jie},
  booktitle = {Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2020}
}

@inproceedings{Hu2020PretrainGNN,
  title = {Strategies for Pre-Training Graph Neural Networks},
  author = {Hu, Weihua and Liu, Bowen and Gomes, Joseph and Zitnik, Marinka and Liang, Percy and Pande, Vijay and Leskovec, Jure},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020}
}

@inproceedings{Thakoor2022BGRL,
  title = {Large-Scale Representation Learning on Graphs via Bootstrapping},
  author = {Thakoor, Shantanu and Tallec, Corentin and Gheshlaghi Azar, Mohammad and Azabou, Mehdi and Dyer, Eva L. and Munos, R{\\'e}mi and Veli{\\v{c}}kovi{\\'c}, Petar and Valko, Michal},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2022}
}

@inproceedings{Xia2022SimGRACE,
  title = {SimGRACE: A Simple Framework for Graph Contrastive Learning without Data Augmentation},
  author = {Xia, Jun and Wu, Lirong and Chen, Jintao and Hu, Bozhen and Li, Stan Z.},
  booktitle = {Proceedings of The Web Conference 2022 (WWW)},
  year = {2022}
}

@inproceedings{Hou2022GraphMAE,
  title = {GraphMAE: Self-Supervised Masked Graph Autoencoders},
  author = {Hou, Zhenyu and Liu, Xiao and Cen, Yukuo and Dong, Yuxiao and Yang, Hongxia and Wang, Chunjie and Tang, Jie},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2022}
}

@inproceedings{Hu2020GPT-GNN,
  title = {GPT-GNN: Generative Pre-Training of Graph Neural Networks},
  author = {Hu, Ziniu and Dong, Yuxiao and Wang, Kuansan and Chang, Kai-Wei and Sun, Yizhou},
  booktitle = {Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year = {2020}
}

@inproceedings{Liu2022GraphMVP,
  title = {Pre-training Molecular Graph Representation with 3D Geometry},
  author = {Liu, Shengchao and Wang, Hanchen and Liu, Weiyang and Lasenby, Joan and Guo, Hongyu and Tang, Jian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2022}
}

@article{Zhu2025PhysicsSSL,
  title = {GNNs' Generalization Improvement for Large-Scale Power System Analysis Based on Physics-Informed Self-Supervised Pre-Training},
  author = {Zhu, Yuhong and Zhou, Yongzhi and Wei, Wei and Li, Peng and Huang, Wenqi},
  journal = {IEEE Transactions on Power Systems},
  year = {2025}
}