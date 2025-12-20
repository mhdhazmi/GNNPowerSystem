# Graph Self-Supervised Learning: Literature Review for IEEE Power Systems

## BibTeX File: agent3_graph_ssl.bib

```bibtex
@inproceedings{hou2022graphmae,
  title={{GraphMAE}: Self-Supervised Masked Graph Autoencoders},
  author={Hou, Zhenyu and Liu, Xiao and Cen, Yukuo and Dong, Yuxiao and Yang, Hongxia and Wang, Chunjie and Tang, Jie},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={594--604},
  year={2022},
  publisher={ACM}
}

@inproceedings{xia2022simgrace,
  title={{SimGRACE}: A Simple Framework for Graph Contrastive Learning without Data Augmentation},
  author={Xia, Jun and Wu, Lirong and Chen, Jintao and Hu, Bozhen and Li, Stan Z.},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1070--1079},
  year={2022},
  publisher={ACM}
}

@inproceedings{thakoor2022bgrl,
  title={Large-Scale Representation Learning on Graphs via Bootstrapping},
  author={Thakoor, Shantanu and Tallec, Corentin and Azar, Mohammad Gheshlaghi and Azabou, Mehdi and Dyer, Eva L and Munos, R{\'e}mi and Veli{\v{c}}kovi{\'c}, Petar and Valko, Michal},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{qiu2020gcc,
  title={{GCC}: Graph Contrastive Coding for Graph Neural Network Pre-Training},
  author={Qiu, Jiezhong and Chen, Qibin and Dong, Yuxiao and Zhang, Jing and Yang, Hongxia and Ding, Ming and Wang, Kuansan and Tang, Jie},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={1150--1160},
  year={2020},
  publisher={ACM}
}

@inproceedings{sun2020infograph,
  title={{InfoGraph}: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization},
  author={Sun, Fan-Yun and Hoffmann, Jordan and Verma, Vikas and Tang, Jian},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

@inproceedings{you2020graphcl,
  title={Graph Contrastive Learning with Augmentations},
  author={You, Yuning and Chen, Tianlong and Sui, Yongduo and Chen, Ting and Wang, Zhangyang and Shen, Yang},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={5812--5823},
  year={2020}
}

@inproceedings{velickovic2019dgi,
  title={Deep Graph Infomax},
  author={Veli{\v{c}}kovi{\'c}, Petar and Fedus, William and Hamilton, William L and Li{\`o}, Pietro and Bengio, Yoshua and Hjelm, R Devon},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{zhu2021gca,
  title={Graph Contrastive Learning with Adaptive Augmentation},
  author={Zhu, Yanqiao and Xu, Yichen and Yu, Feng and Liu, Qiang and Wu, Shu and Wang, Liang},
  booktitle={Proceedings of the Web Conference 2021},
  pages={2069--2080},
  year={2021},
  publisher={ACM}
}

@inproceedings{hou2023graphmae2,
  title={{GraphMAE2}: A Decoding-Enhanced Masked Self-Supervised Graph Learner},
  author={Hou, Zhenyu and He, Yufei and Cen, Yukuo and Liu, Xiao and Dong, Yuxiao and Kharlamov, Evgeny and Tang, Jie},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={737--746},
  year={2023},
  publisher={ACM}
}

@inproceedings{hassani2020mvgrl,
  title={Contrastive Multi-View Representation Learning on Graphs},
  author={Hassani, Kaveh and Khasahmadi, Amir Hosein},
  booktitle={International Conference on Machine Learning},
  pages={4116--4126},
  year={2020},
  organization={PMLR}
}

@article{ji2023stssl,
  title={Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction},
  author={Ji, Jiahao and Wang, Jingyuan and Huang, Chao and Wu, Junjie and Xu, Boren and Wu, Zhenhe and Zhang, Junbo and Zheng, Yu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  pages={4356--4364},
  year={2023}
}

@article{ringsquandl2024safepowergraph,
  title={{SafePowerGraph}: Safety-aware Evaluation of Graph Neural Networks for Transmission Power Grids},
  author={Ringsquandl, Martin and others},
  journal={arXiv preprint arXiv:2407.15929},
  year={2024}
}
```

---

## Two-Paragraph Summary

Graph self-supervised learning (SSL) has emerged as a powerful paradigm for learning representations without labeled data, with two dominant approaches: **contrastive learning** and **generative masked reconstruction**. Contrastive methods such as Deep Graph Infomax, GCC, InfoGraph, and BGRL learn by maximizing agreement between augmented views of graphs, employing augmentation strategies including node dropping, edge perturbation, and subgraph sampling. BGRL and SimGRACE advanced this paradigm by eliminating the need for negative samples and data augmentation respectively—BGRL achieves **2-10× memory reduction** while matching state-of-the-art performance, and SimGRACE generates contrastive views through encoder perturbation with Gaussian noise rather than graph manipulation. In contrast, generative approaches like GraphMAE reconstruct masked node features using a scaled cosine error loss and re-mask decoding strategy, demonstrating that masked autoencoders can **match or exceed contrastive methods** (achieving **84.2%** on Cora versus 82.7% for BGRL). GraphMAE2 further enhanced this approach with multi-view random re-masking and latent representation prediction, achieving strong results on graphs with over 100 million nodes.

These graph SSL methods demonstrate substantial benefits in **low-label regimes**, where labeled training data is scarce. GraphMAE2 shows that with only **1% labeled data** on ogbn-Papers100M, self-supervised pre-training improves accuracy from 43.55% (random initialization) to **49.01%**—a **5.46 percentage point gain**—with similar improvements observed at 5% and 10% label fractions across multiple benchmarks. Applications to physical systems have proven successful in molecular graphs (KPGT, MolGNet) and traffic networks (ST-SSL), where graph structure naturally encodes physical relationships. However, **a significant research gap exists for power grid applications**: while SafePowerGraph (2024) introduced hybrid supervised-SSL approaches for power systems, no pure graph SSL method has been designed specifically for electrical networks with physics-informed constraints such as power flow equations, Kirchhoff's laws, or voltage-angle relationships. This gap presents a clear opportunity to develop graph SSL methods that leverage power system physics as self-supervised pretext tasks—enabling effective learning from the abundant unlabeled operational data available in modern power grids while respecting the underlying physical laws governing electrical networks.


Physics-Informed Neural Networks (PINNs) embed physical laws into neural models to improve learning. The canonical PINN architecture is typically a feed-forward network trained not only on data but also to minimize the residuals of known physics (e.g. governing PDEs) added as a soft constraint (penalty term) in the loss
medium.com
. This multi-term loss approach requires careful weighting between data fit and physics residual, which can be challenging to balance
medium.com
. Recent research has therefore explored hard constraints and specialized architectures to enforce physics. Hard-constrained PINNs satisfy physical laws by construction – for example, using coordinate transformations or distance functions that guarantee boundary conditions are exactly met, thereby removing the need for penalty weights
medium.com
medium.com
. Another approach is to bake domain knowledge into the network architecture itself; for instance, Hamiltonian neural networks hard-code energy conservation as an inductive bias in their design
arxiv.org
. In general, these strategies incorporate prior scientific knowledge (conservation laws, symmetries, known equations) either through the loss function or via network structure, guiding the model toward physically consistent solutions. There are also hybrid physics-ML models that combine neural networks with explicit equation-based components – a form of residual modeling where the network only learns deviations from a known physics baseline
arxiv.org
. This broadens the PINN landscape beyond the original Raissi et al. formulation, offering a spectrum from soft-penalty PINNs to hard-coded physics networks and hybrid architectures.

Benefits and Challenges: Imposing physics-based inductive biases often yields more robust and generalizable models. By constraining the hypothesis space to functions that obey physical laws, PINNs can achieve better sample efficiency and out-of-distribution performance compared to purely data-driven nets
arxiv.org
mdpi.com
. In effect, the model is less likely to learn spurious behaviors, and its predictions remain physically plausible (e.g. conserving energy or mass), which also enhances interpretability
mdpi.com
nature.com
. Studies report that physics-informed models can extrapolate more reliably – for example, adding AC power flow equations as a regularization term kept a neural voltage predictor consistent with Kirchhoff’s laws, reducing errors on unseen grid topologies
arxiv.org
. In power systems applications, numerous PINN variants and physics-guided neural networks have demonstrated improved accuracy and stability with less data, compared to black-box models
arxiv.org
. However, a key gap is that most PINN success stories are in continuum domains (solving PDEs in fluid dynamics, mechanics, etc.), and graph-structured problems (like electrical grids or other networks) remain challenging
mdpi.com
. PINNs have relatively limited flexibility on discrete network topologies
mdpi.com
, indicating the need for new hybrid approaches (e.g. combining graph neural networks with physics constraints) to extend the advantages of physics-informed learning to these structured domains
mdpi.com
.

@article{Raissi2019,
  author    = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  title     = {Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  journal   = {Journal of Computational Physics},
  volume    = {378},
  pages     = {686--707},
  year      = {2019},
  doi       = {10.1016/j.jcp.2018.10.045}
}

@article{Karniadakis2021,
  author    = {Karniadakis, George Em and Kevrekidis, Ioannis G. and Lu, Lu and Perdikaris, Paris and Wang, Sifan and Yang, Liu},
  title     = {Physics-informed machine learning},
  journal   = {Nature Reviews Physics},
  volume    = {3},
  number    = {6},
  pages     = {422--440},
  year      = {2021},
  doi       = {10.1038/s42254-021-00314-5}
}

@article{Sukumar2022,
  author    = {Sukumar, N. and Srivastava, Ankit},
  title     = {Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks},
  journal   = {Computer Methods in Applied Mechanics and Engineering},
  volume    = {389},
  pages     = {114334},
  year      = {2022},
  doi       = {10.1016/j.cma.2021.114334}
}

@inproceedings{Greydanus2019,
  author    = {Greydanus, Samuel and Dzamba, Misko and Yosinski, Jason},
  title     = {Hamiltonian Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {32},
  pages     = {15353--15363},
  year      = {2019},
  doi       = {10.48550/arXiv.1906.01563}
}

@article{Huang2023,
  author    = {Huang, Bin and Wang, Jianhui},
  title     = {Applications of Physics-Informed Neural Networks in Power Systems -- A Review},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {38},
  number    = {1},
  pages     = {572--588},
  year      = {2023},
  doi       = {10.1109/TPWRS.2022.3162473}
}

@article{Hu2021,
  author    = {Hu, Xinyue and Hu, Haoji and Verma, Saurabh and Zhang, Zhi{-}Li},
  title     = {Physics-guided deep neural networks for power flow analysis},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {36},
  number    = {3},
  pages     = {2082--2092},
  year      = {2021},
  doi       = {10.1109/TPWRS.2020.3029557}
}

@article{Gupta2024,
  author    = {Gupta, Kishor Datta and Siddique, Sunzida and George, Roy and Kamal, Marufa and Rifat, Rakib Hossain and Haque, Mohd Ariful},
  title     = {Physics Guided Neural Networks with Knowledge Graph},
  journal   = {Digital},
  volume    = {4},
  number    = {4},
  pages     = {846--865},
  year      = {2024},
  doi       = {10.3390/digital4040042}
}