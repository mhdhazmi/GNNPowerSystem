# Cascading Failure Modeling in Power Grids: A Literature Review

Machine learning approaches for cascading failure prediction have achieved **96%+ accuracy** with 100× computational speedup over physics-based simulators, yet significant gaps remain in explainability and cross-grid generalization. Traditional simulation models like OPA, DCSIMSEP, and the Manchester model established the theoretical foundation for cascade analysis, but their computational costs and accuracy trade-offs have motivated a shift toward data-driven methods. The critical research gap lies in explainable AI—while XAI methods have been successfully applied to power system tasks like frequency stability, **explainability for cascading failure prediction remains significantly underdeveloped**, with recent benchmarks reporting that current XAI methods perform "suboptimally" on cascade explanation tasks.

## Traditional simulation models trade speed for accuracy

The foundational work on cascading failure modeling emerged from the **OPA (Oak Ridge-PSERC-Alaska) model** developed by Carreras, Dobson, and colleagues in the early 2000s. This DC power flow-based approach captures two-timescale dynamics—slow load growth and fast cascading outages—using Monte Carlo simulation with linear programming dispatch. The OPA model demonstrated that power grids exhibit self-organized criticality, where small perturbations can trigger system-wide blackouts. Building on this foundation, the **DCSIMSEP model** introduced by Eppstein and Hines employs a "Random Chemistry" algorithm that requires only O(log n) simulations per contingency identified, achieving at least **two orders of magnitude speedup** over Monte Carlo sampling for risk estimation.

The **Manchester model** extended these approaches to AC power flow, capturing reactive power, voltage deviations, and under-frequency load shedding mechanisms missed by DC approximations. Comparative studies reveal AC models are approximately **7× more computationally expensive** than DC models but provide more accurate cascade predictions, particularly for voltage collapse scenarios. Hidden failure models, introduced by Chen, Thorp, and Dobson, address a critical mechanism where protection system defects remain dormant until triggered by adjacent failures. All traditional approaches face fundamental limitations: DC models provide overly optimistic estimates by ignoring voltage collapse mechanisms, while AC models suffer convergence issues at high stress levels that limit online applications.

## Machine learning methods excel at prediction but struggle with generalization

The application of machine learning to cascading failure prediction has accelerated dramatically since 2018, with **Graph Neural Networks (GNNs)** emerging as the dominant architecture due to their natural fit for grid topology representation. Varbella, Gjorgiev, and Sansavini demonstrated that GNNs coupled with physics-based AC power flow models achieve over **96% accuracy and balanced accuracy** on cascading failure classification tasks, with demonstrated transfer learning capability across different grid topologies. Deep CNNs have achieved **100× acceleration** for N-1 contingency screening with uncertain renewable scenarios, while Random Forest methods provide interpretable feature importance analysis for vulnerability assessment.

Recent work has introduced physics-informed approaches and reinforcement learning for real-time cascade mitigation, yet critical limitations persist. Most models are trained and tested on the same IEEE test systems (14, 30, 39, 118-bus), with **limited validation of cross-grid generalization**. The heavy reliance on synthetic training data—generated from DC power flow simulators—may not capture real-world cascade dynamics involving voltage collapse and transient instability. The **PowerGraph benchmark** (NeurIPS 2024) represents the first systematic attempt to evaluate GNN explainability for cascading failures, but reports that "the performance of [XAI] methods remains suboptimal," underscoring the ongoing need for dedicated research in this critical area.

## The explainability gap presents a significant research opportunity

Explainable AI methods including SHAP, LIME, and attention mechanisms have been successfully applied to power system applications such as frequency stability prediction and voltage security assessment, yet **cascading failure prediction lacks robust explainability frameworks**. The disconnect is striking: major blackout investigations, including the 2003 Northeast blackout and 2021 Texas winter storm, continue to rely on traditional deterministic timeline reconstruction rather than data-driven root cause identification. This gap between the advances in ML-based prediction and the persistent need for interpretable cascade explanations represents a significant opportunity for research contributions.

The PowerGraph benchmark explicitly identifies this gap, noting that "given the crucial role of explainability for power grid operators, this underscores the ongoing need for dedicated research and development in this field." Current GNN-based cascade predictors function as black boxes—they can predict whether an N-k contingency will trigger a cascade with high accuracy, but cannot explain *which specific failure propagation pathways* led to that prediction. For operational deployment, system operators need not just predictions but understanding of the critical lines and failure sequences driving cascade risk.

## Standard test cases dominate research but have important limitations

The IEEE test cases (14-bus through 300-bus systems) and MATPOWER's Polish grid cases (up to 3,375 buses) serve as the primary benchmarks for cascading failure research. The **ACTIVSg synthetic grid models** from Texas A&M provide larger-scale alternatives (up to 82,000 buses) that contain no Critical Energy Infrastructure Information restrictions, enabling more realistic cascade studies. However, most benchmark cases lack coordinated thermal ratings, protection system settings, and probabilistic failure data essential for realistic cascade modeling. The PowerGraph dataset addresses some gaps by providing ground-truth explanations for GNN-based cascade classification across IEEE24, IEEE39, UK, IEEE118, and Texas 2000-bus systems.

---

## BibTeX Citations

```bibtex
@article{carreras2002critical,
  author    = {Carreras, B. A. and Lynch, V. E. and Dobson, I. and Newman, D. E.},
  title     = {Critical Points and Transitions in an Electric Power Transmission Model for Cascading Failure Blackouts},
  journal   = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume    = {12},
  number    = {4},
  pages     = {985--994},
  year      = {2002},
  doi       = {10.1063/1.1505810}
}

@article{dobson2007complex,
  author    = {Dobson, I. and Carreras, B. A. and Lynch, V. E. and Newman, D. E.},
  title     = {Complex Systems Analysis of Series of Blackouts: Cascading Failure, Critical Points, and Self-Organization},
  journal   = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume    = {17},
  number    = {2},
  pages     = {026103},
  year      = {2007},
  doi       = {10.1063/1.2737822}
}

@article{eppstein2012random,
  author    = {Eppstein, Margaret J. and Hines, Paul D. H.},
  title     = {A ``Random Chemistry'' Algorithm for Identifying Collections of Multiple Contingencies That Initiate Cascading Failure},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {27},
  number    = {3},
  pages     = {1698--1705},
  year      = {2012},
  doi       = {10.1109/TPWRS.2012.2183624}
}

@article{nedic2006criticality,
  author    = {Nedic, D. P. and Dobson, I. and Kirschen, D. S. and Carreras, B. A. and Lynch, V. E.},
  title     = {Criticality in a Cascading Failure Blackout Model},
  journal   = {International Journal of Electrical Power \& Energy Systems},
  volume    = {28},
  number    = {9},
  pages     = {627--633},
  year      = {2006},
  doi       = {10.1016/j.ijepes.2006.03.006}
}

@article{chen2005cascading,
  author    = {Chen, J. and Thorp, J. S. and Dobson, I.},
  title     = {Cascading Dynamics and Mitigation Assessment in Power System Disturbances via a Hidden Failure Model},
  journal   = {International Journal of Electrical Power \& Energy Systems},
  volume    = {27},
  number    = {4},
  pages     = {318--326},
  year      = {2005},
  doi       = {10.1016/j.ijepes.2004.12.003}
}

@article{varbella2023geometric,
  author    = {Varbella, Anna and Gjorgiev, Blazhe and Sansavini, Giovanni},
  title     = {Geometric Deep Learning for Online Prediction of Cascading Failures in Power Grids},
  journal   = {Reliability Engineering \& System Safety},
  volume    = {237},
  pages     = {109341},
  year      = {2023},
  doi       = {10.1016/j.ress.2023.109341}
}

@inproceedings{varbella2024powergraph,
  author    = {Varbella, Anna and Amara, Kenza and Gjorgiev, Blazhe and El-Assady, Mennatallah and Sansavini, Giovanni},
  title     = {{PowerGraph}: A Power Grid Benchmark Dataset for Graph Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024}
}

@article{du2019achieving,
  author    = {Du, Y. and Li, F. and Li, J. and Zheng, T.},
  title     = {Achieving 100x Acceleration for {N-1} Contingency Screening With Uncertain Scenarios Using Deep Convolutional Neural Network},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {34},
  number    = {4},
  pages     = {3303--3305},
  year      = {2019},
  doi       = {10.1109/TPWRS.2019.2914860}
}

@article{vaiman2012risk,
  author    = {Vaiman, M. and Bell, K. and Chen, Y. and Chowdhury, B. and Dobson, I. and Hines, P. and Papic, M. and Miller, S. and Zhang, P.},
  title     = {Risk Assessment of Cascading Outages: Methodologies and Challenges},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {27},
  number    = {2},
  pages     = {631--641},
  year      = {2012},
  doi       = {10.1109/TPWRS.2011.2177868}
}

@article{zimmerman2011matpower,
  author    = {Zimmerman, R. D. and Murillo-S{\'a}nchez, C. E. and Thomas, R. J.},
  title     = {{MATPOWER}: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education},
  journal   = {IEEE Transactions on Power Systems},
  volume    = {26},
  number    = {1},
  pages     = {12--19},
  year      = {2011},
  doi       = {10.1109/TPWRS.2010.2051168}
}
```

---

## Summary for Literature Review

**Paragraph 1 (Traditional Simulation-Based Approaches):**
Traditional cascading failure models have established the theoretical foundation for understanding blackout propagation in power systems. The OPA model, developed by Carreras, Dobson, and colleagues, introduced a DC power flow framework with Monte Carlo simulation that demonstrated self-organized criticality in grid behavior. The DCSIMSEP model improved computational efficiency through the Random Chemistry algorithm, achieving two orders of magnitude speedup over Monte Carlo methods for risk estimation. AC-based models like the Manchester model capture voltage collapse and reactive power dynamics missed by DC approximations, but at approximately 7× higher computational cost. A fundamental trade-off persists across all traditional approaches: DC models enable tractable large-scale analysis but provide overly optimistic predictions by ignoring voltage-related failure mechanisms, while AC models offer greater physical fidelity but face convergence issues under high-stress conditions that limit real-time operational applications.

**Paragraph 2 (Machine Learning Approaches and Research Gaps):**
Machine learning methods, particularly Graph Neural Networks, have demonstrated remarkable success in cascading failure prediction, achieving over 96% accuracy with 100× speedup over physics-based simulators. However, these advances are constrained by several critical limitations. Most approaches rely heavily on supervised learning with synthetic training data generated from simplified DC power flow models, raising questions about their fidelity to real-world cascade dynamics. Cross-grid generalization remains poorly validated—models are typically trained and tested on the same IEEE benchmark systems, with limited evidence of transfer to different grid topologies or operating conditions. Most critically, explainability represents a significant and underexplored gap: while XAI methods have proven effective for power system applications like frequency stability, the PowerGraph benchmark (NeurIPS 2024) reports that current explainability methods perform "suboptimally" for cascading failure analysis, and major blackout investigations continue to rely on traditional deterministic analysis rather than interpretable ML methods. These gaps—particularly in explainability and multi-grid generalization—represent significant opportunities for research contributions targeting operational deployment of ML-based cascade prediction tools.

Cascading Failure Prediction & Critical Infrastructure

Power grids have traditionally been analyzed for cascading failures using detailed simulation models. These models iteratively simulate outages and power flow re-distribution to capture how an initial disturbance (or N-k contingencies) might trigger further equipment overloads and outages
overbye.engr.tamu.edu
. Classical approaches range from high-level statistical models (e.g. the CASCADE branching-process model) to exhaustive probabilistic or deterministic simulations
overbye.engr.tamu.edu
. To manage complexity, many tools use simplified DC power flow (linearized) cascade models, which are faster but approximate – for example, ignoring reactive power and voltage limits can lead DC models to underestimate cascade severity compared to full AC models
wimnet.ee.columbia.edu
. More realistic AC power flow and dynamic simulations include generator and protection system responses, capturing phenomena like voltage collapse or transient instability that simpler models miss
overbye.engr.tamu.edu
. However, such detailed simulations are computationally intensive, making comprehensive N-k contingency analysis or Monte Carlo sampling of cascades very slow (often only feasible as offline studies). In summary, traditional cascade modeling provides physics-grounded insight but is slow and expensive, especially as system size and model fidelity (DC vs. AC, static vs. dynamic) grow.

In recent years, researchers have turned to machine learning to predict or assess cascading failure risk much faster than exhaustive simulation. Supervised learning has been applied on data from simulated cascades – for example, using various classifiers (decision trees, SVMs, random forests, neural networks, etc.) to predict outcomes like the “onset time” or size of a cascade from pre-disturbance system features
arxiv.org
. These data-driven models can approximate the complex mapping from grid state and initial outages to cascade severity, after being trained on many scenario simulations. More advanced deep learning methods have also emerged: convolutional and recurrent networks, and especially graph neural networks (GNNs) that naturally represent the power grid topology, have shown improved accuracy in cascade prediction
research-collection.ethz.ch
. For instance, GNN models can learn the dependencies between component failures and were reported to achieve high prediction accuracy (over 95% in test cases) on whether an N-k event will lead to a large blackout
research-collection.ethz.ch
. Nevertheless, key gaps remain. Most ML models are trained and tested on a single network or a narrow range of operating conditions, so their generalization is limited – a model learned on one grid may falter on a different grid or under unseen load/generation patterns
research-collection.ethz.ch
research-collection.ethz.ch
. Recent work addresses this by using transfer learning and GNNs’ inductive ability to adapt models across different grid topologies
research-collection.ethz.ch
research-collection.ethz.ch
, but broadly, poor cross-system generalization is a challenge. Another critical gap is explainability: unlike physics-based simulations that trace clear cause-effect chains, most ML approaches act as black boxes, making it hard for operators to trust their predictions without understanding the drivers. So far, few studies explicitly provide root cause analysis for ML predictions. Some nascent efforts aim to incorporate explainability, for example by creating benchmark datasets with ground-truth “explanations” for each cascade event (identifying the critical line or component whose failure precipitated the cascade) to facilitate training explainable GNN models
research-collection.ethz.ch
. In practice, major blackout post-mortems still rely on detailed offline analysis to identify root causes – for example, the 2003 Northeast US-Canada blackout investigation traced the cascade back to a specific line contact and software failures
arxiv.org
. In summary, machine learning offers promising speed-ups for cascade prediction (enabling near-real-time contingency screening), but ensuring these models are transparent and robust across varied conditions and grids remains an open research problem.

References
@article{Guo2023,
  author = {Guo, Zhenping and Sun, Kai and Su, Xiaowen and Simunovic, Srdjan},
  title = {A review on simulation models of cascading failures in power systems},
  journal = {iEnergy},
  volume = {2},
  number = {4},
  pages = {284--296},
  year = {2023},
  doi = {10.23919/IEN.2023.0039}
}

@article{Cetinay2018,
  author = {Cetinay, Hale and Soltan, Saleh and Kuipers, Fernando A. and Zussman, Gil and Van~Mieghem, Piet},
  title = {Analyzing Cascading Failures in Power Grids under the AC and DC Power Flow Models},
  journal = {SIGMETRICS Perform. Eval. Review},
  volume = {45},
  number = {2},
  pages = {198--203},
  year = {2018},
  doi = {10.1145/3199524.3199559}
}

@inproceedings{Liu2022,
  author = {Liu, Yijing and Zhang, Anna and Dehghanian, Pooria and Jung, Jung~Kyo and Habiba, Ummay and Overbye, Thomas~J.},
  title = {Modeling and Analysis of Cascading Failures in Large-Scale Power Grids},
  booktitle = {Proc. IEEE Kansas Power and Energy Conference (KPEC)},
  year = {2022},
  address = {Manhattan, KS, USA}
}

@article{RahnamayNaeini2014,
  author = {Rahnamay-Naeini, Mahshid and Wang, Zhuo and Ghani, Nasir and Mammoli, A. and Hayat, M. M.},
  title = {Stochastic analysis of cascading-failure dynamics in power grids},
  journal = {IEEE Transactions on Power Systems},
  volume = {29},
  number = {4},
  pages = {1767--1779},
  year = {2014},
  doi = {10.1109/TPWRS.2014.2301797}
}

@article{Varbella2023,
  author = {Varbella, Anna and Gjorgiev, Blazhe and Sansavini, Giovanni},
  title = {Geometric deep learning for online prediction of cascading failures in power grids},
  journal = {Reliability Engineering \& System Safety},
  volume = {237},
  pages = {109341},
  year = {2023},
  doi = {10.1016/j.ress.2023.109341}
}

@article{Sami2024,
  author = {Sami, Naeem~Md and Naeini, Mia},
  title = {Machine learning applications in cascading failure analysis in power systems: A review},
  journal = {Electric Power Systems Research},
  volume = {232},
  pages = {110415},
  year = {2024},
  doi = {10.1016/j.epsr.2024.110415}
}

@misc{Pani2024,
  author = {Pani, Samita~Rani and Bera, Pallav~Kumar and Samal, Rajat~Kanti},
  title = {Predicting Cascading Failures in Power Systems using Machine Learning},
  howpublished = {arXiv preprint arXiv:2503.00567},
  year = {2024}
}

@inproceedings{Chadaga2023,
  author = {Chadaga, Sudeep and Wu, Xuanxuan and Modiano, Eytan},
  title = {Power failure cascade prediction using graph neural networks},
  booktitle = {Proc. IEEE Int. Conf. on Communications, Control, and Computing Tech. for Smart Grids (SmartGridComm)},
  year = {2023},
  pages = {1--7}
}

@inproceedings{VarbellaPowerGraph2023,
  author = {Varbella, Anna and Amara, Kenza and Gjorgiev, Blazhe and Sansavini, Giovanni},
  title = {{PowerGraph}: A power grid benchmark dataset for graph neural networks},
  booktitle = {NeurIPS 2023 Workshop on New Frontiers in Graph Learning (GLFrontiers)},
  year = {2023},
  address = {New Orleans, LA, USA}
}

@techreport{USCanada2004,
  author = {{U.S.-Canada Power System Outage Task Force}},
  title = {Final Report on the August 14, 2003 Blackout in the United States and Canada: Causes and Recommendations},
  institution = {U.S. Dept. of Energy and Natural Resources Canada},
  year = {2004}
}