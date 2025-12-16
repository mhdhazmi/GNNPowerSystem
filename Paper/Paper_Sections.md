Below is a paper “skeleton” that maps directly onto what you already have in **Simulation_Results.md** (canonical results + LaTeX tables + figure assets) and **ModelArchitecture.md** (model/SSL details). I’m outlining it at **section + paragraph** level, and then going **deep** on **Experiments** and **Results** (subsections + what goes in each + exact tables/figures to place).

---

## Proposed paper structure (IEEE-style)

### Title (working)

**Physics-Guided Self-Supervised Graph Neural Networks for Power Grid Analysis: Transfer Across Cascade Prediction, Power Flow, and Line Flow**

(You already frame it as “Physics-Guided GNNs for Power System Analysis” and emphasize SSL transfer across these tasks. )

---

## Abstract (1 paragraph)

**Paragraph 1 (single abstract paragraph):**
State the operational need (fast surrogates + rare-event risk prediction), the core method (physics-guided message passing + SSL masked reconstruction pretraining), and the key quantitative headline improvements (low-label regime + scalability + explainability/robustness as supporting evidence). Your abstract can safely quote the cross-task 10% improvements and IEEE‑118 stability narrative because they are already consolidated in your canonical results.  

---

## Index Terms (1 line)

Power systems, graph neural networks, self-supervised learning, cascading failures, power flow, explainability, robustness.

---

## I. Introduction (4–5 paragraphs)

**P1 — Motivation and stakes.**
Why utilities care: fast screening, near-real-time decision support, and the burden of running expensive solvers / simulations for labeling (tie to low-label framing). You already articulate the “labeled data scarcity + distribution shift + interpretability + scalability” drivers. 

**P2 — Why graphs/GNNs, and the gap.**
Power grids are naturally graphs; but purely supervised GNNs can be label-hungry and unstable in rare-event regimes (connect to IEEE‑118 10% instability). 

**P3 — Proposed approach at a glance.**
Introduce the PhysicsGuidedEncoder (admittance-weighted message passing) and SSL masked reconstruction. Mention it’s **one shared encoder** transferring to multiple heads (PF, line flow, cascade).  

**P4 — Contributions (bulleted).**
Keep it concrete and verifiable:

* Physics-guided message passing mechanism (admittance-inspired weighting). 
* Grid-specific SSL masked reconstruction objective and train-only pretraining disclosure.  
* Transfer gains in low-label regimes across tasks, and stabilizing effect on IEEE‑118 cascade.  
* Explainability fidelity via Integrated Gradients AUC-ROC. 
* Robustness trend under load scaling (clearly labeled as preliminary single-seed if kept). 

**P5 — Paper roadmap.**
One paragraph telling the reader what’s in Sections II–VII.

---

## II. Related Work (3–4 paragraphs)

You can keep this compact in an IEEE conference paper.

**P1:** Learning-based PF/OPF surrogates; GNNs for power networks.
**P2:** Physics-informed ML / inductive biases in graph models.
**P3:** Self-supervised learning on graphs (masked reconstruction / contrastive).
**P4:** Explainability for GNNs (IG/gradients/attention) and what “fidelity” means.

---

## III. Problem Setup and Task Definitions (3–4 paragraphs + 1–2 tables)

### III-A. Graph representation and features (1–2 paragraphs + a table)

**P1:** Define grid as graph (G=(V,E)) with node/edge attributes.
**P2:** Specify the exact feature vectors you use.

**Table A (recommended): “Graph Feature Definitions.”**
Use the node/edge feature table from your architecture doc: node features ([P_{net}, S_{net}, V]) and edge features ([P_{ij}, Q_{ij}, X_{ij}, rating]). 

### III-B. Downstream tasks and metrics (1–2 paragraphs + a table)

**P1:** Introduce the three tasks and why they matter operationally.
**P2:** Define the metric used per task and aggregation rules.

**Table B (keep, near the front): “Task Specifications with Units.”**
This already exists and is clean: cascade (graph-level F1), PF (V_mag MAE), line flow (P_ij,Q_ij MAE). 

### III-C. Deployment/observability assumptions (1 paragraph + a table)

**P1:** Clarify what’s assumed observable at inference and why it’s realistic.

**Table C (keep): “Required Inputs at Inference (Observability).”**
This is very reviewer-friendly. 

---

## IV. Method: Physics-Guided SSL GNN (6–8 paragraphs + 1 figure + 1 algorithm block)

### IV-A. PhysicsGuidedConv (1–2 paragraphs + equation)

**P1:** Start from standard message passing, then introduce admittance-weighted aggregation.
**P2:** Explain the intuition: electrically important lines influence representations more.

You can lift the core equation and explanation directly from the architecture doc. 

### IV-B. PhysicsGuidedEncoder stack (1 paragraph)

**P1:** 4 layers, residuals, LayerNorm, hidden dim 128 (keep details short here; put full hyperparams in Experiments). 

### IV-C. Task-specific heads (2–3 paragraphs)

**P1:** Explain “shared encoder, multiple heads” design.
**P2:** PF head outputs **V_mag only** (important to keep consistent). 
**P3:** LineFlow head outputs **[P_ij, Q_ij]** per edge; cascade head is graph-level probability. 

### IV-D. Self-supervised pretraining objective (2–3 paragraphs + optional algorithm)

**P1:** Motivation for SSL: exploit unlabeled training graphs to learn grid representations.
**P2:** Masked reconstruction: what is masked, how much (15%), what loss (MSE on masked positions). 
**P3:** No label leakage + pretraining uses train split only (explicit disclosure).  

**Algorithm 1 (optional but nice):** “SSL Pretraining and Fine-tuning Pipeline.”
Pseudo-code block: pretrain encoder on masked reconstruction → initialize downstream head → fine-tune with labeled fraction.

### IV-E. Explainability method (1 paragraph)

**P1:** Integrated Gradients for edge attribution; why it is preferred (completeness, path integral idea). 

**Figure 1 (recommended): “Overall Pipeline.”**
One schematic: input graph → physics-guided encoder → heads (cascade/PF/line flow) + SSL pretraining loop.

---

# V. Experimental Setup (detailed outline)

This is where IEEE reviewers look for **protocol clarity**. Your Simulation_Results already contains most of this, so the job is mostly *structuring* and *avoiding omission*.

## V-A. Dataset and splits (2 paragraphs + 2 tables)

**P1 — Dataset description.**
Name PowerGraph benchmark + which IEEE grids used. Provide per-unit normalization (100 MVA). 

**P2 — Split protocol and leakage prevention.**
80/10/10 split; cascade is stratified; metrics reported on held-out test only; validation for early stopping; SSL pretraining uses training partition only.  

**Table D:** “Dataset Statistics.”
Use the grid/buses/lines/samples/train/val/test table. 

**Table E:** “SSL Pretraining Data Split.”
Use the phase/source/samples/labels table + disclosure line. 

---

## V-B. Tasks, label fractions, and metrics (2–3 paragraphs + 1 table)

**P1 — Task definitions + metrics (brief recap).**
Point to Table B (task specs).

**P2 — Low-label protocol.**
Train with {10%, 20%, 50%, 100%} labeled subset; compare SSL-pretrained vs scratch. 

**P3 — “Improvement” computation.**
Spell out how improvement is computed differently for F1 vs MAE (already defined). 

**Table F (optional if space is tight):** “Seed Count and Evaluation Protocol.”
You already have seed justification per task/grid. 

---

## V-C. Model configuration and training (2 paragraphs + 1 table)

**P1 — Architecture settings.**
PhysicsGuidedEncoder, 4 layers, hidden dim 128, dropout etc. 

**P2 — Optimization + stopping.**
AdamW, epochs, batch size; validation-based early stopping.  

**Table G:** “Training Hyperparameters.”
Use the model configuration table. 

---

## V-D. Baselines (this should be explicit) (3 paragraphs + 1–2 tables)

**P1 — Why baselines are needed.**
Set context: trivial heuristics + feature-based ML to show value of graph structure.

**P2 — ML baselines.**
Explain feature representation (aggregated edge statistics for cascade; flattened injection for PF) + CV tuning procedure. 

**P3 — Heuristic baselines + tuning without leakage.**
Explicitly state thresholds are tuned on validation only; applied globally to all test graphs; and add “No test leakage” language. 

**Table H:** “ML Baseline Comparison (Cascade, PF).”
You already have it with 100% and 10% labels. 

**Table I:** “Heuristic Baselines (Cascade).”
Include Always Negative / Max Loading Threshold / Top‑K Loading Check vs GNN(SSL,10%). 

---

## V-E. Reproducibility and artifacts (1 paragraph + optional table)

**P1:** Provide a short “Code Availability / Reproduction” statement and point to the scripts + one-command figure generation. 

Optionally include a short table listing figures/tables produced (or place this in appendix / supplementary). The Progress report already enumerates figure filenames. 

---

# VI. Results (detailed outline)

I’d structure this so the reader gets the “big picture” first (main summary), then task-by-task, then optional robustness/explainability/ablation.

## VI-A. Main transfer summary across tasks and grids (2 paragraphs + 1 table)

**P1 — Headline takeaways.**
Introduce the central claim: SSL improves across tasks, strongest in low-label settings; mention IEEE-118 stabilization.

**P2 — Table walk-through.**
Explain what is reported (test-set mean ± std; seeds). Highlight the 10% regime results and the variance reduction story.

**Table 1 (your current main table): “SSL Transfer Benefits Across Tasks and Grid Scales.”**
This is already perfectly positioned to be Table I or Table II in the final paper (depending on how you number). 

*(Optional companion figure)*: If you want, pair this table with the multi-task bar plot at 10% labels (Figure 6 below). 

---

## VI-B. Cascade prediction on IEEE‑24 (2–3 paragraphs + 1 table + 2 figures + baseline table reference)

**P1 — Setup recap.**
Graph-level classification; positive class definition; F1 metric. 

**P2 — Low-label effect and learning curve.**
Discuss how SSL shifts performance upward especially at 10–20% labels, and why that matters operationally.

**P3 — Baseline context.**
One paragraph comparing against ML/heuristic baselines (briefly; refer to Tables H–I rather than reprinting if space is tight). 

**Table 2 (recommended): “Cascade SSL Transfer Results (IEEE‑24).”**
Use the LaTeX table you already prepared. 

**Figure 2a:** `cascade_ssl_comparison.png` (bar chart, IEEE‑24). 
**Figure 2b:** `cascade_improvement_curve.png` (improvement vs label fraction). 

*(If those figure names are already embedded in Simulation_Results earlier than the excerpt shown, just keep the captions consistent.)*

---

## VI-C. Scalability: Cascade prediction on IEEE‑118 (2–3 paragraphs + 1 table + 2 figures)

**P1 — Why IEEE‑118 is harder.**
More nodes/edges + stronger low-label instability; emphasize variance as the key phenomenon.

**P2 — Quantitative results and stability story.**
Explain that scratch at 10% is unstable (high σ) while SSL is stable; include ΔF1 language (better than % when baseline is near-random). 

**P3 — Practical implication.**
“Reliability over lucky seeds” argument: SSL yields consistent performance when data is scarce.

**Table 3:** IEEE‑118 cascade results across label fractions.
Use the table in the IEEE‑118 section (10/20/50/100 with ΔF1). 

**Figure 3a:** `cascade_118_ssl_comparison.png` (bar chart). 
**Figure 3b:** `cascade_118_improvement_curve.png` (curve with ΔF1). 

---

## VI-D. Power flow prediction (IEEE‑24) (2 paragraphs + 1 table + 2 figures)

**P1 — Task definition + units.**
Predict bus voltage magnitudes **V_mag**; report MAE in p.u. 

**P2 — Results and interpretation.**
Discuss improvements at 10%/20% and diminishing returns at full labels; optionally interpret what MAE means in operational voltage terms.

**Table 4:** Power Flow SSL transfer results (10/20/50/100).
Use your LaTeX table. 

**Figure 4a:** `pf_ssl_comparison.png` 
**Figure 4b:** `pf_improvement_curve.png` 

---

## VI-E. Line flow prediction (IEEE‑24) (2–3 paragraphs + 1 table + 2 figures)

**P1 — Task definition + units.**
Predict ([P_{ij}, Q_{ij}]) per line; MAE in p.u. 

**P2 — Results and low-label advantage.**
Highlight the 10% and 20% improvements.

**P3 — Address the 100% variance note (defensive writing).**
Explicitly include the “one outlier seed; median shown” note exactly once (and move per-seed to appendix). 

**Table 5:** Line Flow SSL transfer results (10/20/50/100).
Use the LaTeX table. 

**Figure 5a:** `lineflow_ssl_comparison.png` 
**Figure 5b:** `lineflow_improvement_curve.png` 

---

## VI-F. Cross-task synthesis (1–2 paragraphs + 1 table + 2 figures)

**P1 — Unified story.**
State the common pattern: largest gains at 10% labels, diminishing at 100%, and SSL improves stability on harder/larger settings.

**P2 — Scalability narrative.**
Use IEEE‑24 vs IEEE‑118 comparison to argue robustness of representation learning.

**Table 6:** “Cross-Task SSL Transfer Summary (10% labels).”
You already have LaTeX for it. 

**Figure 6:** `grid_scalability_comparison.png` 
**Figure 7:** `multi_task_comparison.png` 

---

## VI-G. Robustness under distribution shift (optional / “supporting evidence”) (1 paragraph + 1 table + optional figure)

**P1 — Stress test framing (must be cautious).**
Keep it explicitly “preliminary single-seed” if it remains single-seed. 

**Table 7:** Load scaling results (1.0× to 1.3×). 

**Optional Figure 8:** A simple line plot of F1 vs load multiplier (SSL vs scratch).
(You can generate this from the table; keep in appendix if space is tight.)

---

## VI-H. Explainability fidelity (optional but valuable) (1–2 paragraphs + 1 table + optional qualitative figure)

**P1 — Protocol.**
Explain ground truth (failed-edge masks) and metric (per-graph AUC-ROC averaged). 

**P2 — Results and implication.**
Integrated Gradients beats loading heuristic; indicates learned edge relevance beyond simple loading. 

**Table 8:** “Edge Attribution Fidelity (AUC-ROC).”
Use the method comparison table. 

**Optional Figure 9:** One qualitative visualization: a single IEEE‑24 case showing top‑k edges highlighted (IG vs loading heuristic).
This is often the figure reviewers remember.

---

## VI-I. Encoder ablation (optional / appendix depending on space) (1 paragraph + 1 table)

**P1 — What ablation isolates.**
Clarify it is scratch-only, single-seed, not directly comparable to multi-seed SSL numbers (you already wrote this). 

**Table 9:** Encoder comparison table (PhysicsGuided vs Vanilla vs GCN). 

---

## VII. Discussion (3–4 paragraphs)

**P1:** Why SSL helps most at extreme low-label; connect to representation initialization and grid structure (you have a good “why SSL works” narrative already). 
**P2:** Operational implications: observability table + no-oracle deployment note. 
**P3:** Scalability implications (IEEE‑118 stability). 
**P4:** Limitations (single benchmark; single-seed robustness if kept; need multi-grid generalization; etc.).

---

## VIII. Conclusion (1 paragraph)

One paragraph summarizing contributions + the strongest validated claims (multi-seed low-label improvements and IEEE‑118 stabilization), plus a forward-looking sentence.

---

## Appendices / Supplementary (recommended)

* Per-seed tables for any section where variance is discussed (e.g., line flow 100% outlier). 
* Extended robustness beyond load scaling (if you decide to include noise/edge-drop later).
* Full hyperparameter sweeps or additional baselines.

---

# Quick “exact artifact map” (what you already have)

**Core figures you can directly drop into the paper** (from your artifacts list):
`cascade_ssl_comparison.png`, `cascade_118_ssl_comparison.png`, `grid_scalability_comparison.png`, `pf_ssl_comparison.png`, `lineflow_ssl_comparison.png`, `cascade_improvement_curve.png`, `pf_improvement_curve.png`, `lineflow_improvement_curve.png`, `multi_task_comparison.png`. 

**Core LaTeX tables already drafted** (ready to embed):
PF table, line flow table, cross-task summary table. 
(Plus IEEE‑24 cascade LaTeX table already included in the results doc.) 

---

## Practical recommendation on what to keep in the *main* paper vs appendix

If this is an **IEEE conference** (tight pages), a strong “main paper” set is:

* Tables: Task specs, Dataset stats, Main results summary, + (one) detailed task table (cascade IEEE‑118 or PF)
* Figures: Pipeline schematic, IEEE‑118 cascade curve, one PF/LineFlow curve, multi-task summary bar plot
* Move: full label-fraction tables for all tasks + robustness + ablation to appendix

If this is a **journal/transactions** submission, include the full per-task label-sweep tables and keep robustness + explainability in the main text (they’re compelling supporting contributions).

---

If you want, I can take the above and convert it into a **camera-ready IEEE section template** (with exact Table/Figure numbering, captions, and where each figure should float), but this outline is already aligned to the exact assets and tables you’ve generated.


Absolutely — here’s a **camera-ready IEEE-style template plan** with **(i) exact section numbering**, **(ii) paragraph-by-paragraph intent**, and **(iii) an explicit Table/Figure map with captions, labels, and recommended float placement**. Everything is aligned to what’s already in your canonical results and architecture docs.   

---

## A. Float map (exact tables/figures, captions, labels, and where they go)

### Figures (recommended main-paper set)

**Fig. 1 (top of page 2, two-column * preferred):** *Method overview*

* **File:** *(new schematic you create; can be drawn in TikZ or exported PNG/PDF)*
* **Caption:** “Overview of the proposed physics-guided SSL pipeline: (a) train-only masked reconstruction pretraining of a shared encoder; (b) transfer to task-specific heads for cascade (graph-level), power flow (node-level), and line flow (edge-level).”
* **Label:** `fig:pipeline`

**Fig. 2a/2b (Results—Cascade IEEE-24, same page if possible):**

* **Files:** `cascade_ssl_comparison.png`, `cascade_improvement_curve.png` 
* **Caption:** “Cascade prediction on IEEE-24: (a) SSL vs. scratch at each label fraction; (b) relative improvement vs. label fraction.”
* **Label:** `fig:cascade24`

**Fig. 3a/3b (Results—Cascade IEEE-118):**

* **Files:** `cascade_118_ssl_comparison.png`, `cascade_118_improvement_curve.png` 
* **Caption:** “Scalability on IEEE-118 cascade prediction: (a) SSL vs. scratch; (b) improvement curve highlighting large ΔF1 at 10% labels.”
* **Label:** `fig:cascade118`

**Fig. 4a/4b (Results—Power Flow):**

* **Files:** `pf_ssl_comparison.png`, `pf_improvement_curve.png` 
* **Caption:** “Power flow (V_mag) prediction on IEEE-24: (a) MAE comparison; (b) improvement vs. label fraction.”
* **Label:** `fig:pf`

**Fig. 5a/5b (Results—Line Flow):**

* **Files:** `lineflow_ssl_comparison.png`, `lineflow_improvement_curve.png` 
* **Caption:** “Line flow (P_{ij}, Q_{ij}) prediction on IEEE-24: (a) MAE comparison; (b) improvement vs. label fraction.”
* **Label:** `fig:lineflow`

**Fig. 6 (end of Results, synthesis):**

* **Files:** `grid_scalability_comparison.png`, `multi_task_comparison.png` (use as (a)/(b) subfigures) 
* **Caption:** “Cross-task synthesis: (a) grid scalability comparison (IEEE-24 vs IEEE-118); (b) multi-task summary at 10% labels.”
* **Label:** `fig:synthesis`

### Tables (recommended main-paper set)

**TABLE I (end of Sec. III): Task definitions**

* **From:** “Task Specifications with Units” 
* **Caption:** “Task specifications, I/O, metrics, and units.”
* **Label:** `tab:tasks`

**TABLE II (Sec. V-A): Dataset stats**

* **From:** dataset table with IEEE-24/118 samples + splits 
* **Caption:** “PowerGraph benchmark datasets and train/val/test splits.”
* **Label:** `tab:data`

**TABLE III (Sec. V-C): Training configuration**

* **From:** model configuration + training protocol 
* **Caption:** “Model and optimization hyperparameters (shared across tasks unless noted).”
* **Label:** `tab:hparams`

**TABLE IV (start of Results): Main transfer summary**

* **From:** “Table 1: SSL Transfer Benefits Across Tasks and Grid Scales” 
* **Caption:** “SSL transfer benefits across tasks and grid scales (mean ± std over seeds).”
* **Label:** `tab:main`

**TABLE V (Results—Baselines): ML baselines**

* **From:** “Machine Learning Baselines” 
* **Caption:** “Feature-based ML baselines vs. GNN (scratch/SSL) on cascade and power flow.”
* **Label:** `tab:mlbaselines`

**TABLE VI (Results—Baselines): Heuristic cascade baselines**

* **From:** heuristic table + global tuning protocol 
* **Caption:** “Graph-level heuristic baselines for cascade prediction (thresholds tuned on validation only).”
* **Label:** `tab:heuristics`

**TABLE VII (Results—Explainability, optional but strong):**

* **From:** explainability AUC-ROC method comparison
* **Caption:** “Edge attribution fidelity for cascade prediction (AUC-ROC).”
* **Label:** `tab:explain`

**Appendix tables (recommended):** robustness load scaling, ablation encoder comparison, per-seed line-flow breakdown.

---

## B. Full IEEE section template (sections + paragraphs + exact float placements)

### TITLE + AUTHORS

(IEEE standard)

### ABSTRACT (1 paragraph)

* **1 paragraph**: problem → method → headline multi-task low-label gains → scalability (IEEE-118 stability) → optional mention of explainability/robustness as supporting evidence.

### INDEX TERMS

Power systems; graph neural networks; self-supervised learning; cascading failures; power flow; interpretability.

---

## I. INTRODUCTION (5 paragraphs)

**P1 — Motivation:** operational need + labeled data cost + rare event prediction. 
**P2 — Why GNNs, what’s missing:** topology + physics matter; supervised-only is label-hungry/unstable.
**P3 — Your idea:** physics-guided message passing + SSL pretraining + transfer to three tasks.
**P4 — Contributions (bullets):** physics-guided encoder, SSL pretext, transfer results, explainability fidelity, robustness trend.
**P5 — Roadmap:** Section II–VIII.

---

## II. RELATED WORK (3–4 paragraphs)

**P1:** learning surrogates for PF/OPF + grid ML.
**P2:** physics-informed ML / inductive biases.
**P3:** SSL on graphs (masked reconstruction).
**P4:** explainability for GNNs (IG etc.).

---

## III. PROBLEM FORMULATION (3 subsections)

### III-A. Graph representation (2 paragraphs)

**P1:** define grid graph (G=(V,E)).
**P2:** define node/edge features and labels for cascade + targets for PF/line flow.

> **Place:** **TABLE I** (`tab:tasks`) at end of Sec. III. 

### III-B. Tasks + metrics (1–2 paragraphs)

* Explicitly state: cascade is **graph-level F1**, PF predicts **V_mag**, line flow predicts **(P_{ij},Q_{ij})**.

### III-C. Inference-time observability (1 paragraph + short table if space)

* One paragraph stating what’s assumed observable at test/inference and the “no oracle” stance. 
* (Optional) a compact table here; otherwise keep it in Experiments.

---

## IV. METHOD (recommended: 4 subsections)

### IV-A. Physics-guided message passing (2 paragraphs + 1 equation)

* Define PhysicsGuidedConv and the learned “admittance-like” weighting intuition. 

### IV-B. Shared encoder + task heads (3 paragraphs)

* Encoder stack (layers/residual/LN).
* Heads: cascade head (graph pooling), PF head (node regression), line flow head (edge regression).

### IV-C. Self-supervised pretraining objective (3 paragraphs)

* Masking scheme (15% + 80/10/10), reconstruct masked positions only.
* Train-only pretraining disclosure (no val/test exposure). 

### IV-D. Explainability (1 paragraph)

* Integrated Gradients for edge attribution + how you evaluate fidelity against ground truth edge masks.

> **Place:** **Fig. 1** (`fig:pipeline`) at the end of Method (best as two-column `figure*`).

---

## V. EXPERIMENTAL SETUP (detailed; reviewers care)

### V-A. Datasets and splits (2 paragraphs + TABLE II)

* Describe PowerGraph benchmark; IEEE-24 and IEEE-118; per-unit base; 80/10/10 split; stratification for cascade. 

> **Place:** **TABLE II** (`tab:data`) here.

### V-B. Low-label protocol + seeds (2 paragraphs)

* Label fractions {10,20,50,100}; mean±std; seed counts per task/grid (justify IEEE-118).
* Include one explicit sentence: “All hyperparameters tuned on validation only; test used only once.” 

### V-C. Model/training details (2 paragraphs + TABLE III)

* Architecture config + optimizer + epochs + batch size + early stopping.

> **Place:** **TABLE III** (`tab:hparams`) here.

### V-D. Baselines (3 paragraphs + TABLE V + TABLE VI)

* ML baselines feature construction + tuning. 
* Heuristics: graph-level thresholds, global K/τ tuned on validation only. 

> **Place:** TABLE V (`tab:mlbaselines`) + TABLE VI (`tab:heuristics`) here (or in Results if space).

### V-E. Implementation + reproducibility (1 paragraph)

* Scripts and reproduction command (one paragraph).

---

## VI. RESULTS (detailed subsections + exact artifacts)

### VI-A. Main transfer summary across tasks and grids (2 paragraphs + TABLE IV)

**P1:** State the core claim: SSL improves across tasks, strongest at low labels. 
**P2:** Walk Table IV and highlight “IEEE-118 stabilizes training at 10% labels” with ΔF1 language. 

> **Place:** **TABLE IV** (`tab:main`) at start of Results.

### VI-B. Cascade prediction (IEEE-24) (3 paragraphs + Fig. 2)

**P1:** Task recap (graph-level label; DNS>0).
**P2:** Label fraction trend + emphasize 10–20%. 
**P3:** Compare vs baselines in one tight paragraph (reference Tables V–VI). 

> **Place:** **Fig. 2a/2b** (`fig:cascade24`) after this subsection.

### VI-C. Scalability: Cascade prediction (IEEE-118) (3 paragraphs + Fig. 3)

**P1:** Why IEEE-118 is harder (size + class imbalance + low-label instability).
**P2:** Emphasize variance reduction and ΔF1 at 10%. 
**P3:** Practical implication: reliability across seeds.

> **Place:** **Fig. 3a/3b** (`fig:cascade118`) here.

> **Important note (paper hygiene):** Your docs currently contain *both* “~20% positive” and “~5% positive” statements in different places; ensure the manuscript uses **one consistent rate** everywhere.

### VI-D. Power flow prediction (IEEE-24) (2 paragraphs + Fig. 4)

**P1:** Task definition and MAE aggregation; V_mag only.
**P2:** Result trend: largest gains at 10–20%; interpret p.u. MAE in context. 

> **Place:** **Fig. 4a/4b** (`fig:pf`) here.

### VI-E. Line flow prediction (IEEE-24) (3 paragraphs + Fig. 5)

**P1:** Task definition: predict (P_{ij},Q_{ij}) per edge.
**P2:** Trend + why SSL helps. 
**P3:** One sentence about the 100% variance/median caveat (then push per-seed to appendix).

> **Place:** **Fig. 5a/5b** (`fig:lineflow`) here.

### VI-F. Cross-task synthesis (1–2 paragraphs + Fig. 6)

* Restate the unifying pattern; show scalability + 10% multi-task summary. 

> **Place:** **Fig. 6a/6b** (`fig:synthesis`) here.

### VI-G. Explainability fidelity (optional but strong) (2 paragraphs + TABLE VII)

**P1:** Protocol and why fidelity matters.
**P2:** AUC-ROC results (IG best); implication: learned importance beyond loading heuristic.

> **Place:** **TABLE VII** (`tab:explain`) here.

### VI-H. Robustness under load scaling (appendix unless you have pages) (1 paragraph + appendix table/figure)

Keep it “preliminary/single-seed” if still single-seed.

### VI-I. Encoder ablation (appendix recommended)

Single-seed, scratch-only framing; include in appendix.

---

## VII. DISCUSSION (3–4 paragraphs)

* Why SSL helps low-label; why physics-guided weighting helps; deployment observability; limitations (OOD breadth, dataset scope).

## VIII. CONCLUSION (1 paragraph)

* Restate contributions + strongest validated claims.

---

## C. LaTeX skeleton with numbered float placeholders (IEEEtran)

> This is a **drop-in structure**; you’ll paste your existing LaTeX tables from `Simulation_Results.md` into the matching spots. 

```latex
\documentclass[conference]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath,amsfonts}
\usepackage{booktabs}
\usepackage{subcaption}

\begin{document}

\title{Physics-Guided Self-Supervised Graph Neural Networks for Power Grid Analysis}
\author{...}
\maketitle

\begin{abstract}
% 1 paragraph: motivation -> method -> headline results -> (optional) explainability/robustness
\end{abstract}

\begin{IEEEkeywords}
Power systems, graph neural networks, self-supervised learning, cascading failures, power flow, interpretability
\end{IEEEkeywords}

\section{Introduction}
% P1 motivation
% P2 gap
% P3 approach
% P4 contributions (bullets)
% P5 roadmap

\section{Related Work}
% 3-4 paragraphs

\section{Problem Formulation}
\subsection{Graph representation}
% 2 paragraphs

\subsection{Tasks and metrics}
% 1-2 paragraphs

\begin{table}[t]
\caption{Task specifications, I/O, metrics, and units.}
\label{tab:tasks}
\centering
% paste "Task Specifications with Units" here
\end{table}

\subsection{Inference-time observability}
% 1 paragraph (+ optional small table)

\section{Method}
\subsection{Physics-guided message passing}
% 2 paragraphs + equation

\subsection{Shared encoder and task-specific heads}
% 3 paragraphs

\subsection{Self-supervised pretraining objective}
% 3 paragraphs

\subsection{Explainability method}
% 1 paragraph

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/pipeline.pdf}
\caption{Overview of the proposed physics-guided SSL pipeline: (a) train-only masked reconstruction pretraining of a shared encoder; (b) transfer to task-specific heads for cascade, power flow, and line flow.}
\label{fig:pipeline}
\end{figure*}

\section{Experimental Setup}
\subsection{Datasets and splits}

\begin{table}[t]
\caption{PowerGraph benchmark datasets and train/val/test splits.}
\label{tab:data}
\centering
% paste dataset table here
\end{table}

\subsection{Low-label protocol and evaluation}
% include seeds + mean±std + "no test leakage" sentence

\subsection{Training details}
\begin{table}[t]
\caption{Model and optimization hyperparameters (shared across tasks unless noted).}
\label{tab:hparams}
\centering
% paste model configuration table here
\end{table}

\subsection{Baselines}
\begin{table}[t]
\caption{Feature-based ML baselines vs. GNN (scratch/SSL) on cascade and power flow.}
\label{tab:mlbaselines}
\centering
% paste ML baselines table here
\end{table}

\begin{table}[t]
\caption{Graph-level heuristic baselines for cascade prediction (thresholds tuned on validation only).}
\label{tab:heuristics}
\centering
% paste heuristic table here
\end{table}

\section{Results}
\subsection{Main transfer summary}
\begin{table*}[t]
\caption{SSL transfer benefits across tasks and grid scales (mean $\pm$ std over seeds).}
\label{tab:main}
\centering
% paste your main summary table here
\end{table*}

\subsection{Cascade prediction on IEEE-24}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/cascade_ssl_comparison.png}
\caption{Cascade prediction on IEEE-24: SSL vs. scratch across label fractions.}
\label{fig:cascade24a}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/cascade_improvement_curve.png}
\caption{Cascade prediction on IEEE-24: relative improvement vs. label fraction.}
\label{fig:cascade24b}
\end{figure}

\subsection{Scalability: Cascade prediction on IEEE-118}
% (same pattern: two figs, or subfigures if you prefer)

\subsection{Power flow prediction}
% (two figs)

\subsection{Line flow prediction}
% (two figs)

\subsection{Cross-task synthesis}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/multi_task_comparison.png}
\caption{Multi-task summary at 10\% labels.}
\label{fig:multitask}
\end{figure}

\subsection{Explainability fidelity (optional)}
\begin{table}[t]
\caption{Edge attribution fidelity for cascade prediction (AUC-ROC).}
\label{tab:explain}
\centering
% paste explainability table here
\end{table}

\section{Discussion}
% 3-4 paragraphs

\section{Conclusion}
% 1 paragraph

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

---

## D. Two practical IEEE “layout” tips (so this fits without fighting floats)

1. Put **Table I–III** in the first half (Problem + Setup), then use a **single wide Table IV** at start of Results (`table*`). Your main summary really benefits from being two-column. 
2. Use paired (a)/(b) plots as **two single-column figures** (as above) unless you have abundant space; IEEE often handles this better than cramped subfigures.

---
