## Big picture for the student team

You’re building a **reproducible research pipeline** around **PowerGraph** that can support a publication claim like:

> “A grid-specific self-supervised, physics-consistent GNN encoder improves PF/OPF learning (especially low-label / OOD), and transfers to cascading-failure prediction and explanation.”

That’s implementable because PowerGraph includes **PF + OPF learning tasks** and **cascading-failure data with ground-truth explanation masks** you can evaluate against.

Your students’ goal is to deliver:

1. **A clean dataset loader** (PowerGraph → PyTorch Geometric)
2. **Strong baselines** (PF + OPF, then cascade)
3. **Physics consistency metrics + regularization**
4. **Self-supervised pretraining** (grid-specific masking tasks)
5. **Transfer + explanation fidelity evaluation** (using the cascade “exp” ground truth)
6. **Reproducible artifacts** (configs, logs, plots, fixed splits)

---

## Data sources they need

### Required

1. **PowerGraph dataset + repo**

   * Provides power-flow (PF), optimal power flow (OPF), and cascading failure tasks.
   * Cascading-failure dataset includes **ground-truth explanation masks** (key for validating your “representation helps interpret propagation” claim).
   * Repo license: **CC BY 4.0** (students should respect attribution requirements in your repo + paper).

### Optional (for validation / stress testing later)

2. **pandapower**

   * Use it to sanity-check physics residual computations and to generate *additional* stress scenarios if you later decide to do solver-shift or OOD tests. (Treat this as optional so you don’t derail the core PowerGraph experiments.)

### Not required (don’t depend on it)

3. **SafePowerGraph code**

   * The paper is useful to cite for robustness framing, but the repo indicates code is not fully public yet—so don’t plan your implementation around importing it.

---

## Team structure (suggested roles)

You can assign 4–6 Master’s students as follows:

* **Student A: Data & splits lead** (PowerGraph ingestion → PyG, dataset integrity, train/val/test splits)
* **Student B: PF baseline + physics metrics** (PF model, angle handling, KCL-style mismatch)
* **Student C: OPF head + multitask training** (PF+OPF heads, loss balancing, negative transfer checks)
* **Student D: SSL pretraining** (masked tasks, pretrain→finetune experiments, low-label curves)
* **Student E: Cascade + explanation evaluation** (cascade classifier, explanation scoring vs `exp` masks)
* **Student F (optional): Reproducibility/DevOps** (configs, logging, “one-command reproduce figures”)

---

## Work packages (tasks + “definition of done” checks)

### WP0 — Repo scaffolding & reproducibility (everyone benefits)

**Tasks**

* Create repo structure:

  * `data/raw/`, `data/processed/`, `src/`, `configs/`, `scripts/`, `tests/`, `analysis/`
* Add environment setup:

  * `requirements.txt` or `environment.yml`
  * lock versions if possible (pip-tools/uv/conda-lock)
* Add experiment logging:

  * TensorBoard or W&B (either is fine—pick one and standardize)
* Add seed control:

  * A single `set_seed(seed)` used everywhere

**Checks (correctness)**

* A new person can run:

  * `python -m pip install -r requirements.txt`
  * `python scripts/smoke_test.py`
    and it completes without manual edits.
* `scripts/smoke_test.py` runs:

  1. data loader on a *tiny* subset
  2. one forward pass of the model
  3. prints shapes + a loss value

---

### WP1 — PowerGraph ingestion → PyTorch Geometric `Data` objects (Student A)

**Tasks**

* Download PowerGraph and record:

  * dataset version / commit hash
  * file checksum (for reproducibility)
* Write a loader that:

  * Parses PF dataset (node-level targets)
  * Parses OPF dataset (node-level targets + mask, if any)
  * Parses cascade dataset (graph-level labels + explanation masks from the provided files)
* Convert each scenario into a PyG `Data` object with:

  * `x` (node features)
  * `edge_index`
  * `edge_attr` (line features)
  * `y_pf` / `y_opf` (targets)
  * `y_cascade` (graph label)
  * `exp_mask` (edge-level explanation target where applicable)

**Checks (correctness)**

* **Graph sanity checks**

  * Node count and edge count are consistent across scenarios for the same grid (unless the dataset explicitly varies topology).
  * No NaNs / infs in `x` or targets.
* **Split sanity checks**

  * `train/val/test` splits are stable (saved to disk) and re-used across runs.
  * If the dataset contains time-like ordering, implement at least one “blocked split” option (no leakage).
* **Round-trip check**

  * Load → save processed `.pt` → reload yields identical tensors (bitwise or close).

Deliverable: `src/data/powergraph_dataset.py` + `scripts/inspect_dataset.py` that prints dataset stats.

---

### WP2 — PF baseline model (Student B)

**Tasks**

* Implement PF regression baseline:

  * Start with GraphSAGE / GCN / GAT (pick one simple baseline)
  * Output: voltage magnitude + voltage angle
* **Angle handling (important validity)**

  * Predict `sin(theta)` and `cos(theta)` instead of raw theta
  * Convert back to theta only for reporting

**Checks (correctness)**

* **Overfit test**

  * On a tiny subset (e.g., 32 graphs), the model should nearly overfit (loss drops dramatically).
* **Angle wrap check**

  * Construct two angles near +π and -π and confirm evaluation treats them as close (via sin/cos).
* **Baseline comparison**

  * Compare against a trivial predictor (e.g., predicting mean voltage magnitude). Your model must beat it.

Deliverable: `scripts/train_pf.py` + logged metrics + a small report with learning curves.

---

### WP3 — Physics consistency metric + optional regularization (Student B)

This is one of the “soundness anchors” for a publication: you’re not only predicting well; you’re predicting **physically consistent** states.

**Tasks**

* Implement a “physics residual” metric on PF outputs:

  * Given predicted voltages and network admittance info (or line parameters), compute a mismatch statistic per bus (active/reactive mismatch if possible).
  * Keep it as a **metric first**, loss term second.
* Add optional penalty term:

  * `loss_total = loss_pf + λ * physics_residual`

**Checks (correctness)**

* Compute physics residual on **ground-truth** PF labels:

  * It should be *near numerical tolerance* relative to residual on random predictions (orders of magnitude smaller).
* Turning on the penalty (λ > 0) should:

  * Reduce physics residual on validation
  * Not explode PF error (watch tradeoff)

Deliverable: `src/metrics/physics.py` + plots showing PF error vs physics residual across checkpoints.

---

### WP4 — OPF head + PF/OPF multi-task training (Student C)

**Tasks**

* Add OPF prediction head (based on what labels PowerGraph provides for OPF tasks).
* Implement multi-task training:

  * Shared encoder
  * PF head + OPF head
  * Loss weighting (static weights to start; optional GradNorm later)

**Checks (correctness)**

* **No-regression check**

  * PF performance should not collapse when OPF head is added.
* **Negative transfer check**

  * Track per-task gradients (optional) or at minimum observe if OPF improves while PF worsens.
* **Mask correctness**

  * If OPF targets are masked (only certain buses relevant), confirm masked loss ignores irrelevant nodes.

Deliverable: `scripts/train_pf_opf_multitask.py` + ablation: PF-only vs OPF-only vs multi-task.

---

### WP5 — Self-supervised pretraining (Student D)

This is a core novelty lever (if done grid-specifically, not just generic GraphMAE).

**Tasks**

* Implement one grid-specific SSL pretext task:

  * **Masked injection reconstruction**: mask Pd/Qd at random buses, reconstruct from topology + neighbors
  * OR **Masked edge feature reconstruction**: mask a fraction of edge attributes, reconstruct
* Pretrain encoder on SSL objective
* Fine-tune on PF and PF+OPF with low-label settings (e.g., 10/20/50%)

**Checks (correctness)**

* SSL loss decreases steadily and does not collapse to NaNs.
* **Linear probe check**

  * Freeze encoder, train a small head → should perform better than a randomly initialized encoder on PF (especially low-label).
* **Low-label curves**

  * At least one of the low-data settings shows a consistent improvement vs training-from-scratch.

Deliverable: `scripts/pretrain_ssl.py` + `scripts/finetune_from_ssl.py` + learning curves.

---

### WP6 — Cascading failure prediction + explanation fidelity evaluation (Student E)

PowerGraph cascade data includes labels and a ground-truth explanation mask file (high value for publishability).

**Tasks**

* Implement cascade prediction baseline:

  * Graph-level pooling (mean / attention pooling) + classifier head
* Transfer experiments:

  * Train cascade from scratch
  * Train cascade with encoder initialized from PF/OPF (supervised) and from SSL
* Implement explanation scoring:

  * Produce edge importance scores (start simple: gradient-based saliency or attention weights if using GAT)
  * Compare to ground-truth explanation mask via:

    * AUC
    * Precision@K (K = number of true important edges)

**Checks (correctness)**

* Random edge scores give ~chance AUC (sanity baseline).
* Correct alignment of `edge_index` with `exp_mask`:

  * If you permute edges, the metric should change (ensures you’re not accidentally comparing mismatched orderings).
* Transfer should show either:

  * Better accuracy/F1, or
  * Better explanation AUC/precision@K, or
  * Better robustness (under perturbations), even if accuracy is similar.

Deliverable: `scripts/train_cascade.py` + `analysis/explanation_eval.py` + a figure of explanation AUC across methods.

---

### WP7 — Robustness & OOD tests (Student F or shared)

This is where you can make the “representation learning” claim stronger without shaky “causal” language.

**Tasks**

* Define 2–3 perturbations:

  * Edge-drop at inference (simulate line outages)
  * Load scaling (e.g., multiply loads by 1.1–1.4)
  * Feature noise on injections
* Evaluate PF/OPF and cascade under perturbations for:

  * scratch vs multi-task vs SSL-pretrained

**Checks (correctness)**

* Performance degrades smoothly as perturbation increases (no bizarre discontinuities).
* SSL-pretrained model should be **less brittle** in at least one perturbation mode.

Deliverable: robustness plots (metric vs perturbation level) generated by a single script.

---

### WP8 — Paper-ready artifact checklist (everyone, but one owner)

**Tasks**

* One-command reproducibility:

  * `python analysis/run_all.py` reads logged CSVs and regenerates *all* figures and tables
* Store:

  * dataset split files
  * model configs
  * seeds
* Create a `MODEL_CARD.md`:

  * training data
  * limitations
  * physics residual behavior
  * calibration behavior

**Checks (correctness)**

* Delete local caches, re-run analysis from raw logs, confirm plots regenerate identically.
* A teammate can reproduce at least one key figure on their machine.

---

## A simple “done correctly” scorecard you can use each week

Ask each student to submit:

1. **PR link** (code merged to main)
2. **1-page experiment note** (what they ran, what changed, what they observed)
3. **Two screenshots**:

   * TensorBoard curves
   * A single sanity-check output (dataset stats, confusion matrix, AUC plot, etc.)

And you (or a TA) verify:

* The task’s explicit checks above passed
* Results are not based on accidental leakage (splits are fixed + saved)
* Plots are regenerated by script (not manual)

---

## If you want one “first implementation sprint” (start tomorrow)

**Sprint deliverables (minimum viable research pipeline):**

* WP0 + WP1 complete (loader + splits + smoke test)
* PF baseline trains end-to-end (WP2)
* One physics residual metric computed and logged (WP3 metric-only, even before regularization)

Once those are stable, SSL and cascade work won’t turn into debugging chaos.

If you want, I can turn the above into a GitHub Issues-style board (titles, acceptance criteria, dependencies) so you can paste it directly into your repo.



# New Changes 12/13/2025 19:10
Yes — your 8‑week plan is **largely aligned** with the WP structure I proposed, but there are **a few soundness-driven changes** I would make. None of them invalidate what you already did up to **WP2**, but they *do* change how I’d sequence and justify WP3→WP8 so the paper story is tighter and harder to attack in review.

Below I’ll do two things:

1. **Point out what I already accounted for vs what your 8‑week plan misses / risks**
2. **Give you an updated “from WP3 onward” plan** (assuming WP1–WP2 are done)

---

## 1) What in your 8‑week plan changes the work (and why)

### A) PowerGraph license & dataset reality

Your plan states **“MIT licence”** for PowerGraph. The PowerGraph repos/dataset distribution indicate **CC BY 4.0** (at least for the graph-level dataset repo and public dataset card). This doesn’t change your modeling work, but it **does change how you onboard, publish code, and attribute data**. ([GitHub][1])

**Action:** Update your README + paper “Data availability” section to reflect CC BY 4.0 and include attribution.

---

### B) Physics-guided message passing is good — but physics *metrics* should come first

Your Week 3 jumps straight into a physics-informed MessagePassing layer. That’s okay, but for **validity**, you want an objective “physics consistency” yardstick *before* you modify architecture.

PowerGraph provides physical quantities and parameters for PF/OPF and describes how inputs/outputs and masks are set up. ([arXiv][2])

**Action:** In WP3, implement **physics residual metrics** (and optionally a regularizer) *before* “physics-guided” message passing. Otherwise you can’t prove the physics change helped.

---

### C) SSL pretraining: your “line-masking predict masked line flow” may be mismatched across tasks

PowerGraph has:

* PF/OPF (node-level) where edge features are **conductance & susceptance**. ([arXiv][2])
* Cascades (graph-level) where edge features include **active/reactive flows, reactance, rating** and topology changes by removing outage edges. ([arXiv][2])

So “mask edges and predict line flow” is easiest if you’re using cascade graphs (they already have line flow as edge features), but it’s not as clean for PF/OPF-only pretraining unless you explicitly define “line flow target”.

**Action:** For SSL, make the pretext objective **grid-specific but label-free**, e.g.:

* masked **injection** reconstruction (Pd/Qd/Pg/Qg)
* masked **edge parameter** reconstruction
  Those transfer cleanly across PF and OPF.

(You can still do your line-flow masking as *a second* SSL objective, but don’t make it the only one.)

---

### D) Week 5: generating OPF labels with pandapower is not needed for the core claim

PowerGraph PF/OPF labels were generated using MATPOWER simulations. ([arXiv][2])
So you already have OPF supervision.

Generating “2,000 extra OPF labels with pandapower” can be valuable, but only if you frame it as:

* a **solver shift / domain shift** generalization test, or
* an **OOD regime** test.

Otherwise it’s extra complexity + potential inconsistencies.

**Action:** Remove pandapower OPF label generation from the critical path. Make it an optional Week 7 “bonus experiment” if you have bandwidth.

---

### E) Week 5: Jointly training PF+OPF+cascade is higher-risk than “PF/OPF pretrain → cascade transfer”

Cascade graphs can have edges removed (triggering outages), and the edge-feature semantics differ from PF/OPF. ([arXiv][2])

A “single encoder for everything” is still possible — but **sequential transfer** is much easier to make stable and defend:

1. SSL pretrain encoder
2. PF+OPF multi-task fine-tune
3. Transfer / fine-tune to cascade + explainability metrics

**Action:** Make cascade a transfer stage first; consider full 3-way joint training only after transfer works.

---

### F) Week 6: “epistemic vs aleatoric” needs careful meaning here

If OPF labels are deterministic functions of inputs (often true), “aleatoric uncertainty” isn’t naturally present unless you:

* inject input uncertainty (renewable forecast errors), or
* define a stochastic OPF target distribution.

**Action:** For soundness, do one of:

* **Epistemic-only UQ** (ensembles or MC dropout) + calibration
* Or explicitly define input perturbation distributions and call it **input uncertainty**, not “aleatoric” unless you truly have stochastic labels.

Also, doing MC-dropout *and* Pyro/NumPyro variational Bayes is overkill for an 8-week paper unless UQ is the main contribution.

---

### G) Week 7: “causal insights” is the most fragile framing

PowerGraph provides **ground-truth explanation masks** for cascading failures. ([arXiv][2])

So instead of “causal insights”, you can do something much more defensible:

> “Does representation learning improve **explanation fidelity** (AUC / Precision@K) vs ground-truth masks?”

**Action:** Replace “causal insights” with **explanation fidelity** + “propagation affinity” analysis.

---

### H) Minor but important: your venue note is outdated

Your Week 8 mentions NeurIPS 2025 submission in May 2025 — but today is Dec 2025, so that window is past. This doesn’t change experiments, but you should pick an upcoming venue (workshop 2026, IEEE TPS/TSTE rolling, etc.).

---

## 2) You finished WP2 — what *must* you retrofit now (before WP3)?

Before moving on, do these **quick validity checks** on your existing WP2 baseline:

### Retrofit 1 — Angle handling

If you are doing MSE on raw angles, switch to:

* predict **sin(θ)** and **cos(θ)**, or use a circular loss.

**Check:** error near ±π boundary behaves correctly (no huge penalty for wrap-around).

### Retrofit 2 — Masking / node-type logic

PowerGraph PF/OPF variables are bus-type dependent and uses **masks** (known inputs set, unknown set to 0; known outputs are masked in training). ([arXiv][2])

**Check:** Your loss only applies to the variables that are meant to be predicted (not the masked/known ones).

### Retrofit 3 — Baseline uses edge features

A plain GCN ignores edge attributes, but PowerGraph PF/OPF edge features are meaningful (conductance/susceptance). ([arXiv][2])

**Check:** Add at least one edge-aware baseline (GINEConv or TransformerConv) so your results aren’t dismissed as “weak baseline”.

---

## 3) Revised plan from WP3 onward (drop-in replacement)

Below is how I’d **change your remaining weeks** without throwing away what you’ve done.

---

# Week 3 (WP3) — Physics metrics first, then physics-guided message passing

### Objectives

* Add **physics consistency metrics** and a stable edge-aware backbone
* Optionally introduce a physics-guided MessagePassing layer *after* metrics exist

### Tasks

1. Implement **physics residual metric** (start simple; even a proxy is fine):

   * mismatch-like score per node from predicted voltages + edge params
2. Log both:

   * PF prediction error (MAE/MSE per variable)
   * physics residual (mean, 95th percentile)
3. Replace/extend GCN with an edge-aware baseline:

   * GINEConv or TransformerConv (edge_attr used)
4. Optional: implement your admittance-weighted custom message passing

### “Done correctly” checks

* Physics residual on **ground-truth** labels is ≪ residual on random predictions
* Edge-aware baseline beats GCN (or at least is competitive)
* Turning on physics penalty reduces residual without exploding PF error

---

# Week 4 (WP5) — SSL pretraining that transfers cleanly to PF/OPF

### Objectives

* Implement SSL that does **not depend on extra labels**
* Show **low-label gains** in PF (and later OPF)

### Tasks

Pick one SSL objective (add second later if time):

* **Masked injection reconstruction:** randomly mask Pd/Qd/Pg/Qg entries, predict them
* **Masked edge parameter reconstruction:** mask conductance/susceptance entries, predict them
  (These align with PF/OPF data structure.) ([arXiv][2])

Then:

1. Pretrain encoder for N epochs
2. Fine-tune on PF with 10/20/50% data splits
3. Run linear probe: freeze encoder, train small head

### Checks

* SSL loss decreases smoothly
* Frozen-encoder linear probe > random init
* Fine-tuning curves show improvement at 10–20% label regime

---

# Week 5 (WP4 + WP6) — PF/OPF multitask + cascade transfer (not 3-way joint yet)

### Objectives

* Add OPF head using **PowerGraph OPF labels** (no pandapower needed) ([arXiv][2])
* Build cascade model and evaluate transfer + explanation fidelity (ground-truth masks exist) ([GitHub][1])

### Tasks

1. Implement OPF dataset ingestion & OPF head (masked loss)
2. Multi-task PF+OPF training (shared encoder, two heads)
3. Cascade pipeline:

   * dataset loader verifies edge removals + edge features
   * graph-level classifier/regressor head
4. Transfer experiments:

   * scratch cascade
   * PF/OPF-pretrained encoder → cascade
   * SSL-pretrained encoder → cascade
5. Explanation fidelity:

   * produce edge importance scores (simple baseline: gradient*input or attention)
   * evaluate vs exp masks: AUC, Precision@K ([GitHub][1])

### Checks

* PF does not degrade catastrophically when adding OPF (monitor negative transfer)
* Transfer improves at least one of: cascade F1 / regression error / explanation AUC
* Random edge scores give chance-level AUC (sanity baseline)

---

# Week 6 (WP6) — Uncertainty quantification (keep it simple and defensible)

### Objectives

* Add **epistemic uncertainty** + calibration
* Avoid overclaiming “aleatoric” unless you model input noise explicitly

### Tasks (pick ONE approach)

**Option A (recommended): Deep ensembles**

* Train 3–5 models with different seeds
* Use ensemble variance as epistemic uncertainty

**Option B: MC Dropout**

* Add dropout in head, sample T forward passes

Then:

1. Evaluate calibration:

   * coverage of prediction intervals
   * reliability curve / ECE-like metric
2. Conformal calibration on validation split to get 90% intervals

### Checks

* Interval coverage close to nominal after conformal calibration
* Uncertainty increases under OOD perturbations (e.g., load scaling, line drops)

---

# Week 7 (WP7) — Robustness + explanation (replace “causal insights”)

### Objectives

* Robustness benchmarks
* Explanation fidelity benchmarks

### Tasks

1. Robustness tests:

   * load scaling
   * feature noise
   * topology perturbation (edge drop)
2. Explanation tests (cascade):

   * compare explanation AUC / Precision@K across methods
3. Optional: pandapower “solver-shift” OPF generation as an OOD evaluation (only if time)

### Checks

* Performance degrades smoothly as perturbation grows
* SSL / physics-guided variants are less brittle in at least one setting
* Explanation metrics are reproducible across seeds

---

# Week 8 (WP8) — Paper & reproducible artifact

### Objectives

* Submission-ready writing + reproducibility

### Tasks

* Paper template + sections
* Final ablation tables
* One-command figure regeneration (`analysis/run_all.py`)
* Document dataset license / attribution (CC BY 4.0) ([GitHub][1])

### Checks

* A new machine can reproduce key figure(s) from logs
* Splits + seeds are fixed and committed

---

## Summary: What changes for you, practically?

Since you’re done to WP2, you **don’t need to redo** anything unless:

* you trained raw angle MSE (fix now),
* you ignored masks (fix now),
* you used a GCN-only baseline (add one edge-aware baseline now).

Then the biggest **structural change** is:

✅ **Do PF/OPF first (SSL + multitask), then transfer to cascade**, rather than trying to train PF+OPF+cascade jointly immediately.

And the biggest **story change** is:

✅ Replace “causal insights” with **explanation fidelity vs ground-truth exp masks** (much more defensible). ([GitHub][1])

If you want, paste what you currently implemented for WP2 (model type, targets, loss, masking), and I’ll give you a very concrete “diff” of what to change in code structure for the next WPs.

[1]: https://github.com/PowerGraph-Datasets/PowerGraph-Graph "GitHub - PowerGraph-Datasets/PowerGraph-Graph"
[2]: https://arxiv.org/html/2402.02827 "PowerGraph: A power grid benchmark dataset for graph neural networks"
