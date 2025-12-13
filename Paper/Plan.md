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
