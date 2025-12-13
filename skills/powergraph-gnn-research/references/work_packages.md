# Work Packages & Acceptance Criteria

## Team Structure (4-6 Students)

| Role | Responsibility |
|------|----------------|
| **Student A** | Data & splits (PowerGraph → PyG, integrity, train/val/test) |
| **Student B** | PF baseline + physics metrics (angle handling, KCL mismatch) |
| **Student C** | OPF head + multitask training (loss balancing, negative transfer) |
| **Student D** | SSL pretraining (masked tasks, low-label experiments) |
| **Student E** | Cascade + explanation evaluation (explanation scoring vs exp.mat) |
| **Student F** | Reproducibility/DevOps (configs, logging, one-command reproduce) |

---

## WP0: Repo Scaffolding & Reproducibility

### Tasks
- Create repo structure: `data/raw/`, `data/processed/`, `src/`, `configs/`, `scripts/`, `tests/`, `analysis/`
- Add environment: `requirements.txt` with locked versions
- Add experiment logging: TensorBoard or W&B (pick one, standardize)
- Add seed control: single `set_seed(seed)` used everywhere

### Correctness Checks
```bash
# A new person can run:
python -m pip install -r requirements.txt
python scripts/smoke_test.py
# Should complete without manual edits

# smoke_test.py must:
# 1. Load tiny data subset
# 2. One forward pass
# 3. Print shapes + loss value
```

---

## WP1: PowerGraph Ingestion → PyG Data Objects

### Tasks
- Download PowerGraph, record version/commit hash + file checksum
- Write loader that parses:
  - PF dataset (node-level targets)
  - OPF dataset (node-level targets + mask)
  - Cascade dataset (graph-level labels + explanation masks from exp.mat)
- Convert to PyG `Data` with: `x`, `edge_index`, `edge_attr`, `y_pf`, `y_opf`, `y_cascade`, `exp_mask`

### Correctness Checks

**Graph sanity:**
```python
# Node/edge counts consistent across scenarios for same grid
assert len(data.x) == expected_num_buses
assert data.edge_index.shape[1] == expected_num_edges * 2

# No NaNs/infs
assert not torch.isnan(data.x).any()
assert not torch.isinf(data.y_pf).any()
```

**Split sanity:**
```python
# Splits saved to disk and reused
# train_indices.json, val_indices.json, test_indices.json
# If time-ordered data, use blocked split (no leakage)

# For 1-year 15-min data:
train_months = [1,2,3,4,5,6,7,8,9]   # 75%
val_months = [10]                     # 8%  
test_months = [11,12]                 # 17%
```

**Round-trip check:**
```python
# Load → save → reload yields identical tensors
torch.save(dataset, 'processed.pt')
reloaded = torch.load('processed.pt')
assert torch.allclose(dataset[0].x, reloaded[0].x)
```

### Deliverable
`src/data/powergraph_dataset.py` + `scripts/inspect_dataset.py`

---

## WP2: PF Baseline Model

### Tasks
- Implement PF regression: GraphSAGE/GCN/GAT (pick one baseline)
- Output: voltage magnitude + voltage angle
- **CRITICAL: Angle handling** - predict sin(θ), cos(θ), not raw θ

### Correctness Checks

**Overfit test:**
```python
# On tiny subset (32 graphs), loss should drop dramatically
# If loss plateaus high → architecture/data bug
```

**Angle wrap check:**
```python
# Two angles near +π and -π should evaluate as close
angle1 = torch.tensor(3.1)   # near π
angle2 = torch.tensor(-3.1)  # near -π
# Via sin/cos: angular_error should be ~0.08 rad, not ~6.2 rad
sin1, cos1 = torch.sin(angle1), torch.cos(angle1)
sin2, cos2 = torch.sin(angle2), torch.cos(angle2)
error = torch.atan2(sin1-sin2, cos1-cos2).abs()  # Small
```

**Baseline comparison:**
```python
# Model must beat trivial predictor (mean voltage)
trivial_mae = (y_true - y_true.mean()).abs().mean()
model_mae = (y_pred - y_true).abs().mean()
assert model_mae < trivial_mae
```

### Deliverable
`scripts/train_pf.py` + logged metrics + learning curves

---

## WP3: Physics Consistency Metric + Regularization

### Tasks
- Implement physics residual metric on PF outputs
- Given predicted voltages + network admittance → compute mismatch per bus
- Keep as **metric first**, loss term second

### Correctness Checks

**Ground-truth residual:**
```python
# Physics residual on ground-truth PF labels should be near numerical tolerance
gt_residual = compute_physics_residual(y_true, admittance_matrix)
random_residual = compute_physics_residual(torch.randn_like(y_true), admittance_matrix)
assert gt_residual < 1e-4  # Near zero
assert random_residual > 1.0  # High
assert gt_residual < random_residual * 1e-3  # Orders of magnitude smaller
```

**Regularization effect:**
```python
# λ > 0 should:
# - Reduce physics residual on validation
# - NOT explode PF error (watch tradeoff)
```

### Deliverable
`src/metrics/physics.py` + plots showing PF error vs physics residual

---

## WP4: OPF Head + Multi-Task Training

### Tasks
- Add OPF prediction head (generator setpoints, costs)
- Implement multi-task training: shared encoder + PF head + OPF head
- Loss weighting (static to start; optional GradNorm later)

### Correctness Checks

**No-regression check:**
```python
# PF performance should NOT collapse when OPF head added
pf_only_mae = train_pf_only(...)
multitask_pf_mae = train_multitask(...)['pf_mae']
assert multitask_pf_mae < pf_only_mae * 1.1  # At most 10% worse
```

**Negative transfer check:**
```python
# Monitor if OPF improves while PF worsens
# If so → task conflict, consider separate optimizers or task weights
```

**Mask correctness:**
```python
# If OPF targets masked (only certain buses), loss must ignore irrelevant
masked_loss = loss * opf_mask
assert (masked_loss[~opf_mask] == 0).all()
```

### Deliverable
`scripts/train_pf_opf_multitask.py` + ablation: PF-only vs OPF-only vs multi-task

---

## WP5: Self-Supervised Pretraining

### Tasks
- Implement grid-specific SSL pretext task:
  - **Masked injection reconstruction**: mask Pd/Qd at random buses, reconstruct
  - OR **Masked edge feature reconstruction**: mask fraction of edge attributes
- Pretrain encoder on SSL objective
- Fine-tune on PF and PF+OPF with low-label settings (10/20/50%)

### Correctness Checks

**SSL stability:**
```python
# Loss decreases steadily, no NaN collapse
assert not torch.isnan(ssl_loss)
assert ssl_loss_epoch_10 < ssl_loss_epoch_1
```

**Linear probe check:**
```python
# Freeze encoder, train small head
# Should beat randomly initialized encoder
ssl_pretrained_mae = finetune_frozen(ssl_encoder)
random_init_mae = finetune_frozen(random_encoder)
assert ssl_pretrained_mae < random_init_mae
```

**Low-label curves:**
```python
# At least one low-data setting shows improvement vs scratch
# 10% labeled data + SSL pretrain < 10% labeled data scratch
```

### Deliverable
`scripts/pretrain_ssl.py` + `scripts/finetune_from_ssl.py` + learning curves

---

## WP6: Cascade Prediction + Explanation Fidelity

### Tasks
- Implement cascade prediction: graph-level pooling + classifier
- Transfer experiments: scratch vs PF/OPF-pretrained vs SSL-pretrained
- Explanation scoring: edge importance vs ground-truth exp.mat masks

### Correctness Checks

**Random baseline:**
```python
# Random edge scores → ~0.5 AUC
random_scores = torch.rand(num_edges)
random_auc = compute_auc(random_scores, exp_mask)
assert 0.45 < random_auc < 0.55  # Chance level
```

**Edge alignment:**
```python
# If you permute edges, metric should change
# Ensures not comparing mismatched orderings
original_auc = compute_auc(scores, exp_mask)
shuffled_auc = compute_auc(scores[perm], exp_mask)
assert original_auc != shuffled_auc  # Unless by chance
```

**Transfer benefit (at least one of):**
```python
# Better accuracy/F1, OR
# Better explanation AUC/precision@K, OR
# Better robustness under perturbations
```

### Deliverable
`scripts/train_cascade.py` + `analysis/explanation_eval.py` + explanation AUC figure

---

## WP7: Robustness & OOD Tests

### Tasks
- Define perturbations: edge-drop (5/10/20%), load scaling (×1.1-1.4), feature noise
- Evaluate PF/OPF and cascade under perturbations
- Compare: scratch vs multi-task vs SSL-pretrained

### Correctness Checks

**Smooth degradation:**
```python
# Performance degrades smoothly as perturbation increases
# No bizarre discontinuities
mae_clean < mae_5pct_drop < mae_10pct_drop < mae_20pct_drop
```

**SSL robustness:**
```python
# SSL-pretrained should be less brittle in ≥1 perturbation mode
ssl_degradation = ssl_perturbed_mae - ssl_clean_mae
scratch_degradation = scratch_perturbed_mae - scratch_clean_mae
assert ssl_degradation < scratch_degradation  # In at least one mode
```

### Deliverable
Robustness plots (metric vs perturbation level) generated by single script

---

## WP8: Paper-Ready Artifacts

### Tasks
- One-command reproducibility: `python analysis/run_all.py`
- Store: dataset split files, model configs, seeds
- Create MODEL_CARD.md

### Correctness Checks

**Regeneration:**
```bash
# Delete local caches, re-run analysis from raw logs
rm -rf analysis/figures/*
python analysis/run_all.py
# Plots regenerate identically
```

**Cross-machine:**
```python
# Teammate can reproduce ≥1 key figure on their machine
```

### Deliverable
Complete reproducibility package ready for paper submission

---

## Weekly "Done Correctly" Scorecard

Each student submits:
1. **PR link** (code merged to main)
2. **1-page experiment note** (what they ran, what changed, observations)
3. **Two screenshots**: TensorBoard curves + sanity-check output

Verify:
- [ ] Task's explicit checks passed
- [ ] Results not based on accidental leakage (splits fixed + saved)
- [ ] Plots regenerated by script (not manual)
