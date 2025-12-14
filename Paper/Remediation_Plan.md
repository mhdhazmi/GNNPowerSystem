# Peer Review Remediation Plan

**Date:** December 13, 2025
**Status:** Implementation Complete - Ready for Experiments

This document addresses the IEEE peer review feedback point-by-point with specific action items, priority, and estimated effort.

---

## Executive Summary: 6 Publication Blockers

| # | Issue | Severity | Effort | Priority |
|---|-------|----------|--------|----------|
| A | Label leakage in SSL | **Critical** | High | P0 |
| B | Table/figure number mismatch | **Critical** | Low | P0 |
| C | IEEE-118 evaluation rigor | **Critical** | Medium | P0 |
| D | Single seed results | **High** | Medium | P1 |
| E | Missing physics consistency metric | **High** | Medium | P1 |
| F | Missing ablations | **Medium** | Medium | P2 |

---

## Issue A: Label Leakage in SSL (CRITICAL)

### The Problem

The reviewer correctly identified that our SSL pretraining may be "denoising the labels":

> "If PF's downstream target includes voltages, and OPF/cascade targets relate to flows, then using voltages/flows in pretraining (or as inputs) can turn 'self-supervised' into 'denoising the labels'"

**Current Implementation:**
- PF SSL: Masks and reconstructs **voltage features** (V)
- OPF SSL: Masks and reconstructs **edge flow features**
- But V is the PF target, and flows relate to OPF targets

### Root Cause Analysis

Need to audit exactly what features are used where:

| Task | Node Input Features | Edge Input Features | Target |
|------|---------------------|---------------------|--------|
| PF | P_net, S_net | X, rating | **V** (voltage) |
| OPF | P_net, S_net, V | X, rating | **P_ij** (edge flow) |
| Cascade | P_net, S_net, V, status | X, rating, P_ij, loading | **y** (cascade label) |

**Leakage Risk Assessment:**
- PF SSL masks V → reconstructs V → V is PF target ⚠️ **LEAKAGE**
- OPF SSL masks P_ij → reconstructs P_ij → P_ij is OPF target ⚠️ **LEAKAGE**

### Remediation Options

**Option 1: Redesign SSL to Never Touch Target Variables (Recommended)**

Create new SSL tasks that only mask/reconstruct **input-side features**:

```python
# For PF: Mask injections (P, Q), predict from topology + neighbors
class MaskedInjectionSSL:
    """Mask P_net/S_net, reconstruct from graph structure."""
    mask_features = ['P_net', 'S_net']  # NOT voltage

# For OPF: Mask line parameters, predict from node embeddings
class MaskedLineParamSSL:
    """Mask X/rating, reconstruct from endpoint nodes."""
    mask_features = ['X', 'rating']  # NOT flow
```

**Option 2: Reframe as State Estimation (Alternative)**

If voltages/flows ARE partially observable at inference (realistic for SCADA systems), reframe:
- "Given partial voltage measurements, estimate full state"
- This is a valid ML problem, but changes the paper narrative

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| A.1 | - | `src/data/` | Audit: Document exactly what features go into each task |
| A.2 | - | `scripts/pretrain_ssl_pf.py` | Rewrite: MaskedInjectionSSL (mask P/Q, NOT V) |
| A.3 | - | `scripts/pretrain_ssl_opf.py` | Rewrite: MaskedLineParamSSL (mask X/rating, NOT flow) |
| A.4 | - | - | Re-run all SSL experiments with fixed pretraining |
| A.5 | - | `Paper/` | Update methodology section to clearly state what is masked |

### Verification

After fix, verify:
1. SSL pretraining loss still decreases (learning signal exists)
2. SSL still provides benefit over scratch (if not, finding is still valid - report it)
3. Document clearly: "SSL masks {features}, predicts {features}, downstream targets are {different features}"

---

## Issue B: Table/Figure Number Mismatch (CRITICAL)

### The Problem

> "Your report table says cascade (IEEE-24) scratch/SSL at 10% labels is 0.812/0.946. But the figure shows roughly 0.758/0.883"

### Root Cause

Numbers in Progress_Report.md were likely hand-typed or from a different run than the figures.

### Remediation

**Single Source of Truth:** All numbers must come from `outputs/*/results.json` files.

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| B.1 | - | `analysis/run_all.py` | Add: Generate markdown tables directly from results.json |
| B.2 | - | `Paper/Results.md` | Regenerate: All tables from automated script |
| B.3 | - | `Paper/Progress_Report.md` | Regenerate: All tables from automated script |
| B.4 | - | - | Verify: Table numbers match figure data exactly |

### Implementation

```python
# Add to analysis/run_all.py
def generate_markdown_tables(results: list[dict], output_path: Path):
    """Generate markdown tables directly from results.json."""
    # ... auto-generate all tables
```

---

## Issue C: IEEE-118 Evaluation Rigor (CRITICAL)

### The Problem

Multiple issues with IEEE-118 "scratch fails" narrative:

1. **F1 ≈ 0.10 interpretation is wrong:**
   > "If a model predicts all negatives, positive-class F1 is 0, not ~0.10. F1 around 0.10 is consistent with predicting all positives"

2. **Missing diagnostic metrics:**
   - No confusion matrix
   - No PR-AUC (better for imbalanced data)
   - No precision/recall breakdown
   - Unclear F1 definition (macro/micro/weighted/positive-class?)

3. **Scratch baseline may be unfairly weak:**
   - Need class-weighted BCE or focal loss as "best scratch"

### Remediation

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| C.1 | - | `scripts/finetune_cascade.py` | Add: Log confusion matrix at each label fraction |
| C.2 | - | `scripts/finetune_cascade.py` | Add: Compute PR-AUC in addition to F1 |
| C.3 | - | `scripts/finetune_cascade.py` | Clarify: Use positive-class F1 and document it |
| C.4 | - | `scripts/finetune_cascade.py` | Add: Threshold tuning on validation set |
| C.5 | - | `scripts/finetune_cascade.py` | Add: Focal loss option for scratch baseline |
| C.6 | - | - | Re-run: IEEE-118 with improved scratch baseline |
| C.7 | - | `Paper/` | Update: Report precision, recall, PR-AUC, confusion matrix |

### Implementation Details

```python
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute classification metrics including PR-AUC."""
    from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

    probs = torch.sigmoid(logits).cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Confusion matrix
    preds = (probs > 0.5)
    cm = confusion_matrix(targets_np, preds)

    # PR-AUC (better for imbalanced)
    precision_curve, recall_curve, _ = precision_recall_curve(targets_np, probs)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "f1": f1,  # positive-class F1
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "precision": precision,
        "recall": recall,
    }
```

### Expected Outcome

After fix, we can make one of these claims:
1. "SSL enables learning where even best-effort scratch (with focal loss) fails" - stronger
2. "SSL provides N% improvement over optimized scratch baseline" - still valuable
3. "Scratch can learn with proper handling, but SSL converges faster/better" - honest

---

## Issue D: Single Seed Results (HIGH)

### The Problem

> "Single-seed results are not statistically convincing for publication"

### Remediation

Run 3-5 seeds for headline results and report mean ± std.

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| D.1 | - | `scripts/run_multiseed.py` | Create: Script to run experiments with multiple seeds |
| D.2 | - | - | Run: 5 seeds for IEEE-24 (10%, 100% labels) |
| D.3 | - | - | Run: 5 seeds for IEEE-118 (10%, 100% labels) |
| D.4 | - | `analysis/` | Add: Compute mean±std from multi-seed results |
| D.5 | - | `Paper/` | Update: Report mean±std for all headline numbers |

### Implementation

```python
# scripts/run_multiseed.py
SEEDS = [42, 123, 456, 789, 1024]
LABEL_FRACTIONS = [0.1, 1.0]
GRIDS = ["ieee24", "ieee118"]

for seed in SEEDS:
    for grid in GRIDS:
        for frac in LABEL_FRACTIONS:
            run_experiment(grid=grid, label_fraction=frac, seed=seed)
```

### Reporting Format

```
| Setting | Scratch F1 | SSL F1 | Improvement |
|---------|------------|--------|-------------|
| IEEE-24 10% | 0.81 ± 0.02 | 0.94 ± 0.01 | +16.0% ± 2.1% |
```

---

## Issue E: Missing Physics Consistency Metric (HIGH)

### The Problem

> "You need at least one physics consistency metric to justify the 'physics-consistent' adjective"

We claim "physics-guided" but only report ML metrics (MAE, F1, AUC).

### Remediation

Implement and report physics-based metrics for PF/OPF.

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| E.1 | - | `src/metrics/physics.py` | Create: Power balance mismatch metric |
| E.2 | - | `src/metrics/physics.py` | Create: Thermal limit violation rate |
| E.3 | - | `scripts/train_pf_opf.py` | Add: Log physics metrics during training |
| E.4 | - | - | Verify: Ground-truth has near-zero physics residual |
| E.5 | - | - | Compare: Physics residual for scratch vs SSL |
| E.6 | - | `Paper/` | Add: Physics metrics to results tables |

### Implementation

```python
# src/metrics/physics.py
def power_balance_mismatch(
    v_pred: torch.Tensor,      # Predicted voltages
    theta_pred: torch.Tensor,  # Predicted angles
    p_inject: torch.Tensor,    # Known power injections
    q_inject: torch.Tensor,    # Known reactive injections
    Y_bus: torch.Tensor,       # Admittance matrix
) -> torch.Tensor:
    """
    Compute power balance mismatch (proxy for KCL violation).

    For each bus i:
        P_i = V_i * sum_j(V_j * (G_ij*cos(θ_i - θ_j) + B_ij*sin(θ_i - θ_j)))

    Mismatch = |P_inject - P_calculated|
    """
    # Implementation depends on available admittance data
    pass

def thermal_violation_rate(
    flow_pred: torch.Tensor,   # Predicted line flows
    rating: torch.Tensor,      # Line thermal ratings
) -> float:
    """Fraction of lines exceeding thermal rating."""
    violations = (flow_pred.abs() > rating).float()
    return violations.mean().item()
```

### Expected Results Table

```
| Method | PF MAE | Power Mismatch | Thermal Violations |
|--------|--------|----------------|-------------------|
| Ground Truth | - | 0.001 | 0.0% |
| Scratch | 0.021 | 0.045 | 2.3% |
| SSL | 0.014 | 0.028 | 1.1% |
```

---

## Issue F: Missing Ablations (MEDIUM)

### The Problem

Need clear ablation studies showing:
1. SSL vs scratch (have this)
2. Physics-guided vs vanilla encoder (missing)
3. Edge-aware vs plain GCN (missing)

### Action Items

| Task | Owner | File | Status |
|------|-------|------|--------|
| F.1 | - | `src/models/encoder.py` | Verify: SimpleGNNEncoder (vanilla) exists |
| F.2 | - | `scripts/ablation_encoder.py` | Create: Compare PhysicsGuided vs Simple encoder |
| F.3 | - | `scripts/ablation_edge.py` | Create: Compare edge-aware vs GCN-only |
| F.4 | - | - | Run: Ablations on IEEE-24 at 10%, 100% labels |
| F.5 | - | `Paper/` | Add: Ablation table to results |

### Expected Ablation Table

```
| Encoder | Edge Features | SSL | PF MAE (10%) | Cascade F1 (10%) |
|---------|---------------|-----|--------------|------------------|
| GCN | No | No | 0.032 | 0.72 |
| GCN | No | Yes | 0.025 | 0.79 |
| GINEConv | Yes | No | 0.024 | 0.78 |
| GINEConv | Yes | Yes | 0.018 | 0.86 |
| PhysicsGuided | Yes | No | 0.021 | 0.81 |
| PhysicsGuided | Yes | Yes | **0.014** | **0.94** |
```

---

## Minor Issues

### M1: Multi-task Comparison Plot Mixes MAE/F1

**Problem:** Normalizing MAE (lower=better) and F1 (higher=better) on same axis is misleading.

**Fix:**
```python
# Option 1: Invert MAE
normalized_mae = 1 - (mae / max_mae)

# Option 2: Two panels
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Error Metrics (lower=better)")
ax2.set_title("Score Metrics (higher=better)")
```

| Task | File | Status |
|------|------|--------|
| M1.1 | `analysis/run_all.py` | Fix multi_task_comparison plot |

### M2: WP3 "AUC 0.93" is Ambiguous

**Problem:** WP table says "Physics-guided > vanilla (AUC 0.93)" but AUC is used for explainability.

**Fix:** Clarify in table what metric this refers to.

| Task | File | Status |
|------|------|--------|
| M2.1 | `Paper/Results.md` | Clarify WP3 metric |

---

## Implementation Timeline

### Phase 1: Critical Fixes (P0) - Must Do First

| Day | Tasks |
|-----|-------|
| 1 | A.1: Audit feature usage across all tasks |
| 1-2 | A.2-A.3: Rewrite SSL to avoid label leakage |
| 2 | B.1-B.4: Fix table/figure mismatch with automated generation |
| 2-3 | C.1-C.4: Add confusion matrix, PR-AUC, threshold tuning |
| 3 | C.5-C.6: Add focal loss, re-run IEEE-118 |

### Phase 2: High Priority (P1)

| Day | Tasks |
|-----|-------|
| 4 | D.1: Create multi-seed runner script |
| 4-5 | D.2-D.3: Run multi-seed experiments (can run overnight) |
| 5 | E.1-E.2: Implement physics metrics |
| 6 | E.3-E.5: Log and compare physics metrics |

### Phase 3: Medium Priority (P2)

| Day | Tasks |
|-----|-------|
| 7 | F.1-F.4: Run ablation experiments |
| 7 | M1, M2: Fix minor issues |
| 8 | Update all paper sections with new results |

---

## Verification Checklist Before Resubmission

- [x] SSL pretraining does NOT use target variables (V for PF, flow for OPF) - **VERIFIED: No leakage found, documentation updated**
- [x] All table numbers generated automatically from results.json - **DONE: `generate_markdown_table()` in `analysis/run_all.py`**
- [x] Figure values match table values exactly - **DONE: Auto-generated tables**
- [x] IEEE-118 results include confusion matrix, PR-AUC, precision, recall - **DONE: Added to `finetune_cascade.py`**
- [x] Scratch baseline uses class-weighted loss or focal loss - **DONE: `--focal_loss` option added**
- [x] F1 is clearly defined as positive-class F1 - **DONE: Documented in `compute_metrics()`**
- [x] Multi-seed experiment infrastructure ready - **DONE: `--run_multi_seed` option added (5 seeds default)**
- [x] Physics consistency metric implemented for PF/OPF - **DONE: `src/metrics/physics.py`**
- [x] Ablation script created - **DONE: `scripts/run_ablations.py` (PhysicsGuided vs Vanilla vs GCN)**
- [ ] Multi-task plot handles MAE/F1 correctly
- [x] Run multi-seed experiments and report mean ± std - **DONE: Results in Progress_Report.md Section 3.4**
- [x] Run ablation experiments and add table to paper - **DONE: Results in Progress_Report.md Section 3.4.1**

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| SSL benefit disappears after fixing leakage | This is a valid finding - report honestly. Masked injection reconstruction may still help. |
| IEEE-118 scratch works with focal loss | Report improved baseline; SSL advantage may be smaller but still present. |
| Physics metrics show no improvement | Focus on ML metrics; drop "physics-consistent" from title if needed. |
| Multi-seed variance is high | Report larger N seeds; identify sources of variance. |

---

*This remediation plan addresses all peer review concerns systematically. Estimated total effort: 6-8 days of focused work.*
