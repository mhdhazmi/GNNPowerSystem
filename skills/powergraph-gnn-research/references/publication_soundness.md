# Publication Soundness & Validity Guide

## Core Publication Claim (Defensible)

> "A grid-specific masked pretraining objective + physics-consistency regularization yields a single encoder that improves PF/OPF accuracy and robustness, and transfers better to cascade prediction and explanation—especially in low-label regimes."

This claim is **anchored** to:
- PowerGraph benchmark (real, published, CC BY 4.0)
- Ground-truth explanation masks for cascade tasks
- SafePowerGraph's emphasis on robustness (cite but don't depend on code)

---

## Reviewer Risk Mitigation

### Risk 1: "Just stacking known components"

**Objection:** "This is GNN + SSL + multi-task + dropout. Where is the new insight?"

**Defense:**
- Grid-specific SSL objective (masked injection = forces learning Kirchhoff's laws)
- Physics-consistency as a first-class metric, not just accuracy
- Explanation fidelity evaluation against ground-truth masks (novel evaluation)
- Clear ablations isolating each component's contribution

### Risk 2: "Causal claims are overstated"

**Objection:** "Ranking edges by embedding similarity is not causal identification."

**Defense:** Do NOT use "causal" unless implementing:
- Proper causal discovery with identifiability assumptions
- Interventional experiments (do-calculus)

**Safe alternative framing:**
- "Non-local propagation signatures"
- "Learned vulnerability proximity"
- "Representation-aligned failure affinity"

Evaluate via explanation AUC against exp.mat, NOT causal claims.

### Risk 3: "Results may not generalize"

**Defense:**
- Test on multiple IEEE grids (24, 39, 118-bus)
- Robustness evaluation under perturbations
- Blocked time splits (not random) for temporal data
- Report both IID and OOD performance

---

## Critical Validity Anchors

### 1. Angle Handling

**Problem:** Direct MSE on voltage angles fails near ±π (wrap-around).

**Solution:** Always predict (sin θ, cos θ) and recover via atan2.

```python
# CORRECT
sin_pred, cos_pred = model(x)
angle_pred = torch.atan2(sin_pred, cos_pred)
angle_mae = angular_distance(angle_pred, angle_true).mean()

# WRONG - silent validity bug
angle_pred = model(x)  # Raw angle
loss = F.mse_loss(angle_pred, angle_true)  # Fails at wrap-around
```

**Validation:**
```python
# Test with angles near ±π
a1, a2 = 3.14, -3.14  # Nearly identical angles
sin_cos_error = angular_distance_sincos(a1, a2)  # ~0.003 rad
naive_error = abs(a1 - a2)  # ~6.28 rad (WRONG)
```

### 2. Physics Residual Metric

**Problem:** Model may fit labels but violate physical laws.

**Solution:** Report physics residual (KCL/KVL mismatch) alongside accuracy.

```python
# P_inject = P_gen - P_load
# P_flow_sum = sum of outgoing flows from bus
# Mismatch = P_inject - P_flow_sum (should be ~0)

def physics_residual(V_pred, theta_pred, Y_bus):
    # Compute power flows from predicted state
    P_calc, Q_calc = compute_power_injection(V_pred, theta_pred, Y_bus)
    P_mismatch = P_calc - P_specified
    return torch.abs(P_mismatch).mean()
```

**Expected values:**
| Source | Physics Residual |
|--------|------------------|
| Ground truth | < 1e-4 (numerical) |
| Trained model | < 0.01 (good) |
| Random predictions | > 1.0 (bad) |

### 3. Blocked Time Splits

**Problem:** PowerGraph PF/OPF uses 1-year load at 15-min resolution. Random splits leak seasonal patterns.

**Solution:** Blocked time splits.

```python
# CORRECT - Blocked splits
train_scenarios = filter(lambda s: s.month in [1,2,3,4,5,6,7,8,9], scenarios)
val_scenarios = filter(lambda s: s.month == 10, scenarios)
test_scenarios = filter(lambda s: s.month in [11,12], scenarios)

# WRONG - Random split leaks temporal patterns
train, val, test = random_split(scenarios, [0.75, 0.08, 0.17])
```

**Report both:** If results differ significantly between random and blocked splits, that itself is a finding (temporal structure matters).

### 4. Cascade Explanation Ground Truth

**Problem:** "Embedding analysis" claims need quantitative backing.

**Solution:** PowerGraph provides `exp.mat` with ground-truth explanation masks.

```python
# Load explanation ground truth
exp_mask = load_explanation_mask('exp.mat')  # Binary edge mask

# Compute model's edge importance
edge_scores = model.get_edge_importance(data)  # Attention, gradient, etc.

# Evaluate explanation fidelity
auc = roc_auc_score(exp_mask, edge_scores)
precision_at_k = precision_at_k(exp_mask, edge_scores, k=exp_mask.sum())
```

**Report:**
- Explanation AUC (higher = better alignment with true cascading edges)
- Precision@K where K = number of truly important edges
- Compare: scratch vs SSL-pretrained vs physics-regularized

### 5. Statistical Significance

**Problem:** Small improvements may be noise.

**Solution:** Proper statistical testing.

```python
# For comparing two models across multiple seeds/datasets
from scipy.stats import ttest_rel, bootstrap

# Paired t-test (same test scenarios)
t_stat, p_value = ttest_rel(model_a_errors, model_b_errors)
if p_value < 0.05:
    print("Statistically significant difference")

# Bootstrap confidence intervals
ci = bootstrap((errors,), np.mean, confidence_level=0.95)
print(f"95% CI: [{ci.confidence_interval.low:.4f}, {ci.confidence_interval.high:.4f}]")
```

**Report:** Error bars, p-values, or bootstrap CIs for key comparisons.

---

## Aleatoric vs Epistemic Uncertainty

### Key Issue

OPF labels are typically **deterministic** (given inputs, solution is unique). So "aleatoric uncertainty" is ill-defined unless you explicitly model input noise.

### Options

**Option A: Epistemic only (simpler, recommended for first paper)**
- Use deep ensembles (5 members, different seeds)
- Epistemic uncertainty = variance of ensemble predictions
- Calibrate with conformal prediction

**Option B: Input noise model (more complex)**
- Explicitly perturb inputs (load, generation, line parameters)
- Define stochastic OPF with input distribution
- Aleatoric = expected variance given input distribution
- Epistemic = variance due to model uncertainty

### Calibration Check

```python
# Expected Calibration Error (ECE)
# For 90% confidence interval, ~90% of true values should be inside
coverage = (y_true >= lower_bound) & (y_true <= upper_bound)
actual_coverage = coverage.float().mean()
ece = abs(actual_coverage - 0.90)  # Should be small
```

---

## Negative Results That Are Still Publishable

### Multi-task hurts (negative transfer)
- Show WHY (task conflict diagnostics, gradient cosine similarity)
- Propose fix (adapter heads, task-specific layers)
- This is a contribution

### SSL gives small IID gains
- Show SSL improves **robustness** under perturbations
- Important for safety-critical applications
- Aligns with SafePowerGraph framing

### Physics regularization hurts accuracy slightly
- But reduces physics violations significantly
- Discuss tradeoff: accuracy vs physical plausibility
- May be critical for deployment in real grids

---

## What NOT to Claim

| Avoid | Instead Say |
|-------|-------------|
| "Causal inference from embeddings" | "Learned propagation affinity" |
| "Proves physical laws are satisfied" | "Reduces physics residual by X%" |
| "Generalizes to all power grids" | "Evaluated on IEEE 24/39/118" |
| "Production-ready OPF surrogate" | "Demonstrates feasibility for..." |

---

## Ablation Structure for Publication

### Required Ablations

1. **Architecture ablation**
   - GCN vs GAT vs physics-guided
   - With/without physics regularization

2. **Training regime ablation**
   - Single-task (PF only, OPF only, Cascade only)
   - Multi-task (PF + OPF)
   - SSL pretrain → fine-tune

3. **Data efficiency curves**
   - 10%, 20%, 50%, 100% training data
   - Show where SSL helps most

4. **Robustness ablation**
   - Clean vs edge-drop (5%, 10%, 20%)
   - Clean vs load scaling (×1.1, ×1.2, ×1.4)

### Table Template

```
Table 1: PF Prediction Performance (IEEE 118-bus)

Method              | V_MAE↓  | θ_MAE↓  | Physics↓ | 10% data
--------------------|---------|---------|----------|----------
GCN (baseline)      | 0.XXX   | X.XX°   | X.XXX    | 0.XXX
+ Physics reg       | 0.XXX   | X.XX°   | X.XXX    | 0.XXX  
+ Multi-task        | 0.XXX   | X.XX°   | X.XXX    | 0.XXX
+ SSL pretrain      | 0.XXX   | X.XX°   | X.XXX    | 0.XXX
--------------------|---------|---------|----------|----------
Improvement         | -XX%    | -XX%    | -XX%     | -XX%
```

---

## Target Venues

### Primary (Power Systems + ML)
- **IEEE Transactions on Power Systems (TPS)** - flagship, high impact
- **IEEE Transactions on Sustainable Energy (TSTE)** - if emphasizing renewables
- **Applied Energy** - broader audience

### Secondary (ML Venues)
- **NeurIPS ML for Systems Workshop** - May 2025 submission
- **ICML ML4Science Workshop**
- **ICLR Workshop on AI for Science**

### Conference (Faster Turnaround)
- **IEEE PES General Meeting**
- **ARPA-E Grid Science Conference**
- **IEEE SmartGridComm**

---

## Pre-Submission Checklist

### Reproducibility
- [ ] Fixed random seeds (document in paper)
- [ ] Dataset version/hash recorded
- [ ] Split indices saved to disk
- [ ] requirements.txt with pinned versions
- [ ] One-command script reproduces all figures

### Scientific Validity
- [ ] Angle handling uses sin/cos (not raw angles)
- [ ] Physics residual reported alongside accuracy
- [ ] Blocked time splits (not random)
- [ ] Statistical significance tests with p-values or CIs
- [ ] Ablations isolate each contribution

### Explanation Evaluation
- [ ] Explanation AUC computed against exp.mat ground truth
- [ ] Precision@K reported
- [ ] Random baseline AUC ~0.5 confirmed

### Claims Verification
- [ ] No "causal" claims without causal methodology
- [ ] Limitations section acknowledges scope
- [ ] Generalization claims match evaluation scope
