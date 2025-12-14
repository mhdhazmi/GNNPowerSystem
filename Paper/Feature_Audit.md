# Feature Usage Audit Report

**Date:** December 13, 2025
**Status:** AUDIT COMPLETE - NO LABEL LEAKAGE FOUND

---

## Executive Summary

After a thorough code audit, **the implementation does NOT have label leakage**. The confusion arose from misleading documentation and class names, not actual implementation issues.

---

## Audit Findings by Task

### Power Flow (PF) Task

| Component | Features |
|-----------|----------|
| **Node Input (x)** | P_net, S_net (2 features) |
| **Edge Input** | X, rating (2 features) |
| **Target (y)** | V (voltage) |

**SSL Pretraining (`pretrain_ssl_pf.py` with `--task pf`):**
- Masks: P_net, S_net (node features)
- Reconstructs: P_net, S_net (same node features)
- **V is NOT in the input, NOT masked, NOT reconstructed**

**Verdict: NO LEAKAGE**

### Optimal Power Flow (OPF) Task

| Component | Features |
|-----------|----------|
| **Node Input (x)** | P_net, S_net, V (3 features) |
| **Edge Input** | X, rating (2 features) |
| **Target (y)** | P_flow, Q_flow (edge flows) |

**SSL Pretraining (`pretrain_ssl_pf.py` with `--task opf`):**
- Masks: X, rating (edge features)
- Reconstructs: X, rating (same edge features)
- **Flows are NOT in the input, NOT masked, NOT reconstructed**

**Verdict: NO LEAKAGE**

### Cascade Task

| Component | Features |
|-----------|----------|
| **Node Input (x)** | P_net, S_net, V (3 features) |
| **Edge Input** | P_flow, Q_flow, X, rating (4 features) |
| **Target (y)** | Binary cascade label |

**SSL Pretraining (`pretrain_ssl.py`):**
- Node mask: All 3 node features (P_net, S_net, V)
- Edge mask: All 4 edge features (P_flow, Q_flow, X, rating)

**Verdict: NO DIRECT LEAKAGE**
- The target is a binary label, not V or flows
- V and flows are grid state variables that exist in the input
- This is valid SSL: learning to reconstruct grid state from partial observations

---

## Source of Confusion

### Misleading Class Name
```python
class MaskedVoltageSSL:  # Name suggests voltage masking
    """
    SSL model for PF task: masks voltage and learns to reconstruct it.  # WRONG!
    """
    def __init__(self, node_in_dim: int = 2):  # Actually only P_net, S_net
```

The class is named "MaskedVoltageSSL" and the docstring claims it "masks voltage", but:
- The actual node_in_dim for PF task is 2 (P_net, S_net)
- V is the TARGET, not part of the input
- The class masks P_net/S_net, reconstructs P_net/S_net

### Misleading Progress Report

The Progress_Report.md stated:
> "For Power Flow (Node-level):
> - Mask 15% of node voltage features
> - Objective: Reconstruct original voltage from power injections"

This is incorrect. The actual behavior is:
> "For Power Flow (Node-level):
> - Mask 15% of node injection features (P_net, S_net)
> - Objective: Reconstruct injections from graph structure"

---

## Code Locations Verified

| File | What I Checked |
|------|----------------|
| `src/data/powergraph.py:326-334` | PF task: x=P_net,S_net, y=V, edge_attr=X,rating |
| `src/data/powergraph.py:336-344` | OPF task: x=P_net,S_net,V, y=flows, edge_attr=X,rating |
| `src/data/powergraph.py:346-382` | Cascade task: full features |
| `scripts/pretrain_ssl_pf.py:33-114` | MaskedVoltageSSL: masks x, reconstructs x |
| `scripts/pretrain_ssl_pf.py:117-197` | MaskedFlowSSL: masks edge_attr, reconstructs edge_attr |

---

## Required Fixes

### Fix 1: Rename Class and Update Docstring

```python
# Before
class MaskedVoltageSSL:
    """SSL model for PF task: masks voltage and learns to reconstruct it."""

# After
class MaskedInjectionSSL:
    """SSL model for PF task: masks power injections (P_net, S_net) and reconstructs them."""
```

### Fix 2: Rename OPF SSL Class

```python
# Before
class MaskedFlowSSL:
    """SSL model for OPF task: masks edge flows and learns to reconstruct them."""

# After
class MaskedLineParamSSL:
    """SSL model for OPF task: masks line parameters (X, rating) and reconstructs them."""
```

### Fix 3: Update Progress Report

Correct the methodology section to accurately describe what is being masked.

---

## Conclusion

**The implementation is sound.** The SSL pretraining correctly:
1. For PF: Masks input-side features (P/Q injections), not the target (V)
2. For OPF: Masks input-side features (line parameters), not the target (flows)
3. For Cascade: Masks grid state variables, which is valid since the target is a binary label

The only issue is misleading documentation that made reviewers concerned about leakage. This should be fixed to avoid confusion.
