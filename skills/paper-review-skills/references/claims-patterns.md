# Claims Audit Reference

## Claim Types

| Type | Definition | Evidence Required |
|------|------------|-------------------|
| performance | Quantitative metric improvement | Table cell, exact numbers |
| generalization | Transfer/robustness claims | Multi-dataset results, OOD tests |
| efficiency | Parameter/compute/data efficiency | Model size, training time, data fraction |
| novelty | First/new/unique claims | Literature gap citation |
| physics-validity | Domain correctness claims | Equation, reference to standard |

## Risk Assessment

**HIGH risk** (requires rewrite):
- Claim uses superlatives without qualification
- Evidence missing or in different document
- Statistical significance not reported
- Comparison unfair (different data splits, hyperparams)

**MED risk** (needs softening):
- Evidence exists but not directly cited
- Claim scope broader than evidence
- Missing confidence intervals

**LOW risk** (acceptable):
- Claim directly matches table/figure
- Properly qualified ("on IEEE-24 benchmark")
- Includes uncertainty

## Claim Ledger JSON Schema

```json
{
  "id": "C01",
  "claim": "SSL improves MAE by 29.1%",
  "location": "Abstract, Section 6.1",
  "claim_type": "performance",
  "evidence_present": true,
  "evidence_pointer": "Table 1, PF column, 10% row",
  "risk": "LOW",
  "missing_work": [],
  "rewrite_conservative": null
}
```

## Common Overclaim Patterns in Power Grid ML Papers

1. **Generalization overclaim**: "Generalizes to any power grid" → Fix: "Generalizes to IEEE-24 and IEEE-118 benchmarks"
2. **Efficiency overclaim**: "Lightweight model" → Fix: "274K parameters, 100× smaller than PPGT"
3. **Novelty overclaim**: "First to apply SSL to power grids" → Fix: "Extends SSL pretraining to cascade prediction, which prior work does not address"
4. **Robustness overclaim**: "Robust to distribution shift" → Fix: "Shows +22% advantage under 1.3× load scaling"
