# Positioning Reference

## Closest Prior Art

### PPGT (Physics-informed Pre-trained Graph Transformer)
- **What they do**: Transformer-based, physics-informed pretraining for power grids
- **Strengths**: Strong baselines, thorough experiments
- **Gaps we fill**:
  1. No cascade prediction (we do)
  2. Much larger model (2-15M params vs our 274K)
  3. No explainability analysis (we have Integrated Gradients)
  4. Different SSL approach (we use masked reconstruction)

### Other GNN-for-Power-Systems Work
- PowerGraph, DeepOPF, etc.
- Typically supervised-only
- Often single-task

## Positioning Table Template

| Method | PF | Line Flow | Cascade | SSL | Explainability | Params |
|--------|:--:|:---------:|:-------:|:---:|:--------------:|:------:|
| PPGT | ✓ | ✓ | ✗ | ✓ | ✗ | 2-15M |
| DeepOPF | ✓ | ✗ | ✗ | ✗ | ✗ | ~1M |
| **Ours** | ✓ | ✓ | ✓ | ✓ | ✓ | 274K |

## Novelty Framing Language

**Avoid**:
- "First ever", "novel", "breakthrough", "unique"
- "Outperforms all baselines" (unless statistically verified)

**Use**:
- "Extends prior work to cascade prediction"
- "Achieves comparable accuracy with 100× fewer parameters"
- "Provides interpretable predictions via Integrated Gradients"
- "Demonstrates SSL benefits for training stabilization on larger grids"

## Related Work Structure

1. **GNNs for power systems** — survey existing, note most are supervised
2. **Self-supervised learning for graphs** — general SSL, note limited power grid applications
3. **Cascade failure prediction** — note challenge and limited ML approaches
4. **Physics-informed neural networks** — contrast with physics-guided architecture
5. **Gap statement** — no prior work combines SSL + GNN + cascade + explainability

## Differentiation Statement Template

"While [prior work] addresses [their task] using [their approach], our method differs in three ways: (1) [technical difference], (2) [scope difference], (3) [efficiency/interpretability difference]. Specifically, [concrete comparison with numbers]."
