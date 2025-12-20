# Utility-Scale Grid Integration Plan

## Objective

Add utility-scale (500+ bus) validation to the GNN power systems paper by integrating Texas A&M ACTIVSg synthetic grids. Start with ACTIVSg500 for pipeline validation, then scale to ACTIVSg2000.

---

## RunPod Instance Recommendations

### Recommended Instance: A100 PCIe 40GB

| Instance Type | VRAM | vCPU | RAM | Cost/hr | Use Case |
|---------------|------|------|-----|---------|----------|
| **A100 PCIe 40GB** | 40 GB | 16 | 125 GB | ~$1.50 | **Primary choice** - handles 2000-bus at batch=32 |
| A100 SXM4 80GB | 80 GB | 16 | 250 GB | ~$2.20 | If 40GB insufficient; allows larger batches |
| A6000 | 48 GB | 16 | 125 GB | ~$0.80 | Budget option; may need batch=16 |
| 4090 | 24 GB | 16 | 64 GB | ~$0.50 | Only for ACTIVSg500; too small for 2000-bus |

### Instance Setup Checklist

1. **Template**: PyTorch 2.x + CUDA 12.1
2. **Disk**: 100 GB minimum (data generation creates ~5 GB)
3. **Network Volume**: Optional - for persisting data across sessions
4. **Region**: Any (computation-bound, not network-bound)

### Environment Setup Steps

1. Clone your repository
2. Install dependencies:
   - PyTorch + CUDA
   - PyTorch Geometric + extensions
   - PYPOWER (for cascade simulation)
   - scipy, h5py (for MATLAB file I/O)
3. Verify GPU access
4. Download ACTIVSg case files from MATPOWER GitHub

---

## Phase 0: Preparation

### Data Sources

| Grid | Buses | Branches | Source URL |
|------|-------|----------|------------|
| ACTIVSg500 | 500 | ~600 | github.com/MATPOWER/matpower/blob/master/data/case_ACTIVSg500.m |
| ACTIVSg2000 | 2000 | ~3000 | github.com/MATPOWER/matpower/blob/master/data/case_ACTIVSg2000.m |

Both grids are public domain (Texas A&M synthetic grids based on real Texas ERCOT topology).

### Required MATLAB Files to Generate

For `PowerGraphDataset` compatibility, generate these 7 files in `data/raw/activsg{500,2000}/activsg{500,2000}/raw/`:

| File | Shape | Contents | How to Generate |
|------|-------|----------|-----------------|
| blist.mat | [2, num_edges] | Edge index (1-indexed bus pairs) | Extract from MATPOWER branch table |
| Bf.mat | cell[num_samples] of [3, num_nodes] | Node features (P_net, S_net, V) | Run power flow, extract bus data |
| Ef.mat | cell[num_samples] of [4, num_edges] | Edge features (P_ij, Q_ij, X, rating) | Run power flow, extract branch flows |
| of_bi.mat | [num_samples, 1] | Binary labels (cascade/no cascade) | From cascade simulation |
| of_mc.mat | [num_samples, num_classes] | Multiclass labels (one-hot) | Discretize DNS values |
| of_reg.mat | [num_samples, 1] | Regression labels (DNS in MW) | From cascade simulation |
| exp.mat | cell[num_samples] of variable | Explanation masks (tripped line indices) | Track which lines tripped |

**Note**: Files must be MATLAB v7.3 format (HDF5-based) for cell arrays with >2GB data.

---

## Phase 1: ACTIVSg500 Validation (3-4 days)

### Goal
Validate the entire pipeline end-to-end on a medium-scale grid before committing to ACTIVSg2000.

### Why Start with ACTIVSg500?
- 4x smaller than target (500 vs 2000 buses)
- Faster data generation (~2 hours vs ~12 hours)
- Faster training (~6 hours vs ~12+ hours)
- Early detection of pipeline issues
- If it fails, minimal time wasted

### Tasks

1. **Download case file** from MATPOWER GitHub
2. **Implement cascade simulator**:
   - Load case, perturb loads, apply contingencies
   - Simulate cascade (overload detection, line tripping)
   - Extract pre-outage features
   - Track tripped lines for explanations
3. **Generate dataset**: 5,000 scenarios
4. **Add grid to codebase**: Modify GRIDS list in powergraph.py
5. **Validate data loading**: Run inspect_dataset.py
6. **Smoke test training**: 5 epochs, verify loss decreases
7. **Full training**: 100 epochs, document results

### Success Criteria

- [ ] Power flow converges on base case
- [ ] Data generation completes (<3 hours)
- [ ] Dataset loads correctly
- [ ] Cascade F1 > 0.70 (comparable to IEEE-118)

### Estimated Time
- Data generation: 2-3 hours
- Training: 6-8 hours
- Total with debugging: 1-2 days

---

## Phase 2: ACTIVSg2000 Full Implementation (4-6 days)

### Prerequisites
- ACTIVSg500 validation passed
- Pipeline issues resolved

### Tasks

1. **Download case file** from MATPOWER GitHub
2. **Generate dataset**: 20,000 scenarios (matching IEEE-118 scale)
3. **Training configuration**:
   - Batch size: 32 (A100) or 16 (smaller GPUs)
   - Gradient accumulation if needed
   - Same hyperparameters as IEEE-118
4. **Run experiments**:
   - Baseline (no SSL)
   - SSL pretrained
   - Multi-seed evaluation (5 seeds)
   - Multiple label fractions (10%, 25%, 50%, 100%)
5. **Document results**

### Resource Estimates

| Task | Duration | GPU Hours |
|------|----------|-----------|
| Data generation | 6-12 hours | 12h |
| SSL pretraining | 4-6 hours | 6h |
| Cascade training (1 run) | 8-12 hours | 10h |
| Full experiment suite | 2 days | 40h |

### Memory Considerations

| Grid | Nodes | Edges | Memory/Graph | Batch=32 |
|------|-------|-------|--------------|----------|
| IEEE-118 | 118 | 372 | ~50 KB | ~2 GB |
| ACTIVSg500 | 500 | 1200 | ~200 KB | ~8 GB |
| ACTIVSg2000 | 2000 | 6000 | ~800 KB | ~25 GB |

A100 (40 GB) should handle batch=32 comfortably for ACTIVSg2000.

---

## Phase 3: Paper Integration

### New Results to Add

**Table: Scalability Across Grid Sizes**
| Grid | Buses | Task | Baseline F1 | SSL F1 | Improvement |
|------|-------|------|-------------|--------|-------------|
| IEEE-118 | 118 | Cascade | (existing) | (existing) | (existing) |
| ACTIVSg500 | 500 | Cascade | TBD | TBD | TBD |
| ACTIVSg2000 | 2000 | Cascade | TBD | TBD | TBD |

**Figure: SSL Benefit vs Grid Scale**
- X-axis: Grid size (log scale)
- Y-axis: Relative improvement from SSL
- Demonstrates scaling behavior

### Claims This Enables

With positive ACTIVSg2000 results:
- "SSL pretraining benefits scale to utility-size grids (2000+ buses)"
- "17x increase in grid size does not degrade SSL effectiveness"
- "Physics-informed architecture generalizes across diverse grid topologies"

---

## Timeline Summary

| Phase | Task | Duration | RunPod Hours |
|-------|------|----------|--------------|
| 0 | Environment setup | 0.5 days | 2h |
| 1 | ACTIVSg500 validation | 2 days | 12h |
| 2 | ACTIVSg2000 full experiment | 3-4 days | 50h |
| 3 | Paper integration | 1 day | - |
| **Total** | | **6-8 days** | **~65h** |

**Estimated RunPod cost**: $65-100 (A100 @ $1.50/hr)

---

## Code Changes Required

### Files to Create
1. `scripts/generate_cascading_data.py` - Cascade simulation and data export
2. `scripts/validate_activsg.py` - Data integrity checks

### Files to Modify
1. `src/data/powergraph.py` line 70: Add "activsg500", "activsg2000" to GRIDS list
2. `scripts/inspect_dataset.py` lines 292, 306: Add to argparse choices
3. `configs/base.yaml`: Add grid options to comments

---

## Cascade Simulation Approach

### Algorithm Outline

```
1. Load base case (ACTIVSg500 or ACTIVSg2000)
2. For each scenario:
   a. Perturb loads randomly (+/- 20%)
   b. Run base power flow to get pre-outage state
   c. Apply initial contingency (randomly remove 1-3 lines)
   d. Cascade simulation loop:
      - Compute power flow
      - Identify overloaded lines (|S| > 110% rating)
      - Trip overloaded lines
      - Check for islands, apply load shedding if needed
      - Repeat until no new trips or max iterations
   e. Record:
      - Pre-outage node features (P_net, S_net, V)
      - Pre-outage edge features (P_flow, Q_flow, X, rating)
      - Binary label (DNS > 0?)
      - Regression label (DNS value)
      - Explanation mask (which lines tripped)
3. Export to MATLAB v7.3 format (HDF5-based)
```

### Tools

| Tool | Language | Recommendation |
|------|----------|----------------|
| PYPOWER | Python | Primary choice - pure Python, easy integration |
| pandapower | Python | Alternative - better AC support |
| MATPOWER + Oct2Py | Python/MATLAB | Fallback - if PYPOWER fails on large grids |

---

## Risk Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PYPOWER doesn't converge on 2000-bus | Medium | High | Use DC approximation; try pandapower |
| OOM during training | Low | Medium | Reduce batch size; gradient checkpointing |
| Unrealistic cascade statistics | Medium | Medium | Validate DNS distribution; tune parameters |
| ACTIVSg500 fails | Low | High | Provides early warning; debug before 2000-bus |
| Data generation too slow | Low | Low | Parallelize; use multiple instances |

---

## Decision Points

### After ACTIVSg500 (Go/No-Go)

**Go**: If cascade F1 > 0.70 and pipeline runs smoothly
**No-Go**: If major issues with convergence, data format, or training

### After ACTIVSg2000 Data Generation

**Go**: If 20k scenarios generated with reasonable cascade rate (20-40%)
**No-Go**: If most scenarios fail to converge or cascade rate is extreme

---

## Validation Checklists

### ACTIVSg500 Checkpoint
- [ ] Power flow converges on base case
- [ ] Cascade simulation produces reasonable DNS distribution
- [ ] Data loads via PowerGraphDataset without errors
- [ ] Training loss decreases normally
- [ ] Final F1 > 0.70

### ACTIVSg2000 Checkpoint
- [ ] Power flow converges on base case
- [ ] 20k scenarios generated successfully
- [ ] Training runs without OOM on A100
- [ ] SSL pretrained model improves over baseline
- [ ] Results ready for paper tables

---

## Appendix: Alternative Approaches

If ACTIVSg2000 proves too challenging:

1. **OPFData for PF/OPF only**: DeepMind's dataset has grids up to 6470 buses for power flow/line flow prediction. No cascade data, but demonstrates scalability for two of three tasks.

2. **Strengthen IEEE-118**: More seeds (10→20), more ablations, additional robustness tests. Argue utility-scale is future work.

3. **Subgrid extraction**: Extract 500-bus connected subregions from ACTIVSg2000 if full simulation fails.

4. **Use existing benchmarks**: Check if any pre-computed cascading failure datasets exist for large grids (literature search needed).
