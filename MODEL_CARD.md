# Model Card: Physics-Guided GNN for Power Systems

## Model Overview

**Model Name**: PhysicsGuidedEncoder
**Version**: 1.0
**Task Types**: Cascade Failure Prediction, Power Flow (PF), Optimal Power Flow (OPF)
**Architecture**: Graph Neural Network with physics-informed message passing

### Primary Claim

> "A grid-specific self-supervised, physics-consistent GNN encoder improves PF/OPF learning (especially low-label / OOD), and transfers to cascading-failure prediction and explanation."

---

## Model Architecture

### PhysicsGuidedEncoder

```
Input → Node Embedding → [PhysicsGuidedConv + LayerNorm + ReLU + Residual] × N → Output
```

**Key Components**:

| Component | Description |
|-----------|-------------|
| PhysicsGuidedConv | Message passing weighted by learned admittance |
| Admittance Scaling | `edge_attr → sigmoid(Linear(edge_attr))` for physics-based weighting |
| Residual Connections | Skip connections for gradient flow |
| Layer Normalization | Stable training across graph sizes |

**Default Hyperparameters**:

| Parameter | Value |
|-----------|-------|
| Hidden Dimension | 128 |
| Number of Layers | 4 |
| Dropout | 0.1 |
| Activation | ReLU |

**Parameter Counts**:

| Task | Node Features | Edge Features | Parameters |
|------|---------------|---------------|------------|
| Cascade | 4 | 4 | ~270K |
| PF | 2 | 2 | ~274K |
| OPF | 3 | 2 | ~168K |

---

## Training Data

### Dataset: PowerGraph (IEEE 24-bus)

**Source**: PowerGraph benchmark dataset
**License**: CC BY 4.0
**Grid**: IEEE 24-bus Reliability Test System

**Data Splits** (fixed seed=42):

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 16,125 | 80% |
| Validation | 2,016 | 10% |
| Test | 2,016 | 10% |

**Node Features**:

| Feature | Description | Tasks |
|---------|-------------|-------|
| P_net | Net active power injection (MW) | All |
| S_net | Net apparent power (MVA) | All |
| V | Voltage magnitude (p.u.) | OPF, Cascade |
| status | Component status (0/1) | Cascade |

**Edge Features**:

| Feature | Description | Tasks |
|---------|-------------|-------|
| X | Line reactance (p.u.) | All |
| rating | Thermal rating (MVA) | All |
| P_ij | Active power flow (MW) | Cascade |
| loading | Line loading fraction | Cascade |

---

## Self-Supervised Pretraining

### SSL Tasks

**Cascade Task**: Masked node/edge reconstruction
- Mask ratio: 15%
- BERT-style: 80% mask token, 10% random, 10% unchanged

**PF Task**: Masked voltage reconstruction (MaskedVoltageSSL)
- Predicts voltage from power injections
- Physics-meaningful: learns power flow relationships

**OPF Task**: Masked edge flow reconstruction (MaskedFlowSSL)
- Predicts edge flows from node embeddings
- Physics-meaningful: learns power transfer relationships

---

## Evaluation Results

### SSL Transfer Benefits (10% Labels)

| Task | Metric | Scratch | SSL | Improvement |
|------|--------|---------|-----|-------------|
| Cascade Prediction | F1 Score | 0.758 | 0.883 | **+16.5%** |
| Power Flow (PF) | MAE | 0.0216 | 0.0136 | **+37.1%** |
| Optimal Power Flow (OPF) | MAE | 0.0141 | 0.0096 | **+32.2%** |

### Full Label Results

| Task | Metric | 100% Labels |
|------|--------|-------------|
| Cascade Prediction | F1 Score | 0.957 |
| Power Flow | MAE | 0.0047 |
| Power Flow | R² | 0.998 |
| Optimal Power Flow | MAE | 0.0026 |

### Robustness (OOD Performance)

| Load Multiplier | Scratch F1 | SSL F1 | SSL Advantage |
|-----------------|------------|--------|---------------|
| 1.0x (ID) | 0.936 | 0.956 | +2% |
| 1.3x (OOD) | 0.673 | 0.821 | **+22%** |

### Explainability

| Metric | Value |
|--------|-------|
| Explanation Fidelity (AUC-ROC) | 0.93 |
| Comparison | vs PowerGraph ground-truth masks |

---

## Training Procedure

### Optimizer Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Batch Size | 64 |

### Hardware

- GPU: CUDA-enabled (tested on NVIDIA GPUs)
- Training Time: ~5-10 minutes per task on modern GPU

---

## Limitations

### Known Limitations

1. **Grid Specificity**: Trained and evaluated only on IEEE 24-bus system. Performance on larger grids (e.g., IEEE 118-bus, real utility grids) not validated.

2. **Topology Changes**: Model assumes fixed grid topology. Dynamic topology changes (e.g., islanding) not explicitly handled.

3. **AC vs DC Power Flow**: Current implementation uses simplified power flow features. Full AC power flow with reactive power not included in all tasks.

4. **Temporal Dynamics**: Model treats each snapshot independently. Temporal cascading dynamics not captured.

5. **Rare Events**: Cascading failures are rare events. Model may not generalize to failure modes not represented in training data.

### Physics Residual Behavior

The PhysicsGuidedConv layer uses learned admittance weighting:
- Sigmoid activation bounds weights to [0, 1]
- Does not enforce exact Kirchhoff's laws
- Provides soft physics bias rather than hard constraints

### Out-of-Distribution Behavior

- Performance degrades gracefully under moderate OOD conditions (up to 1.3x load)
- SSL pretraining significantly improves OOD robustness
- Severe OOD (e.g., 2x+ load) not tested

---

## Intended Use

### Primary Use Cases

1. **Research**: Benchmarking physics-informed GNNs for power systems
2. **Education**: Understanding SSL transfer learning for engineering domains
3. **Prototyping**: Initial models before deployment-grade development

### Out-of-Scope Uses

1. **Production Grid Control**: Not validated for real-time operation
2. **Safety-Critical Decisions**: Should not be sole basis for protection decisions
3. **Regulatory Compliance**: Not certified for utility operations

---

## Reproducibility

### Seeds

| Component | Seed |
|-----------|------|
| Global | 42 |
| Data Splits | 42 |
| Model Init | 42 |

### Key Files

```
configs/base.yaml          # Default hyperparameters
scripts/train_cascade.py   # Cascade training
scripts/train_pf_opf.py    # PF/OPF training
scripts/pretrain_ssl.py    # SSL pretraining (cascade)
scripts/pretrain_ssl_pf.py # SSL pretraining (PF/OPF)
```

### One-Command Reproduction

```bash
python analysis/run_all.py
```

---

## Citation

If you use this model, please cite:

```bibtex
@misc{physics_guided_gnn_power,
  title={Physics-Guided Self-Supervised GNN for Power System Analysis},
  year={2024},
  note={PowerGraph benchmark, IEEE 24-bus}
}
```

---

## Model Files

| File | Description |
|------|-------------|
| `outputs/ssl_pf_ieee24_*/best_model.pt` | SSL pretrained encoder (PF) |
| `outputs/ssl_opf_ieee24_*/best_model.pt` | SSL pretrained encoder (OPF) |
| `outputs/comparison_ieee24_*/` | Cascade SSL vs scratch comparison |
| `outputs/pf_comparison_ieee24_*/` | PF SSL vs scratch comparison |
| `outputs/opf_comparison_ieee24_*/` | OPF SSL vs scratch comparison |
