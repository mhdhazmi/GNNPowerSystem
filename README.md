# Physics-Guided GNN for Power Systems

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A physics-informed Graph Neural Network for power system analysis, featuring self-supervised learning for cascade failure prediction, power flow estimation, and line flow prediction.

## Key Results

| Task | Metric | From Scratch | With SSL | Improvement |
|------|--------|--------------|----------|-------------|
| Cascade Prediction | F1 Score | 0.753 | 0.860 | **+14.2%** |
| Power Flow | MAE | 0.0149 | 0.0106 | **+29.1%** |
| Line Flow | MAE | 0.0084 | 0.0062 | **+26.4%** |

*Results on IEEE 24-bus system with 10% labeled data*

## Features

- **Physics-Guided Message Passing**: Learned admittance weighting for power flow relationships
- **Self-Supervised Pretraining**: Masked reconstruction tasks for improved low-label performance
- **Multi-Task Support**: Cascade failure prediction, power flow, and line flow estimation
- **OOD Robustness**: +22% improvement on out-of-distribution load conditions

## Architecture

```
Input → Node Embedding → [PhysicsGuidedConv + LayerNorm + ReLU + Residual] × N → Output
```

The `PhysicsGuidedConv` layer uses learned admittance scaling to weight message passing, providing a soft physics bias without hard constraints.

### Model Parameters

| Task | Node Features | Edge Features | Parameters |
|------|---------------|---------------|------------|
| Cascade | 4 | 4 | ~270K |
| Power Flow | 2 | 2 | ~274K |
| Line Flow | 3 | 2 | ~168K |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mhdhazmi/GNNPowerSystem.git
cd GNNPowerSystem

# Install dependencies (using uv)
uv sync

# Or with pip
pip install -e .

# Run all experiments
python analysis/run_all.py

# Or run individual tasks
python scripts/train_cascade.py
python scripts/train_pf_opf.py
python scripts/pretrain_ssl.py
```

## Project Structure

```
├── src/                    # Core model implementations
├── scripts/                # Training scripts
├── configs/                # Hyperparameter configurations
├── experiments/            # Experiment outputs
├── analysis/               # Analysis and visualization
├── dashboard/              # Visualization dashboard
├── Paper/                  # Research paper materials
└── MODEL_CARD.md          # Detailed model documentation
```

## Dataset

Uses the **PowerGraph** benchmark dataset with IEEE 24-bus and IEEE 118-bus Reliability Test Systems.

| Grid | Train | Val | Test | Total |
|------|-------|-----|------|-------|
| IEEE 24-bus | 16,125 | 2,016 | 2,016 | 20,157 |
| IEEE 118-bus | 91,875 | 11,484 | 11,484 | 114,843 |

## Self-Supervised Learning

The model uses masked reconstruction tasks for pretraining:

- **Cascade Task**: Masked node/edge reconstruction (15% mask ratio, BERT-style)
- **Power Flow Task**: Masked injection reconstruction from topology
- **Line Flow Task**: Masked line parameter reconstruction

SSL pretraining significantly improves performance in low-label regimes and stabilizes training (reduces variance by ~5x at 10% labels).

## Robustness Results

| Load Multiplier | Scratch F1 | SSL F1 | Improvement |
|-----------------|------------|--------|-------------|
| 1.0x (In-Distribution) | 0.936 | 0.956 | +2% |
| 1.3x (Out-of-Distribution) | 0.673 | 0.821 | **+22%** |

## Citation

```bibtex
@misc{physics_guided_gnn_power,
  title={Physics-Guided Self-Supervised GNN for Power System Analysis},
  author={Alhazmi, Mohammed},
  year={2024},
  note={PowerGraph benchmark, IEEE 24-bus and IEEE 118-bus}
}
```

## Documentation

For detailed model specifications, training procedures, and limitations, see [MODEL_CARD.md](MODEL_CARD.md).

## License

MIT License
