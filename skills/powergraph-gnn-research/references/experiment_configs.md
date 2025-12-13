# Experiment Configuration Reference

## Config Structure

Use YAML configs for reproducibility:

```yaml
# configs/base.yaml
project:
  name: powergraph-gnn
  seed: 42
  device: cuda

data:
  root: ./data
  grid: ieee24  # ieee24, ieee39, ieee118, uk
  task: pf      # pf, opf, cascade
  split: blocked  # blocked (temporal), random

model:
  type: physics_guided  # physics_guided, gat, gcn
  hidden_dim: 128
  num_layers: 4
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: cosine
  early_stopping_patience: 15

logging:
  tensorboard: true
  checkpoint_dir: ./checkpoints
  log_every: 10
```

## Experiment Configs

### 1. PF Baseline

```yaml
# configs/pf_baseline.yaml
_base_: base.yaml

experiment:
  name: pf_baseline
  description: Single-task PF baseline

data:
  task: pf

model:
  type: physics_guided

training:
  epochs: 100
  tasks: [pf]
  loss_weights:
    pf: 1.0
```

### 2. Multi-Task PF+OPF

```yaml
# configs/multitask_pf_opf.yaml
_base_: base.yaml

experiment:
  name: multitask_pf_opf
  description: Multi-task PF and OPF

data:
  task: multitask

training:
  epochs: 100
  tasks: [pf, opf]
  loss_weights:
    pf: 1.0
    opf: 0.5
```

### 3. Full Multi-Task

```yaml
# configs/multitask_full.yaml
_base_: base.yaml

experiment:
  name: multitask_full
  description: Full multi-task with cascade

training:
  epochs: 150
  tasks: [pf, opf, cascade]
  loss_weights:
    pf: 1.0
    opf: 0.5
    cascade: 1.0
```

### 4. SSL Pretraining

```yaml
# configs/ssl_pretrain.yaml
_base_: base.yaml

experiment:
  name: ssl_pretrain
  description: Self-supervised pretraining

ssl:
  method: combined  # masked_injection, masked_edge, combined
  node_mask_ratio: 0.15
  edge_mask_ratio: 0.10
  pretrain_epochs: 100
  alpha: 0.5  # weight between node and edge SSL

training:
  epochs: 100
  batch_size: 64
  lr: 1e-3
```

### 5. SSL → Fine-tune

```yaml
# configs/ssl_finetune.yaml
_base_: base.yaml

experiment:
  name: ssl_finetune
  description: Fine-tune from SSL pretrained encoder

pretrained:
  checkpoint: ./checkpoints/ssl_pretrain/best_encoder.pt
  freeze_epochs: 25  # Freeze encoder for first N epochs

training:
  epochs: 100
  lr: 1e-4  # Lower LR for fine-tuning
  tasks: [pf, opf]
```

### 6. Low-Label Experiment

```yaml
# configs/low_label.yaml
_base_: base.yaml

experiment:
  name: low_label_${label_frac}
  description: Low-label experiment

data:
  label_fraction: ${label_frac}  # 0.1, 0.2, 0.5, 1.0

training:
  epochs: 100

# Run with: python train.py --config configs/low_label.yaml label_frac=0.1
```

### 7. Uncertainty Quantification

```yaml
# configs/uncertainty.yaml
_base_: base.yaml

experiment:
  name: uncertainty_ensemble
  description: Deep ensemble for uncertainty

model:
  type: physics_guided
  ensemble_size: 5
  heteroscedastic: true  # Predict mean + variance

training:
  epochs: 100

evaluation:
  conformal_calibration: true
  calibration_alpha: 0.1
```

### 8. Cascade Transfer

```yaml
# configs/cascade_transfer.yaml
_base_: base.yaml

experiment:
  name: cascade_transfer
  description: Transfer to cascade prediction

data:
  task: cascade

pretrained:
  checkpoint: ./checkpoints/multitask_pf_opf/best_encoder.pt

model:
  cascade_classes: 3

training:
  epochs: 100
  freeze_encoder_epochs: 20

evaluation:
  explanation_fidelity: true
```

### 9. Robustness Test

```yaml
# configs/robustness.yaml
_base_: base.yaml

experiment:
  name: robustness_eval
  description: Robustness under perturbations

evaluation:
  perturbations:
    edge_drop: [0.05, 0.1, 0.2]
    load_scale: [0.9, 1.1, 1.2, 1.3]
    noise: [0.01, 0.05, 0.1]
```

### 10. Ablation Study

```yaml
# configs/ablation.yaml
_base_: base.yaml

experiment:
  name: ablation_${ablation_type}
  description: Ablation study

ablation:
  type: ${ablation_type}  # physics_guide, ssl, multitask, ensemble
  
  # Variants
  physics_guide:
    enabled: ${enable_physics}  # true/false
  
  ssl:
    enabled: ${enable_ssl}
  
  multitask:
    tasks: ${task_list}  # [pf], [pf,opf], [pf,opf,cascade]
```

## Config Loading Code

```python
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import copy

def load_config(config_path: str) -> dict:
    """Load config with inheritance support."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if '_base_' in config:
        base_path = Path(config_path).parent / config.pop('_base_')
        base_config = load_config(str(base_path))
        config = deep_merge(base_config, config)
    
    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts."""
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


@dataclass
class ExperimentConfig:
    """Typed experiment config."""
    
    # Project
    name: str
    seed: int = 42
    device: str = 'cuda'
    
    # Data
    data_root: str = './data'
    grid: str = 'ieee24'
    task: str = 'pf'
    split_type: str = 'blocked'
    label_fraction: float = 1.0
    
    # Model
    model_type: str = 'physics_guided'
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    tasks: List[str] = field(default_factory=lambda: ['pf'])
    
    # SSL
    ssl_enabled: bool = False
    ssl_method: str = 'combined'
    node_mask_ratio: float = 0.15
    edge_mask_ratio: float = 0.10
    
    # Pretrained
    pretrained_checkpoint: Optional[str] = None
    freeze_encoder_epochs: int = 0
    
    @classmethod
    def from_yaml(cls, path: str):
        config_dict = load_config(path)
        # Flatten nested dict and create instance
        # ... implementation
        pass
```

## Running Experiments

### Single Experiment

```bash
# Run with config
python scripts/train.py --config configs/pf_baseline.yaml

# Override params
python scripts/train.py --config configs/pf_baseline.yaml training.epochs=200 model.hidden_dim=256
```

### Sweep Experiments

```python
# scripts/run_sweep.py
import subprocess
from itertools import product

grids = ['ieee24', 'ieee39', 'ieee118']
seeds = [42, 43, 44, 45, 46]
label_fracs = [0.1, 0.2, 0.5, 1.0]

for grid, seed, frac in product(grids, seeds, label_fracs):
    cmd = [
        'python', 'scripts/train.py',
        '--config', 'configs/low_label.yaml',
        f'data.grid={grid}',
        f'project.seed={seed}',
        f'data.label_fraction={frac}',
    ]
    
    subprocess.run(cmd)
```

### Ablation Study Script

```python
# scripts/run_ablation.py

ablations = {
    'no_physics': {'model.physics_regularization': False},
    'no_ssl': {'pretrained.checkpoint': None},
    'pf_only': {'training.tasks': ['pf']},
    'pf_opf': {'training.tasks': ['pf', 'opf']},
    'full': {'training.tasks': ['pf', 'opf', 'cascade']},
}

for name, overrides in ablations.items():
    cmd = ['python', 'scripts/train.py', '--config', 'configs/ablation.yaml']
    cmd.append(f'experiment.name=ablation_{name}')
    
    for key, value in overrides.items():
        cmd.append(f'{key}={value}')
    
    subprocess.run(cmd)
```

## Reproducibility Checklist

```yaml
# configs/reproducibility.yaml

# Seeds (set all of these)
seeds:
  torch: 42
  numpy: 42
  random: 42
  cuda_deterministic: true

# Environment capture
environment:
  capture: true
  save_requirements: true
  save_git_hash: true

# Checkpointing
checkpoints:
  save_every: 10
  save_best: true
  save_last: true
  keep_last_k: 3

# Logging
logging:
  log_config: true
  log_model_summary: true
  log_gradients: false
  log_weights: false
```

## Results Directory Structure

```
outputs/
├── pf_baseline_ieee24_seed42/
│   ├── config.yaml           # Full config used
│   ├── checkpoints/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── logs/
│   │   └── tensorboard/
│   ├── metrics/
│   │   ├── train_metrics.csv
│   │   ├── val_metrics.csv
│   │   └── test_metrics.json
│   └── figures/
│       ├── loss_curves.png
│       └── calibration.png
├── multitask_full_ieee24_seed42/
│   └── ...
└── comparison_summary.csv
```

## Experiment Tracking Integration

```python
# Optional: W&B integration
import wandb

def init_tracking(config: dict):
    """Initialize experiment tracking."""
    
    wandb.init(
        project=config['project']['name'],
        name=config['experiment']['name'],
        config=config,
        tags=[config['data']['grid'], config['model']['type']],
    )


def log_metrics(metrics: dict, step: int):
    """Log metrics to tracker."""
    wandb.log(metrics, step=step)


def log_artifact(path: str, name: str, type: str):
    """Log artifact (model, data, etc.)."""
    artifact = wandb.Artifact(name=name, type=type)
    artifact.add_file(path)
    wandb.log_artifact(artifact)
```
