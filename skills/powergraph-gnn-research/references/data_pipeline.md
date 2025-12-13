# Data Pipeline Reference

## PowerGraph Dataset Overview

PowerGraph (CC BY 4.0) provides:
- **PF dataset**: Node-level power flow solutions (V, θ)
- **OPF dataset**: Node-level optimal dispatch results
- **Cascade dataset**: Graph-level failure outcomes + explanation masks

GitHub: https://github.com/PowerGraph-Datasets/PowerGraph-Graph

## Download & Setup

```bash
# Clone repository
git clone https://github.com/PowerGraph-Datasets/PowerGraph-Graph.git
cd PowerGraph-Graph

# Download data (check figshare link in repo README)
# Store in data/raw/
```

## Data Structure

```
PowerGraph-Graph/
├── data/
│   ├── ieee24/
│   │   ├── pf/           # Power flow scenarios
│   │   ├── opf/          # Optimal power flow
│   │   └── cascade/      # Cascading failures
│   ├── ieee39/
│   ├── ieee118/
│   └── uk/
└── ...
```

## PyG Data Object Structure

Convert each scenario to:
```python
from torch_geometric.data import Data

data = Data(
    # Node features
    x=node_features,          # [num_nodes, num_node_features]
    
    # Graph structure
    edge_index=edge_index,    # [2, num_edges]
    edge_attr=edge_attr,      # [num_edges, num_edge_features]
    
    # PF targets (node-level)
    y_v_mag=voltage_magnitude,    # [num_nodes]
    y_v_ang_sin=voltage_ang_sin,  # [num_nodes]
    y_v_ang_cos=voltage_ang_cos,  # [num_nodes]
    
    # OPF targets (node-level, if applicable)
    y_pg=active_generation,       # [num_gen_nodes]
    y_cost=total_cost,            # scalar
    gen_mask=generator_mask,      # [num_nodes] boolean
    
    # Cascade targets (graph-level)
    y_cascade=cascade_label,      # scalar (severity class)
    exp_mask=explanation_mask,    # [num_edges] binary
    
    # Metadata
    grid_name='ieee24',
    scenario_id=scenario_idx,
    timestamp=time_idx,  # For blocked splits
)
```

## Node Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `P_load` | Active load (MW) | Per-grid mean/std |
| `Q_load` | Reactive load (MVAr) | Per-grid mean/std |
| `P_gen` | Active generation (MW) | Per-grid mean/std |
| `Q_gen` | Reactive generation (MVAr) | Per-grid mean/std |
| `V_setpoint` | Voltage setpoint (pu) | Already normalized |
| `bus_type` | Bus type (PQ/PV/Slack) | One-hot encode |

## Edge Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `G_ij` | Conductance | Log-scale recommended |
| `B_ij` | Susceptance | Log-scale recommended |
| `rating` | Thermal limit (MVA) | Per-grid mean/std |
| `tap` | Transformer tap ratio | None needed |

## Data Loader Implementation

```python
import torch
from torch_geometric.data import Dataset, Data
from scipy.io import loadmat
import numpy as np
from pathlib import Path

class PowerGraphDataset(Dataset):
    """PyG dataset for PowerGraph benchmark."""
    
    def __init__(
        self,
        root: str,
        grid: str = 'ieee24',
        task: str = 'pf',  # 'pf', 'opf', 'cascade'
        split: str = 'train',
        transform=None,
        pre_transform=None,
    ):
        self.grid = grid
        self.task = task
        self.split = split
        super().__init__(root, transform, pre_transform)
        
        # Load split indices
        self.indices = self._load_split_indices()
    
    @property
    def raw_file_names(self):
        return [f'{self.grid}/{self.task}']
    
    @property
    def processed_file_names(self):
        return [f'{self.grid}_{self.task}_{self.split}.pt']
    
    def _load_split_indices(self):
        """Load pre-computed blocked time splits."""
        split_file = Path(self.processed_dir) / f'{self.grid}_splits.pt'
        splits = torch.load(split_file)
        return splits[self.split]
    
    def process(self):
        """Convert raw .mat files to PyG Data objects."""
        # Implementation depends on PowerGraph format
        # See PowerGraph repo for exact structure
        pass
    
    def len(self):
        return len(self.indices)
    
    def get(self, idx):
        actual_idx = self.indices[idx]
        data = torch.load(
            Path(self.processed_dir) / f'{self.grid}_{self.task}_{actual_idx}.pt'
        )
        return data
```

## Blocked Time Splits

Prevent temporal leakage:
```python
def create_blocked_splits(
    num_scenarios: int,
    scenarios_per_day: int = 96,  # 15-min resolution
    train_frac: float = 0.75,
    val_frac: float = 0.08,
):
    """Create blocked time splits for 1-year data."""
    days = num_scenarios // scenarios_per_day
    
    train_days = int(days * train_frac)
    val_days = int(days * val_frac)
    
    train_idx = list(range(train_days * scenarios_per_day))
    val_idx = list(range(
        train_days * scenarios_per_day,
        (train_days + val_days) * scenarios_per_day
    ))
    test_idx = list(range(
        (train_days + val_days) * scenarios_per_day,
        num_scenarios
    ))
    
    return {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
    }
```

## Data Validation Checks

Run before training:
```python
def validate_dataset(dataset):
    """Sanity checks for PowerGraph data."""
    
    sample = dataset[0]
    
    # Check shapes
    assert sample.x.dim() == 2, "Node features should be 2D"
    assert sample.edge_index.shape[0] == 2, "edge_index should be [2, E]"
    
    # Check for NaN/Inf
    assert not torch.isnan(sample.x).any(), "NaN in node features"
    assert not torch.isinf(sample.x).any(), "Inf in node features"
    
    # Check targets exist
    if hasattr(sample, 'y_v_mag'):
        assert sample.y_v_mag.shape[0] == sample.x.shape[0]
    
    # Check edge alignment
    if hasattr(sample, 'exp_mask'):
        assert sample.exp_mask.shape[0] == sample.edge_index.shape[1]
    
    print(f"✓ Dataset validated: {len(dataset)} samples")
    print(f"  Nodes: {sample.x.shape[0]}, Edges: {sample.edge_index.shape[1]}")
    print(f"  Node features: {sample.x.shape[1]}")
    print(f"  Edge features: {sample.edge_attr.shape[1] if sample.edge_attr is not None else 0}")
```

## Cascade Explanation Masks

PowerGraph provides ground-truth explanation masks (`exp.mat`):
```python
def load_explanation_masks(cascade_dir: Path, scenario_idx: int):
    """Load ground-truth edge importance for cascade explanation."""
    exp_data = loadmat(cascade_dir / 'exp.mat')
    
    # Binary mask: 1 = edge contributed to cascade, 0 = did not
    exp_mask = torch.tensor(
        exp_data['exp_mask'][scenario_idx],
        dtype=torch.float
    )
    
    return exp_mask
```

## Data Augmentation (Optional)

For robustness testing:
```python
class EdgeDropTransform:
    """Randomly drop edges at inference for robustness testing."""
    
    def __init__(self, drop_prob: float = 0.1):
        self.drop_prob = drop_prob
    
    def __call__(self, data):
        num_edges = data.edge_index.shape[1]
        mask = torch.rand(num_edges) > self.drop_prob
        
        data.edge_index = data.edge_index[:, mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        
        return data

class LoadScalingTransform:
    """Scale loads for OOD testing."""
    
    def __init__(self, scale_range: tuple = (0.9, 1.3)):
        self.scale_range = scale_range
    
    def __call__(self, data):
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        
        # Assuming first 2 features are P_load, Q_load
        data.x[:, :2] *= scale
        
        return data
```
