#!/usr/bin/env python3
"""
PowerGraph Data Loading Script

Converts PowerGraph benchmark data to PyTorch Geometric format.
Implements blocked time splits for temporal data leakage prevention.

Usage:
    python load_powergraph.py --data_dir ./data/raw/PowerGraph-Graph \
                               --output_dir ./data/processed \
                               --grid ieee24
"""

import argparse
import torch
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from typing import Dict, List, Tuple, Optional
import json


class PowerGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for PowerGraph benchmark.
    
    Supports PF, OPF, and Cascade tasks with ground-truth explanation masks.
    """
    
    GRIDS = ['ieee24', 'ieee39', 'ieee118', 'uk']
    TASKS = ['pf', 'opf', 'cascade']
    
    def __init__(
        self,
        root: str,
        grid: str = 'ieee24',
        task: str = 'pf',
        split: str = 'train',
        split_type: str = 'blocked',
        transform=None,
        pre_transform=None,
    ):
        self.grid = grid
        self.task = task
        self.split = split
        self.split_type = split_type
        
        super().__init__(root, transform, pre_transform)
        
        # Load appropriate split
        split_idx = self.split_type.index(split)
        self.data, self.slices = torch.load(self.processed_paths[split_idx])
    
    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.grid}_{self.task}.mat']
    
    @property
    def processed_file_names(self) -> List[str]:
        return [
            f'{self.grid}_{self.task}_{self.split_type}_train.pt',
            f'{self.grid}_{self.task}_{self.split_type}_val.pt',
            f'{self.grid}_{self.task}_{self.split_type}_test.pt',
        ]
    
    def download(self):
        """PowerGraph must be downloaded manually from GitHub/Figshare."""
        raise RuntimeError(
            "PowerGraph dataset must be downloaded manually.\n"
            "Visit: https://github.com/PowerGraph-Datasets/PowerGraph-Graph\n"
            f"Place data in: {self.raw_dir}"
        )
    
    def process(self):
        """Convert raw .mat files to PyG Data objects."""
        
        # Load raw data
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw_data = loadmat(str(raw_path))
        
        # Extract components
        data_list = self._convert_to_pyg(raw_data)
        
        # Create splits
        splits = self._create_splits(len(data_list))
        
        # Save each split
        for split_name, indices in splits.items():
            split_data = [data_list[i] for i in indices]
            
            if self.pre_transform is not None:
                split_data = [self.pre_transform(d) for d in split_data]
            
            data, slices = self.collate(split_data)
            
            split_idx = ['train', 'val', 'test'].index(split_name)
            torch.save((data, slices), self.processed_paths[split_idx])
        
        # Save split indices for reference
        splits_path = Path(self.processed_dir) / f'{self.grid}_{self.task}_splits.json'
        with open(splits_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in splits.items()}, f)
    
    def _convert_to_pyg(self, raw_data: dict) -> List[Data]:
        """Convert MATLAB data to PyG Data objects."""
        
        data_list = []
        num_scenarios = raw_data.get('num_scenarios', 1000)
        
        for i in range(num_scenarios):
            data = self._create_data_object(raw_data, i)
            data_list.append(data)
        
        return data_list
    
    def _create_data_object(self, raw_data: dict, idx: int) -> Data:
        """Create single PyG Data object from scenario."""
        
        # Node features: [P_load, Q_load, P_gen, Q_gen, V_set, bus_type_onehot]
        x = self._extract_node_features(raw_data, idx)
        
        # Edge structure and features
        edge_index = self._extract_edge_index(raw_data)
        edge_attr = self._extract_edge_features(raw_data, idx)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        # Add task-specific targets
        if self.task in ['pf', 'multitask']:
            self._add_pf_targets(data, raw_data, idx)
        
        if self.task in ['opf', 'multitask']:
            self._add_opf_targets(data, raw_data, idx)
        
        if self.task in ['cascade', 'multitask']:
            self._add_cascade_targets(data, raw_data, idx)
        
        # Metadata
        data.scenario_idx = idx
        data.grid_name = self.grid
        
        return data
    
    def _extract_node_features(self, raw_data: dict, idx: int) -> torch.Tensor:
        """Extract node features for scenario."""
        
        # Placeholder - adapt to actual PowerGraph format
        num_nodes = raw_data.get('num_buses', 24)
        
        # Features: P_load, Q_load, P_gen, Q_gen, V_setpoint, bus_type (3 one-hot)
        feature_dim = 8
        
        features = torch.zeros(num_nodes, feature_dim)
        
        # Load from raw_data based on actual format
        # features[:, 0] = raw_data['P_load'][idx]
        # features[:, 1] = raw_data['Q_load'][idx]
        # etc.
        
        return features
    
    def _extract_edge_index(self, raw_data: dict) -> torch.Tensor:
        """Extract edge connectivity."""
        
        # Placeholder - adapt to actual format
        # from_bus = raw_data['branch'][:, 0] - 1  # 0-indexed
        # to_bus = raw_data['branch'][:, 1] - 1
        
        # Bidirectional edges
        # edge_index = torch.tensor([
        #     np.concatenate([from_bus, to_bus]),
        #     np.concatenate([to_bus, from_bus])
        # ], dtype=torch.long)
        
        edge_index = torch.zeros(2, 50, dtype=torch.long)  # Placeholder
        
        return edge_index
    
    def _extract_edge_features(self, raw_data: dict, idx: int) -> torch.Tensor:
        """Extract edge features (line parameters)."""
        
        # Features: G, B, rating, tap_ratio
        # Placeholder
        num_edges = 50
        edge_attr = torch.zeros(num_edges, 4)
        
        return edge_attr
    
    def _add_pf_targets(self, data: Data, raw_data: dict, idx: int):
        """Add power flow targets."""
        
        num_nodes = data.x.shape[0]
        
        # Voltage magnitude
        data.y_v_mag = torch.zeros(num_nodes)  # Placeholder
        
        # Voltage angle as sin/cos (handles wrap-around)
        data.y_v_ang_sin = torch.zeros(num_nodes)
        data.y_v_ang_cos = torch.ones(num_nodes)  # cos(0) = 1
    
    def _add_opf_targets(self, data: Data, raw_data: dict, idx: int):
        """Add optimal power flow targets."""
        
        num_nodes = data.x.shape[0]
        
        # Generator active power setpoints
        data.y_pg = torch.zeros(num_nodes)
        
        # Generator mask (1 for generator buses)
        data.gen_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Total generation cost
        data.y_cost = torch.tensor(0.0)
    
    def _add_cascade_targets(self, data: Data, raw_data: dict, idx: int):
        """Add cascading failure targets with explanation masks."""
        
        num_edges = data.edge_index.shape[1]
        
        # Cascade severity class (0=none, 1=small, 2=large)
        data.y_cascade = torch.tensor(0)
        
        # Ground-truth explanation mask (which edges contributed to cascade)
        data.exp_mask = torch.zeros(num_edges)
    
    def _create_splits(
        self,
        num_samples: int,
        train_frac: float = 0.75,
        val_frac: float = 0.08,
    ) -> Dict[str, np.ndarray]:
        """Create train/val/test splits."""
        
        if self.split_type == 'blocked':
            return self._blocked_split(num_samples, train_frac, val_frac)
        else:
            return self._random_split(num_samples, train_frac, val_frac)
    
    def _blocked_split(
        self,
        num_samples: int,
        train_frac: float,
        val_frac: float,
    ) -> Dict[str, np.ndarray]:
        """Blocked temporal split (no leakage)."""
        
        train_end = int(num_samples * train_frac)
        val_end = int(num_samples * (train_frac + val_frac))
        
        return {
            'train': np.arange(0, train_end),
            'val': np.arange(train_end, val_end),
            'test': np.arange(val_end, num_samples),
        }
    
    def _random_split(
        self,
        num_samples: int,
        train_frac: float,
        val_frac: float,
    ) -> Dict[str, np.ndarray]:
        """Random split (potential leakage)."""
        
        indices = np.random.permutation(num_samples)
        
        train_end = int(num_samples * train_frac)
        val_end = int(num_samples * (train_frac + val_frac))
        
        return {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:],
        }


def validate_dataset(dataset: PowerGraphDataset):
    """Sanity checks for loaded dataset."""
    
    print(f"Dataset: {dataset.grid} / {dataset.task} / {dataset.split}")
    print(f"Samples: {len(dataset)}")
    
    sample = dataset[0]
    
    print(f"Node features shape: {sample.x.shape}")
    print(f"Edge index shape: {sample.edge_index.shape}")
    print(f"Edge attr shape: {sample.edge_attr.shape if sample.edge_attr is not None else 'None'}")
    
    # Check for NaN
    assert not torch.isnan(sample.x).any(), "NaN in node features!"
    
    # Check targets
    if hasattr(sample, 'y_v_mag'):
        print(f"PF targets present: v_mag shape = {sample.y_v_mag.shape}")
    
    if hasattr(sample, 'y_cascade'):
        print(f"Cascade targets present: label = {sample.y_cascade.item()}")
    
    if hasattr(sample, 'exp_mask'):
        print(f"Explanation mask present: shape = {sample.exp_mask.shape}")
    
    print("✓ Dataset validation passed")


def main():
    parser = argparse.ArgumentParser(description='Load PowerGraph dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw PowerGraph data')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--grid', type=str, default='ieee24',
                        choices=['ieee24', 'ieee39', 'ieee118', 'uk'])
    parser.add_argument('--task', type=str, default='pf',
                        choices=['pf', 'opf', 'cascade', 'multitask'])
    parser.add_argument('--split_type', type=str, default='blocked',
                        choices=['blocked', 'random'])
    
    args = parser.parse_args()
    
    # Process dataset
    for split in ['train', 'val', 'test']:
        dataset = PowerGraphDataset(
            root=args.output_dir,
            grid=args.grid,
            task=args.task,
            split=split,
            split_type=args.split_type,
        )
        
        validate_dataset(dataset)


if __name__ == '__main__':
    main()
