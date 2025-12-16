"""
PowerGraph Dataset Loader

Loads PowerGraph benchmark data for power grid GNN tasks:
- CASCADE: Graph-level cascading failure classification
- PF: Node-level power flow (voltage prediction)
- LINE FLOW: Edge-level power flow prediction (uses --task opf/lineflow flag)

Data source: https://figshare.com/articles/dataset/PowerGraph/22820534

Usage:
    from src.data.powergraph import PowerGraphDataset

    dataset = PowerGraphDataset(
        root="./data",
        name="ieee24",
        task="cascade",  # or "pf" or "opf"
        split="train",
    )
"""

import hashlib
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import mat73
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


# Figshare download URL for PowerGraph dataset
FIGSHARE_URL = "https://figshare.com/ndownloader/files/46619158"
EXPECTED_CHECKSUM = None  # Will compute on first download


class PowerGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for PowerGraph benchmark.

    Supports multiple power grids (IEEE24, IEEE39, IEEE118, UK) and tasks
    (CASCADE classification with explanation masks).

    Node features (3):
        - Net active power at bus (P_net)
        - Net apparent power at bus (S_net)
        - Voltage magnitude (V)

    Edge features (4):
        - Active power flow (P_ij)
        - Reactive power flow (Q_ij)
        - Line reactance (X_ij)
        - Line rating (lr_ij)

    CASCADE labels:
        - Binary: DNS=0 vs DNS≠0
        - Multiclass: severity categories
        - Regression: DNS in MW

    Explanation masks:
        - Binary edge mask indicating which edges caused cascade
    """

    GRIDS = ["ieee24", "ieee39", "ieee118", "uk"]
    TASKS = ["cascade", "pf", "opf"]  # PF=voltage prediction, OPF=flow prediction
    LABEL_TYPES = ["binary", "multiclass", "regression"]

    # Raw files expected per grid
    RAW_FILES = [
        "blist.mat",    # Edge index
        "Bf.mat",       # Node features
        "Ef.mat",       # Edge features
        "of_bi.mat",    # Binary labels
        "of_mc.mat",    # Multiclass labels
        "of_reg.mat",   # Regression labels
        "exp.mat",      # Explanation masks
    ]

    def __init__(
        self,
        root: str,
        name: str = "ieee24",
        task: str = "cascade",
        label_type: str = "binary",
        split: str = "train",
        split_type: str = "blocked",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root: Root directory for data storage
            name: Grid name (ieee24, ieee39, ieee118, uk)
            task: Task type (cascade, pf, opf)
            label_type: For cascade task (binary, multiclass, regression)
            split: Data split (train, val, test)
            split_type: Split strategy (blocked for temporal, random)
            transform: Optional transform applied at access time
            pre_transform: Optional transform applied during processing
            pre_filter: Optional filter applied during processing
        """
        self.name = name.lower()
        self.task = task.lower()
        self.label_type = label_type.lower()
        self.split = split.lower()
        self.split_type = split_type.lower()

        assert self.name in self.GRIDS, f"Unknown grid: {name}. Choose from {self.GRIDS}"
        assert self.task in self.TASKS, f"Unknown task: {task}. Choose from {self.TASKS}"
        assert self.label_type in self.LABEL_TYPES, f"Unknown label_type: {label_type}"
        assert self.split in ["train", "val", "test"], f"Unknown split: {split}"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load the appropriate split
        split_idx = ["train", "val", "test"].index(self.split)
        self.data, self.slices = torch.load(
            self.processed_paths[split_idx], weights_only=False
        )

        # Load split indices for reference
        self._load_split_info()

    @property
    def raw_dir(self) -> str:
        # PowerGraph archive has nested structure: grid/grid/raw/
        nested_path = os.path.join(self.root, "raw", self.name, self.name, "raw")
        if os.path.exists(nested_path):
            return nested_path
        # Fallback to flat structure
        return os.path.join(self.root, "raw", self.name)

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            self.root, "processed", self.name,
            f"{self.task}_{self.label_type}_{self.split_type}"
        )

    @property
    def raw_file_names(self) -> List[str]:
        return self.RAW_FILES

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "test.pt", "split_info.json"]

    def download(self):
        """Download PowerGraph data from Figshare."""
        raw_parent = Path(self.root) / "raw"
        raw_parent.mkdir(parents=True, exist_ok=True)

        archive_path = raw_parent / "powergraph_data.zip"

        # Check if already downloaded and extracted
        if self._check_raw_files_exist():
            print(f"Raw files for {self.name} already exist, skipping download")
            return

        # Download if archive doesn't exist
        if not archive_path.exists():
            # Also check for old .tar.gz name
            old_path = raw_parent / "powergraph_data.tar.gz"
            if old_path.exists():
                old_path.rename(archive_path)
            else:
                print(f"Downloading PowerGraph dataset from Figshare...")
                print(f"URL: {FIGSHARE_URL}")
                print("This may take a few minutes (~2.7GB compressed)...")

                _download_with_progress(FIGSHARE_URL, archive_path)
                print(f"Download complete: {archive_path}")

        # Extract - detect format
        print(f"Extracting archive...")
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(raw_parent)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(raw_parent)
        else:
            raise RuntimeError(f"Unknown archive format: {archive_path}")
        print(f"Extraction complete")

        # Verify files exist
        if not self._check_raw_files_exist():
            raise RuntimeError(
                f"Expected raw files not found after extraction.\n"
                f"Expected location: {self.raw_dir}\n"
                f"Please check the archive structure."
            )

    def _check_raw_files_exist(self) -> bool:
        """Check if all raw files exist for this grid."""
        raw_dir = Path(self.raw_dir)
        if not raw_dir.exists():
            return False
        return all((raw_dir / f).exists() for f in self.RAW_FILES)

    def process(self):
        """Process raw .mat files into PyG Data objects."""
        print(f"Processing {self.name} dataset for {self.task} task...")

        # Load raw data
        raw_data = self._load_raw_data()

        # Convert to PyG Data objects
        data_list = self._create_data_list(raw_data)
        print(f"Created {len(data_list)} graph samples")

        # Create splits
        splits = self._create_splits(len(data_list))

        # Save split info
        split_info = {
            "grid": self.name,
            "task": self.task,
            "label_type": self.label_type,
            "split_type": self.split_type,
            "num_samples": len(data_list),
            "train_size": len(splits["train"]),
            "val_size": len(splits["val"]),
            "test_size": len(splits["test"]),
        }

        # Save each split
        for split_name, indices in splits.items():
            split_data = [data_list[i] for i in indices]

            if self.pre_filter is not None:
                split_data = [d for d in split_data if self.pre_filter(d)]

            if self.pre_transform is not None:
                split_data = [self.pre_transform(d) for d in split_data]

            data, slices = self.collate(split_data)

            split_idx = ["train", "val", "test"].index(split_name)
            torch.save((data, slices), self.processed_paths[split_idx])
            print(f"Saved {split_name} split: {len(split_data)} samples")

        # Save split info
        with open(self.processed_paths[3], "w") as f:
            json.dump(split_info, f, indent=2)

    def _load_raw_data(self) -> Dict:
        """Load all raw .mat files."""
        raw_dir = Path(self.raw_dir)

        print(f"Loading raw data from {raw_dir}")

        data = {}

        # Edge index (branch list)
        blist = mat73.loadmat(raw_dir / "blist.mat")
        data["edge_index"] = torch.tensor(blist["bList"] - 1, dtype=torch.long)  # 0-indexed

        # Node features
        bf = mat73.loadmat(raw_dir / "Bf.mat")
        data["node_features"] = bf["B_f_tot"]

        # Edge features
        ef = mat73.loadmat(raw_dir / "Ef.mat")
        data["edge_features"] = ef["E_f_post"]

        # Labels
        of_bi = mat73.loadmat(raw_dir / "of_bi.mat")
        data["labels_binary"] = of_bi["output_features"]

        of_mc = mat73.loadmat(raw_dir / "of_mc.mat")
        data["labels_multiclass"] = of_mc["category"]

        of_reg = mat73.loadmat(raw_dir / "of_reg.mat")
        data["labels_regression"] = of_reg["dns_MW"]

        # Explanation masks
        exp = mat73.loadmat(raw_dir / "exp.mat")
        data["explanations"] = exp["explainations"]  # Note: typo in original data

        return data

    def _create_data_list(self, raw_data: Dict) -> List[Data]:
        """Convert raw data to list of PyG Data objects."""
        data_list = []

        edge_index_base = raw_data["edge_index"]
        num_samples = len(raw_data["node_features"])

        for i in tqdm(range(num_samples), desc="Creating PyG graphs"):
            # Node features: [num_nodes, 3]
            # P_net, S_net, V
            node_feat_full = torch.tensor(
                raw_data["node_features"][i][0],
                dtype=torch.float32
            ).reshape(-1, 3)

            # Edge features: [num_edges, 4]
            # P_flow, Q_flow, X, rating
            edge_feat_raw = torch.tensor(
                raw_data["edge_features"][i][0],
                dtype=torch.float32
            )

            # Find contingency edges (all-zero features = failed lines)
            contingency_mask = (edge_feat_raw.abs().sum(dim=1) == 0)
            valid_mask = ~contingency_mask

            # Remove contingency edges
            edge_attr_full = edge_feat_raw[valid_mask].reshape(-1, 4)
            edge_index = edge_index_base[valid_mask].reshape(-1, 2).T

            # Make bidirectional
            edge_index_rev = edge_index.flip(0)
            edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
            edge_attr_full = torch.cat([edge_attr_full, edge_attr_full], dim=0)

            # Task-specific features and labels
            if self.task == "pf":
                # PF task: predict voltage (V) from P_net, S_net
                # Input: P_net, S_net (2 features)
                # Target: V (voltage magnitude)
                x = node_feat_full[:, :2]  # P_net, S_net only
                y = node_feat_full[:, 2]   # V is the target
                # Edge features: use X and rating only (no flow info, that's derived)
                edge_attr = edge_attr_full[:, 2:4]  # X, rating
                edge_mask = None

            elif self.task == "opf":
                # OPF task: predict edge flows from node features
                # Input: full node features (P_net, S_net, V)
                # Target: P_flow, Q_flow on edges
                x = node_feat_full  # All 3 node features
                y = edge_attr_full[:, :2]  # P_flow, Q_flow as target
                # Edge features: X, rating only (no flow info)
                edge_attr = edge_attr_full[:, 2:4]  # X, rating
                edge_mask = None

            else:  # cascade task
                x = node_feat_full
                edge_attr = edge_attr_full

                # Explanation mask
                exp_raw = raw_data["explanations"][i][0]
                exp_mask = torch.zeros(valid_mask.sum().item(), dtype=torch.float32)
                if exp_raw is not None:
                    exp_arr = np.atleast_1d(exp_raw.astype(int) - 1)  # Ensure 1D
                    exp_indices = torch.tensor(exp_arr, dtype=torch.long)
                    # Filter to valid edges only
                    valid_indices = valid_mask.nonzero().squeeze(-1)
                    if valid_indices.dim() == 0:
                        valid_indices = valid_indices.unsqueeze(0)
                    for idx in exp_indices:
                        match = (valid_indices == idx.item()).nonzero()
                        if len(match) > 0:
                            exp_mask[match[0].item()] = 1.0
                # Make bidirectional
                edge_mask = torch.cat([exp_mask, exp_mask], dim=0)

                # Labels
                if self.label_type == "binary":
                    y = torch.tensor(
                        raw_data["labels_binary"][i][0],
                        dtype=torch.float32
                    ).view(1)
                elif self.label_type == "regression":
                    y = torch.tensor(
                        raw_data["labels_regression"][i],
                        dtype=torch.float32
                    ).view(1)
                elif self.label_type == "multiclass":
                    y = torch.tensor(
                        np.argmax(raw_data["labels_multiclass"][i][0]),
                        dtype=torch.long
                    ).view(1)

            data = Data(
                x=x,
                edge_index=edge_index.contiguous(),
                edge_attr=edge_attr,
                y=y,
                idx=i,
            )

            # Add edge_mask only for cascade task
            if edge_mask is not None:
                data.edge_mask = edge_mask

            data_list.append(data)

        return data_list

    def _create_splits(
        self,
        num_samples: int,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Create train/val/test splits (80/10/10 per configs/splits.yaml)."""

        if self.split_type == "blocked":
            # Blocked temporal split (no data leakage)
            train_end = int(num_samples * train_frac)
            val_end = int(num_samples * (train_frac + val_frac))

            return {
                "train": np.arange(0, train_end),
                "val": np.arange(train_end, val_end),
                "test": np.arange(val_end, num_samples),
            }
        else:
            # Random split (potential leakage for temporal data)
            np.random.seed(42)  # Reproducibility
            indices = np.random.permutation(num_samples)

            train_end = int(num_samples * train_frac)
            val_end = int(num_samples * (train_frac + val_frac))

            return {
                "train": indices[:train_end],
                "val": indices[train_end:val_end],
                "test": indices[val_end:],
            }

    def _load_split_info(self):
        """Load split information."""
        split_info_path = Path(self.processed_dir) / "split_info.json"
        if split_info_path.exists():
            with open(split_info_path) as f:
                self.split_info = json.load(f)
        else:
            self.split_info = {}

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"task={self.task}, "
            f"label_type={self.label_type}, "
            f"split={self.split}, "
            f"num_graphs={len(self)})"
        )


def _download_with_progress(url: str, dest: Path):
    """Download file with progress bar."""

    class DownloadProgress(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def get_dataset_info(root: str, name: str) -> Dict:
    """Get information about a PowerGraph dataset without loading it."""
    raw_dir = Path(root) / "raw" / name

    if not raw_dir.exists():
        return {"exists": False, "message": f"Dataset not found at {raw_dir}"}

    # Load basic info
    try:
        bf = mat73.loadmat(raw_dir / "Bf.mat")
        blist = mat73.loadmat(raw_dir / "blist.mat")

        num_samples = len(bf["B_f_tot"])
        sample_nodes = bf["B_f_tot"][0][0].reshape(-1, 3).shape[0]
        num_edges = len(blist["bList"])

        return {
            "exists": True,
            "grid": name,
            "num_samples": num_samples,
            "num_nodes": sample_nodes,
            "num_edges": num_edges,
            "node_features": 3,
            "edge_features": 4,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}
