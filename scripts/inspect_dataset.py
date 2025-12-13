#!/usr/bin/env python3
"""
PowerGraph Dataset Inspection Script

Validates and inspects the PowerGraph dataset.

Usage:
    python scripts/inspect_dataset.py --grid ieee24
    python scripts/inspect_dataset.py --grid ieee24 --download
    python scripts/inspect_dataset.py --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def inspect_raw_data(data_root: Path, grid: str):
    """Inspect raw .mat files before processing."""
    import mat73

    print(f"\n{'='*60}")
    print(f"RAW DATA INSPECTION: {grid}")
    print(f"{'='*60}")

    # Check nested structure first (from Figshare archive)
    raw_dir = data_root / "raw" / grid / grid / "raw"
    if not raw_dir.exists():
        # Fallback to flat structure
        raw_dir = data_root / "raw" / grid

    if not raw_dir.exists():
        print(f"  ✗ Raw directory not found: {raw_dir}")
        print(f"  Run with --download to fetch the data")
        return False

    files = {
        "blist.mat": "Edge index",
        "Bf.mat": "Node features",
        "Ef.mat": "Edge features",
        "of_bi.mat": "Binary labels",
        "of_mc.mat": "Multiclass labels",
        "of_reg.mat": "Regression labels",
        "exp.mat": "Explanation masks",
    }

    all_present = True
    for fname, desc in files.items():
        fpath = raw_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  ✓ {fname:<15} ({desc:<20}) - {size_mb:.1f} MB")
        else:
            print(f"  ✗ {fname:<15} ({desc:<20}) - MISSING")
            all_present = False

    if not all_present:
        return False

    # Load and inspect
    print(f"\n  Loading data structures...")

    try:
        blist = mat73.loadmat(raw_dir / "blist.mat")
        bf = mat73.loadmat(raw_dir / "Bf.mat")
        ef = mat73.loadmat(raw_dir / "Ef.mat")
        of_bi = mat73.loadmat(raw_dir / "of_bi.mat")
        exp = mat73.loadmat(raw_dir / "exp.mat")

        edge_index = blist["bList"]
        node_features = bf["B_f_tot"]
        edge_features = ef["E_f_post"]
        labels = of_bi["output_features"]
        explanations = exp["explainations"]

        num_samples = len(node_features)
        num_edges = len(edge_index)

        # Sample first graph
        sample_x = np.array(node_features[0][0]).reshape(-1, 3)
        sample_e = np.array(edge_features[0][0]).reshape(-1, 4)
        num_nodes = sample_x.shape[0]

        print(f"\n  Dataset Statistics:")
        print(f"    - Number of graphs: {num_samples}")
        print(f"    - Nodes per graph: {num_nodes}")
        print(f"    - Base edges: {num_edges}")
        print(f"    - Node feature dim: 3 (P_net, S_net, V)")
        print(f"    - Edge feature dim: 4 (P_flow, Q_flow, X, rating)")

        # Label distribution
        pos_count = sum(1 for l in labels if l[0] == 1)
        neg_count = num_samples - pos_count
        print(f"\n  Binary Label Distribution:")
        print(f"    - Class 0 (no cascade): {neg_count} ({100*neg_count/num_samples:.1f}%)")
        print(f"    - Class 1 (cascade): {pos_count} ({100*pos_count/num_samples:.1f}%)")

        # Explanation coverage
        has_exp = sum(1 for e in explanations if e[0] is not None)
        print(f"\n  Explanation Masks:")
        print(f"    - Graphs with explanations: {has_exp} ({100*has_exp/num_samples:.1f}%)")

        print(f"\n  ✓ Raw data inspection passed")
        return True

    except Exception as e:
        print(f"\n  ✗ Error loading data: {e}")
        return False


def inspect_processed_data(data_root: Path, grid: str, task: str = "cascade", label_type: str = "binary"):
    """Inspect processed PyG dataset."""
    from src.data import PowerGraphDataset

    print(f"\n{'='*60}")
    print(f"PROCESSED DATA INSPECTION: {grid}/{task}/{label_type}")
    print(f"{'='*60}")

    try:
        # Load all splits
        for split in ["train", "val", "test"]:
            print(f"\n  Loading {split} split...")
            dataset = PowerGraphDataset(
                root=str(data_root),
                name=grid,
                task=task,
                label_type=label_type,
                split=split,
            )

            print(f"    - Samples: {len(dataset)}")

            if len(dataset) > 0:
                sample = dataset[0]
                print(f"    - Sample graph:")
                print(f"      - x shape: {sample.x.shape}")
                print(f"      - edge_index shape: {sample.edge_index.shape}")
                print(f"      - edge_attr shape: {sample.edge_attr.shape}")
                print(f"      - y: {sample.y}")
                print(f"      - edge_mask shape: {sample.edge_mask.shape}")
                print(f"      - edge_mask sum: {sample.edge_mask.sum().item():.0f}")

        print(f"\n  ✓ Processed data inspection passed")
        return True

    except FileNotFoundError:
        print(f"  ✗ Processed data not found. Run processing first.")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_dataset(data_root: Path, grid: str):
    """Run validation checks on the dataset."""
    from src.data import PowerGraphDataset

    print(f"\n{'='*60}")
    print(f"VALIDATION CHECKS: {grid}")
    print(f"{'='*60}")

    checks_passed = 0
    checks_total = 0

    try:
        dataset = PowerGraphDataset(
            root=str(data_root),
            name=grid,
            task="cascade",
            label_type="binary",
            split="train",
        )

        # Check 1: No NaN in features
        checks_total += 1
        has_nan = False
        for i in range(min(100, len(dataset))):
            if torch.isnan(dataset[i].x).any():
                has_nan = True
                break
            if torch.isnan(dataset[i].edge_attr).any():
                has_nan = True
                break

        if not has_nan:
            print(f"  ✓ Check 1: No NaN values in features")
            checks_passed += 1
        else:
            print(f"  ✗ Check 1: Found NaN values in features")

        # Check 2: Edge index in valid range
        checks_total += 1
        valid_edges = True
        for i in range(min(100, len(dataset))):
            data = dataset[i]
            if data.edge_index.max() >= data.x.shape[0]:
                valid_edges = False
                break
            if data.edge_index.min() < 0:
                valid_edges = False
                break

        if valid_edges:
            print(f"  ✓ Check 2: Edge indices in valid range")
            checks_passed += 1
        else:
            print(f"  ✗ Check 2: Invalid edge indices found")

        # Check 3: Consistent node count per grid
        checks_total += 1
        node_counts = [dataset[i].x.shape[0] for i in range(min(100, len(dataset)))]
        if len(set(node_counts)) == 1:
            print(f"  ✓ Check 3: Consistent node count ({node_counts[0]} nodes)")
            checks_passed += 1
        else:
            print(f"  ✗ Check 3: Inconsistent node counts: {set(node_counts)}")

        # Check 4: Explanation masks are binary
        checks_total += 1
        valid_masks = True
        for i in range(min(100, len(dataset))):
            mask = dataset[i].edge_mask
            if not ((mask == 0) | (mask == 1)).all():
                valid_masks = False
                break

        if valid_masks:
            print(f"  ✓ Check 4: Explanation masks are binary")
            checks_passed += 1
        else:
            print(f"  ✗ Check 4: Non-binary explanation masks found")

        # Check 5: Labels in expected range
        checks_total += 1
        labels = [dataset[i].y.item() for i in range(min(100, len(dataset)))]
        unique_labels = set(labels)
        if unique_labels.issubset({0.0, 1.0}):
            print(f"  ✓ Check 5: Binary labels valid (found: {unique_labels})")
            checks_passed += 1
        else:
            print(f"  ✗ Check 5: Unexpected labels: {unique_labels}")

        # Summary
        print(f"\n  Validation: {checks_passed}/{checks_total} checks passed")
        return checks_passed == checks_total

    except Exception as e:
        print(f"  ✗ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_data(data_root: Path, grid: str):
    """Download and process the PowerGraph dataset."""
    from src.data import PowerGraphDataset

    print(f"\n{'='*60}")
    print(f"DOWNLOADING AND PROCESSING: {grid}")
    print(f"{'='*60}")

    try:
        # This will trigger download and processing
        dataset = PowerGraphDataset(
            root=str(data_root),
            name=grid,
            task="cascade",
            label_type="binary",
            split="train",
        )
        print(f"  ✓ Dataset ready: {len(dataset)} training samples")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Inspect PowerGraph dataset")
    parser.add_argument("--grid", type=str, default="ieee24",
                       choices=["ieee24", "ieee39", "ieee118", "uk"],
                       help="Grid to inspect")
    parser.add_argument("--data_root", type=str, default="./data",
                       help="Data root directory")
    parser.add_argument("--download", action="store_true",
                       help="Download data if not present")
    parser.add_argument("--all", action="store_true",
                       help="Inspect all grids")
    parser.add_argument("--validate", action="store_true",
                       help="Run validation checks")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    grids = ["ieee24", "ieee39", "ieee118", "uk"] if args.all else [args.grid]

    print("\n" + "=" * 60)
    print("POWERGRAPH DATASET INSPECTOR")
    print("=" * 60)

    for grid in grids:
        if args.download:
            download_data(data_root, grid)

        # Raw inspection
        raw_ok = inspect_raw_data(data_root, grid)

        if raw_ok:
            # Try to load processed
            processed_ok = inspect_processed_data(data_root, grid)

            if args.validate and processed_ok:
                validate_dataset(data_root, grid)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
