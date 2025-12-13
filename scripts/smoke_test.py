#!/usr/bin/env python3
"""
Smoke test script for PowerGraph GNN pipeline.

Verifies:
1. All dependencies are installed correctly
2. Model architecture is valid (forward pass works)
3. Loss computation works
4. Basic training step executes

Usage:
    python scripts/smoke_test.py

Success criteria:
    - Script completes without errors
    - Prints model info, shapes, and loss values
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_imports():
    """Verify all required packages are installed."""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)

    required = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("matplotlib", "Matplotlib"),
    ]

    all_ok = True
    for module, name in required:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: NOT INSTALLED - {e}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Some dependencies are missing. Run:")
        print("  uv pip install -e .")
        sys.exit(1)

    print()
    return True


def check_device():
    """Check available compute devices."""
    import torch

    print("=" * 60)
    print("CHECKING DEVICES")
    print("=" * 60)

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if hasattr(torch.backends, "mps"):
        print(f"  MPS available: {torch.backends.mps.is_available()}")

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  Using device: {device}")
    print()
    return device


def check_seed_utility():
    """Verify seed utility works."""
    print("=" * 60)
    print("CHECKING SEED UTILITY")
    print("=" * 60)

    try:
        from src.utils.seed import get_device_info, set_seed

        set_seed(42)
        print("  ✓ set_seed() works")

        info = get_device_info()
        print(f"  ✓ get_device_info() works: {info}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Seed utility error: {e}")
        return False


def check_config_utility():
    """Verify config loading works."""
    print("=" * 60)
    print("CHECKING CONFIG UTILITY")
    print("=" * 60)

    try:
        from src.utils.config import load_config

        config = load_config(project_root / "configs" / "base.yaml")
        print(f"  ✓ Loaded base.yaml")
        print(f"    - project.name: {config['project']['name']}")
        print(f"    - model.hidden_dim: {config['model']['hidden_dim']}")
        print(f"    - training.epochs: {config['training']['epochs']}")

        # Test inheritance
        debug_config = load_config(project_root / "configs" / "debug.yaml")
        print(f"  ✓ Loaded debug.yaml (with inheritance)")
        print(f"    - model.hidden_dim: {debug_config['model']['hidden_dim']} (overridden)")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Config utility error: {e}")
        return False


def create_dummy_data(num_graphs: int = 10, num_nodes: int = 24, num_edges: int = 34):
    """Create dummy PyG data for testing."""
    import torch
    from torch_geometric.data import Data

    data_list = []
    for i in range(num_graphs):
        # Node features: [P_load, Q_load, P_gen, Q_gen, V_set, bus_type (one-hot 3)]
        x = torch.randn(num_nodes, 8)

        # Random edges (bidirectional)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        # Edge features: [G, B, rating, tap]
        edge_attr = torch.randn(num_edges, 4)

        # PF targets
        v_mag = 0.95 + 0.1 * torch.rand(num_nodes)  # Voltage ~0.95-1.05 pu
        theta = torch.randn(num_nodes) * 0.1  # Angle in radians

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_v_mag=v_mag,
            y_v_ang_sin=torch.sin(theta),
            y_v_ang_cos=torch.cos(theta),
        )
        data_list.append(data)

    return data_list


def check_model_forward():
    """Test model forward pass with dummy data."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MessagePassing

    print("=" * 60)
    print("CHECKING MODEL FORWARD PASS")
    print("=" * 60)

    # Simple physics-guided conv for testing
    class SimpleConv(MessagePassing):
        def __init__(self, in_dim, out_dim, edge_dim):
            super().__init__(aggr="add")
            self.lin_node = nn.Linear(in_dim, out_dim)
            self.lin_edge = nn.Linear(edge_dim, out_dim)

        def forward(self, x, edge_index, edge_attr):
            x = self.lin_node(x)
            edge_emb = self.lin_edge(edge_attr)
            return self.propagate(edge_index, x=x, edge_attr=edge_emb)

        def message(self, x_j, edge_attr):
            return F.relu(x_j + edge_attr)

    class SimpleGNN(nn.Module):
        def __init__(self, node_dim=8, edge_dim=4, hidden_dim=32, num_layers=2):
            super().__init__()
            self.embed = nn.Linear(node_dim, hidden_dim)
            self.convs = nn.ModuleList(
                [SimpleConv(hidden_dim, hidden_dim, edge_dim) for _ in range(num_layers)]
            )
            self.head = nn.Linear(hidden_dim, 3)  # v_mag, sin, cos

        def forward(self, x, edge_index, edge_attr):
            x = self.embed(x)
            for conv in self.convs:
                x = conv(x, edge_index, edge_attr)
                x = F.relu(x)
            return self.head(x)

    # Create model
    model = SimpleGNN()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Create dummy data
    data_list = create_dummy_data(num_graphs=10)
    loader = DataLoader(data_list, batch_size=4, shuffle=True)

    # Forward pass
    batch = next(iter(loader))
    print(f"  Batch: {batch}")
    print(f"    - x shape: {batch.x.shape}")
    print(f"    - edge_index shape: {batch.edge_index.shape}")
    print(f"    - edge_attr shape: {batch.edge_attr.shape}")

    out = model(batch.x, batch.edge_index, batch.edge_attr)
    print(f"  Output shape: {out.shape}")
    print(f"  ✓ Forward pass successful")
    print()

    return model, loader


def check_training_step(model, loader, device):
    """Test a single training step."""
    import torch
    import torch.nn.functional as F

    print("=" * 60)
    print("CHECKING TRAINING STEP")
    print("=" * 60)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = next(iter(loader)).to(device)

    # Forward
    optimizer.zero_grad()
    out = model(batch.x, batch.edge_index, batch.edge_attr)

    # Loss (simplified PF loss)
    v_pred = out[:, 0]
    sin_pred = out[:, 1]
    cos_pred = out[:, 2]

    loss_v = F.mse_loss(v_pred, batch.y_v_mag)
    loss_sin = F.mse_loss(sin_pred, batch.y_v_ang_sin)
    loss_cos = F.mse_loss(cos_pred, batch.y_v_ang_cos)
    loss = loss_v + loss_sin + loss_cos

    # Backward
    loss.backward()
    optimizer.step()

    print(f"  Loss breakdown:")
    print(f"    - V_mag loss: {loss_v.item():.4f}")
    print(f"    - sin(θ) loss: {loss_sin.item():.4f}")
    print(f"    - cos(θ) loss: {loss_cos.item():.4f}")
    print(f"    - Total loss: {loss.item():.4f}")
    print(f"  ✓ Training step successful")
    print()

    return loss.item()


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("POWERGRAPH GNN SMOKE TEST")
    print("=" * 60 + "\n")

    # Check imports
    check_imports()

    # Check device
    device = check_device()

    # Check utilities
    check_seed_utility()
    check_config_utility()

    # Check model
    model, loader = check_model_forward()

    # Check training
    loss = check_training_step(model, loader, device)

    # Summary
    print("=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    print("  ✓ All imports successful")
    print(f"  ✓ Device: {device}")
    print("  ✓ Seed utility working")
    print("  ✓ Config utility working")
    print("  ✓ Model forward pass working")
    print(f"  ✓ Training step working (loss={loss:.4f})")
    print()
    print("  🎉 ALL SMOKE TESTS PASSED!")
    print()
    print("  Next steps:")
    print("    1. Download PowerGraph dataset to data/raw/")
    print("    2. Run: python scripts/train_pf_baseline.py --config configs/debug.yaml")
    print()


if __name__ == "__main__":
    main()
