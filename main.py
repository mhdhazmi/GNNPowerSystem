"""
PowerGraph GNN - Physics-guided Graph Neural Networks for Power Grid Analytics

Main entry point for running experiments.

Usage:
    python main.py --config configs/base.yaml
    python main.py --config configs/debug.yaml --task pf
"""

import argparse
from pathlib import Path

from src.utils import ExperimentConfig, get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="PowerGraph GNN Training")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config file"
    )
    parser.add_argument("--task", type=str, choices=["pf", "opf", "cascade", "multitask"])
    parser.add_argument("--seed", type=int, help="Override seed from config")
    parser.add_argument("--debug", action="store_true", help="Use debug config")

    args = parser.parse_args()

    # Load config
    config_path = Path("configs/debug.yaml") if args.debug else Path(args.config)
    config = load_config(config_path)

    # Override with CLI args
    if args.task:
        config["data"]["task"] = args.task
    if args.seed:
        config["project"]["seed"] = args.seed

    # Set seed for reproducibility
    seed = config["project"]["seed"]
    set_seed(seed)

    # Get device
    device = get_device()

    print("=" * 60)
    print("PowerGraph GNN")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Task: {config['data']['task']}")
    print(f"Grid: {config['data']['grid']}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print("=" * 60)

    # TODO: Implement training pipeline in subsequent WPs
    print("\n[INFO] Training pipeline not yet implemented.")
    print("[INFO] Run 'python scripts/smoke_test.py' to verify setup.")


if __name__ == "__main__":
    main()
