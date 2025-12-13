"""
Configuration loading and management utilities.

Supports YAML configs with inheritance via _base_ key.

Usage:
    from src.utils.config import load_config
    config = load_config("configs/pf_baseline.yaml")
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict:
    """
    Load YAML config with inheritance support.

    Configs can inherit from a base config using the _base_ key:
        _base_: base.yaml
        model:
            hidden_dim: 256  # Overrides base

    Args:
        config_path: Path to YAML config file

    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_config(base_path)
        config = deep_merge(base_config, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.

    Args:
        base: Base dictionary
        override: Dictionary with overriding values

    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def save_config(config: dict, path: str | Path) -> None:
    """Save config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


@dataclass
class ExperimentConfig:
    """Typed experiment configuration."""

    # Project
    name: str = "powergraph-gnn"
    seed: int = 42
    device: str = "cuda"

    # Data
    data_root: str = "./data"
    grid: str = "ieee24"
    task: str = "pf"
    split_type: str = "blocked"
    label_fraction: float = 1.0

    # Model
    model_type: str = "physics_guided"
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 15
    tasks: list = field(default_factory=lambda: ["pf"])

    # SSL
    ssl_enabled: bool = False
    ssl_method: str = "combined"
    node_mask_ratio: float = 0.15
    edge_mask_ratio: float = 0.10

    # Logging
    log_dir: str = "./outputs"
    tensorboard: bool = True
    log_every: int = 10

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create config from dictionary, flattening nested structure."""
        flat = {}

        # Flatten nested config
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    flat[key] = val
            else:
                flat[section] = values

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in flat.items() if k in known_fields}

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        config_dict = load_config(path)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
