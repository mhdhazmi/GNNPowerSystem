"""
Data API endpoints
Serve dataset information and samples
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import random

router = APIRouter()


# Dataset statistics
DATASET_STATS = {
    "ieee24": {
        "name": "IEEE 24-bus RTS",
        "nodes": 24,
        "edges": 38,
        "samples": {"train": 8000, "val": 1000, "test": 1000},
        "cascade_rate": 0.067,
        "features": {"node": 6, "edge": 3},
        "feature_names": {
            "node": ["P", "Q", "V", "theta", "bus_type", "zone"],
            "edge": ["X", "rating", "status"]
        }
    },
    "ieee118": {
        "name": "IEEE 118-bus",
        "nodes": 118,
        "edges": 186,
        "samples": {"train": 8000, "val": 1000, "test": 1000},
        "cascade_rate": 0.057,
        "features": {"node": 6, "edge": 3},
        "feature_names": {
            "node": ["P", "Q", "V", "theta", "bus_type", "zone"],
            "edge": ["X", "rating", "status"]
        }
    }
}


@router.get("/statistics/{grid}")
async def get_dataset_statistics(grid: str):
    """Get dataset statistics for a grid"""
    if grid not in DATASET_STATS:
        raise HTTPException(status_code=404, detail=f"Grid {grid} not found")
    return DATASET_STATS[grid]


@router.get("/statistics")
async def get_all_statistics():
    """Get all dataset statistics"""
    return DATASET_STATS


@router.get("/sample/{grid}/{split}")
async def get_sample(grid: str, split: str, sample_id: int = None):
    """Get a sample from the dataset (mock data for demo)"""
    if grid not in DATASET_STATS:
        raise HTTPException(status_code=404, detail=f"Grid {grid} not found")

    if split not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="Split must be train, val, or test")

    stats = DATASET_STATS[grid]
    n_nodes = stats["nodes"]
    n_edges = stats["edges"]

    # Generate mock sample
    sample = {
        "id": sample_id or random.randint(0, stats["samples"][split] - 1),
        "split": split,
        "grid": grid,
        "node_features": [
            [random.gauss(0, 1) for _ in range(6)]
            for _ in range(n_nodes)
        ],
        "edge_features": [
            [random.random() for _ in range(3)]
            for _ in range(n_edges)
        ],
        "label": random.random() < stats["cascade_rate"],
    }

    return sample


@router.get("/graph/{grid}")
async def get_graph_topology(grid: str):
    """Get graph topology for visualization"""
    if grid not in DATASET_STATS:
        raise HTTPException(status_code=404, detail=f"Grid {grid} not found")

    stats = DATASET_STATS[grid]
    n_nodes = stats["nodes"]

    # Generate simplified topology
    if grid == "ieee24":
        # IEEE 24-bus simplified layout
        nodes = [
            {"id": i, "type": "gen" if i in [7, 13, 14, 15, 16, 18, 21, 22, 23] else "load"}
            for i in range(1, n_nodes + 1)
        ]
        edges = [
            [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 9], [3, 24], [4, 9], [5, 10],
            [6, 10], [7, 8], [8, 9], [8, 10], [9, 11], [9, 12], [10, 11], [10, 12],
            [11, 13], [11, 14], [12, 13], [12, 23], [13, 23], [14, 16], [15, 16],
            [15, 21], [15, 24], [16, 17], [16, 19], [17, 18], [17, 22], [18, 21],
            [19, 20], [20, 23], [21, 22]
        ]
    else:
        # IEEE 118-bus (simplified random topology)
        nodes = [
            {"id": i, "type": "gen" if random.random() > 0.7 else "load"}
            for i in range(1, n_nodes + 1)
        ]
        edges = []
        for i in range(1, n_nodes):
            if random.random() > 0.3:
                edges.append([i, i + 1])
            if i + 12 <= n_nodes and random.random() > 0.5:
                edges.append([i, i + 12])

    return {
        "nodes": nodes,
        "edges": edges[:stats["edges"]],
        "stats": stats
    }
