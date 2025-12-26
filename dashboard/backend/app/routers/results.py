"""
Results API endpoints
Serve experimental results data
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

router = APIRouter()

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "results"


@router.get("/comparison")
async def get_comparison_results():
    """Get SSL vs Scratch comparison results"""
    try:
        with open(DATA_DIR / "comparison.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Comparison results not found")


@router.get("/graphmae")
async def get_graphmae_results():
    """Get GraphMAE vs Physics-SSL comparison"""
    try:
        with open(DATA_DIR / "graphmae.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="GraphMAE results not found")


@router.get("/robustness")
async def get_robustness_results():
    """Get robustness analysis results"""
    try:
        with open(DATA_DIR / "robustness.json") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Robustness results not found")


@router.get("/summary")
async def get_summary():
    """Get summary statistics"""
    return {
        "key_metrics": {
            "f1_improvement": "+35.5%",
            "peak_f1": 0.994,
            "variance_reduction": "-54%",
            "tasks_tested": 3,
        },
        "grids": ["ieee24", "ieee118"],
        "tasks": ["cascade", "powerflow", "lineflow"],
        "label_fractions": ["10%", "50%", "100%"],
    }
