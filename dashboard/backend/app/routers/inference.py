"""
Inference API endpoints
Run model predictions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import random
import time

router = APIRouter()


class CascadeRequest(BaseModel):
    grid: str
    sample_id: Optional[int] = None


class PowerFlowRequest(BaseModel):
    grid: str
    sample_id: Optional[int] = None


class LineFlowRequest(BaseModel):
    grid: str
    sample_id: Optional[int] = None


class EmbeddingRequest(BaseModel):
    grid: str
    sample_id: Optional[int] = None
    method: str = "pca"  # pca or tsne


@router.post("/cascade")
async def predict_cascade(request: CascadeRequest):
    """Predict cascade probability for a sample"""
    start_time = time.time()

    # Simulate model inference
    await _simulate_delay()

    # Mock prediction
    is_cascade = random.random() > 0.9
    probability = random.random() * 0.3 + (0.7 if is_cascade else 0)

    latency = (time.time() - start_time) * 1000

    return {
        "prediction": int(is_cascade),
        "probability": round(probability, 4),
        "latency_ms": round(latency, 2),
        "model": "physics_ssl",
        "grid": request.grid,
    }


@router.post("/power_flow")
async def predict_power_flow(request: PowerFlowRequest):
    """Predict bus voltages"""
    start_time = time.time()

    await _simulate_delay()

    n_nodes = 24 if request.grid == "ieee24" else 118

    # Mock voltage predictions
    predictions = [
        {
            "bus_id": i + 1,
            "v": round(0.95 + random.random() * 0.1, 4),
            "theta": round((random.random() - 0.5) * 0.2, 4)
        }
        for i in range(n_nodes)
    ]

    latency = (time.time() - start_time) * 1000

    return {
        "predictions": predictions,
        "mse": round(random.random() * 0.01, 5),
        "latency_ms": round(latency, 2),
        "model": "physics_ssl",
        "grid": request.grid,
    }


@router.post("/line_flow")
async def predict_line_flow(request: LineFlowRequest):
    """Predict branch power flows"""
    start_time = time.time()

    await _simulate_delay()

    n_edges = 38 if request.grid == "ieee24" else 186

    # Mock flow predictions
    predictions = [
        {
            "edge_id": i + 1,
            "p_mw": round(random.random() * 100 - 50, 2),
            "q_mvar": round(random.random() * 50 - 25, 2)
        }
        for i in range(n_edges)
    ]

    latency = (time.time() - start_time) * 1000

    return {
        "predictions": predictions,
        "mse": round(random.random() * 0.05, 5),
        "latency_ms": round(latency, 2),
        "model": "physics_ssl",
        "grid": request.grid,
    }


@router.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    """Get node embeddings projected to 2D"""
    start_time = time.time()

    await _simulate_delay()

    n_nodes = 24 if request.grid == "ieee24" else 118

    # Mock 2D embeddings
    embeddings = [
        {
            "node_id": i + 1,
            "x": round(random.gauss(0, 1), 4),
            "y": round(random.gauss(0, 1), 4),
            "type": "gen" if random.random() > 0.7 else "load"
        }
        for i in range(n_nodes)
    ]

    latency = (time.time() - start_time) * 1000

    return {
        "embeddings": embeddings,
        "method": request.method,
        "latency_ms": round(latency, 2),
        "grid": request.grid,
    }


@router.post("/explain")
async def explain_prediction(grid: str, sample_id: Optional[int] = None):
    """Get feature importance via Integrated Gradients"""
    start_time = time.time()

    await _simulate_delay()

    n_edges = 38 if grid == "ieee24" else 186

    # Mock edge importance scores
    edge_importance = [
        {
            "edge_id": i + 1,
            "importance": round(random.random(), 4)
        }
        for i in range(n_edges)
    ]

    # Sort by importance
    edge_importance.sort(key=lambda x: x["importance"], reverse=True)

    latency = (time.time() - start_time) * 1000

    return {
        "edge_importance": edge_importance,
        "top_5": edge_importance[:5],
        "method": "integrated_gradients",
        "latency_ms": round(latency, 2),
        "grid": grid,
    }


async def _simulate_delay():
    """Simulate model inference delay"""
    import asyncio
    await asyncio.sleep(random.uniform(0.1, 0.3))
