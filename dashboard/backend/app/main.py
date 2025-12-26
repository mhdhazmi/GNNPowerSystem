"""
Physics-SSL Dashboard Backend
FastAPI server for model inference and data serving
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path

from app.routers import inference, data, results

# Initialize FastAPI app
app = FastAPI(
    title="Physics-SSL Dashboard API",
    description="Backend API for the Physics-Guided SSL Power Grid Dashboard",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(results.router, prefix="/api/results", tags=["results"])


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Physics-SSL Dashboard API",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models_loaded": False,  # Will be True when models are loaded
        "gpu_available": False,  # Will check torch.cuda.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
