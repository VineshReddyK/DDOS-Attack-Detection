"""
FastAPI application entry point.

Run locally:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Or via Makefile:
    make serve
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import router
from api.model_registry import registry
from utils.logger import setup_logger

setup_logger("ddos_api", log_file="reports/api.log")
logger = logging.getLogger("ddos_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DDoS Detection API — loading models...")
    registry.load_all()
    loaded = [k for k, v in registry.loaded_models.items() if v]
    logger.info("Models ready: %s", loaded if loaded else "none (train first)")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="DDoS Attack Detection API",
    description=(
        "REST API for real-time DDoS attack detection using machine learning. "
        "Supports Random Forest, K-Means anomaly detection, ANN, and CNN-LSTM models."
    ),
    version="1.0.0",
    contact={
        "name": "Vinesh Reddy Kankanalapally",
        "url": "https://linkedin.com/in/vinesh-reddy-kankanalapally",
        "email": "vineshreddyy.k@gmail.com",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
