"""
FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or via Makefile:
    make serve
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # noqa: E402

from api.model_registry import registry  # noqa: E402
from api.routes import router  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, RedirectResponse  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

setup_logger("ddos_api", log_file="reports/api.log")
logger = logging.getLogger("ddos_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DDoS Detection API v2 — loading models...")
    registry.load_all()
    loaded = [k for k, v in registry.loaded_models.items() if v]
    logger.info("Models ready: %s", loaded if loaded else "none (train first)")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="DDoS Attack Detection API",
    description=(
        "Real-time DDoS attack detection using ML ensemble (RF + ANN + CNN-LSTM + KMeans). "
        "Features: JWT auth, SHAP explainability, data drift detection, WebSocket streaming, "
        "Prometheus metrics, rate limiting."
    ),
    version="2.0.0",
    contact={
        "name": "Vinesh Reddy Kankanalapally",
        "url": "https://linkedin.com/in/vinesh-reddy-kankanalapally",
        "email": "vineshreddyy.k@gmail.com",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler  # noqa: E402
    from slowapi.errors import RateLimitExceeded  # noqa: E402
    from slowapi.util import get_remote_address  # noqa: E402

    limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("Rate limiting enabled: 200 req/min")
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled")

# ── Prometheus metrics ────────────────────────────────────────────────────────
try:
    from api.middleware.metrics import PrometheusMiddleware, metrics_endpoint  # noqa: E402
    app.add_middleware(PrometheusMiddleware)

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        return await metrics_endpoint()

    logger.info("Prometheus metrics enabled at /metrics")
except Exception as e:
    logger.warning("Prometheus middleware unavailable: %s", e)

# ── CORS ──────────────────────────────────────────────────────────────────────
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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
