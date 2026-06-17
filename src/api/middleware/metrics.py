"""
Prometheus metrics middleware — exposes /metrics for Grafana scraping.
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def _prom():
    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
        return Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    except ImportError:
        raise RuntimeError("prometheus-client not installed.")


def build_metrics():
    Counter, Histogram, Gauge, _, _ = _prom()
    return {
        "requests_total": Counter(
            "ddos_api_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        ),
        "request_duration": Histogram(
            "ddos_api_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
        ),
        "predictions_total": Counter(
            "ddos_api_predictions_total",
            "Total predictions made",
            ["model_type", "result"],
        ),
        "attack_detections_total": Counter(
            "ddos_api_attack_detections_total",
            "Total attack detections",
            ["model_type"],
        ),
        "active_connections": Gauge(
            "ddos_api_active_ws_connections",
            "Active WebSocket connections",
        ),
    }


_metrics = None


def get_metrics():
    global _metrics
    if _metrics is None:
        try:
            _metrics = build_metrics()
        except Exception as e:
            logger.warning("Prometheus metrics unavailable: %s", e)
    return _metrics


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        m = get_metrics()
        if m:
            endpoint = request.url.path
            m["requests_total"].labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(response.status_code),
            ).inc()
            m["request_duration"].labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(duration)

        return response


async def metrics_endpoint():
    _, _, _, generate_latest, CONTENT_TYPE_LATEST = _prom()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
