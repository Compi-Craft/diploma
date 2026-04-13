"""
CPU-intensive test service for dataset generation.
Exposes /compute endpoint that does bcrypt hashing with configurable rounds.
bcrypt runs in a thread pool to avoid blocking the async event loop.
Prometheus metrics: request count, request duration, in-flight requests.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import bcrypt
from fastapi import FastAPI, Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

app = FastAPI(title="CPU Load Service")

# Thread pool для CPU-bound bcrypt (не блокує event loop)
# max_workers=32: при rounds=10 (~100ms/req) → ~320 RPS capacity
_executor = ThreadPoolExecutor(max_workers=32)

# ── Prometheus metrics ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
IN_FLIGHT = Gauge(
    "http_requests_in_flight",
    "Number of requests currently being processed",
)


# ── Schemas ─────────────────────────────────────────────────────────
class ComputeRequest(BaseModel):
    data: str = "hello"
    rounds: int = Field(default=10, ge=4, le=16)


class ComputeResponse(BaseModel):
    hash: str
    duration_ms: float


def _bcrypt_hash(data: str, rounds: int) -> bytes:
    return bcrypt.hashpw(data.encode(), bcrypt.gensalt(rounds=rounds))


# ── Endpoints ───────────────────────────────────────────────────────
@app.post("/compute", response_model=ComputeResponse)
async def compute(req: ComputeRequest) -> ComputeResponse:
    """CPU-intensive: bcrypt hash runs in thread pool, event loop stays free."""
    IN_FLIGHT.inc()
    start = time.perf_counter()
    try:
        loop = asyncio.get_event_loop()
        hashed = await loop.run_in_executor(
            _executor, _bcrypt_hash, req.data, req.rounds
        )
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels("POST", "/compute", "200").inc()
        REQUEST_DURATION.labels("/compute").observe(duration)
        return ComputeResponse(hash=hashed.decode(), duration_ms=duration * 1000)
    except Exception:
        REQUEST_COUNT.labels("POST", "/compute", "500").inc()
        raise
    finally:
        IN_FLIGHT.dec()


@app.get("/health")
async def health() -> dict:
    REQUEST_COUNT.labels("GET", "/health", "200").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
