import datetime
import time
from typing import Dict, List

from config import API_URL, MODEL_INPUT_STEPS, PREDICTOR_URL
from shared.logger import send_system_log
from shared.schemas import (
    GenericResponse,
    MetricHistoryRead,
    MetricPoint,
    MetricRead,
    PredictData,
    PredictionRequest,
    PredictionResponse,
    RawCpuCreate,
    SettingsRead,
    StatusResponse,
)
from shared.utils import async_http_request


async def save_raw_reading(ts: datetime.datetime, cpu_value: float) -> None:
    await async_http_request(
        method="POST",
        url=f"{API_URL}/metrics/raw",
        payload=RawCpuCreate(ts=ts, cpu_value=cpu_value),
        response_model=GenericResponse,
    )


async def sync_by_id(row_id: int) -> None:
    """Trigger actual_cpu sync for a row — the API resolves the value from raw_cpu_readings."""
    url = f"{API_URL}/metrics/{row_id}/actual"
    response = await async_http_request(
        method="PUT",
        url=url,
        response_model=GenericResponse,
    )
    if not response or response.status != "success":
        await send_system_log(
            f"Failed to sync row {row_id}: {response.message if response else 'No response'}",
            level="ERROR",
            service="collector",
        )


async def save_new_prediction(
    input_cpu: float,
    input_ram: float,
    input_rps: float,
    predicted_cpu: float,
    horizon_seconds: int,
) -> MetricRead | None:
    """Save a new prediction. Returns the saved row (with id) or None on failure."""
    url = f"{API_URL}/metrics/predict"
    response = await async_http_request(
        method="POST",
        url=url,
        payload=PredictData(
            input_cpu=input_cpu,
            input_ram=input_ram,
            input_rps=input_rps,
            predicted_cpu=predicted_cpu,
            horizon_seconds=horizon_seconds,
        ),
        response_model=MetricRead,
    )
    if not response:
        await send_system_log(
            "Failed to save prediction",
            level="ERROR",
            service="collector",
        )
    return response


_settings_cache: SettingsRead | None = None
_settings_cache_ts: float = 0.0
# 30s TTL: trades toggle immediacy for ~17k fewer daily API requests
_SETTINGS_TTL = 30.0


async def get_system_settings() -> SettingsRead:
    global _settings_cache, _settings_cache_ts
    now = time.monotonic()
    if _settings_cache is not None and (now - _settings_cache_ts) < _SETTINGS_TTL:
        return _settings_cache
    response = await async_http_request(
        method="GET", url=f"{API_URL}/settings", response_model=SettingsRead
    )
    _settings_cache = response
    _settings_cache_ts = now
    return response


async def get_unsynced_rows() -> list[MetricRead]:
    """Return rows where actual_cpu IS NULL and target_ts has already passed."""
    resp = await async_http_request(
        method="GET",
        url=f"{API_URL}/metrics/unsynced",
        response_model=list[MetricRead],
    )
    return resp if resp else []


async def get_recent_history(limit: int = 9) -> list[MetricRead]:
    """Fetch the last N records from the DB to restore the history buffer."""
    resp = await async_http_request(
        method="GET",
        url=f"{API_URL}/metrics/history",
        payload=MetricHistoryRead(limit=limit),
        response_model=list[MetricRead],
    )
    return resp if resp else []


_predictor_status_cache: StatusResponse | None = None
_predictor_status_cache_ts: float = 0.0
_PREDICTOR_STATUS_TTL = 60.0

_FORECAST_HORIZON_FALLBACK = 12


async def get_predictor_model_config() -> tuple[int, int]:
    """
    Return (window_size, forecast_horizon) from the predictor /status endpoint.
    Falls back to (MODEL_INPUT_STEPS, 12) if the predictor is unreachable.
    """
    global _predictor_status_cache, _predictor_status_cache_ts
    now = time.monotonic()
    if (
        _predictor_status_cache is not None
        and (now - _predictor_status_cache_ts) < _PREDICTOR_STATUS_TTL
    ):
        return (
            _predictor_status_cache.window_size,
            _predictor_status_cache.forecast_horizon,
        )
    try:
        response = await async_http_request(
            method="GET",
            url=f"{PREDICTOR_URL}/status",
            response_model=StatusResponse,
        )
        if response:
            _predictor_status_cache = response
            _predictor_status_cache_ts = now
            return response.window_size, response.forecast_horizon
    except Exception:
        pass
    return MODEL_INPUT_STEPS, _FORECAST_HORIZON_FALLBACK


async def get_prediction(history: list[dict[str, float]]) -> float | None:
    """
    Send a window of measurements to the predictor service.
    Returns predicted CPU (float) or None on error.
    """
    history_points = [MetricPoint.model_validate(point) for point in history]

    response = await async_http_request(
        method="POST",
        url=f"{PREDICTOR_URL}/predict",
        payload=PredictionRequest(history=history_points),
        response_model=PredictionResponse,
    )

    if response:
        return response.predicted_cpu

    await send_system_log(
        "Predictor API error: no response received",
        level="ERROR",
        service="collector",
    )
    return None
