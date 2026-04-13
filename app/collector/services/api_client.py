import datetime
from typing import Dict, List

from config import API_URL, PREDICTOR_URL
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
)
from shared.utils import async_http_request


async def save_raw_reading(ts: datetime.datetime, cpu_value: float) -> None:
    """Зберігає сире вимірювання CPU з міткою часу."""
    await async_http_request(
        method="POST",
        url=f"{API_URL}/metrics/raw",
        payload=RawCpuCreate(ts=ts, cpu_value=cpu_value),
        response_model=GenericResponse,
    )


async def sync_by_id(row_id: int) -> None:
    """Тригерить sync для рядка — API сам знайде точне значення з raw_cpu_readings."""
    url = f"{API_URL}/metrics/{row_id}/actual"
    response = await async_http_request(
        method="PUT",
        url=url,
        response_model=GenericResponse,
    )
    if not response or response.status != "success":
        await send_system_log(
            f"⚠️ Failed to sync row {row_id}: {response.message if response else 'No response'}",
            level="ERROR",
            service="collector",
        )


async def save_new_prediction(
    input_cpu: float, input_ram: float, input_rps: float, predicted_cpu: float
) -> MetricRead | None:
    """Зберігає новий прогноз. Повертає збережений рядок (з id) або None."""
    url = f"{API_URL}/metrics/predict"
    response = await async_http_request(
        method="POST",
        url=url,
        payload=PredictData(
            input_cpu=input_cpu,
            input_ram=input_ram,
            input_rps=input_rps,
            predicted_cpu=predicted_cpu,
        ),
        response_model=MetricRead,
    )
    if not response:
        await send_system_log(
            "⚠️ Failed to save prediction",
            level="ERROR",
            service="collector",
        )
    return response


async def get_system_settings() -> SettingsRead:
    response = await async_http_request(
        method="GET", url=f"{API_URL}/settings", response_model=SettingsRead
    )
    return response


async def get_unsynced_rows() -> list[MetricRead]:
    """Повертає рядки де actual_cpu IS NULL і target_ts вже минув."""
    resp = await async_http_request(
        method="GET",
        url=f"{API_URL}/metrics/unsynced",
        response_model=list[MetricRead],
    )
    return resp if resp else []


async def get_recent_history(limit: int = 9) -> list[MetricRead]:
    """Отримує останні N записів з бази для відновлення буфера."""
    resp = await async_http_request(
        method="GET",
        url=f"{API_URL}/metrics/history",
        payload=MetricHistoryRead(limit=limit),
        response_model=list[MetricRead],
    )
    return resp if resp else []


async def get_prediction(history: list[dict[str, float]]) -> float | None:
    """
    Відправляє вікно з 11 останніх вимірів до LSTM сервісу.
    Повертає прогноз CPU (float) або None у разі помилки.
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
        "⚠️ Predictor API Error: No response received",
        level="ERROR",
        service="collector",
    )
    return None
