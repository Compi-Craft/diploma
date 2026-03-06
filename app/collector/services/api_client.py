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
    SettingsRead,
    SyncActualData,
)
from shared.utils import async_http_request


async def sync_actual_values(resource: str, current_val: float) -> None:
    """Відправляє HTTP запит до API для оновлення actual_value"""
    url = f"{API_URL}/metrics/sync"
    responce = await async_http_request(
        method="PUT",
        url=url,
        payload=SyncActualData(resource=resource, actual_value=current_val),
        response_model=GenericResponse,  # Ми не очікуємо конкретної структури відповіді, тому використовуємо GenericResponse
    )
    if responce and responce.status == "success":
        await send_system_log(
            f"✅ Actual value synced for {resource}",
            level="INFO",
            service="collector",
        )
    else:
        await send_system_log(
            f"⚠️ Failed to sync actual value for {resource}: {responce.message if responce else 'No response'}",
            level="ERROR",
            service="collector",
        )


async def save_new_prediction(resource: str, val: float, pred: float) -> None:
    """Відправляє HTTP запит до API для збереження нового прогнозу"""
    url = f"{API_URL}/metrics/predict"
    responce = await async_http_request(
        method="POST",
        url=url,
        payload=PredictData(
            resource=resource,
            input_value=val,
            predicted_value=pred,
        ),
        response_model=MetricRead,
    )
    if responce:
        await send_system_log(
            f"✅ New prediction saved for {resource}: {pred:.2f}",
            level="INFO",
            service="collector",
        )
    else:
        await send_system_log(
            f"⚠️ Failed to save new prediction for {resource}",
            level="ERROR",
            service="collector",
        )


async def get_system_settings() -> SettingsRead:
    response = await async_http_request(
        method="GET", url=f"{API_URL}/settings", response_model=SettingsRead
    )
    return response


async def get_recent_history(resource: str, limit: int = 10) -> list[MetricRead]:
    """Отримує останні N записів з бази для конкретного ресурсу."""
    # Переконайся, що API_BASE_URL вказує на твій API сервіс
    resp = await async_http_request(
        method="GET",
        url=f"{API_URL}/metrics/history?resource={resource}&limit={limit}",
        payload=MetricHistoryRead(resource=resource, limit=limit),
        response_model=list[
            MetricRead
        ],  # Очікуємо список словників з полями ts, resource, actual_value
    )
    return resp


async def get_prediction(history: list[dict[str, float]]) -> MetricPoint | None:
    """
    Відправляє вікно з 10 останніх вимірів до LSTM сервісу.
    Очікує отримати комплексний прогноз для всіх метрик.

    :param history: Список з 10 словників, наприклад: [{"cpu": 0.1, "ram": 10.0, "rps": 150.0}, ...]
    :return: Об'єкт MetricPoint з прогнозами або None у разі помилки.
    """

    # 🌟 МАГІЯ ТУТ: Перетворюємо список словників на список Pydantic-моделей
    # Використовуємо розпакування **point, оскільки point - це звичайний словник
    history_points = [MetricPoint.model_validate(point) for point in history]

    response = await async_http_request(
        method="POST",
        url=f"{PREDICTOR_URL}/predict",
        # Передаємо вже підготовлений список об'єктів
        payload=PredictionRequest(history=history_points),
        response_model=PredictionResponse,
    )

    if response:
        return response.predicted_values

    await send_system_log(
        "⚠️ Predictor API Error: No response received",
        level="ERROR",
        service="collector",
    )
    return None
