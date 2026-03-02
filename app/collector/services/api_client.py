from typing import Any
from logger.logger import send_system_log
import httpx
import aiohttp

from ..config import settings


async def sync_actual_values(resource: str, current_val: float) -> None:
    """Відправляє HTTP запит до API для оновлення actual_value"""
    url = f"{settings.API_URL}/metrics/sync"
    payload = {"resource": resource, "actual_value": current_val}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.put(url, json=payload, timeout=5.0)
            response.raise_for_status()
    except Exception as e:
        await send_system_log(f"⚠️ Помилка синхронізації з API: {e}", level="ERROR", service="collector")


async def save_new_prediction(resource: str, val: float, pred: float) -> None:
    """Відправляє HTTP запит до API для збереження нового прогнозу"""
    url = f"{settings.API_URL}/metrics/predict"
    payload = {"resource": resource, "input_value": val, "predicted_value": pred}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=5.0)
            response.raise_for_status()
    except Exception as e:
        await send_system_log(f"⚠️ Помилка збереження прогнозу в API: {e}", level="ERROR", service="collector")


async def get_system_settings() -> Any:
    async with httpx.AsyncClient() as client:
        # Підстав свій URL до FastAPI
        response = await client.get(f"{settings.API_URL}/settings", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        raise Exception(f"HTTP {response.status_code}")

async def get_recent_history(resource: str, limit: int = 10) -> list:
    """Отримує останні N записів з бази для конкретного ресурсу."""
    # Переконайся, що API_BASE_URL вказує на твій API сервіс
    url = f"{settings.API_URL}/metrics/history?resource={resource}&limit={limit}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception as e:
        print(f"⚠️ Не вдалося завантажити історію для {resource}: {e}")
    return []
