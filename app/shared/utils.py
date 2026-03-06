import asyncio
import time
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

# Створюємо дженерик-тип для Mypy, щоб він розумів, яку саме модель ми повертаємо
T = TypeVar("T", bound=BaseModel)


async def async_http_request(
    method: str,
    url: str,
    payload: Optional[BaseModel] = None,
    response_model: Any = None,  # 👈 Тепер сюди можна передати list[ModelRead]
    retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 10.0,
) -> T | Any:  # Mypy знатиме: якщо передали response_model, повернеться тип T

    json_data = payload.model_dump(mode="json") if payload else None
    current_delay = base_delay

    async with httpx.AsyncClient() as client:
        for attempt in range(1, retries + 1):
            try:
                response = await client.request(
                    method=method, url=url, json=json_data, timeout=timeout
                )

                response.raise_for_status()

                if response.status_code == 204 or not response.content:
                    return None

                raw_data = response.json()

                # 🌟 МАГІЯ ТУТ: Замість BaseModel.model_validate використовуємо TypeAdapter
                if response_model:
                    adapter = TypeAdapter(response_model)
                    return adapter.validate_python(raw_data)

                return raw_data

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status < 500 and status != 429:
                    print(
                        f"❌ [HTTP {status}] Невиправна помилка для {url}: {e.response.text}"
                    )
                    raise e
                print(
                    f"⚠️ [Спроба {attempt}/{retries}] Помилка сервера {status}. Очікування {current_delay}с..."
                )

            except httpx.RequestError as e:
                print(f"🔌 [Спроба {attempt}/{retries}] Мережева помилка до {url}: {e}")

            if attempt < retries:
                await asyncio.sleep(current_delay)
                current_delay *= 2

        raise Exception(f"🚨 Всі {retries} спроби до {url} завершилися невдачею.")


def sync_http_request(
    method: str,
    url: str,
    payload: Optional[BaseModel] = None,
    params: Optional[dict] = None,  # 👈 ДОДАЛИ підтримку Query-параметрів (GET)
    data: Optional[
        dict
    ] = None,  # 👈 ДОДАЛИ підтримку текстових полів форми (POST multipart)
    files: Optional[dict] = None,  # 👈 ДОДАЛИ підтримку файлів (POST multipart)
    response_model: Any = None,
    retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 10.0,
) -> T | Any:

    json_data = payload.model_dump(mode="json") if payload else None
    current_delay = base_delay

    with httpx.Client() as client:
        for attempt in range(1, retries + 1):
            try:
                # 🌟 ПЕРЕДАЄМО ВСІ ПАРАМЕТРИ В HTTPX
                response = client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,  # 👈 httpx сам збере з цього ?limit=100
                    data=data,  # 👈 httpx сам зробить form-data
                    files=files,  # 👈 httpx сам прикріпить файли
                    timeout=timeout,
                )

                response.raise_for_status()

                if response.status_code == 204 or not response.content:
                    return None

                raw_data = response.json()

                if response_model:
                    adapter = TypeAdapter(response_model)
                    return adapter.validate_python(raw_data)

                return raw_data

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status < 500 and status != 429:
                    print(
                        f"❌ [HTTP {status}] Невиправна помилка для {url}: {e.response.text}"
                    )
                    raise e
                print(
                    f"⚠️ [Спроба {attempt}/{retries}] Помилка сервера {status}. Очікування {current_delay}с..."
                )

            except httpx.RequestError as e:
                print(f"🔌 [Спроба {attempt}/{retries}] Мережева помилка до {url}: {e}")

            if attempt < retries:
                time.sleep(current_delay)
                current_delay *= 2

        raise Exception(f"🚨 Всі {retries} спроби до {url} завершилися невдачею.")
