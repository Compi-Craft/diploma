import asyncio
import time
from typing import Any, Optional, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

T = TypeVar("T", bound=BaseModel)

# Module-level client — reused across all calls, avoids per-call TCP handshake
_async_client = httpx.AsyncClient()


async def async_http_request(
    method: str,
    url: str,
    payload: Optional[BaseModel] = None,
    response_model: Any = None,
    retries: int = 3,
    base_delay: float = 1.0,
    timeout: float = 10.0,
) -> T | Any:
    json_data = payload.model_dump(mode="json") if payload else None
    current_delay = base_delay

    for attempt in range(1, retries + 1):
        try:
            response = await _async_client.request(
                method=method, url=url, json=json_data, timeout=timeout
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
                    f"[HTTP {status}] Non-retryable error for {url}: {e.response.text}"
                )
                raise e
            print(
                f"[Attempt {attempt}/{retries}] Server error {status}. Retrying in {current_delay}s..."
            )

        except httpx.RequestError as e:
            print(f"[Attempt {attempt}/{retries}] Network error for {url}: {e}")

        if attempt < retries:
            await asyncio.sleep(current_delay)
            current_delay *= 2

    raise Exception(f"All {retries} attempts to {url} failed.")


def sync_http_request(
    method: str,
    url: str,
    payload: Optional[BaseModel] = None,
    params: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[dict] = None,
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
                response = client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    data=data,
                    files=files,
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
                        f"[HTTP {status}] Non-retryable error for {url}: {e.response.text}"
                    )
                    raise e
                print(
                    f"[Attempt {attempt}/{retries}] Server error {status}. Retrying in {current_delay}s..."
                )

            except httpx.RequestError as e:
                print(f"[Attempt {attempt}/{retries}] Network error for {url}: {e}")

            if attempt < retries:
                time.sleep(current_delay)
                current_delay *= 2

        raise Exception(f"All {retries} attempts to {url} failed.")
