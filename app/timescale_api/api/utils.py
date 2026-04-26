import datetime
import uuid

from config import PREDICTOR_URL
from shared.logger import send_system_log
from shared.schemas import GenericResponse, ReloadRequest
from shared.utils import async_http_request


async def notify_predictor_to_reload(
    version: str,
    model_path: str,
    scaler_x_path: str,
    window_size: int,
    forecast_horizon: int,
) -> None:
    """Background task: sends a reload signal to the predictor service."""
    try:
        request_object = ReloadRequest(
            version=version,
            model_path=model_path,
            scaler_x_path=scaler_x_path,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
        )
        response = await async_http_request(
            method="POST",
            url=f"{PREDICTOR_URL}/reload",
            payload=request_object,
            response_model=GenericResponse,
        )
        if response:
            await send_system_log(
                f"Predictor hot-swapped to {version}",
                level="INFO",
                service="timescale_api",
            )
        else:
            await send_system_log(
                f"Predictor returned an error on hot-swap to {version}",
                level="ERROR",
                service="timescale_api",
            )
    except Exception as e:
        await send_system_log(
            f"Failed to reach predictor: {e}",
            level="ERROR",
            service="timescale_api",
        )


def generate_model_version() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_hash = uuid.uuid4().hex[:4]
    return f"v{timestamp}-{short_hash}"
