from typing import Dict

import numpy as np
from config import API_URL
from services.model_manager import model_manager
from shared.logger import send_system_log
from shared.schemas import MetricHistoryRangeRead, MetricRead, ModelRead, RetrainCommand
from shared.utils import async_http_request


async def run_finetune_pipeline(cmd: RetrainCommand) -> None:
    """Фонова задача: збирає дані, шляхи і запускає навчання."""
    try:
        model_meta = await async_http_request(
            method="GET",
            url=f"{API_URL}/models/byversion/{cmd.target_version}",
            response_model=ModelRead,
        )
        if model_meta is None:
            await send_system_log(
                f"❌ Не вдалося знайти модель {cmd.target_version} для донавчання",
                level="ERROR",
                service="lstm_module",
            )
            return
        url = f"{API_URL}/metrics/history/range"
        payload = MetricHistoryRangeRead(
            start_time=cmd.start_time,
            end_time=cmd.end_time,
        )
        history_data = await async_http_request(
            method="GET",
            url=url,
            payload=payload,
            response_model=list[MetricRead],  # Ми очікуємо список словників з полями
        )
        if history_data is None:
            await send_system_log(
                f"❌ Не вдалося завантажити дані для донавчання",
                level="ERROR",
                service="lstm_module",
            )
            return
        raw_array = await prepare_finetune_data(history_data)
        if len(raw_array) < 50:
            await send_system_log(
                f"❌ Замало повних даних для донавчання: {len(raw_array)} точок.",
                level="ERROR",
                service="lstm_module",
            )
            return
        await send_system_log(
            f"🚀 Запуск Fine-Tuning для {cmd.target_version} на {len(raw_array)} точках...",
            level="INFO",
            service="lstm_module",
        )
        model_manager.fine_tune_specific(
            base_version=cmd.target_version,
            model_path=model_meta.model_path,
            scaler_path=model_meta.scaler_path,
            raw_data=raw_array,
            epochs=cmd.epochs,
            batch_size=cmd.batch_size,
        )
    except Exception as e:
        await send_system_log(
            f"❌ Помилка у пайплайні донавчання: {e}",
            level="ERROR",
            service="lstm_module",
        )


async def prepare_finetune_data(history_data: list[MetricRead]) -> np.ndarray:
    history_data.sort(key=lambda x: x.ts)  # Сортуємо за часом

    raw_values = []
    current_bucket: Dict[str, float] = {}
    last_ts = None

    for item in history_data:
        # 🌟 МАГІЯ ТУТ: Просто беремо готовий datetime об'єкт!
        current_ts = item.ts

        res = item.resource
        val = item.actual_value

        # Якщо це перша точка АБО з моменту минулої пройшло більше 2 секунд
        if last_ts is None or (current_ts - last_ts).total_seconds() > 2.0:
            if (
                "cpu" in current_bucket
                and "ram" in current_bucket
                and "rps" in current_bucket
            ):
                raw_values.append(
                    [
                        current_bucket["cpu"],
                        current_bucket["ram"],
                        current_bucket["rps"],
                    ]
                )
            current_bucket = {}

        if val is not None:
            current_bucket[res] = val

        last_ts = current_ts

    if "cpu" in current_bucket and "ram" in current_bucket and "rps" in current_bucket:
        raw_values.append(
            [current_bucket["cpu"], current_bucket["ram"], current_bucket["rps"]]
        )

    return np.array(raw_values)
