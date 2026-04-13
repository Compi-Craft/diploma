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
            response_model=list[MetricRead],
        )
        if history_data is None:
            await send_system_log(
                "❌ Не вдалося завантажити дані для донавчання",
                level="ERROR",
                service="lstm_module",
            )
            return
        raw_array = prepare_finetune_data(history_data)
        if len(raw_array) < 50:
            await send_system_log(
                f"❌ Замало даних для донавчання: {len(raw_array)} точок.",
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
            scaler_X_path=model_meta.scaler_x_path,
            scaler_y_path=model_meta.scaler_y_path,
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


def prepare_finetune_data(
    history_data: list[MetricRead], interval_sec: int = 5
) -> np.ndarray:
    """
    Конвертує список MetricRead з БД у numpy масив (N, 2) для fine-tuning.
    Використовує input_cpu (CPU на момент ts) і input_rps — обидва з одного моменту часу.
    actual_cpu НЕ використовується тут: таргет будується в fine_tune_specific через shift(-12).
    Виконує лінійну інтерполяцію для пропущених інтервалів.
    """
    history_data.sort(key=lambda x: x.ts)

    raw_values = []
    last_ts = None

    for item in history_data:
        current_ts = item.ts
        if item.input_cpu is None or item.input_rps is None:
            continue

        current_record = [item.input_cpu, item.input_rps]

        # Інтерполяція пропущених кроків
        if last_ts is not None and len(raw_values) > 0:
            gap_seconds = (current_ts - last_ts).total_seconds()
            missing_steps = int(round(gap_seconds / interval_sec)) - 1

            if 0 < missing_steps <= 10:
                last_good = raw_values[-1]
                total_segments = missing_steps + 1
                for step in range(1, missing_steps + 1):
                    interpolated = [
                        last_good[i]
                        + (current_record[i] - last_good[i]) * step / total_segments
                        for i in range(2)
                    ]
                    raw_values.append(interpolated)

        raw_values.append(current_record)
        last_ts = current_ts

    return np.array(raw_values) if raw_values else np.array([]).reshape(0, 2)
