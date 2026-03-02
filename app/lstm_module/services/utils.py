import datetime
from typing import Dict

import aiohttp
import numpy as np
from logger.logger import send_system_log
from lstm_module import API_URL
from models.schemas import RetrainCommand
from services.model_manager import model_manager


async def run_finetune_pipeline(cmd: RetrainCommand) -> None:
    """Фонова задача: збирає дані, шляхи і запускає навчання."""
    try:
        async with aiohttp.ClientSession() as session:
            # 1. Дістаємо шляхи до моделі
            async with session.get(
                f"{API_URL}/models/byversion/{cmd.target_version}"
            ) as resp:
                if resp.status != 200:
                    await send_system_log(
                        f"❌ Не вдалося знайти модель {cmd.target_version}",
                        level="ERROR",
                        service="lstm_module",
                    )
                    return
                model_meta = await resp.json()

            # 2. Дістаємо історичні дані за період
            url = f"{API_URL}/metrics/history/range?start_time={cmd.start_time}&end_time={cmd.end_time}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    await send_system_log(
                        f"❌ Не вдалося завантажити дані для навчання",
                        level="ERROR",
                        service="lstm_module",
                    )
                    return
                history_data = await resp.json()

        history_data.sort(key=lambda x: x["ts"])

        raw_values = []
        current_bucket: Dict[str, float] = {}
        last_ts = None

        # 2. Розумне групування за "вікнами" (Proximity Grouping)
        for item in history_data:
            # Парсимо час (замінюємо Z на +00:00 для сумісності з fromisoformat)
            ts_str = item["ts"].replace("Z", "+00:00")
            current_ts = datetime.datetime.fromisoformat(ts_str)

            res = item["resource"]
            val = item["actual_value"]

            # Якщо це перша точка АБО з моменту минулої пройшло більше 2 секунд -> це новий цикл збору
            if last_ts is None or (current_ts - last_ts).total_seconds() > 2.0:
                # Зберігаємо попередній бакет, якщо в ньому є всі 3 метрики
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

                # Очищаємо бакет для нового циклу
                current_bucket = {}

            # Додаємо метрику в поточний бакет
            current_bucket[res] = val
            last_ts = current_ts

        # 3. Не забуваємо перевірити останній зібраний бакет після виходу з циклу
        if (
            "cpu" in current_bucket
            and "ram" in current_bucket
            and "rps" in current_bucket
        ):
            raw_values.append(
                [current_bucket["cpu"], current_bucket["ram"], current_bucket["rps"]]
            )

        raw_array = np.array(raw_values)

        if len(raw_array) < 50:
            await send_system_log(
                f"❌ Замало повних даних для донавчання: {len(raw_array)} точок.",
                level="ERROR",
                service="lstm_module",
            )
            return

        # 4. Передаємо все в ModelManager
        await send_system_log(
            f"🚀 Запуск Fine-Tuning для {cmd.target_version} на {len(raw_values)} точках...",
            level="INFO",
            service="lstm_module",
        )
        model_manager.fine_tune_specific(
            base_version=cmd.target_version,
            model_path=model_meta["model_path"],
            scaler_path=model_meta["scaler_path"],
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
