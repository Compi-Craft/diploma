import datetime
import os
import threading
import uuid
from typing import Any

import joblib
import numpy as np
import requests
import tensorflow as tf
from config import API_URL
from core.config import settings
from shared.logger import send_system_log_sync
from sklearn.preprocessing import StandardScaler


class ModelManager:
    def __init__(self) -> None:
        # Ініціалізуємо "пустишки"
        self.scaler = self._create_dummy_scaler()
        self.model = self._create_dummy_model()
        self.version = "v0-dummy"
        self._lock = threading.Lock()

    def _create_dummy_scaler(self) -> StandardScaler:
        """Створює пустий скейлер, щоб API не падало до завантаження реальної моделі"""
        scaler = StandardScaler()
        # Фіктивно "навчаємо" його на нулях та одиницях, щоб він просто пропускав дані
        scaler.fit([[0, 0, 0], [1, 1, 1]])
        return scaler

    def _create_dummy_model(self) -> Any:
        """Створює пусту LSTM модель для старту"""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(
                    16,
                    input_shape=(settings.MODEL_INPUT_STEPS, settings.MODEL_FEATURES),
                ),
                tf.keras.layers.Dense(settings.MODEL_FEATURES),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def load_new_model(self, model_path: str, scaler_path: str, version: str) -> None:
        """Гаряча заміна моделі та скейлера (Hot Swap)"""
        try:
            # ДОДАЛИ compile=False ось тут 👇
            new_model = tf.keras.models.load_model(model_path, compile=False)
            new_scaler = joblib.load(scaler_path)

            # Атомарна заміна під локом
            with self._lock:
                self.model = new_model
                self.scaler = new_scaler
                self.version = version
            send_system_log_sync(
                f"✅ Model and Scaler successfully updated to {version}",
                level="INFO",
                service="lstm_module",
            )
        except Exception as e:
            send_system_log_sync(
                f"❌ Failed to load model {version}: {e}",
                level="ERROR",
                service="lstm_module",
            )

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        data має розмірність (1, 10, 3) - один батч, 10 кроків, 3 фічі.
        Scaler від scikit-learn вміє працювати тільки з 2D масивами.
        Тому ми маємо змінити форму, відмасштабувати, і повернути назад.
        """
        with self._lock:
            # 1. Витягуємо наші 10 точок у 2D масив: форма стає (10, 3)
            flat_data = data[0]

            # 2. Нормалізуємо вхідні дані
            scaled_flat_data = self.scaler.transform(flat_data)

            # 3. Повертаємо у 3D форму для LSTM: (1, 10, 3)
            scaled_data = np.array([scaled_flat_data])

            # 4. Робимо прогноз (результат буде мати форму (1, 3))
            prediction_scaled = self.model.predict(scaled_data, verbose=0)

            # 5. Робимо зворотне перетворення (Inverse Transform) у реальні значення
            prediction_real: np.ndarray = np.array(
                self.scaler.inverse_transform(prediction_scaled)
            )

            return prediction_real

    def fine_tune_specific(
        self,
        base_version: str,
        model_path: str,
        scaler_path: str,
        raw_data: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> None:
        """Завантажує конкретну модель, донавчає її та публікує нову версію."""
        try:
            # 1. Завантажуємо модель і скейлер з диска (не чіпаючи активну в пам'яті!)
            target_model = tf.keras.models.load_model(model_path, compile=False)
            target_scaler = joblib.load(scaler_path)

            # 2. Підготовка даних (Data Prep)
            scaled_data = target_scaler.transform(raw_data)

            X_train, y_train = [], []
            lookback = 10  # Твій history_buffer size

            for i in range(len(scaled_data) - lookback):
                X_train.append(scaled_data[i : i + lookback])
                y_train.append(scaled_data[i + lookback])

            X_train, y_train = np.array(X_train), np.array(y_train)  # type: ignore[assignment]

            # 3. Компілюємо з низьким learning rate для fine-tuning
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            target_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

            # 4. Навчаємо
            history = target_model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1
            )

            final_mse = float(history.history["loss"][-1])
            final_mae = float(history.history["mae"][-1])

            # 5. Генеруємо ім'я та зберігаємо
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%d-%H%M%S"
            )
            short_hash = uuid.uuid4().hex[:4]
            # Вказуємо, від якої моделі вона пішла (напр. v1.0_tuned_v2026...)
            new_version = f"{base_version[:10]}_tuned_{timestamp}-{short_hash}"

            base_dir = os.path.dirname(model_path)
            new_model_path = os.path.join(base_dir, f"{new_version}.h5")
            target_model.save(new_model_path)

            send_system_log_sync(
                f"✅ Fine-tuning завершено. Збережено як {new_version}",
                level="INFO",
                service="lstm_module",
            )

            # 6. Відправляємо в БД
            self._sync_publish_new_model(
                new_version, final_mse, final_mae, new_model_path, scaler_path
            )

        except Exception as e:
            send_system_log_sync(
                f"❌ Помилка під час fine-tuning моделі {base_version}: {e}",
                level="ERROR",
                service="lstm_module",
            )

    def _sync_publish_new_model(
        self, version: str, mse: float, mae: float, model_path: str, scaler_path: str
    ) -> None:
        """Оскільки ми вже в окремому потоці (завдяки BackgroundTasks), можемо використовувати синхронний запит або asyncio.run"""
        payload = {
            "version": version,
            "mse": mse,
            "mae": mae,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "is_active": False,
        }
        try:
            # Зміни URL на адресу твого API
            resp = requests.post(f"{API_URL}/models", json=payload)
            if resp.status_code == 200:
                send_system_log_sync(
                    f"📡 Модель {version} успішно опублікована в БД!",
                    level="INFO",
                    service="lstm_module",
                )
            else:
                send_system_log_sync(
                    f"⚠️ Помилка публікації: {resp.text}",
                    level="ERROR",
                    service="lstm_module",
                )
        except Exception as e:
            send_system_log_sync(
                f"❌ Не вдалося достукатись до API для публікації: {e}",
                level="ERROR",
                service="lstm_module",
            )


# Створюємо єдиний екземпляр
model_manager = ModelManager()
