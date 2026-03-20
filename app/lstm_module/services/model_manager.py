import datetime
import joblib # 💡 ЗМІНА: Використовуємо joblib для скейлерів, як в нашому Jupyter
import os
import threading
import uuid
import pickle # 💡 ЗМІНА: Використовуємо pickle, як в нашому Jupyter
from typing import Any

import keras
import numpy as np
import pandas as pd # 💡 Додано для зручного Rolling Max у fine_tune
import requests
import tensorflow as tf
from config import API_URL
from core.config import settings
from shared.logger import send_system_log_sync
from sklearn.preprocessing import StandardScaler


class ModelManager:
    def __init__(self) -> None:
        # 💡 ЗМІНА: Створюємо два РІЗНИХ dummy-скейлери
        self.scaler_X = self._create_dummy_scaler_X()
        self.scaler_y = self._create_dummy_scaler_y()
        self.model = self._create_dummy_model()
        self.version = "v0-dummy-sniper"
        self._lock = threading.Lock()
        self.epsilon = 1e-6

    def _create_dummy_scaler_X(self) -> StandardScaler:
        """Скейлер для 3 фічей (Історія)"""
        scaler = StandardScaler()
        scaler.fit([[0, 0, 0], [1, 1, 1]])
        return scaler

    def _create_dummy_scaler_y(self) -> StandardScaler:
        """Скейлер для 1 фічі (Таргет CPU)"""
        scaler = StandardScaler()
        scaler.fit([[0], [1]])
        return scaler

    def _create_dummy_model(self) -> Any:
        """Створює пусту Снайперську модель для старту"""
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(
                96, 
                input_shape=(settings.MODEL_INPUT_STEPS, settings.MODEL_FEATURES),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1), # 💡 ЗМІНА: Тільки 1 вихід (CPU)
        ])
        model.compile(optimizer="adam", loss="mae") # 💡 ЗМІНА: loss="mae"
        return model

    def load_new_model(self, model_path: str, scaler_X_path: str, scaler_y_path: str, version: str) -> None:
        """Гаряча заміна моделі та обох скейлерів (Hot Swap)"""
        try:
            new_model = keras.models.load_model(model_path, compile=False)
            
            # 💡 ЗМІНА: Використовуємо pickle замість joblib
            new_scaler_X = joblib.load(scaler_X_path)
            new_scaler_y = joblib.load(scaler_y_path)

            with self._lock:
                self.model = new_model
                self.scaler_X = new_scaler_X
                self.scaler_y = new_scaler_y
                self.version = version
                
            send_system_log_sync(
                f"✅ Sniper Model and Scalers successfully updated to {version}",
                level="INFO",
                service="lstm_module",
            )
        except Exception as e:
            send_system_log_sync(f"❌ Failed to load model {version}: {e}", level="ERROR", service="lstm_module")

    def predict(self, raw_data: np.ndarray) -> np.ndarray:
        with self._lock:
            # 1. Витягуємо сирі дані: форма (11, 3)
            flat_raw = raw_data[0]
            
            # Запам'ятовуємо останнє абсолютне значення
            last_absolute_values = flat_raw[-1]
            current_cpu = last_absolute_values[0]
            current_ram = last_absolute_values[1]
            current_rps = last_absolute_values[2]

            # 2. Розраховуємо Log Returns: отримуємо (10, 3)
            log_returns = np.log((flat_raw[1:] + self.epsilon) / (flat_raw[:-1] + self.epsilon))

            # 3. Нормалізуємо (Scaler X)
            scaled_X = self.scaler_X.transform(log_returns)
            model_input = np.array([scaled_X])

            # 4. Прогноз (відмасштабований Log Return ТІЛЬКИ для CPU)
            prediction_scaled = self.model.predict(model_input, verbose=0)

            # 5. Розпаковка (Scaler y)
            prediction_log_return = self.scaler_y.inverse_transform(prediction_scaled)[0][0]

            # 6. Математика: Отримуємо майбутній пік процесора
            pred_cpu = current_cpu * np.exp(prediction_log_return)

            # 7. 💡 ЗАГЛУШКА: Формуємо масив з 3 елементів для сумісності з API
            # CPU = Прогноз моделі. RAM та RPS = Поточні значення.
            return np.array([[pred_cpu, current_ram, current_rps]])

    def fine_tune_specific(
        self,
        base_version: str,
        model_path: str,
        scaler_X_path: str,
        scaler_y_path: str,
        raw_data: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> None:
        """Завантажує модель, донавчає її на логарифмах (Rolling Max) та публікує."""
        try:
            target_model = tf.keras.models.load_model(model_path, compile=False)

            target_scaler_X = joblib.load(scaler_X_path)
            target_scaler_y = joblib.load(scaler_y_path)

            # 💡 ЗМІНА: Використовуємо Pandas для простого прорахунку Rolling Max
            df = pd.DataFrame(raw_data, columns=['cpu', 'ram', 'rps'])
            horizon = 4
            lookback = settings.MODEL_INPUT_STEPS # 10
            
            # X: Log Returns
            log_returns_df = pd.DataFrame()
            for col in ['cpu', 'ram', 'rps']:
                log_returns_df[col] = np.log((df[col] + self.epsilon) / (df[col].shift(1) + self.epsilon))
                
            # Y: Rolling Max для CPU
            future_peak = df['cpu'].rolling(window=horizon).max().shift(-horizon)
            target_y_series = np.log((future_peak + self.epsilon) / (df['cpu'] + self.epsilon))

            # Видаляємо NaN
            valid_idx = target_y_series.dropna().index.intersection(log_returns_df.dropna().index)
            log_returns = log_returns_df.loc[valid_idx].values
            target_y = target_y_series.loc[valid_idx].values.reshape(-1, 1)

            # Масштабуємо
            scaled_X = target_scaler_X.transform(log_returns)
            scaled_y = target_scaler_y.transform(target_y)

            # Нарізаємо вікна
            X_train, y_train = [], []
            for i in range(len(scaled_X) - lookback + 1):
                X_train.append(scaled_X[i : i + lookback])
                y_train.append(scaled_y[i + lookback - 1])

            X_train, y_train = np.array(X_train), np.array(y_train)

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            target_model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])

            history = target_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            final_mae = float(history.history["mae"][-1])

            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
            short_hash = uuid.uuid4().hex[:4]
            new_version = f"{base_version[:10]}_tuned_{timestamp}-{short_hash}"

            base_dir = os.path.dirname(model_path)
            new_model_path = os.path.join(base_dir, f"{new_version}.keras")
            target_model.save(new_model_path)

            send_system_log_sync(f"✅ Fine-tuning (Sniper) завершено. {new_version}", level="INFO", service="lstm_module")

            self._sync_publish_new_model(new_version, final_mae, final_mae, new_model_path, scaler_X_path, scaler_y_path)

        except Exception as e:
            send_system_log_sync(f"❌ Помилка під час fine-tuning: {e}", level="ERROR", service="lstm_module")

    def _sync_publish_new_model(
        self, version: str, mse: float, mae: float, model_path: str, scaler_X_path: str, scaler_y_path: str
    ) -> None:
        payload = {
            "version": version,
            "mse": mse,
            "mae": mae,
            "model_path": model_path,
            "scaler_x_path": scaler_X_path,
            "scaler_y_path": scaler_y_path,
            "is_active": False,
        }
        try:
            resp = requests.post(f"{API_URL}/models", json=payload)
            if resp.status_code == 200:
                send_system_log_sync(f"📡 Модель {version} опублікована!", level="INFO", service="lstm_module")
            else:
                send_system_log_sync(f"⚠️ Помилка публікації: {resp.text}", level="ERROR", service="lstm_module")
        except Exception as e:
            pass

model_manager = ModelManager()
