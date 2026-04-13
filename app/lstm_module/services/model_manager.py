import datetime
import os
import threading
import uuid
from typing import Any

import joblib
import keras
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tensorflow.keras.backend as K
from config import API_URL, MODEL_FEATURES, MODEL_INPUT_STEPS
from shared.logger import send_system_log_sync
from sklearn.preprocessing import StandardScaler


def _quantile_loss(q: float) -> tf.keras.losses.Loss:
    """Asymmetric quantile loss to match training notebook (q=0.75)."""

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e))

    loss.__name__ = f"quantile_loss_q{int(q * 100)}"
    return loss


class ModelManager:
    def __init__(self) -> None:
        self.scaler_X = self._create_dummy_scaler_X()
        self.model = self._create_dummy_model()
        self.version = "v0-dummy-sniper"
        self._lock = threading.Lock()
        self.epsilon = 1e-6

    def _create_dummy_scaler_X(self) -> StandardScaler:
        """Скейлер для 4 фічей: pct_change(cpu_util), pct_change(rps), cpu_level, rps_level."""
        scaler = StandardScaler()
        scaler.fit([[0, 0, 0, 0], [1, 1, 1, 1]])
        return scaler

    def _create_dummy_model(self) -> Any:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.GRU(
                    64,
                    input_shape=(MODEL_INPUT_STEPS, MODEL_FEATURES),
                ),
                tf.keras.layers.Dropout(0.35),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="mae")
        return model

    def load_new_model(
        self,
        model_path: str,
        scaler_X_path: str,
        scaler_y_path: str,  # збережено для сумісності API, не використовується
        version: str,
    ) -> None:
        """Гаряча заміна моделі та scaler_X (Hot Swap). scaler_y більше не потрібен."""
        try:
            new_model = keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"loss": _quantile_loss(0.75)},
            )
            new_scaler_X = joblib.load(scaler_X_path)

            with self._lock:
                self.model = new_model
                self.scaler_X = new_scaler_X
                self.version = version

            send_system_log_sync(
                f"✅ Model and Scaler_X updated to {version}",
                level="INFO",
                service="lstm_module",
            )
        except Exception as e:
            send_system_log_sync(
                f"❌ Failed to load model {version}: {e}",
                level="ERROR",
                service="lstm_module",
            )

    def predict(self, raw_data: np.ndarray) -> float:
        """
        Прогнозує майбутній пік CPU utilization.

        Args:
            raw_data: shape (1, MODEL_INPUT_STEPS+1, 2) — сирі точки [cpu_util, rps]
                      cpu_util вже ∈ [0, 1] (з Prometheus utilization запиту)

        Returns:
            float — прогноз CPU utilization ∈ [0, 1]
        """
        with self._lock:
            flat_raw = raw_data[0]  # (MODEL_INPUT_STEPS+1, 2)

            cpu_util = flat_raw[:, 0]  # CPU utilization [0, 1]
            rps = flat_raw[:, 1]

            # Velocity features: pct_change — shape (MODEL_INPUT_STEPS,)
            pct_cpu = (cpu_util[1:] - cpu_util[:-1]) / (cpu_util[:-1] + self.epsilon)
            pct_rps = (rps[1:] - rps[:-1]) / (rps[:-1] + self.epsilon)

            # Level features: поточна utilization + log-normalized RPS — shape (MODEL_INPUT_STEPS,)
            cpu_level = cpu_util[1:]
            rps_level = np.log1p(rps[1:])  # log(rps+1): bounded, OOD-safe, 0 at idle

            # X: shape (MODEL_INPUT_STEPS, 4)
            X = np.column_stack([pct_cpu, pct_rps, cpu_level, rps_level])
            scaled_X = self.scaler_X.transform(X)
            model_input = np.array([scaled_X])  # (1, MODEL_INPUT_STEPS, 4)

            # Sigmoid output → utilization ∈ (0, 1), scaler_y не потрібен
            prediction = self.model.predict(model_input, verbose=0)
            return float(np.clip(prediction[0][0], 0.0, 1.0))

    def fine_tune_specific(
        self,
        base_version: str,
        model_path: str,
        scaler_X_path: str,
        scaler_y_path: str,  # збережено для сумісності, не використовується
        raw_data: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> None:
        """
        Донавчає модель на нових даних. raw_data shape: (N, 2) — [cpu_util, rps].
        Використовує ту саму feature engineering, що й training notebook.
        """
        try:
            target_model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"loss": _quantile_loss(0.75)},
            )
            target_scaler_X = joblib.load(scaler_X_path)

            df = pd.DataFrame(raw_data, columns=["cpu", "rps"])
            horizon = 12  # MODEL_FORECAST_HORIZON = 12 кроків = 60s
            lookback = MODEL_INPUT_STEPS  # 8 кроків = 40s

            # Feature engineering: відповідає training notebook (4 фічі)
            feat_df = pd.DataFrame()
            feat_df["pct_cpu"] = (df["cpu"] - df["cpu"].shift(1)) / (
                df["cpu"].shift(1) + self.epsilon
            )
            feat_df["pct_rps"] = (df["rps"] - df["rps"].shift(1)) / (
                df["rps"].shift(1) + self.epsilon
            )
            feat_df["cpu_level"] = df["cpu"].values
            feat_df["rps_level"] = np.log1p(df["rps"].values)  # 4-та фіча: log(rps+1)

            # Таргет: point prediction cpu[t+horizon] (не rolling max — застарілий підхід)
            target_y_series = df["cpu"].shift(-horizon)

            valid_idx = target_y_series.dropna().index.intersection(
                feat_df.dropna().index
            )
            X_raw = feat_df.loc[valid_idx].values  # (N, 4)
            y_raw = target_y_series.loc[valid_idx].values  # (N,)

            scaled_X = target_scaler_X.transform(X_raw)

            X_list, y_list = [], []
            for i in range(len(scaled_X) - lookback + 1):
                X_list.append(scaled_X[i : i + lookback])
                y_list.append(y_raw[i + lookback - 1])

            X_train = np.array(X_list)
            y_train = np.array(y_list).reshape(-1, 1)

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
            target_model.compile(optimizer=optimizer, loss=_quantile_loss(0.75))

            history = target_model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1
            )

            final_loss = float(history.history["loss"][-1])
            y_pred = target_model.predict(X_train, verbose=0).flatten()
            final_mae = float(np.mean(np.abs(y_train.flatten() - y_pred)))

            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%d-%H%M%S"
            )
            short_hash = uuid.uuid4().hex[:4]
            new_version = f"{base_version[:10]}_tuned_{timestamp}-{short_hash}"

            base_dir = os.path.dirname(model_path)
            new_model_path = os.path.join(base_dir, f"{new_version}.keras")
            target_model.save(new_model_path)

            send_system_log_sync(
                f"✅ Fine-tuning завершено. {new_version}",
                level="INFO",
                service="lstm_module",
            )

            self._sync_publish_new_model(
                new_version, final_loss, final_mae, new_model_path, scaler_X_path
            )

        except Exception as e:
            send_system_log_sync(
                f"❌ Помилка під час fine-tuning: {e}",
                level="ERROR",
                service="lstm_module",
            )

    def _sync_publish_new_model(
        self,
        version: str,
        mse: float,
        mae: float,
        model_path: str,
        scaler_X_path: str,
    ) -> None:
        payload = {
            "version": version,
            "mse": mse,
            "mae": mae,
            "model_path": model_path,
            "scaler_x_path": scaler_X_path,
            "scaler_y_path": "",  # не використовується в новій моделі
            "is_active": False,
        }
        try:
            resp = requests.post(f"{API_URL}/models", json=payload)
            if resp.status_code == 200:
                send_system_log_sync(
                    f"📡 Модель {version} опублікована!",
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
                f"⚠️ Помилка публікації моделі: {e}",
                level="ERROR",
                service="lstm_module",
            )


model_manager = ModelManager()
