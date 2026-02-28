import os
import threading
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from core.config import settings

class ModelManager:
    def __init__(self):
        # Ініціалізуємо "пустишки"
        self.scaler = self._create_dummy_scaler()
        self.model = self._create_dummy_model()
        self.version = "v0-dummy"
        self._lock = threading.Lock()

    def _create_dummy_scaler(self):
        """Створює пустий скейлер, щоб API не падало до завантаження реальної моделі"""
        scaler = StandardScaler()
        # Фіктивно "навчаємо" його на нулях та одиницях, щоб він просто пропускав дані
        scaler.fit([[0, 0, 0], [1, 1, 1]])
        return scaler

    def _create_dummy_model(self):
        """Створює пусту LSTM модель для старту"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(16, input_shape=(settings.MODEL_INPUT_STEPS, settings.MODEL_FEATURES)),
            tf.keras.layers.Dense(settings.MODEL_FEATURES)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def load_new_model(self, model_path: str, scaler_path: str, version: str):
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
            print(f"✅ Model and Scaler successfully updated to {version}")
        except Exception as e:
            print(f"❌ Failed to load model {version}: {e}")

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
            prediction_real = self.scaler.inverse_transform(prediction_scaled)
            
            return prediction_real

# Створюємо єдиний екземпляр
model_manager = ModelManager()
