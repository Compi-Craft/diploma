import os

API_URL: str = os.getenv("API_URL", "http://timescale_api:5000")
PREDICTOR_URL: str = os.getenv("PREDICTOR_URL", "http://lstm-predictor:6000")

MODELS_DIR: str = os.getenv("MODELS_DIR", "/app/lstm_module/ml_models")
SCALERS_DIR: str = os.getenv("SCALERS_DIR", "/app/lstm_module/scalers")


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
