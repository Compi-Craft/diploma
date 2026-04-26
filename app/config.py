import os

API_URL: str = os.getenv("API_URL", "http://timescale_api:5000")
PREDICTOR_URL: str = os.getenv("PREDICTOR_URL", "http://predictor:6000")

MODELS_DIR: str = os.getenv("MODELS_DIR", "/app/predictor/ml_models")
SCALERS_DIR: str = os.getenv("SCALERS_DIR", "/app/predictor/scalers")

MODEL_INPUT_STEPS: int = int(os.getenv("MODEL_INPUT_STEPS", "10"))
MODEL_FEATURES: int = int(os.getenv("MODEL_FEATURES", "4"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
