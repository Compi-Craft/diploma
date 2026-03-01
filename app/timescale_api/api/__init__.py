import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:lpa_password@timescaledb:5432/lpa_database",
)

PORT = os.getenv("PORT", "5000")

PREDICTOR_URL = os.getenv("PREDICTOR_URL", "http://lstm-predictor:6000")

MODELS_DIR = os.getenv("MODELS_DIR", "/app/lstm_module/ml_models")
SCALERS_DIR = os.getenv("SCALERS_DIR", "/app/lstm_module/scalers")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SCALERS_DIR, exist_ok=True)
