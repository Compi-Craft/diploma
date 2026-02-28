from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LSTM Predictive Service"
    MODEL_INPUT_STEPS: int = 10  # Скільки точок історії беремо
    MODEL_FEATURES: int = 3      # cpu, ram, rps
    
settings = Settings()