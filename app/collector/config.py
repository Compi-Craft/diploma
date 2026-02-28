import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Урли для зв'язку між сервісами
    API_URL: str = os.getenv("API_URL", "http://timescale_api:5000") 
    PREDICTOR_URL: str = os.getenv("PREDICTOR_URL", "http://lstm-predictor:6000")
    
settings = Settings()
