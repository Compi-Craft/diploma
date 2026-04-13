from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "LSTM Predictive Service"


settings = Settings()
