import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:lpa_password@timescaledb:5432/lpa_database",
)

PORT = os.getenv("PORT", "5000")
