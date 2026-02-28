from fastapi import FastAPI
from api.routes import router
from lstm_module import PORT
from core.config import settings
import uvicorn

app = FastAPI(title=settings.PROJECT_NAME)

# Підключаємо наші маршрути
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
