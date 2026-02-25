import uvicorn
from api.main import app
from api import PORT

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
