from contextlib import asynccontextmanager
import logging
import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings
from pyngrok import ngrok

from app.api.v1 import router as api_v1_router
from app.core.runtime.state import app_state
from app.dashboard import router as dashboard_router

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)

# to unite logging style
for uv_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(uv_logger).handlers = logging.getLogger().handlers
    logging.getLogger(uv_logger).setLevel(logging.INFO)

logger = logging.getLogger(__name__)


class ServerSettings(BaseSettings):
    BASE_URL: str = "http://localhost:8000"
    USE_NGROK: bool = os.environ.get("USE_NGROK", "False") == "True"


SERVER_SETTINGS = ServerSettings()

logger.info(
    f"Initializing FastAPI server with settings: {SERVER_SETTINGS.model_dump()}"
)


if SERVER_SETTINGS.USE_NGROK:
    port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "8000"
    public_url = ngrok.connect(port).public_url
    logger.info(f'ngrok tunnel "{public_url}" -> "http://127.0.0.1:{port}"')
    SERVER_SETTINGS.BASE_URL = public_url


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Initializing AppState...")
    await app_state.initialize("./config.yaml")
    logger.info("Startup: AppState ready.")
    yield
    logger.info("Shutdown: Cleaning up...")
    if app_state.worker_manager is not None:
        await app_state.worker_manager.close()
    if SERVER_SETTINGS.USE_NGROK:
        ngrok.kill()
        logger.info("Shutdown: Killed NGrok Tunnels")


app = FastAPI(
    title="Pepper Object Detection Server", version="0.1.0", lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(dashboard_router)

logger.info("Server initialized")
