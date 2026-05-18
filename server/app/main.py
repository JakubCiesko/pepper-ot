from contextlib import asynccontextmanager
import logging
import os

import colorlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings
from pyngrok import ngrok
import uvicorn

from app.api.v1 import router as api_v1_router
from app.core.runtime.state import app_state
from app.dashboard import router as dashboard_router

APPLICATION_PORT = 8000


class ServerSettings(BaseSettings):
    """
    Environment-backed process settings for server URL and optional ngrok tunneling.

    Attributes:
        BASE_URL: Public base URL advertised by the server.
        USE_NGROK: Whether startup should open an ngrok tunnel for APPLICATION_PORT.
    """

    BASE_URL: str = "http://localhost:8000"
    USE_NGROK: bool = os.environ.get("USE_NGROK", "False") == "True"


SERVER_SETTINGS = ServerSettings()


def setup_logging():
    """
    Configure process logging.

    Returns:
        Logger for this module after the root logger and uvicorn loggers are
        configured to use the same colored stream handler.
    """

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s%(reset)s: %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        uv_logger.handlers = root_logger.handlers
        uv_logger.propagate = False

    return logging.getLogger(__name__)


logger = setup_logging()


logger.info(
    f"Initializing FastAPI server with settings: {SERVER_SETTINGS.model_dump()}"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI startup and shutdown resources.

    Args:
        app: FastAPI application instance passed by the lifespan protocol.

    Yields:
        Control to FastAPI while the initialized AppState is active.
    """

    logger.info("Startup: Initializing AppState...")
    await app_state.initialize("./config.yaml")
    logger.info("Startup: AppState ready.")
    if SERVER_SETTINGS.USE_NGROK:
        logger.info("Startup: Setting up NGrok tunneling")
        ngrok.set_auth_token(os.environ.get("NGROK_AUTH_TOKEN", ""))
        tunnel = ngrok.connect(APPLICATION_PORT)
        public_url = tunnel.public_url
        logger.info(
            f'Startup: ngrok tunnel "{public_url}" -> "http://127.0.0.1:{APPLICATION_PORT}"'
        )
        SERVER_SETTINGS.BASE_URL = public_url
    yield
    logger.info("Shutdown: Cleaning up...")
    if app_state.worker_manager is not None:
        logger.info("Shutdown: Closing WorkerManager")
        await app_state.worker_manager.close()
    if SERVER_SETTINGS.USE_NGROK:
        logger.info("Shutdown: Closing NGrok Tunneling")
        ngrok.kill()
        logger.info("Shutdown: Killed NGrok Tunnels")


logger.info("Initializing FastAPI application")
app = FastAPI(
    title="Pepper Object Detection Server", version="1.0.0", lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(dashboard_router)

logger.info("FastAPI server initialized")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=APPLICATION_PORT, reload=True)
