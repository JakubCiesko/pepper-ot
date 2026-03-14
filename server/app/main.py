from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router as api_v1_router
from app.core.runtime.state import ml_state
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
logger.info("Initializing FastAPI server...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: Initializing MLState...")
    await ml_state.initialize("./config.yaml")
    logger.info("Startup: MLState ready.")
    yield
    logger.info("Shutdown: Cleaning up...")
    if ml_state.worker_manager is not None:
        await ml_state.worker_manager.close()


app = FastAPI(
    title="Pepper Object Detection Server", version="0.1.0", lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(dashboard_router)

logger.info("Server initialized")
