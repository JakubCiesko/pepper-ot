from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes import dashboard
from app.routes import detect

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
    logger.info("Startup: Initializing Detection Service Translations...")
    await detect.DETECTION_SERVICE.initialize_translations()
    logger.info("Startup: Translations loaded.")
    yield
    logger.info("Shutdown: Cleaning up...")


app = FastAPI(
    title="Pepper Object Detection Server", version="0.1.0", lifespan=lifespan
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


app.include_router(detect.router, prefix="/api")
app.include_router(dashboard.router)

logger.info("Server initialized")
