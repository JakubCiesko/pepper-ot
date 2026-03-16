from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI

from app.worker.routes import build_worker_router
from app.worker.runtime import WorkerRuntime

logger = logging.getLogger(__name__)
runtime = WorkerRuntime()
# set the logging of worker the same as the top level server
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)

# Keep worker logging consistent with server process logging style.
for uv_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(uv_logger).handlers = logging.getLogger().handlers
    logging.getLogger(uv_logger).setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Worker process startup")
    config_path = os.environ.get("PEPPER_WORKER_CONFIG_PATH")
    if config_path:
        logger.info(f"Worker boot with config path hint: {config_path}")
    yield
    logger.info("Worker process shutdown")


app = FastAPI(title="Pepper Worker Internal API", version="0.1.0", lifespan=lifespan)
app.include_router(build_worker_router(runtime))
