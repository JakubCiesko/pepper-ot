from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI

from app.worker.routes import build_worker_router
from app.worker.runtime import WorkerRuntime

logger = logging.getLogger(__name__)
runtime = WorkerRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config_path = os.environ.get("PEPPER_WORKER_CONFIG_PATH")
    if config_path:
        logger.info(f"Worker boot with config path hint: {config_path}")
    yield


app = FastAPI(title="Pepper Worker Internal API", version="0.1.0", lifespan=lifespan)
app.include_router(build_worker_router(runtime))
