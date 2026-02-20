from app.api.v1 import chat
from app.api.v1 import config
from app.api.v1 import detect
from fastapi import APIRouter

router = APIRouter()

router.include_router(detect.router)
router.include_router(chat.router)
router.include_router(config.router)
