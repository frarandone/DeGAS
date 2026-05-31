from fastapi.routing import APIRouter
from .optimization import router as optimization_router

API_VERSION_PREFIX = "/api"


api_router = APIRouter(prefix=API_VERSION_PREFIX, tags=["api"])
api_router.include_router(optimization_router)
