from app.api.v1.external_routes import router as external_router
from app.api.v1.me_routes import router as me_router
from app.api.v1.prediction_routes import router as prediction_router
from app.api.v1.user_routes import router as user_router
from app.api.v1.voice_routes import router as voice_router
from app.api.v1.xai_routes import router as xai_router

__all__ = [
    "external_router",
    "me_router",
    "prediction_router",
    "user_router",
    "voice_router",
    "xai_router",
]
