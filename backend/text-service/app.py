import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from classification.model_loader import load_models
import classification.model_loader as model_loader
from classification.router import router as classification_router
from xai.router import router as xai_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan handler: loads models at startup and attaches them to app.state
    so both classification and XAI routers can share them.
    """
    # 1. Load models into memory
    load_models()

    # 2. Explicitly attach them to app.state so the XAI router can find them!
    import classification.model_loader as model_loader

    app.state.model = model_loader.get_model()
    app.state.tokenizer = model_loader.get_tokenizer()
    app.state.qwen_model = model_loader.get_qwen_model()
    app.state.qwen_tokenizer = model_loader.get_qwen_tokenizer()

    print("Model 2 loaded successfully and attached to app.state!")
    if model_loader.is_qwen_available():
        print("Qwen model loaded successfully and attached to app.state!")

    yield

    # Shutdown
    print("Shutting down")


app = FastAPI(
    title="AI Text Detection API",
    description="DeBERTa-v3-large Model 2 — AI text classification engine",
    version="1.0.0",
    lifespan=lifespan,
)

# Allowed CORS origins are configurable via env var (comma-separated), so
# different environments (dev/staging/prod) can set their own frontend
# origins without a code change.
_default_origins = ["http://localhost:3000", "http://localhost:5173"]
_env_origins = os.getenv("ALLOWED_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in _env_origins.split(",")]
    if _env_origins
    else _default_origins
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the classification and xai routers.
app.include_router(classification_router)
app.include_router(xai_router)


@app.get("/")
def root():
    """Basic liveness/info endpoint pointing to the interactive API docs."""
    return {"message": "AI Detection API running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
