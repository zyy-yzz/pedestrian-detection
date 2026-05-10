from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.database import init_db
from backend.routers import auth_router, detect_router, results_router, stream_router
from backend.services.model_loader import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    settings = get_settings()
    await init_db()
    try:
        get_model()  # preload model
    except FileNotFoundError:
        pass  # model file may not exist in dev
    yield
    # Shutdown — model unloaded via process exit


app = FastAPI(
    title="Night Pedestrian Detector API",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(detect_router)
app.include_router(results_router)
app.include_router(stream_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model/info")
async def model_info():
    try:
        model = get_model()
        return model.get_model_info()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": f"Model not available: {str(exc)}"},
        )
