from __future__ import annotations

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Night Pedestrian Detector"
    debug: bool = False
    env: str = "development"

    # Database
    database_url: str = ""

    # Auth
    secret_key: str = ""
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    algorithm: str = "HS256"

    # Model
    model_path: str = "models/enhanced_yolov8n.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "cuda:0"

    # File upload
    max_upload_size_mb: int = 20
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/bmp"]
    allowed_video_types: list[str] = ["video/mp4", "video/avi", "video/mov"]

    # Rate limiting
    login_rate_limit: int = 5  # attempts per minute per IP

    # CORS
    cors_origins: list[str] = ["http://localhost:80", "http://localhost:5173"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
