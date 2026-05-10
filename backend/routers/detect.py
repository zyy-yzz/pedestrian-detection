from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.core.dependencies import get_current_user
from backend.services.detection_service import DetectionService

router = APIRouter(prefix="/detect", tags=["detection"])

_settings = get_settings()


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    enhancements: str = Form("clahe,denoise"),
    compare: bool = Form(False),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    if file.content_type not in _settings.allowed_image_types:
        raise HTTPException(status_code=422, detail="Unsupported file type")

    file_bytes = await file.read()
    enh_list = [e.strip() for e in enhancements.split(",") if e.strip()]

    try:
        result = await DetectionService.process_image(
            db=db,
            user_id=user["user_id"],
            file_bytes=file_bytes,
            filename=file.filename or "unknown.jpg",
            content_type=file.content_type or "image/jpeg",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            enhancements=enh_list,
            compare=compare,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Detection processing error: {str(e)}")


@router.post("/video")
async def detect_video(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    raise HTTPException(status_code=501, detail="Video detection not yet implemented")


@router.get("/status/{job_id}")
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    from backend.services.job_service import JobService
    job = await JobService.get(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "pedestrian_count": job.pedestrian_count,
        "processing_time_ms": job.processing_time_ms,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
    }