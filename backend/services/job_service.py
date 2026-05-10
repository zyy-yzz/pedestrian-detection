from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models.db_models import DetectionJob, DetectionResult, JobAuditLog


class JobService:
    """CRUD operations for detection jobs."""

    @staticmethod
    async def create(
        db: AsyncSession,
        job_id: str,
        user_id: int,
        input_type: str,
        original_filename: str | None = None,
        file_path: str | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        model_version: str = "enhanced-yolov8n-640",
    ) -> DetectionJob:
        job = DetectionJob(
            id=job_id,
            user_id=user_id,
            input_type=input_type,
            original_filename=original_filename,
            file_path=file_path,
            status="queued",
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            model_version=model_version,
        )
        db.add(job)
        await db.flush()
        return job

    @staticmethod
    async def get(db: AsyncSession, job_id: str) -> DetectionJob | None:
        return await db.get(DetectionJob, job_id)

    @staticmethod
    async def get_with_results(db: AsyncSession, job_id: str) -> DetectionJob | None:
        result = await db.execute(
            select(DetectionJob)
            .where(DetectionJob.id == job_id)
            .options(selectinload(DetectionJob.results))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_status(
        db: AsyncSession,
        job_id: str,
        status: str,
        pedestrian_count: int | None = None,
        processing_time_ms: int | None = None,
        error_message: str | None = None,
    ) -> DetectionJob | None:
        job = await db.get(DetectionJob, job_id)
        if job is None:
            return None
        job.status = status
        if pedestrian_count is not None:
            job.pedestrian_count = pedestrian_count
        if processing_time_ms is not None:
            job.processing_time_ms = processing_time_ms
        if error_message is not None:
            job.error_message = error_message
        if status in ("completed", "failed"):
            job.completed_at = datetime.now(tz=timezone.utc)
        await db.flush()
        return job

    @staticmethod
    async def add_results(
        db: AsyncSession,
        job_id: str,
        detections: list[dict],
        frame_index: int = 0,
    ) -> None:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            result = DetectionResult(
                job_id=job_id,
                frame_index=frame_index,
                bbox_x1=x1,
                bbox_y1=y1,
                bbox_x2=x2,
                bbox_y2=y2,
                confidence=det["confidence"],
                class_id=det.get("class_id", 0),
                track_id=det.get("track_id"),
            )
            db.add(result)
        await db.flush()

    @staticmethod
    async def add_audit_log(
        db: AsyncSession,
        job_id: str,
        event: str,
        detail: dict | None = None,
    ) -> None:
        log = JobAuditLog(job_id=job_id, event=event, detail=detail)
        db.add(log)
        await db.flush()

    @staticmethod
    async def list_by_user(
        db: AsyncSession,
        user_id: int,
        page: int = 1,
        page_size: int = 10,
    ) -> tuple[list[DetectionJob], int]:
        offset = (page - 1) * page_size
        count_result = await db.execute(
            select(func.count(DetectionJob.id)).where(DetectionJob.user_id == user_id)
        )
        total = count_result.scalar() or 0

        result = await db.execute(
            select(DetectionJob)
            .where(DetectionJob.user_id == user_id)
            .order_by(DetectionJob.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        return list(result.scalars().all()), total

    @staticmethod
    async def delete(db: AsyncSession, job_id: str) -> bool:
        job = await db.get(DetectionJob, job_id)
        if job is None:
            return False
        await db.delete(job)
        await db.flush()
        return True
