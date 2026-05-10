from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.core.dependencies import get_current_user
from backend.services.detection_service import DetectionService
from backend.services.job_service import JobService

router = APIRouter(tags=["results"])




# ---------------------------------------------------------------------------
# Result detail endpoints
# ---------------------------------------------------------------------------

@router.get("/results/{job_id}")
async def get_results(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    result = await DetectionService.get_results(db, job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if result["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {result['status']}")
    return result


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------

@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    jobs, total = await JobService.list_by_user(db, user["user_id"], page=page, page_size=page_size)

    items = [
        {
            "job_id": job.id,
            "input_type": job.input_type,
            "original_filename": job.original_filename,
            "status": job.status,
            "pedestrian_count": job.pedestrian_count,
            "created_at": job.created_at.isoformat() if job.created_at else None,
        }
        for job in jobs
    ]

    return {
        "jobs": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.delete("/history/{job_id}")
async def delete_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    job = await JobService.get(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.user_id != user["user_id"] and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this job")
    deleted = await JobService.delete(db, job_id)
    return {"status": "deleted", "job_id": job_id}
