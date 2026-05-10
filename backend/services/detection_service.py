from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.services.model_loader import get_model, get_vanilla_model
from backend.services.job_service import JobService
from backend.ml.postprocessor import draw_boxes, encode_annotated_image


settings = get_settings()

# Available enhancement methods
ALL_ENHANCEMENTS = ["clahe", "denoise", "histogram_eq", "sharpen", "gamma_correct", "adaptive_threshold"]


class DetectionService:
    """Orchestrates image preprocessing, model inference, and result persistence."""

    @staticmethod
    async def process_image(
        db: AsyncSession,
        user_id: int,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enhancements: list[str] | None = None,
        compare: bool = False,
    ) -> dict:
        # Validate
        if content_type not in settings.allowed_image_types:
            raise ValueError(f"Unsupported file type: {content_type}")
        if len(file_bytes) > settings.max_upload_size_mb * 1024 * 1024:
            raise ValueError(f"File too large (max {settings.max_upload_size_mb} MB)")

        # Sanitize enhancements list
        if enhancements is None:
            enhancements = ["clahe", "denoise"]
        valid_enhancements = [e for e in enhancements if e in ALL_ENHANCEMENTS]
        unrecognized = [e for e in enhancements if e not in ALL_ENHANCEMENTS]
        enhancements = valid_enhancements

        job_id = str(uuid.uuid4())
        model = get_model()

        # Create job
        await JobService.create(
            db, job_id, user_id, input_type="image",
            original_filename=filename,
            conf_threshold=conf_threshold, iou_threshold=iou_threshold,
        )

        try:
            # Decode image
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image")

            await JobService.update_status(db, job_id, "processing")

            # Apply per-request thresholds to model (thread-safe: read-only after load)
            model.conf_threshold = conf_threshold
            model.iou_threshold = iou_threshold

            # Run enhanced detection
            start = time.monotonic()
            detections = await asyncio.to_thread(
                _run_detect_enhanced, model, img, enhancements
            )
            enhanced_ms = int((time.monotonic() - start) * 1000)

            enhanced_annotated = draw_boxes(img.copy(), detections)
            enhanced_items = _format_detections(detections)

            # -----------------------------------------------------------------
            # Compare mode: also run vanilla YOLOv8
            # -----------------------------------------------------------------
            vanilla_items = []
            vanilla_ms = 0
            vanilla_count = 0
            vanilla_annotated_b64 = None

            if compare:
                vanilla_model = get_vanilla_model()
                vanilla_model.conf_threshold = conf_threshold
                vanilla_model.iou_threshold = iou_threshold
                v_start = time.monotonic()
                vanilla_dets = await asyncio.to_thread(_run_detect_vanilla, vanilla_model, img)
                vanilla_ms = int((time.monotonic() - v_start) * 1000)
                vanilla_count = len(vanilla_dets)
                vanilla_annotated = draw_boxes(img.copy(), vanilla_dets)
                vanilla_annotated_b64 = encode_annotated_image(vanilla_annotated)
                vanilla_items = _format_detections(vanilla_dets)

            # Persist enhanced results
            await JobService.add_results(db, job_id, detections)
            await JobService.update_status(
                db, job_id, "completed",
                pedestrian_count=len(detections),
                processing_time_ms=enhanced_ms,
            )
            await JobService.add_audit_log(
                db, job_id, "status_changed",
                {"status": "completed", "pedestrian_count": len(detections),
                 "enhancements": enhancements, "compare": compare},
            )

            # Build response
            enhanced_data = {
                "pedestrian_count": len(detections),
                "processing_time_ms": enhanced_ms,
                "detections": enhanced_items,
                "annotated_image": encode_annotated_image(enhanced_annotated),
            }

            if compare:
                response = {
                    "job_id": job_id,
                    "status": "completed",
                    "warnings": [f"Unrecognized enhancement: {e}" for e in unrecognized] if unrecognized else [],
                    "enhancements_used": enhancements,
                    "enhanced": enhanced_data,
                    "vanilla": {
                        "pedestrian_count": vanilla_count,
                        "processing_time_ms": vanilla_ms,
                        "detections": vanilla_items,
                        "annotated_image": vanilla_annotated_b64,
                    },
                    "comparison": {
                        "delta_count": len(detections) - vanilla_count,
                        "delta_time_ms": enhanced_ms - vanilla_ms,
                        "avg_conf_enhanced": (
                            sum(d["confidence"] for d in detections) / len(detections)
                            if detections else 0.0
                        ),
                        "avg_conf_vanilla": (
                            sum(d["confidence"] for d in vanilla_dets) / len(vanilla_dets)
                            if vanilla_dets else 0.0
                        ),
                    },
                }
            else:
                response = dict(
                    job_id=job_id, status="completed",
                    **({"warnings": [f"Unrecognized enhancement: {e}" for e in unrecognized]}
                       if unrecognized else {}),
                    **enhanced_data,
                )

            return response

        except Exception as exc:
            await JobService.update_status(db, job_id, "failed", error_message=str(exc))
            try:
                await db.commit()
            except Exception:
                pass
            return {"job_id": job_id, "status": "failed", "error_message": str(exc)}

    @staticmethod
    async def get_results(db: AsyncSession, job_id: str) -> dict | None:
        job = await JobService.get_with_results(db, job_id)
        if job is None:
            return None
        detections = [
            {"id": r.id, "bbox": {"x1": r.bbox_x1, "y1": r.bbox_y1,
                                   "x2": r.bbox_x2, "y2": r.bbox_y2},
             "confidence": r.confidence, "class_name": "pedestrian"}
            for r in job.results
        ]
        return {
            "job_id": job.id, "status": job.status,
            "pedestrian_count": job.pedestrian_count,
            "processing_time_ms": job.processing_time_ms,
            "detections": detections,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }

    @staticmethod
    async def process_stream_frame(frame_bytes: bytes) -> dict:
        model = get_model()
        img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode frame")
        start = time.monotonic()
        detections = await asyncio.to_thread(_run_detect, model, img)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return {
            "pedestrian_count": len(detections),
            "detections": [
                {"bbox": d["bbox"], "confidence": d["confidence"], "class_id": d.get("class_id", 0)}
                for d in detections
            ],
            "inference_ms": elapsed_ms,
        }


def _format_detections(detections: list[dict]) -> list[dict]:
    return [
        {"id": i + 1,
         "bbox": {"x1": d["bbox"][0], "y1": d["bbox"][1],
                  "x2": d["bbox"][2], "y2": d["bbox"][3]},
         "confidence": d["confidence"],
         "class_name": d.get("class_name", "pedestrian")}
        for i, d in enumerate(detections)
    ]


def _run_detect(model, img: np.ndarray) -> list[dict]:
    """Run model.detect() using default enhancements."""
    return model.detect(img)


def _run_detect_enhanced(model, img: np.ndarray, enhancements: list[str]) -> list[dict]:
    """Run model.detect() with specified enhancements."""
    return model.detect(img, enhancements=enhancements)


def _run_detect_vanilla(vanilla_model, img: np.ndarray) -> list[dict]:
    """Run vanilla YOLOv8 detection."""
    return vanilla_model.detect(img)