from __future__ import annotations

import threading
from pathlib import Path

from backend.core.config import get_settings
from backend.ml.model import EnhancedYOLOv8, VanillaYOLOv8


_model: EnhancedYOLOv8 | None = None
_model_failed: bool = False
_lock = threading.Lock()


def get_model() -> EnhancedYOLOv8:
    """Lazy-load singleton model instance. Thread-safe."""
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        raise RuntimeError("Model failed to load previously, refusing retry")
    with _lock:
        if _model is not None:
            return _model
        if _model_failed:
            raise RuntimeError("Model failed to load previously, refusing retry")
        settings = get_settings()
        model = EnhancedYOLOv8(
            model_path=Path(settings.model_path),
            device=settings.device,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            img_size=settings.img_size,
        )
        try:
            model.load()
        except Exception:
            _model_failed = True
            raise
        _model = model
    return _model



_vanilla_model: VanillaYOLOv8 | None = None
_vanilla_failed: bool = False
_vanilla_lock = threading.Lock()


def get_vanilla_model() -> VanillaYOLOv8:
    """Lazy-load singleton vanilla YOLOv8n for baseline comparison. Thread-safe."""
    global _vanilla_model, _vanilla_failed
    if _vanilla_model is not None:
        return _vanilla_model
    if _vanilla_failed:
        raise RuntimeError("Vanilla model failed to load previously, refusing retry")
    with _vanilla_lock:
        if _vanilla_model is not None:
            return _vanilla_model
        if _vanilla_failed:
            raise RuntimeError("Vanilla model failed to load previously, refusing retry")
        settings = get_settings()
        vmodel = VanillaYOLOv8(
            model_path=Path(settings.model_path),
            device=settings.device,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            img_size=settings.img_size,
        )
        try:
            vmodel.load()
        except Exception:
            _vanilla_failed = True
            raise
        _vanilla_model = vmodel
    return _vanilla_model

def reload_model() -> EnhancedYOLOv8:
    """Force reload model (e.g. after weights update). Thread-safe."""
    global _model, _model_failed
    with _lock:
        if _model is not None:
            _model.unload()
        _model = None
        _model_failed = False
    return get_model()
