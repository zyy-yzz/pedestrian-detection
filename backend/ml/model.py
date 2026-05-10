"""
Enhanced YOLOv8 model wrapper for inference.

Loads the enhanced model (CBAM + C2fGhost + BiFPN),
handles pre/post-processing, and returns structured detections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.engine.results import Results

from backend.ml.preprocessor import NightPreprocessor


class EnhancedYOLOv8:
    """
    Production inference wrapper for the enhanced YOLOv8 model.

    Handles model loading, pre/post processing, and returns
    structured detection results ready for the API layer.

    Usage:
        model = EnhancedYOLOv8("models/enhanced_yolov8n.pt", device="cuda:0")
        detections = model.detect(bgr_image)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        self.model: YOLO | None = None
        self._loaded = False
        self.preprocessor = NightPreprocessor(img_size=img_size)

    def _resolve_device(self, device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def load(self) -> None:
        """Load model weights. Call once before inference."""
        if not self._loaded:
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            self.model.overrides["conf"] = self.conf_threshold
            self.model.overrides["iou"] = self.iou_threshold
            _reinject_wiou_loss()
            _register_enhancement_hooks(self.model)
            self._loaded = True

    def unload(self) -> None:
        """Release model from memory."""
        _restore_bbox_iou()
        if self.model is not None:
            if self.device != "cpu":
                self.model.to("cpu")
            del self.model
            self.model = None
        self._loaded = False
        if self.device != "cpu":
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    @property
    def model_version(self) -> str:
        return f"enhanced-yolov8n-{self.img_size}"

    def detect(self, img: np.ndarray, enhancements: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Run detection on a single pre-processed image.

        Args:
            img: BGR image as numpy array [H, W, 3] uint8.
            enhancements: list of enhancement names to apply (None = defaults).

        Returns:
            List of detections with bbox, confidence, class_id, class_name.
        """
        if not self._loaded:
            self.load()

        original_h, original_w = img.shape[:2]
        
        tensor = self.preprocessor.process(img, enhancements=enhancements).to(self.device)
        
        results = self.forward_tensor(tensor)
        
        return self._parse_results(results, (original_h, original_w))

    def detect_batch(
        self,
        images: list[np.ndarray],
    ) -> list[list[dict[str, Any]]]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]

    def forward_tensor(self, tensor: torch.Tensor) -> list[dict[str, Any]]:
        """
        Run inference directly on a pre-processed tensor.

        For use with stream processing where pre-processing happens externally.

        Args:
            tensor: [1, 3, H, W] float32, values in [0, 1].

        Returns:
            List of detections (raw, not reverse-letterboxed).
        """
        if not self._loaded:
            self.load()

        assert self.model is not None

        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            results: list[Results] = self.model(
                tensor,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        return self._parse_raw(results)

    def _parse_results(
        self,
        results: list[dict[str, Any]],
        original_shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Parse raw detections and map coordinates back to original image."""
        original_h, original_w = original_shape
        
        if not results:
            return []

        boxes = np.array([det["bbox"] for det in results]) / self.img_size
        confs = np.array([det["confidence"] for det in results])
        cls_ids = np.array([det["class_id"] for det in results])

        original_boxes = self.preprocessor.inverse_letterbox_coords(
            boxes, original_h, original_w
        )

        detections = []
        for box, conf, cls_id in zip(original_boxes, confs, cls_ids):
            detections.append({
                "bbox": box.tolist(),
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": self._class_name(int(cls_id)),
            })

        return detections

    def _parse_raw(self, results: list[Results]) -> list[dict[str, Any]]:
        """Parse results without coordinate scaling (for tensor input)."""
        detections: list[dict[str, Any]] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                detections.append({
                    "bbox": box.tolist(),
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": self._class_name(int(cls_id)),
                })

        return detections

    @staticmethod
    def _class_name(cls_id: int) -> str:
        names = {0: "pedestrian"}
        return names.get(cls_id, "unknown")

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for the /model/info endpoint."""
        return {
            "version": self.model_version,
            "img_size": self.img_size,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "num_classes": 1,
            "class_names": ["pedestrian"],
        }



class VanillaYOLOv8:
    """Vanilla YOLOv8 wrapper for baseline comparison — no custom modules or hooks."""

    def __init__(self, model_path: str | Path, device: str = "cuda:0",
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 img_size: int = 640) -> None:
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model: YOLO | None = None
        self._loaded = False
        from backend.ml.preprocessor import VanillaPreprocessor
        self.preprocessor = VanillaPreprocessor(img_size=img_size)

    def _resolve_device(self, device: str) -> str:
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def load(self) -> None:
        if not self._loaded:
            _restore_bbox_iou()
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            self.model.overrides["conf"] = self.conf_threshold
            self.model.overrides["iou"] = self.iou_threshold
            self._loaded = True

    def unload(self) -> None:
        if self.model is not None:
            if self.device != "cpu":
                self.model.to("cpu")
            del self.model
            self.model = None
        self._loaded = False
        if self.device != "cpu":
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    @property
    def model_version(self) -> str:
        return f"vanilla-yolov8n-{self.img_size}"

    def detect(self, img: np.ndarray) -> list[dict[str, Any]]:
        if not self._loaded:
            self.load()
        original_h, original_w = img.shape[:2]
        tensor = self.preprocessor.process(img).to(self.device)
        results = self._forward_tensor(tensor)
        return self._parse_results(results, (original_h, original_w))

    def _forward_tensor(self, tensor: torch.Tensor) -> list[dict[str, Any]]:
        assert self.model is not None
        tensor = tensor.to(self.device)
        with torch.no_grad():
            results: list[Results] = self.model(
                tensor, conf=self.conf_threshold, iou=self.iou_threshold,
                device=self.device, verbose=False,
            )
        return self._parse_raw(results)

    def _parse_results(self, results, original_shape) -> list[dict[str, Any]]:
        original_h, original_w = original_shape
        if not results:
            return []
        boxes = np.array([det["bbox"] for det in results]) / self.img_size
        confs = np.array([det["confidence"] for det in results])
        cls_ids = np.array([det["class_id"] for det in results])
        original_boxes = self.preprocessor.inverse_letterbox_coords(boxes, original_h, original_w)
        return [{"bbox": b.tolist(), "confidence": float(c), "class_id": int(i),
                 "class_name": self._class_name(int(i))}
                for b, c, i in zip(original_boxes, confs, cls_ids)]

    def _parse_raw(self, results: list[Results]) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            b = result.boxes.xyxy.cpu().numpy()
            c = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for bx, cf, ci in zip(b, c, cls):
                detections.append({"bbox": bx.tolist(), "confidence": float(cf),
                                   "class_id": int(ci), "class_name": self._class_name(int(ci))})
        return detections

    @staticmethod
    def _class_name(cls_id: int) -> str:
        return {0: "pedestrian"}.get(cls_id, "unknown")

    def get_model_info(self) -> dict[str, Any]:
        return {"version": self.model_version, "img_size": self.img_size,
                "conf_threshold": self.conf_threshold, "iou_threshold": self.iou_threshold,
                "device": self.device, "num_classes": 1, "class_names": ["pedestrian"]}

_orig_bbox_iou = None


def _reinject_wiou_loss(delta: float = 1.9) -> None:
    """
    Re-inject WIoU loss monkey-patch for inference-time confidence calibration.
    The training monkey-patch does not persist across process restarts.
    Only affects the enhanced model — vanilla model restores the original.
    """
    global _orig_bbox_iou
    from ultralytics.utils.metrics import bbox_iou as _current_bbox_iou
    import ultralytics.utils.metrics as metrics_module

    if _orig_bbox_iou is None:
        _orig_bbox_iou = _current_bbox_iou

    def _wiou_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        iou = _orig_bbox_iou(box1, box2, xywh=xywh, GIoU=False, DIoU=False, CIoU=False, eps=eps)
        with torch.no_grad():
            beta = (1.0 - iou.detach()) / delta
            weight = torch.exp(beta)
        return 1.0 - weight * (1.0 - iou)

    metrics_module.bbox_iou = _wiou_bbox_iou


def _restore_bbox_iou() -> None:
    """Restore the original bbox_iou after WIoU monkey-patch."""
    global _orig_bbox_iou
    if _orig_bbox_iou is not None:
        import ultralytics.utils.metrics as metrics_module
        metrics_module.bbox_iou = _orig_bbox_iou
        _orig_bbox_iou = None


def _register_enhancement_hooks(model) -> None:
    """
    Register forward hooks for CBAM and BiFPN if the model was trained
    with these modules. Uses model-level attributes saved by train.py.
    """
    inner = model.model

    if hasattr(inner, 'cbam_p3') and hasattr(inner, 'use_cbam') and inner.use_cbam:
        _inject_cbam_hooks(inner)

    if hasattr(inner, 'bifpn') and hasattr(inner, 'use_bifpn') and inner.use_bifpn:
        _inject_bifpn_hooks(inner)


def _inject_cbam_hooks(model) -> None:
    """Inject CBAM modules after backbone P3/P4/P5 feature maps."""
    import logging
    _log = logging.getLogger(__name__)

    try:
        model.cbam_p3.to(model._device)
        model.cbam_p4.to(model._device)
        model.cbam_p5.to(model._device)
    except Exception:
        _log.warning("CBAM device transfer failed, skipping hook injection")
        return

    layers = list(model.model)

    # Locate SPPF (end of backbone) and preceding C2f blocks (P4, P3 outputs)
    sppf_idx = None
    for i, layer in enumerate(layers):
        if 'SPPF' in type(layer).__name__:
            sppf_idx = i
            break
    if sppf_idx is None:
        _log.warning("SPPF layer not found, CBAM hooks not injected")
        return

    p4_idx = None
    for i in range(sppf_idx - 1, -1, -1):
        if 'C2f' in type(layers[i]).__name__:
            p4_idx = i
            break
    if p4_idx is None:
        _log.warning("P4 C2f layer not found, CBAM hooks not injected")
        return

    p3_idx = None
    for i in range(p4_idx - 1, -1, -1):
        if 'C2f' in type(layers[i]).__name__:
            p3_idx = i
            break
    if p3_idx is None:
        _log.warning("P3 C2f layer not found, CBAM hooks not injected")
        return

    def _make_hook(cbam):
        def _hook(_module, _input, output):
            return cbam(output)
        return _hook

    handles = [
        layers[p3_idx].register_forward_hook(_make_hook(model.cbam_p3)),
        layers[p4_idx].register_forward_hook(_make_hook(model.cbam_p4)),
        layers[sppf_idx].register_forward_hook(_make_hook(model.cbam_p5)),
    ]
    model._cbam_handles = handles


def _inject_bifpn_hooks(model) -> None:
    """Replace PANet neck with BiFPN by wrapping model.forward.

    Intercepts backbone outputs (P3/P4/P5), runs them through BiFPN,
    and replaces the features in the save list before the neck consumes them.
    """
    import logging
    _log = logging.getLogger(__name__)

    try:
        model.bifpn.to(model._device)
    except Exception:
        _log.warning("BiFPN device transfer failed, skipping hook injection")
        return

    layers = list(model.model)

    # Locate SPPF (end of backbone)
    sppf_idx = None
    for i, layer in enumerate(layers):
        if 'SPPF' in type(layer).__name__:
            sppf_idx = i
            break
    if sppf_idx is None:
        _log.warning("SPPF layer not found, BiFPN hooks not injected")
        return

    # Get backbone save indices (P3, P4, P5 in the model's save list)
    save = getattr(model, 'save', [])
    bb_save = sorted([s for s in save if s <= sppf_idx])

    if len(bb_save) < 3:
        _log.warning("Insufficient backbone save indices (%d), BiFPN hooks not injected", len(bb_save))
        return

    # Wrap forward to inject BiFPN between backbone and neck
    _orig_forward = model.forward

    def _bifpn_forward(self, x, *args, **kwargs):
        _layers = list(self.model)
        y = []
        dt = []

        # Manual forward through backbone
        for i in range(sppf_idx + 1):
            m = _layers[i]
            if hasattr(m, 'f') and m.f != -1:
                f = m.f
                x = y[f] if isinstance(f, int) else [
                    x if j == -1 else y[j] for j in f
                ]
            x = m(x)
            y.append(x if hasattr(m, 'i') and m.i in self.save else None)

        # Process backbone P3/P4/P5 through BiFPN
        bb_feats = [y[idx] for idx in bb_save if y[idx] is not None]
        if len(bb_feats) == 3:
            enhanced = self.bifpn(bb_feats)
            for idx, feat in zip(bb_save, enhanced):
                y[idx] = feat

        # Continue through neck and head
        for i in range(sppf_idx + 1, len(_layers)):
            m = _layers[i]
            if hasattr(m, 'f') and m.f != -1:
                f = m.f
                x = y[f] if isinstance(f, int) else [
                    x if j == -1 else y[j] for j in f
                ]
            x = m(x)
            y.append(x if hasattr(m, 'i') and m.i in self.save else None)

        return x

    model.forward = _bifpn_forward.__get__(model, type(model))
