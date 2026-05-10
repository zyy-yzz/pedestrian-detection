"""
Post-processing pipeline for YOLOv8 predictions.

- Non-Maximum Suppression (NMS)
- Box coordinate decoding (model space -> original image)
- Bounding box & label rendering
- Confidence-based colour coding
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

try:
    from ultralytics.engine.results import Results as YOLOResults
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    YOLOResults = type(None)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> np.ndarray:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.

    Args:
        boxes:  [N, 4] in xyxy format.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold for suppression.
        max_detections: maximum number of boxes to keep.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    # Sort by score descending
    order = scores.argsort()[::-1]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_detections:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Box decoding
# ---------------------------------------------------------------------------

def decode_predictions(
    raw_output: torch.Tensor | np.ndarray,
    img_size: int = 640,
    conf_threshold: float = 0.25,
    original_shape: tuple[int, int] | None = None,
) -> list[dict]:
    """
    Decode raw YOLO output into structured detection dicts.

    Handles scaling from model coordinates to original image pixel coordinates.

    Args:
        raw_output: ultralytics Results or raw tensor output.
        img_size: model input size.
        conf_threshold: minimum confidence to keep.
        original_shape: (h, w) of the original image for coordinate scaling.

    Returns:
        List of {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}
    """
    detections: list[dict] = []

    if HAS_ULTRALYTICS and isinstance(raw_output, YOLOResults):
        return _decode_yolo_results(raw_output, conf_threshold, original_shape)

    if isinstance(raw_output, torch.Tensor):
        raw_output = raw_output.cpu().numpy()

    if isinstance(raw_output, np.ndarray):
        return _decode_raw_tensor(raw_output, conf_threshold, original_shape, img_size)

    return detections


def _decode_yolo_results(
    results,
    conf_threshold: float,
    original_shape: tuple[int, int] | None,
) -> list[dict]:
    """Decode ultralytics Results object."""
    detections: list[dict] = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    mask = confs >= conf_threshold
    boxes = boxes[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        if original_shape is not None:
            orig_h, orig_w = original_shape
            x1 = x1 / results.orig_shape[1] * orig_w
            y1 = y1 / results.orig_shape[0] * orig_h
            x2 = x2 / results.orig_shape[1] * orig_w
            y2 = y2 / results.orig_shape[0] * orig_h

        detections.append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "class_id": int(cls_id),
        })

    return detections


def _decode_raw_tensor(
    preds: np.ndarray,
    conf_threshold: float,
    original_shape: tuple[int, int] | None,
    img_size: int,
) -> list[dict]:
    """Decode raw prediction tensor [1, N, 6]."""
    if preds.ndim == 3:
        preds = preds[0]  # remove batch dim

    mask = preds[:, 4] >= conf_threshold
    preds = preds[mask]

    detections: list[dict] = []
    for pred in preds:
        x1, y1, x2, y2 = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        conf = float(pred[4])
        cls_id = int(pred[5]) if pred.shape[0] > 5 else 0

        if original_shape is not None:
            orig_h, orig_w = original_shape
            x1 = x1 / img_size * orig_w
            y1 = y1 / img_size * orig_h
            x2 = x2 / img_size * orig_w
            y2 = y2 / img_size * orig_h

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls_id,
        })

    return detections


# ---------------------------------------------------------------------------
# Bounding box rendering
# ---------------------------------------------------------------------------

def draw_boxes(
    img: np.ndarray,
    detections: list[dict],
    class_names: dict[int, str] | None = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Colours are confidence-coded:
      >= 0.85  green
      >= 0.65  amber
      <  0.65  red

    Args:
        img: BGR image [H, W, 3] uint8 (returned as new copy).
        detections: list of detection dicts with "bbox", "confidence", "class_id".
        class_names: optional mapping from class_id to name.
        line_thickness: box border width.
        font_scale: label text size.

    Returns:
        Annotated image (same array, in-place).
    """
    if class_names is None:
        class_names = {0: "pedestrian"}

    result = img.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        conf = det["confidence"]
        cls_id = det.get("class_id", 0)

        colour = _confidence_colour(conf)

        # Draw box
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, line_thickness)

        # Draw label
        label = f"{class_names.get(cls_id, '?')} {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )

        # Label background
        cv2.rectangle(
            result,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            colour,
            -1,
        )
        # Label text
        cv2.putText(
            result, label, (x1 + 2, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            cv2.LINE_AA,
        )

    return result


def _confidence_colour(conf: float) -> tuple[int, int, int]:
    """Map confidence to BGR colour."""
    if conf >= 0.85:
        return (0, 197, 34)    # green  #22c55e
    if conf >= 0.65:
        return (11, 158, 245)  # amber  #f59e0b
    return (68, 68, 239)       # red    #ef4444


# ---------------------------------------------------------------------------
# Utility: encode annotated image for API response
# ---------------------------------------------------------------------------

def encode_annotated_image(img: np.ndarray, quality: int = 90) -> str:
    """
    Encode annotated BGR image as base64 JPEG string.

    Used for returning annotated images directly in API JSON responses.
    """
    import base64
    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")
