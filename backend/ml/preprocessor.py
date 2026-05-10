"""
Night-Time Image Pre-processing Pipeline.

Configurable enhancement pipeline for low-light imagery:
  1. CLAHE on L channel (contrast enhancement)
  2. Bilateral denoising (edge-preserving noise suppression)
  3. Histogram equalization (global contrast)
  4. Unsharp mask sharpening (edge enhancement)
  5. Gamma correction (brightness curve adjustment)
  6. Adaptive threshold normalization (local contrast)

Training augmentations: Mosaic, MixUp, RandomDarken, GaussianNoise,
HorizontalFlip, HSV augment, RandomAffine.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Inference preprocessors
# ---------------------------------------------------------------------------

class NightPreprocessor:
    """
    Configurable night-time image enhancement pipeline.

    Each enhancement can be toggled on/off via the `enhancements` list
    passed to `process()`.

    Usage:
        preprocessor = NightPreprocessor(img_size=640)
        tensor = preprocessor.process(bgr_image, enhancements=["clahe", "denoise"])
        # or: preprocessor.process(bgr_image) for all defaults
    """

    _AVAILABLE_ENHANCEMENTS = [
        "clahe", "denoise", "histogram_eq",
        "sharpen", "gamma_correct", "adaptive_threshold",
    ]

    def __init__(
        self,
        img_size: int = 640,
        clahe_clip_limit: float = 3.0,
        clahe_tile: tuple[int, int] = (8, 8),
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        sharpen_amount: float = 1.5,
        sharpen_radius: int = 3,
        gamma_value: float = 1.2,
        adaptive_clip_limit: float = 2.0,
        adaptive_tile: tuple[int, int] = (8, 8),
        pad_colour: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        self.img_size = img_size
        self.pad_colour = pad_colour

        # CLAHE
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile = clahe_tile
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile)

        # Bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        # Sharpen
        self.sharpen_amount = sharpen_amount
        self.sharpen_radius = sharpen_radius

        # Gamma
        self.gamma_value = gamma_value
        self._gamma_lut: np.ndarray | None = None

        # Adaptive threshold
        self.adaptive_clip_limit = adaptive_clip_limit
        self.adaptive_tile = adaptive_tile
        self._adaptive_clahe = cv2.createCLAHE(clipLimit=adaptive_clip_limit, tileGridSize=adaptive_tile)

        # Thread-local letterbox state (set during process)
        self._tls = threading.local()
        self._tls.scale = 1.0
        self._tls.pad_top = 0
        self._tls.pad_left = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        img: np.ndarray,
        enhancements: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Full pre-processing pipeline with selectable enhancements.

        Args:
            img: BGR image [H, W, 3], uint8.
            enhancements: list of enhancement names to apply.
                          None = apply all defaults ["clahe", "denoise"].
                          Empty list = no enhancements (letterbox + normalize only).

        Returns:
            Float tensor [1, 3, img_size, img_size], values in [0, 1].
        """
        if enhancements is None:
            enhancements = ["clahe", "denoise"]

        for name in enhancements:
            if name == "clahe":
                img = self._clahe_enhance(img)
            elif name == "denoise":
                img = self._bilateral_denoise(img)
            elif name == "histogram_eq":
                img = self._histogram_equalize(img)
            elif name == "sharpen":
                img = self._sharpen(img)
            elif name == "gamma_correct":
                img = self._gamma_correct(img)
            elif name == "adaptive_threshold":
                img = self._adaptive_threshold_norm(img)

        img = self._letterbox(img)
        return self._normalize(img)

    # ------------------------------------------------------------------
    # Enhancement methods
    # ------------------------------------------------------------------

    def _clahe_enhance(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the L channel in LAB space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self.clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _bilateral_denoise(self, img: np.ndarray) -> np.ndarray:
        """Edge-preserving bilateral filter to suppress sensor noise."""
        return cv2.bilateralFilter(
            img, self.bilateral_d,
            self.bilateral_sigma_color, self.bilateral_sigma_space,
        )

    def _histogram_equalize(self, img: np.ndarray) -> np.ndarray:
        """Global histogram equalization on the L channel in LAB space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = cv2.equalizeHist(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Unsharp mask sharpening to enhance edge definition."""
        blurred = cv2.GaussianBlur(img, (0, 0), self.sharpen_radius)
        return cv2.addWeighted(img, self.sharpen_amount, blurred, -0.5, 0)

    def _gamma_correct(self, img: np.ndarray) -> np.ndarray:
        """Gamma correction to adjust overall brightness curve."""
        if self._gamma_lut is None or len(self._gamma_lut) != 256:
            self._gamma_lut = (
                (255.0 * (np.arange(256) / 255.0) ** (1.0 / self.gamma_value))
                .clip(0, 255)
                .astype(np.uint8)
            )
        return cv2.LUT(img, self._gamma_lut)

    def _adaptive_threshold_norm(self, img: np.ndarray) -> np.ndarray:
        """CLAHE-based local contrast normalization on the L channel."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self._adaptive_clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Geometry (shared)
    # ------------------------------------------------------------------

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """Resize to img_size x img_size, maintaining aspect ratio with padding."""
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        self._tls.scale = scale
        self._tls.pad_top = top
        self._tls.pad_left = left
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.pad_colour,
        )

    def _normalize(self, img: np.ndarray) -> torch.Tensor:
        """Scale to [0, 1] and transpose HWC -> CHW, add batch dim."""
        tensor = img.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor)
        tensor = np.expand_dims(tensor, axis=0)
        return torch.from_numpy(tensor)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        return self._tls.scale

    def inverse_letterbox_coords(
        self,
        boxes: np.ndarray,
        original_h: int,
        original_w: int,
    ) -> np.ndarray:
        """Map normalised coords back to original image pixel space."""
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= self.img_size
        boxes[:, [1, 3]] *= self.img_size
        boxes[:, [0, 2]] -= self._tls.pad_left
        boxes[:, [1, 3]] -= self._tls.pad_top
        pad_h = self.img_size - 2 * self._tls.pad_top
        pad_w = self.img_size - 2 * self._tls.pad_left
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, pad_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, pad_h)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self._tls.scale
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self._tls.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h)
        return boxes


class VanillaPreprocessor:
    """
    Minimal preprocessor for baseline YOLOv8 comparison.
    Only performs letterbox + normalize, no night enhancements.
    """

    def __init__(
        self,
        img_size: int = 640,
        pad_colour: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        self.img_size = img_size
        self.pad_colour = pad_colour
        self._tls = threading.local()
        self._tls.scale = 1.0
        self._tls.pad_top = 0
        self._tls.pad_left = 0

    def process(self, img: np.ndarray) -> torch.Tensor:
        img = self._letterbox(img)
        return self._normalize(img)

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        self._tls.scale = scale
        self._tls.pad_top = top
        self._tls.pad_left = left
        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.pad_colour,
        )

    def _normalize(self, img: np.ndarray) -> torch.Tensor:
        tensor = img.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor)
        tensor = np.expand_dims(tensor, axis=0)
        return torch.from_numpy(tensor)

    @property
    def scale_factor(self) -> float:
        return self._tls.scale

    def inverse_letterbox_coords(
        self, boxes: np.ndarray, original_h: int, original_w: int,
    ) -> np.ndarray:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= self.img_size
        boxes[:, [1, 3]] *= self.img_size
        boxes[:, [0, 2]] -= self._tls.pad_left
        boxes[:, [1, 3]] -= self._tls.pad_top
        pad_h = self.img_size - 2 * self._tls.pad_top
        pad_w = self.img_size - 2 * self._tls.pad_left
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, pad_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, pad_h)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self._tls.scale
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self._tls.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h)
        return boxes


# ---------------------------------------------------------------------------
# Training augmentation stack (unchanged)
# ---------------------------------------------------------------------------

class NightAugmentation:
    """
    Training augmentation stack for night-time pedestrian detection.

    Pipeline:
        1. Mosaic (4-image collage)       p = 1.0
        2. MixUp (alpha blending)          p = 0.15
        3. RandomDarken (night simulation)  delta_brightness in [-40, 0]
        4. GaussianNoise (sensor sim)       sigma ~ U(5, 25)
        5. HorizontalFlip                   p = 0.5
        6. HSV augment (hue, sat, val)
        7. RandomAffine (translate, scale)
    """

    def __init__(
        self,
        img_size: int = 640,
        mosaic_p: float = 1.0,
        mixup_p: float = 0.15,
        darken_p: float = 0.5,
        darken_range: tuple[int, int] = (-40, 0),
        noise_p: float = 0.5,
        noise_sigma_range: tuple[float, float] = (5.0, 25.0),
        hflip_p: float = 0.5,
        hsv_h_gain: float = 0.015,
        hsv_s_gain: float = 0.7,
        hsv_v_gain: float = 0.4,
        translate: float = 0.1,
        scale_range: tuple[float, float] = (0.5, 1.5),
    ) -> None:
        self.img_size = img_size
        self.mosaic_p = mosaic_p
        self.mixup_p = mixup_p
        self.darken_p = darken_p
        self.darken_range = darken_range
        self.noise_p = noise_p
        self.noise_sigma_range = noise_sigma_range
        self.hflip_p = hflip_p
        self.hsv_h_gain = hsv_h_gain
        self.hsv_s_gain = hsv_s_gain
        self.hsv_v_gain = hsv_v_gain
        self.translate = translate
        self.scale_range = scale_range

    def __call__(self, img: np.ndarray, bboxes: np.ndarray | None = None):
        if np.random.random() < self.darken_p:
            delta = np.random.randint(*self.darken_range)
            img = self._adjust_brightness(img, delta)
        if np.random.random() < self.noise_p:
            sigma = np.random.uniform(*self.noise_sigma_range)
            noise = np.random.randn(*img.shape).astype(np.float32) * sigma
            img = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        img = self._hsv_augment(img)
        if np.random.random() < self.hflip_p:
            img = cv2.flip(img, 1)
            if bboxes is not None:
                bboxes[:, 1] = 1.0 - bboxes[:, 1]
        if bboxes is not None:
            img, bboxes = self._random_affine(img, bboxes)
        return img, bboxes

    @staticmethod
    def _adjust_brightness(img: np.ndarray, delta: int) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 2] = (hsv[:, :, 2] + delta).clip(0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _hsv_augment(self, img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_gain = 1.0 + np.random.uniform(-self.hsv_h_gain, self.hsv_h_gain)
        s_gain = 1.0 + np.random.uniform(-self.hsv_s_gain, self.hsv_s_gain)
        v_gain = 1.0 + np.random.uniform(-self.hsv_v_gain, self.hsv_v_gain)
        hsv[:, :, 0] = (hsv[:, :, 0] * h_gain) % 180
        hsv[:, :, 1] = (hsv[:, :, 1] * s_gain).clip(0, 255)
        hsv[:, :, 2] = (hsv[:, :, 2] * v_gain).clip(0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _random_affine(self, img, bboxes):
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        scale = np.random.uniform(*self.scale_range)
        tx = np.random.uniform(-self.translate, self.translate) * w
        ty = np.random.uniform(-self.translate, self.translate) * h
        matrix = cv2.getRotationMatrix2D(center, angle=0, scale=scale)
        matrix[:, 2] += [tx, ty]
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        if bboxes is not None:
            bboxes = self._affine_boxes(bboxes, scale, tx / w, ty / h)
        return img, bboxes

    @staticmethod
    def _affine_boxes(bboxes, scale, dx_norm, dy_norm):
        bboxes = bboxes.copy()
        bboxes[:, 1] = bboxes[:, 1] * scale + dx_norm
        bboxes[:, 2] = bboxes[:, 2] * scale + dy_norm
        bboxes[:, 3] *= scale
        bboxes[:, 4] *= scale
        bboxes[:, 1:5] = bboxes[:, 1:5].clip(0, 1)
        return bboxes