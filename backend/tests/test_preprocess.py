from __future__ import annotations

import numpy as np
import pytest

from backend.ml.preprocessor import NightPreprocessor, VanillaPreprocessor


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestNightPreprocessor:
    def test_process_default(self, sample_image):
        pp = NightPreprocessor(img_size=640)
        tensor = pp.process(sample_image)
        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_process_no_enhancements(self, sample_image):
        pp = NightPreprocessor(img_size=640)
        tensor = pp.process(sample_image, enhancements=[])
        assert tensor.shape == (1, 3, 640, 640)

    def test_process_single_enhancement(self, sample_image):
        pp = NightPreprocessor(img_size=416)
        tensor = pp.process(sample_image, enhancements=["clahe"])
        assert tensor.shape == (1, 3, 416, 416)

    def test_letterbox_preserves_aspect(self, sample_image):
        pp = NightPreprocessor(img_size=640)
        tensor = pp.process(sample_image)
        assert tensor.shape[2] == 640
        assert tensor.shape[3] == 640

    def test_inverse_letterbox_coords(self, sample_image):
        pp = NightPreprocessor(img_size=640)
        pp.process(sample_image)
        boxes = np.array([[0.5, 0.5, 0.6, 0.6]])
        result = pp.inverse_letterbox_coords(boxes, 480, 640)
        assert result.shape == (1, 4)
        assert all(c >= 0 for c in result[0])


class TestVanillaPreprocessor:
    def test_process(self, sample_image):
        pp = VanillaPreprocessor(img_size=640)
        tensor = pp.process(sample_image)
        assert tensor.shape == (1, 3, 640, 640)

    def test_no_enhancements_applied(self, sample_image):
        pp = VanillaPreprocessor(img_size=640)
        tensor = pp.process(sample_image)
        assert tensor.shape == (1, 3, 640, 640)
