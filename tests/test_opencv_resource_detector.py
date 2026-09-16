from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.resource_detector import OpenCVResourceDetector


def test_opencv_detector_finds_one_resource() -> None:
    template = np.zeros((10, 12), dtype=np.uint8)
    template[2:8, 3:9] = 255
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[30:40, 60:72] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    detector = OpenCVResourceDetector(
        resource_type="wood",
        templates=[template],
        threshold=0.95,
    )

    detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].resource_type == "wood"
    assert detections[0].x1 == 60
    assert detections[0].y1 == 30
    assert detections[0].confidence >= 0.95


def test_opencv_detector_applies_nms_to_overlapping_template_hits() -> None:
    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 255
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[20:28, 30:38] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    detector = OpenCVResourceDetector(templates=[template], threshold=0.8)

    assert len(detector.detect(frame)) == 1


def test_opencv_detector_requires_templates() -> None:
    with pytest.raises(ValueError, match="至少需要"):
        OpenCVResourceDetector()
