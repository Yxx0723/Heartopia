from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.ui_detector import UiDetector, UiDetectorError


def test_ui_detector_matches_inventory_full_template(tmp_path: Path) -> None:
    frame = np.zeros((50, 70, 3), dtype=np.uint8)
    frame[12:20, 25:35] = (30, 100, 200)
    frame[14:18, 28:32] = (200, 40, 10)
    template_path = tmp_path / "inventory_full.png"
    cv2.imwrite(str(template_path), frame[12:20, 25:35])
    detector = UiDetector(
        inventory_full_template_paths=[template_path],
        inventory_threshold=0.95,
    )

    result = detector.detect(frame)

    assert result.inventory_full is True
    assert result.abnormal_ui is False
    assert result.blocking is True
    assert result.matched_label == "inventory_full"


def test_ui_detector_without_templates_is_non_blocking() -> None:
    result = UiDetector().detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert result.blocking is False
    assert result.inventory_confidence == 0.0


def test_ui_detector_rejects_constant_template(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    cv2.imwrite(str(path), np.full((8, 8), 100, dtype=np.uint8))

    with pytest.raises(UiDetectorError, match="像素变化过小"):
        UiDetector(abnormal_template_paths=[path])
