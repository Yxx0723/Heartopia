from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision.prompt_detector import PromptDetector


def test_prompt_detector_finds_template_in_configured_roi() -> None:
    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 255
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[45:53, 55:63] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    detector = PromptDetector(templates=[template], threshold=0.95)

    result = detector.detect_result(frame)

    assert result.detected is True
    assert result.confidence >= 0.95
    assert result.bbox is not None


def test_prompt_detector_returns_false_without_match_or_template() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    template = np.zeros((8, 8), dtype=np.uint8)
    template[2:6, 2:6] = 255

    assert PromptDetector().detect(frame) is False
    assert PromptDetector(templates=[template], threshold=0.95).detect(frame) is False


def test_prompt_detector_rejects_invalid_roi() -> None:
    with pytest.raises(ValueError, match="roi"):
        PromptDetector(roi=(0.8, 0.0, 0.4, 0.4))
