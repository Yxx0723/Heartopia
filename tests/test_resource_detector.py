from __future__ import annotations

import numpy as np
import pytest

from core.models import Detection
from vision.resource_detector import (
    MockResourceDetector,
    MockScenario,
    ResourceDetector,
)


FRAME = np.zeros((100, 200, 3), dtype=np.uint8)


def test_mock_detector_implements_common_interface() -> None:
    detector = MockResourceDetector()

    assert isinstance(detector, ResourceDetector)
    assert detector.detect(FRAME) == []


def test_mock_scenarios_cover_single_multiple_moving_and_lost() -> None:
    single = MockResourceDetector(MockScenario.SINGLE)
    multiple = MockResourceDetector("multiple")
    moving = MockResourceDetector(MockScenario.MOVING)
    lost = MockResourceDetector(MockScenario.LOST, visible_frames=2)

    assert len(single.detect(FRAME)) == 1
    assert len(multiple.detect(FRAME)) == 3

    first_center = moving.detect(FRAME)[0].center
    second_center = moving.detect(FRAME)[0].center
    assert second_center[0] > first_center[0]

    assert len(lost.detect(FRAME)) == 1
    assert len(lost.detect(FRAME)) == 1
    assert lost.detect(FRAME) == []


def test_mock_detection_respects_frame_bounds_and_resource_type() -> None:
    detections = MockResourceDetector(
        MockScenario.MULTIPLE,
        resource_type="ore",
    ).detect(FRAME)

    for detection in detections:
        assert detection.resource_type == "ore"
        assert 0 <= detection.x1 < detection.x2 <= FRAME.shape[1]
        assert 0 <= detection.y1 < detection.y2 <= FRAME.shape[0]
        assert 0.0 <= detection.confidence <= 1.0


def test_script_can_simulate_target_loss_and_reset() -> None:
    target = Detection("wood", 1, 2, 10, 12, 0.9)
    detector = MockResourceDetector(script=((target,), (),))

    assert detector.detect(FRAME) == [target]
    assert detector.detect(FRAME) == []
    assert detector.detect(FRAME) == []

    detector.reset()
    assert detector.frame_index == 0
    assert detector.detect(FRAME) == [target]


def test_mock_detector_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="未知 Mock 场景"):
        MockResourceDetector("unknown")
    with pytest.raises(ValueError, match="visible_frames"):
        MockResourceDetector(visible_frames=-1)
    with pytest.raises(ValueError, match="BGR"):
        MockResourceDetector().detect(np.zeros((10, 10), dtype=np.uint8))
