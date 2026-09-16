from __future__ import annotations

import numpy as np
import pytest

from vision.motion_detector import MotionDetector


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_static_moving_scene_becomes_stuck_after_threshold() -> None:
    clock = FakeClock()
    detector = MotionDetector(stuck_after_seconds=2.0, clock=clock)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    first = detector.update(frame, moving=True)
    clock.value = 1.0
    second = detector.update(frame, moving=True)
    clock.value = 2.0
    third = detector.update(frame, moving=True)

    assert first.stuck is False
    assert second.stuck is False
    assert third.stuck is True
    assert third.score == pytest.approx(0.0)


def test_motion_or_not_moving_resets_stuck_timer() -> None:
    clock = FakeClock()
    detector = MotionDetector(stuck_after_seconds=1.0, clock=clock)
    static = np.zeros((100, 120, 3), dtype=np.uint8)
    changed = np.full((100, 120, 3), 255, dtype=np.uint8)

    detector.update(static, moving=True)
    clock.value = 0.5
    detector.update(static, moving=True)
    clock.value = 0.6
    moving_result = detector.update(changed, moving=True)
    assert moving_result.stuck is False
    assert moving_result.score > 0.01

    clock.value = 2.0
    not_moving_result = detector.update(changed, moving=False)
    assert not_moving_result.stuck is False
    assert not_moving_result.low_motion_duration == 0.0


def test_reset_and_invalid_configuration() -> None:
    detector = MotionDetector()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    detector.update(frame, moving=True)
    detector.reset()
    assert detector.update(frame, moving=True).low_motion_duration == 0.0

    with pytest.raises(ValueError, match="stuck_after_seconds"):
        MotionDetector(stuck_after_seconds=0)
    with pytest.raises(ValueError, match="BGR"):
        detector.update(np.zeros((20, 20), dtype=np.uint8), moving=True)
