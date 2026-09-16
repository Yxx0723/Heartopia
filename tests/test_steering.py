from __future__ import annotations

import pytest

from core.models import Detection
from navigation.steering import Steering


def test_steering_enters_deadzone_without_mouse_move() -> None:
    steering = Steering(kp=0.25, center_deadzone=0.06)
    target = Detection("wood", 96, 40, 106, 60, 0.9)

    command = steering.compute(target, 200, 100)

    assert command.aligned is True
    assert command.error_x == pytest.approx(1.0)
    assert command.mouse_dx == 0


def test_steering_applies_proportional_control_and_clamps() -> None:
    steering = Steering(kp=0.5, center_deadzone=0.01, max_mouse_dx=30)
    target_right = Detection("wood", 180, 40, 200, 60, 0.9)
    target_left = Detection("wood", 0, 40, 20, 60, 0.9)

    right_command = steering.compute(target_right, 200, 100)
    left_command = steering.compute(target_left, 200, 100)

    assert right_command.aligned is False
    assert right_command.mouse_dx == 30
    assert left_command.mouse_dx == -30


def test_steering_rejects_invalid_frame_size_and_configuration() -> None:
    with pytest.raises(ValueError, match="kp"):
        Steering(kp=float("inf"))
    with pytest.raises(ValueError, match="center_deadzone"):
        Steering(center_deadzone=0.5)
    with pytest.raises(ValueError, match="frame_width"):
        Steering().compute(Detection("wood", 0, 0, 1, 1, 0.9), 0, 100)
