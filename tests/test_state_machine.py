from __future__ import annotations

import numpy as np
import pytest

from core.context import BotContext
from core.models import Detection, Perception
from core.state_machine import Action, BotState, StateMachine
from vision.target_selector import TargetSelector
from vision.resource_detector import MockResourceDetector


FRAME = np.zeros((100, 200, 3), dtype=np.uint8)
TARGET = Detection("wood", 80, 35, 120, 75, 0.9)


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def perception(resources=None, prompt=False, frame_size=None) -> Perception:
    return Perception(
        resources=list(resources or []),
        interaction_prompt=prompt,
        motion_score=1.0,
        frame_size=frame_size,
    )


def test_mock_detector_drives_main_collection_flow() -> None:
    detector = MockResourceDetector(
        script=((TARGET,), (TARGET,), (TARGET,), (),),
    )
    machine = StateMachine()

    assert machine.update(perception(detector.detect(FRAME))).kind == "patrol"
    assert machine.state is BotState.PATROL

    assert machine.update(perception(detector.detect(FRAME))).kind == "stop"
    assert machine.state is BotState.TARGET_FOUND

    assert machine.update(perception(detector.detect(FRAME))).kind == "align_target"
    assert machine.state is BotState.ALIGN_TARGET

    assert machine.update(perception([TARGET])).kind == "approach_target"
    assert machine.state is BotState.APPROACH_TARGET

    assert machine.update(perception([TARGET], prompt=True)).kind == "stop"
    assert machine.state is BotState.INTERACT

    assert machine.update(perception([TARGET], prompt=True)) == Action(
        "interact", key="f"
    )
    assert machine.state is BotState.VERIFY

    assert machine.update(perception([])).kind == "patrol"
    assert machine.state is BotState.PATROL
    assert machine.context.total_collected == 1


def test_approach_timeout_enters_recovery_then_returns_to_patrol() -> None:
    clock = FakeClock()
    machine = StateMachine(clock=clock, target_timeout_seconds=1.0)

    machine.update(perception())
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    assert machine.state is BotState.APPROACH_TARGET

    clock.value += 1.0
    assert machine.update(perception([TARGET])).kind == "stop"
    assert machine.state is BotState.RECOVERY
    assert machine.context.recovery_attempts == 1

    assert machine.update(perception()).kind == "recovery"
    assert machine.state is BotState.RECOVERY
    assert machine.complete_recovery().kind == "patrol"
    assert machine.state is BotState.PATROL


def test_lost_target_returns_to_patrol_after_configured_frames() -> None:
    machine = StateMachine(lost_target_frames=2)
    machine.update(perception())
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    assert machine.state is BotState.APPROACH_TARGET

    assert machine.update(perception()).kind == "hold_target"
    assert machine.state is BotState.APPROACH_TARGET
    assert machine.update(perception()).kind == "patrol"
    assert machine.state is BotState.PATROL


def test_verify_retries_then_abandons_target() -> None:
    clock = FakeClock()
    machine = StateMachine(
        clock=clock,
        verify_timeout_seconds=1.0,
        interaction_retry_count=1,
    )
    machine.update(perception())
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET]))
    machine.update(perception([TARGET], prompt=True))
    machine.update(perception([TARGET], prompt=True))
    assert machine.state is BotState.VERIFY

    clock.value += 1.0
    assert machine.update(perception([TARGET], prompt=True)).kind == "retry_interact"
    assert machine.state is BotState.INTERACT
    machine.update(perception([TARGET], prompt=True))
    assert machine.state is BotState.VERIFY

    clock.value += 1.0
    assert machine.update(perception([TARGET], prompt=True)).kind == "patrol"
    assert machine.state is BotState.PATROL


def test_pause_resume_and_stop() -> None:
    machine = StateMachine(context=BotContext())
    machine.update(perception())
    assert machine.state is BotState.PATROL

    assert machine.pause().kind == "pause"
    assert machine.state is BotState.PAUSED
    machine.update(perception([TARGET]))
    assert machine.state is BotState.PAUSED

    machine.resume()
    assert machine.state is BotState.PATROL
    assert machine.get_action().kind == "idle"

    assert machine.stop().kind == "stop"
    assert machine.state is BotState.STOPPED
    assert machine.update(perception([TARGET])).kind == "stop"


def test_state_machine_rejects_invalid_timing_configuration() -> None:
    with pytest.raises(ValueError, match="target_timeout_seconds"):
        StateMachine(target_timeout_seconds=0)
    with pytest.raises(ValueError, match="lost_target_frames"):
        StateMachine(lost_target_frames=0)


def test_state_machine_uses_injected_target_selector_when_frame_size_exists() -> None:
    selector = TargetSelector()
    low_confidence_center = Detection("wood", 90, 40, 110, 60, 0.75)
    high_confidence_edge = Detection("wood", 0, 40, 20, 60, 0.99)
    machine = StateMachine(target_selector=selector)

    machine.update(perception())
    machine.update(
        perception(
            [low_confidence_center, high_confidence_edge],
            frame_size=(200, 100),
        )
    )

    assert machine.context.current_target == low_confidence_center


def test_state_machine_tracks_nearest_same_type_detection() -> None:
    first = Detection("wood", 80, 40, 100, 60, 0.9)
    nearest = Detection("wood", 84, 40, 104, 60, 0.70)
    farther = Detection("wood", 150, 40, 170, 60, 0.99)
    machine = StateMachine()

    machine.update(perception())
    machine.update(perception([first]))
    machine.update(perception([first]))
    machine.update(perception([nearest, farther]))

    assert machine.context.current_target == nearest


def test_blocking_ui_pauses_state_machine_until_user_resumes() -> None:
    machine = StateMachine()
    machine.update(perception())

    action = machine.update(Perception([], False, 0.0, inventory_full=True))

    assert action.kind == "pause"
    assert machine.state is BotState.PAUSED
    machine.resume()
    assert machine.state is BotState.PATROL
