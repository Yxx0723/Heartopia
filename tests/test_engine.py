from __future__ import annotations

import numpy as np

from actions.controller import InputController
from core.engine import BotEngine
from core.safety import SafetyGuard
from debug.overlay import DebugOverlay
from navigation.navigator import RouteNavigator
from navigation.route import Route
from vision.motion_detector import MotionDetector
from vision.prompt_detector import PromptDetector
from vision.resource_detector import MockResourceDetector


class FakeCapture:
    def __init__(self) -> None:
        self.frame = np.zeros((60, 80, 3), dtype=np.uint8)
        self.closed = False

    def capture(self) -> np.ndarray:
        return self.frame.copy()

    def close(self) -> None:
        self.closed = True


class FakeWindow:
    def __init__(self, foreground: bool = True) -> None:
        self.foreground = foreground

    def is_foreground(self) -> bool:
        return self.foreground


class FakeOverlay:
    def __init__(self) -> None:
        self.rendered = 0
        self.closed = False

    def render(self, frame, state, fps=None, details=None):
        self.rendered += 1
        return frame

    def close(self) -> None:
        self.closed = True


def build_engine(window: FakeWindow | None = None) -> BotEngine:
    route = Route.from_mapping(
        {"route": [{"action": "wait", "duration": 1.0}], "loop": True}
    )
    window = window or FakeWindow()
    return BotEngine(
        capture=FakeCapture(),
        resource_detector=MockResourceDetector(),
        prompt_detector=PromptDetector(),
        motion_detector=MotionDetector(),
        route_navigator=RouteNavigator(route),
        input_controller=InputController(dry_run=True),
        safety_guard=SafetyGuard(window),
        overlay=FakeOverlay(),
    )


def test_engine_update_runs_perception_decision_route_and_overlay() -> None:
    engine = build_engine()
    engine.safety_guard.start(0.0)
    engine.route_navigator.start(0.0)

    snapshot = engine.update(0.0)

    assert snapshot.frame.shape == (60, 80, 3)
    assert snapshot.perception.frame_size == (80, 60)
    assert snapshot.action.kind == "patrol"
    assert snapshot.route_command is not None
    assert snapshot.route_command.action.value == "wait"
    assert engine.overlay.rendered == 1
    engine.stop()


def test_engine_stop_releases_and_closes_resources() -> None:
    engine = build_engine()
    engine.safety_guard.start(0.0)
    engine.route_navigator.start(0.0)
    engine.update(0.0)

    engine.stop()

    assert engine.running is False
    assert engine.input_controller.pressed_keys == frozenset()
    assert engine.capture.closed is True
    assert engine.overlay.closed is True
