from __future__ import annotations

from enum import Enum

import numpy as np
import pytest

from debug.overlay import DebugOverlay, OverlayError


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.windows: list[tuple[str, np.ndarray]] = []
        self.destroyed: list[str] = []
        self.wait_delays: list[int] = []

    def putText(self, image, text, *_args):
        self.texts.append(text)
        return image

    def imshow(self, name: str, image: np.ndarray) -> None:
        self.windows.append((name, image.copy()))

    def waitKey(self, delay_ms: int) -> int:
        self.wait_delays.append(delay_ms)
        return 27

    def destroyWindow(self, name: str) -> None:
        self.destroyed.append(name)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


class State(Enum):
    PATROL = 1


def test_render_draws_fps_window_size_and_state() -> None:
    cv2 = FakeCv2()
    overlay = DebugOverlay(cv2_module=cv2)
    frame = np.zeros((90, 160, 3), dtype=np.uint8)

    result = overlay.render(frame, State.PATROL, fps=10.0)

    assert result.shape == (90, 160, 3)
    assert cv2.texts == ["FPS: 10.0", "WINDOW: 160x90", "STATE: PATROL"]
    assert cv2.windows[0][0] == "Heartopia Bot Debug"


def test_render_estimates_fps_when_not_provided() -> None:
    cv2 = FakeCv2()
    clock = FakeClock()
    overlay = DebugOverlay(cv2_module=cv2, clock=clock)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    overlay.render(frame, "INIT")
    clock.value += 0.5
    overlay.render(frame, "PATROL")

    assert overlay.fps == pytest.approx(2.0)
    assert cv2.texts[-3:] == ["FPS: 2.0", "WINDOW: 20x10", "STATE: PATROL"]


def test_render_can_draw_extended_runtime_details() -> None:
    cv2 = FakeCv2()
    overlay = DebugOverlay(cv2_module=cv2)
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    overlay.render(frame, "APPROACH_TARGET", fps=10.0, details={"prompt": False})

    assert cv2.texts[-1] == "PROMPT: False"


def test_poll_events_and_close_are_forwarded() -> None:
    cv2 = FakeCv2()
    overlay = DebugOverlay(cv2_module=cv2)

    assert overlay.poll_events(5) == 27
    overlay.close()
    overlay.close()

    assert cv2.wait_delays == [5]
    assert cv2.destroyed == ["Heartopia Bot Debug"]


def test_disabled_overlay_does_not_create_or_show_window() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    overlay = DebugOverlay(enabled=False)

    result = overlay.render(frame, "INIT", fps=1.0)

    assert result is frame
    assert overlay.poll_events() is None


def test_overlay_rejects_invalid_frame() -> None:
    overlay = DebugOverlay(cv2_module=FakeCv2())

    with pytest.raises(OverlayError, match="H x W x 3"):
        overlay.render(np.zeros((10, 20), dtype=np.uint8), "INIT")
