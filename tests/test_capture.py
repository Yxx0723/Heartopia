from __future__ import annotations

import numpy as np

from capture.capture import WindowCapture


class FakeWindow:
    def get_rect(self) -> tuple[int, int, int, int]:
        return (10, 20, 14, 23)


class FakeCaptureBackend:
    def __init__(self) -> None:
        self.regions: list[dict[str, int]] = []

    def grab(self, monitor: dict[str, int]) -> np.ndarray:
        self.regions.append(monitor)
        # 模拟 mss 的 BGRA 输出。
        return np.full((3, 4, 4), 17, dtype=np.uint8)


def test_capture_returns_bgr_array_for_repeated_frames() -> None:
    backend = FakeCaptureBackend()
    capture = WindowCapture(FakeWindow(), backend=backend)  # type: ignore[arg-type]

    first = capture.capture()
    second = capture.capture()
    third = capture.capture()

    assert first.shape == (3, 4, 3)
    assert first.dtype == np.uint8
    assert first.flags["C_CONTIGUOUS"]
    assert np.array_equal(first, second)
    assert np.array_equal(second, third)
    assert backend.regions == [
        {"left": 10, "top": 20, "width": 4, "height": 3},
        {"left": 10, "top": 20, "width": 4, "height": 3},
        {"left": 10, "top": 20, "width": 4, "height": 3},
    ]


def test_capture_discards_alpha_channel() -> None:
    class ColorBackend:
        def grab(self, _monitor: dict[str, int]) -> np.ndarray:
            return np.array([[[1, 2, 3, 255]]], dtype=np.uint8)

    capture = WindowCapture(FakeWindow(), backend=ColorBackend())  # type: ignore[arg-type]

    frame = capture.capture()

    assert frame.tolist() == [[[1, 2, 3]]]
