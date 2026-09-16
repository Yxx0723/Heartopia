from __future__ import annotations

import pytest

from core.safety import SafetyGuard


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class FakeWindow:
    def __init__(self) -> None:
        self.foreground = True

    def is_foreground(self) -> bool:
        return self.foreground


def test_safety_requires_start_and_foreground() -> None:
    clock = FakeClock()
    window = FakeWindow()
    guard = SafetyGuard(window, max_runtime_minutes=1, clock=clock)

    assert guard.can_control() is False
    guard.start()
    assert guard.can_control() is True
    window.foreground = False
    assert guard.can_control() is False
    window.foreground = True
    assert guard.can_control() is True


def test_safety_pause_and_runtime_expiration() -> None:
    clock = FakeClock()
    guard = SafetyGuard(FakeWindow(), max_runtime_minutes=1, clock=clock)
    guard.start()
    guard.pause()
    assert guard.can_control() is False
    guard.resume()
    assert guard.can_control() is True
    clock.value = 60.0
    assert guard.can_control() is False
    assert guard.expired is True


def test_safety_rejects_invalid_runtime() -> None:
    with pytest.raises(ValueError, match="max_runtime_minutes"):
        SafetyGuard(FakeWindow(), max_runtime_minutes=0)
