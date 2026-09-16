from __future__ import annotations

import pytest

from navigation.recovery import RecoveryController, RecoveryError


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_recovery_runs_safe_sequence_by_ticks() -> None:
    clock = FakeClock()
    recovery = RecoveryController(clock=clock, camera_turn_dx=100)
    recovery.start(attempt=1)

    release = recovery.update()
    assert release is not None
    assert release.kind == "release_all"

    backward = recovery.update()
    assert backward is not None
    assert backward.kind == "key_hold"
    assert backward.key == "s"
    assert backward.remaining == pytest.approx(0.5)

    clock.value = 0.5
    strafe = recovery.update()
    assert strafe is not None
    assert strafe.key == "a"

    clock.value = 1.0
    turn = recovery.update()
    assert turn is not None
    assert turn.kind == "mouse_move"
    assert turn.mouse_dx == 100

    forward = recovery.update()
    assert forward is not None
    assert forward.key == "w"
    clock.value = 1.5
    assert recovery.update() is None
    assert recovery.active is False


def test_even_recovery_attempt_alternates_direction() -> None:
    recovery = RecoveryController()
    recovery.start(attempt=2, now=0.0)
    recovery.update(now=0.0)
    recovery.update(now=0.0)
    strafe = recovery.update(now=0.5)
    assert strafe is not None
    assert strafe.key == "d"
    turn = recovery.update(now=1.0)
    assert turn is not None
    assert turn.mouse_dx < 0


def test_recovery_rejects_attempt_over_limit() -> None:
    recovery = RecoveryController(max_attempts=3)

    with pytest.raises(RecoveryError, match="上限"):
        recovery.start(attempt=4)
