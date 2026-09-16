from __future__ import annotations

import pytest

from actions.controller import InputController, InputError


class FakeInputBackend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail_on_key_up: set[str] = set()

    def keyDown(self, key: str) -> None:
        self.calls.append(("keyDown", key))

    def keyUp(self, key: str) -> None:
        self.calls.append(("keyUp", key))
        if key in self.fail_on_key_up:
            raise RuntimeError(f"failed to release {key}")

    def moveRel(self, x: int, y: int) -> None:
        self.calls.append(("moveRel", x, y))


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        return self.value


def test_dry_run_never_calls_backend_and_tracks_pressed_keys() -> None:
    controller = InputController(dry_run=True)

    controller.key_down("w")
    controller.key_down("w")
    controller.mouse_move_relative(20, -5)

    assert controller.pressed_keys == frozenset({"w"})
    controller.release_all()
    assert controller.pressed_keys == frozenset()


def test_real_backend_receives_idempotent_key_and_mouse_commands() -> None:
    backend = FakeInputBackend()
    controller = InputController(dry_run=False, backend=backend)

    controller.key_down("w")
    controller.key_down("w")
    controller.mouse_move_relative(20.8, -5.2)
    controller.key_up("w")
    controller.key_up("w")

    assert backend.calls == [
        ("keyDown", "w"),
        ("moveRel", 20, -5),
        ("keyUp", "w"),
    ]
    assert controller.pressed_keys == frozenset()


def test_press_is_non_blocking_and_update_releases_when_due() -> None:
    backend = FakeInputBackend()
    clock = FakeClock()
    controller = InputController(dry_run=False, backend=backend, clock=clock)

    controller.press("f", duration=0.2)
    assert backend.calls == [("keyDown", "f")]
    assert controller.pressed_keys == frozenset({"f"})

    clock.value += 0.19
    controller.update()
    assert controller.pressed_keys == frozenset({"f"})

    clock.value += 0.01
    controller.update()
    assert backend.calls == [("keyDown", "f"), ("keyUp", "f")]
    assert controller.pressed_keys == frozenset()


def test_release_all_attempts_every_pressed_key_and_clears_tracking() -> None:
    backend = FakeInputBackend()
    backend.fail_on_key_up.add("a")
    controller = InputController(dry_run=False, backend=backend)
    controller.key_down("a")
    controller.key_down("d")

    with pytest.raises(InputError, match="释放按键失败"):
        controller.release_all()

    assert backend.calls == [
        ("keyDown", "a"),
        ("keyDown", "d"),
        ("keyUp", "a"),
        ("keyUp", "d"),
    ]
    assert controller.pressed_keys == frozenset()


def test_invalid_key_and_duration_are_rejected() -> None:
    controller = InputController(dry_run=True)

    with pytest.raises(ValueError, match="按键不能为空"):
        controller.key_down(" ")
    with pytest.raises(ValueError, match="duration"):
        controller.press("f", duration=-0.1)
    with pytest.raises(ValueError, match="有限"):
        controller.mouse_move_relative(float("inf"), 0)
