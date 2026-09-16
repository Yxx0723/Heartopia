from __future__ import annotations

import math
from pathlib import Path

import pytest

from navigation.navigator import RouteNavigator
from navigation.route import Route, RouteAction, RouteError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_load_example_route() -> None:
    route = Route.from_yaml(PROJECT_ROOT / "config" / "routes" / "test_route.yaml")

    assert len(route.steps) == 4
    assert route.steps[0].action is RouteAction.MOVE_FORWARD
    assert route.steps[1].action is RouteAction.TURN_RIGHT
    assert route.steps[1].mouse_dx == 350.0
    assert route.loop is True


def test_navigator_progresses_by_tick_and_preserves_remaining_time() -> None:
    clock = FakeClock()
    route = Route.from_mapping(
        {
            "route": [
                {"action": "move_forward", "duration": 3.0},
                {"action": "turn_left", "pixels": 100},
                {"action": "wait", "duration": 1.0},
            ],
            "loop": True,
        }
    )
    navigator = RouteNavigator(route, clock=clock)
    navigator.start()

    command = navigator.update()
    assert command is not None
    assert command.action is RouteAction.MOVE_FORWARD
    assert command.remaining == pytest.approx(3.0)

    clock.value = 2.0
    command = navigator.update()
    assert command is not None
    assert command.elapsed == pytest.approx(2.0)
    assert command.remaining == pytest.approx(1.0)

    navigator.interrupt()
    clock.value = 12.0
    assert navigator.update() is None
    navigator.resume()
    command = navigator.update()
    assert command is not None
    assert command.action is RouteAction.MOVE_FORWARD
    assert command.remaining == pytest.approx(1.0)

    clock.value = 13.0
    command = navigator.update()
    assert command is not None
    assert command.action is RouteAction.TURN_LEFT
    assert command.mouse_dx == -100.0
    assert navigator.route_index == 2


def test_non_loop_route_stops_after_last_timed_action() -> None:
    clock = FakeClock()
    route = Route.from_mapping(
        {"route": [{"action": "wait", "duration": 1.0}], "loop": False}
    )
    navigator = RouteNavigator(route, clock=clock)
    navigator.start()

    assert navigator.update() is not None
    clock.value = 1.0
    assert navigator.update() is None
    assert navigator.active is False


def test_invalid_route_configuration_is_rejected() -> None:
    with pytest.raises(RouteError, match="至少需要"):
        Route.from_mapping({"route": []})
    with pytest.raises(RouteError, match="不支持"):
        Route.from_mapping({"route": [{"action": "jump", "duration": 1}]})
    with pytest.raises(RouteError, match="duration"):
        Route.from_mapping({"route": [{"action": "wait", "duration": 0}]})
    with pytest.raises(RouteError, match="mouse_dx/pixels"):
        Route.from_mapping({"route": [{"action": "turn_right"}]})
    with pytest.raises(RouteError, match="duration"):
        Route.from_mapping(
            {"route": [{"action": "wait", "duration": math.inf}]}
        )
