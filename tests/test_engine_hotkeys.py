from core.state import BotState

from tests.test_engine import FakeWindow, build_engine


def test_unfocused_window_pauses_progress_and_route() -> None:
    window = FakeWindow(foreground=False)
    engine = build_engine(window)
    engine.safety_guard.start(0.0)
    engine.route_navigator.start(0.0)

    snapshot = engine.update(0.0)

    assert snapshot.action.kind == "pause"
    assert engine.state_machine.state is BotState.INIT
    assert engine.route_navigator.paused is True
    engine.stop()


def test_pause_hotkey_releases_input_and_resume_restores_patrol() -> None:
    engine = build_engine()
    engine.safety_guard.start(0.0)
    engine.route_navigator.start(0.0)
    engine.update(0.0)
    engine.input_controller.key_down("w")

    engine._apply_hotkey_events(("pause",), 0.0)
    assert engine.state_machine.state is BotState.PAUSED
    assert engine.input_controller.pressed_keys == frozenset()
    assert engine.safety_guard.paused is True

    engine._apply_hotkey_events(("resume",), 1.0)
    assert engine.state_machine.state is BotState.PATROL
    assert engine.safety_guard.paused is False
    engine.stop()


def test_stop_hotkey_stops_loop_and_releases_input() -> None:
    engine = build_engine()
    engine.safety_guard.start(0.0)
    engine.route_navigator.start(0.0)
    engine.input_controller.key_down("w")
    engine._running = True

    engine._apply_hotkey_events(("stop",), 0.0)

    assert engine.running is False
    assert engine.state_machine.state is BotState.STOPPED
    assert engine.input_controller.pressed_keys == frozenset()
    engine.stop()
