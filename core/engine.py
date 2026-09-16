"""视觉、决策、导航和输入之间的主循环编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from actions.controller import InputController
from capture.capture import WindowCapture
from capture.window import GameWindow
from core.config import Config, load_config
from core.models import Perception
from core.safety import SafetyGuard
from core.state import BotState
from core.state_machine import Action, StateMachine
from debug.overlay import DebugOverlay
from debug.recorder import FailureRecorder
from debug.screenshot import ScreenshotSaver
from navigation.navigator import RouteCommand, RouteNavigator
from navigation.recovery import RecoveryCommand, RecoveryController
from navigation.route import Route
from navigation.steering import Steering
from vision.motion_detector import MotionDetector, MotionResult
from vision.prompt_detector import PromptDetector
from vision.resource_detector import (
    MockResourceDetector,
    OpenCVResourceDetector,
    ResourceDetector,
)
from vision.target_selector import TargetSelector
from vision.ui_detector import UiDetector


from loguru import logger


class EngineError(RuntimeError):
    """引擎装配或运行失败时抛出。"""


class CaptureLike(Protocol):
    def capture(self) -> np.ndarray: ...

    def close(self) -> None: ...


class PromptDetectorLike(Protocol):
    def detect(self, frame: np.ndarray) -> bool: ...


class OverlayLike(Protocol):
    def render(
        self,
        frame: np.ndarray,
        state: Any,
        fps: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EngineSnapshot:
    """一次引擎 tick 的结果，便于调试和集成测试。"""

    frame: np.ndarray
    perception: Perception
    action: Action
    route_command: RouteCommand | None
    motion: MotionResult


class BotEngine:
    """驱动一次完整的视觉采集 tick。"""

    def __init__(
        self,
        *,
        config: Config | Mapping[str, Any] | None = None,
        config_path: str | Path | None = None,
        capture: CaptureLike | None = None,
        resource_detector: ResourceDetector | None = None,
        prompt_detector: PromptDetectorLike | None = None,
        motion_detector: MotionDetector | None = None,
        state_machine: StateMachine | None = None,
        route_navigator: RouteNavigator | None = None,
        input_controller: InputController | None = None,
        safety_guard: SafetyGuard | Any | None = None,
        overlay: OverlayLike | None = None,
        steering: Steering | None = None,
        recovery_controller: RecoveryController | None = None,
        failure_recorder: FailureRecorder | None = None,
        screenshot_saver: ScreenshotSaver | None = None,
        hotkeys: Any | None = None,
        ui_detector: UiDetector | None = None,
        window: GameWindow | Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = self._resolve_config(config, config_path)
        self._clock = clock
        self._sleeper = sleeper
        self._window = window

        if capture is None:
            self._window = self._window or GameWindow(
                str(self.config.get("window.title_contains", "心动小镇"))
            )
            capture = WindowCapture(self._window)
        self.capture = capture

        self.resource_detector = resource_detector or self._build_resource_detector()
        self.prompt_detector = prompt_detector or self._build_prompt_detector()
        self.motion_detector = motion_detector or MotionDetector(
            motion_threshold=float(self.config.get("motion.motion_threshold", 0.01)),
            stuck_after_seconds=float(
                self.config.get("motion.stuck_after_seconds", 2.0)
            ),
            clock=clock,
        )
        self.steering = steering or Steering(
            kp=float(self.config.get("steering.kp", 0.25)),
            center_deadzone=float(
                self.config.get("vision.center_deadzone", 0.06)
            ),
            max_mouse_dx=int(self.config.get("steering.max_mouse_dx", 100)),
        )
        target_selector = TargetSelector()
        self.state_machine = state_machine or StateMachine(
            clock=clock,
            target_timeout_seconds=float(
                self.config.get("navigation.target_timeout_seconds", 8.0)
            ),
            lost_target_frames=int(self.config.get("navigation.lost_target_frames", 5)),
            interaction_retry_count=int(
                self.config.get("interaction.retry_count", 2)
            ),
            target_selector=target_selector,
        )
        self.route_navigator = route_navigator or self._build_route_navigator(clock)
        self.input_controller = input_controller or InputController(
            dry_run=bool(self.config.get("control.dry_run", True))
        )
        if safety_guard is None:
            self._window = self._window or GameWindow(
                str(self.config.get("window.title_contains", "心动小镇"))
            )
            safety_guard = SafetyGuard(
                self._window,
                pause_when_unfocused=bool(
                    self.config.get("safety.pause_when_unfocused", True)
                ),
                max_runtime_minutes=float(
                    self.config.get("safety.max_runtime_minutes", 60)
                ),
                clock=clock,
            )
        self.safety_guard = safety_guard
        self.overlay = overlay or DebugOverlay(
            enabled=bool(self.config.get("debug.overlay", True))
        )
        self.recovery_controller = recovery_controller or RecoveryController(
            max_attempts=int(self.config.get("recovery.max_attempts", 3)),
            clock=clock,
        )
        failure_directory = self.config.get("debug.failure_dir")
        if failure_directory is None:
            failure_directory = Path(__file__).resolve().parents[1] / "debug" / "failures"
        self.failure_recorder = failure_recorder or FailureRecorder(failure_directory)
        screenshot_directory = self.config.get("debug.screenshot_dir")
        if screenshot_directory is None:
            screenshot_directory = Path(__file__).resolve().parents[1] / "dataset" / "images"
        self.screenshot_saver = screenshot_saver or ScreenshotSaver(screenshot_directory)
        self.hotkeys = hotkeys
        self.ui_detector = ui_detector or self._build_ui_detector()
        self._record_frames = bool(self.config.get("debug.record_frames", False))
        self._record_interval = max(
            0.0,
            float(self.config.get("debug.record_interval_seconds", 1.0)),
        )
        self._last_record_time: float | None = None
        self._running = False
        self._last_route_key: str | None = None
        self._last_recovery_key: str | None = None
        self._fps = 0.0
        self._last_tick_time: float | None = None
        self._last_frame: np.ndarray | None = None
        self._last_failure_key: tuple[str, str] | None = None

    @property
    def running(self) -> bool:
        """返回主循环是否运行。"""
        return self._running

    def start(self, max_updates: int | None = None) -> None:
        """启动主循环；``max_updates`` 仅用于测试或短时调试。"""
        self._running = True
        logger.info("ENGINE START dry_run={}", self.input_controller.dry_run)
        self.safety_guard.start(self._clock())
        self.route_navigator.start(self._clock())
        updates = 0
        interval = 1.0 / float(self.config.get("capture.fps", 10))
        try:
            while self._running and (
                max_updates is None or updates < max_updates
            ):
                tick_started = self._clock()
                self.update(tick_started)
                updates += 1
                elapsed = max(0.0, self._clock() - tick_started)
                remaining = interval - elapsed
                if remaining > 0 and self._running:
                    self._sleeper(remaining)
        except Exception:
            logger.exception("ENGINE FAILED")
            raise
        finally:
            self.stop()

    def update(self, now: float | None = None) -> EngineSnapshot:
        """执行一次截图、感知、决策、导航和动作 tick。"""
        current_time = self._clock() if now is None else now
        hotkey_events = self._poll_hotkeys()
        self._apply_hotkey_events(hotkey_events, current_time)
        self.input_controller.update()
        frame = self.capture.capture()
        self._last_frame = frame.copy()
        if "screenshot" in hotkey_events:
            self.save_screenshot()
        if self._record_frames and (
            self._last_record_time is None
            or current_time - self._last_record_time >= self._record_interval
        ):
            self.screenshot_saver.save(frame, prefix="record")
            self._last_record_time = current_time
        previous_state = self.state_machine.state
        resources = self.resource_detector.detect(frame)
        interaction_prompt = self.prompt_detector.detect(frame)
        ui_state = self.ui_detector.detect(frame)
        moving = self.state_machine.state in (
            BotState.PATROL,
            BotState.APPROACH_TARGET,
        )
        motion = self.motion_detector.update(
            frame,
            moving=moving,
            now=current_time,
        )
        perception = Perception(
            resources=resources,
            interaction_prompt=interaction_prompt,
            motion_score=motion.score,
            inventory_full=ui_state.inventory_full,
            abnormal_ui=ui_state.abnormal_ui,
            frame_size=(frame.shape[1], frame.shape[0]),
        )

        can_control = self.safety_guard.can_control(current_time)
        if not can_control:
            if self.route_navigator.active and not self.route_navigator.paused:
                self.route_navigator.pause(current_time)
            if self.recovery_controller.active:
                self.recovery_controller.abort()
            action = Action(
                "stop" if getattr(self.safety_guard, "expired", False) else "pause"
            )
            route_command = None
            recovery_command = None
        else:
            if motion.stuck and self.state_machine.state is BotState.APPROACH_TARGET:
                action = self.state_machine.request_recovery(current_time)
            else:
                action = self.state_machine.update(perception, current_time)

            route_command = self._update_route(current_time)
            action, recovery_command = self._update_recovery(action, current_time)
        if self.state_machine.state is not previous_state:
            logger.info("STATE {} -> {}", previous_state.name, self.state_machine.state.name)
        if resources:
            best = max(resources, key=lambda item: item.confidence)
            logger.debug(
                "TARGET {} confidence={:.3f}",
                best.resource_type,
                best.confidence,
            )
        if interaction_prompt:
            logger.debug("PROMPT detected")
        if ui_state.inventory_full:
            logger.warning("INVENTORY FULL detected")
        if ui_state.abnormal_ui:
            logger.warning("ABNORMAL UI detected label={}", ui_state.matched_label)
        if motion.stuck:
            logger.warning("STUCK detected")
            self._record_failure(frame, "stuck")
        if (
            previous_state is BotState.APPROACH_TARGET
            and self.state_machine.state is BotState.RECOVERY
            and not motion.stuck
        ):
            logger.warning("APPROACH TIMEOUT")
            self._record_failure(frame, "approach_timeout")
        if (
            previous_state is BotState.APPROACH_TARGET
            and self.state_machine.state is BotState.PATROL
            and not resources
            and not interaction_prompt
        ):
            logger.warning("TARGET LOST")
            self._record_failure(frame, "target_lost")
        if action.kind in ("retry_interact",):
            logger.warning("INTERACT FAILED retry={}", self.state_machine.context.interaction_retries)
            self._record_failure(frame, "interact_failed")
        if can_control:
            self._dispatch(action, route_command, perception, recovery_command)
        else:
            self.input_controller.release_all()
            self._last_route_key = None
            self._last_recovery_key = None

        self._update_fps(current_time)
        self.overlay.render(
            frame,
            self.state_machine.state,
            fps=self._fps,
            details=self._overlay_details(perception, motion),
        )
        return EngineSnapshot(frame, perception, action, route_command, motion)

    def _overlay_details(
        self,
        perception: Perception,
        motion: MotionResult,
    ) -> dict[str, Any]:
        target = self.state_machine.context.current_target
        width = perception.frame_size[0] if perception.frame_size else 0
        height = perception.frame_size[1] if perception.frame_size else 0
        target_center = target.center if target else None
        return {
            "target": target.resource_type if target else "none",
            "confidence": f"{target.confidence:.2f}" if target else "-",
            "bbox": (
                f"({target.x1},{target.y1},{target.x2},{target.y2})"
                if target
                else "-"
            ),
            "target_center": target_center if target is not None else "-",
            "screen_center": (width // 2, height // 2) if width and height else "-",
            "prompt": perception.interaction_prompt,
            "motion": f"{motion.score:.4f}",
            "stuck": motion.stuck,
            "window_foreground": getattr(self.safety_guard, "window_foreground", "-"),
            "inventory_full": perception.inventory_full,
            "abnormal_ui": perception.abnormal_ui,
            "route": f"{self.route_navigator.route_index + 1}/{self.route_navigator.route_length}",
            "recovery": self.state_machine.context.recovery_attempts,
            "collected": self.state_machine.context.total_collected,
        }

    def _poll_hotkeys(self) -> tuple[str, ...]:
        if self.hotkeys is None:
            return ()
        events = self.hotkeys.poll()
        return tuple(str(event).strip().lower() for event in events)

    def _apply_hotkey_events(self, events: tuple[str, ...], now: float) -> None:
        """应用控制类热键；截图事件留到当前帧捕获后处理。"""
        for event in events:
            if event == "pause":
                self.state_machine.pause(now)
                self.safety_guard.pause()
                self.route_navigator.pause(now)
                self.input_controller.release_all()
            elif event == "resume":
                self.safety_guard.resume()
                self.state_machine.resume(now)
            elif event == "stop":
                self.input_controller.release_all()
                self.recovery_controller.abort()
                self.route_navigator.stop()
                self.state_machine.stop(now)
                self._running = False

    def stop(self) -> None:
        """停止引擎并释放所有输入和资源。"""
        self._running = False
        try:
            self.input_controller.release_all()
        except Exception:
            # 清理阶段仍要继续关闭其他资源，避免一个按键释放错误阻断整个
            # shutdown 流程；具体异常已记录，调用方不再被清理错误覆盖。
            logger.exception("INPUT CLEANUP FAILED")
        finally:
            self._last_route_key = None
            self._last_recovery_key = None
            self.recovery_controller.abort()
            self.route_navigator.stop()
            if self.state_machine.state is not BotState.STOPPED:
                self.state_machine.stop(self._clock())
            try:
                self.overlay.close()
            except Exception:
                logger.exception("OVERLAY CLEANUP FAILED")
            try:
                self.capture.close()
            except Exception:
                logger.exception("CAPTURE CLEANUP FAILED")

    def save_screenshot(self, prefix: str = "screenshot") -> Path:
        """保存最近一帧截图，供快捷键或调试调用。"""
        if self._last_frame is None:
            raise EngineError("当前没有可保存的截图")
        path = self.screenshot_saver.save(self._last_frame, prefix=prefix)
        logger.info("SCREENSHOT saved={}", path)
        return path

    def _record_failure(self, frame: np.ndarray, reason: str) -> None:
        if not bool(self.config.get("debug.save_failed_targets", True)):
            return
        state_name = self.state_machine.state.name
        key = (reason, state_name)
        if self._last_failure_key == key:
            return
        target = (
            self.state_machine.context.current_target
            or self.state_machine.context.previous_target
        )
        self.failure_recorder.save_failure(
            frame,
            state=self.state_machine.state,
            reason=reason,
            target=target,
        )
        self._last_failure_key = key

    def _update_route(self, now: float) -> RouteCommand | None:
        if self.state_machine.state is BotState.PATROL:
            if self.route_navigator.paused:
                self.route_navigator.resume(now)
            if not self.route_navigator.active:
                self.route_navigator.start(now)
            return self.route_navigator.update(now)

        if self.route_navigator.active and not self.route_navigator.paused:
            self.route_navigator.interrupt(now)
        return None

    def _update_recovery(
        self,
        action: Action,
        now: float,
    ) -> tuple[Action, RecoveryCommand | None]:
        if self.state_machine.state is not BotState.RECOVERY:
            return action, None
        if not self.recovery_controller.active:
            self.recovery_controller.start(
                self.state_machine.context.recovery_attempts,
                now,
            )
        command = self.recovery_controller.update(now)
        if command is None:
            return self.state_machine.complete_recovery(now), None
        return Action("recovery"), command

    def _dispatch(
        self,
        action: Action,
        route_command: RouteCommand | None,
        perception: Perception,
        recovery_command: RecoveryCommand | None,
    ) -> None:
        if action.kind in ("stop", "pause", "idle", "verify_wait", "hold_target"):
            if action.kind in ("stop", "pause"):
                self.input_controller.release_all()
            return
        if action.kind == "recovery":
            self._dispatch_recovery(recovery_command)
            return
        if route_command is not None and self.state_machine.state is BotState.PATROL:
            self._dispatch_route(route_command)
            return
        self._last_route_key = None

        target = self.state_machine.context.current_target
        if target is None or perception.frame_size is None:
            return
        frame_width, frame_height = perception.frame_size
        if action.kind in ("align_target", "approach_target"):
            command = self.steering.compute(target, frame_width, frame_height)
            if command.aligned and action.kind == "approach_target":
                self.input_controller.mouse_move_relative(0, 0)
                self.input_controller.press(
                    str(self.config.get("controls.forward", "w")),
                    duration=float(
                        self.config.get("navigation.approach_step_ms", 150)
                    )
                    / 1000.0,
                )
            elif not command.aligned:
                self.input_controller.release_all()
                self.input_controller.mouse_move_relative(command.mouse_dx, 0)
        elif action.kind in ("interact", "retry_interact"):
            self.input_controller.release_all()
            self.input_controller.press(
                str(self.config.get("controls.interact", "f")),
                duration=0.05,
            )

    def _dispatch_route(self, command: RouteCommand) -> None:
        key_map = {
            "move_forward": "forward",
            "move_backward": "backward",
            "strafe_left": "left",
            "strafe_right": "right",
        }
        if command.action.value in key_map:
            key_name = key_map[command.action.value]
            key = str(self.config.get(f"controls.{key_name}", key_name[0]))
            if self._last_route_key != key:
                self.input_controller.release_all()
                self.input_controller.key_down(key)
                self._last_route_key = key
        elif command.action is not None and command.action.value == "wait":
            self.input_controller.release_all()
            self._last_route_key = None
        elif command.mouse_dx:
            self.input_controller.release_all()
            self.input_controller.mouse_move_relative(command.mouse_dx, 0)
            self._last_route_key = None

    def _dispatch_recovery(self, command: RecoveryCommand | None) -> None:
        if command is None:
            self.input_controller.release_all()
            self._last_recovery_key = None
            return
        if command.kind == "release_all":
            self.input_controller.release_all()
            self._last_recovery_key = None
        elif command.kind == "key_hold" and command.key is not None:
            if self._last_recovery_key != command.key:
                self.input_controller.release_all()
                self.input_controller.key_down(command.key)
                self._last_recovery_key = command.key
        elif command.kind == "mouse_move":
            self.input_controller.release_all()
            self.input_controller.mouse_move_relative(command.mouse_dx, 0)
            self._last_recovery_key = None

    def _build_resource_detector(self) -> ResourceDetector:
        detector_name = str(self.config.get("vision.detector", "mock")).lower()
        if detector_name == "mock":
            return MockResourceDetector()
        if detector_name == "opencv":
            paths = self.config.get("vision.resource_template_paths", [])
            return OpenCVResourceDetector(
                resource_type=str(self.config.get("vision.resource_type", "wood")),
                template_paths=paths,
                threshold=float(self.config.get("vision.confidence_threshold", 0.70)),
            )
        raise EngineError(f"不支持的资源检测器: {detector_name}")

    def _build_prompt_detector(self) -> PromptDetector:
        return PromptDetector(
            template_paths=self.config.get("vision.prompt_template_paths", []),
            threshold=float(self.config.get("vision.prompt_threshold", 0.80)),
        )

    def _build_ui_detector(self) -> UiDetector:
        return UiDetector(
            inventory_full_template_paths=self.config.get(
                "ui.inventory_full_template_paths", []
            ),
            abnormal_template_paths=self.config.get(
                "ui.abnormal_template_paths", []
            ),
            inventory_threshold=float(
                self.config.get("ui.inventory_threshold", 0.85)
            ),
            abnormal_threshold=float(
                self.config.get("ui.abnormal_threshold", 0.85)
            ),
            roi=tuple(self.config.get("ui.roi", (0.0, 0.0, 1.0, 1.0))),
        )

    def _build_route_navigator(self, clock: Callable[[], float]) -> RouteNavigator:
        route_path = self.config.get("route.path")
        if route_path is None:
            route_path = Path(__file__).resolve().parents[1] / "config" / "routes" / "test_route.yaml"
        return RouteNavigator(Route.from_yaml(route_path), clock=clock)

    @staticmethod
    def _resolve_config(
        config: Config | Mapping[str, Any] | None,
        config_path: str | Path | None,
    ) -> Config:
        if isinstance(config, Config):
            return config
        if isinstance(config, Mapping):
            return Config(data=dict(config))
        path = config_path or Path(__file__).resolve().parents[1] / "config" / "default.yaml"
        return load_config(path)

    def _update_fps(self, now: float) -> None:
        if self._last_tick_time is not None:
            elapsed = now - self._last_tick_time
            if elapsed > 0:
                instantaneous = 1.0 / elapsed
                self._fps = instantaneous if self._fps == 0 else self._fps * 0.8 + instantaneous * 0.2
        self._last_tick_time = now
