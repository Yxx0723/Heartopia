"""基于 Perception 的自动采集有限状态机。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from core.context import BotContext
from core.models import Detection, Perception
from core.state import BotState


class TargetSelectorLike(Protocol):
    """StateMachine 使用的目标选择器最小接口。"""

    def select(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
    ) -> Detection | None: ...


@dataclass(frozen=True)
class Action:
    """状态机输出的高层动作，不直接执行键鼠输入。"""

    kind: str
    key: str | None = None
    duration: float = 0.0


class StateMachine:
    """根据视觉结果驱动采集流程。

    状态机只消费已经封装好的 ``Perception``，不依赖 OpenCV、窗口或输入
    后端。每次 ``update`` 最多推进一个主要状态，便于日志和单元测试追踪。
    """

    def __init__(
        self,
        *,
        context: BotContext | None = None,
        clock: Callable[[], float] = time.monotonic,
        target_timeout_seconds: float = 8.0,
        lost_target_frames: int = 5,
        interaction_retry_count: int = 2,
        verify_timeout_seconds: float = 2.0,
        recovery_max_attempts: int = 3,
        target_selector: TargetSelectorLike | None = None,
    ) -> None:
        if target_timeout_seconds <= 0 or not math.isfinite(target_timeout_seconds):
            raise ValueError("target_timeout_seconds 必须是正的有限数")
        if lost_target_frames <= 0:
            raise ValueError("lost_target_frames 必须大于零")
        if interaction_retry_count < 0:
            raise ValueError("interaction_retry_count 不能为负数")
        if verify_timeout_seconds <= 0 or not math.isfinite(verify_timeout_seconds):
            raise ValueError("verify_timeout_seconds 必须是正的有限数")
        if recovery_max_attempts <= 0:
            raise ValueError("recovery_max_attempts 必须大于零")

        self.context = context or BotContext()
        self._clock = clock
        self._target_timeout_seconds = target_timeout_seconds
        self._lost_target_frames_limit = lost_target_frames
        self._interaction_retry_count = interaction_retry_count
        self._verify_timeout_seconds = verify_timeout_seconds
        self._recovery_max_attempts = recovery_max_attempts
        self._target_selector = target_selector
        self._last_action = Action("idle")

    @property
    def state(self) -> BotState:
        """返回当前状态。"""
        return self.context.state

    def update(self, perception: Perception, now: float | None = None) -> Action:
        """消费一次感知结果并返回高层动作。"""
        current_time = self._clock() if now is None else now

        if self.state is BotState.STOPPED:
            return self._set_action(Action("stop"))
        if self.state is BotState.PAUSED:
            return self._set_action(Action("pause"))
        if perception.inventory_full or perception.abnormal_ui:
            self.context.resume_state = self.state
            self._transition(BotState.PAUSED, current_time)
            return self._set_action(Action("pause"))
        if self.state is BotState.INIT:
            self._transition(BotState.PATROL, current_time)
            return self._set_action(Action("patrol"))

        if self.state is BotState.PATROL:
            return self._update_patrol(perception, current_time)
        if self.state is BotState.TARGET_FOUND:
            return self._update_target_found(perception, current_time)
        if self.state is BotState.ALIGN_TARGET:
            return self._update_align(perception, current_time)
        if self.state is BotState.APPROACH_TARGET:
            return self._update_approach(perception, current_time)
        if self.state is BotState.INTERACT:
            return self._update_interact(current_time)
        if self.state is BotState.VERIFY:
            return self._update_verify(perception, current_time)
        if self.state is BotState.RECOVERY:
            return self._update_recovery(current_time)

        return self._set_action(Action("idle"))

    def get_action(self) -> Action:
        """返回最近一次 update 产生的动作。"""
        return self._last_action

    def pause(self, now: float | None = None) -> Action:
        """暂停当前流程并记住恢复状态。"""
        if self.state in (BotState.PAUSED, BotState.STOPPED):
            return self._set_action(Action("pause"))
        current_time = self._clock() if now is None else now
        self.context.resume_state = self.state
        self._transition(BotState.PAUSED, current_time)
        return self._set_action(Action("pause"))

    def resume(self, now: float | None = None) -> Action:
        """恢复暂停流程；没有可恢复状态时回到巡逻。"""
        if self.state is not BotState.PAUSED:
            return self._last_action
        current_time = self._clock() if now is None else now
        resume_state = self.context.resume_state or BotState.PATROL
        self.context.resume_state = None
        self._transition(resume_state, current_time)
        return self._set_action(Action("idle"))

    def stop(self, now: float | None = None) -> Action:
        """停止状态机并输出停止动作。"""
        current_time = self._clock() if now is None else now
        self.context.resume_state = None
        self._transition(BotState.STOPPED, current_time)
        return self._set_action(Action("stop"))

    def request_recovery(self, now: float | None = None) -> Action:
        """请求进入恢复状态，通常由 MotionDetector 的卡死结果触发。"""
        current_time = self._clock() if now is None else now
        if self.state is BotState.STOPPED:
            return self._set_action(Action("stop"))
        if self.state is BotState.RECOVERY:
            return self._set_action(Action("recovery"))
        if self.context.recovery_attempts >= self._recovery_max_attempts:
            self._abandon_target(current_time)
            return self._set_action(Action("patrol"))
        self.context.recovery_attempts += 1
        self._transition(BotState.RECOVERY, current_time)
        return self._set_action(Action("stop"))

    def complete_recovery(self, now: float | None = None) -> Action:
        """在恢复动作执行完毕后回到巡逻状态。"""
        if self.state is not BotState.RECOVERY:
            return self._last_action
        current_time = self._clock() if now is None else now
        self._abandon_target(current_time)
        return self._set_action(Action("patrol"))

    def _update_patrol(self, perception: Perception, now: float) -> Action:
        if not perception.resources:
            return self._set_action(Action("patrol"))

        self._set_target(
            self._select_target(perception.resources, perception.frame_size)
        )
        self._transition(BotState.TARGET_FOUND, now)
        return self._set_action(Action("stop"))

    def _update_target_found(self, perception: Perception, now: float) -> Action:
        if perception.resources:
            self._refresh_target(perception.resources, perception.frame_size)
            self._transition(BotState.ALIGN_TARGET, now)
            return self._set_action(Action("align_target"))

        self._abandon_target(now)
        return self._set_action(Action("patrol"))

    def _update_align(self, perception: Perception, now: float) -> Action:
        if not perception.resources:
            return self._handle_lost_target(now, "patrol")

        self._refresh_target(perception.resources, perception.frame_size)
        self.context.lost_target_frames = 0
        self._transition(BotState.APPROACH_TARGET, now)
        return self._set_action(Action("approach_target"))

    def _update_approach(self, perception: Perception, now: float) -> Action:
        if perception.interaction_prompt:
            self._transition(BotState.INTERACT, now)
            return self._set_action(Action("stop"))

        if perception.resources:
            self._refresh_target(perception.resources, perception.frame_size)
            self.context.lost_target_frames = 0
        else:
            lost_action = self._handle_lost_target(now, "patrol")
            if lost_action is not None:
                return lost_action

        if now - self.context.state_enter_time >= self._target_timeout_seconds:
            if self.context.recovery_attempts < self._recovery_max_attempts:
                self.context.recovery_attempts += 1
                self._transition(BotState.RECOVERY, now)
                return self._set_action(Action("stop"))
            self._abandon_target(now)
            return self._set_action(Action("patrol"))

        return self._set_action(Action("approach_target"))

    def _update_interact(self, now: float) -> Action:
        self._transition(BotState.VERIFY, now)
        return self._set_action(Action("interact", key="f"))

    def _update_verify(self, perception: Perception, now: float) -> Action:
        if not perception.interaction_prompt and not perception.resources:
            self.context.total_collected += 1
            self._abandon_target(now)
            return self._set_action(Action("patrol"))

        if now - self.context.state_enter_time < self._verify_timeout_seconds:
            return self._set_action(Action("verify_wait"))

        if self.context.interaction_retries < self._interaction_retry_count:
            self.context.interaction_retries += 1
            self._transition(BotState.INTERACT, now)
            return self._set_action(Action("retry_interact"))

        self._abandon_target(now)
        return self._set_action(Action("patrol"))

    def _update_recovery(self, now: float) -> Action:
        return self._set_action(Action("recovery"))

    def _handle_lost_target(self, now: float, terminal_action: str) -> Action | None:
        self.context.lost_target_frames += 1
        if self.context.lost_target_frames < self._lost_target_frames_limit:
            return self._set_action(Action("hold_target"))
        self._abandon_target(now)
        return self._set_action(Action(terminal_action))

    def _set_target(self, target: Detection) -> None:
        self.context.previous_target = self.context.current_target
        self.context.current_target = target
        self.context.lost_target_frames = 0
        self.context.interaction_retries = 0
        self.context.recovery_attempts = 0

    def _refresh_target(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        current = self.context.current_target
        same_type = [
            detection
            for detection in detections
            if current is None or detection.resource_type == current.resource_type
        ]
        if same_type:
            if current is None:
                selected = self._select_target(same_type, frame_size)
            else:
                previous_x, previous_y = current.center
                selected = min(
                    same_type,
                    key=lambda detection: (
                        (detection.center[0] - previous_x) ** 2
                        + (detection.center[1] - previous_y) ** 2,
                        -detection.confidence,
                        detection.x1,
                        detection.y1,
                    ),
                )
            self.context.previous_target = current
            self.context.current_target = selected

    def _select_target(
        self,
        detections: list[Detection],
        frame_size: tuple[int, int] | None = None,
    ) -> Detection:
        if self._target_selector is not None and frame_size is not None:
            frame_width, frame_height = frame_size
            selected = self._target_selector.select(
                detections,
                frame_width,
                frame_height,
            )
            if selected is not None:
                return selected
        return max(detections, key=lambda item: item.confidence)

    def _abandon_target(self, now: float) -> None:
        self.context.previous_target = self.context.current_target
        self.context.current_target = None
        self.context.lost_target_frames = 0
        self.context.interaction_retries = 0
        self.context.recovery_attempts = 0
        self._transition(BotState.PATROL, now)

    def _transition(self, new_state: BotState, now: float) -> None:
        self.context.state = new_state
        self.context.state_enter_time = now

    def _set_action(self, action: Action) -> Action:
        self._last_action = action
        return action


# 便于调用方继续从 core.state_machine 导入公开状态类型。
__all__ = ["Action", "BotContext", "BotState", "StateMachine"]
