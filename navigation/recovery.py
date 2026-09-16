"""卡死后的分步恢复动作。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable


class RecoveryError(RuntimeError):
    """恢复计划配置或执行状态错误时抛出。"""


@dataclass(frozen=True)
class RecoveryCommand:
    """恢复控制器输出的一条命令。"""

    kind: str
    step_index: int
    key: str | None = None
    duration: float = 0.0
    remaining: float = 0.0
    mouse_dx: float = 0.0


@dataclass(frozen=True)
class _RecoveryStep:
    kind: str
    key: str | None = None
    duration: float = 0.0
    mouse_dx: float = 0.0


class RecoveryController:
    """按固定安全序列执行一次恢复尝试。"""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        backward_key: str = "s",
        left_key: str = "a",
        right_key: str = "d",
        forward_key: str = "w",
        backward_duration: float = 0.5,
        strafe_duration: float = 0.5,
        forward_duration: float = 0.5,
        camera_turn_dx: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须大于零")
        durations = (backward_duration, strafe_duration, forward_duration)
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError("恢复动作时长必须是正的有限数")
        if not math.isfinite(camera_turn_dx) or camera_turn_dx <= 0:
            raise ValueError("camera_turn_dx 必须是正的有限数")
        self._max_attempts = max_attempts
        self._keys = (backward_key, left_key, right_key, forward_key)
        if any(not isinstance(key, str) or not key.strip() for key in self._keys):
            raise ValueError("恢复按键不能为空")
        self._durations = durations
        self._camera_turn_dx = camera_turn_dx
        self._clock = clock
        self._active = False
        self._attempt = 0
        self._steps: tuple[_RecoveryStep, ...] = ()
        self._step_index = 0
        self._step_started_at: float | None = None

    @property
    def active(self) -> bool:
        """返回恢复计划是否正在执行。"""
        return self._active

    @property
    def attempt(self) -> int:
        """返回当前恢复尝试编号。"""
        return self._attempt

    @property
    def step_index(self) -> int:
        """返回当前恢复步骤的零基索引。"""
        return self._step_index

    def start(self, attempt: int, now: float | None = None) -> None:
        """开始一次恢复尝试。"""
        if attempt < 1 or attempt > self._max_attempts:
            raise RecoveryError("恢复尝试次数超出上限")
        current_time = self._clock() if now is None else now
        backward_key, left_key, right_key, forward_key = self._keys
        strafe_key = left_key if attempt % 2 else right_key
        turn_dx = self._camera_turn_dx if attempt % 2 else -self._camera_turn_dx
        backward_duration, strafe_duration, forward_duration = self._durations
        self._steps = (
            _RecoveryStep("release_all"),
            _RecoveryStep("key_hold", backward_key, backward_duration),
            _RecoveryStep("key_hold", strafe_key, strafe_duration),
            _RecoveryStep("mouse_move", mouse_dx=turn_dx),
            _RecoveryStep("key_hold", forward_key, forward_duration),
        )
        self._attempt = attempt
        self._step_index = 0
        self._step_started_at = current_time
        self._active = True

    def abort(self) -> None:
        """中止当前恢复计划。"""
        self._active = False
        self._step_started_at = None

    def update(self, now: float | None = None) -> RecoveryCommand | None:
        """推进恢复计划一个 tick 并输出当前命令。"""
        if not self._active:
            return None
        current_time = self._clock() if now is None else now
        step = self._steps[self._step_index]
        if step.kind == "release_all" or step.kind == "mouse_move":
            command = self._command_for_step(step)
            self._advance(current_time)
            return command

        step_started_at = (
            self._step_started_at
            if self._step_started_at is not None
            else current_time
        )
        elapsed = max(0.0, current_time - step_started_at)
        if elapsed >= step.duration:
            self._advance(current_time)
            if not self._active:
                return None
            step = self._steps[self._step_index]
            if step.kind in ("release_all", "mouse_move"):
                command = self._command_for_step(step)
                self._advance(current_time)
                return command
            elapsed = 0.0

        return self._command_for_step(
            step,
            elapsed=elapsed,
            remaining=max(0.0, step.duration - elapsed),
        )

    def _command_for_step(
        self,
        step: _RecoveryStep,
        *,
        elapsed: float = 0.0,
        remaining: float = 0.0,
    ) -> RecoveryCommand:
        return RecoveryCommand(
            kind=step.kind,
            step_index=self._step_index,
            key=step.key,
            duration=step.duration,
            remaining=remaining,
            mouse_dx=step.mouse_dx,
        )

    def _advance(self, now: float) -> None:
        self._step_index += 1
        if self._step_index >= len(self._steps):
            self._active = False
            self._step_started_at = None
            return
        self._step_started_at = now
