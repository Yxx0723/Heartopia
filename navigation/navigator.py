"""可中断、可恢复的固定路线导航器。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from .route import Route, RouteAction


@dataclass(frozen=True)
class RouteCommand:
    """导航器输出的一次高层路线命令。"""

    action: RouteAction
    step_index: int
    elapsed: float = 0.0
    remaining: float = 0.0
    mouse_dx: float = 0.0


class RouteNavigator:
    """按 tick 执行路线，并保留被中断动作的进度。"""

    def __init__(
        self,
        route: Route,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._route = route
        self._clock = clock
        self._active = False
        self._paused = False
        self._step_index = 0
        self._step_started_at: float | None = None
        self._paused_elapsed = 0.0

    @property
    def active(self) -> bool:
        """返回路线是否仍在执行。"""
        return self._active

    @property
    def paused(self) -> bool:
        """返回路线是否处于中断暂停状态。"""
        return self._paused

    @property
    def route_index(self) -> int:
        """返回当前路线动作的零基索引。"""
        return self._step_index

    @property
    def route_length(self) -> int:
        """返回路线动作总数。"""
        return len(self._route.steps)

    def start(self, now: float | None = None) -> None:
        """从路线第一条动作开始。"""
        current_time = self._clock() if now is None else now
        self._active = True
        self._paused = False
        self._step_index = 0
        self._step_started_at = current_time
        self._paused_elapsed = 0.0

    def stop(self) -> None:
        """停止路线并清除当前执行状态。"""
        self._active = False
        self._paused = False
        self._step_index = 0
        self._step_started_at = None
        self._paused_elapsed = 0.0

    def pause(self, now: float | None = None) -> None:
        """暂停路线并冻结当前动作已执行时间。"""
        if not self._active or self._paused:
            return
        current_time = self._clock() if now is None else now
        self._paused_elapsed = self._elapsed(current_time)
        self._paused = True

    def interrupt(self, now: float | None = None) -> None:
        """pause 的语义别名，用于资源发现时中断路线。"""
        self.pause(now)

    def resume(self, now: float | None = None) -> None:
        """恢复路线，继续当前动作的剩余时间。"""
        if not self._active or not self._paused:
            return
        current_time = self._clock() if now is None else now
        self._step_started_at = current_time - self._paused_elapsed
        self._paused_elapsed = 0.0
        self._paused = False

    def update(self, now: float | None = None) -> RouteCommand | None:
        """推进路线一个 tick 并返回当前路线命令。"""
        if not self._active or self._paused:
            return None

        current_time = self._clock() if now is None else now
        if self._step_started_at is None:
            self._step_started_at = current_time

        step = self._route.steps[self._step_index]
        if step.action.is_turn:
            command = RouteCommand(
                action=step.action,
                step_index=self._step_index,
                mouse_dx=step.mouse_dx if step.action is RouteAction.TURN_RIGHT else -step.mouse_dx,
            )
            self._advance_step(current_time)
            return command

        elapsed = self._elapsed(current_time)
        if elapsed >= step.duration:
            self._advance_step(current_time)
            if not self._active:
                return None
            step = self._route.steps[self._step_index]
            elapsed = 0.0
            if step.action.is_turn:
                command = RouteCommand(
                    action=step.action,
                    step_index=self._step_index,
                    mouse_dx=step.mouse_dx if step.action is RouteAction.TURN_RIGHT else -step.mouse_dx,
                )
                self._advance_step(current_time)
                return command

        return RouteCommand(
            action=step.action,
            step_index=self._step_index,
            elapsed=max(0.0, elapsed),
            remaining=max(0.0, step.duration - elapsed),
        )

    def _advance_step(self, now: float) -> None:
        next_index = self._step_index + 1
        if next_index >= len(self._route.steps):
            if not self._route.loop:
                self._active = False
                self._step_started_at = None
                return
            next_index = 0

        self._step_index = next_index
        self._step_started_at = now

    def _elapsed(self, now: float) -> float:
        if self._step_started_at is None:
            return 0.0
        elapsed = now - self._step_started_at
        if not math.isfinite(elapsed):
            raise ValueError("路线时间必须是有限数")
        return max(0.0, elapsed)
