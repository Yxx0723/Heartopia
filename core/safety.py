"""运行时输入安全保护。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol


class ForegroundWindow(Protocol):
    """SafetyGuard 所需的窗口最小接口。"""

    def is_foreground(self) -> bool: ...


@dataclass(frozen=True)
class SafetyStatus:
    """可显示在 Overlay 或日志中的窗口与运行安全状态。"""

    window_foreground: bool
    paused: bool
    expired: bool
    can_control: bool


class SafetyGuard:
    """检查窗口焦点、暂停状态和最大运行时间。"""

    def __init__(
        self,
        window: ForegroundWindow,
        *,
        pause_when_unfocused: bool = True,
        max_runtime_minutes: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_runtime_minutes <= 0 or not math.isfinite(max_runtime_minutes):
            raise ValueError("max_runtime_minutes 必须是正的有限数")
        self._window = window
        self._pause_when_unfocused = pause_when_unfocused
        self._max_runtime_seconds = max_runtime_minutes * 60.0
        self._clock = clock
        self._started_at: float | None = None
        self._paused = False
        self._expired = False

    @property
    def paused(self) -> bool:
        """返回安全保护是否被手动暂停。"""
        return self._paused

    @property
    def expired(self) -> bool:
        """返回是否超过最大运行时间。"""
        return self._expired

    @property
    def window_foreground(self) -> bool:
        """返回目标窗口当前是否在前台。"""
        try:
            return bool(self._window.is_foreground())
        except Exception:
            return False

    def status(self, now: float | None = None) -> SafetyStatus:
        """返回一次完整安全状态快照。"""
        return SafetyStatus(
            window_foreground=self.window_foreground,
            paused=self._paused,
            expired=self._expired,
            can_control=self.can_control(now),
        )

    def start(self, now: float | None = None) -> None:
        """开始一次新的运行计时。"""
        self._started_at = self._clock() if now is None else now
        self._paused = False
        self._expired = False

    def pause(self) -> None:
        """暂停输入控制。"""
        self._paused = True

    def resume(self) -> None:
        """恢复输入控制。"""
        if not self._expired:
            self._paused = False

    def can_control(self, now: float | None = None) -> bool:
        """返回当前是否允许向游戏窗口发送输入。"""
        if self._started_at is None or self._paused or self._expired:
            return False
        current_time = self._clock() if now is None else now
        if current_time - self._started_at >= self._max_runtime_seconds:
            self._expired = True
            return False
        if self._pause_when_unfocused and not self.window_foreground:
            return False
        return True
