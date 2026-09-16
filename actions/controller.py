"""安全的键鼠输入控制器。"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Protocol


class InputError(RuntimeError):
    """输入后端操作失败时抛出。"""


class InputBackend(Protocol):
    """pydirectinput 所需的最小接口，便于单元测试替换。"""

    def keyDown(self, key: str) -> Any: ...

    def keyUp(self, key: str) -> Any: ...

    def moveRel(self, x: int, y: int) -> Any: ...


def _load_pydirectinput() -> InputBackend:
    """延迟加载 pydirectinput，dry-run 或单元测试无需安装它。"""
    try:
        import pydirectinput
    except ImportError as exc:
        raise InputError("真实输入模式需要安装 pydirectinput") from exc
    return pydirectinput


class InputController:
    """统一管理键盘和鼠标输入。

    默认 ``dry_run=True``，此时不会触发任何真实输入，但仍会追踪按键状态，
    方便在不操作游戏的情况下运行上层逻辑。``press`` 不使用阻塞式 sleep，
    而是由 ``update`` 在到期后释放按键。
    """

    def __init__(
        self,
        *,
        dry_run: bool = True,
        backend: InputBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dry_run = dry_run
        self._backend = None if dry_run else (backend or _load_pydirectinput())
        self._clock = clock
        self._pressed_keys: dict[str, None] = {}
        self._pending_releases: dict[str, float] = {}

    @property
    def dry_run(self) -> bool:
        """返回是否处于禁止真实输入的 dry-run 模式。"""
        return self._dry_run

    @property
    def pressed_keys(self) -> frozenset[str]:
        """返回当前由控制器认为处于按下状态的按键。"""
        return frozenset(self._pressed_keys)

    def key_down(self, key: str) -> None:
        """按下一个按键；同一按键重复调用不会重复发送。"""
        normalized_key = self._normalize_key(key)
        if normalized_key in self._pressed_keys:
            return

        try:
            if not self._dry_run:
                self._require_backend().keyDown(normalized_key)
        except Exception as exc:
            raise InputError(f"按下按键失败: {normalized_key}") from exc

        self._pressed_keys[normalized_key] = None

    def key_up(self, key: str) -> None:
        """释放一个按键；未追踪的按键不会重复发送释放指令。"""
        normalized_key = self._normalize_key(key)
        if normalized_key not in self._pressed_keys:
            return

        try:
            if not self._dry_run:
                self._require_backend().keyUp(normalized_key)
        except Exception as exc:
            raise InputError(f"释放按键失败: {normalized_key}") from exc
        finally:
            self._pressed_keys.pop(normalized_key, None)
            self._pending_releases.pop(normalized_key, None)

    def press(self, key: str, duration: float = 0.05) -> None:
        """按下按键并安排非阻塞释放。

        调用后需要由主循环持续调用 ``update``。``duration`` 为 0 时立即
        释放，适合需要一次性按键事件的场景。
        """
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration 必须是非负有限数")

        normalized_key = self._normalize_key(key)
        self.key_down(normalized_key)
        if duration == 0:
            self.key_up(normalized_key)
        else:
            self._pending_releases[normalized_key] = self._clock() + duration

    def mouse_move_relative(self, dx: float, dy: float) -> None:
        """发送相对鼠标移动，浮点输入会截断为整数。"""
        if not math.isfinite(dx) or not math.isfinite(dy):
            raise ValueError("鼠标位移必须是有限数")

        try:
            if not self._dry_run:
                self._require_backend().moveRel(int(dx), int(dy))
        except Exception as exc:
            raise InputError("相对鼠标移动失败") from exc

    def update(self) -> None:
        """释放所有已经到期的非阻塞短按。"""
        now = self._clock()
        due_keys = [
            key for key, deadline in self._pending_releases.items() if now >= deadline
        ]
        errors: list[InputError] = []
        for key in due_keys:
            try:
                self.key_up(key)
            except InputError as exc:
                errors.append(exc)

        if errors:
            raise InputError("释放到期按键失败") from errors[0]

    def release_all(self) -> None:
        """尽力释放所有已追踪按键，并清空内部状态。

        即使某一个按键释放失败，也会继续尝试其他按键，避免单个后端异常
        导致 WASD 等按键持续按住。
        """
        errors: list[tuple[str, Exception]] = []
        keys = list(self._pressed_keys)
        try:
            for key in keys:
                try:
                    if not self._dry_run:
                        self._require_backend().keyUp(key)
                except Exception as exc:
                    errors.append((key, exc))
        finally:
            self._pressed_keys.clear()
            self._pending_releases.clear()

        if errors:
            failed_keys = ", ".join(key for key, _error in errors)
            raise InputError(f"释放按键失败: {failed_keys}") from errors[0][1]

    def _require_backend(self) -> InputBackend:
        if self._backend is None:
            raise InputError("输入后端未初始化")
        return self._backend

    @staticmethod
    def _normalize_key(key: str) -> str:
        if not isinstance(key, str):
            raise ValueError("按键必须是字符串")
        normalized_key = key.strip().lower()
        if not normalized_key:
            raise ValueError("按键不能为空")
        return normalized_key

    def __enter__(self) -> "InputController":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release_all()
