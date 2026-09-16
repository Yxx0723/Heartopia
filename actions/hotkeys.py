"""Windows 功能键轮询。

热键只负责把物理按键转换为业务事件，不直接操作引擎或输入控制器。
使用 ``GetAsyncKeyState`` 的边沿检测可以避免引入常驻监听线程，同时保留
可注入 API，便于在非交互测试中验证暂停、恢复和停止逻辑。
"""

from __future__ import annotations

from typing import Any, Mapping


class HotkeyError(RuntimeError):
    """热键后端不可用或配置非法时抛出。"""


class HotkeyManager:
    """将配置的功能键转换为一次性业务事件。"""

    VIRTUAL_KEYS = {
        "F6": 0x75,
        "F8": 0x77,
        "F9": 0x78,
        "F10": 0x79,
    }
    EVENTS = ("screenshot", "pause", "resume", "stop")

    def __init__(
        self,
        bindings: Mapping[str, str] | None = None,
        *,
        api: Any | None = None,
    ) -> None:
        self._api = api or self._load_api()
        source = bindings or {
            "screenshot": "F6",
            "pause": "F8",
            "resume": "F9",
            "stop": "F10",
        }
        self._bindings: dict[str, int] = {}
        self._previous: dict[str, bool] = {}
        for event in self.EVENTS:
            key_name = str(source.get(event, "")).strip().upper()
            if not key_name:
                continue
            if key_name not in self.VIRTUAL_KEYS:
                raise ValueError(f"不支持的热键: {key_name}")
            self._bindings[event] = self.VIRTUAL_KEYS[key_name]
            self._previous[event] = False

    @property
    def bindings(self) -> Mapping[str, int]:
        """返回业务事件到 Windows 虚拟键码的只读视图。"""
        return dict(self._bindings)

    def poll(self) -> tuple[str, ...]:
        """读取一次键盘状态，只在按下沿产生事件。"""
        events: list[str] = []
        for event, virtual_key in self._bindings.items():
            try:
                pressed = bool(int(self._api.GetAsyncKeyState(virtual_key)) & 0x8000)
            except Exception as exc:
                raise HotkeyError(f"读取热键状态失败: {event}") from exc
            if pressed and not self._previous[event]:
                events.append(event)
            self._previous[event] = pressed
        return tuple(events)

    def reset(self) -> None:
        """清除边沿状态，适用于恢复焦点或测试场景。"""
        for event in self._previous:
            self._previous[event] = False

    @staticmethod
    def _load_api() -> Any:
        try:
            import win32api
        except ImportError as exc:
            raise HotkeyError("Windows 热键需要安装 pywin32") from exc
        return win32api


__all__ = ["HotkeyError", "HotkeyManager"]
