from __future__ import annotations

from typing import Any

import pytest

from capture.window import GameWindow, WindowNotFoundError


class FakeWindowApi:
    SW_RESTORE = 9

    def __init__(self) -> None:
        self.windows = {
            101: {"title": "浏览器", "visible": True, "rect": (0, 0, 1280, 720)},
            202: {
                "title": "心动小镇 - 测试",
                "visible": True,
                "rect": (100, 80, 900, 680),
            },
            303: {
                "title": "心动小镇隐藏窗口",
                "visible": False,
                "rect": (0, 0, 800, 600),
            },
        }
        self.foreground = 101
        self.restored: list[int] = []

    def EnumWindows(self, callback: Any, extra: Any) -> None:
        for hwnd in self.windows:
            callback(hwnd, extra)

    def IsWindowVisible(self, hwnd: int) -> bool:
        return self.windows[hwnd]["visible"]

    def GetWindowText(self, hwnd: int) -> str:
        return self.windows[hwnd]["title"]

    def IsWindow(self, hwnd: int) -> bool:
        return hwnd in self.windows

    def GetClientRect(self, hwnd: int) -> tuple[int, int, int, int]:
        _, _, width, height = self.windows[hwnd]["rect"]
        return (0, 0, width, height)

    def ClientToScreen(self, hwnd: int, point: tuple[int, int]) -> tuple[int, int]:
        left, top, _, _ = self.windows[hwnd]["rect"]
        return (left + point[0], top + point[1])

    def GetForegroundWindow(self) -> int:
        return self.foreground

    def IsIconic(self, _hwnd: int) -> bool:
        return False

    def ShowWindow(self, hwnd: int, command: int) -> None:
        self.restored.append(hwnd)

    def SetForegroundWindow(self, hwnd: int) -> None:
        self.foreground = hwnd


def test_find_visible_window_by_case_insensitive_title() -> None:
    api = FakeWindowApi()
    window = GameWindow("心动小镇", api=api)

    assert window.find() is True
    assert window.hwnd == 202


def test_get_rect_returns_client_area_in_screen_coordinates() -> None:
    window = GameWindow("心动小镇", api=FakeWindowApi())
    window.find()

    assert window.get_rect() == (100, 80, 1000, 760)


def test_activate_and_foreground_status() -> None:
    api = FakeWindowApi()
    window = GameWindow("心动小镇", api=api)
    window.find()

    assert window.is_foreground() is False
    window.activate()
    assert window.is_foreground() is True


def test_get_rect_raises_when_window_is_missing() -> None:
    window = GameWindow("不存在的窗口", api=FakeWindowApi())

    with pytest.raises(WindowNotFoundError):
        window.get_rect()
