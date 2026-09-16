"""Windows 游戏窗口查找与客户区坐标处理。"""

from __future__ import annotations

import ctypes
from typing import Any, Protocol


Rect = tuple[int, int, int, int]


class WindowError(RuntimeError):
    """窗口 API 操作失败时抛出。"""


class WindowNotFoundError(WindowError):
    """找不到目标游戏窗口时抛出。"""


class WindowApi(Protocol):
    """GameWindow 所需的最小 Win32 API 接口，便于单元测试替换。"""

    def EnumWindows(self, callback: Any, extra: Any) -> None: ...

    def IsWindowVisible(self, hwnd: int) -> bool: ...

    def GetWindowText(self, hwnd: int) -> str: ...

    def IsWindow(self, hwnd: int) -> bool: ...

    def GetClientRect(self, hwnd: int) -> tuple[int, int, int, int]: ...

    def ClientToScreen(self, hwnd: int, point: tuple[int, int]) -> tuple[int, int]: ...

    def GetForegroundWindow(self) -> int: ...

    def ShowWindow(self, hwnd: int, command: int) -> Any: ...

    def SetForegroundWindow(self, hwnd: int) -> Any: ...


def _load_windows_api() -> WindowApi:
    """延迟加载 pywin32，允许在非 Windows/无依赖环境运行单元测试。"""
    try:
        import win32gui
    except ImportError as exc:
        raise WindowError("GameWindow 需要安装 pywin32") from exc
    _set_process_dpi_awareness()
    return win32gui


def _set_process_dpi_awareness() -> None:
    """让 Win32 客户区坐标与 mss 返回的物理像素保持一致。

    Windows 未启用 DPI 感知时，``GetClientRect`` 可能返回缩放后的逻辑
    坐标，而 mss 会按物理像素截图，导致窗口矩形与帧尺寸不一致。优先使用
    Windows 10 的 per-monitor v2；较旧系统则回退到 Shcore API。DPI 感知
    只能在进程较早阶段设置，因此在加载真实 Win32 API 时执行一次。
    """
    try:
        user32 = ctypes.windll.user32
        set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if set_context is not None:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            if set_context(ctypes.c_void_p(-4)):
                return
    except (AttributeError, OSError, TypeError):
        pass

    try:
        shcore = ctypes.windll.shcore
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError, TypeError):
        # 仅影响坐标缩放，不应阻止后续窗口查找；调用方仍会验证帧尺寸。
        return


class GameWindow:
    """按标题片段查找并操作游戏窗口。

    ``get_rect`` 返回游戏客户区在桌面坐标中的矩形 ``(left, top, right,
    bottom)``。截图模块使用这个矩形，但视觉模块接收到的图像坐标从
    ``(0, 0)`` 开始，因此不会依赖桌面绝对坐标。
    """

    def __init__(self, title_contains: str, api: WindowApi | None = None) -> None:
        if not title_contains.strip():
            raise ValueError("title_contains 不能为空")

        self._title_contains = title_contains.casefold()
        self._api = api or _load_windows_api()
        self._hwnd: int | None = None

    @property
    def hwnd(self) -> int | None:
        """返回当前窗口句柄，未找到时为 ``None``。"""
        return self._hwnd

    def find(self) -> bool:
        """查找第一个可见且标题包含目标文本的窗口。"""
        matched_hwnd: int | None = None

        def callback(hwnd: int, _extra: Any) -> None:
            nonlocal matched_hwnd
            if matched_hwnd is not None:
                return
            if not self._api.IsWindowVisible(hwnd):
                return
            title = self._api.GetWindowText(hwnd)
            if self._title_contains in title.casefold():
                matched_hwnd = hwnd

        try:
            self._api.EnumWindows(callback, None)
        except Exception as exc:  # pywin32 使用的异常类型随系统环境变化
            raise WindowError("枚举 Windows 窗口失败") from exc

        self._hwnd = matched_hwnd
        return matched_hwnd is not None

    def get_rect(self) -> Rect:
        """返回可截图的客户区桌面矩形。"""
        hwnd = self._require_hwnd()
        try:
            client_left, client_top, client_right, client_bottom = (
                self._api.GetClientRect(hwnd)
            )
            screen_left, screen_top = self._api.ClientToScreen(hwnd, (0, 0))
        except Exception as exc:
            raise WindowError("读取游戏窗口客户区失败") from exc

        width = client_right - client_left
        height = client_bottom - client_top
        if width <= 0 or height <= 0:
            raise WindowError("游戏窗口客户区尺寸无效，窗口可能已最小化")

        return (
            int(screen_left),
            int(screen_top),
            int(screen_left + width),
            int(screen_top + height),
        )

    def activate(self) -> None:
        """尝试恢复并激活窗口。

        Windows 可能因前台切换限制而拒绝激活；调用方仍应通过
        ``is_foreground`` 验证结果，不能仅凭本方法返回判断。
        """
        hwnd = self._require_hwnd()
        try:
            if getattr(self._api, "IsIconic", lambda _hwnd: False)(hwnd):
                self._api.ShowWindow(hwnd, 9)  # SW_RESTORE
            self._api.SetForegroundWindow(hwnd)
        except Exception as exc:
            raise WindowError("激活游戏窗口失败") from exc

    def is_foreground(self) -> bool:
        """返回当前游戏窗口是否为前台窗口。"""
        if self._hwnd is None:
            return False
        try:
            return self._api.IsWindow(self._hwnd) and (
                self._api.GetForegroundWindow() == self._hwnd
            )
        except Exception:
            return False

    def _require_hwnd(self) -> int:
        if self._hwnd is not None:
            try:
                if self._api.IsWindow(self._hwnd):
                    return self._hwnd
            except Exception:
                self._hwnd = None

        if self.find() and self._hwnd is not None:
            return self._hwnd
        raise WindowNotFoundError("找不到目标游戏窗口")
