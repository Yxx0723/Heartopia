"""基于 mss 的游戏客户区截图。"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .window import GameWindow, WindowError


class CaptureError(RuntimeError):
    """截图失败或截图格式不正确时抛出。"""


class CaptureBackend(Protocol):
    """mss 截图后端的最小接口，便于测试时注入替身。"""

    def grab(self, monitor: dict[str, int]) -> Any: ...


def _create_mss_backend() -> CaptureBackend:
    """延迟创建 mss 后端。"""
    try:
        import mss
    except ImportError as exc:
        raise CaptureError("WindowCapture 需要安装 mss") from exc
    return mss.mss()


class WindowCapture:
    """截取 GameWindow 客户区并返回 BGR numpy 数组。"""

    def __init__(
        self,
        window: GameWindow,
        backend: CaptureBackend | None = None,
    ) -> None:
        self._window = window
        self._backend = backend or _create_mss_backend()

    def capture(self) -> np.ndarray:
        """截取当前客户区，返回形状为 ``H x W x 3`` 的 BGR 图像。"""
        try:
            left, top, right, bottom = self._window.get_rect()
        except WindowError as exc:
            raise CaptureError("无法获取游戏窗口截图区域") from exc

        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            raise CaptureError("截图区域尺寸无效")

        try:
            raw = self._backend.grab(
                {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                }
            )
        except Exception as exc:
            raise CaptureError("mss 截图失败") from exc

        frame = np.asarray(raw)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise CaptureError("截图后端返回了不支持的图像格式")

        # mss 通常返回 BGRA；去掉 alpha 后仍保持 BGR 通道顺序。
        return np.ascontiguousarray(frame[:, :, :3])

    def close(self) -> None:
        """释放截图后端资源（如果后端提供 close）。"""
        close = getattr(self._backend, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> "WindowCapture":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
