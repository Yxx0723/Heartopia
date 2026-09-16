"""OpenCV 调试 Overlay。"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np


class OverlayError(RuntimeError):
    """调试窗口或输入画面格式错误时抛出。"""


def _load_cv2() -> Any:
    """延迟加载 OpenCV，便于无 GUI 依赖的单元测试运行。"""
    try:
        import cv2
    except ImportError as exc:
        raise OverlayError("DebugOverlay 需要安装 opencv-python") from exc
    return cv2


class DebugOverlay:
    """在截图上绘制基础调试信息并显示 OpenCV 窗口。

    当前阶段只显示 FPS、窗口尺寸和状态。``cv2_module`` 与 ``clock`` 可
    注入测试替身，避免单元测试创建真实 GUI 窗口或依赖实时睡眠。
    """

    def __init__(
        self,
        window_name: str = "Heartopia Bot Debug",
        *,
        enabled: bool = True,
        cv2_module: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not window_name.strip():
            raise ValueError("window_name 不能为空")

        self._window_name = window_name
        self._enabled = enabled
        self._cv2 = cv2_module or (_load_cv2() if enabled else None)
        self._clock = clock
        self._last_render_time: float | None = None
        self._fps = 0.0
        self._closed = False

    @property
    def fps(self) -> float:
        """返回最近计算出的平滑 FPS。"""
        return self._fps

    def render(
        self,
        frame: np.ndarray,
        state: Any,
        fps: float | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """绘制并显示调试画面，返回绘制后的副本。

        Args:
            frame: BGR 图像，形状应为 ``H x W x 3``。
            state: 状态枚举或可转换为字符串的状态对象。
            fps: 可选的外部 FPS；省略时由两次 render 的间隔估算。
        """
        self._validate_frame(frame)
        if not self._enabled:
            return frame
        if self._cv2 is None:
            raise OverlayError("DebugOverlay 未初始化 OpenCV 后端")

        now = self._clock()
        if fps is None:
            self._update_fps(now)
            display_fps = self._fps
        else:
            display_fps = max(0.0, float(fps))
            self._fps = display_fps
            self._last_render_time = now

        height, width = frame.shape[:2]
        state_name = getattr(state, "name", None) or str(state)
        canvas = frame.copy()
        lines = [
            f"FPS: {display_fps:.1f}",
            f"WINDOW: {width}x{height}",
            f"STATE: {state_name}",
        ]
        if details:
            for label, value in details.items():
                lines.append(f"{str(label).upper()}: {value}")

        for index, text in enumerate(lines):
            self._cv2.putText(
                canvas,
                text,
                (12, 28 + index * 28),
                getattr(self._cv2, "FONT_HERSHEY_SIMPLEX", 0),
                0.7,
                (0, 255, 0),
                2,
                getattr(self._cv2, "LINE_AA", 16),
            )

        try:
            self._cv2.imshow(self._window_name, canvas)
        except Exception as exc:
            raise OverlayError("无法显示 Debug Overlay 窗口") from exc
        return canvas

    def poll_events(self, delay_ms: int = 1) -> int | None:
        """刷新 OpenCV 窗口事件并返回按键值。

        本方法只负责读取 Overlay 自身的 OpenCV 事件；F6/F8/F9/F10 等全局
        业务热键由 ``actions.hotkeys.HotkeyManager`` 负责。
        """
        if not self._enabled:
            return None
        if delay_ms < 0:
            raise ValueError("delay_ms 不能为负数")
        if self._cv2 is None:
            raise OverlayError("DebugOverlay 未初始化 OpenCV 后端")
        try:
            return int(self._cv2.waitKey(delay_ms))
        except Exception as exc:
            raise OverlayError("读取 Debug Overlay 窗口事件失败") from exc

    def close(self) -> None:
        """关闭调试窗口；重复调用不会报错。"""
        if self._closed or not self._enabled or self._cv2 is None:
            return
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            # 进程退出或窗口已被用户关闭时，清理动作应保持幂等。
            pass
        finally:
            self._closed = True

    def _update_fps(self, now: float) -> None:
        if self._last_render_time is not None:
            elapsed = now - self._last_render_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                if self._fps == 0.0:
                    self._fps = instantaneous_fps
                else:
                    self._fps = self._fps * 0.8 + instantaneous_fps * 0.2
        self._last_render_time = now

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise OverlayError("Overlay 输入必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise OverlayError("Overlay 输入必须是 H x W x 3 的 BGR 图像")
