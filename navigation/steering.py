"""基于屏幕横向误差的目标居中控制。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.models import Detection


@dataclass(frozen=True)
class SteeringCommand:
    """一次目标居中计算结果。"""

    error_x: float
    mouse_dx: int
    target_center_x: int
    screen_center_x: float
    aligned: bool


class Steering:
    """把目标横向误差转换为受限的相对鼠标移动。"""

    def __init__(
        self,
        *,
        kp: float = 0.25,
        center_deadzone: float = 0.06,
        max_mouse_dx: int = 100,
    ) -> None:
        if not math.isfinite(kp) or kp < 0:
            raise ValueError("kp 必须是非负有限数")
        if not 0.0 <= center_deadzone < 0.5:
            raise ValueError("center_deadzone 必须在 0.0 到 0.5 之间")
        if max_mouse_dx < 0:
            raise ValueError("max_mouse_dx 不能为负数")

        self._kp = kp
        self._center_deadzone = center_deadzone
        self._max_mouse_dx = max_mouse_dx

    def compute(
        self,
        target: Detection,
        frame_width: int,
        frame_height: int,
    ) -> SteeringCommand:
        """根据目标和帧尺寸计算本次转向命令。"""
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_width 和 frame_height 必须大于零")

        target_center_x = target.center[0]
        screen_center_x = frame_width / 2.0
        error_x = target_center_x - screen_center_x
        aligned = abs(error_x) <= frame_width * self._center_deadzone
        if aligned:
            mouse_dx = 0
        else:
            raw_dx = error_x * self._kp
            mouse_dx = int(round(max(-self._max_mouse_dx, min(self._max_mouse_dx, raw_dx))))
            if mouse_dx == 0 and self._max_mouse_dx > 0:
                mouse_dx = 1 if error_x > 0 else -1

        return SteeringCommand(
            error_x=error_x,
            mouse_dx=mouse_dx,
            target_center_x=target_center_x,
            screen_center_x=screen_center_x,
            aligned=aligned,
        )
