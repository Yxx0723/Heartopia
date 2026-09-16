"""跨模块共享的基础数据模型。

这里只放不会依赖具体实现的值对象。检测器、状态机和控制器将在后续
Task 中分别实现。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """单个资源检测结果，坐标使用当前帧内部像素坐标。"""

    resource_type: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """返回检测框中心点。"""
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )


@dataclass(frozen=True)
class Frame:
    """一帧游戏窗口图像及其最小元数据。"""

    image: np.ndarray
    timestamp: float
    sequence: int

    @property
    def height(self) -> int:
        """返回图像高度。"""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """返回图像宽度。"""
        return int(self.image.shape[1])


@dataclass(frozen=True)
class Perception:
    """视觉层提供给决策层的基础结果。"""

    resources: list[Detection]
    interaction_prompt: bool
    motion_score: float
    inventory_full: bool = False
    abnormal_ui: bool = False
    frame_size: tuple[int, int] | None = None
