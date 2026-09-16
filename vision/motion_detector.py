"""基于 ROI 帧差的运动和卡死检测。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np


class MotionDetectorError(RuntimeError):
    """运动检测失败时抛出。"""


@dataclass(frozen=True)
class MotionResult:
    """一次运动检测结果。"""

    score: float
    stuck: bool
    low_motion_duration: float


class MotionDetector:
    """使用连续帧差判断画面是否长期没有变化。"""

    def __init__(
        self,
        *,
        roi: tuple[float, float, float, float] = (0.25, 0.20, 0.50, 0.60),
        motion_threshold: float = 0.01,
        stuck_after_seconds: float = 2.0,
        cv2_module: Any = cv2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= motion_threshold <= 1.0 or not math.isfinite(motion_threshold):
            raise ValueError("motion_threshold 必须在 0.0 到 1.0 之间")
        if stuck_after_seconds <= 0 or not math.isfinite(stuck_after_seconds):
            raise ValueError("stuck_after_seconds 必须是正的有限数")
        self._roi = self._validate_roi(roi)
        self._motion_threshold = motion_threshold
        self._stuck_after_seconds = stuck_after_seconds
        self._cv2 = cv2_module
        self._clock = clock
        self._previous_gray: np.ndarray | None = None
        self._low_motion_since: float | None = None

    def update(
        self,
        frame: np.ndarray,
        *,
        moving: bool,
        now: float | None = None,
    ) -> MotionResult:
        """分析一帧，并在持续移动但变化不足时返回 stuck。"""
        self._validate_frame(frame)
        current_time = self._clock() if now is None else now
        current_gray = self._roi_gray(frame)
        if self._previous_gray is None:
            score = 0.0
        else:
            score = float(
                np.mean(self._cv2.absdiff(self._previous_gray, current_gray)) / 255.0
            )
        self._previous_gray = current_gray

        if not moving or score > self._motion_threshold:
            self._low_motion_since = None
            return MotionResult(score=score, stuck=False, low_motion_duration=0.0)

        if self._low_motion_since is None:
            self._low_motion_since = current_time
        low_motion_duration = max(0.0, current_time - self._low_motion_since)
        return MotionResult(
            score=score,
            stuck=low_motion_duration >= self._stuck_after_seconds,
            low_motion_duration=low_motion_duration,
        )

    def score(self, frame: np.ndarray, now: float | None = None) -> float:
        """只获取帧差分数，不将当前调用视为持续前进。"""
        return self.update(frame, moving=False, now=now).score

    def reset(self) -> None:
        """清除上一帧和低运动计时。"""
        self._previous_gray = None
        self._low_motion_since = None

    def _roi_gray(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x_ratio, y_ratio, w_ratio, h_ratio = self._roi
        x = min(width - 1, int(width * x_ratio))
        y = min(height - 1, int(height * y_ratio))
        roi_width = max(1, min(width - x, int(width * w_ratio)))
        roi_height = max(1, min(height - y, int(height * h_ratio)))
        region = frame[y : y + roi_height, x : x + roi_width]
        return self._cv2.cvtColor(region, self._cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _validate_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if len(roi) != 4 or any(not math.isfinite(value) for value in roi):
            raise ValueError("roi 必须包含四个有限数")
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("roi 必须位于 0.0 到 1.0 范围内")
        return roi

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame 必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame 必须是 H x W x 3 的 BGR 图像")
