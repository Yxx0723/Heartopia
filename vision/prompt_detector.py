"""基于模板匹配的交互提示检测。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


class PromptDetectorError(RuntimeError):
    """交互提示模板加载或检测失败时抛出。"""


@dataclass(frozen=True)
class PromptDetection:
    """交互提示检测结果。"""

    detected: bool
    confidence: float = 0.0
    bbox: tuple[int, int, int, int] | None = None


class PromptDetector:
    """在可配置 ROI 内使用模板匹配检测交互提示。"""

    def __init__(
        self,
        *,
        templates: Sequence[np.ndarray] | None = None,
        template_paths: Sequence[str | Path] | None = None,
        threshold: float = 0.80,
        roi: tuple[float, float, float, float] = (0.25, 0.20, 0.50, 0.60),
        cv2_module: Any = cv2,
    ) -> None:
        if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
            raise ValueError("threshold 必须在 0.0 到 1.0 之间")
        self._threshold = threshold
        self._roi = self._validate_roi(roi)
        self._cv2 = cv2_module
        self._templates = [
            self._to_gray(template, require_variance=True)
            for template in templates or ()
        ]
        self._templates.extend(self._load_templates(template_paths or ()))
        if any(template.size == 0 for template in self._templates):
            raise PromptDetectorError("交互提示模板不能为空")

    def detect(self, frame: np.ndarray) -> bool:
        """检测当前帧是否存在交互提示。"""
        return self.detect_result(frame).detected

    def detect_result(self, frame: np.ndarray) -> PromptDetection:
        """返回提示检测的布尔值、置信度和 ROI 内检测框。"""
        self._validate_frame(frame)
        if not self._templates:
            return PromptDetection(False)

        roi_x, roi_y, roi_w, roi_h = self._pixel_roi(frame)
        region = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        gray_region = self._to_gray(region)
        best: PromptDetection | None = None

        for template in self._templates:
            template_h, template_w = template.shape[:2]
            if template_h > gray_region.shape[0] or template_w > gray_region.shape[1]:
                continue
            result = self._cv2.matchTemplate(
                gray_region,
                template,
                self._cv2.TM_CCOEFF_NORMED,
            )
            _min_value, max_value, _min_location, max_location = self._cv2.minMaxLoc(result)
            confidence = float(np.nan_to_num(max_value, nan=0.0, posinf=0.0, neginf=0.0))
            bbox = (
                roi_x + int(max_location[0]),
                roi_y + int(max_location[1]),
                template_w,
                template_h,
            )
            candidate = PromptDetection(confidence >= self._threshold, confidence, bbox)
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best or PromptDetection(False)

    def _load_templates(self, paths: Sequence[str | Path]) -> list[np.ndarray]:
        loaded: list[np.ndarray] = []
        for path in paths:
            template_path = Path(path)
            image = self._cv2.imread(str(template_path), self._cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise PromptDetectorError(f"无法读取交互提示模板: {template_path}")
            loaded.append(self._to_gray(image, require_variance=True))
        return loaded

    def _to_gray(self, image: np.ndarray, *, require_variance: bool = False) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise PromptDetectorError("模板和帧必须是 numpy.ndarray")
        if image.ndim == 2:
            gray = np.ascontiguousarray(image)
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = self._cv2.cvtColor(image[:, :, :3], self._cv2.COLOR_BGR2GRAY)
        else:
            raise PromptDetectorError("模板必须是灰度或 BGR 图像")
        if require_variance and (gray.size == 0 or np.ptp(gray) == 0):
            raise PromptDetectorError("交互提示模板必须包含像素变化")
        return gray

    def _pixel_roi(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        height, width = frame.shape[:2]
        x_ratio, y_ratio, w_ratio, h_ratio = self._roi
        x = min(width - 1, int(width * x_ratio))
        y = min(height - 1, int(height * y_ratio))
        w = max(1, min(width - x, int(width * w_ratio)))
        h = max(1, min(height - y, int(height * h_ratio)))
        return x, y, w, h

    @staticmethod
    def _validate_roi(roi: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if len(roi) != 4:
            raise ValueError("roi 必须包含 x、y、width、height")
        x, y, width, height = roi
        if any(not math.isfinite(value) for value in roi):
            raise ValueError("roi 必须是有限数")
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise ValueError("roi 必须位于 0.0 到 1.0 范围内")
        return roi

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame 必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame 必须是 H x W x 3 的 BGR 图像")
