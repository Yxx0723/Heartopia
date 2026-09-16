"""背包满和异常 UI 的模板检测。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


class UiDetectorError(ValueError):
    """UI 模板或输入画面无效时抛出。"""


@dataclass(frozen=True)
class UiState:
    """一次 UI 检测结果。"""

    inventory_full: bool = False
    abnormal_ui: bool = False
    inventory_confidence: float = 0.0
    abnormal_confidence: float = 0.0
    matched_label: str | None = None

    @property
    def blocking(self) -> bool:
        """返回是否应阻止继续自动采集。"""
        return self.inventory_full or self.abnormal_ui


class UiDetector:
    """在可配置 ROI 内匹配会阻断自动化的 UI 模板。"""

    def __init__(
        self,
        *,
        inventory_full_template_paths: Sequence[str | Path] = (),
        abnormal_template_paths: Sequence[str | Path] = (),
        inventory_threshold: float = 0.85,
        abnormal_threshold: float = 0.85,
        roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        cv2_module: Any = cv2,
    ) -> None:
        self._cv2 = cv2_module
        self._roi = self._validate_roi(roi)
        self._inventory_threshold = self._validate_threshold(inventory_threshold)
        self._abnormal_threshold = self._validate_threshold(abnormal_threshold)
        self._inventory_templates = self._load_templates(
            inventory_full_template_paths,
            "inventory_full",
        )
        self._abnormal_templates = self._load_templates(
            abnormal_template_paths,
            "abnormal_ui",
        )

    def detect(self, frame: np.ndarray) -> UiState:
        """检测当前画面是否出现阻断性 UI。"""
        self._validate_frame(frame)
        gray = self._roi_gray(frame)
        inventory_confidence = self._best_match(gray, self._inventory_templates)
        abnormal_confidence = self._best_match(gray, self._abnormal_templates)
        inventory_full = inventory_confidence >= self._inventory_threshold
        abnormal_ui = abnormal_confidence >= self._abnormal_threshold
        label = "inventory_full" if inventory_full else None
        if abnormal_ui and (
            label is None or abnormal_confidence >= inventory_confidence
        ):
            label = "abnormal_ui"
        return UiState(
            inventory_full=inventory_full,
            abnormal_ui=abnormal_ui,
            inventory_confidence=inventory_confidence,
            abnormal_confidence=abnormal_confidence,
            matched_label=label,
        )

    def _load_templates(
        self,
        paths: Iterable[str | Path],
        label: str,
    ) -> tuple[np.ndarray, ...]:
        templates: list[np.ndarray] = []
        for path_value in paths:
            path = Path(path_value)
            image = self._cv2.imread(str(path), self._cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise UiDetectorError(f"无法读取 {label} 模板: {path}")
            if image.ndim != 2 or min(image.shape) < 2:
                raise UiDetectorError(f"{label} 模板尺寸无效: {path}")
            if float(np.std(image)) <= 1e-6:
                raise UiDetectorError(f"{label} 模板像素变化过小: {path}")
            templates.append(np.ascontiguousarray(image))
        return tuple(templates)

    def _roi_gray(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        x_ratio, y_ratio, width_ratio, height_ratio = self._roi
        x = min(width - 1, int(width * x_ratio))
        y = min(height - 1, int(height * y_ratio))
        roi_width = max(1, min(width - x, int(width * width_ratio)))
        roi_height = max(1, min(height - y, int(height * height_ratio)))
        return self._cv2.cvtColor(
            frame[y : y + roi_height, x : x + roi_width],
            self._cv2.COLOR_BGR2GRAY,
        )

    def _best_match(
        self,
        gray: np.ndarray,
        templates: Sequence[np.ndarray],
    ) -> float:
        best = 0.0
        for template in templates:
            if template.shape[0] > gray.shape[0] or template.shape[1] > gray.shape[1]:
                continue
            result = self._cv2.matchTemplate(gray, template, self._cv2.TM_CCOEFF_NORMED)
            _minimum, maximum, _minimum_location, _maximum_location = self._cv2.minMaxLoc(result)
            if math.isfinite(float(maximum)):
                best = max(best, float(maximum))
        return best

    @staticmethod
    def _validate_threshold(value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise UiDetectorError("UI threshold 必须在 0.0 到 1.0 之间")
        return float(value)

    @staticmethod
    def _validate_roi(
        roi: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if len(roi) != 4 or any(not math.isfinite(value) for value in roi):
            raise UiDetectorError("UI ROI 必须包含四个有限数")
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 or y + height > 1:
            raise UiDetectorError("UI ROI 必须位于 0.0 到 1.0 范围内")
        return roi

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise UiDetectorError("frame 必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise UiDetectorError("frame 必须是 H x W x 3 的 BGR 图像")


__all__ = ["UiDetector", "UiDetectorError", "UiState"]
