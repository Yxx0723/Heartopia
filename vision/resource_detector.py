"""资源检测器接口与 Mock 实现。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from core.models import Detection


class ResourceDetector(ABC):
    """所有资源检测器必须实现的统一接口。"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """从一帧 BGR 图像中返回资源检测结果。"""


class OpenCVResourceDetector(ResourceDetector):
    """使用 OpenCV 多模板匹配检测一种资源。"""

    def __init__(
        self,
        *,
        resource_type: str = "wood",
        templates: Sequence[np.ndarray] | None = None,
        template_paths: Sequence[str | Path] | None = None,
        threshold: float = 0.70,
        roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        nms_iou_threshold: float = 0.30,
        cv2_module: object = cv2,
    ) -> None:
        if not resource_type.strip():
            raise ValueError("resource_type 不能为空")
        if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
            raise ValueError("threshold 必须在 0.0 到 1.0 之间")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold 必须在 0.0 到 1.0 之间")
        self._resource_type = resource_type.strip()
        self._threshold = threshold
        self._roi = self._validate_roi(roi)
        self._nms_iou_threshold = nms_iou_threshold
        self._cv2 = cv2_module
        self._templates = [
            self._to_gray(template, require_variance=True)
            for template in templates or ()
        ]
        self._templates.extend(self._load_templates(template_paths or ()))
        if not self._templates:
            raise ValueError("OpenCVResourceDetector 至少需要一个模板")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """返回当前帧中经过 NMS 去重的资源检测框。"""
        self._validate_frame(frame)
        roi_x, roi_y, roi_w, roi_h = self._pixel_roi(frame)
        region = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        gray_region = self._to_gray(region)
        candidates: list[Detection] = []

        for template in self._templates:
            template_h, template_w = template.shape[:2]
            if template_h > gray_region.shape[0] or template_w > gray_region.shape[1]:
                continue
            result = self._cv2.matchTemplate(
                gray_region,
                template,
                self._cv2.TM_CCOEFF_NORMED,
            )
            locations = np.argwhere(np.nan_to_num(result, nan=0.0) >= self._threshold)
            for y, x in locations:
                confidence = float(result[y, x])
                candidates.append(
                    Detection(
                        self._resource_type,
                        roi_x + int(x),
                        roi_y + int(y),
                        roi_x + int(x) + template_w,
                        roi_y + int(y) + template_h,
                        confidence,
                    )
                )
        return self._nms(candidates)

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(detection, kept) < self._nms_iou_threshold for kept in selected):
                selected.append(detection)
        return selected

    @staticmethod
    def _iou(first: Detection, second: Detection) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
        second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _load_templates(self, paths: Sequence[str | Path]) -> list[np.ndarray]:
        loaded: list[np.ndarray] = []
        for path in paths:
            template_path = Path(path)
            image = self._cv2.imread(str(template_path), self._cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法读取资源模板: {template_path}")
            loaded.append(self._to_gray(image, require_variance=True))
        return loaded

    def _to_gray(self, image: np.ndarray, *, require_variance: bool = False) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError("模板和帧必须是 numpy.ndarray")
        if image.ndim == 2:
            gray = np.ascontiguousarray(image)
        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = self._cv2.cvtColor(image[:, :, :3], self._cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("模板必须是灰度或 BGR 图像")
        if require_variance and (gray.size == 0 or np.ptp(gray) == 0):
            raise ValueError("资源模板必须包含像素变化")
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


class MockScenario(str, Enum):
    """MockResourceDetector 支持的确定性场景。"""

    NONE = "none"
    SINGLE = "single"
    MULTIPLE = "multiple"
    MOVING = "moving"
    LOST = "lost"


class MockResourceDetector(ResourceDetector):
    """根据预置场景或检测脚本生成资源检测结果。

    Mock 检测器只使用输入帧的宽高，不读取图像内容。它适合状态机测试，
    可以稳定模拟资源出现、多个候选、目标移动和连续丢失。传入 ``script``
    后，每次 ``detect`` 消费一组检测结果，脚本耗尽后返回空列表。
    """

    def __init__(
        self,
        scenario: MockScenario | str = MockScenario.NONE,
        *,
        resource_type: str = "wood",
        script: Sequence[Sequence[Detection]] | None = None,
        visible_frames: int = 3,
        move_step_ratio: float = 0.05,
    ) -> None:
        try:
            self._scenario = (
                scenario
                if isinstance(scenario, MockScenario)
                else MockScenario(str(scenario).strip().lower())
            )
        except ValueError as exc:
            valid = ", ".join(item.value for item in MockScenario)
            raise ValueError(f"未知 Mock 场景，可选值: {valid}") from exc

        if not isinstance(resource_type, str) or not resource_type.strip():
            raise ValueError("resource_type 不能为空")
        if visible_frames < 0:
            raise ValueError("visible_frames 不能为负数")
        if not np.isfinite(move_step_ratio) or move_step_ratio <= 0:
            raise ValueError("move_step_ratio 必须是正的有限数")

        self._resource_type = resource_type.strip()
        self._script = self._validate_script(script)
        self._visible_frames = visible_frames
        self._move_step_ratio = float(move_step_ratio)
        self._frame_index = 0

    @property
    def scenario(self) -> MockScenario:
        """返回当前 Mock 场景。"""
        return self._scenario

    @property
    def frame_index(self) -> int:
        """返回已经处理过的帧数量。"""
        return self._frame_index

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """根据当前场景生成一组检测结果。"""
        self._validate_frame(frame)
        current_index = self._frame_index
        self._frame_index += 1

        if self._script is not None:
            if current_index >= len(self._script):
                return []
            return list(self._script[current_index])

        if self._scenario is MockScenario.NONE:
            return []
        if self._scenario is MockScenario.SINGLE:
            return [self._single_detection(frame)]
        if self._scenario is MockScenario.MULTIPLE:
            return self._multiple_detections(frame)
        if self._scenario is MockScenario.MOVING:
            return [self._moving_detection(frame, current_index)]
        if current_index < self._visible_frames:
            return [self._single_detection(frame)]
        return []

    def reset(self) -> None:
        """重置脚本和场景的帧计数。"""
        self._frame_index = 0

    def _single_detection(self, frame: np.ndarray) -> Detection:
        height, width = frame.shape[:2]
        return self._make_detection(
            width=width,
            height=height,
            center_x=width * 0.5,
            center_y=height * 0.5,
            confidence=0.95,
        )

    def _multiple_detections(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        return [
            self._make_detection(width, height, width * 0.25, height * 0.5, 0.86),
            self._make_detection(width, height, width * 0.50, height * 0.5, 0.92),
            self._make_detection(width, height, width * 0.75, height * 0.5, 0.81),
        ]

    def _moving_detection(self, frame: np.ndarray, frame_index: int) -> Detection:
        height, width = frame.shape[:2]
        center_ratio = min(0.85, 0.25 + frame_index * self._move_step_ratio)
        return self._make_detection(
            width=width,
            height=height,
            center_x=width * center_ratio,
            center_y=height * 0.5,
            confidence=0.90,
        )

    def _make_detection(
        self,
        width: int,
        height: int,
        center_x: float,
        center_y: float,
        confidence: float,
    ) -> Detection:
        box_width = max(2, min(width, int(width * 0.08)))
        box_height = max(2, min(height, int(height * 0.14)))
        x1 = max(0, min(width - box_width, int(center_x - box_width / 2)))
        y1 = max(0, min(height - box_height, int(center_y - box_height / 2)))
        return Detection(
            resource_type=self._resource_type,
            x1=x1,
            y1=y1,
            x2=x1 + box_width,
            y2=y1 + box_height,
            confidence=confidence,
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame 必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame 必须是 H x W x 3 的 BGR 图像")
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("frame 尺寸必须大于零")

    @staticmethod
    def _validate_script(
        script: Sequence[Sequence[Detection]] | None,
    ) -> tuple[tuple[Detection, ...], ...] | None:
        if script is None:
            return None

        normalized: list[tuple[Detection, ...]] = []
        for frame_detections in script:
            detections = tuple(frame_detections)
            if not all(isinstance(item, Detection) for item in detections):
                raise TypeError("script 中的检测结果必须是 Detection")
            normalized.append(detections)
        return tuple(normalized)
