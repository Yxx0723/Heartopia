"""资源候选评分与目标选择。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.models import Detection


@dataclass(frozen=True)
class SelectionWeights:
    """目标评分各项权重。默认值对应 MVP 规格。"""

    confidence: float = 0.40
    center: float = 0.35
    size: float = 0.15
    priority: float = 0.10


@dataclass(frozen=True)
class TargetScore:
    """候选目标的评分明细，便于调试和后续日志记录。"""

    detection: Detection
    total: float
    confidence_score: float
    center_score: float
    size_score: float
    priority_score: float
    edge_multiplier: float


class TargetSelector:
    """按置信度、中心距离、目标尺寸和资源优先级选择目标。"""

    def __init__(
        self,
        *,
        priorities: Mapping[str, float] | None = None,
        weights: SelectionWeights | None = None,
        edge_zone_ratio: float = 0.10,
        edge_multiplier: float = 0.50,
        reference_area_ratio: float = 0.05,
    ) -> None:
        self._weights = weights or SelectionWeights()
        self._validate_weights(self._weights)
        if not 0.0 <= edge_zone_ratio < 0.5:
            raise ValueError("edge_zone_ratio 必须在 0.0 到 0.5 之间")
        if not 0.0 <= edge_multiplier <= 1.0:
            raise ValueError("edge_multiplier 必须在 0.0 到 1.0 之间")
        if not math.isfinite(reference_area_ratio) or reference_area_ratio <= 0:
            raise ValueError("reference_area_ratio 必须是正的有限数")

        self._priorities = dict(priorities or {})
        self._edge_zone_ratio = edge_zone_ratio
        self._edge_multiplier = edge_multiplier
        self._reference_area_ratio = reference_area_ratio

    def select(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
    ) -> Detection | None:
        """返回评分最高的目标；没有候选时返回 ``None``。"""
        ranked = self.rank(detections, frame_width, frame_height)
        return ranked[0].detection if ranked else None

    def rank(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
    ) -> list[TargetScore]:
        """返回按总分降序排列的候选评分。"""
        self._validate_frame_size(frame_width, frame_height)
        scored = [
            self.score_detection(detection, frame_width, frame_height)
            for detection in detections
        ]
        return sorted(
            scored,
            key=lambda item: (
                -item.total,
                -item.confidence_score,
                -item.size_score,
                item.detection.resource_type,
                item.detection.x1,
                item.detection.y1,
            ),
        )

    def score_detection(
        self,
        detection: Detection,
        frame_width: int,
        frame_height: int,
    ) -> TargetScore:
        """计算一个候选目标的评分明细。"""
        self._validate_frame_size(frame_width, frame_height)
        confidence_score = self._clamp(float(detection.confidence), 0.0, 1.0)

        center_x, center_y = detection.center
        normalized_x = center_x / frame_width
        normalized_y = center_y / frame_height
        distance = math.hypot(normalized_x - 0.5, normalized_y - 0.5)
        max_distance = math.hypot(0.5, 0.5)
        center_score = self._clamp(1.0 - distance / max_distance, 0.0, 1.0)

        box_width = max(0, detection.x2 - detection.x1)
        box_height = max(0, detection.y2 - detection.y1)
        area_ratio = (box_width * box_height) / (frame_width * frame_height)
        size_score = self._clamp(
            area_ratio / self._reference_area_ratio,
            0.0,
            1.0,
        )
        priority_score = self._priority_score(detection.resource_type)
        edge = (
            self._edge_multiplier
            if normalized_x <= self._edge_zone_ratio
            or normalized_x >= 1.0 - self._edge_zone_ratio
            else 1.0
        )

        total = edge * (
            confidence_score * self._weights.confidence
            + center_score * self._weights.center
            + size_score * self._weights.size
            + priority_score * self._weights.priority
        )
        return TargetScore(
            detection=detection,
            total=total,
            confidence_score=confidence_score,
            center_score=center_score,
            size_score=size_score,
            priority_score=priority_score,
            edge_multiplier=edge,
        )

    def _priority_score(self, resource_type: str) -> float:
        if not self._priorities:
            return 1.0
        max_priority = max((value for value in self._priorities.values() if value > 0), default=1.0)
        return self._clamp(self._priorities.get(resource_type, 0.0) / max_priority, 0.0, 1.0)

    @staticmethod
    def _validate_frame_size(frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame_width 和 frame_height 必须大于零")

    @staticmethod
    def _validate_weights(weights: SelectionWeights) -> None:
        values = (weights.confidence, weights.center, weights.size, weights.priority)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("评分权重必须是非负有限数")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("评分权重之和必须为 1.0")

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
