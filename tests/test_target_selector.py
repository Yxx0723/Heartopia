from __future__ import annotations

import pytest

from core.models import Detection
from vision.target_selector import SelectionWeights, TargetSelector


FRAME_WIDTH = 200
FRAME_HEIGHT = 100


def test_selector_prefers_center_when_candidates_are_similar() -> None:
    selector = TargetSelector()
    edge = Detection("wood", 0, 40, 20, 60, 0.90)
    center = Detection("wood", 90, 40, 110, 60, 0.85)

    assert selector.select([edge, center], FRAME_WIDTH, FRAME_HEIGHT) == center


def test_selector_applies_priority_and_edge_penalty() -> None:
    selector = TargetSelector(priorities={"wood": 1.0, "ore": 2.0})
    wood = Detection("wood", 85, 40, 115, 60, 0.85)
    ore = Detection("ore", 85, 40, 115, 60, 0.80)

    assert selector.select([wood, ore], FRAME_WIDTH, FRAME_HEIGHT) == ore
    edge_score = selector.score_detection(
        Detection("ore", 0, 40, 20, 60, 0.80),
        FRAME_WIDTH,
        FRAME_HEIGHT,
    )
    center_score = selector.score_detection(ore, FRAME_WIDTH, FRAME_HEIGHT)
    assert edge_score.edge_multiplier == 0.5
    assert edge_score.total < center_score.total


def test_selector_returns_ranked_score_components() -> None:
    selector = TargetSelector()
    target = Detection("wood", 90, 40, 110, 60, 0.8)

    score = selector.score_detection(target, FRAME_WIDTH, FRAME_HEIGHT)

    assert score.detection == target
    assert 0.0 <= score.total <= 1.0
    assert score.center_score == pytest.approx(1.0)
    assert score.priority_score == pytest.approx(1.0)


def test_selector_handles_empty_candidates_and_invalid_configuration() -> None:
    assert TargetSelector().select([], FRAME_WIDTH, FRAME_HEIGHT) is None
    with pytest.raises(ValueError, match="权重之和"):
        TargetSelector(weights=SelectionWeights(confidence=1.0))
    with pytest.raises(ValueError, match="frame_width"):
        TargetSelector().rank([], 0, FRAME_HEIGHT)
