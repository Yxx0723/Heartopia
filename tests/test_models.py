import numpy as np

from core.models import Detection, Frame, Perception


def test_detection_center() -> None:
    detection = Detection("wood", 10, 20, 31, 42, 0.88)

    assert detection.center == (20, 31)


def test_frame_dimensions() -> None:
    frame = Frame(np.zeros((1080, 1920, 3), dtype=np.uint8), 12.5, 7)

    assert frame.height == 1080
    assert frame.width == 1920
    assert frame.sequence == 7


def test_perception_defaults() -> None:
    perception = Perception(resources=[], interaction_prompt=False, motion_score=0.0)

    assert perception.inventory_full is False
    assert perception.abnormal_ui is False
