from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from core.state import BotState
from core.models import Detection
from debug.recorder import FailureRecorder
from debug.screenshot import ScreenshotSaver


def test_screenshot_saver_writes_timestamped_image(tmp_path) -> None:
    saver = ScreenshotSaver(tmp_path)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)

    path = saver.save(
        frame,
        timestamp=datetime(2026, 9, 16, 12, 34, 56, 123000),
    )

    assert path.exists()
    assert path.name == "screenshot_20260916_123456_123.png"


def test_failure_recorder_writes_image_and_json(tmp_path) -> None:
    recorder = FailureRecorder(tmp_path)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    target = Detection("wood", 1, 2, 5, 6, 0.81)

    record = recorder.save_failure(
        frame,
        state=BotState.APPROACH_TARGET,
        reason="target_lost",
        target=target,
        timestamp=datetime(2026, 9, 16, 12, 34, 56, 123000),
    )

    assert record.image_path.exists()
    assert record.metadata_path.exists()
    metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert metadata["state"] == "APPROACH_TARGET"
    assert metadata["target"] == "wood"
    assert metadata["confidence"] == 0.81
    assert metadata["reason"] == "target_lost"
