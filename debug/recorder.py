"""失败样本截图与 JSON 证据记录。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from core.models import Detection
from .screenshot import ScreenshotSaver


class RecorderError(RuntimeError):
    """失败样本记录失败时抛出。"""


@dataclass(frozen=True)
class FailureRecord:
    """失败样本的图片和 JSON 路径。"""

    image_path: Path
    metadata_path: Path


class FailureRecorder:
    """把失败帧和结构化原因保存到同一目录。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        cv2_module: Any = cv2,
    ) -> None:
        self._directory = Path(directory)
        self._screenshot_saver = ScreenshotSaver(self._directory, cv2_module=cv2_module)

    def save_failure(
        self,
        frame: np.ndarray,
        *,
        state: Any,
        reason: str,
        target: Detection | None = None,
        confidence: float | None = None,
        timestamp: datetime | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> FailureRecord:
        """保存失败截图和同名 JSON 元数据。"""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("失败原因不能为空")
        moment = timestamp or datetime.now()
        reason_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", reason.strip()).strip("_")
        image_path = self._screenshot_saver.save(
            frame,
            prefix=f"failure_{reason_slug or 'unknown'}",
            timestamp=moment,
        )
        metadata_path = image_path.with_suffix(".json")
        state_name = getattr(state, "name", None) or str(state)
        payload: dict[str, Any] = {
            "state": state_name,
            "reason": reason,
            "target": target.resource_type if target else None,
            "confidence": confidence if confidence is not None else (
                target.confidence if target else None
            ),
            "bbox": (
                {
                    "x1": target.x1,
                    "y1": target.y1,
                    "x2": target.x2,
                    "y2": target.y2,
                }
                if target
                else None
            ),
            "created_at": moment.isoformat(timespec="milliseconds"),
        }
        if extra:
            payload["extra"] = dict(extra)
        try:
            metadata_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            raise RecorderError(f"保存失败样本 JSON 失败: {metadata_path}") from exc
        return FailureRecord(image_path=image_path, metadata_path=metadata_path.resolve())
