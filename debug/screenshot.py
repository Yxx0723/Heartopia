"""截图保存工具。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ScreenshotError(RuntimeError):
    """截图保存失败时抛出。"""


class ScreenshotSaver:
    """将 BGR 帧保存到指定目录。"""

    def __init__(self, directory: str | Path, *, cv2_module: Any = cv2) -> None:
        self._directory = Path(directory)
        self._cv2 = cv2_module

    def save(
        self,
        frame: np.ndarray,
        *,
        prefix: str = "screenshot",
        timestamp: datetime | None = None,
    ) -> Path:
        """保存截图并返回绝对路径。"""
        self._validate_frame(frame)
        self._directory.mkdir(parents=True, exist_ok=True)
        moment = timestamp or datetime.now()
        stem = f"{prefix}_{moment:%Y%m%d_%H%M%S_%f}"[:-3]
        path = self._unique_path(stem)
        try:
            written = self._cv2.imwrite(str(path), frame)
        except Exception as exc:
            raise ScreenshotError(f"保存截图失败: {path}") from exc
        if not written:
            raise ScreenshotError(f"截图编码失败: {path}")
        return path.resolve()

    def _unique_path(self, stem: str) -> Path:
        candidate = self._directory / f"{stem}.png"
        suffix = 1
        while candidate.exists():
            candidate = self._directory / f"{stem}_{suffix:03d}.png"
            suffix += 1
        return candidate

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame 必须是 numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame 必须是 H x W x 3 的 BGR 图像")
