"""Loguru 日志配置。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def configure_logging(
    directory: str | Path,
    *,
    level: str = "INFO",
    rotation: str = "10 MB",
) -> list[Any]:
    """配置控制台和文件日志，返回 Loguru sink 标识。"""
    log_directory = Path(directory)
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.remove()
    sinks = [logger.add(sys.stderr, level=level, enqueue=False)]
    sinks.append(
        logger.add(
            str(log_directory / "heartopia_bot_{time:YYYYMMDD}.log"),
            level=level,
            rotation=rotation,
            encoding="utf-8",
            enqueue=False,
        )
    )
    return sinks
