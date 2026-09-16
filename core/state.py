"""机器人状态枚举。"""

from __future__ import annotations

from enum import Enum, auto


class BotState(Enum):
    """自动采集流程状态。"""

    INIT = auto()
    PATROL = auto()
    TARGET_FOUND = auto()
    ALIGN_TARGET = auto()
    APPROACH_TARGET = auto()
    INTERACT = auto()
    VERIFY = auto()
    RECOVERY = auto()
    PAUSED = auto()
    STOPPED = auto()
