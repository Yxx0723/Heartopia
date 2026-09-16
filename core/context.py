"""状态机运行上下文。"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import Detection
from core.state import BotState


@dataclass
class BotContext:
    """状态机共享的可变运行数据。"""

    state: BotState = BotState.INIT
    current_target: Detection | None = None
    previous_target: Detection | None = None
    state_enter_time: float = 0.0
    interaction_retries: int = 0
    recovery_attempts: int = 0
    route_index: int = 0
    total_collected: int = 0

    # 这些字段是状态机内部的时间性辅助数据，不改变公开核心字段语义。
    lost_target_frames: int = 0
    resume_state: BotState | None = None
