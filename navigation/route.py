"""固定路线定义与 YAML 加载。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class RouteError(ValueError):
    """路线文件或路线动作配置无效时抛出。"""


class RouteAction(str, Enum):
    """MVP 支持的固定路线动作。"""

    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    WAIT = "wait"

    @property
    def is_turn(self) -> bool:
        """返回动作是否为一次性转向。"""
        return self in (RouteAction.TURN_LEFT, RouteAction.TURN_RIGHT)


@dataclass(frozen=True)
class RouteStep:
    """一条已校验的路线动作。"""

    action: RouteAction
    duration: float = 0.0
    mouse_dx: float = 0.0

    def __post_init__(self) -> None:
        if self.action.is_turn:
            if not math.isfinite(self.mouse_dx) or self.mouse_dx <= 0:
                raise RouteError("转向动作的 mouse_dx 必须大于零")
            if self.duration != 0.0:
                raise RouteError("转向动作不能配置 duration")
        elif self.duration <= 0:
            raise RouteError("移动或等待动作的 duration 必须大于零")
        elif not math.isfinite(self.duration):
            raise RouteError("移动或等待动作的 duration 必须是有限数")
        elif self.mouse_dx != 0.0:
            raise RouteError("移动或等待动作不能配置 mouse_dx")


@dataclass(frozen=True)
class Route:
    """由多个动作组成的固定路线。"""

    steps: tuple[RouteStep, ...]
    loop: bool = True

    def __post_init__(self) -> None:
        if not self.steps:
            raise RouteError("路线至少需要一条动作")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Route":
        """从 YAML 文件加载路线。"""
        route_path = Path(path)
        if not route_path.is_file():
            raise RouteError(f"路线文件不存在: {route_path}")
        try:
            with route_path.open("r", encoding="utf-8") as route_file:
                data = yaml.safe_load(route_file)
        except OSError as exc:
            raise RouteError(f"无法读取路线文件: {route_path}") from exc
        except yaml.YAMLError as exc:
            raise RouteError(f"YAML 路线格式错误: {route_path}") from exc
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "Route":
        """从映射对象加载并校验路线。"""
        if not isinstance(data, Mapping):
            raise RouteError("路线根节点必须是 YAML 映射")

        raw_steps = data.get("route")
        if not isinstance(raw_steps, list):
            raise RouteError("路线必须包含 route 列表")

        steps = tuple(cls._parse_step(raw_step, index) for index, raw_step in enumerate(raw_steps))
        loop = data.get("loop", True)
        if not isinstance(loop, bool):
            raise RouteError("loop 必须是布尔值")
        return cls(steps=steps, loop=loop)

    @staticmethod
    def _parse_step(raw_step: Any, index: int) -> RouteStep:
        if not isinstance(raw_step, Mapping):
            raise RouteError(f"路线动作 #{index} 必须是映射")

        raw_action = raw_step.get("action")
        if not isinstance(raw_action, str):
            raise RouteError(f"路线动作 #{index} 缺少 action")
        try:
            action = RouteAction(raw_action.strip().lower())
        except ValueError as exc:
            valid = ", ".join(item.value for item in RouteAction)
            raise RouteError(f"路线动作 #{index} 不支持，可选值: {valid}") from exc

        if action.is_turn:
            # 兼容原始规格中的 pixels，同时在内部统一为 mouse_dx。
            raw_dx = raw_step.get("mouse_dx", raw_step.get("pixels"))
            if (
                not isinstance(raw_dx, (int, float))
                or not math.isfinite(raw_dx)
                or raw_dx <= 0
            ):
                raise RouteError(f"路线动作 #{index} 的 mouse_dx/pixels 必须大于零")
            return RouteStep(action=action, mouse_dx=float(raw_dx))

        raw_duration = raw_step.get("duration")
        if (
            not isinstance(raw_duration, (int, float))
            or not math.isfinite(raw_duration)
            or raw_duration <= 0
        ):
            raise RouteError(f"路线动作 #{index} 的 duration 必须大于零")
        return RouteStep(action=action, duration=float(raw_duration))
