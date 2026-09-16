"""YAML 配置加载基础设施。

Task 1 只负责安全地加载和访问 YAML 配置，不在这里实现窗口、视觉或控制
逻辑。后续模块可以在此基础上增加面向领域的配置校验。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """配置文件不存在、格式错误或根节点类型不正确时抛出。"""


@dataclass(frozen=True)
class Config:
    """已加载的只读配置。

    ``data`` 保持 YAML 的嵌套字典结构。通过 ``get`` 读取嵌套字段时使用
    点号路径，例如 ``config.get("capture.fps")``。
    """

    data: Mapping[str, Any]
    source: Path | None = None

    def get(self, path: str, default: Any = None) -> Any:
        """按点号路径读取配置，路径不存在时返回 ``default``。"""
        if not path:
            return self.data

        current: Any = self.data
        for key in path.split("."):
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path) -> Config:
    """从 YAML 文件加载配置。

    Args:
        path: YAML 配置文件路径。

    Raises:
        ConfigError: 文件不存在、无法读取、YAML 无法解析，或根节点不是映射。
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"配置文件不存在: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_data = yaml.safe_load(config_file)
    except OSError as exc:
        raise ConfigError(f"无法读取配置文件: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML 配置格式错误: {config_path}") from exc

    if raw_data is None:
        raw_data = {}
    if not isinstance(raw_data, Mapping):
        raise ConfigError("配置文件的根节点必须是 YAML 映射")

    return Config(data=dict(raw_data), source=config_path.resolve())
