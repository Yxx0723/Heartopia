"""心动小镇视觉采集程序的命令行入口。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger

from actions.hotkeys import HotkeyError, HotkeyManager
from core.config import Config, ConfigError, load_config
from core.diagnostics import DiagnosticReport, validate_runtime_config
from core.engine import BotEngine, EngineError
from core.logging_setup import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """创建命令行解析器；不触发窗口、截图或输入初始化。"""
    parser = argparse.ArgumentParser(
        description="基于窗口截图和普通键鼠输入的心动小镇资源采集程序"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用 Debug Overlay，并强制使用 dry-run 输入",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="YAML 配置路径（默认: config/default.yaml）",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="按配置间隔保存当前画面到 dataset/images/",
    )
    parser.add_argument(
        "--route",
        default=None,
        help="覆盖配置中的固定路线 YAML 路径",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只执行配置诊断，不查找窗口、不启动引擎",
    )
    parser.add_argument(
        "--max-updates",
        type=_positive_int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def build_engine(args: argparse.Namespace) -> BotEngine:
    """加载配置、配置日志并装配运行时引擎。"""
    config_path, runtime_config = load_runtime_config(args)
    report = validate_runtime_config(runtime_config, base_dir=PROJECT_ROOT)
    if not report.ok:
        raise ConfigError(f"配置诊断失败:\n{report.format()}")
    log_directory = _resolve_config_path(
        runtime_config.get("debug.log_dir", "logs")
    )
    configure_logging(
        log_directory,
        level=str(runtime_config.get("debug.log_level", "INFO")),
    )
    logger.info("CONFIG loaded={}", config_path)
    logger.info("DRY RUN={}", runtime_config.get("control.dry_run", True))

    hotkeys = HotkeyManager(runtime_config.get("hotkeys", {}))
    return BotEngine(config=runtime_config, hotkeys=hotkeys)


def load_runtime_config(args: argparse.Namespace) -> tuple[Path, Config]:
    """加载配置并应用命令行覆盖，不初始化运行时资源。"""
    config_path = _resolve_path(args.config)
    config = load_config(config_path)
    data = deepcopy(dict(config.data))

    if args.debug:
        _set_nested(data, "debug.overlay", True)
        _set_nested(data, "control.dry_run", True)
    if args.record:
        _set_nested(data, "debug.record_frames", True)
    if args.route:
        _set_nested(data, "route.path", str(_resolve_path(args.route)))
    for path_key in (
        "debug.log_dir",
        "debug.failure_dir",
        "debug.screenshot_dir",
    ):
        configured_path = _get_nested(data, path_key)
        if configured_path is not None:
            _set_nested(data, path_key, str(_resolve_path(configured_path)))
    for paths_key in (
        "vision.resource_template_paths",
        "vision.prompt_template_paths",
        "ui.inventory_full_template_paths",
        "ui.abnormal_template_paths",
    ):
        configured_paths = _get_nested(data, paths_key)
        if isinstance(configured_paths, (list, tuple)):
            _set_nested(
                data,
                paths_key,
                [str(_resolve_path(path)) for path in configured_paths],
            )

    return config_path, Config(data=data, source=config.source)


def main(argv: Sequence[str] | None = None) -> int:
    """运行引擎并在顶层统一处理异常。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    engine: BotEngine | None = None
    try:
        if args.check:
            _config_path, config = load_runtime_config(args)
            report = validate_runtime_config(config, base_dir=PROJECT_ROOT)
            print(report.format())
            return 0 if report.ok else 2
        engine = build_engine(args)
        engine.start(max_updates=args.max_updates)
    except KeyboardInterrupt:
        logger.info("ENGINE interrupted by user")
        if engine is not None:
            engine.stop()
    except (ConfigError, EngineError, HotkeyError, OSError, ValueError):
        logger.exception("PROGRAM FAILED")
        if engine is not None:
            engine.stop()
        return 1
    except Exception:
        logger.exception("UNEXPECTED PROGRAM FAILURE")
        if engine is not None:
            engine.stop()
        return 1
    return 0


def _resolve_path(value: str | Path) -> Path:
    """解析命令行文件路径，兼容从项目根目录外启动。"""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / candidate).resolve()


def _resolve_config_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return _resolve_path(candidate)


def _set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current: dict[str, Any] = data
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, Mapping):
            child = {}
            current[key] = child
        current = child  # type: ignore[assignment]
    current[keys[-1]] = value


def _get_nested(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
