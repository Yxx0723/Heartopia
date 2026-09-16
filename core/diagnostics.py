"""运行前配置和环境诊断。

诊断模块保持只读：它只检查配置值、文件存在性和可推导的运行条件，
不查找窗口、不创建输入后端，也不发送任何输入。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from actions.hotkeys import HotkeyManager
from core.config import Config


class DiagnosticLevel(str, Enum):
    """诊断项目严重程度。"""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class DiagnosticIssue:
    """单条可读诊断信息。"""

    level: DiagnosticLevel
    code: str
    message: str


@dataclass(frozen=True)
class DiagnosticReport:
    """配置诊断结果。"""

    issues: tuple[DiagnosticIssue, ...]

    @property
    def ok(self) -> bool:
        """没有 ERROR 时返回 True。"""
        return not any(issue.level is DiagnosticLevel.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[DiagnosticIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.level is DiagnosticLevel.ERROR
        )

    def format(self) -> str:
        """生成适合终端输出的诊断摘要。"""
        if not self.issues:
            return "配置诊断通过"
        return "\n".join(
            f"[{issue.level.value}] {issue.code}: {issue.message}"
            for issue in self.issues
        )


def validate_runtime_config(
    config: Config,
    *,
    base_dir: str | Path | None = None,
) -> DiagnosticReport:
    """检查运行时配置，不初始化 Windows 资源。"""
    root = Path(base_dir or Path.cwd()).resolve()
    issues: list[DiagnosticIssue] = []

    title = config.get("window.title_contains")
    if not isinstance(title, str) or not title.strip():
        issues.append(_error("WINDOW_TITLE", "window.title_contains 不能为空"))

    fps = config.get("capture.fps", 10)
    if not _positive_finite(fps) or float(fps) > 120:
        issues.append(_error("CAPTURE_FPS", "capture.fps 必须在 0 到 120 之间"))

    detector = str(config.get("vision.detector", "mock")).lower()
    if detector not in {"mock", "opencv"}:
        issues.append(_error("DETECTOR", f"不支持的 vision.detector: {detector}"))
    if detector == "opencv":
        _check_template_paths(
            config.get("vision.resource_template_paths", []),
            root,
            "RESOURCE_TEMPLATE",
            issues,
            required=True,
        )
    _check_template_paths(
        config.get("vision.prompt_template_paths", []),
        root,
        "PROMPT_TEMPLATE",
        issues,
    )
    _check_template_paths(
        config.get("ui.inventory_full_template_paths", []),
        root,
        "INVENTORY_TEMPLATE",
        issues,
    )
    _check_template_paths(
        config.get("ui.abnormal_template_paths", []),
        root,
        "ABNORMAL_UI_TEMPLATE",
        issues,
    )

    route_path = config.get("route.path")
    if route_path is not None and not _resolve_path(route_path, root).is_file():
        issues.append(
            _error("ROUTE_FILE", f"路线文件不存在: {_resolve_path(route_path, root)}")
        )

    max_runtime = config.get("safety.max_runtime_minutes", 60)
    if not _positive_finite(max_runtime):
        issues.append(
            _error("MAX_RUNTIME", "safety.max_runtime_minutes 必须是正的有限数")
        )

    dry_run = config.get("control.dry_run", True)
    if not isinstance(dry_run, bool):
        issues.append(_error("DRY_RUN", "control.dry_run 必须是布尔值"))
    elif not dry_run:
        issues.append(
            _warning(
                "LIVE_INPUT",
                "真实输入已开启；请确认游戏窗口、前台保护和使用权限",
            )
        )

    _check_hotkeys(config.get("hotkeys", {}), issues)
    _check_output_directories(config, root, issues)
    issues.append(
        DiagnosticIssue(
            DiagnosticLevel.INFO,
            "SAFETY_SCOPE",
            "仅使用窗口像素与普通键鼠输入，不读取游戏内存或网络数据",
        )
    )
    return DiagnosticReport(tuple(issues))


def _check_template_paths(
    paths: Any,
    root: Path,
    code: str,
    issues: list[DiagnosticIssue],
    *,
    required: bool = False,
) -> None:
    if paths is None:
        paths = []
    if not isinstance(paths, Iterable) or isinstance(paths, (str, bytes)):
        issues.append(_error(code, "模板路径必须是列表"))
        return
    resolved = [_resolve_path(path, root) for path in paths]
    if required and not resolved:
        issues.append(_error(code, "OpenCV detector 至少需要一个资源模板"))
    for path in resolved:
        if not path.is_file():
            issues.append(_error(code, f"模板文件不存在: {path}"))


def _check_hotkeys(bindings: Any, issues: list[DiagnosticIssue]) -> None:
    if not isinstance(bindings, Mapping):
        issues.append(_error("HOTKEYS", "hotkeys 必须是映射"))
        return
    seen: dict[str, str] = {}
    for event in HotkeyManager.EVENTS:
        value = bindings.get(event)
        if value is None:
            continue
        key = str(value).strip().upper()
        if key not in HotkeyManager.VIRTUAL_KEYS:
            issues.append(_error("HOTKEYS", f"不支持的热键: {key}"))
        if key in seen:
            issues.append(
                _error("HOTKEY_DUPLICATE", f"{event} 与 {seen[key]} 使用了同一个热键 {key}")
            )
        seen[key] = event


def _check_output_directories(
    config: Config,
    root: Path,
    issues: list[DiagnosticIssue],
) -> None:
    for field in ("debug.log_dir", "debug.failure_dir", "debug.screenshot_dir"):
        value = config.get(field)
        if value is None:
            continue
        path = _resolve_path(value, root)
        if path.exists() and not path.is_dir():
            issues.append(_error("OUTPUT_DIR", f"{field} 不是目录: {path}"))


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number > 0 and math.isfinite(number)


def _error(code: str, message: str) -> DiagnosticIssue:
    return DiagnosticIssue(DiagnosticLevel.ERROR, code, message)


def _warning(code: str, message: str) -> DiagnosticIssue:
    return DiagnosticIssue(DiagnosticLevel.WARNING, code, message)


__all__ = [
    "DiagnosticIssue",
    "DiagnosticLevel",
    "DiagnosticReport",
    "validate_runtime_config",
]
