from pathlib import Path

from core.config import Config
from core.diagnostics import DiagnosticLevel, validate_runtime_config


def base_config(tmp_path: Path, **overrides: object) -> Config:
    data: dict[str, object] = {
        "window": {"title_contains": "心动小镇"},
        "capture": {"fps": 10},
        "control": {"dry_run": True},
        "route": {"path": "route.yaml"},
        "hotkeys": {"screenshot": "F6", "pause": "F8", "resume": "F9", "stop": "F10"},
        "debug": {
            "log_dir": "logs",
            "failure_dir": "failures",
            "screenshot_dir": "images",
        },
    }
    data.update(overrides)
    (tmp_path / "route.yaml").write_text(
        "route:\n  - action: wait\n    duration: 0.1\n",
        encoding="utf-8",
    )
    return Config(data=data)


def test_valid_config_report_is_ok(tmp_path: Path) -> None:
    report = validate_runtime_config(base_config(tmp_path), base_dir=tmp_path)

    assert report.ok is True
    assert any(issue.level is DiagnosticLevel.INFO for issue in report.issues)


def test_diagnostics_detect_invalid_route_and_hotkey(tmp_path: Path) -> None:
    config = base_config(
        tmp_path,
        route={"path": "missing.yaml"},
        hotkeys={"stop": "ESC"},
        capture={"fps": 0},
    )

    report = validate_runtime_config(config, base_dir=tmp_path)

    assert report.ok is False
    codes = {issue.code for issue in report.errors}
    assert {"ROUTE_FILE", "HOTKEYS", "CAPTURE_FPS"} <= codes


def test_opencv_detector_requires_existing_template(tmp_path: Path) -> None:
    config = base_config(
        tmp_path,
        vision={
            "detector": "opencv",
            "resource_template_paths": ["wood.png"],
        },
    )

    report = validate_runtime_config(config, base_dir=tmp_path)

    assert report.ok is False
    assert any(issue.code == "RESOURCE_TEMPLATE" for issue in report.errors)
