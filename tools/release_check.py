"""执行发布前的静态、配置和测试验收。"""

from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

from core.config import load_config
from core.diagnostics import validate_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "main.py",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "config/default.yaml",
    "config/routes/test_route.yaml",
    "actions/controller.py",
    "actions/hotkeys.py",
    "capture/window.py",
    "capture/capture.py",
    "core/engine.py",
    "core/diagnostics.py",
    "core/soak.py",
    "debug/overlay.py",
    "debug/recorder.py",
    "debug/screenshot.py",
    "navigation/route.py",
    "navigation/recovery.py",
    "vision/resource_detector.py",
    "vision/prompt_detector.py",
    "vision/ui_detector.py",
    "vision/template_tools.py",
    "start_debug.bat",
    "start_normal.bat",
    "start_record.bat",
)


def run_release_check(*, skip_tests: bool = False) -> int:
    """返回 0 表示发布检查通过。"""
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    if missing:
        print("缺少发布文件:")
        print("\n".join(f"- {path}" for path in missing))
        return 1

    config = load_config(PROJECT_ROOT / "config" / "default.yaml")
    report = validate_runtime_config(config, base_dir=PROJECT_ROOT)
    if not report.ok:
        print(report.format())
        return 1

    if not compileall.compile_dir(
        str(PROJECT_ROOT),
        quiet=1,
        maxlevels=10,
    ):
        print("Python 编译检查失败")
        return 1

    if not skip_tests:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return result.returncode

    print("RELEASE CHECK PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="运行发布前检查")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    try:
        return run_release_check(skip_tests=args.skip_tests)
    except (OSError, ValueError) as exc:
        print(f"发布检查失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
