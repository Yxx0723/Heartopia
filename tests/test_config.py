from pathlib import Path

import pytest

from core.config import ConfigError, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")

    assert config.get("window.title_contains") == "心动小镇"
    assert config.get("capture.fps") == 10
    assert config.get("control.dry_run") is True
    assert config.source == (PROJECT_ROOT / "config" / "default.yaml").resolve()


def test_config_get_returns_default_for_missing_path() -> None:
    config = load_config(PROJECT_ROOT / "config" / "default.yaml")

    assert config.get("missing.value", "fallback") == "fallback"


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="不存在"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="根节点"):
        load_config(config_path)
