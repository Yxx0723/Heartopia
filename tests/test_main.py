from pathlib import Path

import pytest

import main


def test_parser_supports_required_startup_options() -> None:
    args = main.build_parser().parse_args(
        ["--debug", "--record", "--config", "custom.yaml", "--route", "route.yaml"]
    )

    assert args.debug is True
    assert args.record is True
    assert args.config == "custom.yaml"
    assert args.route == "route.yaml"


def test_help_does_not_construct_runtime() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main.main(["--help"])

    assert exc_info.value.code == 0


def test_relative_path_falls_back_to_project_root_for_missing_cwd_path() -> None:
    path = main._resolve_path("config/default.yaml")

    assert path == (Path(main.PROJECT_ROOT) / "config/default.yaml").resolve()


def test_set_nested_creates_missing_mapping() -> None:
    data: dict[str, object] = {}

    main._set_nested(data, "debug.record_frames", True)

    assert data == {"debug": {"record_frames": True}}
