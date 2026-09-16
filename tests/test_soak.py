import pytest

from core.soak import run_offline_soak


def test_offline_soak_processes_twenty_targets_and_faults() -> None:
    report = run_offline_soak(20)

    assert report.ok is True
    assert report.collected_targets == 20
    assert report.recovered_faults == ("target_lost", "recovery", "blocking_ui")


def test_offline_soak_rejects_non_positive_target_count() -> None:
    with pytest.raises(ValueError, match="target_count"):
        run_offline_soak(0)
