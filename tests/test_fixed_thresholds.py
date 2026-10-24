"""Regression: calibrated JSON loads as records, not subscriptable dictionaries."""
import pytest

from scripts.train_continuous import compute_fixed_baseline_A, compute_fixed_baseline_C
from src.context import load_threshold_patch


@pytest.mark.parametrize("compute", [compute_fixed_baseline_A, compute_fixed_baseline_C])
def test_calibrated_patch_is_accepted(compute, tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text('{"low|calm|clean": {"tau_violation": 2, "tau_near_miss": 4}}')
    result = compute(load_threshold_patch(path))
    assert 0 < result["fixed_tau_violation"] <= result["fixed_tau_near_miss"]


def test_fixed_baseline_accepts_mapping_and_record_equally(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text('{"low|calm|clean": {"tau_violation": 2, "tau_near_miss": 4}}')
    mapping = {("low", "calm", "clean"): {"tau_violation": 2, "tau_near_miss": 4}}
    assert compute_fixed_baseline_A(mapping) == compute_fixed_baseline_A(load_threshold_patch(path))
