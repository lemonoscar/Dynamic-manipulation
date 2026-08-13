from dataclasses import replace

import pytest

from conveyor_bench.schema import BenchmarkConfig, EvaluationConfig, PROTOCOL_VERSION


def test_v1_defaults_freeze_dynamicvla_timing_contract() -> None:
    config = BenchmarkConfig.v1()

    assert config.protocol_version == "conveyor-bench-v1" == PROTOCOL_VERSION
    assert config.physics_hz == 400
    assert config.control_hz == 50
    assert config.camera_hz == 25
    assert config.model_hz == 25
    assert config.history_offsets_steps == (-2, 0)
    assert config.m0_chunk_size == 16
    assert config.dynamicvla_chunk_size == 20
    assert config.label_offset_steps == 5
    assert config.future_horizons_steps == (0, 2, 5, 10, 20)
    assert config.chunk_size_for("m0") == 16
    assert config.chunk_size_for("dynamicvla") == 20


def test_v1_rejects_timing_that_changes_model_tick_semantics() -> None:
    config = BenchmarkConfig.v1()

    with pytest.raises(ValueError, match="control_hz"):
        replace(config, model_hz=30)
    with pytest.raises(ValueError, match="end at"):
        replace(config, history_offsets_steps=(-2, -1))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, future_horizons_steps=(0, 5, 5))
    with pytest.raises(ValueError, match="start at 0"):
        replace(config, future_horizons_steps=(2, 5))


def test_v1_rejects_unknown_profile_and_non_finite_thresholds() -> None:
    config = BenchmarkConfig.v1()

    with pytest.raises(ValueError, match="unsupported"):
        config.chunk_size_for("other")
    with pytest.raises(ValueError, match="finite"):
        EvaluationConfig(placement_dwell_s=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        EvaluationConfig(placement_dwell_s=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="require_settled_placement"):
        EvaluationConfig(require_settled_placement=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative integers"):
        replace(config, future_horizons_steps=(0, 2.0, 5))  # type: ignore[arg-type]
