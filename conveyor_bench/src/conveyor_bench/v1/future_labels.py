"""Realized object-future supervision for buffered V1 episodes."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .config import BenchmarkConfig
from .protocol import FutureObjectState, ObjectState, StepSample

SOURCE_OBJECT_INACTIVE = "source_object_inactive"
FUTURE_TICK_UNAVAILABLE = "future_tick_unavailable"
FUTURE_OBJECT_INACTIVE = "future_object_inactive"


def build_realized_future_labels(
    samples: Sequence[StepSample],
    config: BenchmarkConfig | None = None,
) -> tuple[tuple[FutureObjectState, ...], ...]:
    """Build future labels aligned one-to-one with buffered control samples.

    Horizon zero uses the current control sample.  Positive horizons use the
    latest control sample at the requested model tick, matching the V1 exporter
    representative.  No state is extrapolated.
    """

    resolved_samples = tuple(samples)
    if not resolved_samples:
        return ()
    resolved_config = config or BenchmarkConfig.v1()
    _validate_episode_samples(resolved_samples)

    representatives: dict[int, StepSample] = {}
    for sample in resolved_samples:
        representatives[sample.model_tick] = sample
    representative_states = {
        tick: {obj.instance_id: obj for obj in representative.objects}
        for tick, representative in representatives.items()
    }

    return tuple(
        _labels_for_sample(
            sample,
            representative_states,
            resolved_config,
        )
        for sample in resolved_samples
    )


def with_realized_future_labels(
    samples: Sequence[StepSample],
    config: BenchmarkConfig | None = None,
) -> tuple[StepSample, ...]:
    """Return immutable sample copies carrying realized future labels."""

    resolved_samples = tuple(samples)
    labels = build_realized_future_labels(resolved_samples, config)
    return tuple(
        replace(sample, future_object_states=sample_labels)
        for sample, sample_labels in zip(
            resolved_samples,
            labels,
            strict=True,
        )
    )


def _validate_episode_samples(samples: tuple[StepSample, ...]) -> None:
    if any(not isinstance(sample, StepSample) for sample in samples):
        raise ValueError("samples must contain only StepSample values")

    env_id = samples[0].env_id
    object_ids = {obj.instance_id for obj in samples[0].objects}
    previous_sim_step: int | None = None
    previous_sim_time_s: float | None = None
    previous_model_tick: int | None = None
    for sample in samples:
        if sample.env_id != env_id:
            raise ValueError("all buffered samples must belong to one env_id")
        if {obj.instance_id for obj in sample.objects} != object_ids:
            raise ValueError(
                "the per-episode object registry cannot change between samples"
            )
        if (
            previous_sim_step is not None
            and sample.sim_step <= previous_sim_step
        ):
            raise ValueError("sim_step must increase strictly")
        if (
            previous_sim_time_s is not None
            and sample.sim_time_s <= previous_sim_time_s
        ):
            raise ValueError("sim_time_s must increase strictly")
        if (
            previous_model_tick is not None
            and sample.model_tick < previous_model_tick
        ):
            raise ValueError("model_tick cannot decrease")
        previous_sim_step = sample.sim_step
        previous_sim_time_s = sample.sim_time_s
        previous_model_tick = sample.model_tick


def _labels_for_sample(
    sample: StepSample,
    representative_states: dict[int, dict[str, ObjectState]],
    config: BenchmarkConfig,
) -> tuple[FutureObjectState, ...]:
    labels: list[FutureObjectState] = []
    for source_state in sample.objects:
        for horizon in config.future_horizons_steps:
            labels.append(
                _label_for_horizon(
                    source_state=source_state,
                    source_tick=sample.model_tick,
                    horizon=horizon,
                    representative_states=representative_states,
                )
            )
    return tuple(labels)


def _label_for_horizon(
    *,
    source_state: ObjectState,
    source_tick: int,
    horizon: int,
    representative_states: dict[int, dict[str, ObjectState]],
) -> FutureObjectState:
    if not source_state.active:
        return _invalid_label(
            source_state.instance_id,
            horizon,
            SOURCE_OBJECT_INACTIVE,
        )

    if horizon == 0:
        realized_state = source_state
    else:
        future_states = representative_states.get(source_tick + horizon)
        if future_states is None:
            return _invalid_label(
                source_state.instance_id,
                horizon,
                FUTURE_TICK_UNAVAILABLE,
            )
        realized_state = future_states[source_state.instance_id]
        if not realized_state.active:
            return _invalid_label(
                source_state.instance_id,
                horizon,
                FUTURE_OBJECT_INACTIVE,
            )

    return FutureObjectState(
        instance_id=source_state.instance_id,
        horizon_steps=horizon,
        valid=True,
        pose_world=realized_state.pose_world,
        twist_world=realized_state.twist_world,
    )


def _invalid_label(
    instance_id: str,
    horizon: int,
    reason: str,
) -> FutureObjectState:
    return FutureObjectState(
        instance_id=instance_id,
        horizon_steps=horizon,
        valid=False,
        pose_world=None,
        twist_world=None,
        invalid_reason=reason,
    )


__all__ = [
    "FUTURE_OBJECT_INACTIVE",
    "FUTURE_TICK_UNAVAILABLE",
    "SOURCE_OBJECT_INACTIVE",
    "build_realized_future_labels",
    "with_realized_future_labels",
]
