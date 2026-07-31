from dataclasses import replace

import pytest

from conveyor_bench.v1 import (
    BenchmarkConfig,
    CanonicalAction,
    JointState,
    ObjectState,
    Pose,
    StepSample,
    Twist,
)
from conveyor_bench.v1.future_labels import with_realized_future_labels


def _pose(x: float) -> Pose:
    return Pose((x, 0.0, 0.7), (1.0, 0.0, 0.0, 0.0))


def _twist(x: float) -> Twist:
    return Twist((x, 0.0, 0.0), (0.0, 0.0, 0.0))


def _sample(
    *,
    sim_step: int,
    model_tick: int,
    x: float,
    active: bool = True,
) -> StepSample:
    return StepSample(
        sim_step=sim_step,
        sim_time_s=sim_step / 400.0,
        model_tick=model_tick,
        env_id=0,
        robot_root_world=_pose(0.0),
        robot_twist_world=_twist(0.0),
        tcp_base=_pose(0.4),
        joints=JointState(("joint-1",), (0.0,), (0.0,)),
        action=CanonicalAction((0.0,) * 10),
        objects=(
            ObjectState(
                "target",
                _pose(x),
                _twist(x),
                active=active,
            ),
        ),
        left_contact_object_ids=(),
        right_contact_object_ids=(),
        camera_frames=(),
        future_object_states=(),
        phase="track",
        selected_object_id="target",
    )


def _label(sample: StepSample, horizon: int):
    return next(
        label
        for label in sample.future_object_states
        if label.horizon_steps == horizon
    )


def test_uses_latest_control_sample_as_future_model_tick_representative() -> None:
    samples = (
        _sample(sim_step=8, model_tick=0, x=0.0),
        _sample(sim_step=16, model_tick=0, x=0.1),
        _sample(sim_step=24, model_tick=1, x=1.0),
        _sample(sim_step=32, model_tick=1, x=1.1),
        _sample(sim_step=40, model_tick=2, x=2.0),
        _sample(sim_step=48, model_tick=2, x=2.1),
    )
    config = BenchmarkConfig(future_horizons_steps=(0, 1, 2))

    labeled = with_realized_future_labels(samples, config)

    assert samples[0].future_object_states == ()
    assert _label(labeled[0], 0).pose_world == _pose(0.0)
    assert _label(labeled[1], 0).pose_world == _pose(0.1)
    assert _label(labeled[0], 1).pose_world == _pose(1.1)
    assert _label(labeled[0], 1).twist_world == _twist(1.1)
    assert _label(labeled[0], 2).pose_world == _pose(2.1)


def test_masks_tail_and_inactive_states_instead_of_fabricating_them() -> None:
    samples = (
        _sample(sim_step=8, model_tick=0, x=0.0),
        _sample(sim_step=16, model_tick=1, x=1.0, active=False),
        _sample(sim_step=24, model_tick=2, x=2.0),
    )
    config = BenchmarkConfig(future_horizons_steps=(0, 1, 2))

    labeled = with_realized_future_labels(samples, config)

    inactive_future = _label(labeled[0], 1)
    assert not inactive_future.valid
    assert inactive_future.invalid_reason == "future_object_inactive"
    assert inactive_future.pose_world is None
    assert inactive_future.twist_world is None

    assert _label(labeled[0], 2).pose_world == _pose(2.0)
    assert {
        label.invalid_reason for label in labeled[1].future_object_states
    } == {"source_object_inactive"}

    tail = _label(labeled[2], 1)
    assert not tail.valid
    assert tail.invalid_reason == "future_tick_unavailable"
    assert tail.pose_world is None
    assert tail.twist_world is None


def test_rejects_non_episode_order_and_changing_object_registry() -> None:
    config = BenchmarkConfig(future_horizons_steps=(0, 1))
    reversed_ticks = (
        _sample(sim_step=8, model_tick=1, x=1.0),
        _sample(sim_step=16, model_tick=0, x=0.0),
    )

    with pytest.raises(ValueError, match="model_tick"):
        with_realized_future_labels(reversed_ticks, config)

    changed = list(
        _sample(sim_step=16, model_tick=1, x=1.0).objects
    )
    changed[0] = ObjectState(
        "other",
        changed[0].pose_world,
        changed[0].twist_world,
    )
    inconsistent = (
        _sample(sim_step=8, model_tick=0, x=0.0),
        replace(
            _sample(
                sim_step=16,
                model_tick=1,
                x=1.0,
            ),
            objects=tuple(changed),
            selected_object_id="other",
        ),
    )

    with pytest.raises(ValueError, match="object registry"):
        with_realized_future_labels(inconsistent, config)
