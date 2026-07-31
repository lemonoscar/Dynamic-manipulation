from dataclasses import replace

import pytest

from conveyor_bench.v1 import (
    ActionChunkProfile,
    ActionChunkTrace,
    BenchmarkConfig,
    CameraFrameRef,
    CanonicalAction,
    EpisodeManifest,
    FutureObjectState,
    GoalZone,
    JointState,
    ObjectInstance,
    ObjectState,
    Pose,
    RobotMode,
    StepSample,
    TaskManifest,
    TaskType,
    Twist,
)


def pose(xyz=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(xyz, (1.0, 0.0, 0.0, 0.0))


def zero_twist() -> Twist:
    return Twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def zero_action() -> CanonicalAction:
    return CanonicalAction((0.0,) * 10)


def task(robot_mode: RobotMode = RobotMode.FIXED_BASE) -> TaskManifest:
    return TaskManifest(
        task_id="sort-001",
        task_type=TaskType.DYNAMIC_SORT,
        robot_mode=robot_mode,
        instruction="put the red can in the left bin",
        objects=(
            ObjectInstance("can-001", "asset-can", "can", "zone-left"),
            ObjectInstance("box-001", "asset-box", "box"),
        ),
        goal_zones=(GoalZone("zone-left", (0.2, -0.2, 0.6), (0.5, 0.2, 0.9)),),
        scored_object_ids=("can-001",),
        seed=7,
        belt_speed_mps=0.1,
        belt_surface_z_m=0.67,
        transport_direction_xyz=(0.0, -1.0, 0.0),
        exit_plane_point_xyz=(0.7, -0.57, 0.67),
    )


def future_labels(obj: ObjectState) -> tuple[FutureObjectState, ...]:
    return tuple(
        FutureObjectState(
            instance_id=obj.instance_id,
            horizon_steps=horizon,
            valid=True,
            pose_world=obj.pose_world,
            twist_world=obj.twist_world,
        )
        for horizon in BenchmarkConfig.v1().future_horizons_steps
    )


def sample(action: CanonicalAction | None = None) -> StepSample:
    obj = ObjectState("can-001", pose((0.0, 0.0, 0.7)), zero_twist())
    return StepSample(
        sim_step=0,
        sim_time_s=0.0,
        model_tick=0,
        env_id=0,
        robot_root_world=pose(),
        robot_twist_world=zero_twist(),
        tcp_base=pose((0.4, 0.0, 0.5)),
        joints=JointState(("joint-1",), (0.0,), (0.0,)),
        action=action or zero_action(),
        objects=(obj,),
        left_contact_object_ids=(),
        right_contact_object_ids=(),
        camera_frames=(CameraFrameRef("head", 0, 0.0, "rgb/head/000000.png"),),
        future_object_states=future_labels(obj),
        phase="track",
        selected_object_id="can-001",
    )


def test_task_registry_rejects_unknown_or_missing_goal_zone_identity() -> None:
    base = task()

    with pytest.raises(ValueError, match="unknown goal zone"):
        replace(
            base,
            objects=(
                ObjectInstance("can-001", "asset-can", "can", "zone-missing"),
            ),
        )
    with pytest.raises(ValueError, match="requires a goal_zone_id"):
        replace(
            base,
            objects=(ObjectInstance("can-001", "asset-can", "can"),),
        )
    with pytest.raises(ValueError, match="exactly one"):
        replace(
            base,
            objects=(
                *base.objects,
                ObjectInstance("cup-001", "asset-cup", "cup", "zone-left"),
            ),
            scored_object_ids=("can-001", "cup-001"),
        )


def test_canonical_action_is_exactly_ten_finite_values() -> None:
    with pytest.raises(ValueError, match="10"):
        CanonicalAction((0.0,) * 9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CanonicalAction((0.0,) * 9 + (float("nan"),))
    with pytest.raises(ValueError, match="gripper"):
        CanonicalAction((0.0,) * 9 + (1.1,))
    with pytest.raises(ValueError, match="finite"):
        CanonicalAction(("0",) * 10)  # type: ignore[arg-type]


def test_fixed_base_rejects_nonzero_base_body_action() -> None:
    moving_base = CanonicalAction((0.1, 0.0, 0.0) + (0.0,) * 7)

    with pytest.raises(ValueError, match="fixed_base"):
        sample(moving_base).validate_against(task(), BenchmarkConfig.v1())

    sample(moving_base).validate_against(
        task(RobotMode.WHOLE_BODY_POLICY), BenchmarkConfig.v1()
    )


def test_step_rejects_unknown_contacts_and_incomplete_future_horizons() -> None:
    base = sample()

    with pytest.raises(ValueError, match="unknown per-step"):
        replace(base, left_contact_object_ids=("box-001",))
    with pytest.raises(ValueError, match="future horizons"):
        replace(
            base,
            future_object_states=base.future_object_states[:-1],
        ).validate_against(task(), BenchmarkConfig.v1())


def test_action_chunk_accounts_for_stale_and_discarded_actions() -> None:
    actions = tuple(zero_action() for _ in range(16))
    trace = ActionChunkTrace(
        chunk_id="chunk-001",
        profile=ActionChunkProfile.M0,
        source_observation_tick=0,
        source_observation_time_s=0.0,
        valid_from_tick=0,
        valid_until_tick=16,
        execute_from_tick=2,
        execute_until_tick=16,
        actions=actions,
        stale=True,
        discarded_action_count=2,
        discard_reason="inference_latency",
    )

    trace.validate_against(BenchmarkConfig.v1())
    with pytest.raises(ValueError, match="account"):
        replace(trace, discarded_action_count=1)
    with pytest.raises(ValueError, match="require 20"):
        replace(trace, profile=ActionChunkProfile.DYNAMICVLA).validate_against(
            BenchmarkConfig.v1()
        )


def test_nested_protocol_values_and_asset_hash_identities_are_strict() -> None:
    base_task = task()
    with pytest.raises(ValueError, match="ObjectInstance"):
        replace(base_task, objects=(zero_action(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ObjectState"):
        replace(sample(), objects=(zero_action(),))  # type: ignore[arg-type]

    manifest = EpisodeManifest(
        episode_id="episode-001",
        run_id="run-001",
        protocol_version="conveyor-bench-v1",
        task=base_task,
        created_at_utc="2026-07-30T00:00:00+00:00",
        asset_hashes={
            "go2_x5": "sha256:" + "1" * 64,
            "conveyor_station_v1": "2" * 64,
            "locomotion_policy": "sha256:" + "3" * 64,
        },
    )
    assert set(manifest.asset_hashes) == {
        "go2_x5",
        "conveyor_station_v1",
        "locomotion_policy",
    }
    with pytest.raises(ValueError, match="SHA-256"):
        replace(manifest, asset_hashes={"asset-can": ""})
