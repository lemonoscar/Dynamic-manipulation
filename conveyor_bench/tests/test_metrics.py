from dataclasses import replace

import pytest

from conveyor_bench.schema import (
    BenchmarkConfig,
    CanonicalAction,
    EvaluationConfig,
    FailureReason,
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
    evaluate_episode,
)


def pose(xyz=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(xyz, (1.0, 0.0, 0.0, 0.0))


def twist(linear=(0.0, 0.0, 0.0)) -> Twist:
    return Twist(linear, (0.0, 0.0, 0.0))


def task(task_type: TaskType = TaskType.DYNAMIC_SORT) -> TaskManifest:
    objects = (
        ObjectInstance("target-1", "asset-can", "can", "zone-a"),
        ObjectInstance(
            "target-2",
            "asset-box",
            "box",
            "zone-b" if task_type is TaskType.CONTINUOUS_SORT else None,
        ),
        ObjectInstance("distractor", "asset-cup", "cup"),
    )
    return TaskManifest(
        task_id="sort-task",
        task_type=task_type,
        robot_mode=RobotMode.FIXED_BASE,
        instruction="sort the requested objects",
        objects=objects,
        goal_zones=(
            GoalZone("zone-a", (0.0, 0.0, 0.5), (1.0, 1.0, 1.0)),
            GoalZone("zone-b", (1.1, 0.0, 0.5), (2.0, 1.0, 1.0)),
        ),
        scored_object_ids=(
            ("target-1", "target-2")
            if task_type is TaskType.CONTINUOUS_SORT
            else ("target-1",)
        ),
        seed=1,
        belt_speed_mps=0.1,
        belt_surface_z_m=0.67,
        transport_direction_xyz=(0.0, -1.0, 0.0),
        exit_plane_point_xyz=(0.0, -0.6, 0.67),
        max_duration_s=2.0,
    )


def object_state(
    instance_id: str,
    xyz: tuple[float, float, float],
    *,
    held: bool = False,
    linear=(0.0, 0.0, 0.0),
    crossed_exit: bool = False,
) -> ObjectState:
    return ObjectState(
        instance_id,
        pose(xyz),
        twist(linear),
        in_gripper=held,
        crossed_exit=crossed_exit,
    )


def labels(objects: tuple[ObjectState, ...]) -> tuple[FutureObjectState, ...]:
    return tuple(
        FutureObjectState(
            instance_id=obj.instance_id,
            horizon_steps=horizon,
            valid=True,
            pose_world=obj.pose_world,
            twist_world=obj.twist_world,
        )
        for obj in objects
        for horizon in BenchmarkConfig.v1().future_horizons_steps
    )


def sample(
    step: int,
    time_s: float,
    objects: tuple[ObjectState, ...],
    *,
    robot_fallen: bool = False,
) -> StepSample:
    held_ids = tuple(obj.instance_id for obj in objects if obj.in_gripper)
    return StepSample(
        sim_step=step,
        sim_time_s=time_s,
        model_tick=step,
        env_id=0,
        robot_root_world=pose(),
        robot_twist_world=twist(),
        tcp_base=pose((0.4, 0.0, 0.5)),
        joints=JointState(("joint-1",), (0.0,), (0.0,)),
        action=CanonicalAction((0.0,) * 10),
        objects=objects,
        left_contact_object_ids=held_ids,
        right_contact_object_ids=held_ids,
        camera_frames=(),
        future_object_states=labels(objects),
        phase="place",
        selected_object_id=objects[0].instance_id,
        robot_fallen=robot_fallen,
    )


def test_dynamic_sort_requires_release_in_correct_zone_and_settled_dwell() -> None:
    target_held = object_state("target-1", (0.5, 0.5, 0.7), held=True)
    target_released = replace(target_held, in_gripper=False)
    result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [
            sample(0, 0.0, (target_held,)),
            sample(1, 0.1, (target_released,)),
            sample(2, 0.6, (target_released,)),
        ],
    )

    assert result.success
    assert result.failure_reason is FailureReason.NONE
    assert result.metrics["completion_time_s"] == pytest.approx(0.6)
    assert (
        result.metrics["object_outcomes"]["target-1"]["status"]
        == "sorted_correct"
    )


def test_wrong_zone_and_unsettled_placement_are_distinct_failures() -> None:
    held = object_state("target-1", (0.5, 0.5, 0.7), held=True)
    wrong_zone = object_state("target-1", (1.5, 0.5, 0.7))
    moving_in_goal = object_state(
        "target-1", (0.5, 0.5, 0.7), linear=(0.03, 0.0, 0.0)
    )

    wrong_result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [sample(0, 0.0, (held,)), sample(1, 0.1, (wrong_zone,))],
    )
    unsettled_result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [sample(0, 0.0, (held,)), sample(1, 0.1, (moving_in_goal,))],
    )

    assert wrong_result.failure_reason is FailureReason.WRONG_ZONE
    assert unsettled_result.failure_reason is FailureReason.PLACEMENT_NOT_SETTLED


def test_in_bin_mode_accepts_a_released_moving_object_immediately() -> None:
    held = object_state("target-1", (0.5, 0.5, 0.7), held=True)
    moving_in_goal = object_state(
        "target-1", (0.5, 0.5, 0.7), linear=(0.30, 0.0, 0.0)
    )
    config = BenchmarkConfig(
        evaluation=EvaluationConfig(require_settled_placement=False)
    )

    result = evaluate_episode(
        config,
        task(),
        [sample(0, 0.0, (held,)), sample(1, 0.1, (moving_in_goal,))],
    )

    assert result.success
    assert result.metrics["completion_time_s"] == pytest.approx(0.1)
    assert not result.metrics["object_outcomes"]["target-1"]["last_settled"]


def test_grasping_unscored_object_is_wrong_object_failure() -> None:
    target = object_state("target-1", (-0.2, 0.0, 0.7))
    distractor = object_state("distractor", (-0.1, 0.0, 0.7), held=True)

    result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [sample(0, 0.0, (target, distractor))],
    )

    assert result.failure_reason is FailureReason.WRONG_OBJECT
    assert result.metrics["wrong_object_id"] == "distractor"


def test_continuous_sort_tracks_each_registered_object_online() -> None:
    task_manifest = task(TaskType.CONTINUOUS_SORT)
    states = [
        (
            object_state("target-1", (0.5, 0.5, 0.7), held=True),
            object_state("target-2", (1.5, 0.5, 0.7)),
        ),
        (
            object_state("target-1", (0.5, 0.5, 0.7)),
            object_state("target-2", (1.5, 0.5, 0.7), held=True),
        ),
        (
            object_state("target-1", (0.5, 0.5, 0.7)),
            object_state("target-2", (1.5, 0.5, 0.7)),
        ),
        (
            object_state("target-1", (0.5, 0.5, 0.7)),
            object_state("target-2", (1.5, 0.5, 0.7)),
        ),
    ]
    result = evaluate_episode(
        BenchmarkConfig.v1(),
        task_manifest,
        [
            sample(0, 0.0, states[0]),
            sample(1, 0.1, states[1]),
            sample(2, 0.6, states[2]),
            sample(3, 1.1, states[3]),
        ],
    )

    assert result.success
    assert result.metrics["completed_object_count"] == 2
    assert result.metrics["correct_sort_rate"] == pytest.approx(1.0)


def test_empty_episode_and_safety_failure_remain_recordable() -> None:
    assert (
        evaluate_episode(BenchmarkConfig.v1(), task(), []).failure_reason
        is FailureReason.NO_SAMPLES
    )
    target = object_state("target-1", (0.5, 0.5, 0.7), held=True)
    result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [sample(0, 0.0, (target,), robot_fallen=True)],
    )
    assert result.failure_reason is FailureReason.ROBOT_FALLEN


def test_safety_preempts_wrong_object_and_held_target_is_not_missed() -> None:
    target = object_state("target-1", (0.5, 0.5, 0.7))
    distractor = object_state("distractor", (-0.1, 0.0, 0.7), held=True)
    safety_result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [sample(0, 0.0, (target, distractor), robot_fallen=True)],
    )
    assert safety_result.failure_reason is FailureReason.ROBOT_FALLEN

    held_past_exit = object_state(
        "target-1",
        (0.5, 0.5, 0.7),
        held=True,
        crossed_exit=True,
    )
    released = replace(held_past_exit, in_gripper=False, crossed_exit=False)
    placement_result = evaluate_episode(
        BenchmarkConfig.v1(),
        task(),
        [
            sample(0, 0.0, (held_past_exit,)),
            sample(1, 0.1, (released,)),
            sample(2, 0.6, (released,)),
        ],
    )
    assert placement_result.success
