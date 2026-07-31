from dataclasses import replace

import pytest

from conveyor_bench import (
    BenchmarkConfig,
    FailureReason,
    StepSample,
    TaskManifest,
    TaskType,
    evaluate_episode,
)
from conveyor_bench.protocol import make_run_id


def task(
    task_type: TaskType = TaskType.C0_STATIC_PICK,
    *,
    speed: float = 0.0,
    max_duration_s: float = 20.0,
) -> TaskManifest:
    return TaskManifest(
        task_id="task-001",
        task_type=task_type,
        instruction="pick the red cube",
        target_object_id="red_cube",
        object_ids=("red_cube",),
        seed=7,
        belt_speed_mps=speed,
        belt_surface_z_m=0.7,
        exit_x_m=1.0,
        max_duration_s=max_duration_s,
    )


def sample(
    step: int,
    time_s: float,
    *,
    lifted: bool = False,
    secure: bool = False,
    crossed_exit: bool = False,
    robot_fallen: bool = False,
) -> StepSample:
    return StepSample(
        sim_step=step,
        sim_time_s=time_s,
        env_id=0,
        object_xyz=(0.1, 0.0, 0.76 if lifted else 0.71),
        object_linear_velocity=(0.1, 0.0, 0.0),
        tcp_xyz=(0.1 + step * 0.01, 0.0, 0.8),
        belt_command_speed_mps=0.0,
        belt_measured_speed_mps=0.0,
        gripper_closed=secure,
        left_contact=secure,
        right_contact=secure,
        target_in_gripper=secure,
        target_crossed_exit=crossed_exit,
        robot_fallen=robot_fallen,
        forbidden_collision=False,
        phase="verify",
        action={"gripper": 1.0},
    )


def test_v0_configuration_is_self_consistent() -> None:
    config = BenchmarkConfig.v0()
    assert config.protocol_version == "conveyor-bench-v0"
    assert config.physics_hz == 200
    assert config.control_hz == 50
    assert config.camera_hz == 25
    assert config.evaluation.lift_height_m == pytest.approx(0.05)
    assert config.evaluation.hold_time_s == pytest.approx(1.0)


def test_run_ids_do_not_collide_within_one_second() -> None:
    first = make_run_id()
    second = make_run_id()

    assert first != second
    assert first.startswith("run-")
    assert second.startswith("run-")


def test_runtime_error_has_a_distinct_failure_reason() -> None:
    assert FailureReason.RUNTIME_ERROR.value == "runtime_error"
    assert FailureReason.RUNTIME_ERROR is not FailureReason.RECORDER_ERROR


def test_transverse_transport_geometry_projects_world_minus_y() -> None:
    transverse = replace(
        task(TaskType.C1_DYNAMIC_PICK, speed=0.08),
        exit_x_m=None,
        transport_direction_xyz=(0.0, -1.0, 0.0),
        exit_plane_point_xyz=(0.70, -0.57, 0.67),
    )

    assert transverse.transport_progress((0.70, 0.20, 0.71)) == pytest.approx(-0.20)
    assert transverse.forward_speed((0.0, -0.08, 0.0)) == pytest.approx(0.08)
    assert transverse.remaining_distance_to_exit((0.70, 0.20, 0.71)) == pytest.approx(
        0.77
    )
    assert not transverse.has_crossed_exit((0.70, -0.569, 0.71))
    assert transverse.has_crossed_exit((0.70, -0.57, 0.71))


def test_legacy_exit_x_resolves_to_world_plus_x() -> None:
    legacy = task()

    assert legacy.resolved_transport_direction_xyz == (1.0, 0.0, 0.0)
    assert legacy.has_crossed_exit((1.0, 99.0, 0.0))


def test_transport_direction_must_be_a_unit_vector() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        replace(
            task(),
            exit_x_m=None,
            transport_direction_xyz=(0.0, -2.0, 0.0),
            exit_plane_point_xyz=(0.0, -0.57, 0.0),
        )


def test_legacy_and_oriented_exit_geometry_cannot_conflict() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        replace(
            task(),
            transport_direction_xyz=(0.0, -1.0, 0.0),
            exit_plane_point_xyz=(0.0, -0.57, 0.0),
        )


def test_c0_succeeds_after_sustained_verified_grasp() -> None:
    samples = [
        sample(0, 0.0),
        sample(1, 0.5, lifted=True, secure=True),
        sample(2, 1.0, lifted=True, secure=True),
        sample(3, 1.5, lifted=True, secure=True),
    ]
    result = evaluate_episode(BenchmarkConfig.v0(), task(), samples)
    assert result.success
    assert result.failure_reason is FailureReason.NONE
    assert result.metrics["completion_time_s"] == pytest.approx(1.5)
    assert result.metrics["max_lift_m"] == pytest.approx(0.06)


def test_c1_fails_if_target_crosses_exit_before_hold_completes() -> None:
    dynamic_task = task(TaskType.C1_DYNAMIC_PICK, speed=0.1)
    samples = [
        replace(sample(0, 0.0), belt_command_speed_mps=0.1),
        replace(
            sample(1, 0.5, lifted=True, secure=True),
            belt_command_speed_mps=0.1,
        ),
        replace(
            sample(2, 1.0, lifted=True, secure=True, crossed_exit=True),
            belt_command_speed_mps=0.1,
        ),
    ]
    result = evaluate_episode(BenchmarkConfig.v0(), dynamic_task, samples)
    assert not result.success
    assert result.failure_reason is FailureReason.TARGET_MISSED


def test_safety_failure_preempts_grasp_result() -> None:
    samples = [
        sample(0, 0.0, lifted=True, secure=True),
        sample(1, 0.5, lifted=True, secure=True, robot_fallen=True),
        sample(2, 1.0, lifted=True, secure=True),
    ]
    result = evaluate_episode(BenchmarkConfig.v0(), task(), samples)
    assert result.failure_reason is FailureReason.ROBOT_FALLEN


def test_task_mode_rejects_incompatible_nominal_belt_speed() -> None:
    result = evaluate_episode(
        BenchmarkConfig.v0(),
        task(TaskType.C1_DYNAMIC_PICK, speed=0.0),
        [sample(0, 0.0)],
    )
    assert result.failure_reason is FailureReason.INVALID_TASK_CONFIGURATION


def test_full_duration_rollout_is_a_timeout() -> None:
    result = evaluate_episode(
        BenchmarkConfig.v0(),
        task(max_duration_s=1.0),
        [sample(0, 0.02), sample(1, 1.0)],
    )
    assert result.failure_reason is FailureReason.TIMEOUT
    assert result.metrics["duration_s"] == pytest.approx(1.0)


def test_joint_state_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="equal length"):
        replace(
            sample(0, 0.02),
            joint_positions=(0.0, 1.0),
            joint_velocities=(0.0,),
        )
