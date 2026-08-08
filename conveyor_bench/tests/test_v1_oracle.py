from dataclasses import replace
import math

import pytest

from conveyor_bench.v1.oracle import (
    DynamicSortOracle,
    OracleConfig,
    OracleObservation,
    OraclePhase,
    TOP_DOWN_APPROACH_PITCH_DEG,
    TOP_DOWN_X_CLOSING_YAW_DEG,
    top_down_tcp_orientation_wxyz,
)
from conveyor_bench.v1.protocol import RobotMode


TARGET_ID = "target-red-can"
WRONG_ID = "distractor-blue-box"
TARGET_POSITION = (0.50, 0.00, 0.20)
LIFTED_TARGET_POSITION = (0.50, 0.00, 0.55)
TARGET_VELOCITY = (0.10, 0.00, 0.00)


def config(
    robot_mode: RobotMode = RobotMode.MOBILE_KINEMATIC,
) -> OracleConfig:
    return OracleConfig(
        target_object_id=TARGET_ID,
        goal_center_world=(1.20, 0.40, 0.30),
        robot_mode=robot_mode,
        object_height_m=0.10,
        grasp_offset_world=(0.01, -0.02, 0.03),
        intercept_horizon_s=0.20,
        pregrasp_clearance_m=0.10,
        safe_carry_clearance_m=0.20,
        position_tolerance_m=1.0e-6,
        grasp_tolerance_m=1.0e-6,
        settle_duration_s=0.05,
        select_duration_s=0.05,
        grasp_contact_dwell_s=0.05,
        preplace_dwell_s=0.05,
        placement_dwell_s=0.10,
        episode_timeout_s=10.0,
        phase_timeout_s=2.0,
        close_timeout_s=0.50,
        release_timeout_s=0.50,
        verify_timeout_s=0.80,
    )


def observation(
    sim_time_s: float,
    *,
    target_position_world=TARGET_POSITION,
    target_velocity_world=TARGET_VELOCITY,
    tcp_position_world=(0.0, 0.0, 0.0),
    target_held: bool = False,
    target_lifted: bool = False,
    target_in_goal: bool = False,
    target_released: bool = False,
    target_settled: bool = False,
    left_contact_object_ids=(),
    right_contact_object_ids=(),
    wrong_object_grasped: bool = False,
    robot_fallen: bool = False,
    forbidden_collision: bool = False,
    target_crossed_exit: bool = False,
) -> OracleObservation:
    return OracleObservation(
        sim_time_s=sim_time_s,
        target_position_world=target_position_world,
        target_velocity_world=target_velocity_world,
        tcp_position_world=tcp_position_world,
        left_contact_object_ids=left_contact_object_ids,
        right_contact_object_ids=right_contact_object_ids,
        target_held=target_held,
        target_lifted=target_lifted,
        target_in_goal=target_in_goal,
        target_released=target_released,
        target_settled=target_settled,
        wrong_object_grasped=wrong_object_grasped,
        robot_fallen=robot_fallen,
        forbidden_collision=forbidden_collision,
        target_crossed_exit=target_crossed_exit,
    )


def _position(command) -> tuple[float, float, float]:
    return command.target_tcp_pose_world.xyz


def _rotate_vector(wxyz, xyz):
    w, x, y, z = wxyz
    vx, vy, vz = xyz
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


@pytest.mark.parametrize(
    ("closing_axis", "world_axis_index"), (("y", 1), ("x", 0))
)
def test_registered_teacher_pose_is_overhead_and_closing_axis_aligned(
    closing_axis, world_axis_index
) -> None:
    orientation = top_down_tcp_orientation_wxyz(closing_axis)
    approach = _rotate_vector(orientation, (1.0, 0.0, 0.0))
    closing = _rotate_vector(orientation, (0.0, 1.0, 0.0))

    assert TOP_DOWN_APPROACH_PITCH_DEG == pytest.approx(75.0)
    assert approach[2] == pytest.approx(-math.sin(math.radians(75.0)))
    assert approach[2] < -0.96
    expected_alignment = (
        1.0
        if closing_axis == "y"
        else math.sin(math.radians(TOP_DOWN_X_CLOSING_YAW_DEG))
    )
    assert abs(closing[world_axis_index]) == pytest.approx(
        expected_alignment
    )
    assert abs(closing[world_axis_index]) > 0.95


def test_pregrasp_requires_continuous_overhead_observation_dwell() -> None:
    oracle = DynamicSortOracle(
        replace(
            config(),
            intercept_staging_y_world=0.08,
            intercept_entry_tolerance_m=0.04,
            pregrasp_observation_dwell_s=0.50,
        )
    )
    oracle.step(observation(0.00))
    oracle.step(observation(0.06))
    command = oracle.step(observation(0.12))
    staging = _position(command)

    command = oracle.step(
        observation(
            0.14,
            target_position_world=(0.50, 0.08, 0.20),
            target_velocity_world=TARGET_VELOCITY,
            tcp_position_world=staging,
        )
    )
    assert command.phase is OraclePhase.PREGRASP
    command = oracle.step(
        observation(
            0.63,
            target_position_world=(0.50, 0.08, 0.20),
            target_velocity_world=TARGET_VELOCITY,
            tcp_position_world=staging,
        )
    )
    assert command.phase is OraclePhase.PREGRASP
    command = oracle.step(
        observation(
            0.64,
            target_position_world=(0.50, 0.08, 0.20),
            target_velocity_world=TARGET_VELOCITY,
            tcp_position_world=staging,
        )
    )
    assert command.phase is OraclePhase.TRACK


def test_intercept_staging_waits_for_target_entry():
    oracle = DynamicSortOracle(
        replace(
            config(),
            intercept_staging_y_world=0.08,
            intercept_entry_tolerance_m=0.04,
            close_on_target_contact=True,
        )
    )

    command = oracle.step(
        observation(
            0.00,
            target_position_world=(0.50, 0.30, 0.20),
            target_velocity_world=(0.0, 0.0, 0.0),
        )
    )
    assert _position(command)[1] == pytest.approx(0.08)
    command = oracle.step(observation(0.06))
    command = oracle.step(observation(0.12))
    staging_position = _position(command)

    command = oracle.step(
        observation(
            0.13,
            target_position_world=(0.50, 0.30, 0.20),
            target_velocity_world=(0.0, 0.0, 0.0),
            tcp_position_world=staging_position,
        )
    )
    assert command.phase is OraclePhase.PREGRASP
    assert _position(command)[1] == pytest.approx(0.08)
    staging_position = _position(command)

    command = oracle.step(
        observation(
            0.14,
            target_position_world=(0.50, 0.05, 0.20),
            target_velocity_world=(0.0, 0.0, 0.0),
            tcp_position_world=staging_position,
        )
    )
    assert command.phase is OraclePhase.TRACK
    assert _position(command)[1] == pytest.approx(0.08)

    command = oracle.step(
        observation(
            0.15,
            target_position_world=(0.50, 0.04, 0.20),
            target_velocity_world=(0.0, 0.0, 0.0),
            tcp_position_world=staging_position,
        )
    )
    assert command.phase is OraclePhase.DESCEND
    assert _position(command)[1] == pytest.approx(0.08)

    command = oracle.step(
        observation(
            0.16,
            target_position_world=(0.50, 0.03, 0.20),
            target_velocity_world=(0.0, 0.0, 0.0),
            tcp_position_world=staging_position,
            left_contact_object_ids=(TARGET_ID,),
        )
    )
    assert command.phase is OraclePhase.CLOSE
    assert command.gripper_command == 0.0


def run_success_trace(
    robot_mode: RobotMode = RobotMode.MOBILE_KINEMATIC,
):
    oracle = DynamicSortOracle(config(robot_mode))
    commands = []

    command = oracle.step(observation(0.00))
    commands.append(command)
    command = oracle.step(observation(0.06))
    commands.append(command)
    command = oracle.step(observation(0.12))
    commands.append(command)

    command = oracle.step(
        observation(0.13, tcp_position_world=_position(command))
    )
    commands.append(command)
    command = oracle.step(
        observation(0.14, tcp_position_world=_position(command))
    )
    commands.append(command)
    command = oracle.step(
        observation(0.15, tcp_position_world=_position(command))
    )
    commands.append(command)

    bilateral_target_contact = {
        "left_contact_object_ids": (TARGET_ID,),
        "right_contact_object_ids": (TARGET_ID,),
        "target_held": True,
    }
    command = oracle.step(
        observation(
            0.16,
            tcp_position_world=_position(command),
            **bilateral_target_contact,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.22,
            tcp_position_world=_position(command),
            **bilateral_target_contact,
        )
    )
    commands.append(command)

    command = oracle.step(
        observation(
            0.23,
            target_position_world=LIFTED_TARGET_POSITION,
            tcp_position_world=_position(command),
            target_held=True,
            target_lifted=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.24,
            target_position_world=LIFTED_TARGET_POSITION,
            tcp_position_world=_position(command),
            target_held=True,
            target_lifted=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.25,
            target_position_world=LIFTED_TARGET_POSITION,
            tcp_position_world=_position(command),
            target_held=True,
            target_lifted=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.31,
            target_position_world=LIFTED_TARGET_POSITION,
            tcp_position_world=_position(command),
            target_held=True,
            target_lifted=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.32,
            target_position_world=LIFTED_TARGET_POSITION,
            tcp_position_world=_position(command),
            target_held=True,
            target_lifted=True,
        )
    )
    commands.append(command)

    command = oracle.step(
        observation(
            0.33,
            tcp_position_world=_position(command),
            target_released=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.34,
            tcp_position_world=_position(command),
            target_released=True,
            target_in_goal=True,
            target_settled=True,
        )
    )
    commands.append(command)

    command = oracle.step(
        observation(
            0.35,
            tcp_position_world=_position(command),
            target_released=True,
            target_in_goal=True,
            target_settled=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.41,
            tcp_position_world=_position(command),
            target_released=True,
            target_in_goal=False,
            target_settled=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.42,
            tcp_position_world=_position(command),
            target_released=True,
            target_in_goal=True,
            target_settled=True,
        )
    )
    commands.append(command)
    command = oracle.step(
        observation(
            0.53,
            tcp_position_world=_position(command),
            target_released=True,
            target_in_goal=True,
            target_settled=True,
        )
    )
    commands.append(command)
    return commands


def test_constant_velocity_intercept_uses_object_geometry_and_grasp_offset() -> None:
    oracle = DynamicSortOracle(config())

    command = oracle.step(observation(0.0))

    assert command.phase is OraclePhase.SETTLE
    assert command.target_tcp_pose_world.xyz == pytest.approx(
        (0.53, -0.02, 0.33)
    )
    assert command.gripper_command == 1.0


def test_complete_sort_visits_every_phase_and_requires_fresh_goal_dwell() -> None:
    commands = run_success_trace()
    phases = [command.phase for command in commands]

    assert phases == [
        OraclePhase.SETTLE,
        OraclePhase.SELECT,
        OraclePhase.PREGRASP,
        OraclePhase.TRACK,
        OraclePhase.DESCEND,
        OraclePhase.CLOSE,
        OraclePhase.CLOSE,
        OraclePhase.LIFT,
        OraclePhase.CARRY,
        OraclePhase.CARRY,
        OraclePhase.PREPLACE,
        OraclePhase.PLACE_DESCEND,
        OraclePhase.OPEN,
        OraclePhase.RETREAT,
        OraclePhase.VERIFY_PLACE,
        OraclePhase.VERIFY_PLACE,
        OraclePhase.VERIFY_PLACE,
        OraclePhase.VERIFY_PLACE,
        OraclePhase.COMPLETE,
    ]
    assert not commands[-2].terminal
    assert commands[-1].terminal
    assert commands[-1].success
    assert commands[-1].failure_reason is None
    assert commands[-1].gripper_command == 1.0
    assert commands[-1].target_tcp_pose_world.xyz == pytest.approx(
        (1.21, 0.38, 0.58)
    )


def test_base_moves_only_in_select_and_carry_and_fixed_mode_never_moves() -> None:
    mobile_commands = run_success_trace()
    fixed_commands = run_success_trace(RobotMode.FIXED_BASE)

    for command in mobile_commands:
        if command.phase in {OraclePhase.SELECT, OraclePhase.CARRY}:
            assert command.base_command_body != (0.0, 0.0, 0.0)
        else:
            assert command.base_command_body == (0.0, 0.0, 0.0)
    assert all(
        command.base_command_body == (0.0, 0.0, 0.0)
        for command in fixed_commands
    )


def test_wrong_bilateral_object_contact_fails_without_claiming_success() -> None:
    oracle = DynamicSortOracle(config())

    command = oracle.step(
        observation(
            0.01,
            left_contact_object_ids=(WRONG_ID,),
            right_contact_object_ids=(WRONG_ID,),
        )
    )

    assert command.phase is OraclePhase.FAILED
    assert command.terminal
    assert not command.success
    assert command.failure_reason == "wrong_object"
    assert command.base_command_body == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("observation_override", "expected_reason"),
    [
        ({"robot_fallen": True}, "robot_fallen"),
        ({"forbidden_collision": True}, "forbidden_collision"),
        ({"target_crossed_exit": True}, "target_missed"),
    ],
)
def test_safety_and_missed_target_are_terminal_failures(
    observation_override, expected_reason
) -> None:
    oracle = DynamicSortOracle(config())

    command = oracle.step(observation(0.01, **observation_override))

    assert command.phase is OraclePhase.FAILED
    assert command.terminal
    assert command.failure_reason == expected_reason


def test_episode_and_phase_timeouts_are_explicit_failures() -> None:
    phase_timeout_config = replace(
        config(),
        settle_duration_s=1.0,
        phase_timeout_s=0.20,
    )
    phase_timeout_oracle = DynamicSortOracle(phase_timeout_config)
    phase_timeout_oracle.step(observation(0.0))

    phase_failure = phase_timeout_oracle.step(observation(0.20))

    assert phase_failure.failure_reason == "phase_timeout:settle"

    episode_timeout_config = replace(
        config(),
        settle_duration_s=1.0,
        phase_timeout_s=2.0,
        episode_timeout_s=0.20,
    )
    episode_timeout_oracle = DynamicSortOracle(episode_timeout_config)
    episode_timeout_oracle.step(observation(0.0))

    episode_failure = episode_timeout_oracle.step(observation(0.20))

    assert episode_failure.failure_reason == "episode_timeout"


def test_trace_is_deterministic_for_identical_observations() -> None:
    assert run_success_trace() == run_success_trace()


def test_high_goal_drop_opens_without_entering_the_tray() -> None:
    oracle = DynamicSortOracle(
        replace(config(), release_from_high_goal=True)
    )
    oracle._transition(OraclePhase.PREPLACE, 0.0)
    high_goal = oracle._goal_high_position()
    assert high_goal == pytest.approx(oracle._place_position())

    command = oracle.step(
        observation(
            0.06,
            tcp_position_world=high_goal,
            target_held=True,
        )
    )

    assert command.phase is OraclePhase.OPEN
    assert command.gripper_command == 1.0
    assert _position(command) == pytest.approx(high_goal)

    command = oracle.step(
        observation(
            0.07,
            tcp_position_world=high_goal,
            target_held=True,
        )
    )
    assert command.phase is OraclePhase.OPEN
    assert _position(command) == pytest.approx(high_goal)

    command = oracle.step(
        observation(
            0.08,
            tcp_position_world=high_goal,
            target_released=True,
        )
    )
    assert command.phase is OraclePhase.VERIFY_PLACE
    assert command.gripper_command == 1.0
    assert _position(command) == pytest.approx(high_goal)


def test_high_goal_drop_uses_the_registered_carry_tolerance() -> None:
    oracle = DynamicSortOracle(
        replace(
            config(),
            release_from_high_goal=True,
            position_tolerance_m=0.01,
            carry_position_tolerance_m=0.045,
        )
    )
    oracle._transition(OraclePhase.CARRY, 0.0)
    high_goal = oracle._goal_high_position()
    nearby = (high_goal[0] - 0.03, high_goal[1], high_goal[2])

    command = oracle.step(
        observation(0.01, tcp_position_world=nearby, target_held=True)
    )
    assert command.phase is OraclePhase.PREPLACE

    command = oracle.step(
        observation(0.07, tcp_position_world=nearby, target_held=True)
    )
    assert command.phase is OraclePhase.OPEN
    assert command.gripper_command == 1.0


def test_high_goal_drop_can_release_from_a_physically_entered_goal() -> None:
    oracle = DynamicSortOracle(
        replace(
            config(),
            release_from_high_goal=True,
            position_tolerance_m=0.01,
            carry_position_tolerance_m=0.02,
        )
    )
    oracle._transition(OraclePhase.CARRY, 0.0)

    command = oracle.step(
        observation(
            0.01,
            tcp_position_world=(0.0, 0.0, 0.0),
            target_held=True,
            target_in_goal=True,
        )
    )
    assert command.phase is OraclePhase.PREPLACE

    command = oracle.step(
        observation(
            0.07,
            tcp_position_world=(0.0, 0.0, 0.0),
            target_held=True,
            target_in_goal=True,
        )
    )
    assert command.phase is OraclePhase.OPEN
    assert command.gripper_command == 1.0


def test_high_goal_lift_clears_the_object_without_double_counting_bin_height() -> None:
    ordinary = DynamicSortOracle(config())
    high_drop = DynamicSortOracle(
        replace(config(), release_from_high_goal=True)
    )

    assert ordinary._safe_carry_z(0.20) == pytest.approx(0.58)
    assert high_drop._safe_carry_z(0.20) == pytest.approx(0.48)


def test_high_goal_drop_flag_requires_a_bool() -> None:
    with pytest.raises(ValueError, match="release_from_high_goal"):
        replace(config(), release_from_high_goal=1)


def test_carry_position_tolerance_must_be_positive_when_provided() -> None:
    with pytest.raises(ValueError, match="carry_position_tolerance_m"):
        replace(config(), carry_position_tolerance_m=0.0)
