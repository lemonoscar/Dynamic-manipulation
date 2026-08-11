import numpy as np
import pytest

from conveyor_bench.isaac.arm_kinematics import CalibratedArmKinematics
from conveyor_bench.v1.oracle import (
    TOP_DOWN_X_CLOSING_YAW_DEG,
    top_down_tcp_orientation_wxyz,
)


def test_forward_matches_canonical_pct_urdf():
    kinematics = CalibratedArmKinematics()

    position, rotation = kinematics.forward((0.0, 1.2, 1.2, 0.0, 0.0, 0.0))

    # Independent forward composition of the PCT mount, six joint origins and
    # 0.15757 m FinRay tip frame at this pose.
    assert np.allclose(
        position,
        (0.543607549, -0.000503568, 0.832558317),
        atol=1.0e-7,
    )
    # PCT authors the two half-turns as +/-3.1416, hence the few-microradian
    # residual instead of an analytically exact identity.
    assert np.allclose(rotation, np.eye(3), atol=7.0e-6)


def test_solver_reaches_pregrasp_grasp_and_lift_targets():
    kinematics = CalibratedArmKinematics()
    seed = (0.0, 1.0, 1.5, 0.0, 0.0, 0.0)

    for target in (
        (0.70, 0.02, 0.692),
        (0.70, 0.02, 0.547),
        (0.70, 0.02, 0.640),
    ):
        solution = kinematics.solve(
            target,
            (-1.0, 0.0, 0.0, 0.0),
            seed=seed,
        )
        assert solution.position_error_m < 0.001
        assert solution.orientation_error < 0.005
        seed = solution.joint_positions


def test_policy_arm_reaches_low_belt_from_overhead() -> None:
    kinematics = CalibratedArmKinematics.in_policy_usd_root_frame()
    orientation = top_down_tcp_orientation_wxyz("y")
    seed = (0.002, 1.431, 0.746, 0.686, 0.002, 0.0)

    for target in ((0.57, 0.10, 0.179), (0.57, 0.10, 0.079)):
        solution = kinematics.solve(target, orientation, seed=seed)
        assert solution.position_error_m < 0.001
        assert solution.orientation_error < 0.005
        assert solution.joint_positions[3] >= -1.30
        seed = solution.joint_positions


def test_x_closing_overhead_pose_retains_wrist_limit_margin() -> None:
    kinematics = CalibratedArmKinematics.in_policy_usd_root_frame(
        max_iterations=300,
        position_tolerance_m=0.001,
        orientation_tolerance=0.01,
    )
    y_closing_pregrasp = (
        -0.044556522,
        2.098253408,
        1.709825231,
        -0.942128716,
        -0.052071541,
        0.092826496,
    )
    target_position, _ = kinematics.forward(y_closing_pregrasp)

    solution = kinematics.solve(
        target_position,
        top_down_tcp_orientation_wxyz("x"),
        seed=y_closing_pregrasp,
    )

    assert TOP_DOWN_X_CLOSING_YAW_DEG == 73.0
    assert solution.position_error_m < 0.001
    assert solution.orientation_error < 0.01
    assert solution.joint_positions[-1] > -1.40


def test_quaternion_sign_does_not_change_solution():
    kinematics = CalibratedArmKinematics()
    seed = (0.0, 1.0, 1.5, 0.0, 0.0, 0.0)

    positive = kinematics.solve((0.67, 0.0, 0.84), (1.0, 0.0, 0.0, 0.0), seed=seed)
    negative = kinematics.solve((0.67, 0.0, 0.84), (-1.0, 0.0, 0.0, 0.0), seed=seed)

    assert np.allclose(positive.joint_positions, negative.joint_positions)


def test_root_frame_solver_removes_fixed_world_root_height():
    world_solver = CalibratedArmKinematics()
    root_solver = CalibratedArmKinematics.in_robot_root_frame()
    joints = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)

    world_position, world_rotation = world_solver.forward(joints)
    root_position, root_rotation = root_solver.forward(joints)

    assert np.allclose(
        world_position,
        root_position + np.asarray((0.0, 0.0, 0.38)),
        atol=1.0e-9,
    )
    assert np.allclose(world_rotation, root_rotation, atol=1.0e-9)


def test_historical_policy_frame_alias_uses_the_canonical_urdf_mount():
    solver = CalibratedArmKinematics.in_policy_usd_root_frame()
    joints = (
        -0.0002719297,
        0.29707828,
        0.43148327,
        -0.03687394,
        -0.00008331,
        0.000047826,
    )

    position, _ = solver.forward(joints)

    canonical, _ = CalibratedArmKinematics.in_robot_root_frame().forward(
        joints
    )
    assert np.allclose(
        position,
        (0.37591213, -0.00055590, 0.34086516),
        atol=1.0e-7,
    )
    assert np.allclose(position, canonical, atol=1.0e-12)


def test_constructor_rejects_nonpositive_solution_tolerances():
    with pytest.raises(ValueError):
        CalibratedArmKinematics(position_tolerance_m=0.0)
    with pytest.raises(ValueError):
        CalibratedArmKinematics(orientation_tolerance=0.0)
