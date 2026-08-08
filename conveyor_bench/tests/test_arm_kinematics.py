import numpy as np
import pytest

from conveyor_bench.isaac.arm_kinematics import CalibratedArmKinematics
from conveyor_bench.v1.oracle import top_down_tcp_orientation_wxyz


def test_forward_matches_simulator_probe():
    kinematics = CalibratedArmKinematics()

    position, rotation = kinematics.forward((0.0, 1.2, 1.2, 0.0, 0.0, 0.0))

    # With the 0.125 m TCP, the live fixed-URDF probe measured
    # [0.47356, 0.00100, 0.49643] in the root frame, or z=0.87643
    # at the nominal 0.38 m fixed-root height. FK evaluated at the measured
    # joints agreed within 3 micrometres.
    assert np.allclose(position, (0.47354, 0.0010, 0.87656), atol=1.0e-4)
    assert np.allclose(rotation, np.eye(3), atol=1.0e-6)


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
        assert solution.joint_positions[3] >= -1.26
        seed = solution.joint_positions


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


def test_policy_usd_root_frame_matches_live_mobile_probe():
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

    # The live policy-USD probe measured this TCP in the articulation-root
    # frame; calibrated FK differed by 0.429 mm.
    assert np.allclose(
        position,
        (0.34349683, -0.00054965, 0.33769384),
        atol=5.0e-4,
    )


def test_constructor_rejects_nonpositive_solution_tolerances():
    with pytest.raises(ValueError):
        CalibratedArmKinematics(position_tolerance_m=0.0)
    with pytest.raises(ValueError):
        CalibratedArmKinematics(orientation_tolerance=0.0)
