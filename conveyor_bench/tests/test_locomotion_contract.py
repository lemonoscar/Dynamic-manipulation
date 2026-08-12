import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conveyor_bench.isaac.locomotion import (
    ACTION_JOINT_ORDER,
    DEFAULT_ARM_POSE,
    DEFAULT_LEG_POSE,
    FLAT_HEIGHT_SCAN_VALUE,
    OBSERVATION_SLICES,
    POLICY_SHA256,
    STATE_JOINT_ORDER,
    build_observation,
    leg_target,
    load_contract,
    planar_standoff_goal,
    verify_policy_hash,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = (
    PROJECT_ROOT / "assets" / "policies" / "go2_x5_pct_dog_only"
)
POLICY_PATH = POLICY_DIR / "policy.pt"
CONTRACT_PATH = POLICY_DIR / "contract.json"


def _fake_robot(batch_shape=()):
    prefix = batch_shape + (18,)
    default = np.broadcast_to(
        np.linspace(-0.2, 0.2, 18, dtype=np.float32),
        prefix,
    ).copy()
    offset = np.broadcast_to(
        np.linspace(0.01, 0.18, 18, dtype=np.float32),
        prefix,
    )
    return SimpleNamespace(
        root_lin_vel_b=np.broadcast_to(
            np.asarray((1.0, -2.0, 3.0), dtype=np.float32),
            batch_shape + (3,),
        ).copy(),
        root_ang_vel_b=np.broadcast_to(
            np.asarray((0.4, -0.8, 1.2), dtype=np.float32),
            batch_shape + (3,),
        ).copy(),
        projected_gravity_b=np.broadcast_to(
            np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
            batch_shape + (3,),
        ).copy(),
        joint_pos=default + offset,
        default_joint_pos=default,
        joint_vel=np.broadcast_to(
            np.arange(18, dtype=np.float32),
            prefix,
        ).copy(),
    )


def test_contract_and_vendored_policy_are_frozen():
    contract = load_contract(CONTRACT_PATH)

    assert contract["policy"]["sha256"] == POLICY_SHA256
    assert contract["policy"]["size_bytes"] == POLICY_PATH.stat().st_size
    assert verify_policy_hash(POLICY_PATH) == POLICY_SHA256
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256
    assert tuple(contract["joints"]["state_order"]) == STATE_JOINT_ORDER
    assert tuple(contract["joints"]["action_order"]) == ACTION_JOINT_ORDER
    assert tuple(contract["default_pose"]["leg"]) == DEFAULT_LEG_POSE
    assert tuple(contract["default_pose"]["arm"]) == DEFAULT_ARM_POSE
    assert contract["timing"] == {
        "physics_hz": 400,
        "policy_hz": 50,
        "physics_dt_seconds": 0.0025,
        "policy_dt_seconds": 0.02,
        "decimation": 8,
    }

    raw_contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    ranges = raw_contract["command_limits"]
    assert ranges["training_envelope"] == {
        "linear_x_mps": [-0.55, 0.55],
        "linear_y_mps": [-0.25, 0.25],
        "angular_z_radps": [-0.6, 0.6],
    }
    assert ranges["benchmark_v1_guardrail"]["linear_y_mps"] == [0.0, 0.0]
    height_scan = contract["observation"]["slices"]["height_scan"]
    assert height_scan["flat_benchmark_value"] == FLAT_HEIGHT_SCAN_VALUE == -0.2
    assert height_scan["flat_benchmark_mode"] == (
        "constant_direct_flat_approximation"
    )


def test_observation_slices_are_contiguous_and_cover_260_values():
    contract = load_contract(CONTRACT_PATH)
    cursor = 0

    for name, expected_slice in OBSERVATION_SLICES.items():
        frozen = contract["observation"]["slices"][name]
        assert frozen["start"] == cursor
        assert (frozen["start"], frozen["stop"]) == (
            expected_slice.start,
            expected_slice.stop,
        )
        cursor = frozen["stop"]

    assert cursor == contract["observation"]["dimension"] == 260


def test_build_observation_populates_every_slot():
    robot = _fake_robot()
    command = np.asarray((0.2, 0.0, 0.3), dtype=np.float32)
    last_action = np.linspace(-0.6, 0.5, 12, dtype=np.float32)
    arm_target = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    gripper = np.asarray((1.5,), dtype=np.float32)
    height_scan = np.linspace(-1.5, 1.5, 187, dtype=np.float32)

    observation = build_observation(
        robot,
        command,
        last_action,
        arm_target,
        gripper,
        height_scan=height_scan,
    )

    assert observation.shape == (260,)
    assert observation.dtype == np.float32
    assert np.allclose(observation[0:3], robot.root_lin_vel_b * 2.0)
    assert np.allclose(observation[3:6], robot.root_ang_vel_b * 0.25)
    assert np.allclose(observation[6:9], robot.projected_gravity_b)
    assert np.allclose(observation[9:12], command)
    assert np.allclose(
        observation[12:30],
        robot.joint_pos - robot.default_joint_pos,
    )
    assert np.allclose(observation[30:48], robot.joint_vel * 0.05)
    assert np.allclose(observation[48:60], last_action)
    assert np.array_equal(observation[60:66], np.zeros(6, dtype=np.float32))
    assert np.allclose(observation[66:253], np.clip(height_scan, -1.0, 1.0))
    assert np.allclose(observation[253:259], arm_target)
    assert observation[259] == 1.0


def test_build_observation_uses_batched_direct_flat_height_scan():
    robot = _fake_robot((2,))
    command = np.zeros((2, 3), dtype=np.float32)
    last_action = np.zeros((2, 12), dtype=np.float32)
    arm_target = np.broadcast_to(
        np.asarray(DEFAULT_ARM_POSE, dtype=np.float32),
        (2, 6),
    ).copy()
    gripper = np.zeros((2, 1), dtype=np.float32)

    observation = build_observation(
        robot,
        command,
        last_action,
        arm_target,
        gripper,
    )

    assert observation.shape == (2, 260)
    assert np.array_equal(
        observation[:, OBSERVATION_SLICES["height_scan"]],
        np.full((2, 187), FLAT_HEIGHT_SCAN_VALUE, dtype=np.float32),
    )


def test_observation_rejects_wrong_shape_and_nonfinite_values():
    robot = _fake_robot()
    valid = {
        "command": np.zeros(3, dtype=np.float32),
        "last_action": np.zeros(12, dtype=np.float32),
        "arm_target": np.zeros(6, dtype=np.float32),
        "gripper": np.zeros(1, dtype=np.float32),
    }

    with pytest.raises(ValueError, match=r"last_action must have shape"):
        build_observation(
            robot,
            valid["command"],
            np.zeros(11, dtype=np.float32),
            valid["arm_target"],
            valid["gripper"],
        )

    robot.root_lin_vel_b[0] = np.nan
    with pytest.raises(ValueError, match=r"root_lin_vel_b.*finite"):
        build_observation(robot, **valid)


def test_leg_target_uses_frozen_default_pose_and_action_scale():
    action = np.linspace(-1.0, 1.0, 12, dtype=np.float32)

    target = leg_target(action)

    assert target.dtype == np.float32
    assert np.allclose(
        target,
        np.asarray(DEFAULT_LEG_POSE, dtype=np.float32) + 0.25 * action,
    )


def test_policy_hash_rejects_tampered_artifact(tmp_path):
    tampered = tmp_path / "policy.pt"
    tampered.write_bytes(b"not the vendored policy")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_policy_hash(tampered)


def test_planar_standoff_goal_preserves_travel_and_faces_target():
    yaw, goal = planar_standoff_goal(
        (0.08, 0.0),
        (0.34, 0.40),
        standoff_m=0.26,
        minimum_travel_m=0.12,
    )

    assert yaw == pytest.approx(np.arctan2(0.40, 0.26))
    assert np.linalg.norm(np.asarray(goal) - (0.34, 0.40)) == pytest.approx(
        0.26
    )
    assert np.linalg.norm(np.asarray(goal) - (0.08, 0.0)) > 0.12


def test_planar_standoff_goal_rejects_a_fake_navigation_segment():
    with pytest.raises(ValueError, match="too close"):
        planar_standoff_goal(
            (0.0, 0.0),
            (0.30, 0.0),
            standoff_m=0.26,
            minimum_travel_m=0.10,
        )
