"""Local Go2-X5 locomotion policy contract and inference adapter.

The module has no RobotLab, RSL-RL, or Isaac Lab dependency. NumPy inputs make
the observation and target builders testable without Isaac Sim; Torch is
imported only when a tensor or TorchScript policy is used.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np


OBSERVATION_DIM = 260
ACTION_DIM = 12
STATE_JOINT_DIM = 18
HEIGHT_SCAN_DIM = 187
FLAT_HEIGHT_SCAN_VALUE = -0.2
ACTION_SCALE = 0.25
POLICY_SHA256 = (
    "f02e6467472e90671a28d97cd6dc02ed7fdeb59d2ece18e082f254314558d383"
)

STATE_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
)
ACTION_JOINT_ORDER = STATE_JOINT_ORDER[:ACTION_DIM]
DEFAULT_LEG_POSE = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)
DEFAULT_ARM_POSE = (0.0, 0.3, 0.5, 0.0, 0.0, 0.0)
OBSERVATION_SLICES = {
    "base_linear_velocity": slice(0, 3),
    "base_angular_velocity": slice(3, 6),
    "projected_gravity": slice(6, 9),
    "base_velocity_command": slice(9, 12),
    "relative_joint_position": slice(12, 30),
    "joint_velocity": slice(30, 48),
    "last_action_padded": slice(48, 66),
    "height_scan": slice(66, 253),
    "arm_joint_command": slice(253, 259),
    "gripper_command": slice(259, 260),
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_POLICY_DIRECTORY = (
    _PROJECT_ROOT / "assets" / "policies" / "go2_x5_pct_dog_only"
)
DEFAULT_POLICY_PATH = _POLICY_DIRECTORY / "policy.pt"
DEFAULT_CONTRACT_PATH = _POLICY_DIRECTORY / "contract.json"


def planar_standoff_goal(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    standoff_m: float,
    minimum_travel_m: float = 0.0,
) -> tuple[float, tuple[float, float]]:
    """Return a target-facing planar goal before a workcell endpoint."""

    values = (*start_xy, *target_xy, standoff_m, minimum_travel_m)
    if not all(
        isinstance(value, Real) and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("standoff planning values must be finite numbers")
    if standoff_m <= 0.0 or minimum_travel_m < 0.0:
        raise ValueError(
            "standoff must be positive and minimum travel non-negative"
        )
    delta_x = float(target_xy[0] - start_xy[0])
    delta_y = float(target_xy[1] - start_xy[1])
    distance = math.hypot(delta_x, delta_y)
    if distance <= standoff_m + minimum_travel_m:
        raise ValueError("target is too close for the required navigation segment")
    yaw = math.atan2(delta_y, delta_x)
    scale = standoff_m / distance
    return yaw, (
        float(target_xy[0] - delta_x * scale),
        float(target_xy[1] - delta_y * scale),
    )


def heading_hysteresis_active(
    yaw_error_rad: float,
    *,
    was_active: bool,
    enter_tolerance_rad: float,
    exit_tolerance_rad: float,
) -> bool:
    """Latch straight driving until heading error crosses a wider limit."""

    values = (yaw_error_rad, enter_tolerance_rad, exit_tolerance_rad)
    if not all(
        isinstance(value, Real) and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("heading hysteresis values must be finite numbers")
    if not 0.0 < enter_tolerance_rad < exit_tolerance_rad <= math.pi:
        raise ValueError(
            "heading tolerances must satisfy 0 < enter < exit <= pi"
        )
    tolerance = exit_tolerance_rad if was_active else enter_tolerance_rad
    return abs(float(yaw_error_rad)) <= tolerance


def overhead_place_waypoint(
    current_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
    *,
    max_step_m: float,
    planar_tolerance_m: float = 0.005,
) -> tuple[float, float, float]:
    """Stage a place path: lift if needed, move above, then descend."""

    values = (*current_xyz, *target_xyz, max_step_m, planar_tolerance_m)
    if not all(
        isinstance(value, Real) and math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("overhead waypoint values must be finite numbers")
    if max_step_m <= 0.0 or planar_tolerance_m <= 0.0:
        raise ValueError("overhead waypoint limits must be positive")

    current = np.asarray(current_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    delta = target - current
    waypoint = current.copy()
    if delta[2] > planar_tolerance_m:
        waypoint[2] += min(float(delta[2]), max_step_m)
    else:
        planar_distance = float(np.linalg.norm(delta[:2]))
        if planar_distance > planar_tolerance_m:
            step = min(planar_distance, max_step_m)
            waypoint[:2] += delta[:2] * step / planar_distance
        else:
            waypoint[:2] = target[:2]
            waypoint[2] += max(
                -max_step_m, min(max_step_m, float(delta[2]))
            )
    return tuple(float(value) for value in waypoint)


def load_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load and validate the frozen local policy contract."""

    contract_path = Path(path)
    with contract_path.open(encoding="utf-8") as stream:
        contract = json.load(stream)

    if contract.get("schema_version") != 1:
        raise ValueError("locomotion contract schema_version must be 1")
    policy = contract.get("policy", {})
    if policy.get("sha256") != POLICY_SHA256:
        raise ValueError("locomotion contract contains an unexpected policy hash")
    if policy.get("input_dimension") != OBSERVATION_DIM:
        raise ValueError("locomotion contract input dimension must be 260")
    if policy.get("output_dimension") != ACTION_DIM:
        raise ValueError("locomotion contract output dimension must be 12")

    observation = contract.get("observation", {})
    if observation.get("dimension") != OBSERVATION_DIM:
        raise ValueError("locomotion observation dimension must be 260")
    contract_slices = observation.get("slices", {})
    for name, expected in OBSERVATION_SLICES.items():
        item = contract_slices.get(name, {})
        if (item.get("start"), item.get("stop")) != (
            expected.start,
            expected.stop,
        ):
            raise ValueError(f"unexpected observation slice for {name}")

    joints = contract.get("joints", {})
    if tuple(joints.get("state_order", ())) != STATE_JOINT_ORDER:
        raise ValueError("locomotion state joint order does not match the adapter")
    if tuple(joints.get("action_order", ())) != ACTION_JOINT_ORDER:
        raise ValueError("locomotion action joint order does not match the adapter")

    default_pose = contract.get("default_pose", {})
    if tuple(default_pose.get("leg", ())) != DEFAULT_LEG_POSE:
        raise ValueError("locomotion default leg pose does not match the adapter")
    if tuple(default_pose.get("arm", ())) != DEFAULT_ARM_POSE:
        raise ValueError("locomotion default arm pose does not match the adapter")
    if contract.get("control", {}).get("action_scale") != ACTION_SCALE:
        raise ValueError("locomotion action scale must be 0.25")
    return contract


def verify_policy_hash(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    expected_sha256: str = POLICY_SHA256,
) -> str:
    """Verify a policy artifact before allowing it to be loaded."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA256 digest")
    digest = hashlib.sha256()
    with Path(policy_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"locomotion policy SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual}"
        )
    return actual


def load_policy(
    policy_path: str | Path = DEFAULT_POLICY_PATH,
    *,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    device: str | None = None,
) -> Any:
    """Hash-check and load the local TorchScript actor."""

    contract = load_contract(contract_path)
    verify_policy_hash(policy_path, contract["policy"]["sha256"])
    torch = _import_torch()
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()
    return policy


def build_observation(
    robot_data: Any,
    command: Any,
    last_action: Any,
    arm_target: Any,
    gripper: Any,
    *,
    height_scan: Any | None = None,
) -> Any:
    """Build the checkpoint-matched 260-D deterministic policy observation.

    ``robot_data`` may be an object or mapping with ``root_lin_vel_b``,
    ``root_ang_vel_b``, ``projected_gravity_b``, ``joint_pos``,
    ``default_joint_pos``, and ``joint_vel``. Joint tensors must already use
    :data:`STATE_JOINT_ORDER`; this adapter intentionally does not guess or
    reorder articulation joints.
    """

    root_linear = _robot_field(robot_data, "root_lin_vel_b")
    backend = _backend(root_linear)
    leading_shape = _validate_array(
        "root_lin_vel_b", root_linear, 3, backend
    )
    root_angular = _validated(
        "root_ang_vel_b",
        _robot_field(robot_data, "root_ang_vel_b"),
        3,
        backend,
        leading_shape,
    )
    gravity = _validated(
        "projected_gravity_b",
        _robot_field(robot_data, "projected_gravity_b"),
        3,
        backend,
        leading_shape,
    )
    joint_position = _validated(
        "joint_pos",
        _robot_field(robot_data, "joint_pos"),
        STATE_JOINT_DIM,
        backend,
        leading_shape,
    )
    default_joint_position = _validated(
        "default_joint_pos",
        _robot_field(robot_data, "default_joint_pos"),
        STATE_JOINT_DIM,
        backend,
        leading_shape,
    )
    joint_velocity = _validated(
        "joint_vel",
        _robot_field(robot_data, "joint_vel"),
        STATE_JOINT_DIM,
        backend,
        leading_shape,
    )
    command = _validated(
        "command", command, 3, backend, leading_shape
    )
    last_action = _validated(
        "last_action", last_action, ACTION_DIM, backend, leading_shape
    )
    arm_target = _validated(
        "arm_target", arm_target, 6, backend, leading_shape
    )
    gripper = _validated(
        "gripper", gripper, 1, backend, leading_shape
    )
    if height_scan is None:
        height_scan = _zeros(
            leading_shape + (HEIGHT_SCAN_DIM,), root_linear, backend
        ) + FLAT_HEIGHT_SCAN_VALUE
    else:
        height_scan = _validated(
            "height_scan",
            height_scan,
            HEIGHT_SCAN_DIM,
            backend,
            leading_shape,
        )

    padded_action = _concatenate(
        (
            last_action,
            _zeros(leading_shape + (6,), root_linear, backend),
        ),
        backend,
    )
    parts = (
        _clip(root_linear, -100.0, 100.0, backend) * 2.0,
        _clip(root_angular, -100.0, 100.0, backend) * 0.25,
        _clip(gravity, -100.0, 100.0, backend),
        _clip(command, -100.0, 100.0, backend),
        _clip(
            joint_position - default_joint_position,
            -100.0,
            100.0,
            backend,
        ),
        _clip(joint_velocity, -100.0, 100.0, backend) * 0.05,
        _clip(padded_action, -100.0, 100.0, backend),
        _clip(height_scan, -1.0, 1.0, backend),
        _clip(arm_target, -100.0, 100.0, backend),
        _clip(gripper, -1.0, 1.0, backend),
    )
    observation = _concatenate(parts, backend)
    _validate_array(
        "observation",
        observation,
        OBSERVATION_DIM,
        backend,
        leading_shape,
    )
    return observation


def infer(
    policy: Any,
    observation: Any,
    *,
    warmup_scale: float = 1.0,
) -> Any:
    """Run the TorchScript actor and return the finite 12-D applied action."""

    torch = _import_torch()
    if not torch.is_tensor(observation):
        raise TypeError("observation must be a torch.Tensor")
    leading_shape = _validate_array(
        "observation", observation, OBSERVATION_DIM, "torch"
    )
    scale = _validate_warmup_scale(warmup_scale)
    with torch.inference_mode():
        action = policy(observation)
    _validate_array("policy action", action, ACTION_DIM, "torch", leading_shape)
    action = action * scale
    _validate_array("applied action", action, ACTION_DIM, "torch", leading_shape)
    return action


def leg_target(action: Any) -> Any:
    """Convert a finite 12-D applied action into leg position targets."""

    backend = _backend(action)
    leading_shape = _validate_array("action", action, ACTION_DIM, backend)
    default = _constant(DEFAULT_LEG_POSE, action, backend)
    target = default + ACTION_SCALE * action
    _validate_array("leg target", target, ACTION_DIM, backend, leading_shape)
    return target


def _import_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Torch is required only for TorchScript locomotion inference"
        ) from error
    return torch


def _robot_field(robot_data: Any, name: str) -> Any:
    if isinstance(robot_data, Mapping):
        if name not in robot_data:
            raise KeyError(f"robot_data is missing {name}")
        return robot_data[name]
    try:
        return getattr(robot_data, name)
    except AttributeError as error:
        raise AttributeError(f"robot_data is missing {name}") from error


def _backend(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return "numpy"
    torch = _import_torch()
    if torch.is_tensor(value):
        return "torch"
    raise TypeError("locomotion values must be numpy arrays or torch tensors")


def _validated(
    name: str,
    value: Any,
    width: int,
    backend: str,
    leading_shape: tuple[int, ...],
) -> Any:
    _validate_array(name, value, width, backend, leading_shape)
    return value


def _validate_array(
    name: str,
    value: Any,
    width: int,
    backend: str,
    leading_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if backend == "numpy":
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} must use the numpy backend")
        is_float = np.issubdtype(value.dtype, np.floating)
        is_finite = bool(np.isfinite(value).all())
    else:
        torch = _import_torch()
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must use the torch backend")
        is_float = value.dtype.is_floating_point
        is_finite = bool(torch.isfinite(value).all().item())

    if value.ndim < 1 or value.shape[-1] != width:
        raise ValueError(f"{name} must have shape (..., {width})")
    actual_leading_shape = tuple(value.shape[:-1])
    if leading_shape is not None and actual_leading_shape != leading_shape:
        raise ValueError(
            f"{name} leading shape must be {leading_shape}, "
            f"got {actual_leading_shape}"
        )
    if not is_float:
        raise TypeError(f"{name} must have a floating-point dtype")
    if not is_finite:
        raise ValueError(f"{name} must contain only finite values")
    return actual_leading_shape


def _validate_warmup_scale(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("warmup_scale must be a real scalar")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("warmup_scale must be finite and in [0, 1]")
    return value


def _zeros(shape: tuple[int, ...], like: Any, backend: str) -> Any:
    if backend == "numpy":
        return np.zeros(shape, dtype=like.dtype)
    torch = _import_torch()
    return torch.zeros(shape, dtype=like.dtype, device=like.device)


def _constant(values: tuple[float, ...], like: Any, backend: str) -> Any:
    if backend == "numpy":
        return np.asarray(values, dtype=like.dtype)
    torch = _import_torch()
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def _clip(value: Any, lower: float, upper: float, backend: str) -> Any:
    if backend == "numpy":
        return np.clip(value, lower, upper)
    return value.clamp(min=lower, max=upper)


def _concatenate(values: tuple[Any, ...], backend: str) -> Any:
    if backend == "numpy":
        return np.concatenate(values, axis=-1)
    return _import_torch().cat(values, dim=-1)
