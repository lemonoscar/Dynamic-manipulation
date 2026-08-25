"""Shared ConveyorVLA Waypoint Policy v1 contracts and frame math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


MODEL_CONTRACT_ID = "qwen3vl-layerwise-dual-fm-waypoint-v1"
DATASET_SCHEMA_VERSION = "conveyorvla-waypoint-dense-transition-v1"
RUNTIME_PROTOCOL_VERSION = "conveyorvla-waypoint-runtime/v1"
ACTION_HORIZON = 20
NAVIGATION_STRIDE_S = 0.60
MANIPULATION_STRIDE_S = 0.20
HISTORY_SPAN_S = 0.20
CAMERA_CALIBRATION_ID = "liangzhu-0815-go2-x5-head-wrist-v1"
LABEL_FRAME_ID = "query-base-B_t"

PRED_ACTION_TOKEN = "<|pred_action|>"
PRED_DONE_TOKEN = "<|pred_done|>"
SUBTASK_START_TOKEN = "<|subtask|>"
SUBTASK_END_TOKEN = "<|end_subtask|>"


class WaypointRoute(str, Enum):
    NAV_TO_SOURCE = "NAV_TO_SOURCE"
    PICK = "PICK"
    NAV_TO_TARGET = "NAV_TO_TARGET"
    PLACE = "PLACE"
    DONE = "DONE"


class WaypointActionDomain(str, Enum):
    NAVIGATION = "NAVIGATION"
    MANIPULATION = "MANIPULATION"
    NONE = "NONE"


ROUTE_TOKENS: Mapping[WaypointRoute, str] = {
    WaypointRoute.NAV_TO_SOURCE: "<|route_nav_to_source|>",
    WaypointRoute.PICK: "<|route_pick|>",
    WaypointRoute.NAV_TO_TARGET: "<|route_nav_to_target|>",
    WaypointRoute.PLACE: "<|route_place|>",
}
SPECIAL_TOKENS = (
    PRED_ACTION_TOKEN,
    PRED_DONE_TOKEN,
    *ROUTE_TOKENS.values(),
    SUBTASK_START_TOKEN,
    SUBTASK_END_TOKEN,
)
ROUTE_DOMAINS: Mapping[WaypointRoute, WaypointActionDomain] = {
    WaypointRoute.NAV_TO_SOURCE: WaypointActionDomain.NAVIGATION,
    WaypointRoute.PICK: WaypointActionDomain.MANIPULATION,
    WaypointRoute.NAV_TO_TARGET: WaypointActionDomain.NAVIGATION,
    WaypointRoute.PLACE: WaypointActionDomain.MANIPULATION,
    WaypointRoute.DONE: WaypointActionDomain.NONE,
}
ROUTE_SUBTASKS: Mapping[WaypointRoute, str] = {
    WaypointRoute.NAV_TO_SOURCE: "Walk toward the box holding the Coke can.",
    WaypointRoute.PICK: "Move the gripper to grasp, lift, and retract the Coke can.",
    WaypointRoute.NAV_TO_TARGET: "Carry the Coke can toward the empty destination box.",
    WaypointRoute.PLACE: "Move the gripper to place and release the Coke can.",
}


def waypoint_prompt(global_instruction: str) -> str:
    """Return the one train/validation/online user prompt."""

    task = str(global_instruction).strip()
    if not task:
        raise ValueError("global_instruction must be non-empty")
    return (
        "You control a Go2-X5 mobile manipulator using only the ordered camera "
        "images.\n\n"
        f"Task: {task}\n\n"
        "The head-camera images and wrist-camera images are each ordered from "
        "oldest to newest. Decide what the robot should do now from current "
        "visual evidence.\n\n"
        "For an active action, output exactly:\n"
        "<|pred_action|><one route token><|subtask|>one short action "
        "command<|end_subtask|>\n\n"
        "Valid route tokens:\n"
        "<|route_nav_to_source|> approach the source object;\n"
        "<|route_pick|> move the gripper to grasp and lift the object;\n"
        "<|route_nav_to_target|> carry the object toward the destination;\n"
        "<|route_place|> move the gripper to place and release the object.\n\n"
        "If the whole task is visibly complete, output exactly <|pred_done|>. "
        "Otherwise select the best active route from current visual evidence. "
        "Do not describe the scene or output any other text."
    )


def canonical_solution(route: WaypointRoute | str) -> str:
    resolved = WaypointRoute(route)
    if resolved is WaypointRoute.DONE:
        return PRED_DONE_TOKEN
    return (
        PRED_ACTION_TOKEN
        + ROUTE_TOKENS[resolved]
        + SUBTASK_START_TOKEN
        + ROUTE_SUBTASKS[resolved]
        + SUBTASK_END_TOKEN
    )


def action_domain(route: WaypointRoute | str) -> WaypointActionDomain:
    return ROUTE_DOMAINS[WaypointRoute(route)]


def wrap_to_pi(value: float) -> float:
    result = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return -math.pi if result == math.pi else result


def unit_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = _finite_vector(value, 4, "quaternion")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1.0e-8:
        raise ValueError("quaternion has zero norm")
    return tuple(component / norm for component in quaternion)  # type: ignore[return-value]


def quaternion_conjugate(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    w, x, y, z = unit_quaternion(value)
    return w, -x, -y, -z


def quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quaternion_rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    q = unit_quaternion(quaternion)
    x, y, z = _finite_vector(vector, 3, "vector")
    rotated = quaternion_multiply(
        quaternion_multiply(q, (0.0, x, y, z)), quaternion_conjugate(q)
    )
    return rotated[1], rotated[2], rotated[3]


def quaternion_to_rpy(quaternion: Sequence[float]) -> tuple[float, float, float]:
    """Return roll/pitch/yaw for Rz(yaw) * Ry(pitch) * Rx(roll)."""

    w, x, y, z = unit_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_sine)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return wrap_to_pi(roll), pitch, wrap_to_pi(yaw)


def rpy_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Build the quaternion for Rz(yaw) * Ry(pitch) * Rx(roll)."""

    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return unit_quaternion(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _rpy_rotation_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the shortest physical rotation angle between two RPY poses."""

    left_quaternion = rpy_to_quaternion(*_finite_vector(left, 3, "left_rpy"))
    right_quaternion = rpy_to_quaternion(*_finite_vector(right, 3, "right_rpy"))
    dot = abs(
        sum(
            left_value * right_value
            for left_value, right_value in zip(
                left_quaternion, right_quaternion, strict=True
            )
        )
    )
    return 2.0 * math.acos(min(1.0, dot))


def yaw_from_quaternion(quaternion: Sequence[float]) -> float:
    return quaternion_to_rpy(quaternion)[2]


def nav_waypoint_body(
    query_base_world: Sequence[float], target_base_world: Sequence[float]
) -> tuple[float, float, float]:
    query = _finite_vector(query_base_world, 7, "query_base_world")
    target = _finite_vector(target_base_world, 7, "target_base_world")
    yaw = yaw_from_quaternion(query[3:])
    dx_world, dy_world = target[0] - query[0], target[1] - query[1]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        cosine * dx_world + sine * dy_world,
        -sine * dx_world + cosine * dy_world,
        wrap_to_pi(yaw_from_quaternion(target[3:]) - yaw),
    )


def nav_waypoint_world(
    query_base_world: Sequence[float], waypoint_body: Sequence[float]
) -> tuple[float, float, float]:
    query = _finite_vector(query_base_world, 7, "query_base_world")
    dx, dy, dyaw = _finite_vector(waypoint_body, 3, "waypoint_body")
    yaw = yaw_from_quaternion(query[3:])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        query[0] + cosine * dx - sine * dy,
        query[1] + sine * dx + cosine * dy,
        wrap_to_pi(yaw + dyaw),
    )


def arm_target_base(
    query_base_world: Sequence[float],
    target_tcp_world: Sequence[float],
    gripper_open_fraction: float,
) -> tuple[float, ...]:
    query = _finite_vector(query_base_world, 7, "query_base_world")
    target = _finite_vector(target_tcp_world, 7, "target_tcp_world")
    inverse = quaternion_conjugate(query[3:])
    position = quaternion_rotate(
        inverse,
        (target[0] - query[0], target[1] - query[1], target[2] - query[2]),
    )
    relative = unit_quaternion(
        quaternion_multiply(inverse, unit_quaternion(target[3:]))
    )
    gripper = float(gripper_open_fraction)
    if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
        raise ValueError("gripper_open_fraction must be within [0, 1]")
    return (*position, *quaternion_to_rpy(relative), gripper)


def arm_target_world(
    query_base_world: Sequence[float], target_base: Sequence[float]
) -> tuple[float, ...]:
    query = _finite_vector(query_base_world, 7, "query_base_world")
    target = _finite_vector(target_base, 7, "target_base")
    position_delta = quaternion_rotate(query[3:], target[:3])
    orientation = unit_quaternion(
        quaternion_multiply(query[3:], rpy_to_quaternion(*target[3:6]))
    )
    return (
        query[0] + position_delta[0],
        query[1] + position_delta[1],
        query[2] + position_delta[2],
        *orientation,
    )


@dataclass(frozen=True)
class NavWaypointSafety:
    max_segment_translation_m: float = 0.80
    max_segment_yaw_rad: float = math.radians(45.0)
    minimum_translation_m: float = 0.03
    minimum_yaw_rad: float = math.radians(3.0)


@dataclass(frozen=True)
class ArmTargetSafety:
    workspace_min_xyz: tuple[float, float, float] = (0.05, -0.50, -0.10)
    workspace_max_xyz: tuple[float, float, float] = (0.75, 0.50, 0.60)
    max_translation_step_m: float = 0.15
    max_axis_rotation_step_rad: float = math.radians(35.0)


def select_navigation_waypoint(
    waypoints: Sequence[Sequence[float]],
    valid_mask: Sequence[bool],
    *,
    safety: NavWaypointSafety = NavWaypointSafety(),
    validate_full_horizon: bool = True,
) -> tuple[int, tuple[float, float, float]]:
    """Select the first executable waypoint, strictly auditing the chunk by default.

    Disabling the full-horizon audit is reserved for receding-horizon diagnostics;
    every segment through the returned waypoint still uses the unchanged limits.
    """
    if not isinstance(validate_full_horizon, bool):
        raise ValueError("validate_full_horizon must be a boolean")
    rows = _fixed_chunk(waypoints, 3, "nav_waypoints_body")
    valid = _prefix_mask(valid_mask)
    previous = (0.0, 0.0, 0.0)
    selected: tuple[int, tuple[float, float, float]] | None = None
    for index, row in enumerate(rows):
        if not valid[index]:
            break
        translation = math.hypot(row[0] - previous[0], row[1] - previous[1])
        yaw_delta = abs(wrap_to_pi(row[2] - previous[2]))
        if translation > safety.max_segment_translation_m:
            raise ValueError(f"navigation segment {index} exceeds translation limit")
        if yaw_delta > safety.max_segment_yaw_rad:
            raise ValueError(f"navigation segment {index} exceeds yaw limit")
        if selected is None and (
            math.hypot(row[0], row[1]) >= safety.minimum_translation_m
            or abs(row[2]) >= safety.minimum_yaw_rad
        ):
            selected = (index, row)
            if not validate_full_horizon:
                return selected
        previous = row
    if selected is None:
        raise ValueError("navigation chunk has no non-degenerate valid waypoint")
    return selected


def rank_navigation_waypoints(
    waypoints: Sequence[Sequence[float]],
    valid_mask: Sequence[bool],
    *,
    trusted_horizon_points: int,
    minimum_lookahead_m: float,
    target_lookahead_m: float,
    safety: NavWaypointSafety = NavWaypointSafety(),
) -> tuple[tuple[int, tuple[float, float, float]], ...]:
    """Rank model local goals without treating unused rows as executed segments."""

    if (
        isinstance(trusted_horizon_points, bool)
        or not 1 <= trusted_horizon_points <= ACTION_HORIZON
    ):
        raise ValueError(
            f"trusted_horizon_points must be within [1, {ACTION_HORIZON}]"
        )
    minimum = float(minimum_lookahead_m)
    target = float(target_lookahead_m)
    if (
        not math.isfinite(minimum)
        or minimum <= 0.0
        or not math.isfinite(target)
        or target < minimum
    ):
        raise ValueError("lookahead distances must be finite and target >= minimum > 0")

    rows = _fixed_chunk(waypoints, 3, "nav_waypoints_body")
    valid = _prefix_mask(valid_mask)
    candidates: list[tuple[int, tuple[float, float, float], float]] = []
    for index, row in enumerate(rows[:trusted_horizon_points]):
        if not valid[index]:
            break
        radius = math.hypot(row[0], row[1])
        if radius >= safety.minimum_translation_m or abs(row[2]) >= safety.minimum_yaw_rad:
            candidates.append((index, row, radius))
    if not candidates:
        raise ValueError("navigation trusted prefix has no non-degenerate waypoint")

    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item[2] >= minimum else 1,
            abs(item[2] - target) if item[2] >= minimum else -item[2],
            item[0] if item[2] >= minimum else -item[0],
        ),
    )
    return tuple((index, row) for index, row, _radius in ranked)


def validate_arm_targets(
    targets: Sequence[Sequence[float]],
    valid_mask: Sequence[bool],
    current_tcp_base: Sequence[float],
    *,
    safety: ArmTargetSafety = ArmTargetSafety(),
) -> tuple[tuple[float, ...], ...]:
    rows = _fixed_chunk(targets, 7, "arm_targets_base")
    valid = _prefix_mask(valid_mask)
    previous = _finite_vector(current_tcp_base, 6, "current_tcp_base")
    accepted: list[tuple[float, ...]] = []
    for index, row in enumerate(rows):
        if not valid[index]:
            break
        if not all(
            safety.workspace_min_xyz[axis] <= row[axis] <= safety.workspace_max_xyz[axis]
            for axis in range(3)
        ):
            raise ValueError(f"arm target {index} is outside the workspace")
        translation = math.sqrt(sum((row[axis] - previous[axis]) ** 2 for axis in range(3)))
        rotation = _rpy_rotation_distance(previous[3:6], row[3:6])
        if translation > safety.max_translation_step_m:
            raise ValueError(f"arm target {index} exceeds translation step limit")
        if rotation > safety.max_axis_rotation_step_rad:
            raise ValueError(f"arm target {index} exceeds rotation step limit")
        if not 0.0 <= row[6] <= 1.0:
            raise ValueError(f"arm target {index} has an invalid gripper value")
        accepted.append(row)
        previous = row[:6]
    if not accepted:
        raise ValueError("arm chunk has no valid target")
    return tuple(accepted)


def _prefix_mask(value: Sequence[bool]) -> tuple[bool, ...]:
    if len(value) != ACTION_HORIZON:
        raise ValueError(f"action_valid_mask must contain {ACTION_HORIZON} values")
    result = tuple(bool(item) for item in value)
    if any(not earlier and later for earlier, later in zip(result, result[1:])):
        raise ValueError("action_valid_mask must be a true prefix")
    return result


def _fixed_chunk(
    value: Sequence[Sequence[float]], width: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if len(value) != ACTION_HORIZON:
        raise ValueError(f"{name} must contain {ACTION_HORIZON} rows")
    return tuple(_finite_vector(row, width, f"{name}[{index}]") for index, row in enumerate(value))


def _finite_vector(
    value: Sequence[float], length: int, name: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


__all__ = [
    "ACTION_HORIZON",
    "ArmTargetSafety",
    "CAMERA_CALIBRATION_ID",
    "DATASET_SCHEMA_VERSION",
    "HISTORY_SPAN_S",
    "LABEL_FRAME_ID",
    "MANIPULATION_STRIDE_S",
    "MODEL_CONTRACT_ID",
    "NAVIGATION_STRIDE_S",
    "NavWaypointSafety",
    "PRED_ACTION_TOKEN",
    "PRED_DONE_TOKEN",
    "ROUTE_SUBTASKS",
    "ROUTE_TOKENS",
    "RUNTIME_PROTOCOL_VERSION",
    "SPECIAL_TOKENS",
    "SUBTASK_END_TOKEN",
    "SUBTASK_START_TOKEN",
    "WaypointActionDomain",
    "WaypointRoute",
    "action_domain",
    "arm_target_base",
    "arm_target_world",
    "canonical_solution",
    "nav_waypoint_body",
    "nav_waypoint_world",
    "quaternion_to_rpy",
    "rpy_to_quaternion",
    "rank_navigation_waypoints",
    "select_navigation_waypoint",
    "unit_quaternion",
    "validate_arm_targets",
    "waypoint_prompt",
    "wrap_to_pi",
    "yaw_from_quaternion",
]
