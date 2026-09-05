"""Immutable contracts for ConveyorVLA Joint-Trajectory Policy v1.

This successor intentionally does not import the Waypoint v1/v2 horizon,
DONE route, TCP action, prefix selector, or runtime masks.  It reuses the
token strings while initializing Qwen and both action trunks from ABot-M0.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Mapping, Sequence

from conveyor_bench.conveyorvla.waypoint import (
    PRED_ACTION_TOKEN,
    ROUTE_TOKENS as LEGACY_ROUTE_TOKENS,
    SUBTASK_END_TOKEN,
    SUBTASK_START_TOKEN,
    WaypointRoute,
)


MODEL_CONTRACT_ID = "conveyorvla-joint-trajectory-policy-5hz-v1"
DATASET_SCHEMA_VERSION = "conveyorvla-joint-trajectory-5hz-v1"
DATASET_PROFILE = "conveyorvla-liangzhunew500-sampled-control-5hz-v1"
POLICY_CONFIG_SCHEMA_VERSION = "conveyorvla-joint-trajectory-policy-config-5hz-v1"
NORMALIZATION_SCHEMA_VERSION = "conveyorvla-joint-trajectory-normalization-5hz-v1"
RUNTIME_PROTOCOL_VERSION = "conveyorvla-joint-trajectory-runtime/5hz-v1"

ACTION_HORIZON = 10
NAVIGATION_ACTION_DIM = 3
MANIPULATION_ACTION_DIM = 7
MANIPULATION_STATE_DIM = 13
NAVIGATION_STRIDE_S = 0.20
MANIPULATION_STRIDE_S = 0.20
HISTORY_SPAN_S = 0.20
SUCCESS_DWELL_S = 1.0
TRAIN_GLOBAL_BATCH_SIZE = 64
TRAIN_DOMAIN_ROWS_PER_BATCH = 32
TRAIN_BOUNDARY_PAIRS_PER_BATCH = 4


class JointTrajectoryRoute(str, Enum):
    NAV_TO_SOURCE = "NAV_TO_SOURCE"
    PICK = "PICK"
    NAV_TO_TARGET = "NAV_TO_TARGET"
    PLACE = "PLACE"


class JointTrajectoryDomain(str, Enum):
    NAVIGATION = "NAVIGATION"
    MANIPULATION = "MANIPULATION"


ROUTE_TOKENS: Mapping[JointTrajectoryRoute, str] = {
    route: LEGACY_ROUTE_TOKENS[WaypointRoute(route.value)]
    for route in JointTrajectoryRoute
}
ROUTE_DOMAINS: Mapping[JointTrajectoryRoute, JointTrajectoryDomain] = {
    JointTrajectoryRoute.NAV_TO_SOURCE: JointTrajectoryDomain.NAVIGATION,
    JointTrajectoryRoute.PICK: JointTrajectoryDomain.MANIPULATION,
    JointTrajectoryRoute.NAV_TO_TARGET: JointTrajectoryDomain.NAVIGATION,
    JointTrajectoryRoute.PLACE: JointTrajectoryDomain.MANIPULATION,
}
ROUTE_SUBTASKS: Mapping[JointTrajectoryRoute, str] = {
    JointTrajectoryRoute.NAV_TO_SOURCE: "Approach the box holding the Coke can.",
    JointTrajectoryRoute.PICK: "Reach, align, grasp, and lift the Coke can.",
    JointTrajectoryRoute.NAV_TO_TARGET: "Carry the Coke can toward the destination box.",
    JointTrajectoryRoute.PLACE: "Lower, release, and retract from the destination box.",
}
EXPECTED_TRANSITIONS = (
    (JointTrajectoryRoute.NAV_TO_SOURCE, JointTrajectoryRoute.PICK),
    (JointTrajectoryRoute.PICK, JointTrajectoryRoute.NAV_TO_TARGET),
    (JointTrajectoryRoute.NAV_TO_TARGET, JointTrajectoryRoute.PLACE),
)
TRANSITION_TAU_S: Mapping[str, float] = {
    "NAV_TO_SOURCE->PICK": 0.20,
    "PICK->NAV_TO_TARGET": 0.30,
    "NAV_TO_TARGET->PLACE": 0.20,
}
ACTIVE_SPECIAL_TOKENS = (
    PRED_ACTION_TOKEN,
    *ROUTE_TOKENS.values(),
    SUBTASK_START_TOKEN,
    SUBTASK_END_TOKEN,
)


def joint_trajectory_prompt(global_instruction: str) -> str:
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
        "Output exactly:\n"
        "<|pred_action|><one route token><|subtask|>one short action "
        "command<|end_subtask|>\n\n"
        "Valid route tokens:\n"
        "<|route_nav_to_source|> approach the source object;\n"
        "<|route_pick|> reach, grasp, and lift the object;\n"
        "<|route_nav_to_target|> carry the object toward the destination;\n"
        "<|route_place|> place, release, and retract from the object.\n\n"
        "Always select exactly one active route from current visual evidence. "
        "Do not output DONE, describe the scene, or output any other text."
    )


def canonical_solution(route: JointTrajectoryRoute | str) -> str:
    resolved = JointTrajectoryRoute(route)
    return (
        PRED_ACTION_TOKEN
        + ROUTE_TOKENS[resolved]
        + SUBTASK_START_TOKEN
        + ROUTE_SUBTASKS[resolved]
        + SUBTASK_END_TOKEN
    )


def action_domain(
    route: JointTrajectoryRoute | str,
) -> JointTrajectoryDomain:
    return ROUTE_DOMAINS[JointTrajectoryRoute(route)]


def transition_name(old: JointTrajectoryRoute | str, new: JointTrajectoryRoute | str) -> str:
    left = JointTrajectoryRoute(old)
    right = JointTrajectoryRoute(new)
    if (left, right) not in EXPECTED_TRANSITIONS:
        raise ValueError(f"unsupported route transition: {left.value}->{right.value}")
    return f"{left.value}->{right.value}"


def transition_routes(value: str) -> tuple[JointTrajectoryRoute, JointTrajectoryRoute]:
    text = str(value)
    try:
        left, right = text.split("->", maxsplit=1)
        result = JointTrajectoryRoute(left), JointTrajectoryRoute(right)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid route transition: {value!r}") from error
    if result not in EXPECTED_TRANSITIONS:
        raise ValueError(f"unsupported route transition: {text}")
    return result


def fixed_action(
    value: Sequence[Sequence[float]],
    domain: JointTrajectoryDomain | str,
) -> tuple[tuple[float, ...], ...]:
    resolved = JointTrajectoryDomain(domain)
    width = (
        NAVIGATION_ACTION_DIM
        if resolved is JointTrajectoryDomain.NAVIGATION
        else MANIPULATION_ACTION_DIM
    )
    return _fixed_matrix(value, ACTION_HORIZON, width, f"{resolved.value} action")


def terminal_hold(
    valid_prefix: Sequence[Sequence[float]],
    domain: JointTrajectoryDomain | str,
) -> tuple[tuple[float, ...], ...]:
    """Pad a real current-route prefix to ten fully supervised targets."""

    resolved = JointTrajectoryDomain(domain)
    width = (
        NAVIGATION_ACTION_DIM
        if resolved is JointTrajectoryDomain.NAVIGATION
        else MANIPULATION_ACTION_DIM
    )
    if not 1 <= len(valid_prefix) <= ACTION_HORIZON:
        raise ValueError(f"terminal-hold prefix must contain 1..{ACTION_HORIZON} rows")
    prefix = tuple(
        _finite_vector(row, width, f"terminal_hold[{index}]")
        for index, row in enumerate(valid_prefix)
    )
    return prefix + (prefix[-1],) * (ACTION_HORIZON - len(prefix))


def terminal_hold_start_index(action: Sequence[Sequence[float]]) -> int:
    """Return the first repeated suffix index, or 10 when no hold is present."""

    if len(action) != ACTION_HORIZON:
        raise ValueError(f"action must contain {ACTION_HORIZON} rows")
    rows = tuple(tuple(float(component) for component in row) for row in action)
    for index in range(1, ACTION_HORIZON):
        if all(row == rows[index - 1] for row in rows[index:]):
            return index
    return ACTION_HORIZON


def mani_state(value: Sequence[float]) -> tuple[float, ...]:
    result = _finite_vector(value, MANIPULATION_STATE_DIM, "mani_state")
    if not 0.0 <= result[-1] <= 1.0:
        raise ValueError("mani_state gripper must be within [0, 1]")
    return result


def direct_joint_targets(
    query_joint_position: Sequence[float],
    action: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Reconstruct absolute q targets from one query anchor and delta-q rows."""

    anchor = _finite_vector(query_joint_position, 6, "query_joint_position")
    rows = fixed_action(action, JointTrajectoryDomain.MANIPULATION)
    result = []
    for row in rows:
        result.append(
            tuple(anchor[axis] + row[axis] for axis in range(6)) + (row[6],)
        )
    return tuple(result)


def soft_route_probability(boundary_signed_time_s: float, tau_s: float) -> float:
    signed = float(boundary_signed_time_s)
    tau = float(tau_s)
    if not math.isfinite(signed) or not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("boundary time and tau must be finite and tau positive")
    scaled = max(-60.0, min(60.0, signed / tau))
    return 1.0 / (1.0 + math.exp(-scaled))


def _fixed_matrix(
    value: Sequence[Sequence[float]], rows: int, columns: int, name: str
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or len(value) != rows:
        raise ValueError(f"{name} must have shape [{rows},{columns}]")
    return tuple(
        _finite_vector(row, columns, f"{name}[{index}]")
        for index, row in enumerate(value)
    )


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


__all__ = [
    "ACTION_HORIZON",
    "ACTIVE_SPECIAL_TOKENS",
    "DATASET_PROFILE",
    "DATASET_SCHEMA_VERSION",
    "EXPECTED_TRANSITIONS",
    "HISTORY_SPAN_S",
    "JointTrajectoryDomain",
    "JointTrajectoryRoute",
    "MANIPULATION_ACTION_DIM",
    "MANIPULATION_STATE_DIM",
    "MANIPULATION_STRIDE_S",
    "MODEL_CONTRACT_ID",
    "NAVIGATION_ACTION_DIM",
    "NAVIGATION_STRIDE_S",
    "NORMALIZATION_SCHEMA_VERSION",
    "POLICY_CONFIG_SCHEMA_VERSION",
    "ROUTE_SUBTASKS",
    "ROUTE_TOKENS",
    "RUNTIME_PROTOCOL_VERSION",
    "SUCCESS_DWELL_S",
    "TRANSITION_TAU_S",
    "action_domain",
    "canonical_solution",
    "direct_joint_targets",
    "fixed_action",
    "joint_trajectory_prompt",
    "mani_state",
    "soft_route_probability",
    "terminal_hold",
    "terminal_hold_start_index",
    "transition_name",
    "transition_routes",
]
