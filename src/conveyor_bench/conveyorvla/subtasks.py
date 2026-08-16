"""Four-phase task and two-domain runtime contract for ConveyorVLA AL0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence


class Phase(IntEnum):
    """Only legal high-level phases, in execution order."""

    NAV_TO_SOURCE = 0
    PICK = 1
    NAV_TO_TARGET = 2
    PLACE = 3
    DONE = 4


class ActionDomain(IntEnum):
    """Mutually exclusive low-level action spaces."""

    NAVIGATION = 0
    MANIPULATION = 1


PCT_PHASES = {
    "plan_nav_to_pick": Phase.NAV_TO_SOURCE,
    "exec_nav_to_pick": Phase.NAV_TO_SOURCE,
    # Planning observations are physical transition frames, not an online FSM.
    # Planning rows use the next executable subtask.  Verification rows retain
    # the expert whose physical result they are checking; the following 5 Hz
    # row then provides the continuous language-routing switch.
    "verify_pick_reachable": Phase.NAV_TO_SOURCE,
    "plan_pick": Phase.PICK,
    "exec_pick": Phase.PICK,
    "verify_pick_success": Phase.PICK,
    "plan_nav_to_place": Phase.NAV_TO_TARGET,
    "exec_nav_to_place": Phase.NAV_TO_TARGET,
    "verify_place_reachable": Phase.NAV_TO_TARGET,
    "plan_place": Phase.PLACE,
    "exec_place": Phase.PLACE,
}
PHASE_ORDER = (
    Phase.NAV_TO_SOURCE,
    Phase.PICK,
    Phase.NAV_TO_TARGET,
    Phase.PLACE,
)
PHASE_DOMAINS = {
    Phase.NAV_TO_SOURCE: ActionDomain.NAVIGATION,
    Phase.PICK: ActionDomain.MANIPULATION,
    Phase.NAV_TO_TARGET: ActionDomain.NAVIGATION,
    Phase.PLACE: ActionDomain.MANIPULATION,
}
PRED_ACTION_TOKEN = "<|pred_action|>"
SUBTASK_START_TOKEN = "<|subtask|>"
SUBTASK_END_TOKEN = "<|end_subtask|>"
SUBTASK_SPECIAL_TOKENS = (
    PRED_ACTION_TOKEN,
    SUBTASK_START_TOKEN,
    SUBTASK_END_TOKEN,
)
FULL_INSTRUCTION = (
    "Walk to the box holding the Coke can. Keep the base still and pick up "
    "the can. Lift it and retract the arm. Turn around and walk to the other "
    "empty box. Keep the base still and place the can on top of it."
)
PHASE_INSTRUCTIONS = {
    Phase.NAV_TO_SOURCE: "Walk to the box holding the Coke can.",
    Phase.PICK: "Pick up the Coke can, lift it, and retract the arm.",
    Phase.NAV_TO_TARGET: "Turn around and walk to the empty box.",
    Phase.PLACE: "Lower the Coke can onto the empty box and release it.",
}
NAVIGATION_ACTION_INDICES = (0, 2)
MANIPULATION_ACTION_INDICES = tuple(range(3, 10))
NAVIGATION_ACTION_DIM = len(NAVIGATION_ACTION_INDICES)
MANIPULATION_ACTION_DIM = len(MANIPULATION_ACTION_INDICES)
NAVIGATION_REFERENCE_MODES = {
    Phase.NAV_TO_SOURCE: "stow_open",
    Phase.NAV_TO_TARGET: "carry_closed",
}
# Joint-space references are intentionally explicit.  They are the per-phase
# medians of the immutable Liangzhu seen navigation observations; the gripper
# command below, rather than an empty TCP delta, distinguishes stow from carry.
NAVIGATION_ARM_JOINT_REFERENCES = {
    Phase.NAV_TO_SOURCE: (
        -1.2146218068664894e-05,
        8.995759708341211e-05,
        -3.996067607658915e-05,
        -0.001061238581314683,
        3.62528589903377e-05,
        -1.931453425640939e-06,
    ),
    Phase.NAV_TO_TARGET: (
        7.132788596209139e-05,
        0.000671310699544847,
        -4.186293608654523e-06,
        -0.002301583532243967,
        -6.11629438935779e-05,
        -1.2048939424857963e-05,
    ),
}
NAVIGATION_GRIPPER_REFERENCES = {
    Phase.NAV_TO_SOURCE: 1.0,
    Phase.NAV_TO_TARGET: 0.0,
}
NAVIGATION_ARM_MAX_STEP_RAD = (0.008, 0.010, 0.010, 0.010, 0.008, 0.010)


def phase_from_pct(value: object) -> Phase:
    """Map an explicitly enumerated raw PCT execution/transition state."""

    try:
        return PCT_PHASES[str(value)]
    except KeyError as error:
        raise ValueError(f"unsupported PCT execution phase: {value!r}") from error


def action_domain(phase: Phase | int) -> ActionDomain:
    """Return the only action domain allowed for an executable phase."""

    resolved = Phase(phase)
    try:
        return PHASE_DOMAINS[resolved]
    except KeyError as error:
        raise ValueError(f"phase has no action domain: {resolved.name}") from error


def phase_instruction(phase: Phase | int) -> str:
    """Return the grounded low-level instruction for one action head."""

    resolved = Phase(phase)
    try:
        return PHASE_INSTRUCTIONS[resolved]
    except KeyError as error:
        raise ValueError(f"phase has no action instruction: {resolved.name}") from error


def subtask_solution(phase: Phase | int) -> str:
    """Build the exact assistant answer used by both Qwen passes."""

    return (
        PRED_ACTION_TOKEN
        + SUBTASK_START_TOKEN
        + phase_instruction(phase)
        + SUBTASK_END_TOKEN
    )


def subtask_prompt(
    instruction: str,
    previous_prediction: str | None = None,
) -> str:
    """Build a prompt with no privileged task history.

    The optional memory is one model-produced prediction from the preceding
    observation.  Dataset ground truth must never be passed here at inference.
    """

    task = str(instruction).strip()
    if not task:
        raise ValueError("instruction must be non-empty")
    memory = ""
    if previous_prediction is not None:
        prediction = str(previous_prediction).strip()
        if prediction not in PHASE_INSTRUCTIONS.values():
            raise ValueError("previous prediction must be one canonical subtask")
        memory = f"Previous model prediction (may be wrong): {prediction}\n"
    return (
        f"Task: {task}\n"
        "The head and wrist videos are ordered from oldest to newest.\n"
        f"{memory}"
        "What should the robot do now? Output exactly one canonical subtask as "
        f"{PRED_ACTION_TOKEN}{SUBTASK_START_TOKEN}<subtask>{SUBTASK_END_TOKEN}"
    )


@dataclass(frozen=True)
class SubtaskDecision:
    """A fail-closed parse of the language subtask emitted by Qwen."""

    text: str
    phase: Phase
    domain: ActionDomain
    assistant_solution: str


@dataclass(frozen=True)
class NavigationAction:
    """Composed navigation command with an explicit arm/gripper reference."""

    phase: Phase
    base_velocity: tuple[float, float, float]
    arm_joint_positions: tuple[float, ...]
    gripper_open_fraction: float
    reference_mode: str
    joint_reference_kind: str = "joint_space"
    tcp_delta_used: bool = False


def parse_subtask_solution(value: str) -> SubtaskDecision:
    """Parse only the four supervised canonical answers; reject free-form guesses."""

    raw = str(value).strip()
    prefix = PRED_ACTION_TOKEN + SUBTASK_START_TOKEN
    if not raw.startswith(prefix):
        raise ValueError("subtask answer is missing the required assistant prefix")
    end = raw.find(SUBTASK_END_TOKEN, len(prefix))
    if end < 0:
        raise ValueError("subtask answer is missing the end delimiter")
    trailing = raw[end + len(SUBTASK_END_TOKEN) :].strip()
    if trailing not in {"", "<|im_end|>"}:
        raise ValueError("subtask answer contains trailing content")
    text = raw[len(prefix) : end].strip()
    matches = [phase for phase in PHASE_ORDER if phase_instruction(phase) == text]
    if len(matches) != 1:
        raise ValueError(f"unsupported canonical subtask: {text!r}")
    phase = matches[0]
    return SubtaskDecision(
        text=text,
        phase=phase,
        domain=action_domain(phase),
        assistant_solution=subtask_solution(phase),
    )


def project_action10(
    action: Sequence[float],
    domain: ActionDomain | int,
) -> tuple[float, ...]:
    """Project the legacy 10-D target into a domain-specific action space."""

    values = _finite_vector(action, 10, "action10")
    resolved = ActionDomain(domain)
    indices = (
        NAVIGATION_ACTION_INDICES
        if resolved is ActionDomain.NAVIGATION
        else MANIPULATION_ACTION_INDICES
    )
    return tuple(values[index] for index in indices)


def compose_navigation_action(
    phase: Phase | int,
    action: Sequence[float],
    *,
    measured_arm_joint_positions: Sequence[float] | None = None,
) -> NavigationAction:
    """Compose ``[vx, wz]`` with a rate-limited phase-specific joint reference."""

    resolved = Phase(phase)
    if action_domain(resolved) is not ActionDomain.NAVIGATION:
        raise ValueError("navigation composer requires a navigation phase")
    vx, wz = _finite_vector(action, NAVIGATION_ACTION_DIM, "navigation action")
    target = NAVIGATION_ARM_JOINT_REFERENCES[resolved]
    if measured_arm_joint_positions is None:
        arm = target
    else:
        measured = _finite_vector(
            measured_arm_joint_positions,
            len(target),
            "measured arm joints",
        )
        arm = tuple(
            current + max(-limit, min(limit, desired - current))
            for current, desired, limit in zip(
                measured,
                target,
                NAVIGATION_ARM_MAX_STEP_RAD,
                strict=True,
            )
        )
    return NavigationAction(
        phase=resolved,
        base_velocity=(vx, 0.0, wz),
        arm_joint_positions=arm,
        gripper_open_fraction=NAVIGATION_GRIPPER_REFERENCES[resolved],
        reference_mode=NAVIGATION_REFERENCE_MODES[resolved],
    )


def compose_manipulation_action10(action: Sequence[float]) -> tuple[float, ...]:
    """Lift the manipulation expert into 10-D while hard-locking the base."""

    manipulation = _finite_vector(action, MANIPULATION_ACTION_DIM, "manipulation action")
    return (0.0, 0.0, 0.0, *manipulation)


def _finite_vector(value: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


__all__ = [
    "ActionDomain",
    "FULL_INSTRUCTION",
    "MANIPULATION_ACTION_DIM",
    "MANIPULATION_ACTION_INDICES",
    "NAVIGATION_ACTION_DIM",
    "NAVIGATION_ACTION_INDICES",
    "NAVIGATION_ARM_JOINT_REFERENCES",
    "NAVIGATION_GRIPPER_REFERENCES",
    "NAVIGATION_REFERENCE_MODES",
    "PCT_PHASES",
    "PRED_ACTION_TOKEN",
    "PHASE_DOMAINS",
    "PHASE_INSTRUCTIONS",
    "PHASE_ORDER",
    "SUBTASK_END_TOKEN",
    "SUBTASK_SPECIAL_TOKENS",
    "SUBTASK_START_TOKEN",
    "Phase",
    "NavigationAction",
    "SubtaskDecision",
    "action_domain",
    "compose_manipulation_action10",
    "compose_navigation_action",
    "phase_from_pct",
    "phase_instruction",
    "parse_subtask_solution",
    "project_action10",
    "subtask_prompt",
    "subtask_solution",
]
