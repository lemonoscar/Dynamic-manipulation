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
    "exec_nav_to_pick": Phase.NAV_TO_SOURCE,
    "exec_pick": Phase.PICK,
    "exec_nav_to_place": Phase.NAV_TO_TARGET,
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


def phase_from_pct(value: object) -> Phase:
    """Map a raw PCT execution state without guessing unknown labels."""

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


def subtask_history(phase: Phase | int) -> tuple[str, ...]:
    """Return canonical subtasks completed before the current executable phase."""

    resolved = Phase(phase)
    if resolved not in PHASE_ORDER:
        raise ValueError(f"phase has no supervised subtask history: {resolved.name}")
    return tuple(phase_instruction(item) for item in PHASE_ORDER[: int(resolved)])


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
    completed_subtasks: Sequence[str] = (),
) -> str:
    """Build the single seen-task prompt shared by training and inference."""

    task = str(instruction).strip()
    if not task:
        raise ValueError("instruction must be non-empty")
    history = "None" if not completed_subtasks else " ".join(completed_subtasks)
    return (
        f"Task: {task}\n"
        "The head and wrist videos are ordered from oldest to newest.\n"
        f"Completed subtasks: {history}\n"
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


def compose_gated_action10(
    action: Sequence[float],
    domain: ActionDomain | int,
    *,
    gripper_latch: float,
) -> tuple[float, ...]:
    """Lift a compact head output into 10-D while hard-locking the inactive domain."""

    if not math.isfinite(gripper_latch) or not 0.0 <= gripper_latch <= 1.0:
        raise ValueError("gripper_latch must be finite and within [0, 1]")
    resolved = ActionDomain(domain)
    if resolved is ActionDomain.NAVIGATION:
        vx, wz = _finite_vector(action, NAVIGATION_ACTION_DIM, "navigation action")
        return (vx, 0.0, wz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_latch)
    manipulation = _finite_vector(
        action,
        MANIPULATION_ACTION_DIM,
        "manipulation action",
    )
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
    "PCT_PHASES",
    "PRED_ACTION_TOKEN",
    "PHASE_DOMAINS",
    "PHASE_INSTRUCTIONS",
    "PHASE_ORDER",
    "SUBTASK_END_TOKEN",
    "SUBTASK_SPECIAL_TOKENS",
    "SUBTASK_START_TOKEN",
    "Phase",
    "SubtaskDecision",
    "action_domain",
    "compose_gated_action10",
    "phase_from_pct",
    "phase_instruction",
    "parse_subtask_solution",
    "project_action10",
    "subtask_history",
    "subtask_prompt",
    "subtask_solution",
]
