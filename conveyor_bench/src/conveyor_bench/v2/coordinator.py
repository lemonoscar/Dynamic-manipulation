"""Pure-Python target sequencing for multi-object ConveyorBench episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Sequence


class CoordinatorStatus(str, Enum):
    """Episode-level state owned by the target coordinator."""

    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CoordinatorTerminatedError(RuntimeError):
    """Raised when a terminal coordinator receives another outcome."""


@dataclass(frozen=True)
class TargetTransition:
    """Immutable result of one target-level success or failure."""

    target_id: str
    next_target_id: str | None
    status: CoordinatorStatus
    remaining_time_s: float
    failure_reason: str | None = None

    @property
    def episode_terminal(self) -> bool:
        return self.status is not CoordinatorStatus.ACTIVE

    @property
    def episode_success(self) -> bool:
        return self.status is CoordinatorStatus.SUCCEEDED


class SequentialTargetCoordinator:
    """Advance through two or more targets without simulator dependencies.

    The coordinator owns only episode-level sequencing and the shared wall of
    simulation time.  A per-target teacher remains responsible for motion and
    reports its terminal result with the target identity that produced it.
    """

    def __init__(
        self,
        target_ids: Sequence[str],
        *,
        episode_start_time_s: float,
        episode_timeout_s: float,
    ) -> None:
        if isinstance(target_ids, (str, bytes)):
            raise ValueError("target_ids must be a sequence of target IDs")
        resolved_ids = tuple(target_ids)
        if len(resolved_ids) < 2:
            raise ValueError("sequential coordination requires at least two targets")
        if any(
            not isinstance(target_id, str) or not target_id
            for target_id in resolved_ids
        ):
            raise ValueError("target IDs must be non-empty strings")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise ValueError("target IDs must be unique")

        self._validate_number(
            episode_start_time_s,
            "episode_start_time_s",
            allow_zero=True,
        )
        self._validate_number(
            episode_timeout_s,
            "episode_timeout_s",
            allow_zero=False,
        )

        self._target_ids = resolved_ids
        self._episode_start_time_s = float(episode_start_time_s)
        self._episode_timeout_s = float(episode_timeout_s)
        self._current_index = 0
        self._completed_target_ids: list[str] = []
        self._status = CoordinatorStatus.ACTIVE
        self._failure_reason: str | None = None
        self._last_transition_time_s = self._episode_start_time_s

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self._target_ids

    @property
    def status(self) -> CoordinatorStatus:
        return self._status

    @property
    def current_target_id(self) -> str | None:
        if self._status is not CoordinatorStatus.ACTIVE:
            return None
        return self._target_ids[self._current_index]

    @property
    def completed_target_ids(self) -> tuple[str, ...]:
        return tuple(self._completed_target_ids)

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def remaining_time_s(self, sim_time_s: float) -> float:
        """Return the non-negative episode budget remaining at ``sim_time_s``."""

        self._validate_sim_time(sim_time_s)
        elapsed = float(sim_time_s) - self._episode_start_time_s
        return max(0.0, self._episode_timeout_s - elapsed)

    def mark_success(
        self,
        target_id: str,
        *,
        sim_time_s: float,
    ) -> TargetTransition:
        """Complete the current target and advance, or finish the episode."""

        self._ensure_active()
        self._validate_current_target(target_id)
        self._validate_transition_time(sim_time_s)
        remaining = self.remaining_time_s(sim_time_s)
        if remaining <= 0.0:
            raise TimeoutError("episode time budget is exhausted")

        completed_id = self._target_ids[self._current_index]
        self._completed_target_ids.append(completed_id)
        self._last_transition_time_s = float(sim_time_s)
        if self._current_index == len(self._target_ids) - 1:
            self._status = CoordinatorStatus.SUCCEEDED
            next_target_id = None
        else:
            self._current_index += 1
            next_target_id = self._target_ids[self._current_index]

        return TargetTransition(
            target_id=completed_id,
            next_target_id=next_target_id,
            status=self._status,
            remaining_time_s=remaining,
        )

    def mark_failure(
        self,
        target_id: str,
        *,
        sim_time_s: float,
        reason: str,
    ) -> TargetTransition:
        """Terminate the episode immediately on the current target failure."""

        self._ensure_active()
        self._validate_current_target(target_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason must be a non-empty string")
        self._validate_transition_time(sim_time_s)
        remaining = self.remaining_time_s(sim_time_s)

        self._last_transition_time_s = float(sim_time_s)
        self._failure_reason = reason
        self._status = CoordinatorStatus.FAILED
        return TargetTransition(
            target_id=target_id,
            next_target_id=None,
            status=self._status,
            remaining_time_s=remaining,
            failure_reason=reason,
        )

    def _ensure_active(self) -> None:
        if self._status is not CoordinatorStatus.ACTIVE:
            raise CoordinatorTerminatedError(
                f"coordinator already terminated with status {self._status.value!r}"
            )

    def _validate_current_target(self, target_id: str) -> None:
        current = self.current_target_id
        if target_id != current:
            raise ValueError(
                f"target {target_id!r} does not match current target {current!r}"
            )

    def _validate_transition_time(self, sim_time_s: float) -> None:
        self._validate_sim_time(sim_time_s)
        if float(sim_time_s) < self._last_transition_time_s:
            raise ValueError("transition sim_time_s cannot move backwards")

    def _validate_sim_time(self, sim_time_s: float) -> None:
        self._validate_number(sim_time_s, "sim_time_s", allow_zero=True)
        if float(sim_time_s) < self._episode_start_time_s:
            raise ValueError("sim_time_s cannot be before episode start")

    @staticmethod
    def _validate_number(value: float, name: str, *, allow_zero: bool) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}")


__all__ = [
    "CoordinatorStatus",
    "CoordinatorTerminatedError",
    "SequentialTargetCoordinator",
    "TargetTransition",
]
