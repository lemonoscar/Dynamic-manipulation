"""Deterministic C0/C1 episode evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import BenchmarkConfig
from .protocol import FailureReason, StepSample, TaskManifest, TaskType


@dataclass(frozen=True)
class EpisodeEvaluation:
    success: bool
    failure_reason: FailureReason
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.success and self.failure_reason is not FailureReason.NONE:
            raise ValueError("a successful evaluation must use FailureReason.NONE")
        if not self.success and self.failure_reason is FailureReason.NONE:
            raise ValueError("a failed evaluation must include a failure reason")


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _base_metrics(
    task: TaskManifest, samples: Sequence[StepSample], verification_time_s: float | None
) -> dict[str, Any]:
    # Episode time is defined against reset at t=0; the first control sample is
    # normally at 0.02 s and must not make a full-duration rollout look short.
    start_time = 0.0
    max_lift = max(sample.object_xyz[2] - task.belt_surface_z_m for sample in samples)
    path_length = sum(
        _distance(previous.tcp_xyz, current.tcp_xyz)
        for previous, current in zip(samples, samples[1:])
    )
    belt_mae = sum(
        abs(sample.belt_command_speed_mps - sample.belt_measured_speed_mps)
        for sample in samples
    ) / len(samples)
    return {
        "duration_s": samples[-1].sim_time_s - start_time,
        "completion_time_s": (
            None if verification_time_s is None else verification_time_s - start_time
        ),
        "verification_time_s": verification_time_s,
        "max_lift_m": max_lift,
        "tcp_path_length_m": path_length,
        "belt_speed_mae_mps": belt_mae,
        "target_crossed_exit": any(sample.target_crossed_exit for sample in samples),
    }


def evaluate_episode(
    config: BenchmarkConfig,
    task: TaskManifest,
    samples: Sequence[StepSample],
) -> EpisodeEvaluation:
    """Evaluate an ordered episode with the frozen V0 success contract.

    C0 and C1 both require a closed gripper, contact on both fingers, the target
    reported inside the gripper, and a five-centimetre lift sustained for one
    second. C1 additionally requires verification before the target crosses the
    conveyor exit.
    """

    if not samples:
        return EpisodeEvaluation(False, FailureReason.NO_SAMPLES, {"sample_count": 0})

    thresholds = config.evaluation
    invalid_c0 = (
        task.task_type is TaskType.C0_STATIC_PICK
        and abs(task.belt_speed_mps) > thresholds.static_belt_tolerance_mps
    )
    invalid_c1 = (
        task.task_type is TaskType.C1_DYNAMIC_PICK
        and abs(task.belt_speed_mps) < thresholds.dynamic_belt_min_speed_mps
    )
    if invalid_c0 or invalid_c1:
        metrics = _base_metrics(task, samples, None)
        return EpisodeEvaluation(
            False, FailureReason.INVALID_TASK_CONFIGURATION, metrics
        )

    hold_start_s: float | None = None
    had_secure_sample = False
    dropped = False
    verification_time_s: float | None = None
    failure_reason: FailureReason | None = None

    for sample in samples:
        if sample.robot_fallen:
            failure_reason = FailureReason.ROBOT_FALLEN
            break
        if sample.forbidden_collision:
            failure_reason = FailureReason.FORBIDDEN_COLLISION
            break
        if sample.wrong_object_grasped:
            failure_reason = FailureReason.WRONG_OBJECT
            break
        if (
            task.task_type is TaskType.C1_DYNAMIC_PICK
            and sample.target_crossed_exit
        ):
            failure_reason = FailureReason.TARGET_MISSED
            break

        lifted = (
            sample.object_xyz[2] - task.belt_surface_z_m
            >= thresholds.lift_height_m
        )
        secure = (
            sample.gripper_closed
            and sample.left_contact
            and sample.right_contact
            and sample.target_in_gripper
            and lifted
        )
        if secure:
            had_secure_sample = True
            if hold_start_s is None:
                hold_start_s = sample.sim_time_s
            if sample.sim_time_s - hold_start_s >= thresholds.hold_time_s:
                verification_time_s = sample.sim_time_s
                break
        else:
            if hold_start_s is not None:
                dropped = True
            hold_start_s = None

    metrics = _base_metrics(task, samples, verification_time_s)
    metrics["sample_count"] = len(samples)
    metrics["hold_time_required_s"] = thresholds.hold_time_s
    metrics["lift_height_required_m"] = thresholds.lift_height_m

    if verification_time_s is not None:
        return EpisodeEvaluation(True, FailureReason.NONE, metrics)
    if failure_reason is not None:
        return EpisodeEvaluation(False, failure_reason, metrics)

    duration_s = samples[-1].sim_time_s
    if duration_s >= task.max_duration_s:
        failure_reason = FailureReason.TIMEOUT
    elif dropped or had_secure_sample:
        failure_reason = FailureReason.DROPPED
    else:
        failure_reason = FailureReason.GRASP_NOT_SECURED
    return EpisodeEvaluation(False, failure_reason, metrics)
