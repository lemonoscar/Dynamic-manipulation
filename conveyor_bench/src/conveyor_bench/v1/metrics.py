"""Online release-and-settle metrics for ConveyorBench V1 sorting tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import BenchmarkConfig
from .protocol import FailureReason, StepSample, TaskManifest


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


@dataclass
class _ObjectProgress:
    goal_zone_id: str
    ever_held: bool = False
    was_in_gripper: bool = False
    released: bool = False
    release_time_s: float | None = None
    dwell_start_s: float | None = None
    completion_time_s: float | None = None
    crossed_exit: bool = False
    last_zone_ids: tuple[str, ...] = ()
    last_settled: bool = False
    last_seen_time_s: float | None = None


class OnlineEpisodeMetrics:
    """Bounded-memory evaluator updated once per control-rate sample."""

    def __init__(self, config: BenchmarkConfig, task: TaskManifest) -> None:
        self.config = config
        self.task = task
        object_by_id = task.object_by_id
        self._progress = {
            instance_id: _ObjectProgress(
                goal_zone_id=object_by_id[instance_id].goal_zone_id or ""
            )
            for instance_id in task.scored_object_ids
        }
        self._sample_count = 0
        self._object_record_count = 0
        self._last_sim_step: int | None = None
        self._last_model_tick: int | None = None
        self._last_time_s: float | None = None
        self._env_id: int | None = None
        self._failure_reason: FailureReason | None = None
        self._wrong_object_id: str | None = None

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def success(self) -> bool:
        return self._failure_reason is None and all(
            progress.completion_time_s is not None
            for progress in self._progress.values()
        )

    def update(self, sample: StepSample) -> None:
        """Consume one ordered sample without retaining it."""

        sample.validate_against(self.task, self.config)
        if self._last_sim_step is not None and sample.sim_step <= self._last_sim_step:
            raise ValueError("sim_step must increase strictly")
        if self._last_time_s is not None and sample.sim_time_s <= self._last_time_s:
            raise ValueError("sim_time_s must increase strictly")
        if (
            self._last_model_tick is not None
            and sample.model_tick < self._last_model_tick
        ):
            raise ValueError("model_tick cannot decrease")
        if self._env_id is None:
            self._env_id = sample.env_id
        elif sample.env_id != self._env_id:
            raise ValueError("env_id cannot change within an episode")

        self._sample_count += 1
        self._object_record_count += len(sample.objects)
        self._last_sim_step = sample.sim_step
        self._last_model_tick = sample.model_tick
        self._last_time_s = sample.sim_time_s

        if sample.robot_fallen:
            self._failure_reason = FailureReason.ROBOT_FALLEN
        elif (
            sample.forbidden_collision
            and self._failure_reason is not FailureReason.ROBOT_FALLEN
        ):
            self._failure_reason = FailureReason.FORBIDDEN_COLLISION

        scored_ids = set(self._progress)
        for obj in sample.objects:
            if (
                self._failure_reason is None
                and obj.in_gripper
                and obj.instance_id not in scored_ids
            ):
                self._failure_reason = FailureReason.WRONG_OBJECT
                self._wrong_object_id = obj.instance_id
                break

        state_by_id = {obj.instance_id: obj for obj in sample.objects}
        for instance_id, progress in self._progress.items():
            obj = state_by_id.get(instance_id)
            if obj is None:
                continue
            progress.last_seen_time_s = sample.sim_time_s
            progress.crossed_exit = progress.crossed_exit or (
                obj.crossed_exit and not obj.in_gripper
            )
            if (
                progress.crossed_exit
                and progress.completion_time_s is None
                and self._failure_reason is None
            ):
                self._failure_reason = FailureReason.TARGET_MISSED

            if obj.in_gripper:
                progress.ever_held = True
                if progress.released:
                    progress.released = False
                    progress.release_time_s = None
                    progress.dwell_start_s = None
            elif progress.was_in_gripper and progress.ever_held:
                progress.released = True
                progress.release_time_s = sample.sim_time_s
                progress.dwell_start_s = None
            progress.was_in_gripper = obj.in_gripper

            progress.last_zone_ids = tuple(
                zone.zone_id
                for zone in self.task.goal_zones
                if zone.contains(obj.pose_world.xyz)
            )
            linear_speed = math.sqrt(
                sum(value * value for value in obj.twist_world.linear_xyz)
            )
            angular_speed = math.sqrt(
                sum(value * value for value in obj.twist_world.angular_xyz)
            )
            progress.last_settled = (
                linear_speed
                <= self.config.evaluation.settled_linear_speed_mps
                and angular_speed
                <= self.config.evaluation.settled_angular_speed_radps
            )

            eligible_for_dwell = (
                obj.active
                and progress.released
                and not obj.in_gripper
                and progress.goal_zone_id in progress.last_zone_ids
                and progress.last_settled
            )
            if eligible_for_dwell and progress.completion_time_s is None:
                if progress.dwell_start_s is None:
                    progress.dwell_start_s = sample.sim_time_s
                if (
                    sample.sim_time_s - progress.dwell_start_s
                    >= self.config.evaluation.placement_dwell_s
                ):
                    progress.completion_time_s = sample.sim_time_s
            elif progress.completion_time_s is None:
                progress.dwell_start_s = None

    def snapshot(self) -> dict[str, Any]:
        object_outcomes: dict[str, dict[str, Any]] = {}
        for instance_id, progress in self._progress.items():
            if progress.completion_time_s is not None:
                status = "sorted_correct"
            elif progress.crossed_exit:
                status = "target_missed"
            elif progress.released and progress.last_zone_ids:
                status = (
                    "placement_not_settled"
                    if progress.goal_zone_id in progress.last_zone_ids
                    else "wrong_zone"
                )
            elif progress.released:
                status = "dropped"
            elif progress.ever_held:
                status = "held"
            else:
                status = "pending"
            object_outcomes[instance_id] = {
                "status": status,
                "goal_zone_id": progress.goal_zone_id,
                "ever_held": progress.ever_held,
                "released": progress.released,
                "release_time_s": progress.release_time_s,
                "dwell_start_s": progress.dwell_start_s,
                "completion_time_s": progress.completion_time_s,
                "crossed_exit": progress.crossed_exit,
                "last_zone_ids": progress.last_zone_ids,
                "last_settled": progress.last_settled,
                "last_seen_time_s": progress.last_seen_time_s,
            }

        completed = sum(
            progress.completion_time_s is not None
            for progress in self._progress.values()
        )
        completion_times = [
            progress.completion_time_s
            for progress in self._progress.values()
            if progress.completion_time_s is not None
        ]
        return {
            "sample_count": self._sample_count,
            "object_record_count": self._object_record_count,
            "duration_s": self._last_time_s or 0.0,
            "completion_time_s": (
                max(completion_times)
                if len(completion_times) == len(self._progress)
                else None
            ),
            "scored_object_count": len(self._progress),
            "completed_object_count": completed,
            "correct_sort_rate": completed / len(self._progress),
            "wrong_object_id": self._wrong_object_id,
            "object_outcomes": object_outcomes,
        }

    def finalize(self) -> EpisodeEvaluation:
        metrics = self.snapshot()
        if self._sample_count == 0:
            return EpisodeEvaluation(False, FailureReason.NO_SAMPLES, metrics)
        if self._failure_reason is not None:
            return EpisodeEvaluation(False, self._failure_reason, metrics)
        if self.success:
            return EpisodeEvaluation(True, FailureReason.NONE, metrics)

        progress_values = tuple(self._progress.values())
        if any(
            progress.released
            and progress.last_zone_ids
            and progress.goal_zone_id not in progress.last_zone_ids
            for progress in progress_values
        ):
            reason = FailureReason.WRONG_ZONE
        elif any(
            progress.released
            and progress.goal_zone_id in progress.last_zone_ids
            for progress in progress_values
        ):
            reason = FailureReason.PLACEMENT_NOT_SETTLED
        elif any(progress.released for progress in progress_values):
            reason = FailureReason.DROPPED
        elif (self._last_time_s or 0.0) >= self.task.max_duration_s:
            reason = FailureReason.TIMEOUT
        else:
            reason = FailureReason.INCOMPLETE
        return EpisodeEvaluation(False, reason, metrics)

    def failure_evaluation(
        self, reason: FailureReason, metadata: dict[str, Any] | None = None
    ) -> EpisodeEvaluation:
        if reason is FailureReason.NONE:
            raise ValueError("failure_evaluation requires a failure reason")
        metrics = self.snapshot()
        metrics["abort_metadata"] = metadata or {}
        return EpisodeEvaluation(False, reason, metrics)


def evaluate_episode(
    config: BenchmarkConfig,
    task: TaskManifest,
    samples: Iterable[StepSample],
) -> EpisodeEvaluation:
    tracker = OnlineEpisodeMetrics(config, task)
    for sample in samples:
        tracker.update(sample)
    return tracker.finalize()
