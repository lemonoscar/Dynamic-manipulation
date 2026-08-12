"""Offline integrity and quality audit for canonical ConveyorBench V1 episodes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .config import PROTOCOL_VERSION, BenchmarkConfig

_SHA256_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


class IssueCategory(str, Enum):
    DATA_CORRUPTION = "data_corruption"
    QUALITY_RISK = "quality_risk"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class TaskOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class DataStatus(str, Enum):
    CLEAN = "clean"
    WARNING = "warning"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class AuditIssue:
    code: str
    category: IssueCategory
    severity: Severity
    message: str
    stream: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class FrameStats:
    """Image statistics supplied by an optional OpenCV/PIL integration."""

    black_fraction: float | None = None
    blur_score: float | None = None
    object_visibility: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.black_fraction is not None and not (
            isinstance(self.black_fraction, (int, float))
            and not isinstance(self.black_fraction, bool)
            and math.isfinite(self.black_fraction)
            and 0.0 <= self.black_fraction <= 1.0
        ):
            raise ValueError("black_fraction must be finite and within [0, 1]")
        if self.blur_score is not None and not (
            isinstance(self.blur_score, (int, float))
            and not isinstance(self.blur_score, bool)
            and math.isfinite(self.blur_score)
            and self.blur_score >= 0.0
        ):
            raise ValueError("blur_score must be finite and non-negative")
        for instance_id, visibility in self.object_visibility.items():
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError("object_visibility keys must be non-empty strings")
            if not (
                isinstance(visibility, (int, float))
                and not isinstance(visibility, bool)
                and math.isfinite(visibility)
                and 0.0 <= visibility <= 1.0
            ):
                raise ValueError(
                    "object visibility values must be finite and within [0, 1]"
                )


FrameStatsProvider = Callable[
    [Path, Mapping[str, Any]], FrameStats | Mapping[str, Any] | None
]


@dataclass(frozen=True)
class QualityThresholds:
    timestamp_tolerance_s: float = 0.002
    max_action_jump_linf: float = 0.50
    max_black_fraction: float = 0.98
    min_blur_score: float = 10.0
    min_object_visibility: float = 0.01
    min_visible_frame_fraction: float = 0.50
    max_stale_chunk_fraction: float = 0.25
    max_discarded_action_fraction: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "timestamp_tolerance_s",
            "max_action_jump_linf",
            "min_blur_score",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "max_black_fraction",
            "min_object_visibility",
            "min_visible_frame_fraction",
            "max_stale_chunk_fraction",
            "max_discarded_action_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class QualityReport:
    episode_id: str | None
    task_outcome: TaskOutcome
    task_failure_reason: str | None
    data_status: DataStatus
    issues: tuple[AuditIssue, ...]
    metrics: Mapping[str, Any]

    @property
    def data_corrupted(self) -> bool:
        return self.data_status is DataStatus.CORRUPT

    @property
    def training_eligible(self) -> bool:
        """Structural eligibility only; task-success filtering is separate."""

        return not self.data_corrupted

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_outcome": self.task_outcome.value,
            "task_failure_reason": self.task_failure_reason,
            "data_status": self.data_status.value,
            "data_corrupted": self.data_corrupted,
            "training_eligible": self.training_eligible,
            "issues": [
                {
                    **asdict(issue),
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                }
                for issue in self.issues
            ],
            "metrics": dict(self.metrics),
        }


class _Auditor:
    def __init__(
        self,
        episode_directory: Path,
        thresholds: QualityThresholds,
        frame_stats_provider: FrameStatsProvider | None,
    ) -> None:
        self.episode_directory = episode_directory
        self.thresholds = thresholds
        self.frame_stats_provider = frame_stats_provider
        self.issues: list[AuditIssue] = []

    def corruption(
        self,
        code: str,
        message: str,
        stream: str | None = None,
        line: int | None = None,
    ) -> None:
        self.issues.append(
            AuditIssue(
                code,
                IssueCategory.DATA_CORRUPTION,
                Severity.ERROR,
                message,
                stream,
                line,
            )
        )

    def warning(
        self,
        code: str,
        message: str,
        stream: str | None = None,
        line: int | None = None,
    ) -> None:
        self.issues.append(
            AuditIssue(
                code,
                IssueCategory.QUALITY_RISK,
                Severity.WARNING,
                message,
                stream,
                line,
            )
        )

    def read_json(self, name: str) -> Mapping[str, Any] | None:
        path = self.episode_directory / name
        if not path.is_file():
            self.corruption("missing_stream", f"required file is missing: {name}", name)
            return None
        try:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            self.corruption("invalid_json", str(error), name)
            return None
        if not isinstance(value, Mapping):
            self.corruption("schema", "top-level value must be an object", name)
            return None
        self.check_finite_tree(value, name)
        return value

    def read_jsonl(self, name: str) -> list[Mapping[str, Any]]:
        path = self.episode_directory / name
        if not path.is_file():
            self.corruption("missing_stream", f"required file is missing: {name}", name)
            return []
        rows: list[Mapping[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        self.corruption(
                            "invalid_json", str(error), name, line_number
                        )
                        continue
                    if not isinstance(value, Mapping):
                        self.corruption(
                            "schema",
                            "JSONL row must be an object",
                            name,
                            line_number,
                        )
                        continue
                    self.check_finite_tree(value, name, line_number)
                    rows.append(value)
        except OSError as error:
            self.corruption("stream_read_error", str(error), name)
        return rows

    def check_finite_tree(
        self, value: Any, stream: str, line: int | None = None
    ) -> None:
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, float) and not math.isfinite(item):
                self.corruption(
                    "non_finite_numeric",
                    "stream contains NaN or infinity",
                    stream,
                    line,
                )
                return
            if isinstance(item, Mapping):
                stack.extend(item.values())
            elif isinstance(item, Sequence) and not isinstance(
                item, (str, bytes)
            ):
                stack.extend(item)


def _is_int(value: Any, *, nonnegative: bool = True) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (not nonnegative or value >= 0)
    )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_vector(value: Any, size: int) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == size
        and all(_is_number(component) for component in value)
    )


def _valid_pose(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_vector(value.get("xyz"), 3)
        and _is_vector(value.get("wxyz"), 4)
    )


def _valid_twist(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_vector(value.get("linear_xyz"), 3)
        and _is_vector(value.get("angular_xyz"), 3)
    )


def _config_value(
    manifest: Mapping[str, Any] | None,
    name: str,
    default: int | tuple[int, ...],
) -> Any:
    if manifest is None:
        return default
    config = manifest.get("benchmark_config")
    if not isinstance(config, Mapping):
        return default
    return config.get(name, default)


def _validate_manifest(
    auditor: _Auditor,
    manifest: Mapping[str, Any] | None,
) -> tuple[
    str | None,
    Mapping[str, Any],
    dict[str, Mapping[str, Any]],
    set[str],
]:
    if manifest is None:
        return None, {}, {}, set()
    episode = manifest.get("episode")
    if not isinstance(episode, Mapping):
        auditor.corruption("manifest_schema", "manifest.episode is required", "manifest.json")
        return None, {}, {}, set()
    episode_id = episode.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        auditor.corruption("manifest_schema", "episode_id is required", "manifest.json")
        episode_id = None
    if episode.get("protocol_version") != PROTOCOL_VERSION:
        auditor.corruption(
            "protocol_version",
            f"protocol_version must be {PROTOCOL_VERSION}",
            "manifest.json",
        )
    asset_hashes = episode.get("asset_hashes", {})
    if not isinstance(asset_hashes, Mapping) or any(
        not isinstance(component_id, str)
        or not component_id
        or not isinstance(digest, str)
        or not _SHA256_DIGEST.fullmatch(digest)
        for component_id, digest in (
            asset_hashes.items() if isinstance(asset_hashes, Mapping) else ()
        )
    ):
        auditor.corruption(
            "asset_hash_schema",
            "asset_hashes must map component ids to SHA-256 digests",
            "manifest.json",
        )
    task = episode.get("task")
    if not isinstance(task, Mapping):
        auditor.corruption("manifest_schema", "episode.task is required", "manifest.json")
        return episode_id, {}, {}, set()
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        auditor.corruption(
            "language_missing", "task instruction is empty", "manifest.json"
        )
    raw_objects = task.get("objects")
    objects: dict[str, Mapping[str, Any]] = {}
    if not isinstance(raw_objects, Sequence) or isinstance(
        raw_objects, (str, bytes)
    ):
        auditor.corruption(
            "manifest_schema", "task.objects must be a list", "manifest.json"
        )
    else:
        for raw_object in raw_objects:
            if not isinstance(raw_object, Mapping):
                auditor.corruption(
                    "manifest_schema",
                    "task object entry must be an object",
                    "manifest.json",
                )
                continue
            instance_id = raw_object.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                auditor.corruption(
                    "manifest_schema",
                    "task object requires instance_id",
                    "manifest.json",
                )
            elif instance_id in objects:
                auditor.corruption(
                    "duplicate_identity",
                    f"duplicate object instance: {instance_id}",
                    "manifest.json",
                )
            else:
                objects[instance_id] = raw_object
    raw_scored = task.get("scored_object_ids")
    scored = (
        set(raw_scored)
        if isinstance(raw_scored, Sequence)
        and not isinstance(raw_scored, (str, bytes))
        and all(isinstance(value, str) for value in raw_scored)
        else set()
    )
    if not scored or not scored <= set(objects):
        auditor.corruption(
            "manifest_identity",
            "scored_object_ids must reference registered objects",
            "manifest.json",
        )
    config = manifest.get("benchmark_config")
    if not isinstance(config, Mapping):
        auditor.corruption(
            "manifest_schema", "benchmark_config is required", "manifest.json"
        )
    else:
        for name in ("physics_hz", "control_hz", "camera_hz", "model_hz"):
            if not _is_int(config.get(name)) or config[name] <= 0:
                auditor.corruption(
                    "clock_schema",
                    f"{name} must be a positive integer",
                    "manifest.json",
                )
        if (
            config.get("physics_hz") == 200
            and config.get("protocol_version") == PROTOCOL_VERSION
        ):
            auditor.corruption(
                "v1_clock_contract",
                "V1 locomotion-compatible physics_hz must be 400, not 200",
                "manifest.json",
            )
    return episode_id, task, objects, scored


def _validate_summary(
    auditor: _Auditor, summary: Mapping[str, Any] | None
) -> tuple[TaskOutcome, str | None]:
    if summary is None:
        return TaskOutcome.UNKNOWN, None
    success = summary.get("success")
    failure_reason = summary.get("failure_reason")
    if not isinstance(success, bool) or not isinstance(failure_reason, str):
        auditor.corruption(
            "summary_schema",
            "summary requires bool success and string failure_reason",
            "summary.json",
        )
        return TaskOutcome.UNKNOWN, None
    if success and failure_reason != "none":
        auditor.corruption(
            "summary_consistency",
            "successful task must use failure_reason='none'",
            "summary.json",
        )
    if not success and failure_reason == "none":
        auditor.corruption(
            "summary_consistency",
            "failed task requires a non-none failure_reason",
            "summary.json",
        )
    return (
        TaskOutcome.SUCCESS if success else TaskOutcome.FAILURE,
        None if success else failure_reason,
    )


def _validate_steps(
    auditor: _Auditor,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any] | None,
    registered_ids: set[str],
    scored_ids: set[str],
) -> tuple[dict[int, Mapping[str, Any]], dict[str, Any]]:
    control_hz = _config_value(
        manifest, "control_hz", BenchmarkConfig.v1().control_hz
    )
    model_hz = _config_value(
        manifest, "model_hz", BenchmarkConfig.v1().model_hz
    )
    control_dt = 1.0 / control_hz if _is_int(control_hz) and control_hz > 0 else 0.02
    model_dt = 1.0 / model_hz if _is_int(model_hz) and model_hz > 0 else 0.04
    by_sim_step: dict[int, Mapping[str, Any]] = {}
    previous_step: int | None = None
    previous_time: float | None = None
    previous_tick: int | None = None
    previous_action: tuple[float, ...] | None = None
    # Use the first control sample assigned to each model tick.  An episode
    # may terminate after the first half of its final 25 Hz tick; using the
    # last sample would then manufacture a 20 ms "model cadence" error even
    # though successive tick starts remain exactly 40 ms apart.
    tick_first_time: dict[int, float] = {}
    action_jump_count = 0

    for line_number, row in enumerate(rows, start=1):
        sim_step = row.get("sim_step")
        sim_time = row.get("sim_time_s")
        model_tick = row.get("model_tick")
        if not _is_int(sim_step) or not _is_number(sim_time) or not _is_int(
            model_tick
        ):
            auditor.corruption(
                "step_schema",
                "step requires non-negative sim_step/model_tick and finite sim_time_s",
                "steps.jsonl",
                line_number,
            )
            continue
        if previous_step is not None and sim_step <= previous_step:
            auditor.corruption(
                "step_order",
                "sim_step must increase strictly",
                "steps.jsonl",
                line_number,
            )
        if previous_time is not None:
            delta = sim_time - previous_time
            if delta <= 0:
                auditor.corruption(
                    "timestamp_order",
                    "sim_time_s must increase strictly",
                    "steps.jsonl",
                    line_number,
                )
            elif abs(delta - control_dt) > auditor.thresholds.timestamp_tolerance_s:
                auditor.corruption(
                    "control_cadence",
                    f"control timestamp delta {delta:.6f}s differs from {control_dt:.6f}s",
                    "steps.jsonl",
                    line_number,
                )
        if previous_tick is not None and not (
            previous_tick <= model_tick <= previous_tick + 1
        ):
            auditor.corruption(
                "model_tick_cadence",
                "model_tick must stay equal or advance by one",
                "steps.jsonl",
                line_number,
            )
        previous_step = sim_step
        previous_time = float(sim_time)
        previous_tick = model_tick
        by_sim_step[sim_step] = row
        tick_first_time.setdefault(model_tick, float(sim_time))

        required_structures = (
            _valid_pose(row.get("robot_root_world")),
            _valid_twist(row.get("robot_twist_world")),
            _valid_pose(row.get("tcp_base")),
        )
        if not all(required_structures):
            auditor.corruption(
                "robot_state_schema",
                "robot root/twist and tcp pose have invalid dimensions",
                "steps.jsonl",
                line_number,
            )
        joints = row.get("joints")
        if not (
            isinstance(joints, Mapping)
            and isinstance(joints.get("names"), Sequence)
            and not isinstance(joints.get("names"), (str, bytes))
            and _is_vector(
                joints.get("positions"), len(joints.get("names", ()))
            )
            and _is_vector(
                joints.get("velocities"), len(joints.get("names", ()))
            )
        ):
            auditor.corruption(
                "joint_schema",
                "joint names, positions, and velocities must align",
                "steps.jsonl",
                line_number,
            )
        action = row.get("action")
        values = action.get("values") if isinstance(action, Mapping) else None
        if not _is_vector(values, 10):
            auditor.corruption(
                "action_schema",
                "canonical action must contain 10 finite values",
                "steps.jsonl",
                line_number,
            )
        else:
            current_action = tuple(float(value) for value in values)
            if previous_action is not None:
                jump = max(
                    abs(current - previous)
                    for current, previous in zip(
                        current_action[:9], previous_action[:9], strict=True
                    )
                )
                if jump > auditor.thresholds.max_action_jump_linf:
                    action_jump_count += 1
                    auditor.warning(
                        "action_jump",
                        f"canonical action jump {jump:.4f} exceeds threshold",
                        "steps.jsonl",
                        line_number,
                    )
            previous_action = current_action

        selected = row.get("selected_object_id")
        phase = row.get("phase")
        if selected is not None and selected not in registered_ids:
            auditor.corruption(
                "step_identity",
                "selected_object_id is not registered",
                "steps.jsonl",
                line_number,
            )
        active_phases = {
            "pregrasp",
            "track",
            "descend",
            "close",
            "lift",
            "carry",
            "preplace",
            "place",
            "place_descend",
            "open",
            "retreat",
            "verify_place",
        }
        if phase in active_phases and selected is None:
            auditor.warning(
                "language_phase_alignment",
                f"phase {phase!r} has no selected object",
                "steps.jsonl",
                line_number,
            )
        if selected is not None and scored_ids and selected not in scored_ids:
            auditor.warning(
                "language_phase_alignment",
                f"selected object {selected!r} is not scored by the instruction",
                "steps.jsonl",
                line_number,
            )

    ordered_ticks = sorted(tick_first_time)
    for previous, current in zip(ordered_ticks, ordered_ticks[1:]):
        if current != previous + 1:
            auditor.corruption(
                "model_tick_gap",
                f"missing model ticks between {previous} and {current}",
                "steps.jsonl",
            )
            continue
        delta = tick_first_time[current] - tick_first_time[previous]
        if abs(delta - model_dt) > auditor.thresholds.timestamp_tolerance_s:
            auditor.corruption(
                "model_cadence",
                f"model timestamp delta {delta:.6f}s differs from {model_dt:.6f}s",
                "steps.jsonl",
            )
    return by_sim_step, {
        "step_count": len(rows),
        "model_tick_count": len(tick_first_time),
        "action_jump_count": action_jump_count,
        "expected_control_dt_s": control_dt,
        "expected_model_dt_s": model_dt,
    }


def _validate_objects(
    auditor: _Auditor,
    rows: Sequence[Mapping[str, Any]],
    step_by_sim_step: Mapping[int, Mapping[str, Any]],
    registered_ids: set[str],
    expected_horizons: tuple[int, ...],
) -> tuple[dict[int, set[str]], dict[str, Any]]:
    active_by_step: dict[int, set[str]] = {}
    seen_keys: set[tuple[int, str]] = set()
    for line_number, row in enumerate(rows, start=1):
        sim_step = row.get("sim_step")
        state = row.get("state")
        if not _is_int(sim_step) or sim_step not in step_by_sim_step:
            auditor.corruption(
                "object_step_reference",
                "object row references an unknown sim_step",
                "objects.jsonl",
                line_number,
            )
            continue
        if not isinstance(state, Mapping):
            auditor.corruption(
                "object_schema",
                "object row requires state",
                "objects.jsonl",
                line_number,
            )
            continue
        instance_id = state.get("instance_id")
        if not isinstance(instance_id, str) or instance_id not in registered_ids:
            auditor.corruption(
                "object_identity",
                "object state instance_id is not registered",
                "objects.jsonl",
                line_number,
            )
            continue
        key = (sim_step, instance_id)
        if key in seen_keys:
            auditor.corruption(
                "duplicate_identity",
                "duplicate object state for one sim_step",
                "objects.jsonl",
                line_number,
            )
        seen_keys.add(key)
        if not _valid_pose(state.get("pose_world")) or not _valid_twist(
            state.get("twist_world")
        ):
            auditor.corruption(
                "object_schema",
                "object pose/twist dimensions are invalid",
                "objects.jsonl",
                line_number,
            )
        if state.get("active") is True:
            active_by_step.setdefault(sim_step, set()).add(instance_id)
        labels = row.get("future_object_states")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            auditor.corruption(
                "future_label_schema",
                "future_object_states must be a list",
                "objects.jsonl",
                line_number,
            )
            continue
        if not all(
            isinstance(label, Mapping)
            and _is_int(label.get("horizon_steps"))
            and label.get("instance_id") == instance_id
            for label in labels
        ):
            auditor.corruption(
                "future_label_schema",
                "future labels require matching identity and integer horizon",
                "objects.jsonl",
                line_number,
            )
            continue
        for label in labels:
            valid = label.get("valid")
            if valid is True and (
                not _valid_pose(label.get("pose_world"))
                or not _valid_twist(label.get("twist_world"))
                or label.get("invalid_reason") is not None
            ):
                auditor.corruption(
                    "future_label_schema",
                    "valid future label requires finite pose/twist only",
                    "objects.jsonl",
                    line_number,
                )
            elif valid is False and (
                label.get("pose_world") is not None
                or label.get("twist_world") is not None
                or not isinstance(label.get("invalid_reason"), str)
                or not label.get("invalid_reason", "").strip()
            ):
                auditor.corruption(
                    "future_label_schema",
                    "invalid future label requires only invalid_reason",
                    "objects.jsonl",
                    line_number,
                )
            elif not isinstance(valid, bool):
                auditor.corruption(
                    "future_label_schema",
                    "future label valid must be a bool",
                    "objects.jsonl",
                    line_number,
                )
        horizons = tuple(label["horizon_steps"] for label in labels)
        if tuple(sorted(horizons)) != expected_horizons:
            auditor.corruption(
                "future_label_horizons",
                f"future labels must use horizons {expected_horizons}",
                "objects.jsonl",
                line_number,
            )
    return active_by_step, {
        "object_record_count": len(rows),
        "active_object_sample_count": sum(map(len, active_by_step.values())),
    }


def _validate_chunks(
    auditor: _Auditor,
    rows: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    chunk_ids: set[str] = set()
    stale_count = 0
    discarded_total = 0
    action_total = 0
    for line_number, row in enumerate(rows, start=1):
        chunk_id = row.get("chunk_id")
        actions = row.get("actions")
        profile = row.get("profile")
        if not isinstance(chunk_id, str) or not chunk_id:
            auditor.corruption(
                "chunk_schema",
                "action chunk requires chunk_id",
                "action_chunks.jsonl",
                line_number,
            )
            continue
        if chunk_id in chunk_ids:
            auditor.corruption(
                "duplicate_identity",
                f"duplicate action chunk {chunk_id}",
                "action_chunks.jsonl",
                line_number,
            )
        chunk_ids.add(chunk_id)
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            auditor.corruption(
                "chunk_schema",
                "actions must be a list",
                "action_chunks.jsonl",
                line_number,
            )
            continue
        expected_size = (
            config.m0_chunk_size
            if profile == "m0"
            else config.dynamicvla_chunk_size
            if profile == "dynamicvla"
            else None
        )
        if expected_size is None or len(actions) != expected_size:
            auditor.corruption(
                "chunk_shape",
                "action chunk profile or size is invalid",
                "action_chunks.jsonl",
                line_number,
            )
        if any(
            not isinstance(action, Mapping)
            or not _is_vector(action.get("values"), 10)
            for action in actions
        ):
            auditor.corruption(
                "chunk_shape",
                "every chunk action must be canonical 10D",
                "action_chunks.jsonl",
                line_number,
            )
        valid_from = row.get("valid_from_tick")
        valid_until = row.get("valid_until_tick")
        execute_from = row.get("execute_from_tick")
        execute_until = row.get("execute_until_tick")
        discarded = row.get("discarded_action_count")
        if not (
            _is_int(valid_from)
            and _is_int(valid_until)
            and valid_until - valid_from == len(actions)
            and _is_int(discarded)
        ):
            auditor.corruption(
                "chunk_accounting",
                "valid window and discard count are inconsistent",
                "action_chunks.jsonl",
                line_number,
            )
            continue
        if (execute_from is None) != (execute_until is None):
            auditor.corruption(
                "chunk_accounting",
                "execute window bounds must be supplied together",
                "action_chunks.jsonl",
                line_number,
            )
            executed = 0
        elif execute_from is None:
            executed = 0
        elif not (
            _is_int(execute_from)
            and _is_int(execute_until)
            and valid_from <= execute_from < execute_until <= valid_until
        ):
            auditor.corruption(
                "chunk_accounting",
                "execute window must lie inside the valid window",
                "action_chunks.jsonl",
                line_number,
            )
            executed = 0
        else:
            executed = execute_until - execute_from
        if executed + discarded != len(actions):
            auditor.corruption(
                "chunk_accounting",
                "executed and discarded actions do not account for the chunk",
                "action_chunks.jsonl",
                line_number,
            )
        stale = row.get("stale")
        if not isinstance(stale, bool):
            auditor.corruption(
                "chunk_schema",
                "stale must be a bool",
                "action_chunks.jsonl",
                line_number,
            )
        elif stale:
            stale_count += 1
            if discarded == 0:
                auditor.corruption(
                    "chunk_accounting",
                    "stale chunk must discard at least one action",
                    "action_chunks.jsonl",
                    line_number,
                )
        discard_reason = row.get("discard_reason")
        if discarded > 0 and (
            not isinstance(discard_reason, str) or not discard_reason.strip()
        ):
            auditor.corruption(
                "chunk_accounting",
                "discarded actions require discard_reason",
                "action_chunks.jsonl",
                line_number,
            )
        elif discarded == 0 and discard_reason is not None:
            auditor.corruption(
                "chunk_accounting",
                "discard_reason requires discarded actions",
                "action_chunks.jsonl",
                line_number,
            )
        discarded_total += discarded
        action_total += len(actions)

    for line_number, step in enumerate(steps, start=1):
        referenced = step.get("action_chunk_id")
        if referenced is not None and referenced not in chunk_ids:
            auditor.corruption(
                "chunk_reference",
                "step references an unknown action chunk",
                "steps.jsonl",
                line_number,
            )
    chunk_count = len(rows)
    stale_fraction = stale_count / chunk_count if chunk_count else 0.0
    discard_fraction = discarded_total / action_total if action_total else 0.0
    if stale_count:
        auditor.warning(
            "stale_action_chunk",
            f"{stale_count} action chunks were stale",
            "action_chunks.jsonl",
        )
    if stale_fraction > auditor.thresholds.max_stale_chunk_fraction:
        auditor.warning(
            "stale_chunk_rate",
            f"stale chunk fraction {stale_fraction:.3f} exceeds threshold",
            "action_chunks.jsonl",
        )
    if discard_fraction > auditor.thresholds.max_discarded_action_fraction:
        auditor.warning(
            "discarded_action_rate",
            f"discarded action fraction {discard_fraction:.3f} exceeds threshold",
            "action_chunks.jsonl",
        )
    return {
        "action_chunk_count": chunk_count,
        "stale_action_chunk_count": stale_count,
        "stale_action_chunk_fraction": stale_fraction,
        "discarded_action_count": discarded_total,
        "discarded_action_fraction": discard_fraction,
    }


def _validate_events(
    auditor: _Auditor, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    previous_time: float | None = None
    for line_number, row in enumerate(rows, start=1):
        time_s = row.get("time_s")
        if not _is_number(time_s) or time_s < 0:
            auditor.corruption(
                "event_schema",
                "event time_s must be finite and non-negative",
                "events.jsonl",
                line_number,
            )
            continue
        if previous_time is not None and time_s < previous_time:
            auditor.corruption(
                "event_order",
                "event time_s cannot decrease",
                "events.jsonl",
                line_number,
            )
        previous_time = float(time_s)
    return {"event_count": len(rows)}


def _coerce_frame_stats(value: Any) -> FrameStats | None:
    if value is None or isinstance(value, FrameStats):
        return value
    if isinstance(value, Mapping):
        return FrameStats(
            black_fraction=value.get("black_fraction"),
            blur_score=value.get("blur_score"),
            object_visibility=value.get("object_visibility", {}),
        )
    raise ValueError("frame stats provider must return FrameStats, mapping, or None")


def _audit_frames(
    auditor: _Auditor,
    steps: Sequence[Mapping[str, Any]],
    active_by_step: Mapping[int, set[str]],
    scored_ids: set[str],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    placeholder = {
        "frame_stats_available": auditor.frame_stats_provider is not None,
        "camera_frame_ref_count": 0,
        "frame_stats_count": 0,
        "black_frame_count": 0,
        "black_frame_fraction": None,
        "blurred_frame_count": 0,
        "blurred_frame_fraction": None,
        "object_visibility": {},
    }
    checked = 0
    black = 0
    blurred = 0
    visibility_counts: dict[str, list[int]] = {
        instance_id: [0, 0] for instance_id in scored_ids
    }
    camera_hz = _config_value(
        manifest, "camera_hz", BenchmarkConfig.v1().camera_hz
    )
    camera_dt = 1.0 / camera_hz if _is_int(camera_hz) and camera_hz > 0 else 0.04
    last_camera_frame: dict[str, tuple[int, float]] = {}
    audited_frame_keys: set[tuple[str, int]] = set()
    for line_number, step in enumerate(steps, start=1):
        frames = step.get("camera_frames", ())
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            auditor.corruption(
                "camera_schema",
                "camera_frames must be a list",
                "steps.jsonl",
                line_number,
            )
            continue
        placeholder["camera_frame_ref_count"] += len(frames)
        sim_step = step.get("sim_step")
        active = active_by_step.get(sim_step, set()) if _is_int(sim_step) else set()
        for frame in frames:
            if not isinstance(frame, Mapping):
                auditor.corruption(
                    "camera_schema",
                    "camera frame reference must be an object",
                    "steps.jsonl",
                    line_number,
                )
                continue
            camera_id = frame.get("camera_id")
            frame_index = frame.get("frame_index")
            capture_time = frame.get("capture_time_s")
            if (
                not isinstance(camera_id, str)
                or not camera_id
                or not _is_int(frame_index)
                or not _is_number(capture_time)
                or capture_time < 0
            ):
                auditor.corruption(
                    "camera_schema",
                    "camera reference requires id, frame index, and capture time",
                    "steps.jsonl",
                    line_number,
                )
                continue
            previous_frame = last_camera_frame.get(camera_id)
            if previous_frame is not None:
                previous_index, previous_capture = previous_frame
                if frame_index < previous_index or capture_time < previous_capture:
                    auditor.corruption(
                        "camera_cadence",
                        "camera frame index/time cannot decrease",
                        "steps.jsonl",
                        line_number,
                    )
                elif frame_index > previous_index:
                    if frame_index != previous_index + 1:
                        auditor.corruption(
                            "camera_cadence",
                            "camera frame index must advance by one",
                            "steps.jsonl",
                            line_number,
                        )
                    delta = capture_time - previous_capture
                    if (
                        abs(delta - camera_dt)
                        > auditor.thresholds.timestamp_tolerance_s
                    ):
                        auditor.corruption(
                            "camera_cadence",
                            f"camera timestamp delta {delta:.6f}s differs "
                            f"from {camera_dt:.6f}s",
                            "steps.jsonl",
                            line_number,
                        )
            if previous_frame is None or frame_index >= previous_frame[0]:
                last_camera_frame[camera_id] = (frame_index, float(capture_time))
            relative_path = frame.get("relative_path")
            frame_path = auditor.episode_directory
            if relative_path is not None:
                if not isinstance(relative_path, str):
                    auditor.corruption(
                        "camera_schema",
                        "relative_path must be a string or null",
                        "steps.jsonl",
                        line_number,
                    )
                    continue
                path = PurePosixPath(relative_path)
                if path.is_absolute() or ".." in path.parts:
                    auditor.corruption(
                        "camera_path",
                        "camera path escapes the episode directory",
                        "steps.jsonl",
                        line_number,
                    )
                    continue
                frame_path = auditor.episode_directory / relative_path
            if auditor.frame_stats_provider is None:
                continue
            frame_key = (camera_id, frame_index)
            if frame_key in audited_frame_keys:
                continue
            audited_frame_keys.add(frame_key)
            try:
                stats = _coerce_frame_stats(
                    auditor.frame_stats_provider(frame_path, frame)
                )
            except Exception as error:
                auditor.warning(
                    "frame_stats_error",
                    f"frame statistics failed: {error}",
                    "steps.jsonl",
                    line_number,
                )
                continue
            if stats is None:
                continue
            checked += 1
            if (
                stats.black_fraction is not None
                and stats.black_fraction > auditor.thresholds.max_black_fraction
            ):
                black += 1
                auditor.warning(
                    "camera_black_frame",
                    f"black fraction {stats.black_fraction:.3f} exceeds threshold",
                    "steps.jsonl",
                    line_number,
                )
            if (
                stats.blur_score is not None
                and stats.blur_score < auditor.thresholds.min_blur_score
            ):
                blurred += 1
                auditor.warning(
                    "camera_blur",
                    f"blur score {stats.blur_score:.3f} is below threshold",
                    "steps.jsonl",
                    line_number,
                )
            for instance_id in scored_ids & active:
                if instance_id not in stats.object_visibility:
                    continue
                visibility_counts[instance_id][1] += 1
                if (
                    stats.object_visibility[instance_id]
                    >= auditor.thresholds.min_object_visibility
                ):
                    visibility_counts[instance_id][0] += 1
    visibility_metrics: dict[str, float | None] = {}
    for instance_id, (visible, total) in visibility_counts.items():
        fraction = visible / total if total else None
        visibility_metrics[instance_id] = fraction
        if (
            fraction is not None
            and fraction < auditor.thresholds.min_visible_frame_fraction
        ):
            auditor.warning(
                "object_visibility",
                f"{instance_id!r} visible in only {fraction:.3f} of checked frames",
                "steps.jsonl",
            )
    placeholder.update(
        {
            "frame_stats_count": checked,
            "black_frame_count": black,
            "black_frame_fraction": black / checked if checked else None,
            "blurred_frame_count": blurred,
            "blurred_frame_fraction": blurred / checked if checked else None,
            "object_visibility": visibility_metrics,
        }
    )
    return placeholder


def _audit_language(
    auditor: _Auditor,
    task: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    scored_ids: set[str],
) -> dict[str, Any]:
    instruction = task.get("instruction")
    if not isinstance(instruction, str):
        return {"language_target_resolved": False}
    normalized = instruction.lower().replace("-", " ").replace("_", " ")
    target_terms: list[str] = []
    for instance_id in scored_ids:
        obj = objects.get(instance_id, {})
        for value in (instance_id, obj.get("class_id")):
            if isinstance(value, str) and value:
                target_terms.append(
                    value.lower().replace("-", " ").replace("_", " ")
                )
    resolved = any(term in normalized for term in target_terms)
    if target_terms and not resolved:
        auditor.warning(
            "language_target_unresolved",
            "instruction does not name a scored object instance or class",
            "manifest.json",
        )
    return {"language_target_resolved": resolved, "language_target_terms": target_terms}


def audit_episode(
    episode_directory: str | Path,
    *,
    frame_stats_provider: FrameStatsProvider | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Audit data integrity independently from physical task success."""

    episode_path = Path(episode_directory)
    limits = thresholds or QualityThresholds()
    auditor = _Auditor(episode_path, limits, frame_stats_provider)

    manifest = auditor.read_json("manifest.json")
    summary = auditor.read_json("summary.json")
    steps = auditor.read_jsonl("steps.jsonl")
    objects_rows = auditor.read_jsonl("objects.jsonl")
    chunk_rows = auditor.read_jsonl("action_chunks.jsonl")
    event_rows = auditor.read_jsonl("events.jsonl")

    episode_id, task, registered_objects, scored_ids = _validate_manifest(
        auditor, manifest
    )
    task_outcome, failure_reason = _validate_summary(auditor, summary)
    step_by_sim_step, step_metrics = _validate_steps(
        auditor,
        steps,
        manifest,
        set(registered_objects),
        scored_ids,
    )
    raw_horizons = _config_value(
        manifest,
        "future_horizons_steps",
        BenchmarkConfig.v1().future_horizons_steps,
    )
    expected_horizons = (
        tuple(raw_horizons)
        if isinstance(raw_horizons, Sequence)
        and not isinstance(raw_horizons, (str, bytes))
        and all(_is_int(value) for value in raw_horizons)
        else BenchmarkConfig.v1().future_horizons_steps
    )
    active_by_step, object_metrics = _validate_objects(
        auditor,
        objects_rows,
        step_by_sim_step,
        set(registered_objects),
        expected_horizons,
    )
    chunk_metrics = _validate_chunks(
        auditor, chunk_rows, steps, BenchmarkConfig.v1()
    )
    event_metrics = _validate_events(auditor, event_rows)
    frame_metrics = _audit_frames(
        auditor, steps, active_by_step, scored_ids, manifest
    )
    language_metrics = _audit_language(
        auditor, task, registered_objects, scored_ids
    )

    if any(
        issue.category is IssueCategory.DATA_CORRUPTION
        for issue in auditor.issues
    ):
        data_status = DataStatus.CORRUPT
    elif auditor.issues:
        data_status = DataStatus.WARNING
    else:
        data_status = DataStatus.CLEAN
    metrics = {
        **step_metrics,
        **object_metrics,
        **chunk_metrics,
        **event_metrics,
        **frame_metrics,
        **language_metrics,
        "task_success": task_outcome is TaskOutcome.SUCCESS,
        "task_failure_is_data_corruption": False,
    }
    return QualityReport(
        episode_id,
        task_outcome,
        failure_reason,
        data_status,
        tuple(auditor.issues),
        metrics,
    )


__all__ = [
    "AuditIssue",
    "DataStatus",
    "FrameStats",
    "FrameStatsProvider",
    "IssueCategory",
    "QualityReport",
    "QualityThresholds",
    "Severity",
    "TaskOutcome",
    "audit_episode",
]
