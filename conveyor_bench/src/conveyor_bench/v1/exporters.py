"""Lossless offline projections from canonical V1 episodes to VLA views.

The recorder's JSON/JSONL files remain the source of truth.  Exporters create
new files, never rewrite or remove canonical streams, and retain the original
10D action beside each model-specific projection.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4

from .config import PROTOCOL_VERSION, BenchmarkConfig
from .quality import audit_episode
from .validation import validate_v1_episode

EXPORT_SCHEMA_VERSION = "conveyor-bench-v1-export-1"
_REQUIRED_POLICY_CAMERA_IDS = frozenset({"head_rgb", "wrist_rgb"})
_CONTROL_STEPS_PER_MODEL_TICK = (
    BenchmarkConfig.v1().control_hz // BenchmarkConfig.v1().model_hz
)
_CANONICAL_FILENAMES = {
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
}


class ExportError(ValueError):
    """Raised when a canonical episode cannot be projected safely."""


@dataclass(frozen=True)
class ExportSummary:
    profile: str
    source_episode: Path
    output_path: Path
    record_count: int
    source_task_outcome: str
    source_failure_reason: str


@dataclass(frozen=True)
class SourceTaskResult:
    """Canonical task result copied into every derived export artifact."""

    outcome: str
    failure_reason: str


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{name} must be a JSON object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ExportError(f"{name} must be a finite number")
    return float(value)


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExportError(f"{name} must be a numeric sequence")
    if len(value) != size:
        raise ExportError(f"{name} must contain exactly {size} values")
    return tuple(_number(component, name) for component in value)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            return _mapping(json.load(stream), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read {path}: {error}") from error


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExportError(
                        f"{path}:{line_number} is not valid JSON: {error}"
                    ) from error
                yield _mapping(value, f"{path}:{line_number}")
    except OSError as error:
        raise ExportError(f"cannot read {path}: {error}") from error


def _source_task_result(episode_path: Path) -> SourceTaskResult:
    summary = _read_json(episode_path / "summary.json")
    success = summary.get("success")
    failure_reason = summary.get("failure_reason")
    if not isinstance(success, bool):
        raise ExportError("summary.success must be a bool")
    if not isinstance(failure_reason, str) or not failure_reason:
        raise ExportError("summary.failure_reason must be a non-empty string")
    if success and failure_reason != "none":
        raise ExportError(
            "a successful episode must use summary.failure_reason='none'"
        )
    if not success and failure_reason == "none":
        raise ExportError(
            "a failed episode must use a non-none summary.failure_reason"
        )
    if failure_reason == "runtime_error":
        raise ExportError(
            "episodes with failure_reason='runtime_error' cannot be exported"
        )
    return SourceTaskResult(
        outcome="success" if success else "failure",
        failure_reason=failure_reason,
    )


def validate_episode_for_export(
    episode_directory: str | Path,
) -> SourceTaskResult:
    """Reject operationally invalid or structurally corrupt canonical data."""

    episode_path = Path(episode_directory)
    source_result = _source_task_result(episode_path)
    try:
        ticks = load_model_tick_steps(episode_path)
    except ExportError as error:
        if "contains no samples" in str(error):
            raise ExportError(
                "episode contains no valid canonical records"
            ) from error
        raise ExportError(f"canonical record validation failed: {error}") from error
    if not ticks:
        raise ExportError("episode contains no valid canonical records")

    # Report training-observation dropout directly even when the same missing
    # tick would also make the canonical camera index incomplete.
    _validate_training_camera_coverage(episode_path, ticks)

    validation = validate_v1_episode(episode_path)
    if not validation.ok:
        details = "; ".join(validation.errors[:5])
        if len(validation.errors) > 5:
            details += f"; ... ({len(validation.errors) - 5} more)"
        raise ExportError(
            "strict canonical validation failed; episode data corruption "
            f"prevents export: {details}"
        )
    report = audit_episode(episode_path)
    if report.data_corrupted:
        issue_codes = sorted({issue.code for issue in report.issues})
        details = ", ".join(issue_codes) if issue_codes else "unknown"
        raise ExportError(
            f"episode data corruption prevents export ({details})"
        )
    if report.task_outcome.value != source_result.outcome:
        raise ExportError(
            "quality audit task outcome does not match canonical summary"
        )
    audited_reason = report.task_failure_reason or "none"
    if audited_reason != source_result.failure_reason:
        raise ExportError(
            "quality audit failure reason does not match canonical summary"
        )
    return source_result


def _validate_training_camera_coverage(
    episode_path: Path,
    ticks: Sequence[Mapping[str, Any]],
) -> None:
    """Require the frozen policy pair at every complete model tick."""

    manifest = _read_json(episode_path / "manifest.json")
    episode = _mapping(manifest.get("episode"), "manifest.episode")
    metadata = _mapping(episode.get("metadata"), "manifest.episode.metadata")
    cameras = _mapping(
        metadata.get("cameras"),
        "manifest.episode.metadata.cameras",
    )
    declared = {
        camera_id
        for camera_id, contract in cameras.items()
        if (
            isinstance(camera_id, str)
            and isinstance(contract, Mapping)
            and contract.get("role") == "policy_observation"
        )
    }
    missing_contracts = sorted(_REQUIRED_POLICY_CAMERA_IDS - declared)
    unexpected_contracts = sorted(declared - _REQUIRED_POLICY_CAMERA_IDS)
    if missing_contracts or unexpected_contracts:
        problems: list[str] = []
        if missing_contracts:
            problems.append(
                "missing policy camera contract(s): "
                + ", ".join(missing_contracts)
            )
        if unexpected_contracts:
            problems.append(
                "unexpected policy camera contract(s): "
                + ", ".join(unexpected_contracts)
            )
        raise ExportError(
            "episode is not training-export eligible; " + "; ".join(problems)
        )
    _training_source_ticks(ticks)


def _training_source_ticks(
    ticks: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate policy-frame cadence and omit only an unframed partial tail."""

    selected: list[Mapping[str, Any]] = []
    required = tuple(sorted(_REQUIRED_POLICY_CAMERA_IDS))
    for index, tick in enumerate(ticks):
        model_tick = _integer(tick.get("model_tick"), "model_tick")
        source_steps = tick.get("_source_control_sim_steps")
        if (
            isinstance(source_steps, (str, bytes))
            or not isinstance(source_steps, Sequence)
        ):
            raise ExportError("_source_control_sim_steps must be a sequence")
        control_step_count = len(source_steps)
        is_complete = control_step_count == _CONTROL_STEPS_PER_MODEL_TICK
        is_partial_tail = (
            control_step_count == 1 and index == len(ticks) - 1
        )
        if not is_complete and not is_partial_tail:
            raise ExportError(
                f"model_tick {model_tick} has {control_step_count} control "
                "samples; only complete ticks or one final single-sample "
                "partial tick are exportable"
            )

        policy_frames = _select_camera_frames(tick, required)
        recorded = tuple(
            sorted(
                _mapping(frame, "camera frame").get("camera_id")
                for frame in policy_frames
            )
        )
        has_policy_pair = recorded == required
        if is_complete and not has_policy_pair:
            raise ExportError(
                f"model_tick {model_tick} is complete but policy camera "
                f"frames must be exactly {required}; recorded {recorded}"
            )
        if is_partial_tail and recorded and not has_policy_pair:
            raise ExportError(
                f"final partial model_tick {model_tick} must contain either "
                f"no policy camera frames or exactly {required}; recorded "
                f"{recorded}"
            )
        if is_complete or has_policy_pair:
            selected.append(tick)
    return tuple(selected)


def _camera_frames_for_tick(
    group: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for step in group:
        frames = step.get("camera_frames", ())
        if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
            raise ExportError("camera_frames must be a sequence")
        for raw_frame in frames:
            frame = _mapping(raw_frame, "camera frame")
            camera_id = frame.get("camera_id")
            if not isinstance(camera_id, str) or not camera_id:
                raise ExportError("camera frame camera_id must be non-empty")
            frame_index = _integer(frame.get("frame_index"), "camera frame_index")
            previous = latest.get(camera_id)
            if previous is None or frame_index >= _integer(
                previous.get("frame_index"), "camera frame_index"
            ):
                latest[camera_id] = frame
    return [latest[camera_id] for camera_id in sorted(latest)]


def load_model_tick_steps(
    episode_directory: str | Path,
) -> tuple[Mapping[str, Any], ...]:
    """Collapse 50 Hz canonical steps to the latest sample at each 25 Hz tick."""

    steps_path = Path(episode_directory) / "steps.jsonl"
    output: list[Mapping[str, Any]] = []
    group: list[Mapping[str, Any]] = []
    current_tick: int | None = None

    def publish_group() -> None:
        if not group:
            return
        selected = dict(group[-1])
        selected["camera_frames"] = _camera_frames_for_tick(group)
        selected["_source_control_sim_steps"] = [
            _integer(step.get("sim_step"), "sim_step") for step in group
        ]
        output.append(selected)

    for step in _read_jsonl(steps_path):
        tick = _integer(step.get("model_tick"), "model_tick")
        _integer(step.get("sim_step"), "sim_step")
        _number(step.get("sim_time_s"), "sim_time_s")
        _canonical_action(step)
        _pose(step, "robot_root_world")
        _pose(step, "tcp_base")
        if current_tick is not None and tick < current_tick:
            raise ExportError("model_tick cannot decrease in steps.jsonl")
        if current_tick is None or tick == current_tick:
            group.append(step)
            current_tick = tick
            continue
        publish_group()
        group = [step]
        current_tick = tick
    publish_group()
    if not output:
        raise ExportError("steps.jsonl contains no samples")
    return tuple(output)


def _pose(
    step: Mapping[str, Any], field: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pose = _mapping(step.get(field), field)
    xyz = _vector(pose.get("xyz"), 3, f"{field}.xyz")
    wxyz = _vector(pose.get("wxyz"), 4, f"{field}.wxyz")
    norm = math.sqrt(sum(component * component for component in wxyz))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise ExportError(f"{field}.wxyz must be a unit quaternion")
    return xyz, tuple(component / norm for component in wxyz)


def _canonical_action(step: Mapping[str, Any]) -> tuple[float, ...]:
    action = _mapping(step.get("action"), "action")
    return _vector(action.get("values"), 10, "action.values")


def _quat_multiply(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_conjugate(
    quaternion: Sequence[float],
) -> tuple[float, float, float, float]:
    return (
        float(quaternion[0]),
        -float(quaternion[1]),
        -float(quaternion[2]),
        -float(quaternion[3]),
    )


def _rotate_vector(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    pure = (0.0, float(vector[0]), float(vector[1]), float(vector[2]))
    rotated = _quat_multiply(
        _quat_multiply(quaternion, pure), _quat_conjugate(quaternion)
    )
    return rotated[1], rotated[2], rotated[3]


def _quat_to_rotvec(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    q = tuple(float(component) for component in quaternion)
    if q[0] < 0.0:
        q = tuple(-component for component in q)
    vector_norm = math.sqrt(sum(component * component for component in q[1:]))
    if vector_norm <= 1.0e-12:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(vector_norm, max(0.0, q[0]))
    scale = angle / vector_norm
    return q[1] * scale, q[2] * scale, q[3] * scale


def _state6_base(step: Mapping[str, Any]) -> tuple[float, ...]:
    xyz, quaternion = _pose(step, "tcp_base")
    return xyz + _quat_to_rotvec(quaternion)


def _state6_world(step: Mapping[str, Any]) -> tuple[float, ...]:
    root_xyz, root_quaternion = _pose(step, "robot_root_world")
    tcp_xyz, tcp_quaternion = _pose(step, "tcp_base")
    tcp_world_xyz_local = _rotate_vector(root_quaternion, tcp_xyz)
    tcp_world_xyz = tuple(
        root + local
        for root, local in zip(root_xyz, tcp_world_xyz_local, strict=True)
    )
    tcp_world_quaternion = _quat_multiply(root_quaternion, tcp_quaternion)
    return tcp_world_xyz + _quat_to_rotvec(tcp_world_quaternion)


def _future_delta_action7(
    current: Mapping[str, Any], future: Mapping[str, Any]
) -> tuple[float, ...]:
    current_xyz, current_quaternion = _pose(current, "tcp_base")
    future_xyz, future_quaternion = _pose(future, "tcp_base")
    delta_xyz = tuple(
        future_value - current_value
        for current_value, future_value in zip(
            current_xyz, future_xyz, strict=True
        )
    )
    delta_quaternion = _quat_multiply(
        future_quaternion, _quat_conjugate(current_quaternion)
    )
    gripper = _canonical_action(current)[9]
    return delta_xyz + _quat_to_rotvec(delta_quaternion) + (gripper,)


def _m0_world_action7(step: Mapping[str, Any]) -> tuple[float, ...]:
    canonical = _canonical_action(step)
    _, root_quaternion = _pose(step, "robot_root_world")
    delta_xyz_world = _rotate_vector(root_quaternion, canonical[3:6])
    delta_rotvec_world = _rotate_vector(root_quaternion, canonical[6:9])
    return delta_xyz_world + delta_rotvec_world + (canonical[9],)


def _episode_context(
    episode_directory: str | Path,
) -> tuple[
    Path,
    Mapping[str, Any],
    Mapping[str, Any],
    SourceTaskResult,
]:
    episode_path = Path(episode_directory)
    manifest = _read_json(episode_path / "manifest.json")
    episode = _mapping(manifest.get("episode"), "manifest.episode")
    if episode.get("protocol_version") != PROTOCOL_VERSION:
        raise ExportError(f"episode protocol must be {PROTOCOL_VERSION!r}")
    task = _mapping(episode.get("task"), "manifest.episode.task")
    return episode_path, episode, task, _source_task_result(episode_path)


def _camera_ids_for_role(
    episode: Mapping[str, Any], role: str
) -> tuple[str, ...] | None:
    """Resolve frozen camera roles, retaining compatibility with old fixtures."""

    metadata = episode.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    cameras = metadata.get("cameras")
    if not isinstance(cameras, Mapping):
        return None
    resolved: list[str] = []
    for camera_id, raw_contract in cameras.items():
        if (
            isinstance(camera_id, str)
            and isinstance(raw_contract, Mapping)
            and raw_contract.get("role") == role
        ):
            resolved.append(camera_id)
    return tuple(sorted(resolved))


def _select_camera_frames(
    step: Mapping[str, Any],
    camera_ids: tuple[str, ...] | None,
) -> tuple[Mapping[str, Any], ...]:
    frames = step.get("camera_frames", ())
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise ExportError("camera_frames must be a sequence")
    selected: list[Mapping[str, Any]] = []
    allowed = None if camera_ids is None else set(camera_ids)
    for raw_frame in frames:
        frame = _mapping(raw_frame, "camera frame")
        camera_id = frame.get("camera_id")
        if not isinstance(camera_id, str) or not camera_id:
            raise ExportError("camera frame camera_id must be non-empty")
        if allowed is None or camera_id in allowed:
            selected.append(frame)
    return tuple(selected)


def _history_entry(
    step: Mapping[str, Any],
    policy_camera_ids: tuple[str, ...] | None,
) -> dict[str, Any]:
    return {
        "model_tick": _integer(step.get("model_tick"), "model_tick"),
        "sim_step": _integer(step.get("sim_step"), "sim_step"),
        "sim_time_s": _number(step.get("sim_time_s"), "sim_time_s"),
        "source_control_sim_steps": step.get("_source_control_sim_steps", ()),
        # The canonical episode retains observer-only images for debugging.
        # Model projections expose policy cameras only, preventing overview
        # leakage into either training profile.
        "camera_frames": _select_camera_frames(step, policy_camera_ids),
        "state6": _state6_base(step),
    }


def iter_dynamicvla_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield DynamicVLA history/state/future-offset chunks at model rate."""

    benchmark = config or BenchmarkConfig.v1()
    episode_path, episode, task, source_result = _episode_context(
        episode_directory
    )
    ticks = load_model_tick_steps(episode_path)
    policy_camera_ids = _camera_ids_for_role(
        episode, "policy_observation"
    )
    by_tick = {
        _integer(step.get("model_tick"), "model_tick"): step for step in ticks
    }
    zero7 = (0.0,) * 7
    zero10 = (0.0,) * 10
    zero3 = (0.0,) * 3

    for source in _training_source_ticks(ticks):
        source_tick = _integer(source.get("model_tick"), "model_tick")
        history_steps = [
            by_tick.get(source_tick + offset)
            for offset in benchmark.history_offsets_steps
        ]
        action_chunk: list[tuple[float, ...]] = []
        base_chunk: list[tuple[float, ...]] = []
        canonical_chunk: list[tuple[float, ...]] = []
        action_valid: list[bool] = []
        canonical_valid: list[bool] = []
        for index in range(benchmark.dynamicvla_chunk_size):
            action_source = by_tick.get(source_tick + index)
            future = by_tick.get(
                source_tick + index + benchmark.label_offset_steps
            )
            source_valid = action_source is not None
            label_valid = source_valid and future is not None
            canonical = (
                _canonical_action(action_source) if action_source else zero10
            )
            canonical_chunk.append(canonical)
            canonical_valid.append(source_valid)
            base_chunk.append(canonical[0:3] if source_valid else zero3)
            action_chunk.append(
                _future_delta_action7(action_source, future)
                if action_source is not None and future is not None
                else zero7
            )
            action_valid.append(label_valid)

        yield {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "profile": "dynamicvla",
            "source_episode_id": episode.get("episode_id"),
            "source_task_id": task.get("task_id"),
            "source_task_outcome": source_result.outcome,
            "source_failure_reason": source_result.failure_reason,
            "source_steps_path": "steps.jsonl",
            "instruction": task.get("instruction"),
            "model_tick": source_tick,
            "sim_step": source.get("sim_step"),
            "sim_time_s": source.get("sim_time_s"),
            "phase": source.get("phase"),
            "history_offsets_model_ticks": benchmark.history_offsets_steps,
            "history_valid_mask": tuple(
                step is not None for step in history_steps
            ),
            "history": tuple(
                _history_entry(step, policy_camera_ids)
                if step is not None
                else None
                for step in history_steps
            ),
            "policy_camera_ids": policy_camera_ids or (),
            "observer_cameras_excluded": policy_camera_ids is not None,
            "state6": _state6_base(source),
            "state_frame": "robot_root/base",
            "future_ee_offset_model_ticks": benchmark.label_offset_steps,
            "future_ee_offset_s": (
                benchmark.label_offset_steps / benchmark.model_hz
            ),
            "delta_action7_chunk": tuple(action_chunk),
            "delta_action_frame": "robot_root/base",
            "base_action3_chunk": tuple(base_chunk),
            "base_action_frame": "body",
            "canonical_action10_chunk": tuple(canonical_chunk),
            "canonical_valid_mask": tuple(canonical_valid),
            "action_valid_mask": tuple(action_valid),
            "chunk_size": benchmark.dynamicvla_chunk_size,
        }


def iter_m0_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield ABot-M0 right-arm-padded world-frame action chunks."""

    benchmark = config or BenchmarkConfig.v1()
    episode_path, episode, task, source_result = _episode_context(
        episode_directory
    )
    ticks = load_model_tick_steps(episode_path)
    policy_camera_ids = _camera_ids_for_role(
        episode, "policy_observation"
    )
    by_tick = {
        _integer(step.get("model_tick"), "model_tick"): step for step in ticks
    }
    zero7 = (0.0,) * 7
    zero10 = (0.0,) * 10
    zero3 = (0.0,) * 3

    for source in _training_source_ticks(ticks):
        source_tick = _integer(source.get("model_tick"), "model_tick")
        arm_chunk: list[tuple[float, ...]] = []
        padded_chunk: list[tuple[float, ...]] = []
        base_chunk: list[tuple[float, ...]] = []
        canonical_chunk: list[tuple[float, ...]] = []
        valid_mask: list[bool] = []
        for index in range(benchmark.m0_chunk_size):
            action_source = by_tick.get(source_tick + index)
            valid = action_source is not None
            canonical = (
                _canonical_action(action_source) if action_source else zero10
            )
            arm = _m0_world_action7(action_source) if action_source else zero7
            canonical_chunk.append(canonical)
            base_chunk.append(canonical[0:3] if valid else zero3)
            arm_chunk.append(arm)
            padded_chunk.append(zero7 + arm)
            valid_mask.append(valid)

        yield {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "profile": "m0",
            "source_episode_id": episode.get("episode_id"),
            "source_task_id": task.get("task_id"),
            "source_task_outcome": source_result.outcome,
            "source_failure_reason": source_result.failure_reason,
            "source_steps_path": "steps.jsonl",
            "instruction": task.get("instruction"),
            "model_tick": source_tick,
            "sim_step": source.get("sim_step"),
            "sim_time_s": source.get("sim_time_s"),
            "phase": source.get("phase"),
            "policy_camera_ids": policy_camera_ids or (),
            "policy_camera_frames": _select_camera_frames(
                source, policy_camera_ids
            ),
            "observer_cameras_excluded": policy_camera_ids is not None,
            "state6_world": _state6_world(source),
            "world_delta_arm7_chunk": tuple(arm_chunk),
            "right_padded_action14_chunk": tuple(padded_chunk),
            "action_frame": "world",
            "projection": (
                "rotate canonical robot-root/base-frame translation and "
                "rotation-vector deltas by robot_root_world quaternion"
            ),
            "base_action3_chunk": tuple(base_chunk),
            "base_action_frame": "body",
            "canonical_action10_chunk": tuple(canonical_chunk),
            "action_valid_mask": tuple(valid_mask),
            "chunk_size": benchmark.m0_chunk_size,
        }


def _guard_output_path(episode_path: Path, output_path: Path) -> None:
    source_files = {
        (episode_path / filename).resolve()
        for filename in _CANONICAL_FILENAMES
    }
    if output_path.resolve() in source_files:
        raise ExportError("an exporter cannot overwrite a canonical episode file")
    if output_path.exists():
        raise FileExistsError(f"export already exists: {output_path}")


def _write_jsonl_atomic(
    records: Iterable[Mapping[str, Any]], output_path: Path
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )
    count = 0
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for record in records:
                json.dump(
                    record,
                    stream,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                stream.write("\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return count


def export_dynamicvla_episode(
    episode_directory: str | Path,
    output_path: str | Path,
    config: BenchmarkConfig | None = None,
) -> ExportSummary:
    episode_path = Path(episode_directory)
    destination = Path(output_path)
    _guard_output_path(episode_path, destination)
    source_result = validate_episode_for_export(episode_path)
    count = _write_jsonl_atomic(
        iter_dynamicvla_records(episode_path, config), destination
    )
    return ExportSummary(
        "dynamicvla",
        episode_path,
        destination,
        count,
        source_result.outcome,
        source_result.failure_reason,
    )


def export_m0_episode(
    episode_directory: str | Path,
    output_path: str | Path,
    config: BenchmarkConfig | None = None,
) -> ExportSummary:
    episode_path = Path(episode_directory)
    destination = Path(output_path)
    _guard_output_path(episode_path, destination)
    source_result = validate_episode_for_export(episode_path)
    count = _write_jsonl_atomic(
        iter_m0_records(episode_path, config), destination
    )
    return ExportSummary(
        "m0",
        episode_path,
        destination,
        count,
        source_result.outcome,
        source_result.failure_reason,
    )


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "ExportError",
    "ExportSummary",
    "SourceTaskResult",
    "export_dynamicvla_episode",
    "export_m0_episode",
    "iter_dynamicvla_records",
    "iter_m0_records",
    "load_model_tick_steps",
    "validate_episode_for_export",
]
