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

from conveyor_bench.conveyorvla.temporal import (
    ACTION_HORIZON as AL0_TEMPORAL_ACTION_HORIZON,
    CAMERA_IDS as AL0_TEMPORAL_CAMERA_IDS,
    HISTORY_OFFSETS_MODEL_TICKS,
    JOINT_TASK_APPROACH_MIN_DISPLACEMENT_M,
    JOINT_TASK_BACKOFF_MIN_DISPLACEMENT_M,
    JOINT_TASK_CARRY_MIN_DISPLACEMENT_M,
    JOINT_TASK_PLACEMENT_MAX_DISPLACEMENT_M,
    JOINT_TASK_REQUIRED_PHASE_ORDER,
    JOINT_TRAINING_PHASES,
    MODEL_HZ as AL0_TEMPORAL_ACTION_RATE_HZ,
    POLICY_TASK_SCOPE,
    TEMPORAL_PROFILE as AL0_TEMPORAL_PROFILE,
    TEMPORAL_SCHEMA_VERSION as AL0_TEMPORAL_SCHEMA_VERSION,
    relative_tcp_target,
)

from .config import PROTOCOL_VERSION, BenchmarkConfig
from .quality import audit_episode
from .stationary import validate_stationary_episode_contract
from .validation import validate_v1_episode

EXPORT_SCHEMA_VERSION = "conveyor-bench-v1-export-1"
M0_MOBILE_SCHEMA_VERSION = "conveyor-bench-m0-mobile-v1"
M0_MOBILE_ACTION_HORIZON = 16
M0_MOBILE_ACTION_DIMENSION_MASK = (
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
)
M0_MOBILE_STATE_LAYOUT = (
    "root_linear_velocity_body.x",
    "root_linear_velocity_body.y",
    "root_linear_velocity_body.z",
    "root_angular_velocity_body.x",
    "root_angular_velocity_body.y",
    "root_angular_velocity_body.z",
    "projected_gravity_body.x",
    "projected_gravity_body.y",
    "projected_gravity_body.z",
    *(f"arm_joint_position.{index}" for index in range(1, 7)),
    *(f"arm_joint_velocity.{index}" for index in range(1, 7)),
    "tcp_position_base.x",
    "tcp_position_base.y",
    "tcp_position_base.z",
    "tcp_rotation_vector_base.x",
    "tcp_rotation_vector_base.y",
    "tcp_rotation_vector_base.z",
    "gripper_open_fraction",
)
_REQUIRED_POLICY_CAMERA_IDS = frozenset({"head_rgb", "wrist_rgb"})
_M0_DIAGNOSTIC_ASSIST_KEYS = (
    "m0_mobile_approach_assist",
    "m0_pregrasp_staging_assist",
    "m0_carry_retract_teacher_executor",
)
_M0_DIAGNOSTIC_CONTROL_LAYERS = frozenset(
    {
        "diagnostic_mobile_approach_assist",
        "diagnostic_pregrasp_staging_assist",
        "diagnostic_teacher_via_m0_executor",
    }
)
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


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExportError(f"{name} must be a sequence")
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


def canonical_to_m0_mobile_action(
    action: Sequence[float],
) -> tuple[float, ...]:
    """Map canonical gripper ``[-1, 1]`` to the model's ``[0, 1]``."""

    canonical = _vector(action, 10, "canonical action")
    return canonical[:9] + ((canonical[9] + 1.0) / 2.0,)


def m0_mobile_to_canonical_action(
    action: Sequence[float],
) -> tuple[float, ...]:
    """Invert :func:`canonical_to_m0_mobile_action`."""

    model_action = _vector(action, 10, "AL0 legacy-profile action")
    if not 0.0 <= model_action[9] <= 1.0:
        raise ExportError("AL0 gripper must be within [0, 1]")
    return model_action[:9] + (model_action[9] * 2.0 - 1.0,)


def _load_control_steps(
    episode_path: Path,
    benchmark: BenchmarkConfig,
) -> tuple[Mapping[str, Any], ...]:
    """Load the uncollapsed 50 Hz stream used for causal action labels."""

    rows = tuple(_read_jsonl(episode_path / "steps.jsonl"))
    if not rows:
        raise ExportError("steps.jsonl contains no samples")
    physics_steps_per_control = benchmark.physics_hz // benchmark.control_hz
    control_dt_s = 1.0 / benchmark.control_hz
    previous_step: int | None = None
    previous_time: float | None = None
    for row in rows:
        sim_step = _integer(row.get("sim_step"), "sim_step")
        sim_time_s = _number(row.get("sim_time_s"), "sim_time_s")
        _integer(row.get("model_tick"), "model_tick")
        _canonical_action(row)
        _pose(row, "robot_root_world")
        _pose(row, "tcp_base")
        if previous_step is not None and (
            sim_step - previous_step != physics_steps_per_control
            or not math.isclose(
                sim_time_s - previous_time,
                control_dt_s,
                rel_tol=0.0,
                abs_tol=1.0e-8,
            )
        ):
            raise ExportError("steps.jsonl is not a contiguous control-rate stream")
        previous_step = sim_step
        previous_time = sim_time_s
    return rows


def _joint_vectors(
    step: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    joints = _mapping(step.get("joints"), "joints")
    names = joints.get("names")
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ExportError("joints.names must be a sequence")
    if any(not isinstance(name, str) or not name for name in names):
        raise ExportError("joints.names must contain non-empty strings")
    if len(names) != len(set(names)):
        raise ExportError("joints.names must be unique")
    positions = _vector(joints.get("positions"), len(names), "joints.positions")
    velocities = _vector(
        joints.get("velocities"), len(names), "joints.velocities"
    )
    by_name = {
        name: (positions[index], velocities[index])
        for index, name in enumerate(names)
    }
    required = tuple(f"arm_joint{index}" for index in range(1, 9))
    missing = tuple(name for name in required if name not in by_name)
    if missing:
        raise ExportError(
            "AL0 state requires joint(s): " + ", ".join(missing)
        )
    arm_names = required[:6]
    arm_positions = tuple(by_name[name][0] for name in arm_names)
    arm_velocities = tuple(by_name[name][1] for name in arm_names)
    gripper_position = sum(by_name[name][0] for name in required[6:]) / 2.0
    gripper_open_fraction = min(1.0, max(0.0, gripper_position / 0.044))
    return arm_positions, arm_velocities, gripper_open_fraction


def _m0_mobile_state28(step: Mapping[str, Any]) -> tuple[float, ...]:
    _, root_quaternion = _pose(step, "robot_root_world")
    root_inverse = _quat_conjugate(root_quaternion)
    twist = _mapping(step.get("robot_twist_world"), "robot_twist_world")
    linear_world = _vector(
        twist.get("linear_xyz"), 3, "robot_twist_world.linear_xyz"
    )
    angular_world = _vector(
        twist.get("angular_xyz"), 3, "robot_twist_world.angular_xyz"
    )
    linear_body = _rotate_vector(root_inverse, linear_world)
    angular_body = _rotate_vector(root_inverse, angular_world)
    projected_gravity = _rotate_vector(root_inverse, (0.0, 0.0, -1.0))
    arm_positions, arm_velocities, gripper_open_fraction = _joint_vectors(step)
    tcp_xyz, tcp_quaternion = _pose(step, "tcp_base")
    state = (
        linear_body
        + angular_body
        + projected_gravity
        + arm_positions
        + arm_velocities
        + tcp_xyz
        + _quat_to_rotvec(tcp_quaternion)
        + (gripper_open_fraction,)
    )
    if len(state) != len(M0_MOBILE_STATE_LAYOUT):
        raise AssertionError("AL0 state layout and values disagree")
    return state


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


def _reject_m0_diagnostic_assist(
    episode: Mapping[str, Any],
    control_steps: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Keep diagnostic intervention out of standard AL0 exports."""

    metadata = _mapping(episode.get("metadata"), "manifest.episode.metadata")
    for key in _M0_DIAGNOSTIC_ASSIST_KEYS:
        raw_contract = metadata.get(key)
        if raw_contract is None:  # Legacy oracle episodes predate these flags.
            continue
        contract = _mapping(raw_contract, f"manifest.episode.metadata.{key}")
        enabled = contract.get("enabled")
        if not isinstance(enabled, bool):
            raise ExportError(f"{key}.enabled must be a bool")
        assisted = contract.get("assisted", False)
        if not isinstance(assisted, bool):
            raise ExportError(f"{key}.assisted must be a bool")
        if enabled or assisted:
            raise ExportError(
                "diagnostic-assisted episodes cannot be exported for "
                f"standard AL0 training ({key})"
            )

    for step in control_steps or ():
        step_metadata = step.get("metadata")
        if not isinstance(step_metadata, Mapping):
            continue
        online_action = step_metadata.get("m0_online_action")
        if not isinstance(online_action, Mapping):
            continue
        if online_action.get("control_layer") in _M0_DIAGNOSTIC_CONTROL_LAYERS:
            raise ExportError(
                "diagnostic-assisted episodes cannot be exported for "
                "standard AL0 training (recorded control intervention)"
            )


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
    """Yield legacy right-arm-padded world-frame action chunks."""

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


def iter_m0_mobile_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield causal 50 Hz whole-body chunks for AL0 training."""

    benchmark = config or BenchmarkConfig.v1()
    if benchmark.control_hz != 50:
        raise ExportError("AL0 legacy profile requires a 50 Hz control stream")
    episode_path, episode, task, source_result = _episode_context(
        episode_directory
    )
    _reject_m0_diagnostic_assist(episode)
    task_metadata = _mapping(
        task.get("metadata"), "manifest.episode.task.metadata"
    )
    object_curriculum_split = task_metadata.get("curriculum_split")
    if object_curriculum_split not in {"train", "val", "unseen"}:
        raise ExportError(
            "AL0 curriculum_split must be explicitly train, val, or unseen"
        )
    split = object_curriculum_split
    if task.get("task_type") == "stationary_sort":
        if object_curriculum_split != "train":
            raise ExportError(
                "stationary_sort object curriculum_split must be train"
            )
        try:
            stationary_scenario = validate_stationary_episode_contract(episode)
        except ValueError as error:
            raise ExportError(
                "stationary_sort does not match the registered diagnostic "
                f"contract: {error}"
            ) from error
        split = stationary_scenario.split
    robot_mode = task.get("robot_mode", "unspecified")
    if not isinstance(robot_mode, str) or not robot_mode:
        raise ExportError("AL0 robot_mode must be a non-empty string")
    raw_belt_speed = task.get("belt_speed_mps")
    belt_speed_mps = (
        _number(raw_belt_speed, "task.belt_speed_mps")
        if raw_belt_speed is not None
        else None
    )
    model_ticks = load_model_tick_steps(episode_path)
    _validate_training_camera_coverage(episode_path, model_ticks)
    policy_camera_ids = _camera_ids_for_role(episode, "policy_observation")
    required_camera_ids = tuple(sorted(_REQUIRED_POLICY_CAMERA_IDS))
    if policy_camera_ids != required_camera_ids:
        raise ExportError(
            "AL0 policy cameras must be exactly "
            f"{required_camera_ids}; got {policy_camera_ids}"
        )

    control_steps = _load_control_steps(episode_path, benchmark)
    _reject_m0_diagnostic_assist(episode, control_steps)
    horizon = M0_MOBILE_ACTION_HORIZON
    for source_index, source in enumerate(control_steps):
        frames = tuple(
            sorted(
                _select_camera_frames(source, policy_camera_ids),
                key=lambda frame: _mapping(frame, "camera frame").get(
                    "camera_id"
                ),
            )
        )
        recorded_camera_ids = tuple(
            sorted(_mapping(frame, "camera frame").get("camera_id") for frame in frames)
        )
        if recorded_camera_ids != required_camera_ids:
            continue
        future_steps = control_steps[source_index + 1 : source_index + 1 + horizon]
        if len(future_steps) != horizon:
            continue
        canonical_chunk = tuple(_canonical_action(step) for step in future_steps)
        model_chunk = tuple(
            canonical_to_m0_mobile_action(action) for action in canonical_chunk
        )
        observation_sim_step = _integer(source.get("sim_step"), "sim_step")
        observation_time_s = _number(source.get("sim_time_s"), "sim_time_s")
        yield {
            "schema_version": M0_MOBILE_SCHEMA_VERSION,
            "profile": "m0_mobile_v1",
            "source_episode_id": episode.get("episode_id"),
            "source_task_id": task.get("task_id"),
            "source_task_type": task.get("task_type"),
            "source_task_outcome": source_result.outcome,
            "source_failure_reason": source_result.failure_reason,
            "source_assisted": False,
            "split": split,
            "object_curriculum_split": object_curriculum_split,
            "robot_mode": robot_mode,
            "belt_speed_mps": belt_speed_mps,
            "source_steps_path": "steps.jsonl",
            "sample_id": (
                f"{episode.get('episode_id')}:sim-step-{observation_sim_step}"
            ),
            "instruction": task.get("instruction"),
            "model_tick": _integer(source.get("model_tick"), "model_tick"),
            "observation_sim_step": observation_sim_step,
            "observation_time_s": observation_time_s,
            "policy_camera_ids": required_camera_ids,
            "policy_camera_frames": frames,
            "observer_cameras_excluded": True,
            "state28": _m0_mobile_state28(source),
            "state_layout": M0_MOBILE_STATE_LAYOUT,
            "state_frame": "robot_root/body_and_base",
            "action_rate_hz": benchmark.control_hz,
            "action_horizon": horizon,
            "causal_offset_control_steps": 1,
            "label_control_sim_steps": tuple(
                _integer(step.get("sim_step"), "sim_step")
                for step in future_steps
            ),
            "canonical_action10_chunk": canonical_chunk,
            "model_action10_chunk": model_chunk,
            "model_gripper_convention": "0=close,1=open",
            "action_frame": "body_base_canonical",
            "action_dimension_mask": M0_MOBILE_ACTION_DIMENSION_MASK,
        }


def _mean_base_action_for_tick(
    tick: Mapping[str, Any],
    control_by_sim_step: Mapping[int, Mapping[str, Any]],
) -> tuple[float, float, float]:
    source_steps = _sequence(
        tick.get("_source_control_sim_steps"), "_source_control_sim_steps"
    )
    actions = []
    for raw_sim_step in source_steps:
        sim_step = _integer(raw_sim_step, "source control sim_step")
        try:
            actions.append(_canonical_action(control_by_sim_step[sim_step])[:3])
        except KeyError as error:
            raise ExportError(
                f"model tick references missing control sim_step {sim_step}"
            ) from error
    if not actions:
        raise ExportError("model tick has no control actions to average")
    return tuple(
        sum(action[index] for action in actions) / len(actions)
        for index in range(3)
    )


def _camera_clip(
    history_steps: Sequence[Mapping[str, Any]], camera_id: str
) -> dict[str, Any]:
    frames = []
    for step in history_steps:
        matching = tuple(
            frame
            for frame in _select_camera_frames(step, (camera_id,))
            if frame.get("camera_id") == camera_id
        )
        if len(matching) != 1:
            raise ExportError(
                f"temporal history requires one {camera_id} frame per tick"
            )
        frames.append(matching[0])
    return {
        "camera_id": camera_id,
        "history_offsets_model_ticks": HISTORY_OFFSETS_MODEL_TICKS,
        "frames": tuple(frames),
    }


def _joint_task_evidence(
    control_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless one episode follows the joint teacher contract."""

    phase_first_control_indices: dict[str, int] = {}
    cursor = -1
    for required_phase in JOINT_TASK_REQUIRED_PHASE_ORDER:
        match = next(
            (
                index
                for index in range(cursor + 1, len(control_steps))
                if control_steps[index].get("phase") == required_phase
            ),
            None,
        )
        if match is None:
            raise ExportError(
                "joint AL0 episode is missing ordered phase "
                f"{required_phase!r}"
            )
        phase_first_control_indices[required_phase] = match
        cursor = match

    def phase_displacement(phase: str) -> tuple[float, int]:
        rows = [step for step in control_steps if step.get("phase") == phase]
        if len(rows) < 2:
            return 0.0, len(rows)
        first, _ = _pose(rows[0], "robot_root_world")
        last, _ = _pose(rows[-1], "robot_root_world")
        return math.hypot(last[0] - first[0], last[1] - first[1]), len(rows)

    approach_displacement, approach_steps = phase_displacement(
        "mobile_approach"
    )
    backoff_displacement, backoff_steps = phase_displacement("carry_backoff")
    carry_displacement, carry_steps = phase_displacement("carry_navigate")
    if approach_displacement < JOINT_TASK_APPROACH_MIN_DISPLACEMENT_M:
        raise ExportError(
            "joint AL0 approach navigation displacement is too short: "
            f"{approach_displacement:.3f} m"
        )
    if backoff_displacement < JOINT_TASK_BACKOFF_MIN_DISPLACEMENT_M:
        raise ExportError(
            "joint AL0 post-grasp backoff displacement is too short: "
            f"{backoff_displacement:.3f} m"
        )
    if carry_displacement < JOINT_TASK_CARRY_MIN_DISPLACEMENT_M:
        raise ExportError(
            "joint AL0 loaded navigation displacement is too short: "
            f"{carry_displacement:.3f} m"
        )
    backoff_actions = tuple(
        _canonical_action(step)
        for step in control_steps
        if step.get("phase") == "carry_backoff"
    )
    if not backoff_actions or any(
        action[0] >= -0.16 or abs(action[1]) > 1.0e-9
        for action in backoff_actions
    ):
        raise ExportError(
            "joint AL0 carry_backoff must use a negative longitudinal "
            "command with zero lateral command"
        )
    placement_phases = {"carry", "preplace", "place_descend", "open"}
    placement_steps = tuple(
        step for step in control_steps if step.get("phase") in placement_phases
    )
    if any(
        any(abs(value) > 1.0e-9 for value in _canonical_action(step)[:3])
        for step in placement_steps
    ):
        raise ExportError(
            "joint AL0 base must remain locked throughout loaded placement"
        )
    placement_start, _ = _pose(placement_steps[0], "robot_root_world")
    placement_max_displacement = max(
        math.hypot(
            _pose(step, "robot_root_world")[0][0] - placement_start[0],
            _pose(step, "robot_root_world")[0][1] - placement_start[1],
        )
        for step in placement_steps
    )
    if placement_max_displacement > JOINT_TASK_PLACEMENT_MAX_DISPLACEMENT_M:
        raise ExportError(
            "joint AL0 base moved during loaded placement: "
            f"{placement_max_displacement:.3f} m"
        )
    return {
        "schema_version": "conveyor-vla-al0-joint-task-evidence-1",
        "required_phase_order": JOINT_TASK_REQUIRED_PHASE_ORDER,
        "phase_first_control_indices": phase_first_control_indices,
        "approach_conveyor": {
            "phase": "mobile_approach",
            "control_steps": approach_steps,
            "planar_displacement_m": approach_displacement,
            "minimum_planar_displacement_m": (
                JOINT_TASK_APPROACH_MIN_DISPLACEMENT_M
            ),
        },
        "back_away_from_conveyor": {
            "phase": "carry_backoff",
            "control_steps": backoff_steps,
            "planar_displacement_m": backoff_displacement,
            "minimum_planar_displacement_m": (
                JOINT_TASK_BACKOFF_MIN_DISPLACEMENT_M
            ),
            "command_direction": "negative_longitudinal",
        },
        "carry_to_sort_bin": {
            "phase": "carry_navigate",
            "control_steps": carry_steps,
            "planar_displacement_m": carry_displacement,
            "minimum_planar_displacement_m": (
                JOINT_TASK_CARRY_MIN_DISPLACEMENT_M
            ),
        },
        "placement_base_lock": {
            "phases": tuple(sorted(placement_phases)),
            "maximum_planar_displacement_m": placement_max_displacement,
            "allowed_planar_displacement_m": (
                JOINT_TASK_PLACEMENT_MAX_DISPLACEMENT_M
            ),
            "base_action": "zero",
        },
    }


def iter_conveyorvla_al0_temporal_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield joint navigation/manipulation observations and future targets."""

    benchmark = config or BenchmarkConfig.v1()
    if benchmark.control_hz != 50 or benchmark.model_hz != 25:
        raise ExportError(
            "AL0 temporal v3 requires 50 Hz control and 25 Hz model clocks"
        )
    episode_path, episode, task, source_result = _episode_context(
        episode_directory
    )
    _reject_m0_diagnostic_assist(episode)
    if source_result.outcome != "success":
        return

    task_metadata = _mapping(
        task.get("metadata"), "manifest.episode.task.metadata"
    )
    if task_metadata.get("active_object_count") != 1:
        raise ExportError(
            "AL0 temporal joint data requires exactly one active object"
        )
    if task.get("robot_mode") != "whole_body_policy":
        raise ExportError(
            "AL0 temporal joint data requires whole_body_policy episodes"
        )
    policy_camera_ids = _camera_ids_for_role(episode, "policy_observation")
    if policy_camera_ids != AL0_TEMPORAL_CAMERA_IDS:
        raise ExportError(
            f"AL0 temporal cameras must be {AL0_TEMPORAL_CAMERA_IDS}; "
            f"got {policy_camera_ids}"
        )

    model_ticks = _training_source_ticks(load_model_tick_steps(episode_path))
    by_model_tick = {
        _integer(step.get("model_tick"), "model_tick"): step
        for step in model_ticks
    }
    control_steps = _load_control_steps(episode_path, benchmark)
    _reject_m0_diagnostic_assist(episode, control_steps)
    joint_task_evidence = _joint_task_evidence(control_steps)
    control_by_sim_step = {
        _integer(step.get("sim_step"), "sim_step"): step for step in control_steps
    }
    control_index_by_sim_step = {
        _integer(step.get("sim_step"), "sim_step"): index
        for index, step in enumerate(control_steps)
    }
    belt_speed = _number(task.get("belt_speed_mps"), "task.belt_speed_mps")
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ExportError("joint AL0 task instruction must be a non-empty string")
    instruction = instruction.strip()

    for source in model_ticks:
        source_tick = _integer(source.get("model_tick"), "model_tick")
        source_phase = source.get("phase")
        if source_phase not in JOINT_TRAINING_PHASES:
            continue
        history_steps = tuple(
            by_model_tick.get(source_tick + offset)
            for offset in HISTORY_OFFSETS_MODEL_TICKS
        )
        future_steps = tuple(
            by_model_tick.get(source_tick + offset)
            for offset in range(1, AL0_TEMPORAL_ACTION_HORIZON + 1)
        )
        if any(step is None for step in history_steps + future_steps):
            continue
        history = tuple(step for step in history_steps if step is not None)
        future = tuple(step for step in future_steps if step is not None)
        if any(
            step.get("phase") not in JOINT_TRAINING_PHASES
            for step in future
        ):
            continue

        source_root_xyz, source_root_wxyz = _pose(source, "robot_root_world")
        source_tcp_xyz, source_tcp_wxyz = _pose(source, "tcp_base")
        action_chunk = []
        for target in future:
            future_root_xyz, future_root_wxyz = _pose(
                target, "robot_root_world"
            )
            future_tcp_xyz, future_tcp_wxyz = _pose(target, "tcp_base")
            base_action = _mean_base_action_for_tick(
                target, control_by_sim_step
            )
            tcp_target = relative_tcp_target(
                source_root_xyz,
                source_root_wxyz,
                source_tcp_xyz,
                source_tcp_wxyz,
                future_root_xyz,
                future_root_wxyz,
                future_tcp_xyz,
                future_tcp_wxyz,
            )
            _, _, gripper = _joint_vectors(target)
            action_chunk.append(base_action + tcp_target + (gripper,))

        observation_sim_step = _integer(source.get("sim_step"), "sim_step")
        try:
            observation_control_tick = control_index_by_sim_step[
                observation_sim_step
            ]
        except KeyError as error:
            raise ExportError(
                "temporal observation is absent from the control stream"
            ) from error
        yield {
            "schema_version": AL0_TEMPORAL_SCHEMA_VERSION,
            "profile": AL0_TEMPORAL_PROFILE,
            "source_episode_id": episode.get("episode_id"),
            "source_task_id": task.get("task_id"),
            "source_task_type": task.get("task_type"),
            "source_task_outcome": source_result.outcome,
            "source_failure_reason": source_result.failure_reason,
            "source_assisted": False,
            "source_steps_path": "steps.jsonl",
            "source_instruction": task.get("instruction"),
            "instruction": instruction,
            "policy_task_scope": POLICY_TASK_SCOPE,
            "joint_task_evidence": joint_task_evidence,
            "sample_id": (
                f"{episode.get('episode_id')}:model-tick-{source_tick}"
            ),
            "phase": source_phase,
            "belt_speed_mps": belt_speed,
            "observation_model_tick": source_tick,
            "observation_control_tick": observation_control_tick,
            "observation_sim_step": observation_sim_step,
            "observation_time_s": _number(
                source.get("sim_time_s"), "sim_time_s"
            ),
            "history_offsets_model_ticks": HISTORY_OFFSETS_MODEL_TICKS,
            "history_model_ticks": tuple(
                _integer(step.get("model_tick"), "history model_tick")
                for step in history
            ),
            "history_sim_times_s": tuple(
                _number(step.get("sim_time_s"), "history sim_time_s")
                for step in history
            ),
            "camera_clips": tuple(
                _camera_clip(history, camera_id)
                for camera_id in AL0_TEMPORAL_CAMERA_IDS
            ),
            "observer_cameras_excluded": True,
            "state28": _m0_mobile_state28(source),
            "state_layout": M0_MOBILE_STATE_LAYOUT,
            "state_frame": "robot_root/body_and_base",
            "object_state_is_model_input": False,
            "observation_reference": {
                "robot_root_world": {
                    "xyz": source_root_xyz,
                    "wxyz": source_root_wxyz,
                },
                "tcp_base": {
                    "xyz": source_tcp_xyz,
                    "wxyz": source_tcp_wxyz,
                },
                "model_input": False,
            },
            "action_rate_hz": AL0_TEMPORAL_ACTION_RATE_HZ,
            "control_rate_hz": benchmark.control_hz,
            "action_horizon": AL0_TEMPORAL_ACTION_HORIZON,
            "future_offsets_model_ticks": tuple(
                range(1, AL0_TEMPORAL_ACTION_HORIZON + 1)
            ),
            "future_model_ticks": tuple(
                _integer(step.get("model_tick"), "future model_tick")
                for step in future
            ),
            "future_target_control_ticks": tuple(
                control_index_by_sim_step[
                    _integer(step.get("sim_step"), "future sim_step")
                ]
                for step in future
            ),
            "model_action10_chunk": tuple(action_chunk),
            "action_semantics": (
                "future mean base velocity plus independent future TCP pose "
                "relative to observation root/TCP plus future realized "
                "gripper opening"
            ),
            "tcp_target_frame": "observation_robot_root",
            "model_gripper_convention": "0=close,1=open",
            "gripper_action_source": (
                "future_measured_joint_open_fraction"
            ),
            "action_dimension_mask": M0_MOBILE_ACTION_DIMENSION_MASK,
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


def export_m0_mobile_episode(
    episode_directory: str | Path,
    output_path: str | Path,
    config: BenchmarkConfig | None = None,
) -> ExportSummary:
    episode_path = Path(episode_directory)
    destination = Path(output_path)
    _guard_output_path(episode_path, destination)
    source_result = validate_episode_for_export(episode_path)
    count = _write_jsonl_atomic(
        iter_m0_mobile_records(episode_path, config), destination
    )
    return ExportSummary(
        "m0_mobile_v1",
        episode_path,
        destination,
        count,
        source_result.outcome,
        source_result.failure_reason,
    )


def export_conveyorvla_al0_temporal_episode(
    episode_directory: str | Path,
    output_path: str | Path,
    config: BenchmarkConfig | None = None,
) -> ExportSummary:
    episode_path = Path(episode_directory)
    destination = Path(output_path)
    _guard_output_path(episode_path, destination)
    source_result = validate_episode_for_export(episode_path)
    count = _write_jsonl_atomic(
        iter_conveyorvla_al0_temporal_records(episode_path, config),
        destination,
    )
    if source_result.outcome == "success" and count == 0:
        raise ExportError("successful episode produced no AL0 temporal joint records")
    return ExportSummary(
        AL0_TEMPORAL_PROFILE,
        episode_path,
        destination,
        count,
        source_result.outcome,
        source_result.failure_reason,
    )


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "M0_MOBILE_ACTION_DIMENSION_MASK",
    "M0_MOBILE_ACTION_HORIZON",
    "M0_MOBILE_SCHEMA_VERSION",
    "M0_MOBILE_STATE_LAYOUT",
    "AL0_TEMPORAL_PROFILE",
    "AL0_TEMPORAL_SCHEMA_VERSION",
    "ExportError",
    "ExportSummary",
    "SourceTaskResult",
    "export_dynamicvla_episode",
    "export_m0_episode",
    "export_m0_mobile_episode",
    "export_conveyorvla_al0_temporal_episode",
    "canonical_to_m0_mobile_action",
    "iter_dynamicvla_records",
    "iter_m0_records",
    "iter_m0_mobile_records",
    "iter_conveyorvla_al0_temporal_records",
    "load_model_tick_steps",
    "m0_mobile_to_canonical_action",
    "validate_episode_for_export",
]
