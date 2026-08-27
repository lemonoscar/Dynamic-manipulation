"""Atomic raw-episode recording for fresh joint-trajectory data.

This recorder publishes only raw evidence.  It never creates a formal dataset
manifest, normalizer, or training row; the immutable materializer remains the
only path from successful raw episodes to model data.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from conveyor_bench.conveyorvla.joint_trajectory import (
    HISTORY_SPAN_S,
    JointTrajectoryRoute,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import (
    CLOCK_ABS_TOLERANCE_S,
    CONTROL_STRIDE_S,
    PROGRESS_PROVENANCE,
    validate_applied_control_sample,
)
from conveyor_bench.conveyorvla.joint_trajectory_system import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    JointControlTick,
    measured_named_joint_state,
)


RAW_EPISODE_SCHEMA_VERSION = "conveyorvla-joint-trajectory-raw-episode-v1"
QUERY_STRIDE_S = 0.20
QUERY_STRIDE_TICKS = round(QUERY_STRIDE_S / CONTROL_STRIDE_S)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CameraAsset:
    stream: str
    frame_id: str
    timestamp_s: float
    relative_path: str


def applied_control_sample_from_isaac(
    event: JointControlTick,
    *,
    control_tick_id: int,
    model_tick_id: int,
    arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
    gripper_joint_names: Sequence[str] = GRIPPER_JOINT_NAMES,
    gripper_closed_position: float = 0.0,
    gripper_open_position: float = 0.04,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one proven applied-command row from an Isaac post-step report."""

    if event.route is None:
        raise ValueError("raw control evidence requires a committed route")
    state = event.state_after
    before_metadata = getattr(event.state_before, "metadata", {})
    metadata = getattr(state, "metadata", {})
    if not isinstance(before_metadata, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Isaac state metadata must be mappings")
    arm_report = metadata.get("last_arm_joint_position_target_report")
    gripper_report = metadata.get("last_gripper_joint_position_target_report")
    if not isinstance(arm_report, Mapping) or arm_report.get("applied") is not True:
        raise ValueError("Isaac did not prove an applied arm position target")
    if not isinstance(gripper_report, Mapping) or gripper_report.get("applied") is not True:
        raise ValueError("Isaac did not prove an applied gripper position target")
    _fresh_apply_count(
        before_metadata,
        metadata,
        "arm_joint_position_target_apply_count",
    )
    _fresh_apply_count(
        before_metadata,
        metadata,
        "gripper_joint_position_target_apply_count",
    )
    action = event.action
    action_metadata = getattr(action, "metadata", {})
    if not isinstance(action_metadata, Mapping):
        raise ValueError("RobotAction metadata must be a mapping")
    requested_q = _finite_vector(
        getattr(action, "arm_joint_positions", ()), 6, "requested arm target"
    )
    applied_q = _finite_vector(
        arm_report.get("target_positions", ()), 6, "applied arm target"
    )
    measured = measured_named_joint_state(
        state,
        arm_joint_names=arm_joint_names,
        gripper_joint_names=gripper_joint_names,
        gripper_closed_position=gripper_closed_position,
        gripper_open_position=gripper_open_position,
    )
    requested_gripper = _unit_fraction(
        action_metadata.get("gripper_open_fraction_requested"),
        "requested gripper open fraction",
    )
    physical_gripper = _finite_vector(
        gripper_report.get("target_positions", ()),
        1,
        "applied gripper target",
    )[0]
    span = float(gripper_open_position) - float(gripper_closed_position)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("gripper physical range must be finite and positive")
    applied_gripper = _unit_fraction(
        (physical_gripper - gripper_closed_position) / span,
        "applied gripper open fraction",
    )
    requested_base = _finite_vector(
        getattr(action, "base_velocity", ()), 3, "requested base command"
    )
    applied_base = tuple(
        _finite_scalar(metadata.get(key), key)
        for key in ("command_seen_vx", "command_seen_vy", "command_seen_wz")
    )
    sample = {
        "tick_id": _nonnegative_integer(control_tick_id, "control_tick_id"),
        "sim_step": _nonnegative_integer(state.step_index, "sim_step"),
        "model_tick": _nonnegative_integer(model_tick_id, "model_tick_id"),
        "timestamp_s": _finite_nonnegative(state.timestamp, "timestamp_s"),
        "q_measured": list(measured.joint_position),
        "dq_measured": list(measured.joint_velocity),
        "gripper_measured": measured.gripper_open_fraction,
        "q_command_requested": list(requested_q),
        "q_command_applied": list(applied_q),
        "gripper_command_requested": requested_gripper,
        "gripper_command_applied": applied_gripper,
        "base_command_requested": list(requested_base),
        "base_command_applied": list(applied_base),
        "base_pose_world": list(
            _finite_vector(state.robot_root_pose, 7, "base_pose_world")
        ),
        "base_twist_world": list(
            _finite_vector(state.robot_root_velocity, 6, "base_twist_world")
        ),
        "route": event.route.value,
        "q_command_source": "controller_applied_after_saturation",
        "arm_target_apply_count": int(
            metadata["arm_joint_position_target_apply_count"]
        ),
        "gripper_target_apply_count": int(
            metadata["gripper_joint_position_target_apply_count"]
        ),
    }
    if extra:
        overlap = set(sample).intersection(extra)
        if overlap:
            raise ValueError(f"extra control evidence overwrites contract fields: {sorted(overlap)}")
        sample.update(dict(extra))
    validate_applied_control_sample(sample)
    _finite_vector(sample["base_command_requested"], 3, "base_command_requested")
    return sample


class FreshJointTrajectoryEpisodeRecorder:
    """Write one immutable raw episode through a staging directory."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        episode_id: str,
        split: str,
        episode_metadata: Mapping[str, Any],
        jpeg_quality: int = 90,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve()
        self.episode_id = _safe_component(episode_id, "episode_id")
        if split not in {"train", "val", "test"}:
            raise ValueError("raw episode split must be train, val, or test")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("JPEG quality must be within [1,100]")
        self.split = split
        self.jpeg_quality = int(jpeg_quality)
        self.episode_metadata = dict(episode_metadata)
        self.final_path = self.output_root / self.episode_id
        if self.final_path.exists():
            raise FileExistsError(f"raw episode already exists: {self.final_path}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.output_root / (
            f".{self.episode_id}.{uuid.uuid4().hex}.staging"
        )
        self.staging_path.mkdir()
        self._control_stream = (self.staging_path / "joint_commands_50hz.jsonl").open(
            "x", encoding="utf-8"
        )
        self._query_stream = (self.staging_path / "joint_queries_5hz.jsonl").open(
            "x", encoding="utf-8"
        )
        self._control_count = 0
        self._query_count = 0
        self._last_control_tick: int | None = None
        self._last_control_timestamp: float | None = None
        self._last_query_tick: int | None = None
        self._controls: dict[int, tuple[float, JointTrajectoryRoute]] = {}
        self._finalized = False

    def save_camera_frame(
        self,
        stream: str,
        *,
        frame_id: str | int,
        timestamp_s: float,
        image: Any,
    ) -> CameraAsset:
        self._require_open()
        if stream not in {"head", "wrist", "overview"}:
            raise ValueError("camera stream must be head, wrist, or overview")
        frame = _safe_component(str(frame_id), "frame_id")
        timestamp = _finite_nonnegative(timestamp_s, "camera timestamp")
        directory = self.staging_path / "images" / stream
        directory.mkdir(parents=True, exist_ok=True)
        relative = Path("images") / stream / f"{frame}.jpg"
        path = self.staging_path / relative
        if path.exists():
            raise FileExistsError(f"camera frame already exists: {relative}")
        _save_jpeg_atomic(path, image, self.jpeg_quality)
        return CameraAsset(stream, frame, timestamp, relative.as_posix())

    def record_control(self, sample: Mapping[str, Any]) -> None:
        self._require_open()
        row = dict(sample)
        validate_applied_control_sample(row)
        _finite_vector(row.get("base_command_requested", ()), 3, "base_command_requested")
        tick = _nonnegative_integer(row["tick_id"], "tick_id")
        timestamp = _finite_nonnegative(row["timestamp_s"], "timestamp_s")
        route = JointTrajectoryRoute(str(row["route"]))
        if self._last_control_tick is not None:
            if tick != self._last_control_tick + 1:
                raise ValueError("50 Hz control ticks must be contiguous")
            if not math.isclose(
                timestamp - float(self._last_control_timestamp),
                CONTROL_STRIDE_S,
                rel_tol=0.0,
                abs_tol=CLOCK_ABS_TOLERANCE_S,
            ):
                raise ValueError("control timestamps must remain exactly 50 Hz")
        self._write_jsonl(self._control_stream, row)
        self._last_control_tick = tick
        self._last_control_timestamp = timestamp
        self._controls[tick] = (timestamp, route)
        self._control_count += 1

    def record_query(self, query: Mapping[str, Any]) -> None:
        self._require_open()
        row = dict(query)
        if str(row.get("episode_id", "")) != self.episode_id:
            raise ValueError("query episode_id does not match its recorder")
        if str(row.get("split", "")) != self.split:
            raise ValueError("query split does not match its recorder")
        _safe_component(str(row.get("sample_id", "")), "sample_id")
        if not str(row.get("global_instruction", "")).strip():
            raise ValueError("query global_instruction must be non-empty")
        route = JointTrajectoryRoute(str(row.get("route", "")))
        tick = _nonnegative_integer(row.get("control_tick_id"), "control_tick_id")
        if tick not in self._controls:
            raise ValueError("query control tick has not been recorded")
        timestamp, control_route = self._controls[tick]
        if route is not control_route:
            raise ValueError("query route does not match its control tick")
        if self._last_query_tick is not None and tick - self._last_query_tick != QUERY_STRIDE_TICKS:
            raise ValueError("query anchors must remain exactly 5 Hz")
        history = _finite_vector(
            row.get("history_timestamps_s", ()), 2, "history_timestamps_s"
        )
        if not (
            math.isclose(history[1], timestamp, rel_tol=0.0, abs_tol=CLOCK_ABS_TOLERANCE_S)
            and math.isclose(
                history[1] - history[0],
                HISTORY_SPAN_S,
                rel_tol=0.0,
                abs_tol=CLOCK_ABS_TOLERANCE_S,
            )
        ):
            raise ValueError("query images must bind exact [t-0.20,t] timestamps")
        for key, stream in (("head_images", "head"), ("wrist_images", "wrist")):
            paths = row.get(key)
            if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence) or len(paths) != 2:
                raise ValueError(f"{key} must contain exact [t-0.20,t] assets")
            for value in paths:
                self._require_asset(str(value), stream)
        overview = row.get("overview_images")
        if overview is not None:
            if isinstance(overview, (str, bytes)) or not isinstance(overview, Sequence):
                raise ValueError("overview_images must be a sequence")
            for value in overview:
                self._require_asset(str(value), "overview")
        progress_valid = bool(row.get("physical_progress_valid"))
        if progress_valid:
            progress = _finite_scalar(row.get("physical_progress"), "physical_progress")
            if not 0.0 <= progress <= 1.0:
                raise ValueError("physical_progress must be within [0,1]")
            if row.get("physical_progress_provenance") not in PROGRESS_PROVENANCE:
                raise ValueError("physical progress requires route-specific provenance")
        elif row.get("physical_progress") is not None:
            raise ValueError("invalid physical progress must be null")
        self._write_jsonl(self._query_stream, row)
        self._last_query_tick = tick
        self._query_count += 1

    def finalize(
        self,
        *,
        success: bool,
        outcome_metadata: Mapping[str, Any],
    ) -> Path:
        self._require_open()
        if success and (self._control_count == 0 or self._query_count == 0):
            raise ValueError("successful raw episode requires control and query evidence")
        self._flush_and_close(self._control_stream)
        self._flush_and_close(self._query_stream)
        summary = {
            "schema_version": RAW_EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "split": self.split,
            "success": bool(success),
            "control_row_count": self._control_count,
            "query_row_count": self._query_count,
            "episode_metadata": self.episode_metadata,
            "outcome_metadata": dict(outcome_metadata),
        }
        summary_path = self.staging_path / "summary.json"
        with summary_path.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if self.final_path.exists():
            raise FileExistsError(f"raw episode appeared during recording: {self.final_path}")
        os.replace(self.staging_path, self.final_path)
        directory_fd = os.open(self.output_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._finalized = True
        return self.final_path

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _require_asset(self, relative_value: str, expected_stream: str) -> None:
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("query image paths must remain episode-relative")
        expected_prefix = ("images", expected_stream)
        if relative.parts[:2] != expected_prefix:
            raise ValueError(f"query asset is not from the {expected_stream} stream")
        path = (self.staging_path / relative).resolve()
        if self.staging_path not in path.parents or not path.is_file():
            raise ValueError(f"query asset is missing: {relative_value}")

    def _require_open(self) -> None:
        if self._finalized or self._control_stream.closed or self._query_stream.closed:
            raise RuntimeError("raw episode recorder is already closed")

    @staticmethod
    def _write_jsonl(stream: Any, row: Mapping[str, Any]) -> None:
        json.dump(dict(row), stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")

    @staticmethod
    def _flush_and_close(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()


def _save_jpeg_atomic(path: Path, image: Any, quality: int) -> None:
    import numpy as np
    from PIL import Image

    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"camera image must be [H,W,3|4], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    Image.fromarray(array).convert("RGB").save(
        temporary, format="JPEG", quality=quality
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _fresh_apply_count(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> None:
    previous = int(before.get(key, 0))
    current = int(after.get(key, 0))
    if current <= previous:
        raise ValueError(f"Isaac {key} did not advance on this control tick")


def _safe_component(value: str, name: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"{name} must be a safe non-empty path component")
    return text


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _unit_fraction(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return result


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


__all__ = [
    "CameraAsset",
    "FreshJointTrajectoryEpisodeRecorder",
    "RAW_EPISODE_SCHEMA_VERSION",
    "applied_control_sample_from_isaac",
]
