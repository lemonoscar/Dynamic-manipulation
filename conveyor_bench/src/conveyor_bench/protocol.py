"""Serializable task, episode, sample, and event contracts."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import uuid4

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


class TaskType(str, Enum):
    C0_STATIC_PICK = "c0_static_pick"
    C1_DYNAMIC_PICK = "c1_dynamic_pick"


class EpisodeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class FailureReason(str, Enum):
    NONE = "none"
    NO_SAMPLES = "no_samples"
    INVALID_TASK_CONFIGURATION = "invalid_task_configuration"
    ROBOT_FALLEN = "robot_fallen"
    FORBIDDEN_COLLISION = "forbidden_collision"
    WRONG_OBJECT = "wrong_object"
    TARGET_MISSED = "target_missed"
    DROPPED = "dropped"
    TIMEOUT = "timeout"
    GRASP_NOT_SECURED = "grasp_not_secured"
    ABORTED = "aborted"
    RUNTIME_ERROR = "runtime_error"
    RECORDER_ERROR = "recorder_error"


class EventKind(str, Enum):
    EPISODE_START = "episode_start"
    OBJECT_SPAWNED = "object_spawned"
    PHASE_CHANGED = "phase_changed"
    CAMERA_FRAME = "camera_frame"
    GRIPPER_CLOSED = "gripper_closed"
    TARGET_LIFTED = "target_lifted"
    GRASP_VERIFIED = "grasp_verified"
    TARGET_CROSSED_EXIT = "target_crossed_exit"
    FAILURE = "failure"
    EPISODE_END = "episode_end"


def _finite_nonnegative(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_vec3(value: Sequence[float], name: str) -> None:
    if len(value) != 3:
        raise ValueError(f"{name} must have exactly three elements")
    if any(not math.isfinite(float(component)) for component in value):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class TimingTrace:
    """Optional timestamps for observation-to-execution latency analysis."""

    observation_capture_s: float | None = None
    inference_start_s: float | None = None
    inference_end_s: float | None = None
    action_enqueue_s: float | None = None
    action_execute_start_s: float | None = None
    action_execute_end_s: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            _finite_nonnegative(value, name)
        ordered_pairs = (
            ("inference_start_s", self.inference_start_s, "inference_end_s", self.inference_end_s),
            (
                "action_execute_start_s",
                self.action_execute_start_s,
                "action_execute_end_s",
                self.action_execute_end_s,
            ),
        )
        for start_name, start, end_name, end in ordered_pairs:
            if start is not None and end is not None and end < start:
                raise ValueError(f"{end_name} cannot precede {start_name}")


@dataclass(frozen=True)
class TaskManifest:
    """A frozen task instance; random choices are resolved before execution."""

    task_id: str
    task_type: TaskType
    instruction: str
    target_object_id: str
    object_ids: tuple[str, ...]
    seed: int
    belt_speed_mps: float
    belt_surface_z_m: float
    # ``exit_x_m`` is retained only so development datasets produced before
    # the transverse-belt layout remain readable.  New tasks describe an
    # oriented exit plane explicitly.
    exit_x_m: float | None = None
    transport_direction_xyz: Vec3 | None = None
    exit_plane_point_xyz: Vec3 | None = None
    max_duration_s: float = 20.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")
        if not self.target_object_id:
            raise ValueError("target_object_id cannot be empty")
        if self.target_object_id not in self.object_ids:
            raise ValueError("target_object_id must be present in object_ids")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("object_ids must be unique")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        for name in ("belt_speed_mps", "belt_surface_z_m", "max_duration_s"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.exit_x_m is not None and not math.isfinite(self.exit_x_m):
            raise ValueError("exit_x_m must be finite")
        has_direction = self.transport_direction_xyz is not None
        has_exit_point = self.exit_plane_point_xyz is not None
        if has_direction != has_exit_point:
            raise ValueError(
                "transport_direction_xyz and exit_plane_point_xyz must be provided together"
            )
        if not has_direction:
            if self.exit_x_m is None:
                raise ValueError(
                    "task requires oriented exit geometry or legacy exit_x_m"
                )
        else:
            assert self.transport_direction_xyz is not None
            assert self.exit_plane_point_xyz is not None
            _validate_vec3(
                self.transport_direction_xyz,
                "transport_direction_xyz",
            )
            _validate_vec3(self.exit_plane_point_xyz, "exit_plane_point_xyz")
            norm = math.sqrt(
                sum(float(component) ** 2 for component in self.transport_direction_xyz)
            )
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
                raise ValueError("transport_direction_xyz must be a unit vector")
            if self.exit_x_m is not None:
                legacy_equivalent = (
                    all(
                        math.isclose(
                            float(actual),
                            float(expected),
                            rel_tol=0.0,
                            abs_tol=1.0e-9,
                        )
                        for actual, expected in zip(
                            self.transport_direction_xyz,
                            (1.0, 0.0, 0.0),
                            strict=True,
                        )
                    )
                    and math.isclose(
                        float(self.exit_plane_point_xyz[0]),
                        self.exit_x_m,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                )
                if not legacy_equivalent:
                    raise ValueError(
                        "exit_x_m conflicts with the oriented exit geometry"
                    )
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")

    @property
    def resolved_transport_direction_xyz(self) -> Vec3:
        """Return the forward unit vector, including the legacy +X fallback."""

        if self.transport_direction_xyz is None:
            return (1.0, 0.0, 0.0)
        return tuple(
            float(component) for component in self.transport_direction_xyz
        )

    @property
    def resolved_exit_plane_point_xyz(self) -> Vec3:
        """Return a point on the exit plane, including the legacy fallback."""

        if self.exit_plane_point_xyz is None:
            assert self.exit_x_m is not None
            return (float(self.exit_x_m), 0.0, 0.0)
        return tuple(float(component) for component in self.exit_plane_point_xyz)

    def transport_progress(self, xyz: Sequence[float]) -> float:
        """Project a world-space point/vector onto the forward transport axis."""

        _validate_vec3(xyz, "xyz")
        return sum(
            float(component) * direction
            for component, direction in zip(
                xyz,
                self.resolved_transport_direction_xyz,
                strict=True,
            )
        )

    def forward_speed(self, velocity_xyz: Sequence[float]) -> float:
        """Return signed speed along the task's positive transport direction."""

        return self.transport_progress(velocity_xyz)

    def remaining_distance_to_exit(self, xyz: Sequence[float]) -> float:
        """Return signed forward distance from ``xyz`` to the exit plane."""

        point = self.resolved_exit_plane_point_xyz
        direction = self.resolved_transport_direction_xyz
        return sum(
            (exit_component - float(component)) * axis_component
            for component, exit_component, axis_component in zip(
                xyz,
                point,
                direction,
                strict=True,
            )
        )

    def has_crossed_exit(self, xyz: Sequence[float]) -> bool:
        """Whether a point lies on or beyond the oriented exit plane."""

        return self.remaining_distance_to_exit(xyz) <= 0.0


_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def make_run_id() -> str:
    """Return a sortable run identifier that remains unique within one second."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{timestamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class EpisodeManifest:
    """Identity and provenance for one attempted episode."""

    episode_id: str
    run_id: str
    protocol_version: str
    task: TaskManifest
    created_at_utc: str
    env_id: int = 0
    git_commit: str | None = None
    asset_hashes: Mapping[str, str] = field(default_factory=dict)
    seeds: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SAFE_EPISODE_ID.fullmatch(self.episode_id) or self.episode_id in {".", ".."}:
            raise ValueError("episode_id contains unsafe characters")
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if not self.protocol_version:
            raise ValueError("protocol_version cannot be empty")
        if self.env_id < 0:
            raise ValueError("env_id cannot be negative")


@dataclass(frozen=True)
class StepSample:
    """One control-rate state/action sample used by recording and evaluation."""

    sim_step: int
    sim_time_s: float
    env_id: int
    object_xyz: Vec3
    object_linear_velocity: Vec3
    tcp_xyz: Vec3
    belt_command_speed_mps: float
    belt_measured_speed_mps: float
    gripper_closed: bool
    left_contact: bool
    right_contact: bool
    target_in_gripper: bool
    target_crossed_exit: bool
    robot_fallen: bool
    forbidden_collision: bool
    phase: str
    action: Mapping[str, Any]
    timing: TimingTrace = field(default_factory=TimingTrace)
    wrong_object_grasped: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tcp_wxyz: Vec4 = (1.0, 0.0, 0.0, 0.0)
    joint_positions: tuple[float, ...] = ()
    joint_velocities: tuple[float, ...] = ()
    camera_frame_index: int | None = None

    def __post_init__(self) -> None:
        if self.sim_step < 0:
            raise ValueError("sim_step cannot be negative")
        _finite_nonnegative(self.sim_time_s, "sim_time_s")
        if self.env_id < 0:
            raise ValueError("env_id cannot be negative")
        _validate_vec3(self.object_xyz, "object_xyz")
        _validate_vec3(self.object_linear_velocity, "object_linear_velocity")
        _validate_vec3(self.tcp_xyz, "tcp_xyz")
        if len(self.tcp_wxyz) != 4 or any(
            not math.isfinite(float(component)) for component in self.tcp_wxyz
        ):
            raise ValueError("tcp_wxyz must contain four finite values")
        if len(self.joint_positions) != len(self.joint_velocities):
            raise ValueError(
                "joint_positions and joint_velocities must have equal length"
            )
        if any(
            not math.isfinite(float(component))
            for values in (self.joint_positions, self.joint_velocities)
            for component in values
        ):
            raise ValueError("joint state must contain only finite values")
        if self.camera_frame_index is not None and self.camera_frame_index < 0:
            raise ValueError("camera_frame_index cannot be negative")
        for name in ("belt_command_speed_mps", "belt_measured_speed_mps"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.phase:
            raise ValueError("phase cannot be empty")


@dataclass(frozen=True)
class Event:
    kind: EventKind
    time_s: float
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite_nonnegative(self.time_s, "time_s")


def to_jsonable(value: Any) -> Any:
    """Convert protocol values to objects accepted by ``json.dump``."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value
