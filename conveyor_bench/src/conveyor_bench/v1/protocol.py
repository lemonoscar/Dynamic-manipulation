"""Serializable identities, states, actions, labels, and events for V1."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from numbers import Real
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from uuid import uuid4

from .config import PROTOCOL_VERSION

if TYPE_CHECKING:
    from .config import BenchmarkConfig

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
Action10 = tuple[float, float, float, float, float, float, float, float, float, float]

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


class TaskType(str, Enum):
    STATIONARY_SORT = "stationary_sort"
    DYNAMIC_SORT = "dynamic_sort"
    CONTINUOUS_SORT = "continuous_sort"


class RobotMode(str, Enum):
    FIXED_BASE = "fixed_base"
    MOBILE_KINEMATIC = "mobile_kinematic"
    WHOLE_BODY_POLICY = "whole_body_policy"


class ActionChunkProfile(str, Enum):
    M0 = "m0"
    DYNAMICVLA = "dynamicvla"


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
    WRONG_ZONE = "wrong_zone"
    PLACEMENT_NOT_SETTLED = "placement_not_settled"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    RUNTIME_ERROR = "runtime_error"
    RECORDER_ERROR = "recorder_error"


class EventKind(str, Enum):
    EPISODE_START = "episode_start"
    OBJECT_SPAWNED = "object_spawned"
    OBJECT_RECYCLED = "object_recycled"
    TARGET_SELECTED = "target_selected"
    PHASE_CHANGED = "phase_changed"
    CONTACT_BEGIN = "contact_begin"
    CONTACT_END = "contact_end"
    GRASP_ATTEMPT = "grasp_attempt"
    OBJECT_RELEASED = "object_released"
    OBJECT_PLACED = "object_placed"
    TARGET_MISSED = "target_missed"
    FAILURE = "failure"
    EPISODE_END = "episode_end"


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} contains unsafe or empty characters")


def _validate_finite_vector(
    value: Sequence[float], length: int, name: str
) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    if len(value) != length:
        raise ValueError(f"{name} must have exactly {length} elements")
    if any(
        isinstance(component, bool)
        or not isinstance(component, Real)
        or not math.isfinite(component)
        for component in value
    ):
        raise ValueError(f"{name} must contain only finite values")


def _validate_nonnegative_number(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _validate_unique(values: Sequence[str], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")


@dataclass(frozen=True)
class Pose:
    xyz: Vec3
    wxyz: Vec4

    def __post_init__(self) -> None:
        _validate_finite_vector(self.xyz, 3, "xyz")
        _validate_finite_vector(self.wxyz, 4, "wxyz")
        norm = math.sqrt(sum(float(component) ** 2 for component in self.wxyz))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError("wxyz must be a unit quaternion")


@dataclass(frozen=True)
class Twist:
    linear_xyz: Vec3
    angular_xyz: Vec3

    def __post_init__(self) -> None:
        _validate_finite_vector(self.linear_xyz, 3, "linear_xyz")
        _validate_finite_vector(self.angular_xyz, 3, "angular_xyz")


@dataclass(frozen=True)
class JointState:
    names: tuple[str, ...]
    positions: tuple[float, ...]
    velocities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("joint names cannot be empty")
        for name in self.names:
            _validate_id(name, "joint name")
        _validate_unique(self.names, "joint names")
        if len(self.positions) != len(self.names) or len(self.velocities) != len(
            self.names
        ):
            raise ValueError("joint names, positions, and velocities must have equal length")
        _validate_finite_vector(self.positions, len(self.names), "joint positions")
        _validate_finite_vector(self.velocities, len(self.names), "joint velocities")


@dataclass(frozen=True)
class CanonicalAction:
    """Canonical 10D command.

    Layout: base body-frame ``vx, vy, wz``; end-effector delta ``x, y, z``
    and rotation-vector delta ``rx, ry, rz`` in the robot-root/base frame;
    gripper command. Legacy world-pose deltas belong only in an exporter
    projection and must never be stored as this canonical action.
    """

    values: Action10

    def __post_init__(self) -> None:
        _validate_finite_vector(self.values, 10, "canonical action")
        if not -1.0 <= float(self.values[9]) <= 1.0:
            raise ValueError("gripper command must be within [-1, 1]")

    @property
    def base_body_twist(self) -> tuple[float, float, float]:
        return self.values[0:3]

    @property
    def ee_delta_xyz_base(self) -> tuple[float, float, float]:
        return self.values[3:6]

    @property
    def ee_delta_rotvec(self) -> tuple[float, float, float]:
        return self.values[6:9]

    @property
    def gripper(self) -> float:
        return self.values[9]

    def validate_for_robot_mode(self, robot_mode: RobotMode) -> None:
        if robot_mode is RobotMode.FIXED_BASE and any(
            not math.isclose(float(value), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            for value in self.base_body_twist
        ):
            raise ValueError("fixed_base actions must have zero base body twist")


@dataclass(frozen=True)
class GoalZone:
    zone_id: str
    min_xyz: Vec3
    max_xyz: Vec3

    def __post_init__(self) -> None:
        _validate_id(self.zone_id, "zone_id")
        _validate_finite_vector(self.min_xyz, 3, "min_xyz")
        _validate_finite_vector(self.max_xyz, 3, "max_xyz")
        if any(
            float(lower) >= float(upper)
            for lower, upper in zip(self.min_xyz, self.max_xyz, strict=True)
        ):
            raise ValueError("each goal-zone minimum must be less than its maximum")

    def contains(self, xyz: Sequence[float]) -> bool:
        _validate_finite_vector(xyz, 3, "xyz")
        return all(
            float(lower) <= float(value) <= float(upper)
            for value, lower, upper in zip(
                xyz, self.min_xyz, self.max_xyz, strict=True
            )
        )


@dataclass(frozen=True)
class ObjectInstance:
    instance_id: str
    asset_id: str
    class_id: str
    goal_zone_id: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.instance_id, "instance_id")
        _validate_id(self.asset_id, "asset_id")
        _validate_id(self.class_id, "class_id")
        if self.goal_zone_id is not None:
            _validate_id(self.goal_zone_id, "goal_zone_id")


@dataclass(frozen=True)
class TaskManifest:
    """A fully resolved dynamic-sort or continuous-sort task."""

    task_id: str
    task_type: TaskType
    robot_mode: RobotMode
    instruction: str
    objects: tuple[ObjectInstance, ...]
    goal_zones: tuple[GoalZone, ...]
    scored_object_ids: tuple[str, ...]
    seed: int
    belt_speed_mps: float
    belt_surface_z_m: float
    transport_direction_xyz: Vec3
    exit_plane_point_xyz: Vec3
    max_duration_s: float = 20.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_id(self.task_id, "task_id")
        if not isinstance(self.task_type, TaskType):
            raise ValueError("task_type must be a TaskType")
        if not isinstance(self.robot_mode, RobotMode):
            raise ValueError("robot_mode must be a RobotMode")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction cannot be empty")
        if not self.objects:
            raise ValueError("objects cannot be empty")
        if not self.goal_zones:
            raise ValueError("goal_zones cannot be empty")
        if not self.scored_object_ids:
            raise ValueError("scored_object_ids cannot be empty")
        for instance_id in self.scored_object_ids:
            _validate_id(instance_id, "scored_object_id")
        if any(not isinstance(obj, ObjectInstance) for obj in self.objects):
            raise ValueError("objects must contain only ObjectInstance values")
        if any(not isinstance(zone, GoalZone) for zone in self.goal_zones):
            raise ValueError("goal_zones must contain only GoalZone values")
        object_ids = tuple(obj.instance_id for obj in self.objects)
        zone_ids = tuple(zone.zone_id for zone in self.goal_zones)
        _validate_unique(object_ids, "object instance ids")
        _validate_unique(zone_ids, "goal zone ids")
        _validate_unique(self.scored_object_ids, "scored_object_ids")
        unknown_targets = set(self.scored_object_ids) - set(object_ids)
        if unknown_targets:
            raise ValueError(
                f"scored_object_ids contain unknown instances: {sorted(unknown_targets)}"
            )
        zone_id_set = set(zone_ids)
        for obj in self.objects:
            if obj.goal_zone_id is not None and obj.goal_zone_id not in zone_id_set:
                raise ValueError(
                    f"object {obj.instance_id!r} references an unknown goal zone"
                )
        object_by_id = {obj.instance_id: obj for obj in self.objects}
        for instance_id in self.scored_object_ids:
            if object_by_id[instance_id].goal_zone_id is None:
                raise ValueError(
                    f"scored object {instance_id!r} requires a goal_zone_id"
                )
        if (
            self.task_type
            in {TaskType.STATIONARY_SORT, TaskType.DYNAMIC_SORT}
            and len(self.scored_object_ids) != 1
        ):
            raise ValueError(
                f"{self.task_type.value} requires exactly one scored object"
            )
        if self.task_type is TaskType.STATIONARY_SORT and len(self.objects) != 1:
            raise ValueError("stationary_sort requires exactly one object")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("belt_speed_mps", "belt_surface_z_m", "max_duration_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.task_type is TaskType.STATIONARY_SORT:
            if not math.isclose(
                float(self.belt_speed_mps), 0.0, rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError("stationary_sort requires belt_speed_mps=0")
        elif self.belt_speed_mps <= 0:
            raise ValueError(
                "dynamic and continuous V1 sorting tasks require a positive "
                "belt_speed_mps"
            )
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        _validate_finite_vector(
            self.transport_direction_xyz, 3, "transport_direction_xyz"
        )
        _validate_finite_vector(
            self.exit_plane_point_xyz, 3, "exit_plane_point_xyz"
        )
        norm = math.sqrt(
            sum(float(component) ** 2 for component in self.transport_direction_xyz)
        )
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("transport_direction_xyz must be a unit vector")

    @property
    def object_by_id(self) -> dict[str, ObjectInstance]:
        return {obj.instance_id: obj for obj in self.objects}

    @property
    def goal_zone_by_id(self) -> dict[str, GoalZone]:
        return {zone.zone_id: zone for zone in self.goal_zones}


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run-{timestamp}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class EpisodeManifest:
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
        if (
            not isinstance(self.episode_id, str)
            or not _SAFE_EPISODE_ID.fullmatch(self.episode_id)
            or self.episode_id in {".", ".."}
        ):
            raise ValueError("episode_id contains unsafe characters")
        _validate_id(self.run_id, "run_id")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
        if not isinstance(self.task, TaskManifest):
            raise ValueError("task must be a TaskManifest")
        if isinstance(self.env_id, bool) or not isinstance(self.env_id, int) or self.env_id < 0:
            raise ValueError("env_id must be a non-negative integer")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc:
            raise ValueError("created_at_utc cannot be empty")
        for name, value in self.seeds.items():
            _validate_id(name, "seed name")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("all seeds must be non-negative integers")
        for component_id, digest in self.asset_hashes.items():
            _validate_id(component_id, "asset hash component_id")
            if not isinstance(digest, str) or not _SHA256_DIGEST.fullmatch(digest):
                raise ValueError(
                    "asset hashes must be 64 hexadecimal SHA-256 digests, "
                    "optionally prefixed by 'sha256:'"
                )


@dataclass(frozen=True)
class ObjectState:
    instance_id: str
    pose_world: Pose
    twist_world: Twist
    active: bool = True
    in_gripper: bool = False
    crossed_exit: bool = False

    def __post_init__(self) -> None:
        _validate_id(self.instance_id, "instance_id")
        if not isinstance(self.pose_world, Pose):
            raise ValueError("pose_world must be a Pose")
        if not isinstance(self.twist_world, Twist):
            raise ValueError("twist_world must be a Twist")
        _validate_bool(self.active, "active")
        _validate_bool(self.in_gripper, "in_gripper")
        _validate_bool(self.crossed_exit, "crossed_exit")
        if self.in_gripper and not self.active:
            raise ValueError("an inactive object cannot be in the gripper")


@dataclass(frozen=True)
class FutureObjectState:
    instance_id: str
    horizon_steps: int
    valid: bool
    pose_world: Pose | None
    twist_world: Twist | None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.instance_id, "instance_id")
        if (
            isinstance(self.horizon_steps, bool)
            or not isinstance(self.horizon_steps, int)
            or self.horizon_steps < 0
        ):
            raise ValueError("horizon_steps must be a non-negative integer")
        _validate_bool(self.valid, "valid")
        if self.valid:
            if not isinstance(self.pose_world, Pose) or not isinstance(
                self.twist_world, Twist
            ):
                raise ValueError("valid future labels require pose_world and twist_world")
            if self.invalid_reason is not None:
                raise ValueError("valid future labels cannot have invalid_reason")
        else:
            if self.pose_world is not None or self.twist_world is not None:
                raise ValueError("invalid future labels cannot carry state values")
            if not isinstance(self.invalid_reason, str) or not self.invalid_reason.strip():
                raise ValueError("invalid future labels require invalid_reason")


@dataclass(frozen=True)
class CameraFrameRef:
    camera_id: str
    frame_index: int
    capture_time_s: float
    relative_path: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.camera_id, "camera_id")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        _validate_nonnegative_number(self.capture_time_s, "capture_time_s")
        if self.relative_path is not None:
            path = PurePosixPath(self.relative_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("relative_path must be a safe relative path")


@dataclass(frozen=True)
class ActionChunkTrace:
    """Final accounting for one model-rate action chunk.

    All window bounds are half-open model-tick intervals.
    """

    chunk_id: str
    profile: ActionChunkProfile
    source_observation_tick: int
    source_observation_time_s: float
    valid_from_tick: int
    valid_until_tick: int
    execute_from_tick: int | None
    execute_until_tick: int | None
    actions: tuple[CanonicalAction, ...]
    stale: bool = False
    discarded_action_count: int = 0
    discard_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.chunk_id, "chunk_id")
        if not isinstance(self.profile, ActionChunkProfile):
            raise ValueError("profile must be an ActionChunkProfile")
        for name in (
            "source_observation_tick",
            "valid_from_tick",
            "valid_until_tick",
            "discarded_action_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _validate_nonnegative_number(
            self.source_observation_time_s, "source_observation_time_s"
        )
        if not self.actions:
            raise ValueError("actions cannot be empty")
        if any(not isinstance(action, CanonicalAction) for action in self.actions):
            raise ValueError("actions must contain only CanonicalAction values")
        _validate_bool(self.stale, "stale")
        if self.source_observation_tick > self.valid_from_tick:
            raise ValueError("source observation cannot follow the valid window")
        if self.valid_until_tick <= self.valid_from_tick:
            raise ValueError("valid action window must be non-empty")
        if self.valid_until_tick - self.valid_from_tick != len(self.actions):
            raise ValueError("valid action window length must equal actions length")

        has_execute_start = self.execute_from_tick is not None
        has_execute_end = self.execute_until_tick is not None
        if has_execute_start != has_execute_end:
            raise ValueError("execute window bounds must be provided together")
        executed_count = 0
        if has_execute_start:
            assert self.execute_from_tick is not None
            assert self.execute_until_tick is not None
            for name, value in (
                ("execute_from_tick", self.execute_from_tick),
                ("execute_until_tick", self.execute_until_tick),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
            if not (
                self.valid_from_tick
                <= self.execute_from_tick
                < self.execute_until_tick
                <= self.valid_until_tick
            ):
                raise ValueError("execute window must be inside the valid window")
            executed_count = self.execute_until_tick - self.execute_from_tick

        if executed_count + self.discarded_action_count != len(self.actions):
            raise ValueError("executed and discarded actions must account for the chunk")
        if self.discarded_action_count > 0:
            if not isinstance(self.discard_reason, str) or not self.discard_reason.strip():
                raise ValueError("discarded actions require discard_reason")
        elif self.discard_reason is not None:
            raise ValueError("discard_reason requires discarded actions")
        if self.stale and self.discarded_action_count == 0:
            raise ValueError("a stale chunk must discard at least one action")

    def validate_against(self, config: "BenchmarkConfig") -> None:
        expected = config.chunk_size_for(self.profile.value)
        if len(self.actions) != expected:
            raise ValueError(
                f"{self.profile.value} chunks require {expected} actions, got {len(self.actions)}"
            )


@dataclass(frozen=True)
class StepSample:
    """One control-rate state record with model-tick and object supervision."""

    sim_step: int
    sim_time_s: float
    model_tick: int
    env_id: int
    robot_root_world: Pose
    robot_twist_world: Twist
    tcp_base: Pose
    joints: JointState
    action: CanonicalAction
    objects: tuple[ObjectState, ...]
    left_contact_object_ids: tuple[str, ...]
    right_contact_object_ids: tuple[str, ...]
    camera_frames: tuple[CameraFrameRef, ...]
    future_object_states: tuple[FutureObjectState, ...]
    phase: str
    selected_object_id: str | None = None
    action_chunk_id: str | None = None
    action_index_in_chunk: int | None = None
    robot_fallen: bool = False
    forbidden_collision: bool = False
    belt_measured_speed_mps: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sim_step", "model_tick", "env_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _validate_nonnegative_number(self.sim_time_s, "sim_time_s")
        if (
            isinstance(self.belt_measured_speed_mps, bool)
            or not isinstance(self.belt_measured_speed_mps, Real)
            or not math.isfinite(self.belt_measured_speed_mps)
        ):
            raise ValueError("belt_measured_speed_mps must be finite")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("phase cannot be empty")
        if not isinstance(self.robot_root_world, Pose):
            raise ValueError("robot_root_world must be a Pose")
        if not isinstance(self.robot_twist_world, Twist):
            raise ValueError("robot_twist_world must be a Twist")
        if not isinstance(self.tcp_base, Pose):
            raise ValueError("tcp_base must be a Pose")
        if not isinstance(self.joints, JointState):
            raise ValueError("joints must be a JointState")
        if not isinstance(self.action, CanonicalAction):
            raise ValueError("action must be a CanonicalAction")
        if any(not isinstance(obj, ObjectState) for obj in self.objects):
            raise ValueError("objects must contain only ObjectState values")
        if any(
            not isinstance(frame, CameraFrameRef) for frame in self.camera_frames
        ):
            raise ValueError("camera_frames must contain only CameraFrameRef values")
        if any(
            not isinstance(label, FutureObjectState)
            for label in self.future_object_states
        ):
            raise ValueError(
                "future_object_states must contain only FutureObjectState values"
            )
        _validate_bool(self.robot_fallen, "robot_fallen")
        _validate_bool(self.forbidden_collision, "forbidden_collision")

        object_ids = tuple(obj.instance_id for obj in self.objects)
        _validate_unique(object_ids, "per-step object ids")
        for name, values in (
            ("left_contact_object_ids", self.left_contact_object_ids),
            ("right_contact_object_ids", self.right_contact_object_ids),
        ):
            for instance_id in values:
                _validate_id(instance_id, name)
            _validate_unique(values, name)
            unknown = set(values) - set(object_ids)
            if unknown:
                raise ValueError(f"{name} contain unknown per-step objects: {sorted(unknown)}")
        active_by_id = {obj.instance_id: obj.active for obj in self.objects}
        contacted = set(self.left_contact_object_ids) | set(
            self.right_contact_object_ids
        )
        if any(not active_by_id[instance_id] for instance_id in contacted):
            raise ValueError("contact object ids must refer to active objects")

        camera_ids = tuple(frame.camera_id for frame in self.camera_frames)
        _validate_unique(camera_ids, "camera ids")
        if self.selected_object_id is not None:
            _validate_id(self.selected_object_id, "selected_object_id")
            if self.selected_object_id not in object_ids:
                raise ValueError("selected_object_id must be present in per-step objects")

        future_keys = tuple(
            (label.instance_id, label.horizon_steps)
            for label in self.future_object_states
        )
        if len(set(future_keys)) != len(future_keys):
            raise ValueError("future object labels must be unique by instance and horizon")
        unknown_future = {
            label.instance_id for label in self.future_object_states
        } - set(object_ids)
        if unknown_future:
            raise ValueError(
                f"future labels contain unknown per-step objects: {sorted(unknown_future)}"
            )

        has_chunk_id = self.action_chunk_id is not None
        has_chunk_index = self.action_index_in_chunk is not None
        if has_chunk_id != has_chunk_index:
            raise ValueError(
                "action_chunk_id and action_index_in_chunk must be provided together"
            )
        if self.action_chunk_id is not None:
            _validate_id(self.action_chunk_id, "action_chunk_id")
            assert self.action_index_in_chunk is not None
            if (
                isinstance(self.action_index_in_chunk, bool)
                or not isinstance(self.action_index_in_chunk, int)
                or self.action_index_in_chunk < 0
            ):
                raise ValueError(
                    "action_index_in_chunk must be a non-negative integer"
                )

    def validate_against(
        self, task: TaskManifest, config: "BenchmarkConfig"
    ) -> None:
        registered_ids = set(task.object_by_id)
        sample_ids = {obj.instance_id for obj in self.objects}
        unknown = sample_ids - registered_ids
        if unknown:
            raise ValueError(
                f"sample contains unregistered object instances: {sorted(unknown)}"
            )
        self.action.validate_for_robot_mode(task.robot_mode)

        expected_horizons = set(config.future_horizons_steps)
        labels_by_object = {
            instance_id: {
                label.horizon_steps
                for label in self.future_object_states
                if label.instance_id == instance_id
            }
            for instance_id in sample_ids
        }
        for instance_id, horizons in labels_by_object.items():
            if horizons != expected_horizons:
                raise ValueError(
                    f"future horizons for {instance_id!r} must be "
                    f"{config.future_horizons_steps}"
                )

        state_by_id = {obj.instance_id: obj for obj in self.objects}
        for label in self.future_object_states:
            if label.horizon_steps != 0 or not label.valid:
                continue
            state = state_by_id[label.instance_id]
            assert label.pose_world is not None
            assert label.twist_world is not None
            if label.pose_world != state.pose_world or label.twist_world != state.twist_world:
                raise ValueError("valid horizon-0 labels must equal the current object state")


@dataclass(frozen=True)
class Event:
    kind: EventKind
    time_s: float
    sim_step: int | None = None
    object_instance_id: str | None = None
    goal_zone_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            raise ValueError("kind must be an EventKind")
        _validate_nonnegative_number(self.time_s, "time_s")
        if self.sim_step is not None and (
            isinstance(self.sim_step, bool)
            or not isinstance(self.sim_step, int)
            or self.sim_step < 0
        ):
            raise ValueError("sim_step must be a non-negative integer")
        if self.object_instance_id is not None:
            _validate_id(self.object_instance_id, "object_instance_id")
        if self.goal_zone_id is not None:
            _validate_id(self.goal_zone_id, "goal_zone_id")

    def validate_against(self, task: TaskManifest) -> None:
        if (
            self.object_instance_id is not None
            and self.object_instance_id not in task.object_by_id
        ):
            raise ValueError("event references an unregistered object instance")
        if (
            self.goal_zone_id is not None
            and self.goal_zone_id not in task.goal_zone_by_id
        ):
            raise ValueError("event references an unregistered goal zone")


def to_jsonable(value: Any) -> Any:
    """Convert protocol values to objects accepted by strict ``json.dump``."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value
