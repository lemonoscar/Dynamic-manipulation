"""Temporal data contract and SE(3) targets for ConveyorVLA AL0."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conveyor_bench.m0_mobile import M0MobileError, M0MobileNormalizer


TEMPORAL_CONFIG_SCHEMA_VERSION = "conveyor-vla-al0-temporal-config-2"
TEMPORAL_SCHEMA_VERSION = "conveyor-vla-al0-temporal-v2"
TEMPORAL_PROFILE = "conveyorvla_al0_temporal_v2"
GRIPPER_ACTION_SOURCE = "future_measured_joint_open_fraction"
DEFAULT_TEMPORAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "conveyorvla_al0_temporal.json"
)
CAMERA_IDS = ("head_rgb", "wrist_rgb")
HISTORY_OFFSETS_MODEL_TICKS = (-2, 0)
STATE_DIM = 28
ACTION_DIM = 10
ACTION_HORIZON = 20
MODEL_HZ = 25
CONTROL_HZ = 50
ACTION_DIMENSION_MASK = (
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
GRASP_TRAINING_PHASES = frozenset(
    {
        "mobile_settle",
        "mobile_approach",
        "mobile_stabilize",
        "arm_preposition",
        "settle",
        "select",
        "pregrasp",
        "track",
        "descend",
        "close",
        "lift",
        "carry_retract",
    }
)


@dataclass(frozen=True)
class TemporalSample:
    """One two-frame/two-camera observation with independent future targets."""

    sample_id: str
    instruction: str
    camera_paths: tuple[tuple[Path, Path], tuple[Path, Path]]
    state: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    episode_id: str
    observation_model_tick: int
    observation_control_tick: int

    def as_model_example(
        self,
        normalizer: M0MobileNormalizer,
        image_loader: Callable[[Path], Any] | None = None,
    ) -> dict[str, Any]:
        videos = tuple(
            tuple(
                image_loader(path) if image_loader is not None else path
                for path in clip
            )
            for clip in self.camera_paths
        )
        return {
            "video": videos,
            "lang": self.instruction,
            "state": (normalizer.normalize_state(self.state),),
            "action": tuple(
                normalizer.normalize_action(action) for action in self.actions
            ),
            "action_mask": ACTION_DIMENSION_MASK,
            "stream_identity": {
                "episode_id": self.episode_id,
                "observation_model_tick": self.observation_model_tick,
                "observation_control_tick": self.observation_control_tick,
            },
        }


def load_temporal_config(
    path: str | Path = DEFAULT_TEMPORAL_CONFIG_PATH,
) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read temporal config {source}: {error}") from error
    if not isinstance(value, dict):
        raise M0MobileError("temporal config must be a JSON object")
    _validate_temporal_config(value)
    return value


def temporal_sample_from_record(
    record: Mapping[str, Any],
    episode_root: str | Path,
    *,
    require_images: bool = True,
) -> TemporalSample:
    if record.get("schema_version") != TEMPORAL_SCHEMA_VERSION:
        raise M0MobileError("record has an unsupported temporal schema")
    if record.get("profile") != TEMPORAL_PROFILE:
        raise M0MobileError("record has an unsupported temporal profile")
    if record.get("policy_task_scope") != "grasp_only":
        raise M0MobileError("temporal record must use grasp_only task scope")
    if record.get("gripper_action_source") != GRIPPER_ACTION_SOURCE:
        raise M0MobileError(
            "temporal record must use future measured gripper actions"
        )

    root = Path(episode_root).expanduser().resolve()
    raw_clips = _sequence(record.get("camera_clips"), "camera_clips")
    if len(raw_clips) != len(CAMERA_IDS):
        raise M0MobileError("camera_clips must contain head then wrist")
    camera_paths: list[tuple[Path, Path]] = []
    for expected_camera, raw_clip in zip(CAMERA_IDS, raw_clips, strict=True):
        clip = _mapping(raw_clip, "camera clip")
        if clip.get("camera_id") != expected_camera:
            raise M0MobileError("camera clips must be ordered head then wrist")
        offsets = tuple(
            _integer(value, "history offset")
            for value in _sequence(
                clip.get("history_offsets_model_ticks"), "history offsets"
            )
        )
        if offsets != HISTORY_OFFSETS_MODEL_TICKS:
            raise M0MobileError("camera clip history offsets are not [-2, 0]")
        frames = _sequence(clip.get("frames"), "camera clip frames")
        if len(frames) != 2:
            raise M0MobileError("each camera clip must contain exactly two frames")
        paths: list[Path] = []
        for raw_frame in frames:
            frame = _mapping(raw_frame, "camera frame")
            if frame.get("camera_id") != expected_camera:
                raise M0MobileError("camera frame ID disagrees with its clip")
            relative = Path(_string(frame.get("relative_path"), "relative_path"))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise M0MobileError(
                    f"camera frame escapes episode root: {relative}"
                ) from error
            if require_images and not path.is_file():
                raise M0MobileError(f"missing temporal camera frame: {path}")
            paths.append(path)
        camera_paths.append((paths[0], paths[1]))

    actions = tuple(
        _finite_vector(row, ACTION_DIM, "future action")
        for row in _sequence(record.get("model_action10_chunk"), "action chunk")
    )
    if len(actions) != ACTION_HORIZON:
        raise M0MobileError(f"temporal action chunk must contain {ACTION_HORIZON} rows")
    if record.get("action_rate_hz") != MODEL_HZ:
        raise M0MobileError("temporal action rate must be 25 Hz")
    if record.get("future_offsets_model_ticks") != list(
        range(1, ACTION_HORIZON + 1)
    ) and tuple(record.get("future_offsets_model_ticks", ())) != tuple(
        range(1, ACTION_HORIZON + 1)
    ):
        raise M0MobileError("future offsets must be independent ticks 1..20")

    return TemporalSample(
        sample_id=_string(record.get("sample_id"), "sample_id"),
        instruction=_string(record.get("instruction"), "instruction"),
        camera_paths=(camera_paths[0], camera_paths[1]),
        state=_finite_vector(record.get("state28"), STATE_DIM, "state28"),
        actions=actions,
        episode_id=_string(record.get("source_episode_id"), "source_episode_id"),
        observation_model_tick=_nonnegative_integer(
            record.get("observation_model_tick"), "observation_model_tick"
        ),
        observation_control_tick=_nonnegative_integer(
            record.get("observation_control_tick"), "observation_control_tick"
        ),
    )


def relative_tcp_target(
    source_root_xyz: Sequence[float],
    source_root_wxyz: Sequence[float],
    source_tcp_xyz: Sequence[float],
    source_tcp_wxyz: Sequence[float],
    future_root_xyz: Sequence[float],
    future_root_wxyz: Sequence[float],
    future_tcp_xyz: Sequence[float],
    future_tcp_wxyz: Sequence[float],
) -> tuple[float, ...]:
    """Express a future world TCP pose relative to the observation root/TCP.

    Translation is in the root frame at observation time. Rotation is the
    left-composed change from the observed TCP orientation to the future TCP
    orientation, also expressed in that frozen root frame. The six values are
    independently decodable even when earlier action rows are skipped.
    """

    root_xyz = _finite_vector(source_root_xyz, 3, "source root xyz")
    root_q = _unit_quaternion(source_root_wxyz, "source root quaternion")
    tcp_xyz = _finite_vector(source_tcp_xyz, 3, "source TCP xyz")
    tcp_q = _unit_quaternion(source_tcp_wxyz, "source TCP quaternion")
    future_root_xyz = _finite_vector(future_root_xyz, 3, "future root xyz")
    future_root_q = _unit_quaternion(
        future_root_wxyz, "future root quaternion"
    )
    future_tcp_xyz = _finite_vector(future_tcp_xyz, 3, "future TCP xyz")
    future_tcp_q = _unit_quaternion(future_tcp_wxyz, "future TCP quaternion")

    future_world_xyz = _add(
        future_root_xyz, _rotate(future_root_q, future_tcp_xyz)
    )
    future_world_q = _multiply(future_root_q, future_tcp_q)
    source_root_inverse = _conjugate(root_q)
    future_in_source_xyz = _rotate(
        source_root_inverse, _subtract(future_world_xyz, root_xyz)
    )
    future_in_source_q = _multiply(source_root_inverse, future_world_q)
    delta_xyz = _subtract(future_in_source_xyz, tcp_xyz)
    delta_q = _multiply(future_in_source_q, _conjugate(tcp_q))
    return delta_xyz + _quaternion_to_rotation_vector(delta_q)


def reconstruct_tcp_world(
    source_root_xyz: Sequence[float],
    source_root_wxyz: Sequence[float],
    source_tcp_xyz: Sequence[float],
    source_tcp_wxyz: Sequence[float],
    delta_target6: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Invert :func:`relative_tcp_target` for runtime target reconstruction."""

    root_xyz = _finite_vector(source_root_xyz, 3, "source root xyz")
    root_q = _unit_quaternion(source_root_wxyz, "source root quaternion")
    tcp_xyz = _finite_vector(source_tcp_xyz, 3, "source TCP xyz")
    tcp_q = _unit_quaternion(source_tcp_wxyz, "source TCP quaternion")
    delta = _finite_vector(delta_target6, 6, "delta target")
    target_source_xyz = _add(tcp_xyz, delta[:3])
    target_source_q = _multiply(_rotation_vector_to_quaternion(delta[3:]), tcp_q)
    target_world_xyz = _add(root_xyz, _rotate(root_q, target_source_xyz))
    target_world_q = _unit_quaternion(
        _multiply(root_q, target_source_q), "reconstructed TCP quaternion"
    )
    return target_world_xyz, target_world_q


def _validate_temporal_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != TEMPORAL_CONFIG_SCHEMA_VERSION:
        raise M0MobileError("temporal config has the wrong schema_version")
    data = _mapping(config.get("data"), "config.data")
    expected = {
        "profile": TEMPORAL_PROFILE,
        "camera_order": list(CAMERA_IDS),
        "history_offsets_model_ticks": list(HISTORY_OFFSETS_MODEL_TICKS),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_rate_hz": MODEL_HZ,
        "control_rate_hz": CONTROL_HZ,
        "gripper_action_source": GRIPPER_ACTION_SOURCE,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise M0MobileError(f"temporal config data.{key} must be {value!r}")
    if data.get("object_state_is_model_input") is not False:
        raise M0MobileError("object state must not be a temporal model input")
    if data.get("overview_camera_is_model_input") is not False:
        raise M0MobileError("overview camera must not be a temporal model input")
    streaming = _mapping(config.get("streaming"), "config.streaming")
    if streaming.get("require_episode_generation_id") is not True:
        raise M0MobileError("streaming must require an episode generation ID")
    if streaming.get("drop_fully_stale_chunks") is not True:
        raise M0MobileError("streaming must drop fully stale chunks")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M0MobileError(f"{name} must be a sequence")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise M0MobileError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise M0MobileError(f"{name} must be an integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    value = _integer(value, name)
    if value < 0:
        raise M0MobileError(f"{name} must be non-negative")
    return value


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    values = _sequence(value, name)
    if len(values) != length:
        raise M0MobileError(f"{name} must contain exactly {length} values")
    result = []
    for component in values:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise M0MobileError(f"{name} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise M0MobileError(f"{name} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _unit_quaternion(value: Any, name: str) -> tuple[float, float, float, float]:
    q = _finite_vector(value, 4, name)
    norm = math.sqrt(sum(component * component for component in q))
    if norm <= 1.0e-12:
        raise M0MobileError(f"{name} cannot be zero")
    q = tuple(component / norm for component in q)
    return tuple(-component for component in q) if q[0] < 0.0 else q


def _add(first: Sequence[float], second: Sequence[float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(first, second, strict=True))  # type: ignore[return-value]


def _subtract(
    first: Sequence[float], second: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(first, second, strict=True))  # type: ignore[return-value]


def _multiply(
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


def _conjugate(value: Sequence[float]) -> tuple[float, float, float, float]:
    return value[0], -value[1], -value[2], -value[3]


def _rotate(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    pure = (0.0, vector[0], vector[1], vector[2])
    rotated = _multiply(_multiply(quaternion, pure), _conjugate(quaternion))
    return rotated[1], rotated[2], rotated[3]


def _quaternion_to_rotation_vector(
    value: Sequence[float],
) -> tuple[float, float, float]:
    q = _unit_quaternion(value, "relative TCP quaternion")
    vector_norm = math.sqrt(sum(component * component for component in q[1:]))
    if vector_norm <= 1.0e-12:
        return 0.0, 0.0, 0.0
    angle = 2.0 * math.atan2(vector_norm, max(0.0, q[0]))
    return tuple(component * angle / vector_norm for component in q[1:])  # type: ignore[return-value]


def _rotation_vector_to_quaternion(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    vector = _finite_vector(value, 3, "rotation vector")
    angle = math.sqrt(sum(component * component for component in vector))
    if angle <= 1.0e-12:
        return 1.0, 0.0, 0.0, 0.0
    scale = math.sin(angle / 2.0) / angle
    return _unit_quaternion(
        (
            math.cos(angle / 2.0),
            vector[0] * scale,
            vector[1] * scale,
            vector[2] * scale,
        ),
        "rotation-vector quaternion",
    )


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_DIMENSION_MASK",
    "CAMERA_IDS",
    "CONTROL_HZ",
    "DEFAULT_TEMPORAL_CONFIG_PATH",
    "GRASP_TRAINING_PHASES",
    "GRIPPER_ACTION_SOURCE",
    "HISTORY_OFFSETS_MODEL_TICKS",
    "MODEL_HZ",
    "STATE_DIM",
    "TEMPORAL_CONFIG_SCHEMA_VERSION",
    "TEMPORAL_PROFILE",
    "TEMPORAL_SCHEMA_VERSION",
    "TemporalSample",
    "load_temporal_config",
    "reconstruct_tcp_world",
    "relative_tcp_target",
    "temporal_sample_from_record",
]
