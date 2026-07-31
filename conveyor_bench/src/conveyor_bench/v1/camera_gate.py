"""Fail-closed temporal camera gate for canonical ConveyorBench V1 episodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_CAMERA_ROLES = {
    "head_rgb": "policy_observation",
    "wrist_rgb": "policy_observation",
    "overview_rgb": "observer_only",
}


class CameraGateError(ValueError):
    """Raised when an episode cannot be audited safely."""


@dataclass(frozen=True)
class CameraGateThresholds:
    min_frame_count: int = 8
    max_sampled_frames: int = 32
    pooling_block_px: int = 6
    changed_channel_delta: float = 30.0
    min_changed_block_fraction: float = 0.002
    min_dynamic_sample_fraction: float = 0.20
    min_policy_target_sample_fraction: float = 0.25
    min_physical_translation_m: float = 0.02


@dataclass(frozen=True)
class _Frame:
    index: int
    sim_step: int
    capture_time_s: float
    entries: Mapping[str, tuple[Path, str, int, int]]


ImageLoader = Callable[[Path], np.ndarray]


def audit_camera_episode(
    episode_directory: str | Path,
    *,
    thresholds: CameraGateThresholds = CameraGateThresholds(),
    image_loader: ImageLoader | None = None,
) -> dict[str, Any]:
    """Audit temporal rendering and policy-camera target evidence.

    RGB cannot prove semantic visibility.  The target check uses a conservative
    proxy instead: after ``object_spawned``, a target must cause sustained
    geometry-scale pixel changes in a policy camera.  The overview camera is
    explicitly excluded because its contract is observer-only.
    """

    episode = Path(episode_directory).resolve()
    if not episode.is_dir():
        raise CameraGateError(f"episode directory is missing: {episode}")
    frames = _load_frames(episode)
    if len(frames) < thresholds.min_frame_count:
        raise CameraGateError(
            f"too few synchronized camera frames: {len(frames)}"
        )
    steps = _read_jsonl(episode / "steps.jsonl")
    events = _read_jsonl(episode / "events.jsonl")
    _validate_step_references(frames, steps)

    load = image_loader or _load_png
    cache: dict[tuple[int, str], np.ndarray] = {}

    def pooled(frame: _Frame, camera_id: str) -> np.ndarray:
        key = (frame.index, camera_id)
        if key in cache:
            return cache[key]
        path, relative, width, height = frame.entries[camera_id]
        image = np.asarray(load(path))
        if image.ndim != 3 or image.shape[2] < 3:
            raise CameraGateError(f"{relative} must decode to an HxWx3 image")
        if image.shape[:2] != (height, width):
            raise CameraGateError(
                f"{relative} shape {image.shape[:2]} != {(height, width)}"
            )
        cache[key] = _pool_rgb(image[..., :3], thresholds.pooling_block_px)
        return cache[key]

    issues: list[dict[str, Any]] = []
    physical_motion = _physical_motion(steps)
    if physical_motion["max_translation_m"] < (
        thresholds.min_physical_translation_m
    ):
        issues.append(
            {
                "code": "insufficient_physical_motion",
                "message": (
                    "episode lacks enough physical translation to test "
                    "temporal rendering"
                ),
            }
        )

    sampled = [
        frames[index]
        for index in _uniform_indices(
            len(frames), thresholds.max_sampled_frames
        )
    ]
    dynamics: dict[str, dict[str, float | int | str]] = {}
    for camera_id, role in EXPECTED_CAMERA_ROLES.items():
        baseline = pooled(sampled[0], camera_id)
        changes = [
            _changed_fraction(
                baseline,
                pooled(frame, camera_id),
                thresholds.changed_channel_delta,
            )
            for frame in sampled[1:]
        ]
        maximum = max(changes, default=0.0)
        dynamic_fraction = sum(
            value >= thresholds.min_changed_block_fraction
            for value in changes
        ) / max(1, len(changes))
        dynamics[camera_id] = {
            "role": role,
            "sample_count": len(changes),
            "max_changed_block_fraction": maximum,
            "dynamic_sample_fraction": dynamic_fraction,
        }
        if (
            maximum < thresholds.min_changed_block_fraction
            or dynamic_fraction < thresholds.min_dynamic_sample_fraction
        ):
            issues.append(
                {
                    "code": "camera_geometry_frozen",
                    "camera_id": camera_id,
                    "message": (
                        f"{camera_id} lacks sustained geometry-scale change"
                    ),
                }
            )

    target_metrics, target_issues = _target_metrics(
        frames,
        steps,
        events,
        pooled,
        thresholds,
    )
    issues.extend(target_issues)
    return {
        "schema_version": "conveyor-bench-camera-gate-v1",
        "episode_directory": str(episode),
        "passed": not issues,
        "issues": issues,
        "metrics": {
            "frame_count": len(frames),
            "sampled_frame_count": len(sampled),
            "camera_roles": dict(EXPECTED_CAMERA_ROLES),
            "physical_motion": physical_motion,
            "camera_dynamics": dynamics,
            "policy_target_visibility_proxy": target_metrics,
            "observer_only_counts_as_policy_evidence": False,
        },
        "thresholds": vars(thresholds),
    }


def _load_frames(episode: Path) -> list[_Frame]:
    result: list[_Frame] = []
    previous_step = -1
    previous_time = -1.0
    for expected_index, row in enumerate(
        _read_jsonl(episode / "camera_frames.jsonl")
    ):
        index = _int(row.get("frame_index"), "frame_index")
        sim_step = _int(row.get("sim_step"), "sim_step")
        capture_time = _number(row.get("capture_time_s"), "capture_time_s")
        if index != expected_index:
            raise CameraGateError("frame indices must be contiguous from zero")
        if sim_step <= previous_step or capture_time <= previous_time:
            raise CameraGateError(
                "camera time and sim_step must be strictly increasing"
            )
        previous_step, previous_time = sim_step, capture_time
        raw_entries = row.get("frames")
        if not isinstance(raw_entries, Mapping):
            raise CameraGateError("camera frame row must contain frames")
        if set(raw_entries) != set(EXPECTED_CAMERA_ROLES):
            raise CameraGateError(
                f"camera ids must be {sorted(EXPECTED_CAMERA_ROLES)}"
            )
        entries: dict[str, tuple[Path, str, int, int]] = {}
        for camera_id, role in EXPECTED_CAMERA_ROLES.items():
            entry = raw_entries[camera_id]
            if not isinstance(entry, Mapping):
                raise CameraGateError(f"{camera_id} entry must be an object")
            if entry.get("role") != role:
                raise CameraGateError(f"{camera_id} must retain role {role!r}")
            relative = _safe_camera_path(
                entry.get("relative_path"), camera_id
            )
            path = (episode / relative).resolve()
            if not path.is_relative_to(episode) or not path.is_file():
                raise CameraGateError(f"missing or unsafe PNG: {relative}")
            resolution = entry.get("resolution")
            if (
                not isinstance(resolution, Sequence)
                or isinstance(resolution, (str, bytes))
                or len(resolution) != 2
            ):
                raise CameraGateError(
                    f"{camera_id} resolution must be [width, height]"
                )
            width = _int(resolution[0], f"{camera_id} width", positive=True)
            height = _int(resolution[1], f"{camera_id} height", positive=True)
            entries[camera_id] = (path, relative, width, height)
        result.append(_Frame(index, sim_step, capture_time, entries))
    return result


def _validate_step_references(
    frames: Sequence[_Frame],
    steps: Sequence[Mapping[str, Any]],
) -> None:
    by_step: dict[int, Mapping[str, Any]] = {}
    for step in steps:
        sim_step = _int(step.get("sim_step"), "step sim_step")
        if sim_step in by_step:
            raise CameraGateError(f"duplicate step sim_step: {sim_step}")
        by_step[sim_step] = step
    for frame in frames:
        step = by_step.get(frame.sim_step)
        if step is None:
            raise CameraGateError(
                f"camera frame {frame.index} has no matching step"
            )
        raw_refs = step.get("camera_frames")
        if not isinstance(raw_refs, Sequence) or isinstance(
            raw_refs, (str, bytes)
        ):
            raise CameraGateError("step camera_frames must be a list")
        refs = {
            value.get("camera_id"): value
            for value in raw_refs
            if isinstance(value, Mapping)
            and isinstance(value.get("camera_id"), str)
        }
        if set(refs) != set(EXPECTED_CAMERA_ROLES):
            raise CameraGateError(
                f"step {frame.sim_step} must reference all cameras"
            )
        for camera_id, (_, relative, _, _) in frame.entries.items():
            ref = refs[camera_id]
            if (
                ref.get("frame_index") != frame.index
                or ref.get("relative_path") != relative
            ):
                raise CameraGateError(
                    f"step {frame.sim_step} camera reference mismatch"
                )


def _physical_motion(
    steps: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    roots: list[np.ndarray] = []
    tcps: list[np.ndarray] = []
    for step in steps:
        root = _xyz(step.get("robot_root_world"))
        if root is not None:
            roots.append(root)
        metadata = step.get("metadata")
        tcp = _xyz(metadata.get("tcp_world")) if isinstance(
            metadata, Mapping
        ) else None
        if tcp is None:
            tcp = _xyz(step.get("tcp_base"))
        if tcp is not None:
            tcps.append(tcp)
    if len(roots) < 2 or len(tcps) < 2:
        raise CameraGateError("steps lack robot root or TCP pose history")
    root_motion = _max_translation(roots)
    tcp_motion = _max_translation(tcps)
    return {
        "robot_root_max_translation_m": root_motion,
        "tcp_max_translation_m": tcp_motion,
        "max_translation_m": max(root_motion, tcp_motion),
    }


def _target_metrics(
    frames: Sequence[_Frame],
    steps: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    pooled: Callable[[_Frame, str], np.ndarray],
    thresholds: CameraGateThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spawns = {
        str(event["object_instance_id"]): event
        for event in events
        if event.get("kind") == "object_spawned"
        and isinstance(event.get("object_instance_id"), str)
        and event["object_instance_id"]
    }
    selected = {
        str(step["selected_object_id"])
        for step in steps
        if isinstance(step.get("selected_object_id"), str)
        and step["selected_object_id"]
    }
    target_ids = selected or set(spawns)
    if not target_ids:
        raise CameraGateError("no spawned target object is identifiable")
    if not target_ids <= set(spawns):
        raise CameraGateError(
            f"selected targets lack spawn events: {sorted(target_ids - set(spawns))}"
        )

    policy_ids = ("head_rgb", "wrist_rgb")
    metrics: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    for target_id in sorted(target_ids):
        spawn_step = _int(
            spawns[target_id].get("sim_step"),
            f"{target_id} spawn sim_step",
        )
        end_step = frames[-1].sim_step
        placed_steps = [
            _int(event.get("sim_step"), f"{target_id} placed sim_step")
            for event in events
            if event.get("kind") == "object_placed"
            and event.get("object_instance_id") == target_id
        ]
        if placed_steps:
            end_step = min(end_step, placed_steps[0])
        active = [
            frame
            for frame in frames
            if spawn_step <= frame.sim_step <= end_step
        ]
        if len(active) < 2:
            raise CameraGateError(
                f"{target_id} has too few frames after spawn"
            )
        before_spawn = [
            frame for frame in frames if frame.sim_step < spawn_step
        ]
        baseline = before_spawn[-1] if before_spawn else active[0]
        sampled = [
            active[index]
            for index in _uniform_indices(
                len(active), thresholds.max_sampled_frames
            )
        ]
        changes = [
            max(
                _changed_fraction(
                    pooled(baseline, camera_id),
                    pooled(frame, camera_id),
                    thresholds.changed_channel_delta,
                )
                for camera_id in policy_ids
            )
            for frame in sampled
            if frame.index != baseline.index
        ]
        maximum = max(changes, default=0.0)
        visible_fraction = sum(
            value >= thresholds.min_changed_block_fraction
            for value in changes
        ) / max(1, len(changes))
        metrics[target_id] = {
            "spawn_sim_step": spawn_step,
            "end_sim_step": end_step,
            "policy_camera_ids": list(policy_ids),
            "observer_camera_ids_excluded": ["overview_rgb"],
            "sample_count": len(changes),
            "max_changed_block_fraction": maximum,
            "changed_sample_fraction": visible_fraction,
        }
        if (
            maximum < thresholds.min_changed_block_fraction
            or visible_fraction
            < thresholds.min_policy_target_sample_fraction
        ):
            issues.append(
                {
                    "code": "target_not_visible_in_policy_cameras",
                    "object_instance_id": target_id,
                    "message": (
                        f"{target_id} lacks sustained RGB change evidence in "
                        "policy cameras; overview_rgb is observer-only"
                    ),
                }
            )
    return metrics, issues


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise CameraGateError(f"required stream is missing: {path.name}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise CameraGateError(
                        f"{path.name}:{line_number} must be an object"
                    )
                rows.append(value)
    except json.JSONDecodeError as error:
        raise CameraGateError(
            f"{path.name} contains invalid JSON: {error.msg}"
        ) from error
    if not rows:
        raise CameraGateError(f"required stream is empty: {path.name}")
    return rows


def _uniform_indices(length: int, maximum: int) -> tuple[int, ...]:
    if length <= maximum:
        return tuple(range(length))
    return tuple(
        int(value)
        for value in np.linspace(
            0, length - 1, maximum, dtype=np.int64
        )
    )


def _pool_rgb(image: np.ndarray, block: int) -> np.ndarray:
    height = image.shape[0] // block * block
    width = image.shape[1] // block * block
    if height == 0 or width == 0:
        raise CameraGateError(f"image is smaller than {block}px pooling block")
    cropped = image[:height, :width].astype(np.float32, copy=False)
    return cropped.reshape(
        height // block,
        block,
        width // block,
        block,
        3,
    ).mean(axis=(1, 3))


def _changed_fraction(
    before: np.ndarray,
    after: np.ndarray,
    channel_delta: float,
) -> float:
    if before.shape != after.shape:
        raise CameraGateError("camera resolution changed within episode")
    delta = np.max(np.abs(after - before), axis=2)
    return float(np.mean(delta >= channel_delta))


def _load_png(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise CameraGateError("OpenCV is required to audit PNG files") from error
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise CameraGateError(f"could not decode PNG: {path}")
    return image


def _safe_camera_path(value: Any, camera_id: str) -> str:
    if not isinstance(value, str) or not value:
        raise CameraGateError(f"{camera_id} path must be non-empty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:2] != ("cameras", camera_id)
    ):
        raise CameraGateError(f"unsafe {camera_id} path: {value}")
    return path.as_posix()


def _xyz(value: Any) -> np.ndarray | None:
    if not isinstance(value, Mapping):
        return None
    xyz = value.get("xyz")
    if (
        not isinstance(xyz, Sequence)
        or isinstance(xyz, (str, bytes))
        or len(xyz) != 3
    ):
        return None
    try:
        result = np.asarray(xyz, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if np.all(np.isfinite(result)) else None


def _max_translation(values: Sequence[np.ndarray]) -> float:
    return max(
        float(np.linalg.norm(value - values[0]))
        for value in values
    )


def _int(value: Any, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise CameraGateError(f"{name} must be a {qualifier} integer")
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise CameraGateError(f"{name} must be finite and non-negative")
    return float(value)
