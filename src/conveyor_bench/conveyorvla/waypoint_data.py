"""Immutable, state-free Liangzhu dataset for Waypoint Policy v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.pct_dataset import (
    audit_pct_episode,
    discover_pct_episodes,
)
from conveyor_bench.conveyorvla.subtasks import PCT_PHASES, Phase
from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    CAMERA_CALIBRATION_ID,
    DATASET_SCHEMA_VERSION,
    HISTORY_SPAN_S,
    LABEL_FRAME_ID,
    MANIPULATION_STRIDE_S,
    NAVIGATION_STRIDE_S,
    ROUTE_SUBTASKS,
    ROUTE_TOKENS,
    WaypointActionDomain,
    WaypointRoute,
    action_domain,
    arm_target_base,
    arm_target_world,
    canonical_solution,
    nav_waypoint_body,
    nav_waypoint_world,
    unit_quaternion,
    waypoint_prompt,
    wrap_to_pi,
    yaw_from_quaternion,
)


QUERY_HZ = 5
CONTROL_HZ = 50
QUERY_CONTROL_STEPS = CONTROL_HZ // QUERY_HZ
DONE_STABLE_DURATION_S = 1.0
DONE_STABLE_SAMPLE_COUNT = int(round(DONE_STABLE_DURATION_S * QUERY_HZ)) + 1
NORMALIZATION_SCHEMA_VERSION = "conveyorvla-waypoint-normalization-v1"
SPLIT_SEED = "conveyor-vla-al0-liangzhu-seen-split-v2"
FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "state",
        "state28",
        "observation.state",
        "joint_positions",
        "joint_velocities",
        "tcp_pose",
        "base_pose",
        "phase",
        "operation",
        "subtask_history",
        "previous_subtask",
        "object_state",
    }
)
MODEL_BATCH_KEYS = frozenset(
    {
        "video",
        "lang",
        "solution",
        "route",
        "route_token",
        "action_domain",
        "action",
        "action_valid_mask",
        "sample_id",
        "split",
    }
)

_PHASE_ROUTES = {
    Phase.NAV_TO_SOURCE: WaypointRoute.NAV_TO_SOURCE,
    Phase.PICK: WaypointRoute.PICK,
    Phase.NAV_TO_TARGET: WaypointRoute.NAV_TO_TARGET,
    Phase.PLACE: WaypointRoute.PLACE,
}


def discover_eligible_waypoint_episodes(
    source_roots: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Return only source episodes that pass the existing physical-data gate."""

    eligible = []
    for root in discover_pct_episodes(source_roots):
        if audit_pct_episode(root)["eligible"]:
            eligible.append(root)
    if not eligible:
        raise M0MobileError("no eligible Liangzhu waypoint source episodes were found")
    return tuple(eligible)


def iter_waypoint_records(
    episode_root: str | Path,
    *,
    split: str | None = None,
    calibration_id: str = CAMERA_CALIBRATION_ID,
    require_source_audit: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield state-free, typed waypoint records directly from one raw episode."""

    root = Path(episode_root).expanduser().resolve()
    if require_source_audit:
        audit = audit_pct_episode(root)
        if not audit["eligible"]:
            raise M0MobileError(
                f"waypoint source is not eligible: {root}: "
                + "; ".join(audit["problems"])
            )
    task = _read_json(root / "task.json")
    instruction = str(task.get("instruction", "")).strip()
    if not instruction:
        raise M0MobileError(f"waypoint source instruction is missing: {root}")
    samples = tuple(_read_jsonl(root / "samples.jsonl"))
    if len(samples) < 2:
        raise M0MobileError(f"waypoint source contains fewer than two samples: {root}")
    steps = tuple(_nonnegative_integer(row.get("simulation_step"), "simulation_step") for row in samples)
    timestamps = tuple(_finite_number(row.get("timestamp"), "timestamp") for row in samples)
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise M0MobileError("waypoint source simulation steps must increase")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise M0MobileError("waypoint source timestamps must increase")

    episode_id = _source_episode_id(root)
    resolved_split = _episode_split(episode_id) if split is None else split
    if resolved_split not in {"train", "val", "test"}:
        raise M0MobileError("waypoint split must be train, val, or test")
    step_to_index = {step: index for index, step in enumerate(steps)}
    if len(step_to_index) != len(steps):
        raise M0MobileError("waypoint source contains duplicate sample steps")

    labels: list[WaypointRoute | None] = []
    for sample in samples:
        phase = PCT_PHASES.get(str(sample.get("pipeline_state", "")))
        labels.append(None if phase is None else _PHASE_ROUTES[phase])
    done_indices = _stable_done_indices(samples, labels)
    for index in done_indices:
        labels[index] = WaypointRoute.DONE

    for index in range(1, len(samples)):
        current = samples[index]
        route = labels[index]
        if route is None:
            continue
        if steps[index] - steps[index - 1] != QUERY_CONTROL_STEPS or not math.isclose(
            timestamps[index] - timestamps[index - 1],
            HISTORY_SPAN_S,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            continue
        head_images, wrist_images = _camera_history(root, samples[index - 1], current)
        domain = action_domain(route)
        valid_mask = [False] * ACTION_HORIZON
        nav_waypoints: list[list[float]] | None = None
        arm_targets: list[list[float]] | None = None
        time_offsets = [0.0] * ACTION_HORIZON
        nav_roundtrip_max = 0.0
        arm_roundtrip_max = 0.0
        target_rows: list[int | None] = [None] * ACTION_HORIZON

        if domain is not WaypointActionDomain.NONE:
            stride_s = (
                NAVIGATION_STRIDE_S
                if domain is WaypointActionDomain.NAVIGATION
                else MANIPULATION_STRIDE_S
            )
            stride_steps = int(round(stride_s * CONTROL_HZ))
            values = [[0.0] * (3 if domain is WaypointActionDomain.NAVIGATION else 7) for _ in range(ACTION_HORIZON)]
            query_base = _finite_vector(current.get("base_pose"), 7, "base_pose")
            for horizon_index in range(ACTION_HORIZON):
                time_offsets[horizon_index] = stride_s * (horizon_index + 1)
                target_step = steps[index] + stride_steps * (horizon_index + 1)
                target_index = step_to_index.get(target_step)
                if target_index is None or not _route_interval_is_pure(
                    labels,
                    steps,
                    step_to_index,
                    start_step=steps[index],
                    target_step=target_step,
                    route=route,
                ):
                    break
                target = samples[target_index]
                if domain is WaypointActionDomain.NAVIGATION:
                    waypoint = nav_waypoint_body(
                        query_base,
                        _finite_vector(target.get("base_pose"), 7, "base_pose"),
                    )
                    values[horizon_index] = list(waypoint)
                    reconstructed = nav_waypoint_world(query_base, waypoint)
                    target_pose = _finite_vector(target.get("base_pose"), 7, "base_pose")
                    nav_roundtrip_max = max(
                        nav_roundtrip_max,
                        math.hypot(reconstructed[0] - target_pose[0], reconstructed[1] - target_pose[1]),
                        abs(wrap_to_pi(reconstructed[2] - yaw_from_quaternion(target_pose[3:]))),
                    )
                else:
                    target_pose = _finite_vector(target.get("tcp_pose"), 7, "tcp_pose")
                    target_base = arm_target_base(
                        query_base,
                        target_pose,
                        _gripper_fraction(target),
                    )
                    values[horizon_index] = list(target_base)
                    reconstructed = arm_target_world(query_base, target_base)
                    arm_roundtrip_max = max(
                        arm_roundtrip_max,
                        _position_distance(reconstructed[:3], target_pose[:3]),
                        _quaternion_angle(reconstructed[3:], target_pose[3:]),
                    )
                valid_mask[horizon_index] = True
                target_rows[horizon_index] = int(target.get("frame_index", target_index))
            if domain is WaypointActionDomain.NAVIGATION:
                nav_waypoints = values
            else:
                arm_targets = values

        previous_route = _nearest_route(labels, index, -1)
        next_route = _nearest_route(labels, index, 1)
        boundary = _boundary_transition(labels, index)
        yield {
            "schema_version": DATASET_SCHEMA_VERSION,
            "source_dataset_id": episode_id.split(":", 1)[0],
            "source_episode_id": episode_id,
            "source_row_id": int(current.get("frame_index", index)),
            "split": resolved_split,
            "timestamp": timestamps[index],
            "global_instruction": instruction,
            "head_images": list(head_images),
            "wrist_images": list(wrist_images),
            "history_timestamps_s": [timestamps[index - 1], timestamps[index]],
            "route": route.value,
            "route_token": None if route is WaypointRoute.DONE else ROUTE_TOKENS[route],
            "subtask_text": "" if route is WaypointRoute.DONE else ROUTE_SUBTASKS[route],
            "assistant_solution": canonical_solution(route),
            "action_domain": domain.value,
            "nav_waypoints_body": nav_waypoints,
            "arm_targets_base": arm_targets,
            "action_valid_mask": valid_mask,
            "waypoint_time_offsets_s": time_offsets,
            "label_frame_id": LABEL_FRAME_ID,
            "calibration_id": calibration_id,
            "previous_route": None if previous_route is None else previous_route.value,
            "next_route": None if next_route is None else next_route.value,
            "is_boundary_window": boundary is not None,
            "boundary_transition": boundary,
            "label_provenance": {
                "source_samples_relative_path": "samples.jsonl",
                "query_source_row": int(current.get("frame_index", index)),
                "target_source_rows": target_rows,
                "query_simulation_step": steps[index],
                "source_pose_fields": ["base_pose", "tcp_pose", "gripper_position"],
                "source_data_used_as_model_input": False,
            },
            "roundtrip_error": {
                "navigation_max_m_or_rad": nav_roundtrip_max,
                "arm_max_m_or_rad": arm_roundtrip_max,
            },
        }


class WaypointNormalizer:
    """Per-horizon q01/q99 normalization with a fixed gripper mapping."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
            raise M0MobileError("waypoint normalization schema is incompatible")
        self.payload = dict(payload)
        self.nav_q01, self.nav_q99 = _normalization_arrays(payload, "navigation", (ACTION_HORIZON, 3))
        self.arm_q01, self.arm_q99 = _normalization_arrays(payload, "manipulation", (ACTION_HORIZON, 6))

    @classmethod
    def from_path(cls, path: str | Path) -> "WaypointNormalizer":
        return cls(_read_json(Path(path).expanduser().resolve()))

    def normalize(
        self, route: WaypointRoute | str, action: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        resolved = WaypointRoute(route)
        domain = action_domain(resolved)
        if domain is WaypointActionDomain.NONE:
            raise ValueError("DONE has no action to normalize")
        array = np.asarray(action, dtype=np.float64)
        expected = (ACTION_HORIZON, 3 if domain is WaypointActionDomain.NAVIGATION else 7)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"waypoint action must have shape {expected} and be finite")
        lower, upper = (
            (self.nav_q01, self.nav_q99)
            if domain is WaypointActionDomain.NAVIGATION
            else (self.arm_q01, self.arm_q99)
        )
        continuous = np.clip(2.0 * (array[:, : lower.shape[1]] - lower) / (upper - lower) - 1.0, -1.0, 1.0)
        if domain is WaypointActionDomain.MANIPULATION:
            continuous = np.concatenate((continuous, (2.0 * array[:, 6:7] - 1.0)), axis=1)
        return tuple(tuple(float(value) for value in row) for row in continuous)

    def denormalize(
        self, route: WaypointRoute | str, action: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], ...]:
        resolved = WaypointRoute(route)
        domain = action_domain(resolved)
        if domain is WaypointActionDomain.NONE:
            raise ValueError("DONE has no action to denormalize")
        array = np.asarray(action, dtype=np.float64)
        expected = (ACTION_HORIZON, 3 if domain is WaypointActionDomain.NAVIGATION else 7)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"normalized waypoint action must have shape {expected}")
        lower, upper = (
            (self.nav_q01, self.nav_q99)
            if domain is WaypointActionDomain.NAVIGATION
            else (self.arm_q01, self.arm_q99)
        )
        continuous = lower + (np.clip(array[:, : lower.shape[1]], -1.0, 1.0) + 1.0) * 0.5 * (upper - lower)
        if domain is WaypointActionDomain.MANIPULATION:
            continuous = np.concatenate((continuous, np.clip((array[:, 6:7] + 1.0) * 0.5, 0.0, 1.0)), axis=1)
        return tuple(tuple(float(value) for value in row) for row in continuous)


def materialize_waypoint_dataset(
    episode_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    calibration_id: str = CAMERA_CALIBRATION_ID,
) -> dict[str, Any]:
    """Atomically create the metadata-only waypoint-v1 derived dataset."""

    roots = tuple(Path(value).expanduser().resolve() for value in episode_roots)
    if not roots or len(roots) != len(set(roots)):
        raise M0MobileError("waypoint source episodes must be non-empty and unique")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise M0MobileError(f"waypoint output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    streams: dict[str, Any] = {}
    route_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    episode_reports = []
    train_nav: list[list[list[float]]] = [[[] for _ in range(3)] for _ in range(ACTION_HORIZON)]
    train_arm: list[list[list[float]]] = [[[] for _ in range(6)] for _ in range(ACTION_HORIZON)]
    try:
        streams = {split: (staging / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "val", "test")}
        for root in roots:
            episode_id = _source_episode_id(root)
            split = _episode_split(episode_id)
            row_count = 0
            episode_route_counts: Counter[str] = Counter()
            for record in iter_waypoint_records(root, split=split, calibration_id=calibration_id):
                streams[split].write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                key = f"{split}:{record['route']}"
                route_counts[key] += 1
                episode_route_counts[str(record["route"])] += 1
                if record["boundary_transition"] is not None:
                    boundary_counts[f"{split}:{record['boundary_transition']}"] += 1
                if split == "train":
                    _append_normalization_values(record, train_nav, train_arm)
                row_count += 1
            if row_count == 0:
                raise M0MobileError(f"waypoint episode produced no rows: {root}")
            episode_reports.append(
                {
                    "source_episode_id": episode_id,
                    "source_episode_root": str(root),
                    "split": split,
                    "row_count": row_count,
                    "route_counts": dict(sorted(episode_route_counts.items())),
                    "samples_sha256": _sha256(root / "samples.jsonl"),
                    "task_sha256": _sha256(root / "task.json"),
                    "summary_sha256": _sha256(root / "summary.json"),
                    "source_manifest_sha256": _sha256(root / "lerobot_manifest.json"),
                }
            )
        for stream in streams.values():
            stream.close()
        streams = {}
        normalization = _build_normalization(train_nav, train_arm)
        normalization_path = staging / "normalization.json"
        normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        record_files = {
            split: {
                "relative_path": f"{split}.jsonl",
                "sha256": _sha256(staging / f"{split}.jsonl"),
            }
            for split in ("train", "val", "test")
        }
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "source_format": "pct_full_physics_raw",
            "source_collections": sorted({report["source_episode_id"].split(":", 1)[0] for report in episode_reports}),
            "source_read_only": True,
            "split_seed": SPLIT_SEED,
            "split_unit": "source_episode_id",
            "records": record_files,
            "normalization_relative_path": "normalization.json",
            "normalization_sha256": _sha256(normalization_path),
            "episode_count": len(episode_reports),
            "row_count": sum(report["row_count"] for report in episode_reports),
            "route_split_counts": dict(sorted(route_counts.items())),
            "boundary_split_counts": dict(sorted(boundary_counts.items())),
            "episodes": episode_reports,
            "model_input_contract": {
                "keys": sorted(MODEL_BATCH_KEYS),
                "robot_state_field_count": 0,
                "robot_state_tensor_count": 0,
                "forbidden_keys": sorted(FORBIDDEN_MODEL_KEYS),
                "semantic_history": "empty_only",
            },
            "visual_history": {
                "head_frames": 2,
                "wrist_frames": 2,
                "order": "oldest_to_newest",
                "offsets_s": [-HISTORY_SPAN_S, 0.0],
            },
            "action_contract": {
                "navigation": {"shape": [ACTION_HORIZON, 3], "stride_s": NAVIGATION_STRIDE_S, "frame": LABEL_FRAME_ID},
                "manipulation": {"shape": [ACTION_HORIZON, 7], "stride_s": MANIPULATION_STRIDE_S, "frame": LABEL_FRAME_ID, "pose": "absolute_tcp_target"},
                "done": {"action_domain": WaypointActionDomain.NONE.value, "action_fields": None},
            },
            "camera_calibration_id": calibration_id,
            "label_roundtrip_tolerance": 1.0e-5,
            "done_stable_duration_s": DONE_STABLE_DURATION_S,
            "prompt_sha256": hashlib.sha256(waypoint_prompt("{global_instruction}").encode("utf-8")).hexdigest(),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, output)
        return {**manifest, "dataset_root": str(output), "manifest_sha256": _sha256(output / "manifest.json")}
    except Exception:
        for stream in streams.values():
            stream.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def audit_waypoint_dataset(root: str | Path) -> dict[str, Any]:
    """Audit split, state leakage, shapes, temporal order, masks, and hashes."""

    dataset_root = Path(root).expanduser().resolve()
    manifest = _read_json(dataset_root / "manifest.json")
    problems: list[str] = []
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        problems.append("dataset schema_version is incompatible")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, Sequence):
        raise M0MobileError("waypoint manifest episodes must be a sequence")
    episode_splits: dict[str, set[str]] = {}
    counts: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    max_nav_roundtrip = 0.0
    max_arm_roundtrip = 0.0
    row_count = 0
    for split in ("train", "val", "test"):
        record_info = _mapping(_mapping(manifest.get("records"), "records").get(split), f"records.{split}")
        path = dataset_root / str(record_info.get("relative_path", ""))
        if not path.is_file() or _sha256(path) != record_info.get("sha256"):
            problems.append(f"{split} record file is missing or corrupt")
            continue
        for line_number, record in enumerate(_read_jsonl(path), start=1):
            row_count += 1
            source_episode_id = str(record.get("source_episode_id", ""))
            episode_splits.setdefault(source_episode_id, set()).add(split)
            route = WaypointRoute(str(record.get("route")))
            domain = WaypointActionDomain(str(record.get("action_domain")))
            counts[f"{split}:{route.value}"] += 1
            boundary = record.get("boundary_transition")
            if boundary is not None:
                boundaries[f"{split}:{boundary}"] += 1
            forbidden = FORBIDDEN_MODEL_KEYS.intersection(record)
            if forbidden:
                problems.append(f"{split}:{line_number} contains forbidden top-level fields {sorted(forbidden)}")
            history = record.get("history_timestamps_s")
            if not isinstance(history, Sequence) or len(history) != 2 or not math.isclose(float(history[1]) - float(history[0]), HISTORY_SPAN_S, rel_tol=0.0, abs_tol=1.0e-6):
                problems.append(f"{split}:{line_number} has invalid visual history")
            if any(not isinstance(record.get(key), list) or len(record[key]) != 2 for key in ("head_images", "wrist_images")):
                problems.append(f"{split}:{line_number} has invalid camera frame count")
            valid = tuple(bool(value) for value in record.get("action_valid_mask", ()))
            if len(valid) != ACTION_HORIZON or any(not earlier and later for earlier, later in zip(valid, valid[1:])):
                problems.append(f"{split}:{line_number} has invalid action_valid_mask")
            if route is WaypointRoute.DONE:
                if domain is not WaypointActionDomain.NONE or record.get("nav_waypoints_body") is not None or record.get("arm_targets_base") is not None or any(valid):
                    problems.append(f"{split}:{line_number} has invalid DONE action fields")
            elif domain is WaypointActionDomain.NAVIGATION:
                if not _shape(record.get("nav_waypoints_body"), (ACTION_HORIZON, 3)) or record.get("arm_targets_base") is not None:
                    problems.append(f"{split}:{line_number} has invalid NAV action fields")
            elif domain is WaypointActionDomain.MANIPULATION:
                if not _shape(record.get("arm_targets_base"), (ACTION_HORIZON, 7)) or record.get("nav_waypoints_body") is not None:
                    problems.append(f"{split}:{line_number} has invalid ARM action fields")
            roundtrip = record.get("roundtrip_error")
            if isinstance(roundtrip, Mapping):
                max_nav_roundtrip = max(max_nav_roundtrip, float(roundtrip.get("navigation_max_m_or_rad", 0.0)))
                max_arm_roundtrip = max(max_arm_roundtrip, float(roundtrip.get("arm_max_m_or_rad", 0.0)))
    leaked = [episode for episode, splits in episode_splits.items() if len(splits) != 1]
    if leaked:
        problems.append(f"source episodes leak across splits: {leaked[:3]}")
    for split in ("train", "val", "test"):
        for route in WaypointRoute:
            if counts[f"{split}:{route.value}"] == 0:
                problems.append(f"missing {split}/{route.value} rows")
        for left, right in zip(tuple(WaypointRoute), tuple(WaypointRoute)[1:]):
            transition = f"{left.value}->{right.value}"
            if boundaries[f"{split}:{transition}"] == 0:
                problems.append(f"missing {split}/{transition} boundary window")
    tolerance = float(manifest.get("label_roundtrip_tolerance", 1.0e-5))
    if max_nav_roundtrip >= tolerance:
        problems.append(f"NAV frame round-trip {max_nav_roundtrip:.6g} exceeds {tolerance:.6g}")
    if max_arm_roundtrip >= tolerance:
        problems.append(f"ARM frame round-trip {max_arm_roundtrip:.6g} exceeds {tolerance:.6g}")
    normalization_path = dataset_root / str(manifest.get("normalization_relative_path", ""))
    if not normalization_path.is_file() or _sha256(normalization_path) != manifest.get("normalization_sha256"):
        problems.append("normalization file is missing or corrupt")
        normalization = None
    else:
        normalization = _read_json(normalization_path)
        try:
            WaypointNormalizer(normalization)
        except (M0MobileError, ValueError) as error:
            problems.append(str(error))
        clip_rate = float(normalization.get("train_continuous_clip_rate", math.inf))
        if clip_rate >= 0.01:
            problems.append(f"train normalization clip rate {clip_rate:.6g} is not below 1%")
    if row_count != int(manifest.get("row_count", -1)):
        problems.append("manifest row_count does not match records")
    return {
        "schema_version": "conveyorvla-waypoint-data-audit-v1",
        "ok": not problems,
        "dataset_root": str(dataset_root),
        "manifest_sha256": _sha256(dataset_root / "manifest.json"),
        "row_count": row_count,
        "episode_count": len(episode_splits),
        "route_split_counts": dict(sorted(counts.items())),
        "boundary_split_counts": dict(sorted(boundaries.items())),
        "state_field_count": 0,
        "state_tensor_count": 0,
        "navigation_roundtrip_max_m_or_rad": max_nav_roundtrip,
        "arm_roundtrip_max_m_or_rad": max_arm_roundtrip,
        "train_continuous_clip_rate": None if normalization is None else normalization.get("train_continuous_clip_rate"),
        "problems": problems,
    }


class ConveyorVLAWaypointDataset:
    """Lazy JSONL/PIL loader whose returned batch schema contains no state."""

    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split not in {"train", "val", "test"}:
            raise M0MobileError("waypoint split must be train, val, or test")
        self.root = Path(root).expanduser().resolve()
        self.manifest = _read_json(self.root / "manifest.json")
        if self.manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise M0MobileError("waypoint dataset schema is incompatible")
        record_info = _mapping(_mapping(self.manifest.get("records"), "records").get(split), f"records.{split}")
        self.path = self.root / str(record_info["relative_path"])
        if not self.path.is_file() or _sha256(self.path) != record_info.get("sha256"):
            raise M0MobileError("waypoint split record file is missing or corrupt")
        self.normalizer = WaypointNormalizer.from_path(self.root / str(self.manifest["normalization_relative_path"]))
        self.split = split
        self.offsets: list[int] = []
        self.routes: list[str] = []
        self.boundaries: list[str | None] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                if FORBIDDEN_MODEL_KEYS.intersection(record):
                    raise M0MobileError("waypoint record exposes a forbidden model field")
                self.offsets.append(offset)
                self.routes.append(str(record["route"]))
                self.boundaries.append(record.get("boundary_transition"))
        if not self.offsets:
            raise M0MobileError(f"waypoint {split} split is empty")
        self._stream: Any = None
        self._stream_pid: int | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_stream"] = None
        state["_stream_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._record(index)
        try:
            from PIL import Image
        except ImportError as error:
            raise M0MobileError("Pillow is required for waypoint image loading") from error
        clips = []
        for key in ("head_images", "wrist_images"):
            frames = []
            for value in record[key]:
                with Image.open(value) as image:
                    frames.append(image.convert("RGB"))
            clips.append(frames)
        route = WaypointRoute(record["route"])
        domain = action_domain(route)
        raw_action = (
            record["nav_waypoints_body"]
            if domain is WaypointActionDomain.NAVIGATION
            else record["arm_targets_base"]
            if domain is WaypointActionDomain.MANIPULATION
            else None
        )
        example = {
            "video": clips,
            "lang": waypoint_prompt(record["global_instruction"]),
            "solution": record["assistant_solution"],
            "route": route.value,
            "route_token": record["route_token"],
            "action_domain": domain.value,
            "action": None if raw_action is None else self.normalizer.normalize(route, raw_action),
            "action_valid_mask": tuple(bool(value) for value in record["action_valid_mask"]),
            "sample_id": f"{record['source_episode_id']}:{record['source_row_id']}",
            "split": self.split,
        }
        if set(example) != MODEL_BATCH_KEYS or FORBIDDEN_MODEL_KEYS.intersection(example):
            raise M0MobileError("waypoint model batch schema changed or leaks state")
        return example

    def sample_weights(self) -> tuple[float, ...]:
        route_counts = Counter(self.routes)
        boundary_counts = Counter(value for value in self.boundaries if value is not None)
        route_groups = len(route_counts)
        boundary_groups = len(boundary_counts)
        total = len(self)
        return tuple(
            total / (route_groups * route_counts[route])
            + (
                total / (boundary_groups * boundary_counts[boundary])
                if boundary is not None and boundary_groups
                else 0.0
            )
            for route, boundary in zip(self.routes, self.boundaries, strict=True)
        )

    def _record(self, index: int) -> Mapping[str, Any]:
        pid = os.getpid()
        if self._stream is None or self._stream_pid != pid:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_pid = pid
        self._stream.seek(self.offsets[index])
        value = json.loads(self._stream.readline())
        return _mapping(value, "waypoint record")


def make_waypoint_loader(
    dataset: ConveyorVLAWaypointDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> Any:
    """Create the route/boundary-balanced training loader."""

    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    try:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
    except ImportError as error:
        raise M0MobileError("torch is required for waypoint training") from error
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        dataset.sample_weights(),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        pin_memory=True,
    )


def _identity_collate(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def _stable_done_indices(
    samples: Sequence[Mapping[str, Any]], labels: Sequence[WaypointRoute | None]
) -> tuple[int, ...]:
    release_indices = [
        index
        for index, sample in enumerate(samples)
        if labels[index] is WaypointRoute.PLACE and _release_has_occurred(sample)
    ]
    if not release_indices:
        return ()
    release_index = release_indices[0]
    suffix: list[int] = []
    for index in range(len(samples) - 1, release_index - 1, -1):
        if labels[index] is not WaypointRoute.PLACE or not _stable_done_sample(samples[index]):
            if suffix:
                break
            continue
        if suffix:
            later = suffix[-1]
            if _nonnegative_integer(samples[later].get("simulation_step"), "simulation_step") - _nonnegative_integer(samples[index].get("simulation_step"), "simulation_step") != QUERY_CONTROL_STEPS:
                break
        suffix.append(index)
    suffix.reverse()
    if len(suffix) < DONE_STABLE_SAMPLE_COUNT:
        return ()
    first_time = _finite_number(samples[suffix[0]].get("timestamp"), "timestamp")
    last_time = _finite_number(samples[suffix[-1]].get("timestamp"), "timestamp")
    if last_time - first_time + 1.0e-6 < DONE_STABLE_DURATION_S:
        return ()
    return tuple(suffix)


def _release_has_occurred(sample: Mapping[str, Any]) -> bool:
    signals = sample.get("subtask_signals")
    command = str(sample.get("gripper_command", "")).lower()
    if isinstance(signals, Mapping):
        command = str(signals.get("gripper_command", command)).lower()
        segment = str(signals.get("segment_name", "")).lower()
        segment_type = str(signals.get("segment_type", "")).lower()
    else:
        segment = segment_type = ""
    return command in {"open", "hold"} and (
        "release" in segment or "release" in segment_type or "retreat_place" in segment or "return_home_after_place" in segment or "final_motion_hold" in segment
    )


def _stable_done_sample(sample: Mapping[str, Any]) -> bool:
    try:
        base = _finite_vector(sample.get("base_velocity"), 3, "base_velocity")
        object_state = _finite_vector(sample.get("object_state"), 13, "object_state")
        return (
            max(abs(value) for value in base) <= 0.03
            and math.sqrt(sum(value * value for value in object_state[7:10])) <= 0.03
            and math.sqrt(sum(value * value for value in object_state[10:13])) <= 0.10
            and _gripper_fraction(sample) >= 0.80
            and _release_has_occurred(sample)
        )
    except (M0MobileError, ValueError):
        return False


def _route_interval_is_pure(
    labels: Sequence[WaypointRoute | None],
    steps: Sequence[int],
    step_to_index: Mapping[int, int],
    *,
    start_step: int,
    target_step: int,
    route: WaypointRoute,
) -> bool:
    for step in range(start_step + QUERY_CONTROL_STEPS, target_step + 1, QUERY_CONTROL_STEPS):
        index = step_to_index.get(step)
        if index is None or steps[index] != step or labels[index] is not route:
            return False
    return True


def _camera_history(
    root: Path,
    history: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[tuple[str, str], tuple[str, str]]:
    clips = []
    for key in ("front", "wrist"):
        paths = []
        for sample in (history, current):
            camera_frames = _mapping(sample.get("camera_frames"), "camera_frames")
            frame = _mapping(camera_frames.get(key), f"camera_frames.{key}")
            relative = Path(str(frame.get("raw_image_path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise M0MobileError(f"invalid {key} image path")
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise M0MobileError(f"{key} image escapes source episode") from error
            if not resolved.is_file():
                raise M0MobileError(f"missing {key} image: {resolved}")
            paths.append(str(resolved))
        clips.append(tuple(paths))
    return clips[0], clips[1]  # type: ignore[return-value]


def _nearest_route(
    labels: Sequence[WaypointRoute | None], index: int, direction: int
) -> WaypointRoute | None:
    cursor = index + direction
    while 0 <= cursor < len(labels):
        if labels[cursor] is not None:
            return labels[cursor]
        cursor += direction
    return None


def _boundary_transition(
    labels: Sequence[WaypointRoute | None], index: int
) -> str | None:
    route = labels[index]
    if route is None:
        return None
    for distance in range(1, QUERY_HZ + 1):
        for cursor in (index - distance, index + distance):
            if 0 <= cursor < len(labels) and labels[cursor] is not None and labels[cursor] is not route:
                left, right = (labels[cursor], route) if cursor < index else (route, labels[cursor])
                return f"{left.value}->{right.value}"  # type: ignore[union-attr]
    return None


def _append_normalization_values(
    record: Mapping[str, Any],
    nav: list[list[list[float]]],
    arm: list[list[list[float]]],
) -> None:
    valid = tuple(bool(value) for value in record["action_valid_mask"])
    if record["action_domain"] == WaypointActionDomain.NAVIGATION.value:
        values = record["nav_waypoints_body"]
        target = nav
        width = 3
    elif record["action_domain"] == WaypointActionDomain.MANIPULATION.value:
        values = record["arm_targets_base"]
        target = arm
        width = 6
    else:
        return
    for step in range(ACTION_HORIZON):
        if not valid[step]:
            break
        for dimension in range(width):
            target[step][dimension].append(float(values[step][dimension]))


def _build_normalization(
    nav: list[list[list[float]]], arm: list[list[list[float]]]
) -> dict[str, Any]:
    nav_q01, nav_q99, nav_clip, nav_count = _quantiles(nav, "navigation")
    arm_q01, arm_q99, arm_clip, arm_count = _quantiles(arm, "manipulation")
    total = nav_count + arm_count
    clip_rate = (nav_clip + arm_clip) / total if total else math.inf
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "split": "train",
        "continuous_mapping": "per_horizon_q01_q99_to_minus1_plus1",
        "gripper_mapping": "2*g-1",
        "navigation": {"shape": [ACTION_HORIZON, 3], "frame": LABEL_FRAME_ID, "unit": ["m", "m", "rad"], "q01": nav_q01, "q99": nav_q99, "valid_value_count": nav_count, "clip_rate": nav_clip / nav_count},
        "manipulation": {"shape": [ACTION_HORIZON, 6], "frame": LABEL_FRAME_ID, "unit": ["m", "m", "m", "rad", "rad", "rad"], "q01": arm_q01, "q99": arm_q99, "valid_value_count": arm_count, "clip_rate": arm_clip / arm_count},
        "train_continuous_clip_rate": clip_rate,
    }


def _quantiles(
    values: list[list[list[float]]], name: str
) -> tuple[list[list[float]], list[list[float]], int, int]:
    lower: list[list[float]] = []
    upper: list[list[float]] = []
    clipped = count = 0
    for step, dimensions in enumerate(values):
        lower_row = []
        upper_row = []
        for dimension, raw in enumerate(dimensions):
            array = np.asarray(raw, dtype=np.float64)
            if array.size < 100 or not np.isfinite(array).all():
                raise M0MobileError(f"{name}[{step},{dimension}] lacks 100 finite train labels")
            q01, q99 = np.quantile(array, (0.01, 0.99))
            if q99 - q01 <= 1.0e-6:
                raise M0MobileError(f"{name}[{step},{dimension}] q99-q01 is too small")
            lower_row.append(float(q01))
            upper_row.append(float(q99))
            clipped += int(np.logical_or(array < q01, array > q99).sum())
            count += int(array.size)
        lower.append(lower_row)
        upper.append(upper_row)
    return lower, upper, clipped, count


def _normalization_arrays(
    payload: Mapping[str, Any], key: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    section = _mapping(payload.get(key), key)
    q01 = np.asarray(section.get("q01"), dtype=np.float64)
    q99 = np.asarray(section.get("q99"), dtype=np.float64)
    if q01.shape != shape or q99.shape != shape or not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 - q01 <= 1.0e-6):
        raise M0MobileError(f"{key} normalization must contain finite q01/q99 arrays of shape {shape}")
    return q01, q99


def _episode_split(episode_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{episode_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def _source_episode_id(root: Path) -> str:
    collection = next((parent.name for parent in root.parents if parent.name.startswith("liangzhu_") and "_n" in parent.name), "liangzhu_pct")
    return f"{collection}:{root.name}"


def _gripper_fraction(sample: Mapping[str, Any]) -> float:
    value = _finite_number(sample.get("gripper_position"), "gripper_position")
    return min(1.0, max(0.0, value / 0.04))


def _position_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    first, second = unit_quaternion(left), unit_quaternion(right)
    dot = abs(sum(a * b for a, b in zip(first, second, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _shape(value: Any, shape: tuple[int, int]) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != shape[0]:
        return False
    return all(isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == shape[1] and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in row) for row in value)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read waypoint JSON {path}: {error}") from error


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise M0MobileError(f"{path}:{line_number}: {error}") from error
                yield _mapping(value, f"{path}:{line_number}")
    except OSError as error:
        raise M0MobileError(f"cannot read waypoint JSONL {path}: {error}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise M0MobileError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise M0MobileError(f"{name} must contain finite values")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise M0MobileError(f"{name} must be finite")
    return float(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M0MobileError(f"{name} must be a non-negative integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ConveyorVLAWaypointDataset",
    "FORBIDDEN_MODEL_KEYS",
    "MODEL_BATCH_KEYS",
    "NORMALIZATION_SCHEMA_VERSION",
    "SPLIT_SEED",
    "WaypointNormalizer",
    "audit_waypoint_dataset",
    "discover_eligible_waypoint_episodes",
    "iter_waypoint_records",
    "make_waypoint_loader",
    "materialize_waypoint_dataset",
]
