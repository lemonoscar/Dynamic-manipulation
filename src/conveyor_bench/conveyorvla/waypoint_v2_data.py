"""Immutable, state-free Liangzhu dataset for Waypoint Policy v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    CAMERA_CALIBRATION_ID,
    HISTORY_SPAN_S,
    LABEL_FRAME_ID,
    MANIPULATION_STRIDE_S,
    NAVIGATION_STRIDE_S,
    WaypointActionDomain,
    WaypointRoute,
    action_domain,
    arm_target_base,
    waypoint_prompt,
)
from conveyor_bench.conveyorvla.waypoint_data import (
    FORBIDDEN_MODEL_KEYS,
    NORMALIZATION_SCHEMA_VERSION,
    SPLIT_SEED,
    WaypointNormalizer,
    _append_normalization_values,
    _build_normalization,
    _dataset_clip_rates,
    _episode_split,
    _finite_vector,
    _gripper_fraction,
    _mapping,
    _read_json,
    _read_jsonl,
    _sha256,
    _shape,
    _source_episode_id,
    discover_eligible_waypoint_episodes,
    iter_waypoint_records,
)
from conveyor_bench.conveyorvla.waypoint_v2 import (
    BOUNDARY_EVENTS,
    DATASET_SCHEMA_VERSION_V2,
    DATASET_TRANSFORM_VERSION_V2,
    EXPECTED_NEXT_ROUTE,
    LOCAL_CRL_GOALS,
)


NORMALIZATION_SCHEMA_VERSION_V2 = "conveyorvla-waypoint-normalization-v2"
BOUNDARY_WINDOW_S = 1.0
SUFFIX_REASONS = frozenset(
    {"none", "boundary", "source-tail", "episode-tail", "done-no-action"}
)
MODEL_BATCH_KEYS_V2 = frozenset(
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
        "source_episode_id",
        "next_route",
        "boundary_class",
        "boundary_signed_time_s",
        "time_to_boundary_s",
        "time_to_boundary_valid",
        "phase_progress",
        "phase_segment_id",
        "transition_id",
        "transition_window",
        "original_valid_prefix_k",
        "prefix_target_k",
        "terminal_hold_mask",
        "suffix_reason",
        "crl_goal_index",
        "crl_goal_text",
        "on_policy_correction",
    }
)


class WaypointV2Normalizer(WaypointNormalizer):
    """Use unchanged action math while rejecting v1 normalizer identity."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION_V2:
            raise M0MobileError("waypoint-v2 normalization schema is incompatible")
        if payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION_V2:
            raise M0MobileError("waypoint-v2 normalization dataset identity is incompatible")
        compatible = dict(payload)
        compatible["schema_version"] = NORMALIZATION_SCHEMA_VERSION
        super().__init__(compatible)
        self.payload = dict(payload)


def iter_waypoint_v2_records(
    episode_root: str | Path,
    *,
    split: str | None = None,
    calibration_id: str = CAMERA_CALIBRATION_ID,
    require_source_audit: bool = True,
) -> Iterator[dict[str, Any]]:
    """Derive terminal-hold and transition labels without changing v1 rows."""

    root = Path(episode_root).expanduser().resolve()
    records = list(
        iter_waypoint_records(
            root,
            split=split,
            calibration_id=calibration_id,
            require_source_audit=require_source_audit,
        )
    )
    if not records:
        raise M0MobileError(f"waypoint-v2 source produced no v1-compatible rows: {root}")
    raw_samples = {
        int(row.get("frame_index", index)): row
        for index, row in enumerate(_read_jsonl(root / "samples.jsonl"))
    }
    yield from upgrade_waypoint_records(records, raw_samples)


def upgrade_waypoint_records(
    records: Sequence[Mapping[str, Any]],
    raw_samples: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Upgrade one ordered episode while preserving the original action prefix."""

    if not records:
        raise M0MobileError("waypoint-v2 upgrade needs a non-empty episode")
    episode_ids = {str(record["source_episode_id"]) for record in records}
    if len(episode_ids) != 1:
        raise M0MobileError("waypoint-v2 upgrade cannot mix source episodes")
    timestamps = [float(record["timestamp"]) for record in records]
    if any(not math.isfinite(value) for value in timestamps) or any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        raise M0MobileError("waypoint-v2 episode timestamps must strictly increase")
    routes = [WaypointRoute(str(record["route"])) for record in records]
    segments, segment_by_position = _segments(routes, timestamps, records)
    events = _expected_events(routes, timestamps, records)
    episode_end = timestamps[-1]
    upgraded = []
    for position, source in enumerate(records):
        route = routes[position]
        record = dict(source)
        record["schema_version"] = DATASET_SCHEMA_VERSION_V2
        segment = segments[segment_by_position[position]]
        next_event = next((event for event in events if event["position"] > position), None)
        previous_event = next(
            (event for event in reversed(events) if event["position"] <= position),
            None,
        )
        boundary = _boundary_metadata(
            position,
            timestamps[position],
            previous_event,
            next_event,
        )
        progress = (
            1.0
            if segment["duration_s"] <= 1.0e-9
            else (timestamps[position] - segment["start_time_s"])
            / segment["duration_s"]
        )
        record.update(boundary)
        record.update(
            {
                "next_route": (
                    None if next_event is None else str(next_event["right_route"])
                ),
                "phase_progress": min(1.0, max(0.0, progress)),
                "phase_segment_id": segment["segment_id"],
                "phase_segment_duration_s": segment["duration_s"],
                "crl_goal_index": (
                    None
                    if route is WaypointRoute.DONE
                    else tuple(LOCAL_CRL_GOALS).index(route)
                ),
                "crl_goal_text": (
                    None if route is WaypointRoute.DONE else LOCAL_CRL_GOALS[route]
                ),
                "on_policy_correction": False,
            }
        )
        _upgrade_action_suffix(
            record,
            route,
            raw_samples,
            episode_end=episode_end,
            next_event=next_event,
        )
        upgraded.append(record)
    return tuple(upgraded)


def materialize_waypoint_v2_dataset(
    episode_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    calibration_id: str = CAMERA_CALIBRATION_ID,
) -> dict[str, Any]:
    """Atomically create the metadata-only waypoint-v2 derived dataset."""

    roots = tuple(Path(value).expanduser().resolve() for value in episode_roots)
    if not roots or len(roots) != len(set(roots)):
        raise M0MobileError("waypoint-v2 source episodes must be non-empty and unique")
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise M0MobileError(f"waypoint-v2 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    streams: dict[str, Any] = {}
    route_counts: Counter[str] = Counter()
    boundary_counts: Counter[str] = Counter()
    suffix_counts: Counter[str] = Counter()
    original_prefix_counts: Counter[str] = Counter()
    episode_reports = []
    segment_durations: dict[str, dict[str, float]] = defaultdict(dict)
    train_nav: list[list[list[float]]] = [[[] for _ in range(3)] for _ in range(ACTION_HORIZON)]
    train_arm: list[list[list[float]]] = [[[] for _ in range(6)] for _ in range(ACTION_HORIZON)]
    try:
        streams = {
            split: (staging / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in ("train", "val", "test")
        }
        for root in roots:
            episode_id = _source_episode_id(root)
            split = _episode_split(episode_id)
            row_count = 0
            episode_route_counts: Counter[str] = Counter()
            for record in iter_waypoint_v2_records(
                root,
                split=split,
                calibration_id=calibration_id,
            ):
                streams[split].write(
                    json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                )
                route_counts[f"{split}:{record['route']}"] += 1
                episode_route_counts[str(record["route"])] += 1
                suffix_counts[f"{split}:{record['suffix_reason']}"] += 1
                if record["boundary_transition"] is not None:
                    boundary_counts[f"{split}:{record['boundary_transition']}"] += 1
                if record["original_valid_prefix_k"] is not None:
                    original_prefix_counts[
                        f"{split}:{record['route']}:{record['original_valid_prefix_k']}"
                    ] += 1
                if split == "train":
                    _append_normalization_values(record, train_nav, train_arm)
                    if record["route"] != WaypointRoute.DONE.value:
                        segment_durations[str(record["route"])].setdefault(
                            str(record["phase_segment_id"]),
                            float(record["phase_segment_duration_s"]),
                        )
                row_count += 1
            if row_count == 0:
                raise M0MobileError(f"waypoint-v2 episode produced no rows: {root}")
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
        normalization["schema_version"] = NORMALIZATION_SCHEMA_VERSION_V2
        normalization["dataset_schema_version"] = DATASET_SCHEMA_VERSION_V2
        compatible = _v1_compatible_normalization(normalization)
        split_clip_rates = _dataset_clip_rates(staging, compatible)
        normalization["clip_rate_definition"] = "maximum_one_sided_saturation_fraction"
        normalization["two_sided_clip_rate_reported"] = True
        normalization["split_continuous_clip_rates"] = split_clip_rates
        normalization["train_continuous_clip_rate"] = split_clip_rates["train"]["overall"][
            "max_one_sided_clip_rate"
        ]
        normalization_path = staging / "normalization.json"
        normalization_path.write_text(
            json.dumps(normalization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tau = {
            route.value: statistics.median(segment_durations[route.value].values())
            for route in LOCAL_CRL_GOALS
            if segment_durations[route.value]
        }
        if set(tau) != {route.value for route in LOCAL_CRL_GOALS}:
            raise M0MobileError("waypoint-v2 train split lacks a route duration for CRL tau")
        records = {
            split: {
                "relative_path": f"{split}.jsonl",
                "sha256": _sha256(staging / f"{split}.jsonl"),
            }
            for split in ("train", "val", "test")
        }
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION_V2,
            "transform": {
                "identity": DATASET_TRANSFORM_VERSION_V2,
                "source_schema_version": "conveyorvla-waypoint-dense-transition-v1",
                "terminal_hold_boundary_only": True,
                "non_boundary_action_mask_unchanged": True,
            },
            "source_format": "pct_full_physics_raw",
            "source_collections": sorted(
                {
                    report["source_episode_id"].split(":", 1)[0]
                    for report in episode_reports
                }
            ),
            "source_read_only": True,
            "split_seed": SPLIT_SEED,
            "split_unit": "source_episode_id",
            "records": records,
            "normalization_relative_path": "normalization.json",
            "normalization_sha256": _sha256(normalization_path),
            "episode_count": len(episode_reports),
            "row_count": sum(report["row_count"] for report in episode_reports),
            "route_split_counts": dict(sorted(route_counts.items())),
            "boundary_split_counts": dict(sorted(boundary_counts.items())),
            "suffix_reason_split_counts": dict(sorted(suffix_counts.items())),
            "original_prefix_split_counts": dict(sorted(original_prefix_counts.items())),
            "episodes": episode_reports,
            "model_input_contract": {
                "keys": sorted(MODEL_BATCH_KEYS_V2),
                "robot_state_field_count": 0,
                "robot_state_tensor_count": 0,
                "forbidden_keys": sorted(FORBIDDEN_MODEL_KEYS),
                "semantic_history": "empty_only",
                "supervision_metadata_is_not_qwen_input": True,
            },
            "visual_history": {
                "head_frames": 2,
                "wrist_frames": 2,
                "order": "oldest_to_newest",
                "offsets_s": [-HISTORY_SPAN_S, 0.0],
            },
            "action_contract": {
                "navigation": {
                    "shape": [ACTION_HORIZON, 3],
                    "stride_s": NAVIGATION_STRIDE_S,
                    "frame": LABEL_FRAME_ID,
                },
                "manipulation": {
                    "shape": [ACTION_HORIZON, 7],
                    "stride_s": MANIPULATION_STRIDE_S,
                    "frame": LABEL_FRAME_ID,
                    "pose": "absolute_tcp_target",
                },
                "terminal_hold": {
                    "boundary_only": True,
                    "full_horizon_loss": True,
                    "original_valid_prefix_preserved": True,
                    "zero_prefix_hold_source": "query_pose",
                },
            },
            "transition_contract": {
                "boundary_window_s": BOUNDARY_WINDOW_S,
                "events": dict(BOUNDARY_EVENTS),
                "suffix_reasons": sorted(SUFFIX_REASONS),
                "prefix_candidates": list(range(1, ACTION_HORIZON + 1)),
                "zero_original_prefix_maps_to_training_target": 1,
            },
            "crl_contract": {
                "goals": {route.value: text for route, text in LOCAL_CRL_GOALS.items()},
                "tau_route_s": tau,
                "tau_estimator": "train_median_route_segment_duration_s",
                "runtime_route_override": False,
            },
            "camera_calibration_id": calibration_id,
            "label_roundtrip_tolerance": 1.0e-5,
            "prompt_sha256": hashlib.sha256(
                waypoint_prompt("{global_instruction}").encode("utf-8")
            ).hexdigest(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return {
            **manifest,
            "dataset_root": str(output),
            "manifest_sha256": _sha256(output / "manifest.json"),
        }
    except Exception:
        for stream in streams.values():
            stream.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def select_waypoint_v2_episodes_per_split(
    episode_roots: Iterable[str | Path], count: int
) -> tuple[Path, ...]:
    """Select a deterministic, source-order-preserving smoke subset."""

    if count <= 0:
        raise ValueError("waypoint-v2 per-split episode count must be positive")
    selected: list[Path] = []
    counts: Counter[str] = Counter()
    for value in episode_roots:
        root = Path(value).expanduser().resolve()
        split = _episode_split(_source_episode_id(root))
        if counts[split] >= count:
            continue
        selected.append(root)
        counts[split] += 1
        if all(counts[split_name] == count for split_name in ("train", "val", "test")):
            break
    missing = {
        split: count - counts[split]
        for split in ("train", "val", "test")
        if counts[split] != count
    }
    if missing:
        raise M0MobileError(f"waypoint-v2 source cannot fill split subset: {missing}")
    return tuple(selected)


def audit_waypoint_v2_dataset(root: str | Path) -> dict[str, Any]:
    """Audit v2 identity, suffix semantics, state boundary, hashes, and geometry."""

    dataset_root = Path(root).expanduser().resolve()
    manifest = _read_json(dataset_root / "manifest.json")
    problems: list[str] = []
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION_V2:
        problems.append("dataset schema_version is not waypoint-v2")
    transform = manifest.get("transform")
    if not isinstance(transform, Mapping) or transform.get("identity") != DATASET_TRANSFORM_VERSION_V2:
        problems.append("dataset transform identity is not waypoint-v2")
    transition_contract = manifest.get("transition_contract")
    if not isinstance(transition_contract, Mapping) or transition_contract.get(
        "prefix_candidates"
    ) != list(range(1, ACTION_HORIZON + 1)):
        problems.append("manifest does not freeze all prefix candidates 1..20")
    model_contract = manifest.get("model_input_contract")
    if (
        not isinstance(model_contract, Mapping)
        or model_contract.get("keys") != sorted(MODEL_BATCH_KEYS_V2)
        or model_contract.get("robot_state_field_count") != 0
        or model_contract.get("robot_state_tensor_count") != 0
    ):
        problems.append("manifest model input contract is missing or leaks state")
    reports = manifest.get("episodes")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise M0MobileError("waypoint-v2 manifest episodes must be a sequence")
    report_by_episode: dict[str, Mapping[str, Any]] = {}
    for report in reports:
        episode_report = _mapping(report, "episodes[]")
        episode_id = str(episode_report.get("source_episode_id", ""))
        if not episode_id or episode_id in report_by_episode:
            problems.append("manifest has an empty or duplicate source episode")
            continue
        if episode_report.get("split") != _episode_split(episode_id):
            problems.append(f"manifest episode split is wrong: {episode_id}")
        report_by_episode[episode_id] = episode_report
    episode_splits: dict[str, set[str]] = defaultdict(set)
    episode_row_counts: Counter[str] = Counter()
    episode_route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    episode_suffix_counts: dict[str, Counter[str]] = defaultdict(Counter)
    counts: Counter[str] = Counter()
    boundaries: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    original_prefixes: Counter[str] = Counter()
    transition_id_contracts: dict[str, tuple[str, str, tuple[float, float]]] = {}
    episode_transition_ids: dict[str, set[str]] = defaultdict(set)
    segment_progress: dict[str, float] = {}
    max_nav_roundtrip = 0.0
    max_arm_roundtrip = 0.0
    row_count = 0
    for split in ("train", "val", "test"):
        info = _mapping(_mapping(manifest.get("records"), "records").get(split), f"records.{split}")
        path = dataset_root / str(info.get("relative_path", ""))
        if not path.is_file() or _sha256(path) != info.get("sha256"):
            problems.append(f"{split} record file is missing or corrupt")
            continue
        for line_number, record in enumerate(_read_jsonl(path), start=1):
            row_count += 1
            prefix = f"{split}:{line_number}"
            episode_id = str(record.get("source_episode_id", ""))
            episode_splits[episode_id].add(split)
            episode_row_counts[episode_id] += 1
            if episode_id not in report_by_episode:
                problems.append(f"{prefix} references an undeclared source episode")
            if record.get("schema_version") != DATASET_SCHEMA_VERSION_V2:
                problems.append(f"{prefix} has a non-v2 row schema")
            if record.get("split") != split:
                problems.append(f"{prefix} row split does not match its file")
            try:
                route = WaypointRoute(str(record.get("route")))
                domain = WaypointActionDomain(str(record.get("action_domain")))
            except ValueError as error:
                problems.append(f"{prefix} has invalid route/domain: {error}")
                continue
            counts[f"{split}:{route.value}"] += 1
            episode_route_counts[episode_id][route.value] += 1
            transition = record.get("boundary_transition")
            if transition is not None:
                boundaries[f"{split}:{transition}"] += 1
            reason = str(record.get("suffix_reason"))
            suffixes[f"{split}:{reason}"] += 1
            episode_suffix_counts[episode_id][reason] += 1
            original_k_value = record.get("original_valid_prefix_k")
            if original_k_value is not None:
                original_prefixes[
                    f"{split}:{route.value}:{original_k_value}"
                ] += 1
            if reason not in SUFFIX_REASONS:
                problems.append(f"{prefix} has an invalid suffix_reason")
            forbidden = FORBIDDEN_MODEL_KEYS.intersection(record)
            if forbidden:
                problems.append(f"{prefix} contains forbidden top-level fields {sorted(forbidden)}")
            history = record.get("history_timestamps_s")
            if (
                not isinstance(history, Sequence)
                or isinstance(history, (str, bytes))
                or len(history) != 2
                or not math.isclose(
                    float(history[1]) - float(history[0]),
                    HISTORY_SPAN_S,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            ):
                problems.append(f"{prefix} has invalid visual history")
            if any(
                not isinstance(record.get(key), list) or len(record[key]) != 2
                for key in ("head_images", "wrist_images")
            ):
                problems.append(f"{prefix} has invalid camera frame count")
            original = tuple(bool(value) for value in record.get("original_action_valid_mask", ()))
            valid = tuple(bool(value) for value in record.get("action_valid_mask", ()))
            terminal = tuple(bool(value) for value in record.get("terminal_hold_mask", ()))
            if any(len(value) != ACTION_HORIZON for value in (original, valid, terminal)):
                problems.append(f"{prefix} has an invalid action mask length")
                continue
            if any(not left and right for left, right in zip(original, original[1:])):
                problems.append(f"{prefix} original mask is not a true prefix")
            original_k = record.get("original_valid_prefix_k")
            if route is WaypointRoute.DONE:
                if (
                    domain is not WaypointActionDomain.NONE
                    or original_k is not None
                    or record.get("padded_action") is not None
                    or any(valid)
                    or any(terminal)
                    or reason != "done-no-action"
                ):
                    problems.append(f"{prefix} has invalid DONE fields")
                continue
            if original_k != sum(original):
                problems.append(f"{prefix} original K does not match its mask")
            if record.get("prefix_target_k") != max(1, int(original_k)):
                problems.append(f"{prefix} prefix target does not preserve the zero-K rule")
            width = 3 if domain is WaypointActionDomain.NAVIGATION else 7
            action_key = "nav_waypoints_body" if width == 3 else "arm_targets_base"
            counterpart_key = "arm_targets_base" if width == 3 else "nav_waypoints_body"
            action = record.get(action_key)
            if (
                not _shape(action, (ACTION_HORIZON, width))
                or record.get(counterpart_key) is not None
                or record.get("padded_action") != action
            ):
                problems.append(f"{prefix} has invalid or inconsistent padded action")
                continue
            if reason == "boundary":
                if (
                    int(original_k) >= ACTION_HORIZON
                    or not all(valid)
                    or record.get("terminal_hold_applied") is not True
                ):
                    problems.append(f"{prefix} boundary suffix is not fully supervised")
                expected_terminal = tuple(index >= int(original_k) for index in range(ACTION_HORIZON))
                if terminal != expected_terminal:
                    problems.append(f"{prefix} boundary terminal-hold mask is wrong")
                if any(action[index] != action[int(original_k)] for index in range(int(original_k), ACTION_HORIZON)):
                    problems.append(f"{prefix} boundary suffix is not a constant hold")
            elif reason in {"source-tail", "episode-tail", "none"}:
                if (
                    valid != original
                    or any(terminal)
                    or record.get("terminal_hold_applied") is not False
                    or record.get("terminal_hold_target") is not None
                ):
                    problems.append(f"{prefix} non-boundary suffix changed supervision")
            progress = record.get("phase_progress")
            if not isinstance(progress, (int, float)) or not 0.0 <= float(progress) <= 1.0:
                problems.append(f"{prefix} has invalid phase progress")
            else:
                segment_id = str(record.get("phase_segment_id", ""))
                previous_progress = segment_progress.get(segment_id)
                if previous_progress is not None and float(progress) + 1.0e-9 < previous_progress:
                    problems.append(f"{prefix} phase progress is not monotonic")
                segment_progress[segment_id] = float(progress)
            transition_window = bool(record.get("transition_window"))
            boundary_class = str(record.get("boundary_class"))
            if transition_window:
                transition_id = record.get("transition_id")
                interval = record.get("boundary_interval_s")
                transition_name = record.get("boundary_transition")
                if (
                    not isinstance(transition_id, str)
                    or transition_name not in BOUNDARY_EVENTS
                    or record.get("boundary_event") != BOUNDARY_EVENTS.get(str(transition_name))
                    or not isinstance(interval, Sequence)
                    or len(interval) != 2
                    or not float(interval[0]) < float(interval[1])
                    or boundary_class not in {"BEFORE", "AFTER"}
                ):
                    problems.append(f"{prefix} has invalid transition-window metadata")
                else:
                    contract = (
                        episode_id,
                        str(transition_name),
                        (float(interval[0]), float(interval[1])),
                    )
                    if transition_id in transition_id_contracts and transition_id_contracts[transition_id] != contract:
                        problems.append(f"{prefix} transition_id changes meaning")
                    transition_id_contracts[transition_id] = contract
                    episode_transition_ids[episode_id].add(transition_id)
                signed = record.get("boundary_signed_time_s")
                if not isinstance(signed, (int, float)) or (
                    boundary_class == "BEFORE" and float(signed) > 1.0e-9
                ) or (boundary_class == "AFTER" and float(signed) < -1.0e-9):
                    problems.append(f"{prefix} boundary signed time has the wrong sign")
            elif (
                boundary_class != "INTERIOR"
                or record.get("transition_id") is not None
                or record.get("boundary_transition") is not None
                or record.get("boundary_interval_s") is not None
            ):
                problems.append(f"{prefix} interior row exposes boundary metadata")
            roundtrip = record.get("roundtrip_error")
            if isinstance(roundtrip, Mapping):
                max_nav_roundtrip = max(max_nav_roundtrip, float(roundtrip.get("navigation_max_m_or_rad", 0.0)))
                max_arm_roundtrip = max(max_arm_roundtrip, float(roundtrip.get("arm_max_m_or_rad", 0.0)))
    leaked = [episode for episode, splits in episode_splits.items() if len(splits) != 1]
    if leaked:
        problems.append(f"source episodes leak across splits: {leaked[:3]}")
    if len(report_by_episode) != int(manifest.get("episode_count", -1)):
        problems.append("manifest episode_count does not match episode reports")
    if set(episode_splits) != set(report_by_episode):
        problems.append("manifest episode reports do not match record episodes")
    for episode_id, report in report_by_episode.items():
        if episode_row_counts[episode_id] != int(report.get("row_count", -1)):
            problems.append(f"manifest row count is wrong for {episode_id}")
        if dict(sorted(episode_route_counts[episode_id].items())) != report.get("route_counts"):
            problems.append(f"manifest route counts are wrong for {episode_id}")
        transition_names = {
            transition_id_contracts[transition_id][1]
            for transition_id in episode_transition_ids[episode_id]
        }
        expected_transitions = set(BOUNDARY_EVENTS)
        has_done = episode_route_counts[episode_id][WaypointRoute.DONE.value] > 0
        if not has_done:
            expected_transitions.remove("PLACE->DONE")
        if transition_names != expected_transitions:
            problems.append(
                f"source episode transition set is inconsistent: {episode_id}: "
                f"expected={sorted(expected_transitions)}, actual={sorted(transition_names)}"
            )
        if not has_done and episode_suffix_counts[episode_id]["episode-tail"] == 0:
            problems.append(
                f"source episode without DONE does not preserve episode-tail: {episode_id}"
            )
    for split in ("train", "val", "test"):
        for route in WaypointRoute:
            if counts[f"{split}:{route.value}"] == 0:
                problems.append(f"missing {split}/{route.value} rows")
        for transition in BOUNDARY_EVENTS:
            if boundaries[f"{split}:{transition}"] == 0:
                problems.append(f"missing {split}/{transition} transition window")
        if suffixes[f"{split}:boundary"] == 0:
            problems.append(f"missing {split} boundary terminal-hold rows")
    tolerance = float(manifest.get("label_roundtrip_tolerance", 1.0e-5))
    if max_nav_roundtrip >= tolerance:
        problems.append(f"NAV frame round-trip {max_nav_roundtrip:.6g} exceeds {tolerance:.6g}")
    if max_arm_roundtrip >= tolerance:
        problems.append(f"ARM frame round-trip {max_arm_roundtrip:.6g} exceeds {tolerance:.6g}")
    normalization_path = dataset_root / str(manifest.get("normalization_relative_path", ""))
    clip_rates = None
    if not normalization_path.is_file() or _sha256(normalization_path) != manifest.get("normalization_sha256"):
        problems.append("normalization file is missing or corrupt")
    else:
        normalization = _read_json(normalization_path)
        try:
            WaypointV2Normalizer(normalization)
            clip_rates = _dataset_clip_rates(
                dataset_root, _v1_compatible_normalization(normalization)
            )
        except (M0MobileError, OSError, TypeError, ValueError) as error:
            problems.append(f"cannot validate waypoint-v2 normalization: {error}")
        else:
            if normalization.get("split_continuous_clip_rates") != clip_rates:
                problems.append("normalization clip-rate report does not match records")
            if float(clip_rates["train"]["overall"]["max_one_sided_clip_rate"]) >= 0.01:
                problems.append("train normalization clip rate is not below 1%")
    if row_count != int(manifest.get("row_count", -1)):
        problems.append("manifest row_count does not match records")
    for key, value in (
        ("route_split_counts", counts),
        ("boundary_split_counts", boundaries),
        ("suffix_reason_split_counts", suffixes),
        ("original_prefix_split_counts", original_prefixes),
    ):
        if manifest.get(key) != dict(sorted(value.items())):
            problems.append(f"manifest {key} does not match records")
    return {
        "schema_version": "conveyorvla-waypoint-v2-data-audit-v1",
        "ok": not problems,
        "dataset_root": str(dataset_root),
        "manifest_sha256": _sha256(dataset_root / "manifest.json"),
        "row_count": row_count,
        "episode_count": len(episode_splits),
        "route_split_counts": dict(sorted(counts.items())),
        "boundary_split_counts": dict(sorted(boundaries.items())),
        "suffix_reason_split_counts": dict(sorted(suffixes.items())),
        "state_field_count": 0,
        "state_tensor_count": 0,
        "navigation_roundtrip_max_m_or_rad": max_nav_roundtrip,
        "arm_roundtrip_max_m_or_rad": max_arm_roundtrip,
        "split_continuous_clip_rates": clip_rates,
        "problems": problems,
    }


class ConveyorVLAWaypointV2Dataset:
    """Lazy JSONL/PIL loader exposing v2 supervision but no robot state."""

    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split not in {"train", "val", "test"}:
            raise M0MobileError("waypoint-v2 split must be train, val, or test")
        self.root = Path(root).expanduser().resolve()
        self.manifest = _read_json(self.root / "manifest.json")
        if self.manifest.get("schema_version") != DATASET_SCHEMA_VERSION_V2:
            raise M0MobileError("waypoint-v2 dataset schema is incompatible")
        info = _mapping(_mapping(self.manifest.get("records"), "records").get(split), f"records.{split}")
        self.path = self.root / str(info["relative_path"])
        if not self.path.is_file() or _sha256(self.path) != info.get("sha256"):
            raise M0MobileError("waypoint-v2 split record file is missing or corrupt")
        self.normalizer = WaypointV2Normalizer.from_path(
            self.root / str(self.manifest["normalization_relative_path"])
        )
        self.split = split
        self.offsets: list[int] = []
        self.routes: list[str] = []
        self.boundaries: list[str | None] = []
        self.transition_ids: list[str | None] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = json.loads(line)
                if FORBIDDEN_MODEL_KEYS.intersection(record):
                    raise M0MobileError("waypoint-v2 record exposes a forbidden model field")
                self.offsets.append(offset)
                self.routes.append(str(record["route"]))
                self.boundaries.append(record.get("boundary_transition"))
                self.transition_ids.append(record.get("transition_id"))
        if not self.offsets:
            raise M0MobileError(f"waypoint-v2 {split} split is empty")
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
            raise M0MobileError("Pillow is required for waypoint-v2 image loading") from error
        video = []
        for key in ("head_images", "wrist_images"):
            frames = []
            for value in record[key]:
                with Image.open(value) as image:
                    frames.append(image.convert("RGB"))
            video.append(frames)
        route = WaypointRoute(str(record["route"]))
        domain = action_domain(route)
        raw_action = record.get("padded_action")
        example = {
            "video": video,
            "lang": waypoint_prompt(str(record["global_instruction"])),
            "solution": str(record["assistant_solution"]),
            "route": route.value,
            "route_token": record.get("route_token"),
            "action_domain": domain.value,
            "action": None if raw_action is None else self.normalizer.normalize(route, raw_action),
            "action_valid_mask": tuple(bool(value) for value in record["action_valid_mask"]),
            "sample_id": f"{record['source_episode_id']}:{record['source_row_id']}",
            "split": self.split,
            "source_episode_id": str(record["source_episode_id"]),
            "next_route": record.get("next_route"),
            "boundary_class": str(record["boundary_class"]),
            "boundary_signed_time_s": record.get("boundary_signed_time_s"),
            "time_to_boundary_s": (
                0.0 if record.get("time_to_boundary_s") is None else float(record["time_to_boundary_s"])
            ),
            "time_to_boundary_valid": record.get("time_to_boundary_s") is not None,
            "phase_progress": float(record["phase_progress"]),
            "phase_segment_id": str(record["phase_segment_id"]),
            "transition_id": record.get("transition_id"),
            "transition_window": bool(record["transition_window"]),
            "original_valid_prefix_k": (
                0 if record.get("original_valid_prefix_k") is None else int(record["original_valid_prefix_k"])
            ),
            "prefix_target_k": (
                0 if record.get("prefix_target_k") is None else int(record["prefix_target_k"])
            ),
            "terminal_hold_mask": tuple(bool(value) for value in record["terminal_hold_mask"]),
            "suffix_reason": str(record["suffix_reason"]),
            "crl_goal_index": (
                -1 if record.get("crl_goal_index") is None else int(record["crl_goal_index"])
            ),
            "crl_goal_text": record.get("crl_goal_text"),
            "on_policy_correction": bool(record["on_policy_correction"]),
        }
        if set(example) != MODEL_BATCH_KEYS_V2 or FORBIDDEN_MODEL_KEYS.intersection(example):
            raise M0MobileError("waypoint-v2 model batch schema changed or leaks state")
        return example

    def _record(self, index: int) -> Mapping[str, Any]:
        pid = os.getpid()
        if self._stream is None or self._stream_pid != pid:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_pid = pid
        self._stream.seek(self.offsets[index])
        return _mapping(json.loads(self._stream.readline()), "waypoint-v2 record")


def _upgrade_action_suffix(
    record: dict[str, Any],
    route: WaypointRoute,
    raw_samples: Mapping[int, Mapping[str, Any]],
    *,
    episode_end: float,
    next_event: Mapping[str, Any] | None,
) -> None:
    original = tuple(bool(value) for value in record["action_valid_mask"])
    if len(original) != ACTION_HORIZON or any(
        not left and right for left, right in zip(original, original[1:])
    ):
        raise M0MobileError("waypoint-v2 source action mask is not a true 20-step prefix")
    record["original_action_valid_mask"] = list(original)
    record["terminal_hold_mask"] = [False] * ACTION_HORIZON
    if route is WaypointRoute.DONE:
        record.update(
            {
                "original_valid_prefix_k": None,
                "prefix_target_k": None,
                "suffix_reason": "done-no-action",
                "terminal_hold_applied": False,
                "terminal_hold_target": None,
                "padded_action": None,
            }
        )
        return
    domain = action_domain(route)
    stride = NAVIGATION_STRIDE_S if domain is WaypointActionDomain.NAVIGATION else MANIPULATION_STRIDE_S
    original_k = sum(original)
    query_time = float(record["timestamp"])
    expected_k = None
    if next_event is not None and str(next_event["left_route"]) == route.value:
        time_to_boundary = float(next_event["timestamp_s"]) - query_time
        expected_k = sum(
            stride * (index + 1) < time_to_boundary - 1.0e-6
            for index in range(ACTION_HORIZON)
        )
    if original_k == ACTION_HORIZON:
        reason = "none"
    elif expected_k is not None and original_k == expected_k:
        reason = "boundary"
    elif episode_end - query_time < stride * ACTION_HORIZON - 1.0e-6:
        reason = "episode-tail"
    else:
        reason = "source-tail"
    key = (
        "nav_waypoints_body"
        if domain is WaypointActionDomain.NAVIGATION
        else "arm_targets_base"
    )
    action = [list(row) for row in record[key]]
    if reason == "boundary":
        if original_k:
            hold = list(action[original_k - 1])
        elif domain is WaypointActionDomain.NAVIGATION:
            hold = [0.0, 0.0, 0.0]
        else:
            source_row = int(record["source_row_id"])
            sample = raw_samples.get(source_row)
            if sample is None:
                raise M0MobileError("waypoint-v2 cannot resolve a zero-prefix ARM query row")
            hold = list(
                arm_target_base(
                    _finite_vector(sample.get("base_pose"), 7, "base_pose"),
                    _finite_vector(sample.get("tcp_pose"), 7, "tcp_pose"),
                    _gripper_fraction(sample),
                )
            )
        for index in range(original_k, ACTION_HORIZON):
            action[index] = list(hold)
            record["terminal_hold_mask"][index] = True
        record["action_valid_mask"] = [True] * ACTION_HORIZON
        record["terminal_hold_target"] = hold
    else:
        record["action_valid_mask"] = list(original)
        record["terminal_hold_target"] = None
    record[key] = action
    record.update(
        {
            "original_valid_prefix_k": original_k,
            "prefix_target_k": max(1, original_k),
            "suffix_reason": reason,
            "terminal_hold_applied": reason == "boundary",
            "padded_action": action,
        }
    )


def _segments(
    routes: Sequence[WaypointRoute],
    timestamps: Sequence[float],
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    segments = []
    segment_by_position = [0] * len(routes)
    start = 0
    for end in range(1, len(routes) + 1):
        if end < len(routes) and routes[end] is routes[start]:
            continue
        segment_index = len(segments)
        segment_id = (
            f"{records[start]['source_episode_id']}:{routes[start].value}:"
            f"{records[start]['source_row_id']}"
        )
        segments.append(
            {
                "segment_id": segment_id,
                "start_time_s": timestamps[start],
                "end_time_s": timestamps[end - 1],
                "duration_s": timestamps[end - 1] - timestamps[start],
            }
        )
        for position in range(start, end):
            segment_by_position[position] = segment_index
        start = end
    return segments, segment_by_position


def _expected_events(
    routes: Sequence[WaypointRoute],
    timestamps: Sequence[float],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for position in range(1, len(routes)):
        left, right = routes[position - 1], routes[position]
        if EXPECTED_NEXT_ROUTE.get(left) is not right:
            continue
        transition = f"{left.value}->{right.value}"
        events.append(
            {
                "position": position,
                "transition": transition,
                "event": BOUNDARY_EVENTS[transition],
                "left_route": left.value,
                "right_route": right.value,
                "timestamp_s": timestamps[position],
                "interval_s": [timestamps[position - 1], timestamps[position]],
                "transition_id": (
                    f"{records[position]['source_episode_id']}:{transition}:"
                    f"{records[position]['source_row_id']}"
                ),
            }
        )
    return events


def _boundary_metadata(
    position: int,
    timestamp: float,
    previous_event: Mapping[str, Any] | None,
    next_event: Mapping[str, Any] | None,
) -> dict[str, Any]:
    time_since = math.inf if previous_event is None else timestamp - float(previous_event["timestamp_s"])
    time_until = math.inf if next_event is None else float(next_event["timestamp_s"]) - timestamp
    previous_is_near = time_since <= BOUNDARY_WINDOW_S + 1.0e-6
    next_is_near = time_until <= BOUNDARY_WINDOW_S + 1.0e-6
    if previous_is_near and (not next_is_near or time_since < time_until):
        selected = previous_event
        boundary_class = "AFTER"
        signed = time_since
    elif next_is_near:
        selected = next_event
        boundary_class = "BEFORE"
        signed = -time_until
    else:
        selected = None
        boundary_class = "INTERIOR"
        signed = None
    return {
        "boundary_class": boundary_class,
        "boundary_signed_time_s": signed,
        "transition_window": selected is not None,
        "transition_id": None if selected is None else selected["transition_id"],
        "boundary_transition": None if selected is None else selected["transition"],
        "boundary_event": None if selected is None else selected["event"],
        "boundary_interval_s": None if selected is None else selected["interval_s"],
        "boundary_uncertainty_s": (
            None
            if selected is None
            else float(selected["interval_s"][1]) - float(selected["interval_s"][0])
        ),
        "time_to_boundary_s": None if next_event is None else time_until,
        "transition_position": None if selected is None else int(selected["position"]),
    }


def _v1_compatible_normalization(payload: Mapping[str, Any]) -> dict[str, Any]:
    compatible = dict(payload)
    compatible["schema_version"] = NORMALIZATION_SCHEMA_VERSION
    return compatible


__all__ = [
    "BOUNDARY_WINDOW_S",
    "ConveyorVLAWaypointV2Dataset",
    "MODEL_BATCH_KEYS_V2",
    "NORMALIZATION_SCHEMA_VERSION_V2",
    "SUFFIX_REASONS",
    "WaypointV2Normalizer",
    "audit_waypoint_v2_dataset",
    "discover_eligible_waypoint_episodes",
    "iter_waypoint_v2_records",
    "materialize_waypoint_v2_dataset",
    "select_waypoint_v2_episodes_per_split",
    "upgrade_waypoint_records",
]
