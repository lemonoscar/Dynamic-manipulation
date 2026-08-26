"""Data boundary for the fresh ConveyorVLA joint-trajectory dataset.

The module deliberately contains validators and a lazy reader, but no adapter
from old Waypoint/TCP rows.  Fresh manipulation labels must come from applied
joint commands; future measured joints are never an accepted fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    DATASET_SCHEMA_VERSION,
    HISTORY_SPAN_S,
    MANIPULATION_STRIDE_S,
    MANIPULATION_STATE_DIM,
    NAVIGATION_STRIDE_S,
    NORMALIZATION_SCHEMA_VERSION,
    ROUTE_TOKENS,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
    canonical_solution,
    fixed_action,
    joint_trajectory_prompt,
    mani_state,
    terminal_hold,
    transition_routes,
)
from conveyor_bench.conveyorvla.waypoint import nav_waypoint_body


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
        "mani_state",
        "sample_id",
        "episode_id",
        "split",
        "transition_id",
        "boundary_transition",
        "boundary_signed_time_s",
        "transition_window",
        "physical_progress",
        "physical_progress_valid",
        "physical_progress_provenance",
        "progress_bucket",
        "terminal_hold_start_index",
        "terminal_hold_reason",
        "gripper_transition",
        "route_importance_weight",
    }
)
FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "state",
        "state28",
        "base_pose",
        "base_twist",
        "tcp_pose",
        "phase",
        "operation",
        "object_state",
        "object_truth",
        "previous_route",
        "previous_subtask",
        "subtask_history",
        "prefix_target_k",
        "original_valid_prefix_k",
        "trusted_prefix_k",
        "elapsed_phase_fraction",
        "row_index_progress",
    }
)
PROGRESS_PROVENANCE = frozenset(
    {
        "source_distance_and_settle",
        "pick_reach_alignment_grasp_lift",
        "target_distance_carry_and_settle",
        "place_alignment_release_and_separation",
    }
)
TERMINAL_HOLD_REASONS = frozenset(
    {"boundary", "episode_tail", "success_tail"}
)
CONTROL_STRIDE_S = 0.02
JOINT_TARGET_STEP_LIMIT_25HZ = (0.016, 0.020, 0.020, 0.020, 0.016, 0.020)
CLOCK_ABS_TOLERANCE_S = 1.0e-4


class JointTrajectoryNormalizer:
    """Train-only, channel-wise normalizer shared across routes and horizon."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
            raise M0MobileError("joint-trajectory normalization schema is incompatible")
        if payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
            raise M0MobileError("joint-trajectory normalizer targets another dataset schema")
        self.payload = dict(payload)
        self.nav = _normalization_pair(payload, "navigation", 3)
        manipulation = _mapping(payload.get("manipulation_action"), "manipulation_action")
        self.mani_delta = _normalization_pair(manipulation, "delta_q", 6)
        state = _mapping(payload.get("manipulation_state"), "manipulation_state")
        self.state_q = _normalization_pair(state, "q", 6)
        self.state_dq = _normalization_pair(state, "dq", 6)
        gripper = _mapping(manipulation.get("gripper"), "manipulation_action.gripper")
        state_gripper = _mapping(state.get("gripper"), "manipulation_state.gripper")
        if gripper != {"minimum": 0.0, "maximum": 1.0, "normalized": [-1.0, 1.0]}:
            raise M0MobileError("manipulation action gripper mapping must be fixed [0,1]->[-1,1]")
        if state_gripper != gripper:
            raise M0MobileError("manipulation state/action gripper mappings differ")
        expected_id = normalization_identity(payload)
        if payload.get("normalizer_id") != expected_id:
            raise M0MobileError("joint-trajectory normalizer identity is invalid")

    @classmethod
    def from_path(cls, path: str | Path) -> "JointTrajectoryNormalizer":
        return cls(_read_json(Path(path)))

    @classmethod
    def fit(cls, records: Iterable[Mapping[str, Any]]) -> "JointTrajectoryNormalizer":
        nav: list[Sequence[float]] = []
        mani: list[Sequence[float]] = []
        state_q: list[Sequence[float]] = []
        state_dq: list[Sequence[float]] = []
        count = 0
        for record in records:
            validate_joint_trajectory_record(record, expected_split="train")
            count += 1
            route = JointTrajectoryRoute(str(record["route"]))
            if action_domain(route) is JointTrajectoryDomain.NAVIGATION:
                nav.extend(record["nav_trajectory_body"])
            else:
                mani.extend(row[:6] for row in record["mani_delta_q_gripper"])
                state = record["mani_state"]
                state_q.append(state[:6])
                state_dq.append(state[6:12])
        if not count or not nav or not mani or not state_q:
            raise M0MobileError("normalizer fit needs train NAV and Mani records")
        payload: dict[str, Any] = {
            "schema_version": NORMALIZATION_SCHEMA_VERSION,
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "fit_split": "train",
            "fit_row_count": count,
            "shared_across_routes": True,
            "shared_across_horizon": True,
            "navigation": _quantile_payload(nav, 3),
            "manipulation_action": {
                "delta_q": _quantile_payload(mani, 6),
                "gripper": _gripper_payload(),
            },
            "manipulation_state": {
                "q": _quantile_payload(state_q, 6),
                "dq": _quantile_payload(state_dq, 6),
                "gripper": _gripper_payload(),
            },
        }
        payload["normalizer_id"] = normalization_identity(payload)
        return cls(payload)

    def normalize_action(
        self,
        route: JointTrajectoryRoute | str,
        value: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        resolved = JointTrajectoryRoute(route)
        domain = action_domain(resolved)
        rows = fixed_action(value, domain)
        if domain is JointTrajectoryDomain.NAVIGATION:
            return tuple(_normalize(row, *self.nav) for row in rows)
        return tuple(
            _normalize(row[:6], *self.mani_delta) + (2.0 * row[6] - 1.0,)
            for row in rows
        )

    def denormalize_action(
        self,
        route: JointTrajectoryRoute | str,
        value: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        resolved = JointTrajectoryRoute(route)
        domain = action_domain(resolved)
        rows = fixed_action(value, domain)
        if domain is JointTrajectoryDomain.NAVIGATION:
            return tuple(_denormalize(row, *self.nav) for row in rows)
        return tuple(
            _denormalize(row[:6], *self.mani_delta)
            + ((row[6] + 1.0) / 2.0,)
            for row in rows
        )

    def normalize_mani_state(self, value: Sequence[float]) -> tuple[float, ...]:
        raw = mani_state(value)
        return (
            _normalize(raw[:6], *self.state_q)
            + _normalize(raw[6:12], *self.state_dq)
            + (2.0 * raw[12] - 1.0,)
        )

    def denormalize_mani_state(self, value: Sequence[float]) -> tuple[float, ...]:
        normalized = _finite_vector(value, MANIPULATION_STATE_DIM, "normalized mani_state")
        return (
            _denormalize(normalized[:6], *self.state_q)
            + _denormalize(normalized[6:12], *self.state_dq)
            + (max(0.0, min(1.0, (normalized[12] + 1.0) / 2.0)),)
        )


def normalization_identity(payload: Mapping[str, Any]) -> str:
    identity_free = {key: value for key, value in payload.items() if key != "normalizer_id"}
    digest = hashlib.sha256(_canonical_json(identity_free)).hexdigest()
    return f"joint-trajectory-normalizer:{digest[:24]}"


def validate_applied_control_sample(sample: Mapping[str, Any]) -> None:
    """Validate one raw control tick without accepting measured-action fallback."""

    _nonnegative_integer(sample.get("tick_id"), "tick_id")
    timestamp = float(sample.get("timestamp_s", float("nan")))
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("timestamp_s must be finite and non-negative")
    _finite_vector(sample.get("q_measured", ()), 6, "q_measured")
    _finite_vector(sample.get("dq_measured", ()), 6, "dq_measured")
    _unit_fraction(sample.get("gripper_measured"), "gripper_measured")
    _finite_vector(sample.get("q_command_requested", ()), 6, "q_command_requested")
    _finite_vector(sample.get("q_command_applied", ()), 6, "q_command_applied")
    _unit_fraction(sample.get("gripper_command_requested"), "gripper_command_requested")
    _unit_fraction(sample.get("gripper_command_applied"), "gripper_command_applied")
    _finite_vector(sample.get("base_command_applied", ()), 3, "base_command_applied")
    _finite_vector(sample.get("base_pose_world", ()), 7, "base_pose_world")
    _finite_vector(sample.get("base_twist_world", ()), 6, "base_twist_world")
    route = JointTrajectoryRoute(str(sample.get("route", "")))
    if action_domain(route) is JointTrajectoryDomain.MANIPULATION and any(
        value != 0.0
        for value in _finite_vector(
            sample.get("base_command_applied", ()), 3, "base_command_applied"
        )
    ):
        raise ValueError("PICK/PLACE applied base command must be exactly zero")
    if sample.get("q_command_source") != "controller_applied_after_saturation":
        raise ValueError("q_command_source must prove the applied controller target")


def mani_action_from_applied_commands(
    query_state: Sequence[float],
    applied_samples: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], ...]:
    """Construct ten query-relative targets from exact applied-command ticks."""

    state = mani_state(query_state)
    if len(applied_samples) != ACTION_HORIZON:
        raise ValueError(f"Mani labels need exactly {ACTION_HORIZON} applied samples")
    result = []
    previous_tick = -1
    previous_timestamp: float | None = None
    previous_target: tuple[float, ...] | None = None
    for index, sample in enumerate(applied_samples):
        validate_applied_control_sample(sample)
        tick = int(sample["tick_id"])
        if tick <= previous_tick:
            raise ValueError("applied command ticks must be strictly increasing")
        timestamp = float(sample["timestamp_s"])
        if previous_tick >= 0 and (
            tick - previous_tick != 2
            or not math.isclose(
                timestamp - float(previous_timestamp),
                MANIPULATION_STRIDE_S,
                rel_tol=0.0,
                abs_tol=CLOCK_ABS_TOLERANCE_S,
            )
        ):
            raise ValueError("Mani applied targets must be aligned at 25 Hz")
        previous_tick = tick
        previous_timestamp = timestamp
        target = _finite_vector(sample["q_command_applied"], 6, "q_command_applied")
        if previous_target is not None:
            _validate_joint_target_step(previous_target, target)
        previous_target = target
        gripper = _unit_fraction(sample["gripper_command_applied"], "gripper_command_applied")
        result.append(tuple(target[axis] - state[axis] for axis in range(6)) + (gripper,))
    return fixed_action(result, JointTrajectoryDomain.MANIPULATION)


def derive_fresh_joint_trajectory_record(
    query: Mapping[str, Any],
    control_samples: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive one row from fresh 50 Hz logs with no measured-action fallback."""

    query_tick = _nonnegative_integer(query.get("control_tick_id"), "control_tick_id")
    current = control_samples.get(query_tick)
    if current is None:
        raise ValueError("query control tick is missing from the fresh command log")
    validate_applied_control_sample(current)
    route = JointTrajectoryRoute(str(query.get("route", "")))
    if JointTrajectoryRoute(str(current["route"])) is not route:
        raise ValueError("query route and control-log route do not match")
    domain = action_domain(route)
    stride_s = (
        NAVIGATION_STRIDE_S
        if domain is JointTrajectoryDomain.NAVIGATION
        else MANIPULATION_STRIDE_S
    )
    stride_ticks = round(stride_s / CONTROL_STRIDE_S)
    prefix: list[tuple[float, ...]] = []
    suffix_reason: str | None = None
    previous_joint_target = (
        None
        if domain is JointTrajectoryDomain.NAVIGATION
        else _finite_vector(current["q_command_applied"], 6, "q_command_applied")
    )
    for index in range(ACTION_HORIZON):
        future = control_samples.get(query_tick + stride_ticks * (index + 1))
        if future is None:
            suffix_reason = str(query.get("tail_reason") or "")
            if suffix_reason not in {"episode_tail", "success_tail"}:
                raise ValueError("missing future command is not a declared episode/success tail")
            break
        validate_applied_control_sample(future)
        if not math.isclose(
            float(future["timestamp_s"]) - float(current["timestamp_s"]),
            stride_s * (index + 1),
            rel_tol=0.0,
            abs_tol=CLOCK_ABS_TOLERANCE_S,
        ):
            raise ValueError("future applied command timestamp is not aligned to its action clock")
        if JointTrajectoryRoute(str(future["route"])) is not route:
            suffix_reason = "boundary"
            break
        if domain is JointTrajectoryDomain.NAVIGATION:
            prefix.append(
                nav_waypoint_body(
                    current["base_pose_world"], future["base_pose_world"]
                )
            )
        else:
            query_q = _finite_vector(current["q_measured"], 6, "q_measured")
            target = _finite_vector(
                future["q_command_applied"], 6, "q_command_applied"
            )
            if previous_joint_target is None:
                raise AssertionError("Mani target validator lost its query anchor")
            _validate_joint_target_step(previous_joint_target, target)
            previous_joint_target = target
            prefix.append(
                tuple(target[axis] - query_q[axis] for axis in range(6))
                + (
                    _unit_fraction(
                        future["gripper_command_applied"],
                        "gripper_command_applied",
                    ),
                )
            )
    if not prefix:
        raise ValueError("query has no legal future target in its committed route")
    action = terminal_hold(prefix, domain)
    hold_start = len(prefix) if len(prefix) < ACTION_HORIZON else ACTION_HORIZON
    progress_valid = bool(query.get("physical_progress_valid"))
    progress = query.get("physical_progress") if progress_valid else None
    provenance = query.get("physical_progress_provenance") if progress_valid else None
    bucket = progress_bucket(float(progress)) if progress_valid else None
    transition_window = bool(query.get("transition_window"))
    transition = query.get("boundary_transition") if transition_window else None
    transition_id = query.get("transition_id") if transition_window else None
    signed = query.get("boundary_signed_time_s") if transition_window else None
    timestamp = float(current["timestamp_s"])
    history = query.get("history_timestamps_s")
    if history is None:
        history = [timestamp - HISTORY_SPAN_S, timestamp]
    gripper_values = (
        [] if domain is JointTrajectoryDomain.NAVIGATION else [row[6] for row in action]
    )
    record = {
        "sample_id": str(query.get("sample_id", "")),
        "episode_id": str(query.get("episode_id", "")),
        "split": str(query.get("split", "")),
        "query_timestamp_s": timestamp,
        "history_timestamps_s": list(history),
        "global_instruction": str(query.get("global_instruction", "")),
        "head_images": list(query.get("head_images", ())),
        "wrist_images": list(query.get("wrist_images", ())),
        "route": route.value,
        "route_token": ROUTE_TOKENS[route],
        "assistant_solution": canonical_solution(route),
        "action_domain": domain.value,
        "nav_trajectory_body": (
            [list(row) for row in action]
            if domain is JointTrajectoryDomain.NAVIGATION
            else None
        ),
        "mani_delta_q_gripper": (
            [list(row) for row in action]
            if domain is JointTrajectoryDomain.MANIPULATION
            else None
        ),
        "mani_state": (
            None
            if domain is JointTrajectoryDomain.NAVIGATION
            else [
                *_finite_vector(current["q_measured"], 6, "q_measured"),
                *_finite_vector(current["dq_measured"], 6, "dq_measured"),
                _unit_fraction(current["gripper_measured"], "gripper_measured"),
            ]
        ),
        "action_provenance": (
            "teacher_base_reference"
            if domain is JointTrajectoryDomain.NAVIGATION
            else "controller_applied_after_saturation"
        ),
        "action_valid_mask": [True] * ACTION_HORIZON,
        "terminal_hold_start_index": hold_start,
        "terminal_hold_reason": suffix_reason,
        "transition_window": transition_window,
        "boundary_transition": transition,
        "transition_id": transition_id,
        "boundary_signed_time_s": signed,
        "physical_progress": progress,
        "physical_progress_valid": progress_valid,
        "physical_progress_provenance": provenance,
        "progress_bucket": bucket,
        "gripper_transition": (
            bool(gripper_values)
            and max(gripper_values) - min(gripper_values) > 1.0e-3
        ),
    }
    validate_joint_trajectory_record(record)
    return record


def materialize_fresh_joint_trajectory_dataset(
    episode_roots: Sequence[str | Path],
    output_root: str | Path,
) -> Mapping[str, Any]:
    """Publish an immutable dataset from fresh-only control/query logs.

    Each episode must contain ``summary.json`` with ``success=true``,
    ``joint_commands_50hz.jsonl``, and ``joint_queries_5hz.jsonl``.  Asset
    paths in queries are episode-relative and are copied into the release.
    """

    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"joint-trajectory output already exists: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    streams = {
        split: (staging / f"{split}.jsonl").open("x", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    split_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    episode_splits: dict[str, str] = {}
    sample_ids: set[str] = set()
    try:
        for root_value in episode_roots:
            root = Path(root_value).expanduser().resolve()
            summary = _read_json(root / "summary.json")
            if summary.get("success") is not True:
                raise M0MobileError(f"fresh episode is not successful: {root}")
            controls = {}
            for sample in _read_jsonl(root / "joint_commands_50hz.jsonl"):
                validate_applied_control_sample(sample)
                tick = int(sample["tick_id"])
                if tick in controls:
                    raise M0MobileError(f"fresh episode repeats control tick {tick}: {root}")
                controls[tick] = sample
            if not controls:
                raise M0MobileError(f"fresh episode has no control samples: {root}")
            queries = tuple(_read_jsonl(root / "joint_queries_5hz.jsonl"))
            if not queries:
                raise M0MobileError(f"fresh episode has no query rows: {root}")
            episode_id = str(queries[0].get("episode_id", ""))
            split = str(queries[0].get("split", ""))
            if split not in streams or not _safe_path_component(episode_id):
                raise M0MobileError("fresh query episode/split identity is invalid")
            if any(
                str(query.get("episode_id")) != episode_id
                or str(query.get("split")) != split
                for query in queries
            ):
                raise M0MobileError("fresh episode queries change episode/split identity")
            if episode_id in episode_splits:
                raise M0MobileError(f"fresh materialization repeats episode ID: {episode_id}")
            episode_splits[episode_id] = split
            episode_records = [
                derive_fresh_joint_trajectory_record(query, controls) for query in queries
            ]
            _validate_complete_episode_records(episode_records)
            for record in episode_records:
                sample_id = str(record["sample_id"])
                if sample_id in sample_ids:
                    raise M0MobileError(f"fresh materialization repeats sample ID: {sample_id}")
                sample_ids.add(sample_id)
                for key in ("head_images", "wrist_images"):
                    record[key] = [
                        _copy_episode_asset(root, staging, episode_id, str(value))
                        for value in record[key]
                    ]
                json.dump(record, streams[split], sort_keys=True, separators=(",", ":"))
                streams[split].write("\n")
                split_counts[split] += 1
                route_counts[f"{split}:{record['route']}"] += 1
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        if any(split_counts[split] == 0 for split in streams):
            raise M0MobileError("fresh materialization requires non-empty train/val/test")
        normalizer = JointTrajectoryNormalizer.fit(
            _read_jsonl(staging / "train.jsonl")
        )
        normalization_path = staging / "normalization.json"
        normalization_path.write_text(
            json.dumps(normalizer.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records = {
            split: {
                "relative_path": f"{split}.jsonl",
                "row_count": split_counts[split],
                "sha256": _sha256(staging / f"{split}.jsonl"),
            }
            for split in streams
        }
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": f"joint-trajectory-{uuid.uuid4().hex}",
            "immutable": True,
            "source": "fresh_applied_joint_commands_only",
            "row_count": sum(split_counts.values()),
            "episode_count": len(episode_splits),
            "records": records,
            "route_split_counts": dict(sorted(route_counts.items())),
            "normalization_relative_path": "normalization.json",
            "normalization_sha256": _sha256(normalization_path),
            "normalizer_id": normalizer.payload["normalizer_id"],
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
            if not stream.closed:
                stream.close()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_joint_trajectory_record(
    record: Mapping[str, Any], *, expected_split: str | None = None
) -> None:
    if FORBIDDEN_MODEL_KEYS.intersection(record):
        raise ValueError(
            "joint-trajectory record contains forbidden fields: "
            + ", ".join(sorted(FORBIDDEN_MODEL_KEYS.intersection(record)))
        )
    split = str(record.get("split", ""))
    if split not in {"train", "val", "test"} or (
        expected_split is not None and split != expected_split
    ):
        raise ValueError("joint-trajectory split is invalid")
    for name in ("sample_id", "episode_id", "global_instruction"):
        if not str(record.get(name, "")).strip():
            raise ValueError(f"{name} must be non-empty")
    query_time = float(record.get("query_timestamp_s", float("nan")))
    if not math.isfinite(query_time) or query_time < 0.0:
        raise ValueError("query_timestamp_s must be finite and non-negative")
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
        or not math.isclose(float(history[1]), query_time, rel_tol=0.0, abs_tol=1.0e-6)
    ):
        raise ValueError("visual history must be [query-0.20s, query]")
    for name in ("head_images", "wrist_images"):
        values = record.get(name)
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != 2
            or any(not str(value).strip() for value in values)
        ):
            raise ValueError(f"{name} must contain two image paths")

    route = JointTrajectoryRoute(str(record.get("route")))
    domain = action_domain(route)
    if record.get("route_token") != ROUTE_TOKENS[route]:
        raise ValueError("record route token does not match route")
    if record.get("assistant_solution") != canonical_solution(route):
        raise ValueError("record assistant solution is not canonical")
    if record.get("action_domain") != domain.value:
        raise ValueError("record action domain does not match route")
    valid = tuple(bool(value) for value in record.get("action_valid_mask", ()))
    if valid != (True,) * ACTION_HORIZON:
        raise ValueError("joint-trajectory action_valid_mask must be all true")
    nav = record.get("nav_trajectory_body")
    manipulation = record.get("mani_delta_q_gripper")
    state = record.get("mani_state")
    if domain is JointTrajectoryDomain.NAVIGATION:
        fixed_action(nav, domain)
        if manipulation is not None or state is not None:
            raise ValueError("NAV record exposes Mani action/state")
    else:
        rows = fixed_action(manipulation, domain)
        mani_state(state)
        if nav is not None:
            raise ValueError("Mani record exposes NAV action")
        if any(not 0.0 <= row[6] <= 1.0 for row in rows):
            raise ValueError("Mani gripper action must stay within [0, 1]")
        if record.get("action_provenance") != "controller_applied_after_saturation":
            raise ValueError("Mani action lacks applied-command provenance")

    hold_start = _bounded_integer(
        record.get("terminal_hold_start_index"),
        1,
        ACTION_HORIZON,
        "terminal_hold_start_index",
    )
    hold_reason = record.get("terminal_hold_reason")
    if hold_start < ACTION_HORIZON:
        if hold_reason not in TERMINAL_HOLD_REASONS:
            raise ValueError("terminal-hold suffix needs an allowed physical reason")
        rows = nav if domain is JointTrajectoryDomain.NAVIGATION else manipulation
        anchor = tuple(float(value) for value in rows[hold_start - 1])
        if any(tuple(float(value) for value in row) != anchor for row in rows[hold_start:]):
            raise ValueError("terminal-hold suffix does not repeat its last valid target")
    elif hold_reason is not None:
        raise ValueError("terminal_hold_reason must be null when no suffix is held")

    transition_window = record.get("transition_window")
    if not isinstance(transition_window, bool):
        raise ValueError("transition_window must be a boolean")
    transition = record.get("boundary_transition")
    transition_id = record.get("transition_id")
    signed = record.get("boundary_signed_time_s")
    if transition_window:
        old, new = transition_routes(str(transition))
        if route not in {old, new}:
            raise ValueError("boundary row route is not a transition endpoint")
        if not str(transition_id or "").strip():
            raise ValueError("boundary row needs transition_id")
        if not math.isfinite(float(signed)):
            raise ValueError("boundary row needs finite signed time")
    elif any(value is not None for value in (transition, transition_id, signed)):
        raise ValueError("interior row contains boundary metadata")

    progress_valid = record.get("physical_progress_valid")
    if not isinstance(progress_valid, bool):
        raise ValueError("physical_progress_valid must be a boolean")
    progress = record.get("physical_progress")
    provenance = record.get("physical_progress_provenance")
    if progress_valid:
        numeric = float(progress)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("physical progress must be within [0, 1]")
        if provenance not in PROGRESS_PROVENANCE:
            raise ValueError("physical progress provenance is not route-specific")
        expected_provenance = {
            JointTrajectoryRoute.NAV_TO_SOURCE: "source_distance_and_settle",
            JointTrajectoryRoute.PICK: "pick_reach_alignment_grasp_lift",
            JointTrajectoryRoute.NAV_TO_TARGET: "target_distance_carry_and_settle",
            JointTrajectoryRoute.PLACE: "place_alignment_release_and_separation",
        }[route]
        if provenance != expected_provenance:
            raise ValueError("physical progress provenance does not match route")
        if record.get("progress_bucket") != progress_bucket(numeric):
            raise ValueError("progress bucket does not match physical progress")
    elif progress is not None or provenance is not None or record.get("progress_bucket") is not None:
        raise ValueError("invalid progress row must mask value, provenance, and bucket")
    if not isinstance(record.get("gripper_transition"), bool):
        raise ValueError("gripper_transition must be a boolean")
    if domain is JointTrajectoryDomain.NAVIGATION and record["gripper_transition"]:
        raise ValueError("NAV row cannot be a gripper transition")


def progress_bucket(progress: float) -> str:
    value = float(progress)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("physical progress must be within [0, 1]")
    return "early" if value < 1.0 / 3.0 else "middle" if value < 2.0 / 3.0 else "late"


def _validate_joint_target_step(
    previous: Sequence[float], current: Sequence[float]
) -> None:
    left = _finite_vector(previous, 6, "previous q_command_applied")
    right = _finite_vector(current, 6, "current q_command_applied")
    if any(
        abs(after - before) > limit + 1.0e-9
        for before, after, limit in zip(
            left, right, JOINT_TARGET_STEP_LIMIT_25HZ, strict=True
        )
    ):
        raise ValueError("Mani 25 Hz applied joint target exceeds the collection envelope")


def _validate_complete_episode_records(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise M0MobileError("fresh episode has no derived records")
    collapsed: list[JointTrajectoryRoute] = []
    for record in records:
        route = JointTrajectoryRoute(str(record["route"]))
        if not collapsed or collapsed[-1] is not route:
            collapsed.append(route)
    if tuple(collapsed) != tuple(JointTrajectoryRoute):
        raise M0MobileError(
            "fresh successful episode must contain the four routes exactly once in order"
        )
    for old, new in zip(tuple(JointTrajectoryRoute), tuple(JointTrajectoryRoute)[1:]):
        transition = f"{old.value}->{new.value}"
        matching = [
            record
            for record in records
            if record.get("boundary_transition") == transition
            and bool(record.get("transition_window"))
        ]
        transition_ids = {str(record.get("transition_id")) for record in matching}
        before = sum(float(record["boundary_signed_time_s"]) < 0.0 for record in matching)
        after = sum(float(record["boundary_signed_time_s"]) >= 0.0 for record in matching)
        if len(transition_ids) != 1 or before < 2 or after < 2:
            raise M0MobileError(
                f"fresh episode needs one {transition} event with at least two before/after rows"
            )


class ConveyorVLAJointTrajectoryDataset:
    """Lazy immutable JSONL/PIL dataset for the new model contract."""

    def __init__(self, root: str | Path, *, split: str = "train") -> None:
        if split not in {"train", "val", "test"}:
            raise M0MobileError("joint-trajectory split must be train, val, or test")
        self.root = Path(root).expanduser().resolve()
        self.manifest = _read_json(self.root / "manifest.json")
        if self.manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise M0MobileError("joint-trajectory dataset schema is incompatible")
        records = _mapping(self.manifest.get("records"), "records")
        info = _mapping(records.get(split), f"records.{split}")
        self.path = self.root / str(info.get("relative_path", ""))
        if not self.path.is_file() or _sha256(self.path) != info.get("sha256"):
            raise M0MobileError("joint-trajectory record file is missing or corrupt")
        normalization_path = self.root / str(self.manifest.get("normalization_relative_path", ""))
        if not normalization_path.is_file() or _sha256(normalization_path) != self.manifest.get(
            "normalization_sha256"
        ):
            raise M0MobileError("joint-trajectory normalizer is missing or corrupt")
        self.normalizer = JointTrajectoryNormalizer.from_path(normalization_path)
        self.split = split
        self.offsets: list[int] = []
        self.routes: list[str] = []
        self.episode_ids: list[str] = []
        self.transition_ids: list[str | None] = []
        self.boundary_signed_times: list[float | None] = []
        self.progress_buckets: list[str | None] = []
        self.gripper_transitions: list[bool] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                record = _mapping(json.loads(line), "joint-trajectory record")
                validate_joint_trajectory_record(record, expected_split=split)
                self.offsets.append(offset)
                self.routes.append(str(record["route"]))
                self.episode_ids.append(str(record["episode_id"]))
                self.transition_ids.append(record["transition_id"])
                self.boundary_signed_times.append(record["boundary_signed_time_s"])
                self.progress_buckets.append(record["progress_bucket"])
                self.gripper_transitions.append(bool(record["gripper_transition"]))
        if not self.offsets:
            raise M0MobileError(f"joint-trajectory {split} split is empty")
        route_counts = Counter(self.routes)
        self.route_importance_weights = {
            route: len(JointTrajectoryRoute) * count / len(self.routes)
            for route, count in route_counts.items()
        }
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
            raise M0MobileError("Pillow is required for joint-trajectory image loading") from error
        clips = []
        for key in ("head_images", "wrist_images"):
            frames = []
            for value in record[key]:
                path = _relative_asset(self.root, str(value))
                with Image.open(path) as image:
                    frames.append(image.convert("RGB"))
            clips.append(frames)
        route = JointTrajectoryRoute(str(record["route"]))
        domain = action_domain(route)
        raw_action = (
            record["nav_trajectory_body"]
            if domain is JointTrajectoryDomain.NAVIGATION
            else record["mani_delta_q_gripper"]
        )
        example = {
            "video": clips,
            "lang": joint_trajectory_prompt(str(record["global_instruction"])),
            "solution": str(record["assistant_solution"]),
            "route": route.value,
            "route_token": str(record["route_token"]),
            "action_domain": domain.value,
            "action": self.normalizer.normalize_action(route, raw_action),
            "action_valid_mask": (True,) * ACTION_HORIZON,
            "mani_state": (
                None
                if domain is JointTrajectoryDomain.NAVIGATION
                else self.normalizer.normalize_mani_state(record["mani_state"])
            ),
            "sample_id": str(record["sample_id"]),
            "episode_id": str(record["episode_id"]),
            "split": self.split,
            "transition_id": record["transition_id"],
            "boundary_transition": record["boundary_transition"],
            "boundary_signed_time_s": record["boundary_signed_time_s"],
            "transition_window": bool(record["transition_window"]),
            "physical_progress": record["physical_progress"],
            "physical_progress_valid": bool(record["physical_progress_valid"]),
            "physical_progress_provenance": record["physical_progress_provenance"],
            "progress_bucket": record["progress_bucket"],
            "terminal_hold_start_index": int(record["terminal_hold_start_index"]),
            "terminal_hold_reason": record["terminal_hold_reason"],
            "gripper_transition": bool(record["gripper_transition"]),
            # Correct the balanced four-route sampler back toward the natural
            # train distribution for the route-only objective.
            "route_importance_weight": self.route_importance_weights[route.value],
        }
        if set(example) != MODEL_BATCH_KEYS or FORBIDDEN_MODEL_KEYS.intersection(example):
            raise M0MobileError("joint-trajectory model batch schema changed or leaks forbidden state")
        return example

    def _record(self, index: int) -> Mapping[str, Any]:
        pid = os.getpid()
        if self._stream is None or self._stream_pid != pid:
            if self._stream is not None:
                self._stream.close()
            self._stream = self.path.open("rb")
            self._stream_pid = pid
        self._stream.seek(self.offsets[index])
        return _mapping(json.loads(self._stream.readline()), "joint-trajectory record")


def audit_joint_trajectory_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = _read_json(manifest_path)
    problems: list[str] = []
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        problems.append("dataset schema is incompatible")
    if manifest.get("immutable") is not True:
        problems.append("dataset manifest is not immutable")
    if manifest.get("source") != "fresh_applied_joint_commands_only":
        problems.append("dataset source is not fresh applied commands")
    if not str(manifest.get("dataset_id", "")).strip():
        problems.append("dataset ID is missing")
    records = _mapping(manifest.get("records"), "records")
    route_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    episode_splits: dict[str, set[str]] = {}
    sample_ids: set[str] = set()
    row_count = 0
    for split in ("train", "val", "test"):
        info = _mapping(records.get(split), f"records.{split}")
        path = dataset_root / str(info.get("relative_path", ""))
        if not path.is_file() or _sha256(path) != info.get("sha256"):
            problems.append(f"{split} records are missing or corrupt")
            continue
        split_row_count = 0
        for line_number, record in enumerate(_read_jsonl(path), start=1):
            row_count += 1
            split_row_count += 1
            try:
                validate_joint_trajectory_record(record, expected_split=split)
            except (TypeError, ValueError) as error:
                problems.append(f"{split}:{line_number}: {error}")
                continue
            sample_id = str(record["sample_id"])
            if sample_id in sample_ids:
                problems.append(f"duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            for image_key in ("head_images", "wrist_images"):
                for value in record[image_key]:
                    try:
                        _relative_asset(dataset_root, str(value))
                    except M0MobileError as error:
                        problems.append(f"{split}:{line_number}: {error}")
            route_counts[f"{split}:{record['route']}"] += 1
            if record["boundary_transition"] is not None:
                transition_counts[f"{split}:{record['boundary_transition']}"] += 1
            episode_splits.setdefault(str(record["episode_id"]), set()).add(split)
        if split_row_count != int(info.get("row_count", -1)):
            problems.append(f"manifest {split} row_count does not match records")
    for split in ("train", "val", "test"):
        for route in JointTrajectoryRoute:
            if route_counts[f"{split}:{route.value}"] == 0:
                problems.append(f"missing {split}/{route.value} rows")
        for left, right in zip(tuple(JointTrajectoryRoute), tuple(JointTrajectoryRoute)[1:]):
            name = f"{left.value}->{right.value}"
            if transition_counts[f"{split}:{name}"] == 0:
                problems.append(f"missing {split}/{name} boundary rows")
    leaked = sorted(episode for episode, splits in episode_splits.items() if len(splits) != 1)
    if leaked:
        problems.append(f"episodes leak across splits: {leaked[:3]}")
    normalization_path = dataset_root / str(manifest.get("normalization_relative_path", ""))
    try:
        if _sha256(normalization_path) != manifest.get("normalization_sha256"):
            raise M0MobileError("normalization hash mismatch")
        normalizer = JointTrajectoryNormalizer.from_path(normalization_path)
    except (M0MobileError, OSError, ValueError) as error:
        problems.append(str(error))
        normalizer_id = None
    else:
        normalizer_id = normalizer.payload["normalizer_id"]
        if manifest.get("normalizer_id") != normalizer_id:
            problems.append("manifest normalizer ID does not match normalization")
    if row_count != int(manifest.get("row_count", -1)):
        problems.append("manifest row_count does not match records")
    if len(episode_splits) != int(manifest.get("episode_count", -1)):
        problems.append("manifest episode_count does not match records")
    if manifest.get("route_split_counts") != dict(sorted(route_counts.items())):
        problems.append("manifest route_split_counts do not match records")
    return {
        "schema_version": "conveyorvla-joint-trajectory-data-audit-v1",
        "ok": not problems,
        "dataset_root": str(dataset_root),
        "manifest_sha256": _sha256(manifest_path),
        "normalizer_id": normalizer_id,
        "row_count": row_count,
        "episode_count": len(episode_splits),
        "route_split_counts": dict(sorted(route_counts.items())),
        "transition_split_counts": dict(sorted(transition_counts.items())),
        "problems": problems,
    }


def _quantile_payload(values: Sequence[Sequence[float]], width: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
        raise M0MobileError(f"normalizer values must have shape [rows,{width}] and be finite")
    lower = np.quantile(array, 0.01, axis=0)
    upper = np.quantile(array, 0.99, axis=0)
    center = (lower + upper) / 2.0
    half = np.maximum((upper - lower) / 2.0, 1.0e-6)
    return {
        "q01": lower.tolist(),
        "q99": upper.tolist(),
        "center": center.tolist(),
        "half_range": half.tolist(),
    }


def _gripper_payload() -> dict[str, Any]:
    return {"minimum": 0.0, "maximum": 1.0, "normalized": [-1.0, 1.0]}


def _normalization_pair(
    payload: Mapping[str, Any], key: str, width: int
) -> tuple[np.ndarray, np.ndarray]:
    values = _mapping(payload.get(key), key)
    center = np.asarray(values.get("center"), dtype=np.float64)
    half = np.asarray(values.get("half_range"), dtype=np.float64)
    if (
        center.shape != (width,)
        or half.shape != (width,)
        or not np.isfinite(center).all()
        or not np.isfinite(half).all()
        or np.any(half <= 0.0)
    ):
        raise M0MobileError(f"{key} normalization is invalid")
    return center, half


def _normalize(
    value: Sequence[float], center: np.ndarray, half: np.ndarray
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != center.shape or not np.isfinite(array).all():
        raise ValueError("normalization input has the wrong shape or non-finite values")
    return tuple(float(component) for component in (array - center) / half)


def _denormalize(
    value: Sequence[float], center: np.ndarray, half: np.ndarray
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != center.shape or not np.isfinite(array).all():
        raise ValueError("denormalization input has the wrong shape or non-finite values")
    return tuple(float(component) for component in array * half + center)


def _relative_asset(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise M0MobileError("joint-trajectory asset paths must be dataset-relative")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise M0MobileError(f"joint-trajectory asset is missing or escapes root: {value}")
    return path


def _copy_episode_asset(
    episode_root: Path, staging: Path, episode_id: str, value: str
) -> str:
    relative = Path(value)
    if relative.is_absolute():
        raise M0MobileError("fresh episode asset paths must be episode-relative")
    source = (episode_root / relative).resolve()
    if episode_root not in source.parents or not source.is_file():
        raise M0MobileError(f"fresh episode asset is missing or escapes root: {value}")
    destination_relative = Path("assets") / episode_id / relative
    destination = staging / destination_relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    elif _sha256(destination) != _sha256(source):
        raise M0MobileError(f"fresh asset collision has different content: {value}")
    return destination_relative.as_posix()


def _safe_path_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise M0MobileError(f"required JSON file is missing: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield _mapping(json.loads(line), "joint-trajectory JSONL row")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be a mapping")
    return value


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _unit_fraction(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    return _bounded_integer(value, 0, 2**63 - 1, name)


def _bounded_integer(value: Any, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within [{minimum}, {maximum}]")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise M0MobileError(f"file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ConveyorVLAJointTrajectoryDataset",
    "FORBIDDEN_MODEL_KEYS",
    "JointTrajectoryNormalizer",
    "MODEL_BATCH_KEYS",
    "PROGRESS_PROVENANCE",
    "TERMINAL_HOLD_REASONS",
    "audit_joint_trajectory_dataset",
    "derive_fresh_joint_trajectory_record",
    "mani_action_from_applied_commands",
    "materialize_fresh_joint_trajectory_dataset",
    "normalization_identity",
    "progress_bucket",
    "validate_applied_control_sample",
    "validate_joint_trajectory_record",
]
