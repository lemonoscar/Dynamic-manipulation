"""Strict, stdlib-only structural validation for published V1 datasets."""

from __future__ import annotations

import json
import math
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .tasking import (
    TASKING_SCHEMA_VERSION,
    CurriculumSplit,
    split_object_ids,
)
from .stationary import (
    stationary_spawn_xy,
    validate_stationary_episode_contract,
)


PROTOCOL_VERSION = "conveyor-bench-v1"
_REQUIRED_STREAMS = (
    "steps.jsonl",
    "objects.jsonl",
    "events.jsonl",
    "action_chunks.jsonl",
)
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_EVENT_KINDS = {
    "episode_start",
    "object_spawned",
    "object_recycled",
    "target_selected",
    "phase_changed",
    "contact_begin",
    "contact_end",
    "grasp_attempt",
    "object_released",
    "object_placed",
    "target_missed",
    "failure",
    "episode_end",
}


@dataclass
class ValidationResult:
    """Machine-readable validation outcome used by the CLI and tests."""

    errors: list[str] = field(default_factory=list)
    run_count: int = 0
    episode_count: int = 0
    sample_count: int = 0
    object_record_count: int = 0
    camera_frame_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, path: Path, message: str, line: int | None = None) -> None:
        location = f"{path}:{line}" if line is not None else str(path)
        self.errors.append(f"{location}: {message}")


@dataclass(frozen=True)
class _EpisodeContext:
    episode_id: str
    run_id: str
    env_id: int
    task_id: str
    task_type: str
    belt_speed_mps: float
    expected_spawn_xy_by_object: Mapping[str, tuple[float, float]]
    robot_mode: str
    registered_ids: tuple[str, ...]
    scored_ids: tuple[str, ...]
    goal_by_object: Mapping[str, str]
    goal_bounds: Mapping[str, tuple[tuple[float, ...], tuple[float, ...]]]
    control_hz: int
    model_hz: int
    future_horizons: tuple[int, ...]
    settled_linear_speed_mps: float
    settled_angular_speed_radps: float
    placement_dwell_s: float
    physics_hz: int
    camera_hz: int
    camera_resolutions: Mapping[str, tuple[int, int]]
    camera_roles: Mapping[str, str]
    chunk_sizes: Mapping[str, int]


def validate_v1_dataset(source: str | Path) -> ValidationResult:
    """Validate one V1 run summary or every V1 summary in an output root."""

    path = Path(source)
    result = ValidationResult()
    if path.is_file():
        output_root = path.parent
        summaries = (path,)
    elif path.is_dir():
        output_root = path
        summaries = tuple(sorted(path.glob("*-summary.json")))
    else:
        result.fail(path, "input must be a run summary JSON file or output root")
        return result

    episodes_root = output_root / "episodes"
    if episodes_root.is_dir():
        for unfinished in sorted(episodes_root.rglob("*.inprogress")):
            result.fail(unfinished, "unfinished .inprogress episode is present")
    if not summaries:
        result.fail(output_root, "no run summary JSON files were found")
        return result

    for summary_path in summaries:
        _validate_run(summary_path, output_root, result)
    return result


def validate_v1_episode(source: str | Path) -> ValidationResult:
    """Strictly validate one atomically published canonical V1 episode."""

    episode_dir = Path(source)
    result = ValidationResult()
    if not episode_dir.is_dir():
        result.fail(episode_dir, "input must be a canonical episode directory")
        return result
    _validate_episode(
        episode_dir,
        run_id=None,
        report=None,
        result=result,
    )
    return result


def validate_v1_run_summary(source: str | Path) -> ValidationResult:
    """Explicit alias for validating one run-summary path."""

    return validate_v1_dataset(source)


def _validate_run(
    summary_path: Path,
    output_root: Path,
    result: ValidationResult,
) -> None:
    summary = _read_json(summary_path, result)
    if summary is None:
        return
    result.run_count += 1
    if summary.get("protocol_version") != PROTOCOL_VERSION:
        result.fail(summary_path, f"protocol_version must be {PROTOCOL_VERSION!r}")
        return

    run_id = summary.get("run_id")
    reports = summary.get("episodes")
    requested = summary.get("requested_episodes")
    successful = summary.get("successful_episodes")
    if not isinstance(run_id, str) or not run_id:
        result.fail(summary_path, "run_id must be a non-empty string")
        return
    if not _is_int(requested) or requested < 0:
        result.fail(summary_path, "requested_episodes must be a non-negative integer")
    if not _is_int(successful) or successful < 0:
        result.fail(
            summary_path,
            "successful_episodes must be a non-negative integer",
        )
    if not _is_sequence(reports):
        result.fail(summary_path, "episodes must be a list")
        return
    if _is_int(requested) and requested != len(reports):
        result.fail(
            summary_path,
            "requested_episodes does not match the run episode count",
        )

    seen_ids: set[str] = set()
    observed_successes = 0
    for index, report in enumerate(reports, start=1):
        if not isinstance(report, Mapping):
            result.fail(summary_path, "episode report must be an object", index)
            continue
        episode_id = report.get("episode_id")
        if (
            not isinstance(episode_id, str)
            or not _SAFE_EPISODE_ID.fullmatch(episode_id)
        ):
            result.fail(summary_path, "episode_id is missing or unsafe", index)
            continue
        if episode_id in seen_ids:
            result.fail(summary_path, f"duplicate episode_id {episode_id!r}", index)
            continue
        seen_ids.add(episode_id)

        report_success = report.get("success")
        failure_reason = report.get("failure_reason")
        if not isinstance(report_success, bool):
            result.fail(summary_path, "episode success must be a bool", index)
        else:
            observed_successes += int(report_success)
        if not isinstance(failure_reason, str) or not failure_reason:
            result.fail(
                summary_path,
                "episode failure_reason must be a non-empty string",
                index,
            )
        elif failure_reason == "runtime_error":
            result.fail(
                summary_path,
                f"episode {episode_id!r} ended with runtime_error",
                index,
            )
        if report_success is True and failure_reason != "none":
            result.fail(
                summary_path,
                "successful report must use failure_reason none",
                index,
            )
        if report_success is False and failure_reason == "none":
            result.fail(summary_path, "failed report requires a failure reason", index)

        episode_dir = _resolve_episode_directory(
            output_root,
            episode_id,
            report.get("path"),
        )
        if episode_dir is None:
            result.fail(
                summary_path,
                f"published episode directory is missing for {episode_id!r}",
                index,
            )
            continue
        _validate_episode(
            episode_dir,
            run_id,
            report,
            result,
        )

    if _is_int(successful) and successful != observed_successes:
        result.fail(
            summary_path,
            "successful_episodes does not match episode reports",
        )


def _resolve_episode_directory(
    output_root: Path,
    episode_id: str,
    recorded_path: Any,
) -> Path | None:
    portable_path = output_root / "episodes" / episode_id
    if portable_path.is_dir():
        return portable_path
    if not isinstance(recorded_path, str) or not recorded_path:
        return None
    candidate = Path(recorded_path)
    if not candidate.is_absolute():
        candidate = output_root / candidate
    return candidate if candidate.is_dir() and candidate.name == episode_id else None


def _validate_episode(
    episode_dir: Path,
    run_id: str | None,
    report: Mapping[str, Any] | None,
    result: ValidationResult,
) -> None:
    result.episode_count += 1
    if episode_dir.name.endswith(".inprogress"):
        result.fail(episode_dir, "episode was not atomically published")
        return

    manifest_path = episode_dir / "manifest.json"
    summary_path = episode_dir / "summary.json"
    manifest = _read_json(manifest_path, result)
    summary = _read_json(summary_path, result)
    streams = {
        name: _read_jsonl(episode_dir / name, result)
        for name in _REQUIRED_STREAMS
    }
    if manifest is None or summary is None or any(
        rows is None for rows in streams.values()
    ):
        return
    steps = streams["steps.jsonl"] or []
    objects = streams["objects.jsonl"] or []
    events = streams["events.jsonl"] or []
    chunks = streams["action_chunks.jsonl"] or []

    context = _parse_manifest(manifest_path, manifest, result)
    if context is None:
        return
    if context.episode_id != episode_dir.name:
        result.fail(manifest_path, "episode_id does not match directory name")
    if run_id is not None and context.run_id != run_id:
        result.fail(manifest_path, "episode run_id does not match run summary")
    if report is not None and report.get("episode_id") != context.episode_id:
        result.fail(manifest_path, "episode_id does not match run report")

    resolved_report = (
        report
        if report is not None
        else {
            "episode_id": context.episode_id,
            "success": summary.get("success"),
            "failure_reason": summary.get("failure_reason"),
            "metrics": summary.get("metrics"),
        }
    )
    _validate_summary(
        summary_path,
        summary,
        context,
        resolved_report,
        len(steps),
        len(objects),
        len(chunks),
        len(events),
        result,
    )
    step_by_id, capture_count = _validate_steps(
        episode_dir,
        steps,
        context,
        result,
    )
    _validate_camera_index(
        episode_dir,
        steps,
        step_by_id,
        capture_count,
        context,
        result,
    )
    states_by_step = _validate_objects(
        episode_dir / "objects.jsonl",
        objects,
        step_by_id,
        context,
        result,
    )
    _validate_chunks(
        episode_dir / "action_chunks.jsonl",
        chunks,
        steps,
        context,
        result,
    )
    _validate_events(
        episode_dir / "events.jsonl",
        events,
        step_by_id,
        summary,
        context,
        result,
    )
    if summary.get("success") is True:
        _validate_success_evidence(
            episode_dir,
            steps,
            states_by_step,
            events,
            summary,
            context,
            result,
        )

    if report is not None:
        reported_frames = report.get("camera_frames", 0)
        if not _is_int(reported_frames) or reported_frames < 0:
            result.fail(
                summary_path,
                "run report camera_frames must be non-negative",
            )
        elif reported_frames != capture_count:
            result.fail(
                summary_path,
                "run report camera_frames does not match recorded captures",
            )
    result.sample_count += len(steps)
    result.object_record_count += len(objects)


def _parse_manifest(
    path: Path,
    manifest: Mapping[str, Any],
    result: ValidationResult,
) -> _EpisodeContext | None:
    episode = manifest.get("episode")
    config = manifest.get("benchmark_config")
    if not isinstance(episode, Mapping) or not isinstance(config, Mapping):
        result.fail(path, "manifest requires episode and benchmark_config objects")
        return None
    if episode.get("protocol_version") != PROTOCOL_VERSION:
        result.fail(path, f"episode protocol_version must be {PROTOCOL_VERSION!r}")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        result.fail(path, f"benchmark protocol_version must be {PROTOCOL_VERSION!r}")

    episode_id = episode.get("episode_id")
    run_id = episode.get("run_id")
    env_id = episode.get("env_id")
    task = episode.get("task")
    if not isinstance(episode_id, str) or not _SAFE_EPISODE_ID.fullmatch(episode_id):
        result.fail(path, "manifest episode_id is missing or unsafe")
        return None
    if not isinstance(run_id, str) or not run_id:
        result.fail(path, "manifest run_id must be non-empty")
        return None
    if not _is_int(env_id) or env_id < 0:
        result.fail(path, "manifest env_id must be non-negative")
        return None
    if not isinstance(task, Mapping):
        result.fail(path, "episode task must be an object")
        return None

    task_id = task.get("task_id")
    task_type = task.get("task_type")
    robot_mode = task.get("robot_mode")
    task_objects = task.get("objects")
    goal_zones = task.get("goal_zones")
    scored_ids = task.get("scored_object_ids")
    if not all(
        isinstance(value, str) and value
        for value in (task_id, task_type, robot_mode)
    ):
        result.fail(path, "task_id, task_type, and robot_mode must be strings")
        return None
    allowed_task_types = {
        "stationary_sort",
        "dynamic_sort",
        "continuous_sort",
    }
    if task_type not in allowed_task_types:
        result.fail(path, "task_type must be a registered V1 sorting task")
        return None
    belt_speed_mps = task.get("belt_speed_mps")
    if not _is_number(belt_speed_mps) or belt_speed_mps < 0.0:
        result.fail(path, "belt_speed_mps must be finite and non-negative")
        return None
    if task_type == "stationary_sort" and not math.isclose(
        float(belt_speed_mps), 0.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        result.fail(path, "stationary_sort requires belt_speed_mps=0")
        return None
    if task_type != "stationary_sort" and belt_speed_mps <= 0.0:
        result.fail(
            path,
            "dynamic and continuous sorting require positive belt_speed_mps",
        )
        return None
    if not _is_sequence(task_objects) or not task_objects:
        result.fail(path, "task objects must be a non-empty list")
        return None
    if not _is_sequence(goal_zones) or not goal_zones:
        result.fail(path, "task goal_zones must be a non-empty list")
        return None
    if not _is_sequence(scored_ids) or not scored_ids:
        result.fail(path, "task scored_object_ids must be a non-empty list")
        return None
    if task_type == "stationary_sort" and len(task_objects) != 1:
        result.fail(path, "stationary_sort requires exactly one object")
        return None
    stationary_scenario = None
    if task_type == "stationary_sort":
        try:
            stationary_scenario = validate_stationary_episode_contract(episode)
        except ValueError as error:
            result.fail(path, f"stationary diagnostic contract: {error}")
            return None

    registered_ids: list[str] = []
    task_asset_ids: list[str] = []
    goal_by_object: dict[str, str] = {}
    for item in task_objects:
        if not isinstance(item, Mapping):
            result.fail(path, "every task object must be an object")
            continue
        instance_id = item.get("instance_id")
        asset_id = item.get("asset_id")
        class_id = item.get("class_id")
        if not all(
            isinstance(value, str) and value
            for value in (instance_id, asset_id, class_id)
        ):
            result.fail(path, "task object IDs must be non-empty strings")
            continue
        registered_ids.append(instance_id)
        task_asset_ids.append(asset_id)
        goal_id = item.get("goal_zone_id")
        if goal_id is not None:
            if not isinstance(goal_id, str) or not goal_id:
                result.fail(path, "goal_zone_id must be a string or null")
            else:
                goal_by_object[instance_id] = goal_id
    if len(registered_ids) != len(set(registered_ids)):
        result.fail(path, "task object instance IDs must be unique")
    _validate_tasking_split(
        path,
        episode,
        task.get("metadata"),
        task_asset_ids,
        result,
    )

    goal_bounds: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for zone in goal_zones:
        if not isinstance(zone, Mapping):
            result.fail(path, "every goal zone must be an object")
            continue
        zone_id = zone.get("zone_id")
        minimum = zone.get("min_xyz")
        maximum = zone.get("max_xyz")
        if not isinstance(zone_id, str) or not zone_id:
            result.fail(path, "goal zone ID must be a non-empty string")
            continue
        if not _is_vector(minimum, 3) or not _is_vector(maximum, 3):
            result.fail(path, f"goal zone {zone_id!r} bounds are invalid")
            continue
        resolved_minimum = tuple(float(value) for value in minimum)
        resolved_maximum = tuple(float(value) for value in maximum)
        if any(
            lower >= upper
            for lower, upper in zip(resolved_minimum, resolved_maximum, strict=True)
        ):
            result.fail(path, f"goal zone {zone_id!r} bounds are empty")
            continue
        goal_bounds[zone_id] = (resolved_minimum, resolved_maximum)
    if len(goal_bounds) != len(goal_zones):
        result.fail(path, "goal zone IDs must be valid and unique")

    resolved_scored = tuple(value for value in scored_ids if isinstance(value, str))
    if len(resolved_scored) != len(scored_ids) or len(set(resolved_scored)) != len(
        resolved_scored
    ):
        result.fail(path, "scored_object_ids must contain unique strings")
    for object_id in resolved_scored:
        if object_id not in registered_ids:
            result.fail(path, f"scored object {object_id!r} is not registered")
        elif goal_by_object.get(object_id) not in goal_bounds:
            result.fail(path, f"scored object {object_id!r} lacks a valid destination")
    if any(
        object_id not in registered_ids
        or goal_by_object.get(object_id) not in goal_bounds
        for object_id in resolved_scored
    ):
        return None
    if (
        task_type in {"stationary_sort", "dynamic_sort"}
        and len(resolved_scored) != 1
    ):
        result.fail(path, f"{task_type} requires exactly one scored object")
        return None

    physics_hz = config.get("physics_hz")
    control_hz = config.get("control_hz")
    camera_hz = config.get("camera_hz")
    model_hz = config.get("model_hz")
    if physics_hz != 400 or camera_hz != 25:
        result.fail(path, "V1 camera cadence requires physics_hz=400 and camera_hz=25")
        return None
    horizons = config.get("future_horizons_steps")
    if control_hz != 50 or model_hz != 25:
        result.fail(path, "V1 dataset cadence requires control_hz=50 and model_hz=25")
        return None
    if (
        not _is_sequence(horizons)
        or not horizons
        or any(not _is_int(value) or value < 0 for value in horizons)
        or list(horizons) != sorted(set(horizons))
    ):
        result.fail(path, "future_horizons_steps must be sorted unique integers")
        return None

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        result.fail(path, "benchmark evaluation config must be an object")
        return None
    thresholds: list[float] = []
    for key in (
        "settled_linear_speed_mps",
        "settled_angular_speed_radps",
        "placement_dwell_s",
    ):
        value = evaluation.get(key)
        if not _is_number(value) or value < 0:
            result.fail(path, f"evaluation {key} must be finite and non-negative")
            return None
        thresholds.append(float(value))
    if thresholds[2] <= 0:
        result.fail(path, "placement_dwell_s must be positive")
        return None

    camera_resolutions: dict[str, tuple[int, int]] = {}
    camera_roles: dict[str, str] = {}
    metadata = episode.get("metadata")
    cameras = metadata.get("cameras") if isinstance(metadata, Mapping) else None
    if cameras is not None:
        if not isinstance(cameras, Mapping):
            result.fail(path, "episode metadata cameras must be an object")
        else:
            for camera_id, specification in cameras.items():
                resolution = (
                    specification.get("resolution")
                    if isinstance(specification, Mapping)
                    else None
                )
                role = (
                    specification.get("role")
                    if isinstance(specification, Mapping)
                    else None
                )
                if (
                    not isinstance(camera_id, str)
                    or not camera_id
                    or not _is_sequence(resolution)
                    or len(resolution) != 2
                    or any(not _is_int(value) or value <= 0 for value in resolution)
                    or role not in {"policy_observation", "observer_only"}
                ):
                    result.fail(
                        path,
                        "camera contracts require positive resolution and a valid role",
                    )
                    continue
                camera_resolutions[camera_id] = (resolution[0], resolution[1])
                camera_roles[camera_id] = role

    chunk_sizes = {
        "m0": config.get("m0_chunk_size"),
        "dynamicvla": config.get("dynamicvla_chunk_size"),
    }
    if any(not _is_int(value) or value <= 0 for value in chunk_sizes.values()):
        result.fail(path, "action chunk sizes must be positive integers")
        return None
    return _EpisodeContext(
        episode_id=episode_id,
        run_id=run_id,
        env_id=env_id,
        task_id=task_id,
        task_type=task_type,
        belt_speed_mps=float(belt_speed_mps),
        expected_spawn_xy_by_object=(
            {registered_ids[0]: stationary_spawn_xy(stationary_scenario)}
            if stationary_scenario is not None
            else {}
        ),
        robot_mode=robot_mode,
        registered_ids=tuple(registered_ids),
        scored_ids=resolved_scored,
        goal_by_object=goal_by_object,
        goal_bounds=goal_bounds,
        control_hz=control_hz,
        model_hz=model_hz,
        future_horizons=tuple(horizons),
        settled_linear_speed_mps=thresholds[0],
        settled_angular_speed_radps=thresholds[1],
        placement_dwell_s=thresholds[2],
        physics_hz=physics_hz,
        camera_hz=camera_hz,
        camera_resolutions=camera_resolutions,
        camera_roles=camera_roles,
        chunk_sizes=chunk_sizes,
    )


def _validate_tasking_split(
    path: Path,
    episode: Mapping[str, Any],
    metadata: Any,
    asset_ids: Sequence[str],
    result: ValidationResult,
) -> None:
    if not isinstance(metadata, Mapping):
        return
    tasking_keys = {"tasking_schema_version", "curriculum_split"}
    if not tasking_keys & metadata.keys():
        return

    if metadata.get("tasking_schema_version") != TASKING_SCHEMA_VERSION:
        result.fail(
            path,
            f"tasking_schema_version must be {TASKING_SCHEMA_VERSION!r}",
        )

    split_value = metadata.get("curriculum_split")
    try:
        split = CurriculumSplit(split_value)
    except (TypeError, ValueError):
        result.fail(
            path,
            "curriculum_split must be train, val, or unseen",
        )
        split = None

    active_asset_ids = metadata.get("active_asset_ids")
    if (
        not _is_sequence(active_asset_ids)
        or any(not isinstance(value, str) for value in active_asset_ids)
        or tuple(active_asset_ids) != tuple(asset_ids)
    ):
        result.fail(
            path,
            "task metadata active_asset_ids do not match task objects",
        )

    if split is None:
        return
    episode_metadata = episode.get("metadata")
    scene = (
        episode_metadata.get("scene_profile")
        if isinstance(episode_metadata, Mapping)
        else None
    )
    if (
        isinstance(scene, Mapping)
        and scene.get("backend") == "isaac_rtx_native_nurec"
    ):
        fixture = scene.get("object_fixture_contract")
        fixture_objects = (
            fixture.get("objects") if isinstance(fixture, Mapping) else None
        )
        profile_ids = (
            tuple(
                item.get("object_id")
                for item in fixture_objects
                if isinstance(item, Mapping)
                and isinstance(item.get("object_id"), str)
            )
            if _is_sequence(fixture_objects)
            else ()
        )
        fixtures_valid = (
            isinstance(fixture, Mapping)
            and fixture.get("all_rigid_bodies_valid") is True
            and fixture.get("all_visuals_composed") is True
            and len(profile_ids) == len(fixture_objects)
            and len(profile_ids) == len(set(profile_ids))
            and bool(profile_ids)
            and all(profile_ids)
        )
        if not fixtures_valid:
            result.fail(path, "V3 object fixture contract is invalid")
            return
        allowed_assets = (
            set(profile_ids) if split is CurriculumSplit.TRAIN else set()
        )
    else:
        try:
            allowed_assets = set(split_object_ids()[split])
        except (OSError, ValueError) as error:
            result.fail(path, f"cannot resolve curriculum asset splits: {error}")
            return
    leaked_assets = sorted(set(asset_ids) - allowed_assets)
    if leaked_assets:
        result.fail(
            path,
            f"task assets violate curriculum split {split.value!r}: "
            f"{leaked_assets}",
        )


def _validate_summary(
    path: Path,
    summary: Mapping[str, Any],
    context: _EpisodeContext,
    report: Mapping[str, Any],
    step_count: int,
    object_count: int,
    chunk_count: int,
    event_count: int,
    result: ValidationResult,
) -> None:
    expected_identity = {
        "episode_id": context.episode_id,
        "task_id": context.task_id,
        "task_type": context.task_type,
        "robot_mode": context.robot_mode,
    }
    for key, expected in expected_identity.items():
        if summary.get(key) != expected:
            result.fail(path, f"{key} does not match manifest")

    expected_counts = {
        "sample_count": step_count,
        "object_record_count": object_count,
        "action_chunk_count": chunk_count,
        "event_count": event_count,
    }
    for key, expected in expected_counts.items():
        value = summary.get(key)
        if not _is_int(value) or value != expected:
            result.fail(path, f"{key} does not match {key.removesuffix('_count')} data")

    success = summary.get("success")
    reason = summary.get("failure_reason")
    status = summary.get("status")
    if not isinstance(success, bool):
        result.fail(path, "success must be a bool")
    if not isinstance(reason, str) or not reason:
        result.fail(path, "failure_reason must be a non-empty string")
    if success is True and (status != "success" or reason != "none"):
        result.fail(path, "successful summary requires status success and reason none")
    if success is False and (status != "failure" or reason == "none"):
        result.fail(path, "failed summary requires status failure and a reason")
    if reason == "runtime_error":
        result.fail(path, "runtime_error episodes fail dataset validation")
    if success != report.get("success") or reason != report.get("failure_reason"):
        result.fail(path, "episode summary outcome does not match run report")
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        result.fail(path, "metrics must be an object")
    else:
        if metrics.get("sample_count") != step_count:
            result.fail(path, "metrics sample_count does not match steps")
        if metrics.get("object_record_count") != object_count:
            result.fail(path, "metrics object_record_count does not match objects")
        if report.get("metrics") != metrics:
            result.fail(path, "episode metrics do not match run report")


def _validate_steps(
    episode_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    context: _EpisodeContext,
    result: ValidationResult,
) -> tuple[dict[int, Mapping[str, Any]], int]:
    path = episode_dir / "steps.jsonl"
    previous_step: int | None = None
    previous_time: float | None = None
    by_step: dict[int, Mapping[str, Any]] = {}
    frame_state: dict[str, tuple[int, float]] = {}
    frame_paths: set[Path] = set()
    capture_count = 0
    repeat_count = context.control_hz // context.model_hz

    for line, row in enumerate(rows, start=1):
        sim_step = row.get("sim_step")
        sim_time = row.get("sim_time_s")
        model_tick = row.get("model_tick")
        if not _is_int(sim_step) or sim_step < 0:
            result.fail(path, "sim_step must be a non-negative integer", line)
            continue
        if sim_step in by_step:
            result.fail(path, "sim_step must be unique", line)
        by_step[sim_step] = row
        if previous_step is not None and sim_step <= previous_step:
            result.fail(path, "sim_step must increase strictly", line)
        previous_step = sim_step
        if not _is_number(sim_time) or sim_time < 0:
            result.fail(path, "sim_time_s must be finite and non-negative", line)
        elif not math.isclose(
            float(sim_time),
            sim_step / context.physics_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            result.fail(
                path,
                "sim_time_s must match the canonical physics clock",
                line,
            )
        elif previous_time is not None and sim_time <= previous_time:
            result.fail(path, "sim_time_s must increase strictly", line)
        else:
            previous_time = float(sim_time)

        expected_tick = (line - 1) // repeat_count
        if model_tick != expected_tick:
            result.fail(
                path,
                f"model_tick must follow the 25/50 cadence; expected {expected_tick}",
                line,
            )
        if row.get("env_id") != context.env_id:
            result.fail(path, "env_id does not match manifest", line)
        if not _valid_pose(row.get("robot_root_world")):
            result.fail(path, "robot_root_world is not a finite pose", line)
        if not _valid_twist(row.get("robot_twist_world")):
            result.fail(path, "robot_twist_world is not a finite twist", line)
        if not _valid_pose(row.get("tcp_base")):
            result.fail(path, "tcp_base is not a finite pose", line)
        if not _valid_joints(row.get("joints")):
            result.fail(path, "joint names/positions/velocities are malformed", line)
        if not _valid_action(row.get("action")):
            result.fail(path, "action must contain ten finite values", line)
        measured_belt_speed = row.get("belt_measured_speed_mps")
        if not _is_number(measured_belt_speed):
            result.fail(
                path,
                "belt_measured_speed_mps must be finite",
                line,
            )
        elif not math.isclose(
            float(measured_belt_speed),
            context.belt_speed_mps,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            result.fail(
                path,
                "belt_measured_speed_mps does not match task belt_speed_mps",
                line,
            )
        if not isinstance(row.get("phase"), str) or not row["phase"]:
            result.fail(path, "phase must be a non-empty string", line)
        if row.get("selected_object_id") not in (None, *context.registered_ids):
            result.fail(path, "selected_object_id is not registered", line)
        for key in ("left_contact_object_ids", "right_contact_object_ids"):
            values = row.get(key)
            if (
                not _is_sequence(values)
                or any(not isinstance(value, str) for value in values)
                or len(values) != len(set(values))
                or any(value not in context.registered_ids for value in values)
            ):
                result.fail(path, f"{key} contains invalid object IDs", line)

        frames = row.get("camera_frames")
        if not _is_sequence(frames):
            result.fail(path, "camera_frames must be a list", line)
            continue
        if frames:
            capture_count += 1
            frame_ids = [
                frame.get("camera_id")
                for frame in frames
                if isinstance(frame, Mapping)
                and isinstance(frame.get("camera_id"), str)
            ]
            if (
                len(frames) != len(context.camera_resolutions)
                or len(frame_ids) != len(frames)
                or set(frame_ids) != set(context.camera_resolutions)
            ):
                result.fail(
                    path,
                    "camera frame IDs must occur exactly once per manifest camera",
                    line,
                )
        for frame in frames:
            if not isinstance(frame, Mapping):
                result.fail(path, "camera frame reference must be an object", line)
                continue
            _validate_camera_frame(
                episode_dir,
                frame,
                sim_time,
                frame_state,
                frame_paths,
                context,
                path,
                line,
                result,
            )
    return by_step, capture_count


def _validate_camera_frame(
    episode_dir: Path,
    frame: Mapping[str, Any],
    step_time: Any,
    frame_state: dict[str, tuple[int, float]],
    frame_paths: set[Path],
    context: _EpisodeContext,
    stream_path: Path,
    line: int,
    result: ValidationResult,
) -> None:
    camera_id = frame.get("camera_id")
    frame_index = frame.get("frame_index")
    capture_time = frame.get("capture_time_s")
    relative_path = frame.get("relative_path")
    if (
        not isinstance(camera_id, str)
        or camera_id not in context.camera_resolutions
        or not _is_int(frame_index)
        or frame_index < 0
        or not _is_number(capture_time)
        or capture_time < 0
    ):
        result.fail(stream_path, "camera frame identity/time is invalid", line)
        return
    if _is_number(step_time) and not math.isclose(
        float(capture_time), float(step_time), rel_tol=0.0, abs_tol=1.0e-6
    ):
        result.fail(stream_path, "camera capture_time_s does not match step", line)

    previous = frame_state.get(camera_id)
    if previous is None and frame_index != 0:
        result.fail(stream_path, "camera frame indices must start at zero", line)
    elif previous is not None:
        if frame_index != previous[0] + 1:
            result.fail(stream_path, "camera frame indices must be consecutive", line)
        if float(capture_time) <= previous[1]:
            result.fail(
                stream_path,
                "camera capture times must increase strictly",
                line,
            )
    frame_state[camera_id] = (frame_index, float(capture_time))

    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
    ):
        result.fail(stream_path, "camera relative_path must be a safe PNG path", line)
        return
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.suffix.lower() != ".png"
    ):
        result.fail(stream_path, "camera relative_path must be a safe PNG path", line)
        return
    frame_path = (episode_dir / Path(*relative.parts)).resolve()
    if not frame_path.is_relative_to(episode_dir.resolve()):
        result.fail(stream_path, "camera path escapes the episode directory", line)
        return
    if frame_path in frame_paths:
        result.fail(stream_path, "camera PNG path is referenced more than once", line)
    frame_paths.add(frame_path)
    dimensions, error = _read_png_dimensions(frame_path)
    if error is not None:
        result.fail(frame_path, error)
        return
    if dimensions != context.camera_resolutions[camera_id]:
        result.fail(
            frame_path,
            f"PNG dimensions {dimensions} do not match "
            f"{context.camera_resolutions[camera_id]}",
        )
    result.camera_frame_count += 1


def _validate_camera_index(
    episode_dir: Path,
    steps: Sequence[Mapping[str, Any]],
    step_by_id: Mapping[int, Mapping[str, Any]],
    capture_count: int,
    context: _EpisodeContext,
    result: ValidationResult,
) -> None:
    """Validate the camera index as an exact, clocked mirror of step refs."""

    path = episode_dir / "camera_frames.jsonl"
    if not path.is_file():
        if capture_count:
            result.fail(path, "camera captures require a camera_frames.jsonl index")
        return
    rows = _read_jsonl(path, result)
    if rows is None:
        return
    if not context.camera_resolutions:
        if rows:
            result.fail(
                path,
                "camera index is present without a manifest camera contract",
            )
        return

    captured_steps = {
        sim_step
        for sim_step, step in step_by_id.items()
        if _is_sequence(step.get("camera_frames"))
        and bool(step["camera_frames"])
    }
    indexed_steps: set[int] = set()
    indexed_ticks: list[int] = []
    previous_step: int | None = None
    previous_time: float | None = None
    stride = context.physics_hz // context.camera_hz
    period_s = 1.0 / context.camera_hz

    for line, row in enumerate(rows, start=1):
        frame_index = row.get("frame_index")
        sim_step = row.get("sim_step")
        capture_time = row.get("capture_time_s")
        if not _is_int(frame_index) or frame_index != line - 1:
            result.fail(
                path,
                "camera index frame_index must be contiguous from zero",
                line,
            )
        if (
            not _is_int(sim_step)
            or sim_step < 0
            or not _is_number(capture_time)
            or capture_time < 0
        ):
            result.fail(
                path,
                "camera index sim_step/time must be finite and non-negative",
                line,
            )
            continue
        resolved_time = float(capture_time)
        if sim_step in indexed_steps:
            result.fail(path, "camera index references a step more than once", line)
        indexed_steps.add(sim_step)
        if not math.isclose(
            resolved_time,
            sim_step / context.physics_hz,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            result.fail(
                path,
                "camera index capture_time_s must match the physics clock",
                line,
            )
        if previous_step is not None:
            if sim_step - previous_step != stride:
                result.fail(
                    path,
                    f"camera index cadence must be exactly {stride} physics steps",
                    line,
                )
            if not math.isclose(
                resolved_time - previous_time,
                period_s,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                result.fail(
                    path,
                    f"camera index cadence must be exactly {period_s:.2f}s",
                    line,
                )
        previous_step = sim_step
        previous_time = resolved_time

        step = step_by_id.get(sim_step)
        if step is None:
            result.fail(path, "camera index row has no matching step", line)
            continue
        refs = step.get("camera_frames")
        if not _is_sequence(refs) or not refs:
            result.fail(
                path,
                "camera index row references a step without a capture",
                line,
            )
            continue
        if not _same_number(capture_time, step.get("sim_time_s")):
            result.fail(
                path,
                "camera index capture_time_s does not match step sim_time_s",
                line,
            )
        model_tick = step.get("model_tick")
        if _is_int(model_tick):
            indexed_ticks.append(model_tick)

        refs_by_id = {
            ref.get("camera_id"): ref
            for ref in refs
            if isinstance(ref, Mapping)
            and isinstance(ref.get("camera_id"), str)
        }
        if len(refs) != len(context.camera_resolutions) or set(
            refs_by_id
        ) != set(context.camera_resolutions):
            result.fail(
                path,
                "indexed step must contain exactly one reference per camera",
                line,
            )

        entries = row.get("frames")
        if not isinstance(entries, Mapping) or set(entries) != set(
            context.camera_resolutions
        ):
            result.fail(
                path,
                "camera index frames must exactly match the manifest contract",
                line,
            )
            continue
        for camera_id in context.camera_resolutions:
            entry = entries[camera_id]
            ref = refs_by_id.get(camera_id)
            if not isinstance(entry, Mapping) or not isinstance(ref, Mapping):
                result.fail(path, f"camera index entry {camera_id!r} is invalid", line)
                continue
            if (
                ref.get("frame_index") != frame_index
                or not _same_number(ref.get("capture_time_s"), capture_time)
                or entry.get("relative_path") != ref.get("relative_path")
            ):
                result.fail(
                    path,
                    f"camera index entry {camera_id!r} disagrees with step reference",
                    line,
                )
            resolution = entry.get("resolution")
            if (
                not _is_sequence(resolution)
                or len(resolution) != 2
                or any(not _is_int(value) or value <= 0 for value in resolution)
                or tuple(resolution) != context.camera_resolutions[camera_id]
            ):
                result.fail(
                    path,
                    f"camera index entry {camera_id!r} resolution "
                    "disagrees with manifest",
                    line,
                )
            if entry.get("role") != context.camera_roles[camera_id]:
                result.fail(
                    path,
                    f"camera index entry {camera_id!r} role disagrees with manifest",
                    line,
                )

    if len(rows) != capture_count or indexed_steps != captured_steps:
        missing = sorted(captured_steps - indexed_steps)
        extra = sorted(indexed_steps - captured_steps)
        result.fail(
            path,
            "camera index and captured steps must have a one-to-one mapping; "
            f"missing={missing}, extra={extra}",
        )

    if rows:
        if len(indexed_ticks) != len(set(indexed_ticks)):
            result.fail(path, "each model tick may have at most one camera capture")
        tick_counts: dict[int, int] = {}
        for step in steps:
            model_tick = step.get("model_tick")
            if _is_int(model_tick):
                tick_counts[model_tick] = tick_counts.get(model_tick, 0) + 1
        repeat_count = context.control_hz // context.model_hz
        complete_ticks = {
            tick for tick, count in tick_counts.items() if count == repeat_count
        }
        captured_ticks = set(indexed_ticks)
        missing_ticks = sorted(complete_ticks - captured_ticks)
        if missing_ticks:
            result.fail(
                path,
                f"complete model ticks lack camera captures: {missing_ticks}",
            )
        extra_ticks = captured_ticks - complete_ticks
        final_tick = max(tick_counts, default=-1)
        if extra_ticks and not (
            extra_ticks == {final_tick} and tick_counts.get(final_tick) == 1
        ):
            result.fail(path, "camera capture occurs on a non-final partial model tick")


def _validate_objects(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    step_by_id: Mapping[int, Mapping[str, Any]],
    context: _EpisodeContext,
    result: ValidationResult,
) -> dict[int, dict[str, Mapping[str, Any]]]:
    states_by_step: dict[int, dict[str, Mapping[str, Any]]] = {
        sim_step: {} for sim_step in step_by_id
    }
    object_records: list[
        tuple[int, Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for line, row in enumerate(rows, start=1):
        sim_step = row.get("sim_step")
        state = row.get("state")
        if not _is_int(sim_step) or sim_step not in step_by_id:
            result.fail(path, "object row references an unknown sim_step", line)
            continue
        if not isinstance(state, Mapping):
            result.fail(path, "object row state must be an object", line)
            continue
        instance_id = state.get("instance_id")
        if instance_id not in context.registered_ids:
            result.fail(path, "object state instance_id is not registered", line)
            continue
        step = step_by_id[sim_step]
        if (
            row.get("model_tick") != step.get("model_tick")
            or row.get("env_id") != step.get("env_id")
            or not _same_number(row.get("sim_time_s"), step.get("sim_time_s"))
        ):
            result.fail(path, "object row timing does not match its step", line)
        if instance_id in states_by_step[sim_step]:
            result.fail(path, "duplicate object record for one step", line)
        states_by_step[sim_step][instance_id] = state
        object_records.append((line, row, state))

        if not _valid_pose(state.get("pose_world")):
            result.fail(path, "object pose_world is malformed", line)
        if not _valid_twist(state.get("twist_world")):
            result.fail(path, "object twist_world is malformed", line)
        if any(
            not isinstance(state.get(key), bool)
            for key in ("active", "in_gripper", "crossed_exit")
        ):
            result.fail(path, "object flags must be bool values", line)
        _validate_future_labels(
            path,
            line,
            row.get("future_object_states"),
            instance_id,
            state,
            context,
            result,
        )

    expected = set(context.registered_ids)
    for sim_step, states in states_by_step.items():
        actual = set(states)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            result.fail(
                path,
                f"sim_step {sim_step} must have one record per object; "
                f"missing={missing}, extra={extra}",
            )

    representatives = _model_tick_object_representatives(
        step_by_id,
        states_by_step,
    )
    for line, row, state in object_records:
        _validate_realized_future_values(
            path,
            line,
            row.get("future_object_states"),
            state,
            row.get("model_tick"),
            representatives,
            result,
        )
    return states_by_step


def _validate_future_labels(
    path: Path,
    line: int,
    labels: Any,
    instance_id: str,
    state: Mapping[str, Any],
    context: _EpisodeContext,
    result: ValidationResult,
) -> None:
    if not _is_sequence(labels):
        result.fail(path, "future_object_states must be a list", line)
        return
    horizons: list[int] = []
    for label in labels:
        if not isinstance(label, Mapping):
            result.fail(path, "future label must be an object", line)
            continue
        horizon = label.get("horizon_steps")
        if label.get("instance_id") != instance_id or not _is_int(horizon):
            result.fail(path, "future label identity/horizon is invalid", line)
            continue
        horizons.append(horizon)
        valid = label.get("valid")
        if valid is True:
            if (
                not _valid_pose(label.get("pose_world"))
                or not _valid_twist(label.get("twist_world"))
                or label.get("invalid_reason") is not None
            ):
                result.fail(path, "valid future label has malformed state", line)
            if horizon == 0 and (
                label.get("pose_world") != state.get("pose_world")
                or label.get("twist_world") != state.get("twist_world")
            ):
                result.fail(
                    path,
                    "valid horizon-0 label must equal current state",
                    line,
                )
        elif valid is False:
            if (
                label.get("pose_world") is not None
                or label.get("twist_world") is not None
                or not isinstance(label.get("invalid_reason"), str)
                or not label["invalid_reason"]
            ):
                result.fail(path, "invalid future label requires only a reason", line)
        else:
            result.fail(path, "future label valid must be a bool", line)
    if tuple(sorted(horizons)) != context.future_horizons or len(horizons) != len(
        set(horizons)
    ):
        result.fail(
            path,
            f"future horizons must be exactly {context.future_horizons}",
            line,
        )


def _model_tick_object_representatives(
    step_by_id: Mapping[int, Mapping[str, Any]],
    states_by_step: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[int, Mapping[str, Mapping[str, Any]]]:
    """Select the latest control sample for every recorded model tick."""

    representative_step_by_tick: dict[int, int] = {}
    for sim_step, step in step_by_id.items():
        model_tick = step.get("model_tick")
        if not _is_int(model_tick):
            continue
        previous = representative_step_by_tick.get(model_tick)
        if previous is None or sim_step > previous:
            representative_step_by_tick[model_tick] = sim_step
    return {
        model_tick: states_by_step.get(sim_step, {})
        for model_tick, sim_step in representative_step_by_tick.items()
    }


def _validate_realized_future_values(
    path: Path,
    line: int,
    labels: Any,
    source_state: Mapping[str, Any],
    source_tick: Any,
    representatives: Mapping[
        int,
        Mapping[str, Mapping[str, Any]],
    ],
    result: ValidationResult,
) -> None:
    """Cross-check every valid label against the recorded future state."""

    if not _is_sequence(labels) or not _is_int(source_tick):
        return
    instance_id = source_state.get("instance_id")
    source_active = source_state.get("active")
    if not isinstance(instance_id, str) or not isinstance(source_active, bool):
        return

    for label in labels:
        if (
            not isinstance(label, Mapping)
            or label.get("instance_id") != instance_id
            or not _is_int(label.get("horizon_steps"))
            or label.get("valid") is not True
            or not _valid_pose(label.get("pose_world"))
            or not _valid_twist(label.get("twist_world"))
        ):
            continue
        horizon = label["horizon_steps"]
        target_tick = source_tick + horizon
        if not source_active:
            result.fail(
                path,
                "valid future label cannot originate from an inactive object",
                line,
            )
            continue

        if horizon == 0:
            realized_state = source_state
        else:
            future_states = representatives.get(target_tick)
            if future_states is None:
                result.fail(
                    path,
                    f"valid future label targets unavailable model_tick "
                    f"{target_tick}",
                    line,
                )
                continue
            realized_state = future_states.get(instance_id)
            if realized_state is None:
                result.fail(
                    path,
                    f"valid future label has no object state at model_tick "
                    f"{target_tick}",
                    line,
                )
                continue
            future_active = realized_state.get("active")
            if future_active is False:
                result.fail(
                    path,
                    f"valid future label targets inactive object at model_tick "
                    f"{target_tick}",
                    line,
                )
                continue
            if future_active is not True:
                continue

        if not _same_pose(
            label.get("pose_world"),
            realized_state.get("pose_world"),
        ):
            result.fail(
                path,
                f"valid future pose does not match realized object state at "
                f"model_tick {target_tick}",
                line,
            )
        if not _same_twist(
            label.get("twist_world"),
            realized_state.get("twist_world"),
        ):
            result.fail(
                path,
                f"valid future twist does not match realized object state at "
                f"model_tick {target_tick}",
                line,
            )


def _validate_chunks(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    context: _EpisodeContext,
    result: ValidationResult,
) -> None:
    by_id: dict[str, Mapping[str, Any]] = {}
    for line, chunk in enumerate(rows, start=1):
        chunk_id = chunk.get("chunk_id")
        profile = chunk.get("profile")
        actions = chunk.get("actions")
        if not isinstance(chunk_id, str) or not chunk_id:
            result.fail(path, "action chunk requires chunk_id", line)
            continue
        if chunk_id in by_id:
            result.fail(path, f"duplicate action chunk {chunk_id!r}", line)
        by_id[chunk_id] = chunk
        if not isinstance(profile, str) or profile not in context.chunk_sizes:
            result.fail(path, "action chunk profile is unsupported", line)
            continue
        if not _is_sequence(actions) or len(actions) != context.chunk_sizes[profile]:
            result.fail(path, "action chunk has the wrong action count", line)
            continue
        if any(not _valid_action(action) for action in actions):
            result.fail(path, "action chunk contains a malformed action", line)
        valid_from = chunk.get("valid_from_tick")
        valid_until = chunk.get("valid_until_tick")
        source_tick = chunk.get("source_observation_tick")
        source_time = chunk.get("source_observation_time_s")
        execute_from = chunk.get("execute_from_tick")
        execute_until = chunk.get("execute_until_tick")
        discarded = chunk.get("discarded_action_count")
        if (
            not _is_int(source_tick)
            or source_tick < 0
            or not _is_number(source_time)
            or source_time < 0
            or not all(
                _is_int(value) and value >= 0
                for value in (valid_from, valid_until, discarded)
            )
        ):
            result.fail(path, "action chunk window/count fields are invalid", line)
            continue
        if source_tick > valid_from:
            result.fail(path, "action chunk source tick follows its valid window", line)
        if valid_until - valid_from != len(actions):
            result.fail(path, "valid action window must equal actions length", line)
        executed = 0
        if execute_from is None and execute_until is None:
            executed = 0
        elif (
            _is_int(execute_from)
            and _is_int(execute_until)
            and valid_from <= execute_from < execute_until <= valid_until
        ):
            executed = execute_until - execute_from
        else:
            result.fail(path, "execute window is invalid", line)
        if executed + discarded != len(actions):
            result.fail(
                path,
                "executed and discarded actions do not account for chunk",
                line,
            )
        stale = chunk.get("stale")
        discard_reason = chunk.get("discard_reason")
        if not isinstance(stale, bool):
            result.fail(path, "action chunk stale must be a bool", line)
        if discarded > 0 and (
            not isinstance(discard_reason, str) or not discard_reason
        ):
            result.fail(path, "discarded actions require a reason", line)
        if discarded == 0 and discard_reason is not None:
            result.fail(path, "discard_reason requires discarded actions", line)
        if stale is True and discarded == 0:
            result.fail(path, "stale action chunk must discard actions", line)

    for line, step in enumerate(steps, start=1):
        chunk_id = step.get("action_chunk_id")
        action_index = step.get("action_index_in_chunk")
        if chunk_id is None and action_index is None:
            continue
        if not isinstance(chunk_id, str) or not _is_int(action_index):
            result.fail(path, "step action chunk reference is incomplete", line)
            continue
        chunk = by_id.get(chunk_id)
        if chunk is None:
            result.fail(
                path,
                f"step references unknown action chunk {chunk_id!r}",
                line,
            )
            continue
        actions = chunk.get("actions")
        if not _is_sequence(actions) or not 0 <= action_index < len(actions):
            result.fail(path, "step action_index_in_chunk is out of range", line)
            continue
        if step.get("model_tick") != chunk.get("valid_from_tick") + action_index:
            result.fail(path, "step model_tick disagrees with action chunk index", line)
        if step.get("action") != actions[action_index]:
            result.fail(
                path,
                "step action disagrees with referenced chunk action",
                line,
            )


def _validate_events(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    step_by_id: Mapping[int, Mapping[str, Any]],
    summary: Mapping[str, Any],
    context: _EpisodeContext,
    result: ValidationResult,
) -> None:
    previous_time: float | None = None
    end_events: list[Mapping[str, Any]] = []
    stationary_spawned_ids: set[str] = set()
    for line, event in enumerate(rows, start=1):
        kind = event.get("kind")
        time_s = event.get("time_s")
        if not isinstance(kind, str) or kind not in _EVENT_KINDS:
            result.fail(path, "event kind is unsupported", line)
        if not _is_number(time_s) or time_s < 0:
            result.fail(path, "event time_s must be finite and non-negative", line)
        elif previous_time is not None and time_s < previous_time:
            result.fail(path, "event time_s cannot decrease", line)
        else:
            previous_time = float(time_s)
        if event.get("object_instance_id") not in (None, *context.registered_ids):
            result.fail(path, "event object identity is not registered", line)
        if event.get("goal_zone_id") not in (None, *context.goal_bounds):
            result.fail(path, "event goal zone is not registered", line)
        sim_step = event.get("sim_step")
        if sim_step is not None and (not _is_int(sim_step) or sim_step < 0):
            result.fail(path, "event sim_step must be non-negative or null", line)
        elif sim_step is not None and _is_number(time_s):
            matching_step = step_by_id.get(sim_step)
            expected_time = (
                matching_step.get("sim_time_s")
                if matching_step is not None
                else sim_step / context.physics_hz
            )
            if not _same_number(time_s, expected_time):
                result.fail(
                    path,
                    "event time_s does not match its sim_step clock",
                    line,
                )
        event_payload = event.get("payload", {})
        if not isinstance(event_payload, Mapping):
            result.fail(path, "event payload must be an object", line)
        if (
            kind == "object_spawned"
            and event.get("object_instance_id")
            in context.expected_spawn_xy_by_object
        ):
            object_id = event["object_instance_id"]
            spawn_xyz = (
                event_payload.get("spawn_xyz")
                if isinstance(event_payload, Mapping)
                else None
            )
            expected_xy = context.expected_spawn_xy_by_object[object_id]
            if object_id in stationary_spawned_ids:
                result.fail(path, "stationary object spawned more than once", line)
            stationary_spawned_ids.add(object_id)
            if (
                not _is_vector(spawn_xyz, 3)
                or any(
                    not math.isclose(
                        float(actual),
                        expected,
                        rel_tol=0.0,
                        abs_tol=1.0e-9,
                    )
                    for actual, expected in zip(
                        spawn_xyz[:2], expected_xy, strict=True
                    )
                )
            ):
                result.fail(
                    path,
                    "stationary object_spawned position does not match registry",
                    line,
                )
        if kind == "episode_end":
            end_events.append(event)
    if stationary_spawned_ids != set(context.expected_spawn_xy_by_object):
        result.fail(path, "stationary object_spawned evidence is incomplete")
    if len(end_events) != 1 or not rows or rows[-1].get("kind") != "episode_end":
        result.fail(path, "events must end with exactly one episode_end")
        return
    payload = end_events[0].get("payload")
    if not isinstance(payload, Mapping):
        result.fail(path, "episode_end payload must be an object")
    elif (
        payload.get("success") != summary.get("success")
        or payload.get("failure_reason") != summary.get("failure_reason")
    ):
        result.fail(path, "episode_end outcome does not match summary")


def _validate_success_evidence(
    episode_dir: Path,
    steps: Sequence[Mapping[str, Any]],
    states_by_step: Mapping[int, Mapping[str, Mapping[str, Any]]],
    events: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    context: _EpisodeContext,
    result: ValidationResult,
) -> None:
    path = episode_dir / "summary.json"
    step_times = {
        step["sim_step"]: float(step["sim_time_s"])
        for step in steps
        if _is_int(step.get("sim_step")) and _is_number(step.get("sim_time_s"))
    }
    wrong_object_held = any(
        state.get("in_gripper") is True
        for states in states_by_step.values()
        for object_id, state in states.items()
        if object_id not in context.scored_ids
    )
    if wrong_object_held:
        result.fail(path, "success contradicts wrong-object grasp evidence")

    completion_times: dict[str, float] = {}
    for target_id in context.scored_ids:
        ever_held = False
        was_held = False
        released = False
        dwell_start: float | None = None
        goal_id = context.goal_by_object[target_id]
        bounds = context.goal_bounds[goal_id]
        for sim_step in step_times:
            state = states_by_step.get(sim_step, {}).get(target_id)
            if state is None:
                continue
            held = state.get("in_gripper") is True
            if held:
                ever_held = True
                if released:
                    released = False
                    dwell_start = None
            elif was_held and ever_held:
                released = True
                dwell_start = None
            was_held = held

            eligible = (
                state.get("active") is True
                and released
                and not held
                and _pose_inside(state.get("pose_world"), bounds)
                and _settled(state.get("twist_world"), context)
            )
            if eligible:
                if dwell_start is None:
                    dwell_start = step_times[sim_step]
                if (
                    step_times[sim_step] - dwell_start
                    >= context.placement_dwell_s
                ):
                    completion_times.setdefault(
                        target_id,
                        step_times[sim_step],
                    )
            elif target_id not in completion_times:
                dwell_start = None
        if not ever_held:
            result.fail(path, f"success lacks correct-object grasp for {target_id!r}")
        if target_id not in completion_times:
            result.fail(
                path,
                f"success lacks correct-zone settled dwell for {target_id!r}",
            )

    placed_pairs = {
        (event.get("object_instance_id"), event.get("goal_zone_id"))
        for event in events
        if event.get("kind") == "object_placed"
        and isinstance(event.get("object_instance_id"), str)
        and isinstance(event.get("goal_zone_id"), str)
    }
    for target_id in context.scored_ids:
        expected_pair = (target_id, context.goal_by_object[target_id])
        if expected_pair not in placed_pairs:
            result.fail(
                episode_dir / "events.jsonl",
                f"success lacks correct object_placed event {expected_pair}",
            )

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        return
    if metrics.get("completed_object_count") != len(context.scored_ids):
        result.fail(path, "success metrics completed_object_count is inconsistent")
    if metrics.get("scored_object_count") != len(context.scored_ids):
        result.fail(path, "success metrics scored_object_count is inconsistent")
    rate = metrics.get("correct_sort_rate")
    if not _is_number(rate) or not math.isclose(
        float(rate), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        result.fail(path, "success metrics correct_sort_rate must be one")
    reported_completion = metrics.get("completion_time_s")
    expected_completion = (
        max(completion_times.values())
        if len(completion_times) == len(context.scored_ids)
        else None
    )
    if (
        expected_completion is None
        or not _is_number(reported_completion)
        or not math.isclose(
            float(reported_completion),
            expected_completion,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        result.fail(path, "success metrics completion_time_s is inconsistent")
    outcomes = metrics.get("object_outcomes")
    if not isinstance(outcomes, Mapping):
        result.fail(path, "success metrics require object_outcomes")
        return
    for target_id in context.scored_ids:
        outcome = outcomes.get(target_id)
        if (
            not isinstance(outcome, Mapping)
            or outcome.get("status") != "sorted_correct"
            or outcome.get("goal_zone_id") != context.goal_by_object[target_id]
            or not _is_number(outcome.get("completion_time_s"))
        ):
            result.fail(path, f"success outcome for {target_id!r} is inconsistent")
        elif target_id not in completion_times or not math.isclose(
            float(outcome["completion_time_s"]),
            completion_times.get(target_id, math.inf),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            result.fail(
                path,
                f"success outcome time for {target_id!r} is inconsistent",
            )


def _pose_inside(
    pose: Any,
    bounds: tuple[tuple[float, ...], tuple[float, ...]],
) -> bool:
    if not _valid_pose(pose):
        return False
    minimum, maximum = bounds
    return all(
        lower <= float(value) <= upper
        for value, lower, upper in zip(
            pose["xyz"],
            minimum,
            maximum,
            strict=True,
        )
    )


def _settled(twist: Any, context: _EpisodeContext) -> bool:
    if not _valid_twist(twist):
        return False
    linear_speed = math.sqrt(
        sum(float(value) ** 2 for value in twist["linear_xyz"])
    )
    angular_speed = math.sqrt(
        sum(float(value) ** 2 for value in twist["angular_xyz"])
    )
    return (
        linear_speed <= context.settled_linear_speed_mps
        and angular_speed <= context.settled_angular_speed_radps
    )


def _read_png_dimensions(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    if not path.is_file():
        return None, "referenced camera PNG is missing"
    try:
        with path.open("rb") as stream:
            header = stream.read(33)
    except OSError as error:
        return None, f"cannot read camera PNG: {error}"
    if len(header) != 33 or header[:8] != _PNG_SIGNATURE:
        return None, "camera file is not a decodable PNG header"
    chunk_length = struct.unpack(">I", header[8:12])[0]
    if chunk_length != 13 or header[12:16] != b"IHDR":
        return None, "PNG must begin with a 13-byte IHDR chunk"
    expected_crc = struct.unpack(">I", header[29:33])[0]
    actual_crc = zlib.crc32(header[12:29]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        return None, "PNG IHDR checksum is invalid"
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        return None, "PNG dimensions must be positive"
    return (width, height), None


def _read_json(path: Path, result: ValidationResult) -> dict[str, Any] | None:
    if not path.is_file():
        result.fail(path, "required JSON file is missing")
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result.fail(path, f"cannot read strict JSON: {error}")
        return None
    if not isinstance(value, dict):
        result.fail(path, "top-level JSON value must be an object")
        return None
    return value


def _read_jsonl(
    path: Path,
    result: ValidationResult,
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        result.fail(path, "required JSONL stream is missing")
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line, raw in enumerate(stream, start=1):
                if not raw.strip():
                    result.fail(path, "blank JSONL row", line)
                    continue
                try:
                    value = json.loads(
                        raw,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    result.fail(path, f"invalid strict JSON row: {error}", line)
                    continue
                if not isinstance(value, dict):
                    result.fail(path, "JSONL row must be an object", line)
                    continue
                rows.append(value)
    except (OSError, UnicodeError) as error:
        result.fail(path, f"cannot read JSONL stream: {error}")
        return None
    return rows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_vector(value: Any, size: int) -> bool:
    return (
        _is_sequence(value)
        and len(value) == size
        and all(_is_number(component) for component in value)
    )


def _valid_pose(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_vector(value.get("xyz"), 3)
        and _is_vector(value.get("wxyz"), 4)
    )


def _valid_twist(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_vector(value.get("linear_xyz"), 3)
        and _is_vector(value.get("angular_xyz"), 3)
    )


def _same_pose(left: Any, right: Any) -> bool:
    return (
        _valid_pose(left)
        and _valid_pose(right)
        and _same_vector(left["xyz"], right["xyz"], 3)
        and _same_vector(left["wxyz"], right["wxyz"], 4)
    )


def _same_twist(left: Any, right: Any) -> bool:
    return (
        _valid_twist(left)
        and _valid_twist(right)
        and _same_vector(left["linear_xyz"], right["linear_xyz"], 3)
        and _same_vector(left["angular_xyz"], right["angular_xyz"], 3)
    )


def _same_vector(left: Any, right: Any, size: int) -> bool:
    return (
        _is_vector(left, size)
        and _is_vector(right, size)
        and all(
            _same_number(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    )


def _valid_joints(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    names = value.get("names")
    return (
        _is_sequence(names)
        and bool(names)
        and all(isinstance(name, str) and name for name in names)
        and len(names) == len(set(names))
        and _is_vector(value.get("positions"), len(names))
        and _is_vector(value.get("velocities"), len(names))
    )


def _valid_action(value: Any) -> bool:
    return isinstance(value, Mapping) and _is_vector(value.get("values"), 10)


def _same_number(left: Any, right: Any) -> bool:
    return _is_number(left) and _is_number(right) and math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )


__all__ = [
    "ValidationResult",
    "validate_v1_dataset",
    "validate_v1_episode",
    "validate_v1_run_summary",
]
