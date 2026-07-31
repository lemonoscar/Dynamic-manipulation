#!/usr/bin/env python3
"""Validate ConveyorBench run summaries and episode artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_KNOWN_CAMERA_NAMES = ("head_rgb", "wrist_rgb", "overview_rgb")


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    run_count: int = 0
    episode_count: int = 0
    sample_count: int = 0
    video_frame_count: int = 0

    def fail(self, path: Path, message: str) -> None:
        self.errors.append(f"{path}: {message}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_json(path: Path, result: ValidationResult) -> dict[str, Any] | None:
    if not path.is_file():
        result.fail(path, "required JSON file is missing")
        return None
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        result.fail(path, f"cannot read JSON: {error}")
        return None
    if not isinstance(value, dict):
        result.fail(path, "top-level JSON value must be an object")
        return None
    return value


def _read_jsonl(path: Path, result: ValidationResult) -> list[dict[str, Any]] | None:
    if not path.is_file():
        result.fail(path, "required JSONL file is missing")
        return None

    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    result.fail(path, f"line {line_number} is blank")
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    result.fail(path, f"line {line_number} is invalid JSON: {error}")
                    continue
                if not isinstance(value, dict):
                    result.fail(path, f"line {line_number} must contain a JSON object")
                    continue
                rows.append(value)
    except (OSError, UnicodeError) as error:
        result.fail(path, f"cannot read JSONL: {error}")
        return None
    return rows


def _require_string(
    mapping: dict[str, Any],
    key: str,
    path: Path,
    result: ValidationResult,
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        result.fail(path, f"{key!r} must be a non-empty string")
        return None
    return value


def _require_nonnegative_int(
    mapping: dict[str, Any],
    key: str,
    path: Path,
    result: ValidationResult,
) -> int | None:
    value = mapping.get(key)
    if not _is_int(value) or value < 0:
        result.fail(path, f"{key!r} must be a non-negative integer")
        return None
    return value


def _resolve_transport_geometry(
    task: dict[str, Any],
    path: Path,
    result: ValidationResult,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Validate new oriented geometry while accepting legacy +X manifests."""

    direction = task.get("transport_direction_xyz")
    exit_point = task.get("exit_plane_point_xyz")
    has_direction = direction is not None
    has_exit_point = exit_point is not None
    if has_direction != has_exit_point:
        result.fail(
            path,
            "transport_direction_xyz and exit_plane_point_xyz must appear together",
        )
        return None

    if not has_direction:
        exit_x = task.get("exit_x_m")
        if not _is_number(exit_x):
            result.fail(path, "task lacks valid conveyor exit geometry")
            return None
        return (1.0, 0.0, 0.0), (float(exit_x), 0.0, 0.0)

    if (
        not isinstance(direction, list)
        or len(direction) != 3
        or not all(_is_number(value) for value in direction)
    ):
        result.fail(path, "transport_direction_xyz must be three finite numbers")
        return None
    if (
        not isinstance(exit_point, list)
        or len(exit_point) != 3
        or not all(_is_number(value) for value in exit_point)
    ):
        result.fail(path, "exit_plane_point_xyz must be three finite numbers")
        return None

    resolved_direction = tuple(float(value) for value in direction)
    norm = math.sqrt(sum(value * value for value in resolved_direction))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
        result.fail(path, "transport_direction_xyz must be a unit vector")
        return None
    return resolved_direction, tuple(float(value) for value in exit_point)


def _validate_exit_flags(
    path: Path,
    steps: list[dict[str, Any]],
    direction: tuple[float, float, float],
    exit_point: tuple[float, float, float],
    result: ValidationResult,
) -> None:
    for row_number, step in enumerate(steps, start=1):
        object_xyz = step.get("object_xyz")
        crossed = step.get("target_crossed_exit")
        if (
            not isinstance(object_xyz, list)
            or len(object_xyz) != 3
            or not all(_is_number(value) for value in object_xyz)
            or not isinstance(crossed, bool)
        ):
            continue
        signed_progress = sum(
            (float(value) - plane_value) * axis_value
            for value, plane_value, axis_value in zip(
                object_xyz,
                exit_point,
                direction,
                strict=True,
            )
        )
        expected = signed_progress >= 0.0
        if crossed is not expected:
            result.fail(
                path,
                f"row {row_number} target_crossed_exit disagrees with exit geometry",
            )


def _validate_step_order(
    path: Path,
    steps: list[dict[str, Any]],
    result: ValidationResult,
) -> None:
    previous_step: int | None = None
    previous_time: float | None = None
    for index, step in enumerate(steps):
        sim_step = step.get("sim_step")
        sim_time = step.get("sim_time_s")
        if not _is_int(sim_step) or sim_step < 0:
            result.fail(path, f"row {index + 1} has invalid sim_step")
        elif previous_step is not None and sim_step <= previous_step:
            result.fail(path, "sim_step must increase strictly")
        else:
            previous_step = sim_step

        if not _is_number(sim_time) or sim_time < 0:
            result.fail(path, f"row {index + 1} has invalid sim_time_s")
        elif previous_time is not None and sim_time <= previous_time:
            result.fail(path, "sim_time_s must increase strictly")
        else:
            previous_time = float(sim_time)


def _validate_event_order(
    path: Path,
    events: list[dict[str, Any]],
    result: ValidationResult,
) -> None:
    previous_time: float | None = None
    for index, event in enumerate(events):
        time_s = event.get("time_s")
        if not _is_number(time_s) or time_s < 0:
            result.fail(path, f"row {index + 1} has invalid time_s")
            continue
        if previous_time is not None and time_s < previous_time:
            result.fail(path, "event time_s must be non-decreasing")
        previous_time = float(time_s)


_SUCCESS_STEP_FIELDS = (
    "object_xyz",
    "gripper_closed",
    "left_contact",
    "right_contact",
    "target_in_gripper",
    "target_crossed_exit",
    "robot_fallen",
    "forbidden_collision",
    "wrong_object_grasped",
)

_SUCCESS_METRIC_FIELDS = (
    "verification_time_s",
    "completion_time_s",
    "max_lift_m",
    "hold_time_required_s",
    "lift_height_required_m",
    "sample_count",
)


def _validate_success_evidence(
    episode_dir: Path,
    episode: dict[str, Any],
    benchmark_config: dict[str, Any],
    summary: dict[str, Any],
    steps: list[dict[str, Any]],
    events: list[dict[str, Any]],
    result: ValidationResult,
) -> None:
    steps_path = episode_dir / "steps.jsonl"
    summary_path = episode_dir / "summary.json"
    task = episode.get("task")
    evaluation = benchmark_config.get("evaluation")
    metrics = summary.get("metrics")
    if not isinstance(task, dict):
        result.fail(episode_dir / "manifest.json", "'episode.task' must be an object")
        return
    if not isinstance(evaluation, dict):
        result.fail(
            episode_dir / "manifest.json",
            "'benchmark_config.evaluation' must be an object",
        )
        return
    if not isinstance(metrics, dict):
        result.fail(summary_path, "'metrics' must be an object")
        return

    for field_name in _SUCCESS_METRIC_FIELDS:
        if field_name not in metrics:
            result.fail(
                summary_path,
                f"successful episode metric {field_name!r} is missing",
            )

    belt_surface_z = task.get("belt_surface_z_m")
    hold_time = evaluation.get("hold_time_s")
    lift_height = evaluation.get("lift_height_m")
    task_type = task.get("task_type")
    if not _is_number(belt_surface_z):
        result.fail(episode_dir / "manifest.json", "invalid task belt_surface_z_m")
        return
    if not _is_number(hold_time) or hold_time < 0:
        result.fail(episode_dir / "manifest.json", "invalid evaluation hold_time_s")
        return
    if not _is_number(lift_height) or lift_height < 0:
        result.fail(episode_dir / "manifest.json", "invalid evaluation lift_height_m")
        return

    hold_start: float | None = None
    verified_at: float | None = None
    for row_number, step in enumerate(steps, start=1):
        missing = [field for field in _SUCCESS_STEP_FIELDS if field not in step]
        if missing:
            result.fail(
                steps_path,
                f"row {row_number} lacks success fields: {', '.join(missing)}",
            )
            continue

        object_xyz = step.get("object_xyz")
        booleans = {
            field: step.get(field)
            for field in _SUCCESS_STEP_FIELDS
            if field != "object_xyz"
        }
        if (
            not isinstance(object_xyz, list)
            or len(object_xyz) != 3
            or not all(_is_number(component) for component in object_xyz)
        ):
            result.fail(steps_path, f"row {row_number} has invalid object_xyz")
            continue
        invalid_boolean = next(
            (field for field, value in booleans.items() if not isinstance(value, bool)),
            None,
        )
        if invalid_boolean is not None:
            result.fail(
                steps_path,
                f"row {row_number} has non-boolean {invalid_boolean}",
            )
            continue

        if (
            step["robot_fallen"]
            or step["forbidden_collision"]
            or step["wrong_object_grasped"]
            or (task_type == "c1_dynamic_pick" and step["target_crossed_exit"])
        ):
            break

        secure = (
            step["gripper_closed"]
            and step["left_contact"]
            and step["right_contact"]
            and step["target_in_gripper"]
            and float(object_xyz[2]) - float(belt_surface_z) >= float(lift_height)
        )
        if secure:
            sim_time = step.get("sim_time_s")
            if not _is_number(sim_time):
                continue
            if hold_start is None:
                hold_start = float(sim_time)
            if float(sim_time) - hold_start >= float(hold_time):
                verified_at = float(sim_time)
                break
        else:
            hold_start = None

    if verified_at is None:
        result.fail(steps_path, "successful episode has no sustained secure-grasp evidence")

    verification_time = metrics.get("verification_time_s")
    if not _is_number(verification_time):
        result.fail(summary_path, "successful episode needs numeric verification_time_s")
    elif verified_at is not None and not math.isclose(
        float(verification_time), verified_at, rel_tol=0.0, abs_tol=1e-9
    ):
        result.fail(
            summary_path,
            "verification_time_s does not match the step evidence",
        )

    if not any(event.get("kind") == "grasp_verified" for event in events):
        result.fail(
            episode_dir / "events.jsonl",
            "successful episode lacks a grasp_verified event",
        )


def _validate_video(
    episode_dir: Path,
    report: dict[str, Any],
    episode: dict[str, Any],
    steps: list[dict[str, Any]],
    result: ValidationResult,
) -> None:
    frame_count = report.get("video_frames")
    if not _is_int(frame_count) or frame_count < 0:
        result.fail(episode_dir, "'video_frames' must be a non-negative integer")
        return

    metadata = episode.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    explicitly_enabled = metadata.get("video_enabled") is True or metadata.get(
        "save_video"
    ) is True
    camera_contract = metadata.get("cameras")
    if isinstance(camera_contract, dict):
        camera_names = tuple(
            name for name in _KNOWN_CAMERA_NAMES if name in camera_contract
        )
    else:
        camera_names = ("head_rgb", "wrist_rgb")
    if not camera_names:
        camera_names = ("head_rgb", "wrist_rgb")

    video_paths = tuple(episode_dir / f"{name}.mp4" for name in camera_names)
    known_video_paths = tuple(
        episode_dir / f"{name}.mp4" for name in _KNOWN_CAMERA_NAMES
    )
    frames_path = episode_dir / "camera_frames.jsonl"
    artifact_present = any(
        path.exists() for path in (*known_video_paths, frames_path)
    )
    video_enabled = explicitly_enabled or frame_count > 0 or artifact_present
    if not video_enabled:
        return

    for video_path in video_paths:
        if not video_path.is_file():
            result.fail(video_path, "enabled video artifact is missing")
        elif video_path.stat().st_size == 0:
            result.fail(video_path, "video artifact is empty")

    frames = _read_jsonl(frames_path, result)
    if frames is None:
        return
    result.video_frame_count += len(frames)
    if len(frames) != frame_count:
        result.fail(
            frames_path,
            f"frame count {len(frames)} does not match video_frames {frame_count}",
        )
    if explicitly_enabled and not frames:
        result.fail(frames_path, "video is enabled but the frame index is empty")

    step_times = {
        step.get("sim_step"): float(step["sim_time_s"])
        for step in steps
        if _is_int(step.get("sim_step")) and _is_number(step.get("sim_time_s"))
    }
    previous_step: int | None = None
    previous_time: float | None = None
    for expected_index, frame in enumerate(frames):
        frame_index = frame.get("frame_index")
        sim_step = frame.get("sim_step")
        sim_time = frame.get("sim_time_s")
        if frame_index != expected_index:
            result.fail(
                frames_path,
                f"frame_index must be contiguous from zero; expected {expected_index}",
            )
        if not _is_int(sim_step) or sim_step < 0:
            result.fail(frames_path, f"row {expected_index + 1} has invalid sim_step")
        elif previous_step is not None and sim_step <= previous_step:
            result.fail(frames_path, "frame sim_step must increase strictly")
        else:
            previous_step = sim_step
        if not _is_number(sim_time) or sim_time < 0:
            result.fail(frames_path, f"row {expected_index + 1} has invalid sim_time_s")
        elif previous_time is not None and sim_time <= previous_time:
            result.fail(frames_path, "frame sim_time_s must increase strictly")
        else:
            previous_time = float(sim_time)

        if _is_int(sim_step) and sim_step not in step_times:
            result.fail(
                frames_path,
                f"frame_index {expected_index} references absent sim_step {sim_step}",
            )
        elif (
            _is_int(sim_step)
            and _is_number(sim_time)
            and not math.isclose(
                step_times[sim_step],
                float(sim_time),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            result.fail(
                frames_path,
                f"frame_index {expected_index} time does not match its step",
            )


def _validate_episode(
    summary_path: Path,
    run_summary: dict[str, Any],
    report: dict[str, Any],
    result: ValidationResult,
) -> None:
    episode_id = _require_string(report, "episode_id", summary_path, result)
    if episode_id is None:
        return
    episode_dir = summary_path.parent / "episodes" / episode_id
    manifest_path = episode_dir / "manifest.json"
    events_path = episode_dir / "events.jsonl"
    steps_path = episode_dir / "steps.jsonl"
    episode_summary_path = episode_dir / "summary.json"

    manifest = _read_json(manifest_path, result)
    events = _read_jsonl(events_path, result)
    steps = _read_jsonl(steps_path, result)
    episode_summary = _read_json(episode_summary_path, result)
    if (
        manifest is None
        or events is None
        or steps is None
        or episode_summary is None
    ):
        return

    result.episode_count += 1
    result.sample_count += len(steps)
    _validate_step_order(steps_path, steps, result)
    _validate_event_order(events_path, events, result)

    episode = manifest.get("episode")
    benchmark_config = manifest.get("benchmark_config")
    if not isinstance(episode, dict):
        result.fail(manifest_path, "'episode' must be an object")
        return
    if not isinstance(benchmark_config, dict):
        result.fail(manifest_path, "'benchmark_config' must be an object")
        return

    run_id = run_summary.get("run_id")
    protocol_version = run_summary.get("protocol_version")
    if episode.get("episode_id") != episode_id:
        result.fail(manifest_path, "episode_id does not match the run summary")
    if episode.get("run_id") != run_id:
        result.fail(manifest_path, "run_id does not match the run summary")
    if episode.get("protocol_version") != protocol_version:
        result.fail(manifest_path, "protocol_version does not match the run summary")
    if benchmark_config.get("protocol_version") != protocol_version:
        result.fail(manifest_path, "benchmark protocol_version is inconsistent")

    task = episode.get("task")
    if not isinstance(task, dict):
        result.fail(manifest_path, "'episode.task' must be an object")
        task = {}
    if task.get("task_type") != run_summary.get("task_type"):
        result.fail(manifest_path, "task_type does not match the run summary")
    geometry = _resolve_transport_geometry(task, manifest_path, result)
    if geometry is not None:
        _validate_exit_flags(
            steps_path,
            steps,
            geometry[0],
            geometry[1],
            result,
        )

    success = report.get("success")
    if not isinstance(success, bool):
        result.fail(summary_path, f"episode {episode_id!r} success must be boolean")
        success = False
    failure_reason = report.get("failure_reason")
    if not isinstance(failure_reason, str) or not failure_reason:
        result.fail(
            summary_path,
            f"episode {episode_id!r} failure_reason must be a non-empty string",
        )

    if episode_summary.get("episode_id") != episode_id:
        result.fail(episode_summary_path, "episode_id is inconsistent")
    if episode_summary.get("task_id") != task.get("task_id"):
        result.fail(episode_summary_path, "task_id is inconsistent")
    if episode_summary.get("task_type") != task.get("task_type"):
        result.fail(episode_summary_path, "task_type is inconsistent")
    if episode_summary.get("success") is not success:
        result.fail(episode_summary_path, "success is inconsistent with run summary")
    expected_status = "success" if success else "failure"
    if episode_summary.get("status") != expected_status:
        result.fail(episode_summary_path, f"status must be {expected_status!r}")
    if episode_summary.get("failure_reason") != failure_reason:
        result.fail(
            episode_summary_path,
            "failure_reason is inconsistent with run summary",
        )
    if success and failure_reason != "none":
        result.fail(episode_summary_path, "successful episode must use failure_reason 'none'")
    if not success and failure_reason == "none":
        result.fail(episode_summary_path, "failed episode must include a failure reason")

    sample_count = _require_nonnegative_int(
        episode_summary, "sample_count", episode_summary_path, result
    )
    if sample_count is not None and sample_count != len(steps):
        result.fail(
            episode_summary_path,
            f"sample_count {sample_count} does not match {len(steps)} step rows",
        )
    event_count = _require_nonnegative_int(
        episode_summary, "event_count", episode_summary_path, result
    )
    if event_count is not None and event_count != len(events):
        result.fail(
            episode_summary_path,
            f"event_count {event_count} does not match {len(events)} event rows",
        )

    for metrics_path, metrics in (
        (episode_summary_path, episode_summary.get("metrics")),
        (summary_path, report.get("metrics")),
    ):
        if not isinstance(metrics, dict):
            result.fail(metrics_path, f"episode {episode_id!r} metrics must be an object")
        elif "sample_count" in metrics and metrics["sample_count"] != len(steps):
            result.fail(metrics_path, "metrics.sample_count is inconsistent")

    if not events or events[-1].get("kind") != "episode_end":
        result.fail(events_path, "last event must be episode_end")
    else:
        payload = events[-1].get("payload")
        if not isinstance(payload, dict):
            result.fail(events_path, "episode_end payload must be an object")
        else:
            if payload.get("success") is not success:
                result.fail(events_path, "episode_end success is inconsistent")
            if payload.get("failure_reason") != failure_reason:
                result.fail(events_path, "episode_end failure_reason is inconsistent")

    if success:
        _validate_success_evidence(
            episode_dir,
            episode,
            benchmark_config,
            episode_summary,
            steps,
            events,
            result,
        )
    _validate_video(episode_dir, report, episode, steps, result)


def validate_run_summary(path: Path, result: ValidationResult) -> None:
    summary = _read_json(path, result)
    if summary is None:
        return
    result.run_count += 1

    run_id = _require_string(summary, "run_id", path, result)
    _require_string(summary, "protocol_version", path, result)
    _require_string(summary, "task_type", path, result)
    requested = _require_nonnegative_int(summary, "requested_episodes", path, result)
    successful = _require_nonnegative_int(
        summary, "successful_episodes", path, result
    )
    reports = summary.get("episodes")
    if not isinstance(reports, list):
        result.fail(path, "'episodes' must be a list")
        return
    if requested is not None and requested != len(reports):
        result.fail(
            path,
            f"requested_episodes {requested} does not match {len(reports)} reports",
        )

    valid_reports = [report for report in reports if isinstance(report, dict)]
    if len(valid_reports) != len(reports):
        result.fail(path, "every episode report must be an object")
    actual_successes = sum(report.get("success") is True for report in valid_reports)
    if successful is not None and successful != actual_successes:
        result.fail(
            path,
            f"successful_episodes {successful} does not match {actual_successes}",
        )

    episode_ids = [report.get("episode_id") for report in valid_reports]
    string_ids = [value for value in episode_ids if isinstance(value, str)]
    if len(string_ids) != len(set(string_ids)):
        result.fail(path, "episode_id values must be unique within a run")
    if run_id is None:
        return
    for report in valid_reports:
        _validate_episode(path, summary, report, result)


def _find_run_summaries(path: Path, result: ValidationResult) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        result.fail(path, "input path does not exist")
        return []
    summaries = sorted(path.rglob("run-*-summary.json"))
    if not summaries:
        result.fail(path, "no run-*-summary.json files found")
    return summaries


def validate_dataset(path: Path) -> ValidationResult:
    result = ValidationResult()
    for summary_path in _find_run_summaries(path, result):
        validate_run_summary(summary_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Run summary JSON or a directory containing run-*-summary.json files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = validate_dataset(args.path.resolve())
    if result.errors:
        for error in result.errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"FAILED: {len(result.errors)} validation error(s)",
            file=sys.stderr,
        )
        return 1
    print(
        "OK: "
        f"{result.run_count} run(s), "
        f"{result.episode_count} episode(s), "
        f"{result.sample_count} sample(s), "
        f"{result.video_frame_count} video frame(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
