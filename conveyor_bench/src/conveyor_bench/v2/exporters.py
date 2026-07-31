"""V2 task-context annotations over lossless V1 model projections."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from conveyor_bench.v1.config import BenchmarkConfig
from conveyor_bench.v1.exporters import (
    ExportError,
    iter_dynamicvla_records as _iter_dynamicvla_records_v1,
    iter_m0_records as _iter_m0_records_v1,
)

from .config import (
    BENCHMARK_SUITE_VERSION,
    CANONICAL_PROTOCOL_VERSION,
    TASK_CONTEXT_SCHEMA_VERSION,
)


EXPORT_SCHEMA_VERSION = "conveyor-bench-v2-export-1"
SUPERVISION_ONLY_FIELDS = (
    "current_target_id",
    "current_subtask_index",
)


def iter_dynamicvla_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the V1 DynamicVLA projection plus V2 task supervision."""

    episode_path = Path(episode_directory)
    context = _load_context(episode_path)
    selected_by_tick = _selected_target_by_tick(episode_path)
    for record in _iter_dynamicvla_records_v1(episode_path, config):
        yield _annotate(record, context, selected_by_tick)


def iter_m0_records(
    episode_directory: str | Path,
    config: BenchmarkConfig | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the V1 ABot-M0 projection plus V2 task supervision."""

    episode_path = Path(episode_directory)
    context = _load_context(episode_path)
    selected_by_tick = _selected_target_by_tick(episode_path)
    for record in _iter_m0_records_v1(episode_path, config):
        yield _annotate(record, context, selected_by_tick)


def _annotate(
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    selected_by_tick: Mapping[int, str | None],
) -> dict[str, Any]:
    record = dict(source)
    targets = context["target_sequence_ids"]
    model_tick = source.get("model_tick")
    current_target = (
        selected_by_tick.get(model_tick)
        if isinstance(model_tick, int) and not isinstance(model_tick, bool)
        else None
    )
    current_index = (
        targets.index(current_target) if current_target in targets else None
    )
    record.update(
        {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "scene_id": context["scene_id"],
            "task_family": context["task_family"],
            "target_sequence_ids": targets,
            "destination_zone_by_target": dict(
                context["destination_zone_by_target"]
            ),
            "current_target_id": current_target,
            "current_subtask_index": current_index,
            "supervision_only_fields": SUPERVISION_ONLY_FIELDS,
        }
    )
    return record


def _load_context(episode_path: Path) -> Mapping[str, Any]:
    manifest = _read_json(episode_path / "manifest.json")
    episode = _mapping(manifest.get("episode"), "manifest.episode")
    task = _mapping(episode.get("task"), "manifest.episode.task")
    metadata = _mapping(task.get("metadata"), "manifest.episode.task.metadata")
    suite = _mapping(
        metadata.get("benchmark_suite"),
        "manifest.episode.task.metadata.benchmark_suite",
    )
    expected_versions = {
        "schema_version": TASK_CONTEXT_SCHEMA_VERSION,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "canonical_protocol_version": CANONICAL_PROTOCOL_VERSION,
    }
    for key, expected in expected_versions.items():
        if suite.get(key) != expected:
            raise ExportError(f"benchmark_suite.{key} must be {expected!r}")

    scene_id = suite.get("scene_id")
    task_family = suite.get("task_family")
    targets = suite.get("target_sequence_ids")
    destinations = suite.get("destination_zone_by_target")
    if not isinstance(scene_id, str) or not scene_id:
        raise ExportError("benchmark_suite.scene_id must be non-empty")
    if not isinstance(task_family, str) or not task_family:
        raise ExportError("benchmark_suite.task_family must be non-empty")
    if (
        not _is_sequence(targets)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or len(targets) != len(set(targets))
    ):
        raise ExportError(
            "benchmark_suite.target_sequence_ids must contain unique target IDs"
        )
    resolved_targets = tuple(targets)
    if (
        not isinstance(destinations, Mapping)
        or set(destinations) != set(resolved_targets)
        or any(
            not isinstance(zone_id, str) or not zone_id
            for zone_id in destinations.values()
        )
    ):
        raise ExportError(
            "benchmark_suite.destination_zone_by_target is inconsistent"
        )
    return {
        "scene_id": scene_id,
        "task_family": task_family,
        "target_sequence_ids": resolved_targets,
        "destination_zone_by_target": dict(destinations),
    }


def _selected_target_by_tick(episode_path: Path) -> dict[int, str | None]:
    selected: dict[int, str | None] = {}
    path = episode_path / "steps.jsonl"
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    step = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ExportError(
                        f"{path}:{line_number} is not valid JSON: {error}"
                    ) from error
                step = _mapping(step, f"{path}:{line_number}")
                model_tick = step.get("model_tick")
                target_id = step.get("selected_object_id")
                if isinstance(model_tick, bool) or not isinstance(model_tick, int):
                    raise ExportError(f"{path}:{line_number} model_tick is invalid")
                if target_id is not None and (
                    not isinstance(target_id, str) or not target_id
                ):
                    raise ExportError(
                        f"{path}:{line_number} selected_object_id is invalid"
                    )
                selected[model_tick] = target_id
    except OSError as error:
        raise ExportError(f"cannot read {path}: {error}") from error
    return selected


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            return _mapping(json.load(stream), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot read {path}: {error}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExportError(f"{name} must be a JSON object")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "ExportError",
    "SUPERVISION_ONLY_FIELDS",
    "iter_dynamicvla_records",
    "iter_m0_records",
]
