"""V2 suite validation layered over the frozen V1 canonical protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from conveyor_bench.v1.validation import ValidationResult, validate_v1_episode

from .camera_contracts import camera_contract_for_scene
from .config import (
    BENCHMARK_SUITE_VERSION,
    CANONICAL_PROTOCOL_VERSION,
    DEFAULT_SUITE_CONFIG,
    TASK_CONTEXT_SCHEMA_VERSION,
    SceneId,
)
from .tasking import validate_task_combination


MINIMUM_REMOTE_LOADED_DISPLACEMENT_M = 0.65


def validate_v2_episode(source: str | Path) -> ValidationResult:
    """Validate one V2 task while retaining canonical V1 validation."""

    episode_path = Path(source)
    result = validate_v1_episode(episode_path)
    manifest = _read_json(episode_path / "manifest.json", result)
    summary = _read_json(episode_path / "summary.json", result)
    if manifest is None or summary is None:
        return result

    task, suite = _validate_suite_metadata(
        episode_path / "manifest.json",
        manifest,
        result,
    )
    if task is None or suite is None:
        return result

    events = _read_jsonl(episode_path / "events.jsonl", result)
    if events is None:
        return result

    family = suite["task_family"]
    targets = tuple(suite["target_sequence_ids"])
    _validate_task_event_sequence(
        episode_path,
        events,
        suite,
        targets,
        require_complete=summary.get("success") is True,
        result=result,
    )

    if summary.get("success") is not True:
        return result

    steps = _read_jsonl(episode_path / "steps.jsonl", result)
    if steps is None:
        return result

    if family == "continuous_multi_target":
        _validate_continuous_success(
            episode_path,
            steps,
            events,
            targets,
            result,
        )

    if suite["scene_id"] == SceneId.MOBILE_REMOTE_DELIVERY_V2.value:
        objects = _read_jsonl(episode_path / "objects.jsonl", result)
        if objects is not None:
            _validate_remote_success(
                episode_path,
                steps,
                objects,
                events,
                targets[0],
                result,
            )
    return result


def _validate_suite_metadata(
    path: Path,
    manifest: Mapping[str, Any],
    result: ValidationResult,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    episode = manifest.get("episode")
    task = episode.get("task") if isinstance(episode, Mapping) else None
    metadata = task.get("metadata") if isinstance(task, Mapping) else None
    suite = (
        metadata.get("benchmark_suite")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(episode, Mapping) or not isinstance(task, Mapping):
        result.fail(path, "V2 manifest requires episode.task metadata")
        return None, None
    if not isinstance(metadata, Mapping) or not isinstance(suite, Mapping):
        result.fail(path, "task metadata requires benchmark_suite")
        return task, None

    expected_versions = {
        "schema_version": TASK_CONTEXT_SCHEMA_VERSION,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "canonical_protocol_version": CANONICAL_PROTOCOL_VERSION,
    }
    for key, expected in expected_versions.items():
        if suite.get(key) != expected:
            result.fail(path, f"benchmark_suite.{key} must be {expected!r}")

    scene_id = suite.get("scene_id")
    family = suite.get("task_family")
    robot_mode = suite.get("robot_mode")
    if not all(
        isinstance(value, str) and value
        for value in (scene_id, family, robot_mode)
    ):
        result.fail(
            path,
            "benchmark_suite scene_id, task_family, and robot_mode must be strings",
        )
        return task, None
    if task.get("robot_mode") != robot_mode:
        result.fail(path, "benchmark_suite robot_mode does not match task")
    if metadata.get("task_family") != family:
        result.fail(path, "benchmark_suite task_family does not match task metadata")
    try:
        validate_task_combination(scene_id, family, robot_mode)
    except (TypeError, ValueError) as error:
        result.fail(
            path,
            "unsupported scene/task/mode combination: "
            f"{scene_id}/{family}/{robot_mode} ({error})",
        )
    try:
        scene_contract = DEFAULT_SUITE_CONFIG.scene(scene_id)
    except ValueError:
        scene_contract = None
    if scene_contract is not None:
        if suite.get("layout_id") != scene_contract.layout_id:
            result.fail(
                path,
                "benchmark_suite.layout_id does not match the frozen scene",
            )
        if metadata.get("layout_id") != scene_contract.layout_id:
            result.fail(
                path,
                "task metadata layout_id does not match the frozen scene",
            )
        expected_contracts = {
            zone.zone_id: zone.to_snapshot()
            for zone in scene_contract.goal_zones
        }
        if suite.get("destination_zone_contracts") != expected_contracts:
            result.fail(
                path,
                "destination_zone_contracts do not match the frozen scene",
            )
        expected_goal_zones = {
            zone.zone_id: {
                "min_xyz": list(zone.min_xyz),
                "max_xyz": list(zone.max_xyz),
            }
            for zone in scene_contract.goal_zones
        }
        raw_goal_zones = task.get("goal_zones")
        actual_goal_zones = {
            zone.get("zone_id"): {
                "min_xyz": zone.get("min_xyz"),
                "max_xyz": zone.get("max_xyz"),
            }
            for zone in raw_goal_zones
            if isinstance(zone, Mapping)
            and isinstance(zone.get("zone_id"), str)
        } if _is_sequence(raw_goal_zones) else {}
        if (
            not _is_sequence(raw_goal_zones)
            or len(raw_goal_zones) != len(expected_goal_zones)
            or actual_goal_zones != expected_goal_zones
        ):
            result.fail(
                path,
                "task.goal_zones do not match the frozen scene geometry",
            )
        episode_metadata = episode.get("metadata")
        cameras = (
            episode_metadata.get("cameras")
            if isinstance(episode_metadata, Mapping)
            else None
        )
        if cameras != camera_contract_for_scene(scene_contract.scene_id):
            result.fail(
                path,
                "episode camera contract does not match the frozen V2 scene",
            )
    if (
        suite.get("object_split") not in {"train", "val", "unseen"}
        or suite.get("object_split") != metadata.get("curriculum_split")
    ):
        result.fail(
            path,
            "benchmark_suite.object_split does not match curriculum_split",
        )

    targets = suite.get("target_sequence_ids")
    if (
        not _is_sequence(targets)
        or not targets
        or any(not isinstance(value, str) or not value for value in targets)
        or len(targets) != len(set(targets))
    ):
        result.fail(
            path,
            "benchmark_suite.target_sequence_ids must contain unique target IDs",
        )
        return task, None
    resolved_targets = tuple(targets)
    scored_object_ids = task.get("scored_object_ids")
    if (
        not _is_sequence(scored_object_ids)
        or tuple(scored_object_ids) != resolved_targets
    ):
        result.fail(path, "target_sequence_ids do not match scored_object_ids")
    metadata_target_ids = metadata.get("target_ids")
    if (
        not _is_sequence(metadata_target_ids)
        or tuple(metadata_target_ids) != resolved_targets
    ):
        result.fail(path, "target_sequence_ids do not match task metadata target_ids")

    expected_task_type = (
        "continuous_sort"
        if family == "continuous_multi_target"
        else "dynamic_sort"
    )
    if task.get("task_type") != expected_task_type:
        result.fail(
            path,
            f"task_type must be {expected_task_type!r} for {family!r}",
        )
    if metadata.get("scene_id") != scene_id:
        result.fail(path, "task metadata scene_id does not match benchmark_suite")

    belt_speed = task.get("belt_speed_mps")
    metadata_belt_speed = metadata.get("belt_speed_mps")
    if (
        not _is_number(belt_speed)
        or not any(
            math.isclose(
                float(belt_speed),
                allowed,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for allowed in DEFAULT_SUITE_CONFIG.belt_speeds_mps
        )
        or not _is_number(metadata_belt_speed)
        or not math.isclose(
            float(metadata_belt_speed),
            float(belt_speed),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        result.fail(
            path,
            "task and metadata belt_speed_mps must match a frozen V2 speed",
        )

    destinations = suite.get("destination_zone_by_target")
    if (
        not isinstance(destinations, Mapping)
        or set(destinations) != set(resolved_targets)
        or any(
            not isinstance(zone_id, str) or not zone_id
            for zone_id in destinations.values()
        )
    ):
        result.fail(
            path,
            "destination_zone_by_target must map every and only sequence target",
        )
        return task, None
    if metadata.get("destination_zone_by_target") != destinations:
        result.fail(path, "suite destination map does not match task metadata")

    objects = task.get("objects")
    object_by_id = {
        item.get("instance_id"): item
        for item in objects
        if isinstance(item, Mapping) and isinstance(item.get("instance_id"), str)
    } if _is_sequence(objects) else {}
    distractors = metadata.get("distractors")
    resolved_distractors = (
        tuple(distractors) if _is_sequence(distractors) else ()
    )
    expected_cardinality = {
        "single_target": (1, 1, 0),
        "language_conditioned": (2, 1, 1),
        "continuous_multi_target": (2, 2, 0),
    }.get(family)
    if expected_cardinality is not None:
        expected_objects, expected_targets, expected_distractors = (
            expected_cardinality
        )
        if (
            len(object_by_id) != expected_objects
            or len(resolved_targets) != expected_targets
            or len(resolved_distractors) != expected_distractors
            or set(object_by_id)
            != set(resolved_targets).union(resolved_distractors)
        ):
            result.fail(
                path,
                f"{family} object/target/distractor cardinality is inconsistent",
            )
    for target_id in resolved_targets:
        target = object_by_id.get(target_id)
        if not isinstance(target, Mapping) or target.get(
            "goal_zone_id"
        ) != destinations[target_id]:
            result.fail(path, f"destination for target {target_id!r} is inconsistent")

    instance_asset_map = metadata.get("instance_asset_map")
    expected_instance_asset_map = {
        instance_id: item.get("asset_id")
        for instance_id, item in object_by_id.items()
    }
    if instance_asset_map != expected_instance_asset_map:
        result.fail(path, "instance_asset_map does not match task objects")

    _validate_service_gates(path, suite, resolved_targets, result)
    _validate_schedule_gate_mirror(
        path,
        metadata,
        suite,
        object_by_id,
        resolved_targets,
        destinations,
        result,
    )
    _validate_task_duration(path, task, metadata, result)
    contracts = suite.get("destination_zone_contracts")
    if not isinstance(contracts, Mapping) or not set(destinations.values()) <= set(
        contracts
    ):
        result.fail(path, "destination_zone_contracts omit a target destination")

    minimum_displacement = suite.get("minimum_loaded_base_displacement_m")
    try:
        expected_displacement = DEFAULT_SUITE_CONFIG.scene(
            scene_id
        ).minimum_loaded_base_displacement_m
    except ValueError:
        expected_displacement = None
    if (
        not _is_number(minimum_displacement)
        or float(minimum_displacement) < 0.0
        or (
            expected_displacement is not None
            and not math.isclose(
                float(minimum_displacement),
                expected_displacement,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
    ):
        result.fail(
            path,
            "benchmark_suite minimum_loaded_base_displacement_m is inconsistent",
        )

    episode_metadata = episode.get("metadata")
    mirror = (
        episode_metadata.get("benchmark_suite")
        if isinstance(episode_metadata, Mapping)
        else None
    )
    if mirror is not None and mirror != suite:
        result.fail(path, "episode benchmark_suite mirror does not match task metadata")
    return task, suite


def _validate_service_gates(
    path: Path,
    suite: Mapping[str, Any],
    targets: tuple[str, ...],
    result: ValidationResult,
) -> None:
    expected_policy = "service_gated" if len(targets) > 1 else "episode_start"
    if suite.get("spawn_policy") != expected_policy:
        result.fail(path, f"benchmark_suite.spawn_policy must be {expected_policy!r}")
    gates = suite.get("service_gates")
    if not _is_sequence(gates) or len(gates) != len(targets):
        result.fail(path, "service_gates must cover target_sequence_ids")
        return
    for index, (target_id, gate) in enumerate(zip(targets, gates, strict=True)):
        predecessor = targets[index - 1] if index else None
        expected_kind = "previous_target_completed" if index else "episode_start"
        if not isinstance(gate, Mapping) or (
            gate.get("service_index") != index
            or gate.get("target_instance_id") != target_id
            or gate.get("gate_kind") != expected_kind
            or gate.get("after_target_instance_id") != predecessor
            or not _is_number(gate.get("not_before_s"))
            or float(gate["not_before_s"]) < 0.0
        ):
            result.fail(path, f"service gate {index} does not match target sequence")


def _validate_schedule_gate_mirror(
    path: Path,
    metadata: Mapping[str, Any],
    suite: Mapping[str, Any],
    object_by_id: Mapping[str, Mapping[str, Any]],
    targets: tuple[str, ...],
    destinations: Mapping[str, str],
    result: ValidationResult,
) -> None:
    """Keep object declarations, spawn schedule, and service gates identical."""

    schedule = metadata.get("spawn_schedule")
    gates = suite.get("service_gates")
    if not _is_sequence(schedule) or not _is_sequence(gates):
        result.fail(path, "spawn_schedule and service_gates must be sequences")
        return
    schedule_by_object: dict[str, Mapping[str, Any]] = {}
    for entry in schedule:
        if not isinstance(entry, Mapping):
            result.fail(path, "spawn_schedule entries must be objects")
            continue
        object_id = entry.get("object_instance_id")
        if not isinstance(object_id, str) or not object_id:
            result.fail(path, "spawn_schedule object_instance_id must be a string")
            continue
        if object_id in schedule_by_object:
            result.fail(path, f"spawn_schedule repeats object {object_id!r}")
            continue
        schedule_by_object[object_id] = entry
    if (
        len(schedule) != len(object_by_id)
        or set(schedule_by_object) != set(object_by_id)
    ):
        result.fail(path, "spawn_schedule must cover every and only task object")

    target_set = set(targets)
    for object_id, task_object in object_by_id.items():
        entry = schedule_by_object.get(object_id)
        if entry is None:
            continue
        is_target = object_id in target_set
        expected_role = "target" if is_target else "distractor"
        expected_destination = destinations.get(object_id)
        if entry.get("asset_id") != task_object.get("asset_id"):
            result.fail(
                path,
                f"spawn_schedule asset_id for {object_id!r} is inconsistent",
            )
        if entry.get("role") != expected_role:
            result.fail(
                path,
                f"spawn_schedule role for {object_id!r} must be {expected_role!r}",
            )
        if entry.get("destination_zone_id") != expected_destination:
            result.fail(
                path,
                f"spawn_schedule destination for {object_id!r} is inconsistent",
            )
        spawn_time = entry.get("spawn_time_s")
        initialization_end = entry.get("initialization_end_s")
        if (
            not _is_number(spawn_time)
            or float(spawn_time) < 0.0
            or not _is_number(initialization_end)
            or float(initialization_end) <= float(spawn_time)
        ):
            result.fail(
                path,
                f"spawn_schedule initialization window for {object_id!r} is invalid",
            )
        if is_target:
            continue
        service_gate = entry.get("service_gate")
        if (
            not isinstance(service_gate, Mapping)
            or service_gate.get("gate_kind") != "episode_start"
            or service_gate.get("after_target_instance_id") is not None
            or service_gate.get("target_instance_id") is not None
            or service_gate.get("service_index") is not None
            or not _is_number(service_gate.get("not_before_s"))
            or not _is_number(spawn_time)
            or not math.isclose(
                float(service_gate["not_before_s"]),
                float(spawn_time),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            result.fail(
                path,
                f"distractor service gate for {object_id!r} is inconsistent",
            )

    if tuple(schedule_by_object) != tuple(object_by_id):
        result.fail(path, "spawn_schedule order must match task.objects")
    previous_initialization_end: float | None = None
    for entry in schedule:
        if not isinstance(entry, Mapping):
            continue
        spawn_time = entry.get("spawn_time_s")
        initialization_end = entry.get("initialization_end_s")
        if not _is_number(spawn_time) or not _is_number(initialization_end):
            continue
        if (
            previous_initialization_end is not None
            and float(spawn_time) < previous_initialization_end
        ):
            result.fail(path, "spawn_schedule initialization windows overlap")
        previous_initialization_end = float(initialization_end)

    gate_by_target = {
        str(gate.get("target_instance_id")): gate
        for gate in gates
        if isinstance(gate, Mapping)
        and gate.get("target_instance_id") in targets
    }
    for target_id in targets:
        entry = schedule_by_object.get(target_id)
        gate = gate_by_target.get(target_id)
        if entry is None or gate is None:
            result.fail(
                path,
                f"spawn_schedule and service_gates must both cover {target_id!r}",
            )
            continue
        if entry.get("service_gate") != gate:
            result.fail(
                path,
                f"spawn_schedule service_gate for {target_id!r} is inconsistent",
            )
        spawn_time = entry.get("spawn_time_s")
        not_before = gate.get("not_before_s")
        if (
            not _is_number(spawn_time)
            or not _is_number(not_before)
            or not math.isclose(
                float(spawn_time),
                float(not_before),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            result.fail(
                path,
                f"spawn_time_s for {target_id!r} must equal gate not_before_s",
            )


def _validate_task_duration(
    path: Path,
    task: Mapping[str, Any],
    metadata: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    schedule = metadata.get("spawn_schedule")
    initialization_ends = [
        float(entry["initialization_end_s"])
        for entry in schedule
        if isinstance(entry, Mapping)
        and _is_number(entry.get("initialization_end_s"))
    ] if _is_sequence(schedule) else []
    max_duration_s = task.get("max_duration_s")
    if (
        not _is_number(max_duration_s)
        or not initialization_ends
        or float(max_duration_s) <= max(initialization_ends)
    ):
        result.fail(
            path,
            "task.max_duration_s must exceed every initialization_end_s",
        )


def _validate_task_event_sequence(
    episode_path: Path,
    events: Sequence[Mapping[str, Any]],
    suite: Mapping[str, Any],
    targets: tuple[str, ...],
    *,
    require_complete: bool,
    result: ValidationResult,
) -> None:
    """Validate observed target lifecycle events against the V2 task context.

    Failed episodes may contain any valid prefix of the lifecycle.  Successful
    continuous episodes must contain all target selection, spawn, and placement
    events.  Other V2 families retain the frozen V1 lifecycle requirements and
    validate any observed selection/spawn events without inventing new required
    records.  Non-target events remain valid failure evidence and are not part
    of the scored target sequence.
    """

    path = episode_path / "events.jsonl"
    target_set = set(targets)
    by_kind: dict[str, list[tuple[str, int, int, float | None]]] = {
        "target_selected": [],
        "object_spawned": [],
        "object_placed": [],
    }
    for event_index, event in enumerate(events):
        kind = event.get("kind")
        target_id = event.get("object_instance_id")
        if kind not in by_kind or target_id not in target_set:
            continue
        time_s = event.get("time_s")
        by_kind[kind].append(
            (
                str(target_id),
                event_index,
                event_index + 1,
                float(time_s) if _is_number(time_s) else None,
            )
        )

    continuous = suite.get("task_family") == "continuous_multi_target"
    for kind, occurrences in by_kind.items():
        observed = tuple(item[0] for item in occurrences)
        complete = require_complete and (
            kind == "object_placed" or continuous
        )
        expected = targets if complete else targets[: len(observed)]
        if observed != expected:
            qualifier = "complete sequence" if complete else "sequence prefix"
            result.fail(
                path,
                f"V2 {kind} target order must match the {qualifier} of "
                "target_sequence_ids",
            )

    selected_by_id = {item[0]: item for item in by_kind["target_selected"]}
    spawned_by_id = {item[0]: item for item in by_kind["object_spawned"]}
    placed_by_id = {item[0]: item for item in by_kind["object_placed"]}

    raw_gates = suite.get("service_gates")
    gates_by_id = {
        gate.get("target_instance_id"): gate
        for gate in raw_gates
        if isinstance(gate, Mapping)
        and isinstance(gate.get("target_instance_id"), str)
    } if _is_sequence(raw_gates) else {}

    for index, target_id in enumerate(targets):
        selected = selected_by_id.get(target_id)
        spawned = spawned_by_id.get(target_id)
        placed = placed_by_id.get(target_id)
        gate = gates_by_id.get(target_id)

        if spawned is not None and isinstance(gate, Mapping):
            not_before_s = gate.get("not_before_s")
            if (
                _is_number(not_before_s)
                and spawned[3] is not None
                and spawned[3] + 1.0e-12 < float(not_before_s)
            ):
                result.fail(
                    path,
                    f"object_spawned for {target_id!r} occurs before service "
                    f"gate not_before_s={float(not_before_s):g}",
                    spawned[2],
                )

        if continuous and spawned is not None:
            if selected is None:
                result.fail(
                    path,
                    f"object_spawned for {target_id!r} lacks target_selected",
                    spawned[2],
                )
            elif not _event_precedes_or_ties(selected, spawned):
                result.fail(
                    path,
                    f"target_selected for {target_id!r} must precede its "
                    "object_spawned event",
                    spawned[2],
                )

        if placed is not None and (continuous or spawned is not None):
            if spawned is None:
                result.fail(
                    path,
                    f"object_placed for {target_id!r} lacks object_spawned",
                    placed[2],
                )
            elif not _event_precedes_or_ties(spawned, placed):
                result.fail(
                    path,
                    f"object_spawned for {target_id!r} must precede its "
                    "object_placed event",
                    placed[2],
                )

        if not continuous or index == 0:
            continue
        predecessor_id = targets[index - 1]
        predecessor_placed = placed_by_id.get(predecessor_id)
        if selected is not None and (
            predecessor_placed is None
            or not _event_precedes_or_ties(predecessor_placed, selected)
        ):
            result.fail(
                path,
                f"target_selected for {target_id!r} must not precede previous "
                f"target object_placed for {predecessor_id!r}",
                selected[2],
            )
        if spawned is not None and (
            predecessor_placed is None
            or not _event_precedes_or_ties(predecessor_placed, spawned)
        ):
            result.fail(
                path,
                f"object_spawned for {target_id!r} must not precede previous "
                f"target object_placed for {predecessor_id!r}",
                spawned[2],
            )


def _event_precedes_or_ties(
    earlier: tuple[str, int, int, float | None],
    later: tuple[str, int, int, float | None],
) -> bool:
    """Return whether two JSONL events respect row order and event time."""

    if earlier[1] >= later[1]:
        return False
    if earlier[3] is None or later[3] is None:
        return True
    return earlier[3] <= later[3] + 1.0e-12

def _validate_continuous_success(
    episode_path: Path,
    steps: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    targets: tuple[str, ...],
    result: ValidationResult,
) -> None:
    selected: list[str] = []
    for step in steps:
        target_id = step.get("selected_object_id")
        if target_id is None:
            continue
        if not selected or selected[-1] != target_id:
            selected.append(target_id)
    if tuple(selected) != targets:
        result.fail(
            episode_path / "steps.jsonl",
            "continuous success selected_object_id order does not match "
            "target_sequence_ids",
        )

    placed = tuple(
        event.get("object_instance_id")
        for event in events
        if event.get("kind") == "object_placed"
    )
    if placed != targets:
        result.fail(
            episode_path / "events.jsonl",
            "continuous success object_placed order does not match "
            "target_sequence_ids",
        )


def _validate_remote_success(
    episode_path: Path,
    steps: Sequence[Mapping[str, Any]],
    objects: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    target_id: str,
    result: ValidationResult,
) -> None:
    releases = [
        event
        for event in events
        if event.get("kind") == "object_released"
        and event.get("object_instance_id") == target_id
    ]
    release_step = releases[0].get("sim_step") if releases else None
    release_time = releases[0].get("time_s") if releases else None

    root_by_step: dict[int, tuple[float, float]] = {}
    for step in steps:
        sim_step = step.get("sim_step")
        pose = step.get("robot_root_world")
        xyz = pose.get("xyz") if isinstance(pose, Mapping) else None
        if isinstance(sim_step, int) and not isinstance(sim_step, bool) and _is_vector(
            xyz, 3
        ):
            root_by_step[sim_step] = (float(xyz[0]), float(xyz[1]))

    maximum = 0.0
    segment_origin: tuple[float, float] | None = None
    for row in objects:
        state = row.get("state")
        if not isinstance(state, Mapping) or state.get("instance_id") != target_id:
            continue
        sim_step = row.get("sim_step")
        sim_time = row.get("sim_time_s")
        before_release = (
            isinstance(release_step, int)
            and isinstance(sim_step, int)
            and sim_step <= release_step
        ) or (
            release_step is None
            and _is_number(release_time)
            and _is_number(sim_time)
            and float(sim_time) <= float(release_time)
        )
        root_xy = root_by_step.get(sim_step) if isinstance(sim_step, int) else None
        if state.get("in_gripper") is True and before_release and root_xy is not None:
            if segment_origin is None:
                segment_origin = root_xy
            maximum = max(maximum, math.dist(segment_origin, root_xy))
        else:
            segment_origin = None

    if maximum + 1.0e-12 < MINIMUM_REMOTE_LOADED_DISPLACEMENT_M:
        result.fail(
            episode_path / "objects.jsonl",
            "remote success loaded base displacement must be at least "
            f"{MINIMUM_REMOTE_LOADED_DISPLACEMENT_M:.2f} m; observed {maximum:.3f} m",
        )


def _read_json(path: Path, result: ValidationResult) -> Mapping[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        result.fail(path, f"cannot read JSON: {error}")
        return None
    if not isinstance(value, Mapping):
        result.fail(path, "JSON root must be an object")
        return None
    return value


def _read_jsonl(
    path: Path,
    result: ValidationResult,
) -> list[Mapping[str, Any]] | None:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    result.fail(path, "JSONL row must be an object", line_number)
                    return None
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        result.fail(path, f"cannot read JSONL: {error}")
        return None
    return rows


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _is_vector(value: Any, size: int) -> bool:
    return (
        _is_sequence(value)
        and len(value) == size
        and all(_is_number(component) for component in value)
    )


__all__ = [
    "MINIMUM_REMOTE_LOADED_DISPLACEMENT_M",
    "validate_v2_episode",
]
