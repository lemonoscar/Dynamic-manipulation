"""Deterministic task contracts for the ConveyorBench V2 suite.

The returned context contains only protocol values and JSON-safe metadata.  It
does not depend on Isaac Sim or on the physical runtime's private resolved-task
types, so invalid scene/family/mode combinations can be rejected before the
simulator starts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import Any, Mapping

from conveyor_bench.v1.protocol import (
    GoalZone,
    ObjectInstance,
    RobotMode,
    TaskManifest,
)
from conveyor_bench.v1.tasking import (
    TASKING_SCHEMA_VERSION,
    CurriculumConfig,
    CurriculumSplit,
    InstructionLanguage,
    TaskFamily,
    build_episode_manifest,
)

from .config import (
    BENCHMARK_SUITE_VERSION,
    CANONICAL_PROTOCOL_VERSION,
    DEFAULT_SUITE_CONFIG,
    TASK_CONTEXT_SCHEMA_VERSION,
    SceneContract,
    SceneId,
    V2SuiteConfig,
)


class ServiceGateKind(str, Enum):
    EPISODE_START = "episode_start"
    PREVIOUS_TARGET_COMPLETED = "previous_target_completed"


@dataclass(frozen=True)
class ServiceGate:
    """Condition that makes one target eligible to spawn and be serviced."""

    service_index: int
    target_instance_id: str
    gate_kind: ServiceGateKind
    after_target_instance_id: str | None
    not_before_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.service_index, bool)
            or not isinstance(self.service_index, int)
            or self.service_index < 0
        ):
            raise ValueError("service_index must be a non-negative integer")
        if (
            not isinstance(self.target_instance_id, str)
            or not self.target_instance_id
        ):
            raise ValueError("target_instance_id cannot be empty")
        if not isinstance(self.gate_kind, ServiceGateKind):
            raise ValueError("gate_kind must be a ServiceGateKind")
        if (
            isinstance(self.not_before_s, bool)
            or not isinstance(self.not_before_s, Real)
            or not math.isfinite(self.not_before_s)
            or self.not_before_s < 0.0
        ):
            raise ValueError("not_before_s must be finite and non-negative")
        if self.gate_kind is ServiceGateKind.EPISODE_START:
            if self.service_index != 0 or self.after_target_instance_id is not None:
                raise ValueError(
                    "episode_start is valid only for the first service gate"
                )
        else:
            if self.service_index == 0:
                raise ValueError(
                    "previous_target_completed cannot gate the first target"
                )
            if (
                not isinstance(self.after_target_instance_id, str)
                or not self.after_target_instance_id
            ):
                raise ValueError(
                    "previous_target_completed requires a predecessor target"
                )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "service_index": self.service_index,
            "target_instance_id": self.target_instance_id,
            "gate_kind": self.gate_kind.value,
            "after_target_instance_id": self.after_target_instance_id,
            "not_before_s": float(self.not_before_s),
        }


@dataclass(frozen=True)
class V2TaskContext:
    """Resolved V2 suite context with a V1 canonical task manifest."""

    canonical_protocol_version: str
    benchmark_suite_version: str
    scene_id: SceneId
    task_family: TaskFamily
    task: TaskManifest
    target_sequence_ids: tuple[str, ...]
    distractor_ids: tuple[str, ...]
    destination_zone_by_target: Mapping[str, str]
    instance_asset_map: Mapping[str, str]
    service_gates: tuple[ServiceGate, ...]

    def __post_init__(self) -> None:
        if self.canonical_protocol_version != CANONICAL_PROTOCOL_VERSION:
            raise ValueError("V2 task context must use the V1 canonical protocol")
        if self.benchmark_suite_version != BENCHMARK_SUITE_VERSION:
            raise ValueError("benchmark_suite_version is inconsistent")
        if not isinstance(self.scene_id, SceneId):
            raise ValueError("scene_id must be a SceneId")
        if not isinstance(self.task_family, TaskFamily):
            raise ValueError("task_family must be a TaskFamily")
        if not isinstance(self.task, TaskManifest):
            raise ValueError("task must be a TaskManifest")
        if self.target_sequence_ids != self.task.scored_object_ids:
            raise ValueError("target sequence must match scored_object_ids")
        if not self.target_sequence_ids:
            raise ValueError("target_sequence_ids cannot be empty")
        object_ids = tuple(obj.instance_id for obj in self.task.objects)
        expected_map = {obj.instance_id: obj.asset_id for obj in self.task.objects}
        if dict(self.instance_asset_map) != expected_map:
            raise ValueError("instance_asset_map must match task objects")
        if any(instance_id not in object_ids for instance_id in self.distractor_ids):
            raise ValueError("distractor_ids must reference task objects")
        if set(self.destination_zone_by_target) != set(
            self.target_sequence_ids
        ):
            raise ValueError("every and only target requires a destination")
        if len(self.service_gates) != len(self.target_sequence_ids):
            raise ValueError("service gates must cover every target")
        for index, (target_id, gate) in enumerate(
            zip(self.target_sequence_ids, self.service_gates, strict=True)
        ):
            if gate.service_index != index or gate.target_instance_id != target_id:
                raise ValueError("service gates must follow target sequence order")
            predecessor = self.target_sequence_ids[index - 1] if index else None
            if gate.after_target_instance_id != predecessor:
                raise ValueError("service gate predecessor is inconsistent")
        suite = self.task.metadata.get("benchmark_suite")
        if not isinstance(suite, Mapping):
            raise ValueError("task metadata requires benchmark_suite")
        if suite.get("scene_id") != self.scene_id.value:
            raise ValueError("task suite scene_id is inconsistent")
        if suite.get("task_family") != self.task_family.value:
            raise ValueError("task suite task_family is inconsistent")


def _coerce_scene(
    value: SceneId | str,
    config: V2SuiteConfig,
) -> tuple[SceneId, SceneContract]:
    scene = config.scene(value)
    return scene.scene_id, scene


def _coerce_family(value: TaskFamily | str) -> TaskFamily:
    try:
        return value if isinstance(value, TaskFamily) else TaskFamily(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported V2 task_family: {value!r}") from error


def _coerce_mode(value: RobotMode | str) -> RobotMode:
    if value in (RobotMode.FIXED_BASE, "fixed_base"):
        return RobotMode.FIXED_BASE
    if value in (
        RobotMode.WHOLE_BODY_POLICY,
        "whole_body",
        "whole_body_policy",
    ):
        return RobotMode.WHOLE_BODY_POLICY
    raise ValueError("V2 robot mode must be fixed_base or whole_body_policy")


def validate_task_combination(
    scene_id: SceneId | str,
    family: TaskFamily | str,
    mode: RobotMode | str,
    *,
    config: V2SuiteConfig = DEFAULT_SUITE_CONFIG,
) -> tuple[SceneId, TaskFamily, RobotMode]:
    """Resolve and validate a suite combination without importing Isaac."""

    if not isinstance(config, V2SuiteConfig):
        raise TypeError("config must be a V2SuiteConfig")
    resolved_scene, _ = _coerce_scene(scene_id, config)
    resolved_family = _coerce_family(family)
    resolved_mode = _coerce_mode(mode)

    # Keep these messages specific because the CLI can surface them before
    # AppLauncher creates a simulation process.
    if (
        resolved_family is TaskFamily.CONTINUOUS_MULTI_TARGET
        and resolved_mode is not RobotMode.FIXED_BASE
    ):
        raise ValueError(
            "continuous_multi_target initially requires fixed_base"
        )
    if (
        resolved_scene is SceneId.MOBILE_REMOTE_DELIVERY_V2
        and resolved_mode is not RobotMode.WHOLE_BODY_POLICY
    ):
        raise ValueError("remote delivery requires whole_body_policy")
    if not config.supports(
        resolved_scene,
        resolved_family.value,
        resolved_mode.value,
    ):
        raise ValueError(
            "unsupported V2 scene/task/mode combination: "
            f"{resolved_scene.value}/{resolved_family.value}/"
            f"{resolved_mode.value}"
        )
    return resolved_scene, resolved_family, resolved_mode


def _destination_mapping(
    scene: SceneContract,
    original: Mapping[str, str],
    id_map: Mapping[str, str],
) -> dict[str, str]:
    zone_map = {
        "sort_bin_blue": (
            "delivery_bin_blue"
            if scene.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2
            else "sort_bin_blue"
        ),
        "sort_bin_yellow": (
            "delivery_bin_yellow"
            if scene.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2
            else "sort_bin_yellow"
        ),
    }
    return {
        id_map[old_instance_id]: zone_map[zone_id]
        for old_instance_id, zone_id in original.items()
    }


def _adapt_remote_language(value: str) -> str:
    replacements = {
        "blue sorting tray": "blue remote delivery bin",
        "yellow sorting tray": "yellow remote delivery bin",
        "蓝色分拣盘": "蓝色远端交付箱",
        "黄色分拣盘": "黄色远端交付箱",
    }
    result = value
    for source, destination in replacements.items():
        result = result.replace(source, destination)
    return result


def _service_gates(
    target_ids: tuple[str, ...],
    spawn_schedule: tuple[Mapping[str, Any], ...],
) -> tuple[ServiceGate, ...]:
    not_before_by_id = {
        str(entry["object_instance_id"]): float(entry["spawn_time_s"])
        for entry in spawn_schedule
    }
    return tuple(
        ServiceGate(
            service_index=index,
            target_instance_id=target_id,
            gate_kind=(
                ServiceGateKind.EPISODE_START
                if index == 0
                else ServiceGateKind.PREVIOUS_TARGET_COMPLETED
            ),
            after_target_instance_id=(target_ids[index - 1] if index else None),
            not_before_s=not_before_by_id[target_id],
        )
        for index, target_id in enumerate(target_ids)
    )


def build_task_context(
    *,
    seed: int,
    scene_id: SceneId | str,
    family: TaskFamily | str,
    mode: RobotMode | str,
    split: CurriculumSplit | str,
    instruction_language: InstructionLanguage | str = (
        InstructionLanguage.BILINGUAL
    ),
    config: V2SuiteConfig = DEFAULT_SUITE_CONFIG,
) -> V2TaskContext:
    """Build one deterministic, runtime-independent V2 task context."""

    resolved_scene_id, resolved_family, resolved_mode = (
        validate_task_combination(
            scene_id,
            family,
            mode,
            config=config,
        )
    )
    scene = config.scene(resolved_scene_id)
    base = build_episode_manifest(
        seed=seed,
        split=split,
        family=resolved_family,
        mode=resolved_mode,
        instruction_language=instruction_language,
        config=CurriculumConfig(
            belt_speeds_mps=config.belt_speeds_mps,
            max_duration_s=scene.default_max_duration_s,
        ),
    ).task

    id_map = {obj.instance_id: obj.asset_id for obj in base.objects}
    if len(set(id_map.values())) != len(id_map):
        raise ValueError("V2 tasking cannot repeat an asset within one episode")
    original_destinations = base.metadata["destination_zone_by_target"]
    if not isinstance(original_destinations, Mapping):
        raise ValueError("base task lacks destination_zone_by_target")
    destinations = _destination_mapping(
        scene,
        original_destinations,
        id_map,
    )
    old_target_ids = tuple(base.scored_object_ids)
    target_ids = tuple(id_map[instance_id] for instance_id in old_target_ids)
    old_distractor_ids = tuple(base.metadata["distractors"])
    distractor_ids = tuple(id_map[instance_id] for instance_id in old_distractor_ids)

    objects = tuple(
        ObjectInstance(
            instance_id=obj.asset_id,
            asset_id=obj.asset_id,
            class_id=obj.class_id,
            goal_zone_id=destinations.get(obj.asset_id),
        )
        for obj in base.objects
    )
    goal_zones = tuple(
        GoalZone(zone.zone_id, zone.min_xyz, zone.max_xyz)
        for zone in scene.goal_zones
    )

    raw_schedule = tuple(base.metadata["spawn_schedule"])
    normalized_schedule = tuple(
        {
            **dict(entry),
            "object_instance_id": id_map[str(entry["object_instance_id"])],
            "destination_zone_id": (
                destinations.get(id_map[str(entry["object_instance_id"])])
            ),
        }
        for entry in raw_schedule
    )
    gates = _service_gates(target_ids, normalized_schedule)
    gate_by_target = {gate.target_instance_id: gate for gate in gates}
    gated_schedule = tuple(
        {
            **entry,
            "service_gate": (
                gate_by_target[entry["object_instance_id"]].to_metadata()
                if entry["object_instance_id"] in gate_by_target
                else {
                    "gate_kind": ServiceGateKind.EPISODE_START.value,
                    "after_target_instance_id": None,
                    "not_before_s": float(entry["spawn_time_s"]),
                }
            ),
        }
        for entry in normalized_schedule
    )

    english = str(base.metadata["canonical_instruction_en"])
    chinese = str(base.metadata["canonical_instruction_zh"])
    instruction = base.instruction
    if resolved_scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
        english = _adapt_remote_language(english)
        chinese = _adapt_remote_language(chinese)
        instruction = _adapt_remote_language(instruction)

    destination_contracts = {
        zone.zone_id: zone.to_snapshot() for zone in scene.goal_zones
    }
    suite_metadata = {
        "schema_version": TASK_CONTEXT_SCHEMA_VERSION,
        "benchmark_suite_version": BENCHMARK_SUITE_VERSION,
        "canonical_protocol_version": CANONICAL_PROTOCOL_VERSION,
        "scene_id": resolved_scene_id.value,
        "layout_id": scene.layout_id,
        "task_family": resolved_family.value,
        "robot_mode": resolved_mode.value,
        "object_split": str(base.metadata["curriculum_split"]),
        "target_sequence_ids": target_ids,
        "destination_zone_by_target": dict(destinations),
        "spawn_policy": (
            "service_gated" if len(target_ids) > 1 else "episode_start"
        ),
        "service_gates": tuple(gate.to_metadata() for gate in gates),
        "destination_zone_contracts": destination_contracts,
        "minimum_loaded_base_displacement_m": (
            scene.minimum_loaded_base_displacement_m
        ),
    }
    metadata = {
        **dict(base.metadata),
        "tasking_schema_version": TASKING_SCHEMA_VERSION,
        "active_object_ids": tuple(obj.instance_id for obj in objects),
        "active_asset_ids": tuple(obj.asset_id for obj in objects),
        "target_id": target_ids[0],
        "target_ids": target_ids,
        "destination_zone": destinations[target_ids[0]],
        "destination_zone_by_target": dict(destinations),
        "distractors": distractor_ids,
        "spawn_schedule": gated_schedule,
        "canonical_instruction": instruction,
        "canonical_instruction_en": english,
        "canonical_instruction_zh": chinese,
        "layout_id": scene.layout_id,
        "scene_id": resolved_scene_id.value,
        "instance_asset_map": {
            obj.instance_id: obj.asset_id for obj in objects
        },
        "benchmark_suite": suite_metadata,
    }
    task = replace(
        base,
        task_id=(
            f"v2-{resolved_scene_id.value}-{resolved_family.value}-"
            f"{resolved_mode.value}-seed-{seed}"
        ),
        instruction=instruction,
        objects=objects,
        goal_zones=goal_zones,
        scored_object_ids=target_ids,
        max_duration_s=scene.default_max_duration_s,
        metadata=metadata,
    )
    return V2TaskContext(
        canonical_protocol_version=CANONICAL_PROTOCOL_VERSION,
        benchmark_suite_version=BENCHMARK_SUITE_VERSION,
        scene_id=resolved_scene_id,
        task_family=resolved_family,
        task=task,
        target_sequence_ids=target_ids,
        distractor_ids=distractor_ids,
        destination_zone_by_target=dict(destinations),
        instance_asset_map={obj.instance_id: obj.asset_id for obj in objects},
        service_gates=gates,
    )


def build_task_manifest(**kwargs: Any) -> TaskManifest:
    """Convenience projection for callers that need only the V1 task value."""

    return build_task_context(**kwargs).task


__all__ = [
    "ServiceGate",
    "ServiceGateKind",
    "V2TaskContext",
    "build_task_context",
    "build_task_manifest",
    "validate_task_combination",
]
