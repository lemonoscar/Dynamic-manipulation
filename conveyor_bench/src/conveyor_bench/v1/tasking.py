"""Deterministic, oracle-independent episode curricula for ConveyorBench V1.

The generator resolves every task decision into an :class:`EpisodeManifest`:
object identities, destinations, language, belt speed, and non-overlapping
initialization windows.  Runtime code consumes the manifest; it does not need
to ask an oracle which object or destination is correct.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence

from .assets import (
    ObjectAsset,
    ReceptacleAsset,
    load_object_registry,
    load_receptacles,
    load_workcell_manifest,
)
from .config import PROTOCOL_VERSION
from .protocol import (
    EpisodeManifest,
    GoalZone,
    ObjectInstance,
    RobotMode,
    TaskManifest,
    TaskType,
)

TASKING_SCHEMA_VERSION = "conveyor-bench-tasking-v1"

# The registry owns the coarse seen/unseen split.  These stable project-local
# IDs freeze a non-leaking development split inside its six "seen" objects.
TRAIN_OBJECT_IDS = (
    "part_red_block",
    "part_blue_bar",
    "part_yellow_bushing",
    "part_green_shaft",
)
VAL_OBJECT_IDS = (
    "part_silver_hex",
    "part_orange_flange",
)
UNSEEN_OBJECT_IDS = (
    "part_purple_bracket",
    "part_cyan_gear",
)
SORT_ZONE_IDS = ("sort_bin_blue", "sort_bin_yellow")

_ZONE_LANGUAGE = {
    "sort_bin_blue": {
        "en": "blue sorting tray",
        "zh": "蓝色分拣盘",
    },
    "sort_bin_yellow": {
        "en": "yellow sorting tray",
        "zh": "黄色分拣盘",
    },
}
_GENERATED_AT_UTC = "2000-01-01T00:00:00+00:00"


class CurriculumSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    UNSEEN = "unseen"


class TaskFamily(str, Enum):
    SINGLE_TARGET = "single_target"
    LANGUAGE_CONDITIONED = "language_conditioned"
    CONTINUOUS_MULTI_TARGET = "continuous_multi_target"


class InstructionLanguage(str, Enum):
    ENGLISH = "en"
    BILINGUAL = "en_zh"


@dataclass(frozen=True)
class CurriculumConfig:
    """Small deterministic design space used by the task generator."""

    belt_speeds_mps: tuple[float, ...] = (0.08, 0.10, 0.12)
    first_spawn_time_s: float = 0.50
    initialization_window_s: float = 0.75
    spawn_gap_s: float = 0.25
    max_duration_s: float = 20.0
    generated_at_utc: str = _GENERATED_AT_UTC

    def __post_init__(self) -> None:
        if not self.belt_speeds_mps:
            raise ValueError("belt_speeds_mps cannot be empty")
        if any(
            isinstance(speed, bool)
            or not isinstance(speed, Real)
            or not math.isfinite(speed)
            or speed <= 0.0
            for speed in self.belt_speeds_mps
        ):
            raise ValueError("belt_speeds_mps must contain positive finite values")
        for name in (
            "first_spawn_time_s",
            "initialization_window_s",
            "spawn_gap_s",
            "max_duration_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.first_spawn_time_s < 0.0:
            raise ValueError("first_spawn_time_s cannot be negative")
        if self.initialization_window_s <= 0.0:
            raise ValueError("initialization_window_s must be positive")
        if self.spawn_gap_s < 0.0:
            raise ValueError("spawn_gap_s cannot be negative")
        if self.max_duration_s <= 0.0:
            raise ValueError("max_duration_s must be positive")
        if not isinstance(self.generated_at_utc, str) or not self.generated_at_utc:
            raise ValueError("generated_at_utc cannot be empty")


@dataclass(frozen=True)
class SpawnScheduleEntry:
    """One object's exclusive initialization window on the conveyor."""

    object_instance_id: str
    asset_id: str
    role: str
    spawn_time_s: float
    initialization_end_s: float
    destination_zone_id: str | None

    def __post_init__(self) -> None:
        if not self.object_instance_id or not self.asset_id:
            raise ValueError("spawn entries require object and asset IDs")
        if self.role not in {"target", "distractor"}:
            raise ValueError("spawn role must be target or distractor")
        for name in ("spawn_time_s", "initialization_end_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.initialization_end_s <= self.spawn_time_s:
            raise ValueError("initialization_end_s must follow spawn_time_s")
        if self.role == "target" and self.destination_zone_id is None:
            raise ValueError("target spawn entries require a destination")
        if self.role == "distractor" and self.destination_zone_id is not None:
            raise ValueError("distractor spawn entries cannot have a destination")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "object_instance_id": self.object_instance_id,
            "asset_id": self.asset_id,
            "role": self.role,
            "spawn_time_s": self.spawn_time_s,
            "initialization_end_s": self.initialization_end_s,
            "destination_zone_id": self.destination_zone_id,
        }


def split_object_ids(
    registry: Sequence[ObjectAsset] | None = None,
) -> dict[CurriculumSplit, tuple[str, ...]]:
    """Return and validate the frozen train/val/unseen registry partition."""

    assets = tuple(registry) if registry is not None else load_object_registry()
    registry_ids = {asset.object_id for asset in assets}
    expected_ids = set(TRAIN_OBJECT_IDS + VAL_OBJECT_IDS + UNSEEN_OBJECT_IDS)
    if registry_ids != expected_ids:
        missing = sorted(expected_ids - registry_ids)
        extra = sorted(registry_ids - expected_ids)
        raise ValueError(
            "object registry does not match V1 task split; "
            f"missing={missing}, extra={extra}"
        )
    if len(assets) != len(registry_ids):
        raise ValueError("object registry contains duplicate IDs")

    seen_ids = {asset.object_id for asset in assets if asset.split == "seen"}
    unseen_ids = {asset.object_id for asset in assets if asset.split == "unseen"}
    if seen_ids != set(TRAIN_OBJECT_IDS + VAL_OBJECT_IDS):
        raise ValueError(
            "train and val IDs must exactly partition registry seen assets"
        )
    if unseen_ids != set(UNSEEN_OBJECT_IDS):
        raise ValueError("unseen IDs must exactly match registry unseen assets")

    result = {
        CurriculumSplit.TRAIN: TRAIN_OBJECT_IDS,
        CurriculumSplit.VAL: VAL_OBJECT_IDS,
        CurriculumSplit.UNSEEN: UNSEEN_OBJECT_IDS,
    }
    split_sets = tuple(set(values) for values in result.values())
    if any(
        left & right
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    ):
        raise ValueError("curriculum object splits overlap")
    return result


def validate_spawn_schedule(entries: Sequence[SpawnScheduleEntry]) -> None:
    """Reject duplicate objects and overlapping initialization windows."""

    if not entries:
        raise ValueError("spawn schedule cannot be empty")
    object_ids = tuple(entry.object_instance_id for entry in entries)
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("spawn schedule object IDs must be unique")
    for previous, current in zip(entries, entries[1:]):
        if current.spawn_time_s < previous.spawn_time_s:
            raise ValueError("spawn schedule must be ordered by spawn_time_s")
        if current.spawn_time_s < previous.initialization_end_s:
            raise ValueError("spawn initialization windows cannot overlap")


def build_episode_manifest(
    *,
    seed: int,
    split: CurriculumSplit | str,
    family: TaskFamily | str,
    mode: RobotMode | str = RobotMode.FIXED_BASE,
    instruction_language: InstructionLanguage | str = InstructionLanguage.ENGLISH,
    config: CurriculumConfig | None = None,
    run_id: str | None = None,
) -> EpisodeManifest:
    """Resolve one deterministic task into the existing V1 protocol."""

    _validate_seed(seed)
    resolved_split = _coerce_split(split)
    resolved_family = _coerce_family(family)
    robot_mode, mode_label = _coerce_mode(mode)
    language = _coerce_language(instruction_language)
    resolved_config = config or CurriculumConfig()

    registry = load_object_registry()
    assets_by_id = {asset.object_id: asset for asset in registry}
    pool_ids = split_object_ids(registry)[resolved_split]
    ordered_ids = _deterministic_order(
        pool_ids,
        seed=seed,
        namespace=f"{resolved_split.value}:{resolved_family.value}:objects",
    )

    if resolved_family is TaskFamily.SINGLE_TARGET:
        target_asset_ids = ordered_ids[:1]
        distractor_asset_ids: tuple[str, ...] = ()
    elif resolved_family is TaskFamily.LANGUAGE_CONDITIONED:
        if len(ordered_ids) < 2:
            raise ValueError(
                "language-conditioned tasks require two split-local assets"
            )
        target_asset_ids = ordered_ids[:1]
        distractor_asset_ids = ordered_ids[1:2]
    else:
        if len(ordered_ids) < 2:
            raise ValueError("continuous tasks require two split-local assets")
        target_asset_ids = ordered_ids[:2]
        distractor_asset_ids = ()

    target_assets = tuple(assets_by_id[asset_id] for asset_id in target_asset_ids)
    distractor_assets = tuple(
        assets_by_id[asset_id] for asset_id in distractor_asset_ids
    )
    active_assets = target_assets + distractor_assets

    destination_by_index = _destinations_for_targets(seed, len(target_assets))
    objects: list[ObjectInstance] = []
    destination_by_target: dict[str, str] = {}
    target_ids: list[str] = []
    distractor_ids: list[str] = []
    for index, asset in enumerate(active_assets):
        instance_id = f"obj-{index:02d}-{asset.object_id}"
        if index < len(target_assets):
            destination = destination_by_index[index]
            target_ids.append(instance_id)
            destination_by_target[instance_id] = destination
        else:
            destination = None
            distractor_ids.append(instance_id)
        objects.append(
            ObjectInstance(
                instance_id=instance_id,
                asset_id=asset.object_id,
                class_id=asset.category,
                goal_zone_id=destination,
            )
        )

    english, chinese = _canonical_instructions(
        resolved_family,
        target_assets,
        distractor_assets,
        tuple(destination_by_index),
    )
    canonical_instruction = (
        english
        if language is InstructionLanguage.ENGLISH
        else f"[EN] {english} [ZH] {chinese}"
    )

    schedule = _build_spawn_schedule(
        tuple(objects),
        frozenset(target_ids),
        destination_by_target,
        resolved_config,
    )
    belt_speed = resolved_config.belt_speeds_mps[
        _digest_int(seed, "belt-speed") % len(resolved_config.belt_speeds_mps)
    ]
    goal_zones, belt_surface_z, transport_direction, exit_plane = (
        _load_task_geometry()
    )

    target_id = target_ids[0]
    destination_zone = destination_by_target[target_id]
    metadata: dict[str, Any] = {
        "tasking_schema_version": TASKING_SCHEMA_VERSION,
        "curriculum_split": resolved_split.value,
        "registry_split": (
            "unseen" if resolved_split is CurriculumSplit.UNSEEN else "seen"
        ),
        "task_family": resolved_family.value,
        "mode": mode_label,
        "instruction_language": language.value,
        "canonical_instruction": canonical_instruction,
        "canonical_instruction_en": english,
        "canonical_instruction_zh": chinese,
        "active_object_ids": tuple(obj.instance_id for obj in objects),
        "active_asset_ids": tuple(obj.asset_id for obj in objects),
        "target_id": target_id,
        "target_ids": tuple(target_ids),
        "destination_zone": destination_zone,
        "destination_zone_by_target": dict(destination_by_target),
        "distractors": tuple(distractor_ids),
        "distractor_asset_ids": tuple(distractor_asset_ids),
        "belt_speed_mps": float(belt_speed),
        "spawn_schedule": tuple(entry.to_metadata() for entry in schedule),
    }

    seed_label = str(seed)
    language_label = language.value.replace("_", "-")
    episode_stem = (
        f"{resolved_split.value}-{resolved_family.value.replace('_', '-')}-"
        f"{mode_label.replace('_', '-')}-{language_label}-{seed_label}"
    )
    task = TaskManifest(
        task_id=f"task-{episode_stem}",
        task_type=(
            TaskType.CONTINUOUS_SORT
            if resolved_family is TaskFamily.CONTINUOUS_MULTI_TARGET
            else TaskType.DYNAMIC_SORT
        ),
        robot_mode=robot_mode,
        instruction=canonical_instruction,
        objects=tuple(objects),
        goal_zones=goal_zones,
        scored_object_ids=tuple(target_ids),
        seed=seed,
        belt_speed_mps=float(belt_speed),
        belt_surface_z_m=belt_surface_z,
        transport_direction_xyz=transport_direction,
        exit_plane_point_xyz=exit_plane,
        max_duration_s=resolved_config.max_duration_s,
        metadata=metadata,
    )
    episode = EpisodeManifest(
        episode_id=f"episode-{episode_stem}",
        run_id=run_id or f"curriculum-{resolved_split.value}-{seed_label}",
        protocol_version=PROTOCOL_VERSION,
        task=task,
        created_at_utc=resolved_config.generated_at_utc,
        seeds={
            "episode": seed,
            "task_selection": seed,
            "spawn_schedule": seed,
        },
        metadata=dict(metadata),
    )
    validate_episode_manifest(episode)
    return episode


def build_smoke_suite(
    *,
    base_seed: int = 7300,
    config: CurriculumConfig | None = None,
) -> tuple[EpisodeManifest, ...]:
    """Build 18 tiny manifests: every split/family/mode combination once."""

    _validate_seed(base_seed)
    resolved_config = config or CurriculumConfig()
    run_id = f"curriculum-smoke-{base_seed}"
    episodes: list[EpisodeManifest] = []
    ordinal = 0
    for split_index, split in enumerate(CurriculumSplit):
        for family_index, family in enumerate(TaskFamily):
            for mode_index, mode in enumerate(("fixed_base", "whole_body")):
                language = (
                    InstructionLanguage.ENGLISH
                    if (split_index + family_index + mode_index) % 2 == 0
                    else InstructionLanguage.BILINGUAL
                )
                episodes.append(
                    build_episode_manifest(
                        seed=base_seed + ordinal,
                        split=split,
                        family=family,
                        mode=mode,
                        instruction_language=language,
                        config=resolved_config,
                        run_id=run_id,
                    )
                )
                ordinal += 1
    result = tuple(episodes)
    validate_curriculum_suite(result)
    return result


def validate_episode_manifest(episode: EpisodeManifest) -> None:
    """Validate tasking metadata against the resolved protocol task."""

    task = episode.task
    metadata = task.metadata
    split = _coerce_split(metadata.get("curriculum_split", ""))
    family = _coerce_family(metadata.get("task_family", ""))
    _, expected_mode_label = _coerce_mode(task.robot_mode)
    if metadata.get("mode") != expected_mode_label:
        raise ValueError("tasking mode metadata does not match robot_mode")
    if episode.metadata != metadata:
        raise ValueError("episode and task tasking metadata must match")
    if task.seed != episode.seeds.get("episode"):
        raise ValueError("episode seed does not match task seed")
    if metadata.get("canonical_instruction") != task.instruction:
        raise ValueError("canonical instruction does not match task instruction")
    if metadata.get("belt_speed_mps") != task.belt_speed_mps:
        raise ValueError("belt speed metadata does not match task")

    object_ids = tuple(obj.instance_id for obj in task.objects)
    asset_ids = tuple(obj.asset_id for obj in task.objects)
    if tuple(metadata.get("active_object_ids", ())) != object_ids:
        raise ValueError("active_object_ids do not match task objects")
    if tuple(metadata.get("active_asset_ids", ())) != asset_ids:
        raise ValueError("active_asset_ids do not match task assets")
    allowed_assets = set(split_object_ids()[split])
    if not set(asset_ids) <= allowed_assets:
        raise ValueError("episode contains an asset from another curriculum split")

    target_ids = tuple(metadata.get("target_ids", ()))
    if target_ids != task.scored_object_ids:
        raise ValueError("target_ids do not match scored_object_ids")
    if not target_ids or metadata.get("target_id") != target_ids[0]:
        raise ValueError("target_id must identify the primary scored object")
    destination_by_target = metadata.get("destination_zone_by_target")
    if not isinstance(destination_by_target, Mapping):
        raise ValueError("destination_zone_by_target must be a mapping")
    if set(destination_by_target) != set(target_ids):
        raise ValueError("every and only target IDs require destinations")
    if metadata.get("destination_zone") != destination_by_target[target_ids[0]]:
        raise ValueError("destination_zone must identify the primary destination")

    object_by_id = task.object_by_id
    for target_id in target_ids:
        if object_by_id[target_id].goal_zone_id != destination_by_target[target_id]:
            raise ValueError("target destination does not match ObjectInstance")
    distractor_ids = tuple(metadata.get("distractors", ()))
    expected_distractors = tuple(
        object_id for object_id in object_ids if object_id not in set(target_ids)
    )
    if distractor_ids != expected_distractors:
        raise ValueError("distractors do not match non-target task objects")
    if any(
        object_by_id[object_id].goal_zone_id is not None
        for object_id in distractor_ids
    ):
        raise ValueError("distractors cannot carry a scored destination")

    if family is TaskFamily.SINGLE_TARGET:
        if len(target_ids) != 1 or distractor_ids:
            raise ValueError("single_target requires one target and no distractors")
        if task.task_type not in {
            TaskType.STATIONARY_SORT,
            TaskType.DYNAMIC_SORT,
        }:
            raise ValueError(
                "single_target must use stationary_sort or dynamic_sort"
            )
    elif family is TaskFamily.LANGUAGE_CONDITIONED:
        if len(target_ids) != 1 or not distractor_ids:
            raise ValueError(
                "language_conditioned requires one target and distractors"
            )
        if task.task_type is not TaskType.DYNAMIC_SORT:
            raise ValueError(
                "language_conditioned must use dynamic_sort"
            )
    else:
        if len(target_ids) < 2 or distractor_ids:
            raise ValueError(
                "continuous_multi_target requires multiple targets and no distractors"
            )
        if task.task_type is not TaskType.CONTINUOUS_SORT:
            raise ValueError("continuous_multi_target must use continuous_sort")

    raw_schedule = metadata.get("spawn_schedule")
    if not isinstance(raw_schedule, Sequence) or isinstance(
        raw_schedule, (str, bytes)
    ):
        raise ValueError("spawn_schedule must be a sequence")
    schedule = tuple(_spawn_entry_from_metadata(value) for value in raw_schedule)
    validate_spawn_schedule(schedule)
    if tuple(entry.object_instance_id for entry in schedule) != object_ids:
        raise ValueError("spawn_schedule must cover active objects in task order")
    if schedule[-1].initialization_end_s >= task.max_duration_s:
        raise ValueError("spawn initialization must finish before task timeout")
    for entry in schedule:
        obj = object_by_id[entry.object_instance_id]
        if entry.asset_id != obj.asset_id:
            raise ValueError("spawn schedule asset does not match task object")
        expected_role = (
            "target" if entry.object_instance_id in target_ids else "distractor"
        )
        if entry.role != expected_role:
            raise ValueError("spawn schedule role does not match target identity")
        if entry.destination_zone_id != obj.goal_zone_id:
            raise ValueError("spawn schedule destination does not match task object")


def validate_curriculum_suite(episodes: Sequence[EpisodeManifest]) -> None:
    """Validate manifests and assert split-local asset identities."""

    if not episodes:
        raise ValueError("curriculum suite cannot be empty")
    assets_by_split = {split: set() for split in CurriculumSplit}
    for episode in episodes:
        validate_episode_manifest(episode)
        split = _coerce_split(episode.task.metadata["curriculum_split"])
        assets_by_split[split].update(obj.asset_id for obj in episode.task.objects)

    for index, left in enumerate(CurriculumSplit):
        for right in tuple(CurriculumSplit)[index + 1 :]:
            leaked = assets_by_split[left] & assets_by_split[right]
            if leaked:
                raise ValueError(
                    f"curriculum split leakage between {left.value} and "
                    f"{right.value}: {sorted(leaked)}"
                )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _coerce_split(value: CurriculumSplit | str) -> CurriculumSplit:
    try:
        return value if isinstance(value, CurriculumSplit) else CurriculumSplit(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported curriculum split: {value!r}") from error


def _coerce_family(value: TaskFamily | str) -> TaskFamily:
    try:
        return value if isinstance(value, TaskFamily) else TaskFamily(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported task family: {value!r}") from error


def _coerce_language(
    value: InstructionLanguage | str,
) -> InstructionLanguage:
    if value == "bilingual":
        value = InstructionLanguage.BILINGUAL
    try:
        return (
            value
            if isinstance(value, InstructionLanguage)
            else InstructionLanguage(value)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"unsupported instruction language: {value!r}") from error


def _coerce_mode(value: RobotMode | str) -> tuple[RobotMode, str]:
    if value in (RobotMode.FIXED_BASE, "fixed_base"):
        return RobotMode.FIXED_BASE, "fixed_base"
    if value in (
        RobotMode.WHOLE_BODY_POLICY,
        "whole_body",
        "whole_body_policy",
    ):
        return RobotMode.WHOLE_BODY_POLICY, "whole_body"
    raise ValueError("tasking mode must be fixed_base or whole_body")


def _digest_int(seed: int, namespace: str, value: str = "") -> int:
    encoded = (
        f"{TASKING_SCHEMA_VERSION}|{seed}|{namespace}|{value}".encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(encoded).digest(), byteorder="big")


def _deterministic_order(
    values: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            values,
            key=lambda value: (_digest_int(seed, namespace, value), value),
        )
    )


def _destinations_for_targets(seed: int, target_count: int) -> tuple[str, ...]:
    offset = _digest_int(seed, "destination-zone") % len(SORT_ZONE_IDS)
    return tuple(
        SORT_ZONE_IDS[(offset + index) % len(SORT_ZONE_IDS)]
        for index in range(target_count)
    )


def _alias(asset: ObjectAsset, language: str) -> str:
    aliases = asset.language_aliases.get(language)
    if not aliases or not aliases[0].strip():
        raise ValueError(f"{asset.object_id} lacks a canonical {language} alias")
    return aliases[0].strip()


def _canonical_instructions(
    family: TaskFamily,
    targets: Sequence[ObjectAsset],
    distractors: Sequence[ObjectAsset],
    destinations: Sequence[str],
) -> tuple[str, str]:
    if family is TaskFamily.SINGLE_TARGET:
        target = targets[0]
        destination = destinations[0]
        return (
            f"Pick the {_alias(target, 'en')} from the moving conveyor and "
            f"place it in the {_ZONE_LANGUAGE[destination]['en']}.",
            f"从移动传送带上抓取{_alias(target, 'zh')}，并将其放入"
            f"{_ZONE_LANGUAGE[destination]['zh']}。",
        )
    if family is TaskFamily.LANGUAGE_CONDITIONED:
        target = targets[0]
        distractor = distractors[0]
        destination = destinations[0]
        return (
            f"When the {_alias(target, 'en')} and {_alias(distractor, 'en')} "
            f"appear, pick the {_alias(target, 'en')} and place it in the "
            f"{_ZONE_LANGUAGE[destination]['en']}; ignore the other object.",
            f"当{_alias(target, 'zh')}和{_alias(distractor, 'zh')}出现时，"
            f"抓取{_alias(target, 'zh')}并将其放入"
            f"{_ZONE_LANGUAGE[destination]['zh']}；忽略另一个物体。",
        )

    english_steps = tuple(
        f"the {_alias(target, 'en')} into the {_ZONE_LANGUAGE[destination]['en']}"
        for target, destination in zip(targets, destinations, strict=True)
    )
    chinese_steps = tuple(
        f"将{_alias(target, 'zh')}放入{_ZONE_LANGUAGE[destination]['zh']}"
        for target, destination in zip(targets, destinations, strict=True)
    )
    return (
        f"Sort {english_steps[0]}, then {english_steps[1]}.",
        f"依次{chinese_steps[0]}，然后{chinese_steps[1]}。",
    )


def _build_spawn_schedule(
    objects: Sequence[ObjectInstance],
    target_ids: frozenset[str],
    destination_by_target: Mapping[str, str],
    config: CurriculumConfig,
) -> tuple[SpawnScheduleEntry, ...]:
    interval = config.initialization_window_s + config.spawn_gap_s
    entries = tuple(
        SpawnScheduleEntry(
            object_instance_id=obj.instance_id,
            asset_id=obj.asset_id,
            role="target" if obj.instance_id in target_ids else "distractor",
            spawn_time_s=round(config.first_spawn_time_s + index * interval, 9),
            initialization_end_s=round(
                config.first_spawn_time_s
                + index * interval
                + config.initialization_window_s,
                9,
            ),
            destination_zone_id=destination_by_target.get(obj.instance_id),
        )
        for index, obj in enumerate(objects)
    )
    validate_spawn_schedule(entries)
    return entries


def _goal_zone(asset: ReceptacleAsset) -> GoalZone:
    minimum = tuple(
        center - half
        for center, half in zip(
            asset.center_xyz_m,
            asset.goal_half_extents_xyz_m,
            strict=True,
        )
    )
    maximum = tuple(
        center + half
        for center, half in zip(
            asset.center_xyz_m,
            asset.goal_half_extents_xyz_m,
            strict=True,
        )
    )
    return GoalZone(asset.zone_id, minimum, maximum)


def _load_task_geometry() -> tuple[
    tuple[GoalZone, ...],
    float,
    tuple[float, float, float],
    tuple[float, float, float],
]:
    receptacles = {asset.zone_id: asset for asset in load_receptacles()}
    if set(SORT_ZONE_IDS) - set(receptacles):
        raise ValueError("project-local sorting receptacles are missing")
    goal_zones = tuple(_goal_zone(receptacles[zone_id]) for zone_id in SORT_ZONE_IDS)

    design = load_workcell_manifest().get("design")
    if not isinstance(design, Mapping):
        raise ValueError("workcell manifest lacks design geometry")
    direction = tuple(float(value) for value in design["transport_axis_world"])
    center = tuple(float(value) for value in design["belt_center_xyz_m"])
    size = tuple(float(value) for value in design["belt_size_xyz_m"])
    belt_surface_z = float(design["belt_top_z_m"])
    half_length = 0.5 * sum(
        abs(axis) * extent for axis, extent in zip(direction, size, strict=True)
    )
    exit_plane = tuple(
        center[axis] + direction[axis] * half_length for axis in range(3)
    )
    exit_plane = (exit_plane[0], exit_plane[1], belt_surface_z)
    return goal_zones, belt_surface_z, direction, exit_plane


def _spawn_entry_from_metadata(value: Any) -> SpawnScheduleEntry:
    if not isinstance(value, Mapping):
        raise ValueError("spawn_schedule entries must be mappings")
    return SpawnScheduleEntry(
        object_instance_id=str(value.get("object_instance_id", "")),
        asset_id=str(value.get("asset_id", "")),
        role=str(value.get("role", "")),
        spawn_time_s=value.get("spawn_time_s"),
        initialization_end_s=value.get("initialization_end_s"),
        destination_zone_id=value.get("destination_zone_id"),
    )


# Readable aliases for callers that use "generate" rather than "build".
generate_episode_manifest = build_episode_manifest
generate_smoke_suite = build_smoke_suite
validate_suite = validate_curriculum_suite


__all__ = [
    "TASKING_SCHEMA_VERSION",
    "TRAIN_OBJECT_IDS",
    "VAL_OBJECT_IDS",
    "UNSEEN_OBJECT_IDS",
    "SORT_ZONE_IDS",
    "CurriculumConfig",
    "CurriculumSplit",
    "InstructionLanguage",
    "SpawnScheduleEntry",
    "TaskFamily",
    "build_episode_manifest",
    "build_smoke_suite",
    "generate_episode_manifest",
    "generate_smoke_suite",
    "split_object_ids",
    "validate_curriculum_suite",
    "validate_episode_manifest",
    "validate_spawn_schedule",
    "validate_suite",
]
