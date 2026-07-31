from dataclasses import replace
import json

import pytest

from conveyor_bench.v1.assets import load_object_registry
from conveyor_bench.v1.protocol import RobotMode, TaskType, to_jsonable
from conveyor_bench.v1.tasking import (
    SORT_ZONE_IDS,
    TRAIN_OBJECT_IDS,
    UNSEEN_OBJECT_IDS,
    VAL_OBJECT_IDS,
    CurriculumConfig,
    CurriculumSplit,
    InstructionLanguage,
    SpawnScheduleEntry,
    TaskFamily,
    build_episode_manifest,
    build_smoke_suite,
    split_object_ids,
    validate_curriculum_suite,
    validate_episode_manifest,
    validate_spawn_schedule,
)


def test_registry_partition_is_complete_disjoint_and_respects_seen_unseen() -> None:
    registry = load_object_registry()
    registry_by_id = {asset.object_id: asset for asset in registry}
    partitions = split_object_ids(registry)

    assert partitions == {
        CurriculumSplit.TRAIN: TRAIN_OBJECT_IDS,
        CurriculumSplit.VAL: VAL_OBJECT_IDS,
        CurriculumSplit.UNSEEN: UNSEEN_OBJECT_IDS,
    }
    split_sets = {split: set(ids) for split, ids in partitions.items()}
    assert len(split_sets[CurriculumSplit.TRAIN]) == 4
    assert len(split_sets[CurriculumSplit.VAL]) == 2
    assert len(split_sets[CurriculumSplit.UNSEEN]) == 2
    assert set.union(*split_sets.values()) == set(registry_by_id)
    assert not split_sets[CurriculumSplit.TRAIN] & split_sets[CurriculumSplit.VAL]
    assert not split_sets[CurriculumSplit.TRAIN] & split_sets[CurriculumSplit.UNSEEN]
    assert not split_sets[CurriculumSplit.VAL] & split_sets[CurriculumSplit.UNSEEN]
    assert all(
        registry_by_id[asset_id].split == "seen"
        for asset_id in TRAIN_OBJECT_IDS + VAL_OBJECT_IDS
    )
    assert all(
        registry_by_id[asset_id].split == "unseen"
        for asset_id in UNSEEN_OBJECT_IDS
    )


def test_episode_generation_is_deterministic_and_has_canonical_languages() -> None:
    kwargs = {
        "seed": 41,
        "split": CurriculumSplit.TRAIN,
        "family": TaskFamily.LANGUAGE_CONDITIONED,
        "mode": "whole_body",
        "instruction_language": InstructionLanguage.BILINGUAL,
    }

    first = build_episode_manifest(**kwargs)
    second = build_episode_manifest(**kwargs)

    assert first == second
    assert first.created_at_utc == "2000-01-01T00:00:00+00:00"
    assert first.task.robot_mode is RobotMode.WHOLE_BODY_POLICY
    assert first.task.metadata["mode"] == "whole_body"
    assert first.task.metadata["instruction_language"] == "en_zh"
    assert first.task.instruction.startswith("[EN] ")
    assert " [ZH] " in first.task.instruction
    assert first.task.metadata["canonical_instruction"] == first.task.instruction
    assert first.task.metadata["canonical_instruction_en"]
    assert first.task.metadata["canonical_instruction_zh"]
    assert first.task.seed == first.seeds["episode"] == 41

    english_only = build_episode_manifest(
        **{**kwargs, "instruction_language": "en"}
    )
    assert english_only.task.instruction == (
        english_only.task.metadata["canonical_instruction_en"]
    )
    assert english_only.episode_id != first.episode_id


@pytest.mark.parametrize(
    ("family", "target_count", "distractor_count", "task_type"),
    (
        (TaskFamily.SINGLE_TARGET, 1, 0, TaskType.DYNAMIC_SORT),
        (TaskFamily.LANGUAGE_CONDITIONED, 1, 1, TaskType.DYNAMIC_SORT),
        (
            TaskFamily.CONTINUOUS_MULTI_TARGET,
            2,
            0,
            TaskType.CONTINUOUS_SORT,
        ),
    ),
)
def test_families_resolve_correct_objects_destinations_and_distractors(
    family: TaskFamily,
    target_count: int,
    distractor_count: int,
    task_type: TaskType,
) -> None:
    episode = build_episode_manifest(
        seed=123,
        split="val",
        family=family,
        mode="fixed_base",
    )
    task = episode.task
    metadata = task.metadata
    targets = tuple(metadata["target_ids"])
    distractors = tuple(metadata["distractors"])
    destinations = metadata["destination_zone_by_target"]

    assert task.task_type is task_type
    assert task.robot_mode is RobotMode.FIXED_BASE
    assert len(targets) == target_count
    assert len(distractors) == distractor_count
    assert targets == task.scored_object_ids
    assert metadata["target_id"] == targets[0]
    assert metadata["destination_zone"] == destinations[targets[0]]
    assert set(destinations) == set(targets)
    assert set(destinations.values()) <= set(SORT_ZONE_IDS)
    assert len(task.goal_zones) == 2
    assert {zone.zone_id for zone in task.goal_zones} == set(SORT_ZONE_IDS)
    for target_id in targets:
        assert task.object_by_id[target_id].goal_zone_id == destinations[target_id]
    for distractor_id in distractors:
        assert task.object_by_id[distractor_id].goal_zone_id is None


def test_spawn_schedule_covers_active_objects_without_temporal_overlap() -> None:
    episode = build_episode_manifest(
        seed=7,
        split="unseen",
        family="continuous_multi_target",
        mode="whole_body",
    )
    schedule = episode.task.metadata["spawn_schedule"]

    assert tuple(entry["object_instance_id"] for entry in schedule) == tuple(
        episode.task.metadata["active_object_ids"]
    )
    assert {entry["asset_id"] for entry in schedule} == {
        obj.asset_id for obj in episode.task.objects
    }
    for previous, current in zip(schedule, schedule[1:]):
        assert previous["initialization_end_s"] <= current["spawn_time_s"]
    assert all(entry["role"] == "target" for entry in schedule)
    assert all(entry["destination_zone_id"] in SORT_ZONE_IDS for entry in schedule)

    overlapping = (
        SpawnScheduleEntry("obj-a", "part-a", "distractor", 0.0, 1.0, None),
        SpawnScheduleEntry("obj-b", "part-b", "distractor", 0.5, 1.5, None),
    )
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_spawn_schedule(overlapping)


def test_smoke_suite_covers_matrix_without_collecting_or_leaking_assets() -> None:
    suite = build_smoke_suite(base_seed=900)
    repeated = build_smoke_suite(base_seed=900)
    partitions = split_object_ids()

    assert suite == repeated
    assert len(suite) == 18
    assert len({episode.episode_id for episode in suite}) == 18
    assert len({episode.task.seed for episode in suite}) == 18
    assert {episode.run_id for episode in suite} == {"curriculum-smoke-900"}
    assert {
        episode.task.metadata["curriculum_split"] for episode in suite
    } == {split.value for split in CurriculumSplit}
    assert {episode.task.metadata["task_family"] for episode in suite} == {
        family.value for family in TaskFamily
    }
    assert {episode.task.metadata["mode"] for episode in suite} == {
        "fixed_base",
        "whole_body",
    }
    assert {episode.task.metadata["instruction_language"] for episode in suite} == {
        "en",
        "en_zh",
    }

    used_assets: dict[CurriculumSplit, set[str]] = {
        split: set() for split in CurriculumSplit
    }
    for episode in suite:
        split = CurriculumSplit(episode.task.metadata["curriculum_split"])
        episode_assets = {obj.asset_id for obj in episode.task.objects}
        assert episode_assets <= set(partitions[split])
        used_assets[split].update(episode_assets)
        assert tuple(episode.task.metadata["active_object_ids"]) == tuple(
            obj.instance_id for obj in episode.task.objects
        )
        json.dumps(to_jsonable(episode), ensure_ascii=False, allow_nan=False)

    assert not used_assets[CurriculumSplit.TRAIN] & used_assets[CurriculumSplit.VAL]
    assert not used_assets[CurriculumSplit.TRAIN] & used_assets[CurriculumSplit.UNSEEN]
    assert not used_assets[CurriculumSplit.VAL] & used_assets[CurriculumSplit.UNSEEN]
    validate_curriculum_suite(suite)


def test_generator_rejects_unsupported_modes_seeds_and_short_timeout() -> None:
    common = {
        "split": "train",
        "family": "single_target",
    }
    with pytest.raises(ValueError, match="non-negative integer"):
        build_episode_manifest(seed=-1, **common)
    with pytest.raises(ValueError, match="non-negative integer"):
        build_episode_manifest(seed=True, **common)
    with pytest.raises(ValueError, match="fixed_base or whole_body"):
        build_episode_manifest(
            seed=1,
            mode=RobotMode.MOBILE_KINEMATIC,
            **common,
        )
    with pytest.raises(ValueError, match="before task timeout"):
        build_episode_manifest(
            seed=1,
            config=CurriculumConfig(max_duration_s=1.0),
            **common,
        )


def test_manifest_validator_detects_cross_split_asset_tampering() -> None:
    episode = build_episode_manifest(
        seed=5,
        split="unseen",
        family="single_target",
    )
    original = episode.task.objects[0]
    tampered_object = replace(original, asset_id=TRAIN_OBJECT_IDS[0])
    tampered_metadata = {
        **episode.task.metadata,
        "active_asset_ids": (TRAIN_OBJECT_IDS[0],),
        "spawn_schedule": (
            {
                **episode.task.metadata["spawn_schedule"][0],
                "asset_id": TRAIN_OBJECT_IDS[0],
            },
        ),
    }
    tampered_task = replace(
        episode.task,
        objects=(tampered_object,),
        metadata=tampered_metadata,
    )
    tampered_episode = replace(
        episode,
        task=tampered_task,
        metadata=dict(tampered_metadata),
    )

    with pytest.raises(ValueError, match="another curriculum split"):
        validate_episode_manifest(tampered_episode)
