from __future__ import annotations

import json
from pathlib import Path

import pytest

from conveyor_bench.v1.config import PROTOCOL_VERSION
from conveyor_bench.v1.protocol import (
    EpisodeManifest,
    RobotMode,
    TaskType,
    to_jsonable,
)
from conveyor_bench.v1.tasking import (
    TASKING_SCHEMA_VERSION,
    TaskFamily,
    validate_episode_manifest,
)
from conveyor_bench.v2.config import (
    BENCHMARK_SUITE_VERSION,
    CANONICAL_PROTOCOL_VERSION,
    DEFAULT_SUITE_CONFIG,
    TASK_CONTEXT_SCHEMA_VERSION,
    SceneId,
)
from conveyor_bench.v2.tasking import (
    ServiceGateKind,
    build_task_context,
    validate_task_combination,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v2_snapshot_and_python_contract_have_one_canonical_shape() -> None:
    snapshot = json.loads(
        (PROJECT_ROOT / "configs" / "v2.json").read_text(encoding="utf-8")
    )

    assert CANONICAL_PROTOCOL_VERSION == PROTOCOL_VERSION == "conveyor-bench-v1"
    assert BENCHMARK_SUITE_VERSION == "conveyor-bench-v2"
    assert TASK_CONTEXT_SCHEMA_VERSION == "conveyor-bench-v2-task-context-1"
    assert {scene.value for scene in SceneId} == {
        "transverse_near_sort_v2",
        "mobile_remote_delivery_v2",
    }
    assert snapshot == DEFAULT_SUITE_CONFIG.to_snapshot()


def test_continuous_context_is_deterministic_and_service_gated() -> None:
    kwargs = {
        "seed": 81,
        "scene_id": SceneId.TRANSVERSE_NEAR_SORT_V2,
        "family": TaskFamily.CONTINUOUS_MULTI_TARGET,
        "mode": RobotMode.FIXED_BASE,
        "split": "train",
        "instruction_language": "en_zh",
    }
    first = build_task_context(**kwargs)
    second = build_task_context(**kwargs)

    assert first == second
    assert first.canonical_protocol_version == PROTOCOL_VERSION
    assert first.task.task_type is TaskType.CONTINUOUS_SORT
    assert first.task.robot_mode is RobotMode.FIXED_BASE
    assert len(first.target_sequence_ids) == 2
    assert first.target_sequence_ids == first.task.scored_object_ids
    assert not first.distractor_ids
    assert tuple(obj.instance_id for obj in first.task.objects) == tuple(
        obj.asset_id for obj in first.task.objects
    )
    assert first.instance_asset_map == {
        obj.instance_id: obj.asset_id for obj in first.task.objects
    }

    gates = first.service_gates
    assert tuple(gate.target_instance_id for gate in gates) == (
        first.target_sequence_ids
    )
    assert gates[0].gate_kind is ServiceGateKind.EPISODE_START
    assert gates[0].after_target_instance_id is None
    assert gates[1].gate_kind is ServiceGateKind.PREVIOUS_TARGET_COMPLETED
    assert gates[1].after_target_instance_id == first.target_sequence_ids[0]

    metadata = first.task.metadata
    suite = metadata["benchmark_suite"]
    assert metadata["tasking_schema_version"] == TASKING_SCHEMA_VERSION
    assert metadata["instance_asset_map"] == first.instance_asset_map
    assert suite["schema_version"] == TASK_CONTEXT_SCHEMA_VERSION
    assert suite["benchmark_suite_version"] == BENCHMARK_SUITE_VERSION
    assert suite["canonical_protocol_version"] == PROTOCOL_VERSION
    assert suite["scene_id"] == SceneId.TRANSVERSE_NEAR_SORT_V2.value
    assert suite["task_family"] == TaskFamily.CONTINUOUS_MULTI_TARGET.value
    assert suite["spawn_policy"] == "service_gated"
    assert tuple(suite["target_sequence_ids"]) == first.target_sequence_ids
    json.dumps(to_jsonable(first.task), ensure_ascii=False, allow_nan=False)
    validate_episode_manifest(
        EpisodeManifest(
            episode_id="v2-contract-compatibility",
            run_id="v2-contract",
            protocol_version=PROTOCOL_VERSION,
            task=first.task,
            created_at_utc="2000-01-01T00:00:00+00:00",
            seeds={"episode": first.task.seed},
            metadata=first.task.metadata,
        )
    )


def test_single_and_language_tasks_keep_distinct_target_distractor_semantics() -> None:
    single = build_task_context(
        seed=5,
        scene_id="transverse_near_sort_v2",
        family="single_target",
        mode="whole_body_policy",
        split="val",
    )
    language = build_task_context(
        seed=5,
        scene_id="transverse_near_sort_v2",
        family="language_conditioned",
        mode="whole_body_policy",
        split="val",
    )

    assert len(single.target_sequence_ids) == 1
    assert not single.distractor_ids
    assert len(single.task.objects) == 1
    assert len(language.target_sequence_ids) == 1
    assert len(language.distractor_ids) == 1
    assert len(language.task.objects) == 2
    assert language.task.metadata["distractors"] == language.distractor_ids
    assert language.task.instruction != single.task.instruction


@pytest.mark.parametrize(
    ("seed", "expected_zone"),
    ((1, "delivery_bin_blue"), (0, "delivery_bin_yellow")),
)
def test_remote_delivery_resolves_remote_goal_and_navigation_contract(
    seed: int,
    expected_zone: str,
) -> None:
    context = build_task_context(
        seed=seed,
        scene_id=SceneId.MOBILE_REMOTE_DELIVERY_V2,
        family=TaskFamily.SINGLE_TARGET,
        mode=RobotMode.WHOLE_BODY_POLICY,
        split="train",
        instruction_language="en_zh",
    )

    target_id = context.target_sequence_ids[0]
    suite = context.task.metadata["benchmark_suite"]
    destination = context.destination_zone_by_target[target_id]
    zone = context.task.goal_zone_by_id[destination]
    contract = suite["destination_zone_contracts"][destination]

    assert destination == expected_zone
    assert context.task.robot_mode is RobotMode.WHOLE_BODY_POLICY
    assert set(context.task.goal_zone_by_id) == {
        "delivery_bin_blue",
        "delivery_bin_yellow",
    }
    assert zone.min_xyz[1] < (-1.0 if destination.endswith("yellow") else 1.2)
    assert zone.max_xyz[1] > (-1.2 if destination.endswith("yellow") else 1.0)
    assert contract["delivery_root_goal_xy_m"] in (
        [-0.16, 0.78],
        [-0.16, -0.78],
    )
    assert contract["delivery_standoff_m"] == pytest.approx(0.42)
    assert suite["minimum_loaded_base_displacement_m"] == pytest.approx(0.65)
    assert "remote delivery bin" in context.task.metadata[
        "canonical_instruction_en"
    ]
    assert "远端交付箱" in context.task.metadata["canonical_instruction_zh"]


@pytest.mark.parametrize(
    ("scene_id", "family", "mode", "message"),
    (
        (
            "mobile_remote_delivery_v2",
            "single_target",
            "fixed_base",
            "remote delivery requires whole_body_policy",
        ),
        (
            "transverse_near_sort_v2",
            "continuous_multi_target",
            "whole_body_policy",
            "continuous_multi_target initially requires fixed_base",
        ),
        (
            "mobile_remote_delivery_v2",
            "continuous_multi_target",
            "whole_body_policy",
            "continuous_multi_target initially requires fixed_base",
        ),
    ),
)
def test_invalid_scene_family_mode_combinations_fail_before_runtime(
    scene_id: str,
    family: str,
    mode: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_task_combination(scene_id, family, mode)
    with pytest.raises(ValueError, match=message):
        build_task_context(
            seed=0,
            scene_id=scene_id,
            family=family,
            mode=mode,
            split="train",
        )


def test_unknown_scene_is_rejected_without_falling_back() -> None:
    with pytest.raises(ValueError, match="unsupported V2 scene_id"):
        build_task_context(
            seed=0,
            scene_id="warehouse_magic",
            family="single_target",
            mode="whole_body_policy",
            split="train",
        )
