from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from conveyor_bench.v1.validation import ValidationResult
from conveyor_bench.v2.camera_contracts import camera_contract_for_scene
from conveyor_bench.v2.config import DEFAULT_SUITE_CONFIG
from conveyor_bench.v2 import validation


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _suite(
    *,
    scene_id: str,
    task_family: str,
    robot_mode: str,
    targets: tuple[str, ...],
) -> dict:
    scene = DEFAULT_SUITE_CONFIG.scene(scene_id)
    zone_ids = tuple(zone.zone_id for zone in scene.goal_zones)
    destinations = {
        target_id: zone_ids[index % len(zone_ids)]
        for index, target_id in enumerate(targets)
    }
    return {
        "schema_version": "conveyor-bench-v2-task-context-1",
        "benchmark_suite_version": "conveyor-bench-v2",
        "canonical_protocol_version": "conveyor-bench-v1",
        "scene_id": scene_id,
        "layout_id": scene.layout_id,
        "task_family": task_family,
        "robot_mode": robot_mode,
        "object_split": "train",
        "target_sequence_ids": list(targets),
        "destination_zone_by_target": destinations,
        "spawn_policy": (
            "service_gated" if len(targets) > 1 else "episode_start"
        ),
        "service_gates": [
            {
                "service_index": index,
                "target_instance_id": target_id,
                "gate_kind": (
                    "episode_start"
                    if index == 0
                    else "previous_target_completed"
                ),
                "after_target_instance_id": (
                    None if index == 0 else targets[index - 1]
                ),
                "not_before_s": index * 0.5,
            }
            for index, target_id in enumerate(targets)
        ],
        "destination_zone_contracts": {
            zone.zone_id: zone.to_snapshot() for zone in scene.goal_zones
        },
        "minimum_loaded_base_displacement_m": (
            0.65 if scene_id == "mobile_remote_delivery_v2" else 0.0
        ),
    }


def _make_episode(
    root: Path,
    *,
    scene_id: str = "transverse_near_sort_v2",
    task_family: str = "single_target",
    robot_mode: str = "fixed_base",
    targets: tuple[str, ...] = ("target-a",),
    success: bool = True,
) -> tuple[Path, dict]:
    episode = root / "ep-v2"
    episode.mkdir()
    suite = _suite(
        scene_id=scene_id,
        task_family=task_family,
        robot_mode=robot_mode,
        targets=targets,
    )
    destinations = suite["destination_zone_by_target"]
    gates_by_target = {
        gate["target_instance_id"]: gate
        for gate in suite["service_gates"]
    }
    scene = DEFAULT_SUITE_CONFIG.scene(scene_id)
    distractors = (
        ("distractor-a",)
        if task_family == "language_conditioned"
        else ()
    )
    object_ids = targets + distractors
    task = {
        "task_id": "task-v2",
        "task_type": (
            "continuous_sort"
            if task_family == "continuous_multi_target"
            else "dynamic_sort"
        ),
        "robot_mode": robot_mode,
        "belt_speed_mps": 0.06,
        "max_duration_s": 45.0,
        "goal_zones": [
            {
                "zone_id": zone.zone_id,
                "min_xyz": list(zone.min_xyz),
                "max_xyz": list(zone.max_xyz),
            }
            for zone in scene.goal_zones
        ],
        "objects": [
            {
                "instance_id": object_id,
                "asset_id": f"asset-{object_id}",
                "class_id": "part",
                "goal_zone_id": destinations.get(object_id),
            }
            for object_id in object_ids
        ],
        "scored_object_ids": list(targets),
        "metadata": {
            "task_family": task_family,
            "curriculum_split": "train",
            "scene_id": scene_id,
            "layout_id": suite["layout_id"],
            "belt_speed_mps": 0.06,
            "target_ids": list(targets),
            "distractors": list(distractors),
            "destination_zone_by_target": destinations,
            "instance_asset_map": {
                object_id: f"asset-{object_id}" for object_id in object_ids
            },
            "spawn_schedule": [
                {
                    "object_instance_id": target_id,
                    "asset_id": f"asset-{target_id}",
                    "destination_zone_id": destinations[target_id],
                    "role": "target",
                    "spawn_time_s": gates_by_target[target_id][
                        "not_before_s"
                    ],
                    "initialization_end_s": (
                        gates_by_target[target_id]["not_before_s"] + 0.25
                    ),
                    "service_gate": deepcopy(gates_by_target[target_id]),
                }
                for target_id in targets
            ]
            + [
                {
                    "object_instance_id": distractor_id,
                    "asset_id": f"asset-{distractor_id}",
                    "destination_zone_id": None,
                    "role": "distractor",
                    "spawn_time_s": 1.5,
                    "initialization_end_s": 2.25,
                    "service_gate": {
                        "gate_kind": "episode_start",
                        "after_target_instance_id": None,
                        "not_before_s": 1.5,
                    },
                }
                for distractor_id in distractors
            ],
            "benchmark_suite": suite,
        },
    }
    manifest = {
        "episode": {
            "protocol_version": "conveyor-bench-v1",
            "task": task,
            "metadata": {
                "benchmark_suite": deepcopy(suite),
                "cameras": camera_contract_for_scene(scene_id),
            },
        }
    }
    _write_json(episode / "manifest.json", manifest)
    _write_json(
        episode / "summary.json",
        {
            "success": success,
            "failure_reason": "none" if success else "target_missed",
        },
    )
    _write_jsonl(episode / "steps.jsonl", [])
    _write_jsonl(episode / "objects.jsonl", [])
    _write_jsonl(episode / "events.jsonl", [])
    return episode, manifest


def _stub_v1_ok(monkeypatch) -> None:
    monkeypatch.setattr(
        validation,
        "validate_v1_episode",
        lambda _path: ValidationResult(episode_count=1),
    )


def test_v2_validator_runs_v1_first_and_preserves_canonical_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(tmp_path, success=False)
    calls: list[Path] = []

    def fake_v1(path: str | Path) -> ValidationResult:
        calls.append(Path(path))
        return ValidationResult(errors=["canonical failure"])

    monkeypatch.setattr(validation, "validate_v1_episode", fake_v1)

    result = validation.validate_v2_episode(episode)

    assert calls == [episode]
    assert "canonical failure" in result.errors


def test_v2_validator_accepts_supported_metadata_and_rejects_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, manifest = _make_episode(tmp_path, success=False)
    _stub_v1_ok(monkeypatch)

    assert validation.validate_v2_episode(episode).ok

    manifest["episode"]["metadata"]["benchmark_suite"]["scene_id"] = (
        "mobile_remote_delivery_v2"
    )
    _write_json(episode / "manifest.json", manifest)
    result = validation.validate_v2_episode(episode)
    assert any("benchmark_suite mirror" in error for error in result.errors)


def test_v2_validator_rejects_frozen_scene_and_schedule_mirror_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, manifest = _make_episode(tmp_path, success=False)
    _stub_v1_ok(monkeypatch)
    task_metadata = manifest["episode"]["task"]["metadata"]
    suite = task_metadata["benchmark_suite"]

    suite["layout_id"] = "drifted-layout"
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(suite)
    _write_json(episode / "manifest.json", manifest)
    result = validation.validate_v2_episode(episode)
    assert any("layout_id does not match" in error for error in result.errors)

    suite["layout_id"] = DEFAULT_SUITE_CONFIG.scene(
        suite["scene_id"]
    ).layout_id
    suite["destination_zone_contracts"]["sort_bin_blue"][
        "center_xyz_m"
    ][0] += 0.01
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(suite)
    _write_json(episode / "manifest.json", manifest)
    result = validation.validate_v2_episode(episode)
    assert any(
        "destination_zone_contracts do not match" in error
        for error in result.errors
    )

    suite["destination_zone_contracts"] = {
        zone.zone_id: zone.to_snapshot()
        for zone in DEFAULT_SUITE_CONFIG.scene(suite["scene_id"]).goal_zones
    }
    task_metadata["spawn_schedule"][0]["spawn_time_s"] += 0.1
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(suite)
    _write_json(episode / "manifest.json", manifest)
    result = validation.validate_v2_episode(episode)
    assert any(
        "must equal gate not_before_s" in error for error in result.errors
    )


def test_v2_validator_rejects_task_semantics_and_camera_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, clean_manifest = _make_episode(tmp_path, success=False)
    _stub_v1_ok(monkeypatch)

    mutations = (
        (
            lambda manifest: manifest["episode"]["task"]["goal_zones"][0][
                "min_xyz"
            ].__setitem__(0, -99.0),
            "task.goal_zones do not match",
        ),
        (
            lambda manifest: manifest["episode"]["task"].__setitem__(
                "task_type", "continuous_sort"
            ),
            "task_type must be",
        ),
        (
            lambda manifest: manifest["episode"]["task"]["metadata"].__setitem__(
                "scene_id", "mobile_remote_delivery_v2"
            ),
            "task metadata scene_id does not match",
        ),
        (
            lambda manifest: manifest["episode"]["task"].__setitem__(
                "belt_speed_mps", 0.123
            ),
            "belt_speed_mps must match a frozen V2 speed",
        ),
        (
            lambda manifest: manifest["episode"]["metadata"]["cameras"][
                "overview_rgb"
            ].__setitem__("role", "policy_observation"),
            "camera contract does not match",
        ),
        (
            lambda manifest: manifest["episode"]["metadata"]["cameras"][
                "wrist_rgb"
            ]["mount"]["xyz_m"].__setitem__(2, 9.0),
            "camera contract does not match",
        ),
    )

    for mutate, expected_error in mutations:
        manifest = deepcopy(clean_manifest)
        mutate(manifest)
        _write_json(episode / "manifest.json", manifest)
        result = validation.validate_v2_episode(episode)
        assert any(expected_error in error for error in result.errors)


def test_v2_validator_reports_null_targets_and_invalid_duration_without_crash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, clean_manifest = _make_episode(tmp_path, success=False)
    _stub_v1_ok(monkeypatch)

    for field_path in (("scored_object_ids",), ("metadata", "target_ids")):
        manifest = deepcopy(clean_manifest)
        value = manifest["episode"]["task"]
        for key in field_path[:-1]:
            value = value[key]
        value[field_path[-1]] = None
        _write_json(episode / "manifest.json", manifest)
        result = validation.validate_v2_episode(episode)
        assert not result.ok
        assert any("target_sequence_ids do not match" in e for e in result.errors)

    manifest = deepcopy(clean_manifest)
    schedule = manifest["episode"]["task"]["metadata"]["spawn_schedule"]
    manifest["episode"]["task"]["max_duration_s"] = schedule[0][
        "initialization_end_s"
    ]
    _write_json(episode / "manifest.json", manifest)
    result = validation.validate_v2_episode(episode)
    assert any("max_duration_s must exceed" in e for e in result.errors)


def test_v2_validator_fails_closed_on_unsupported_scene_task_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        scene_id="mobile_remote_delivery_v2",
        robot_mode="fixed_base",
        success=False,
    )
    _stub_v1_ok(monkeypatch)

    result = validation.validate_v2_episode(episode)

    assert any("unsupported scene/task/mode" in error for error in result.errors)


def test_continuous_success_requires_selection_and_placement_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    targets = ("target-a", "target-b")
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=targets,
    )
    _stub_v1_ok(monkeypatch)
    steps = [
        {"sim_step": 0, "selected_object_id": "target-a"},
        {"sim_step": 1, "selected_object_id": "target-a"},
        {"sim_step": 2, "selected_object_id": None},
        {"sim_step": 3, "selected_object_id": "target-b"},
    ]
    events = [
        {
            "kind": "target_selected",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_placed",
            "time_s": 0.25,
            "object_instance_id": "target-a",
        },
        {
            "kind": "target_selected",
            "time_s": 0.25,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.5,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_placed",
            "time_s": 0.75,
            "object_instance_id": "target-b",
        },
    ]
    _write_jsonl(episode / "steps.jsonl", steps)
    _write_jsonl(episode / "events.jsonl", events)

    assert validation.validate_v2_episode(episode).ok

    steps[-1]["selected_object_id"] = "target-a"
    events[3]["object_instance_id"] = "target-a"
    events[2]["object_instance_id"] = "target-b"
    events[5]["object_instance_id"] = "target-a"
    _write_jsonl(episode / "steps.jsonl", steps)
    _write_jsonl(episode / "events.jsonl", events)
    result = validation.validate_v2_episode(episode)
    assert any("selected_object_id order" in error for error in result.errors)
    assert any("target_selected target order" in error for error in result.errors)
    assert any("object_placed target order" in error for error in result.errors)


def test_continuous_service_gate_rejects_early_spawn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=("target-a", "target-b"),
    )
    _stub_v1_ok(monkeypatch)
    events = [
        {
            "kind": "target_selected",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_placed",
            "time_s": 0.2,
            "object_instance_id": "target-a",
        },
        {
            "kind": "target_selected",
            "time_s": 0.2,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.49,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_placed",
            "time_s": 0.75,
            "object_instance_id": "target-b",
        },
    ]
    _write_jsonl(episode / "events.jsonl", events)

    result = validation.validate_v2_episode(episode)

    assert any("before service gate not_before_s=0.5" in error for error in result.errors)


def test_continuous_selection_must_precede_same_time_spawn_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=("target-a", "target-b"),
        success=False,
    )
    _stub_v1_ok(monkeypatch)
    _write_jsonl(
        episode / "events.jsonl",
        [
            {
                "kind": "object_spawned",
                "time_s": 0.0,
                "object_instance_id": "target-a",
            },
            {
                "kind": "target_selected",
                "time_s": 0.0,
                "object_instance_id": "target-a",
            },
        ],
    )

    result = validation.validate_v2_episode(episode)

    assert any(
        "target_selected for 'target-a' must precede its object_spawned event"
        in error
        for error in result.errors
    )


def test_continuous_service_gate_requires_previous_placement_first(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=("target-a", "target-b"),
    )
    _stub_v1_ok(monkeypatch)
    events = [
        {
            "kind": "target_selected",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "target_selected",
            "time_s": 0.25,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.5,
            "object_instance_id": "target-b",
        },
        {
            "kind": "object_placed",
            "time_s": 0.6,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_placed",
            "time_s": 0.75,
            "object_instance_id": "target-b",
        },
    ]
    _write_jsonl(episode / "events.jsonl", events)

    result = validation.validate_v2_episode(episode)

    assert any(
        "object_spawned for 'target-b' must not precede previous target "
        "object_placed"
        in error
        for error in result.errors
    )


def test_failed_continuous_episode_accepts_incomplete_valid_event_prefix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=("target-a", "target-b"),
        success=False,
    )
    _stub_v1_ok(monkeypatch)
    events = [
        {
            "kind": "target_selected",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "object_instance_id": "target-a",
        },
    ]
    _write_jsonl(episode / "events.jsonl", events)

    assert validation.validate_v2_episode(episode).ok

    events.append(
        {
            "kind": "target_selected",
            "time_s": 0.25,
            "object_instance_id": "target-b",
        }
    )
    _write_jsonl(episode / "events.jsonl", events)
    result = validation.validate_v2_episode(episode)
    assert any(
        "target_selected for 'target-b' must not precede previous target "
        "object_placed"
        in error
        for error in result.errors
    )


def test_single_success_does_not_invent_target_selected_requirement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(tmp_path)
    _stub_v1_ok(monkeypatch)
    _write_jsonl(
        episode / "events.jsonl",
        [
            {
                "kind": "object_spawned",
                "time_s": 0.0,
                "object_instance_id": "target-a",
            },
            {
                "kind": "object_placed",
                "time_s": 1.0,
                "object_instance_id": "target-a",
            },
        ],
    )

    assert validation.validate_v2_episode(episode).ok


def test_failed_episode_does_not_require_success_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        task_family="continuous_multi_target",
        targets=("target-a", "target-b"),
        success=False,
    )
    _stub_v1_ok(monkeypatch)

    assert validation.validate_v2_episode(episode).ok


def test_remote_success_requires_loaded_whole_body_displacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    episode, _ = _make_episode(
        tmp_path,
        scene_id="mobile_remote_delivery_v2",
        robot_mode="whole_body_policy",
    )
    _stub_v1_ok(monkeypatch)
    steps = [
        {
            "sim_step": index,
            "sim_time_s": index * 0.1,
            "robot_root_world": {
                "xyz": [x, 0.0, 0.3],
                "wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
        for index, x in enumerate((0.0, 0.30, 0.66, 0.70))
    ]
    objects = [
        {
            "sim_step": index,
            "state": {
                "instance_id": "target-a",
                "in_gripper": index < 3,
            },
        }
        for index in range(4)
    ]
    events = [
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "sim_step": 0,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_released",
            "time_s": 0.3,
            "sim_step": 3,
            "object_instance_id": "target-a",
        },
        {
            "kind": "object_placed",
            "time_s": 0.4,
            "sim_step": 4,
            "object_instance_id": "target-a",
        },
    ]
    _write_jsonl(episode / "steps.jsonl", steps)
    _write_jsonl(episode / "objects.jsonl", objects)
    _write_jsonl(episode / "events.jsonl", events)

    assert validation.validate_v2_episode(episode).ok

    steps[2]["robot_root_world"]["xyz"][0] = 0.64
    _write_jsonl(episode / "steps.jsonl", steps)
    result = validation.validate_v2_episode(episode)
    assert any("loaded base displacement" in error for error in result.errors)
