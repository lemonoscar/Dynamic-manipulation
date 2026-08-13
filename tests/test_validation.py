from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

from conveyor_bench.schema.tasking import (
    TASKING_SCHEMA_VERSION,
    TRAIN_OBJECT_IDS,
    UNSEEN_OBJECT_IDS,
)
from conveyor_bench.schema.stationary import (
    STATIONARY_DESTINATION_ZONE_ID,
    STATIONARY_SCENARIOS,
    STATIONARY_SPAWN_ORIGIN_XY_M,
    STATIONARY_TARGET_ASSET_ID,
)
from conveyor_bench.schema.validation import validate_v1_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate.py"
HORIZONS = [0, 2, 5, 10, 20]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _write_png(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def _pose(x: float, *, inside: bool = False):
    return {
        "xyz": [0.5 if inside else x, 0.5 if inside else 0.0, 0.7],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _twist():
    return {
        "linear_xyz": [0.0, 0.0, 0.0],
        "angular_xyz": [0.0, 0.0, 0.0],
    }


def _future_labels(instance_id: str, state: dict):
    return [
        {
            "instance_id": instance_id,
            "horizon_steps": horizon,
            "valid": horizon == 0,
            "pose_world": state["pose_world"] if horizon == 0 else None,
            "twist_world": state["twist_world"] if horizon == 0 else None,
            "invalid_reason": None if horizon == 0 else "fixture_future_masked",
        }
        for horizon in HORIZONS
    ]


def _make_dataset(
    root: Path,
    *,
    success: bool = True,
    failure_reason: str = "target_missed",
    with_camera: bool = False,
    with_distractor: bool = True,
) -> dict[str, Path]:
    run_id = "run-v1-validation"
    episode_id = "ep-v1-validation"
    episode_dir = root / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    sample_count = 30 if success else 4
    camera_contract = (
        {
            "head_rgb": {
                "resolution": [2, 3],
                "fps": 25,
                "role": "policy_observation",
            }
        }
        if with_camera
        else {}
    )
    task_objects = [
        {
            "instance_id": "target",
            "asset_id": "asset-target",
            "class_id": "part",
            "goal_zone_id": "zone-a",
        }
    ]
    if with_distractor:
        task_objects.append(
            {
                "instance_id": "distractor",
                "asset_id": "asset-distractor",
                "class_id": "part",
                "goal_zone_id": None,
            }
        )
    task = {
        "task_id": "sort-validation",
        "task_type": "dynamic_sort",
        "robot_mode": "fixed_base",
        "instruction": "place target in zone a",
        "objects": task_objects,
        "goal_zones": [
            {
                "zone_id": "zone-a",
                "min_xyz": [0.0, 0.0, 0.5],
                "max_xyz": [1.0, 1.0, 1.0],
            }
        ],
        "scored_object_ids": ["target"],
        "seed": 7,
        "belt_speed_mps": 0.1,
        "belt_surface_z_m": 0.5,
        "transport_direction_xyz": [0.0, -1.0, 0.0],
        "exit_plane_point_xyz": [0.7, -0.6, 0.5],
        "max_duration_s": 20.0,
        "metadata": {},
    }
    manifest = {
        "benchmark_config": {
            "protocol_version": "conveyor-bench-v1",
            "physics_hz": 400,
            "control_hz": 50,
            "camera_hz": 25,
            "model_hz": 25,
            "history_offsets_steps": [-2, 0],
            "m0_chunk_size": 16,
            "dynamicvla_chunk_size": 20,
            "label_offset_steps": 5,
            "future_horizons_steps": HORIZONS,
            "evaluation": {
                "settled_linear_speed_mps": 0.02,
                "settled_angular_speed_radps": 0.1,
                "placement_dwell_s": 0.5,
            },
        },
        "episode": {
            "episode_id": episode_id,
            "run_id": run_id,
            "protocol_version": "conveyor-bench-v1",
            "task": task,
            "created_at_utc": "2026-07-30T00:00:00+00:00",
            "env_id": 0,
            "asset_hashes": {},
            "seeds": {"episode": 7},
            "metadata": {"cameras": camera_contract},
        },
    }

    steps = []
    object_rows = []
    for index in range(sample_count):
        sim_step = (index + 1) * 8
        sim_time = (index + 1) / 50.0
        camera_frames = []
        if with_camera and index % 2 == 1:
            frame_index = index // 2
            camera_frames = [
                {
                    "camera_id": "head_rgb",
                    "frame_index": frame_index,
                    "capture_time_s": sim_time,
                    "relative_path": (
                        f"cameras/head_rgb/{frame_index:06d}.png"
                    ),
                }
            ]
        steps.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_time,
                "model_tick": index // 2,
                "env_id": 0,
                "robot_root_world": _pose(0.0),
                "robot_twist_world": _twist(),
                "tcp_base": _pose(0.4),
                "joints": {
                    "names": ["joint-1"],
                    "positions": [0.0],
                    "velocities": [0.0],
                },
                "action": {"values": [0.0] * 10},
                "left_contact_object_ids": ["target"] if success and index < 2 else [],
                "right_contact_object_ids": ["target"] if success and index < 2 else [],
                "camera_frames": camera_frames,
                "phase": "place",
                "selected_object_id": "target",
                "action_chunk_id": None,
                "action_index_in_chunk": None,
                "robot_fallen": False,
                "forbidden_collision": False,
                "belt_measured_speed_mps": 0.1,
                "metadata": {},
            }
        )
        target = {
            "instance_id": "target",
            "pose_world": _pose(
                -0.2,
                inside=success and index >= 2,
            ),
            "twist_world": _twist(),
            "active": True,
            "in_gripper": success and index < 2,
            "crossed_exit": not success and index >= 2,
        }
        distractor = {
            "instance_id": "distractor",
            "pose_world": _pose(-0.4),
            "twist_world": _twist(),
            "active": True,
            "in_gripper": False,
            "crossed_exit": False,
        }
        states = (target, distractor) if with_distractor else (target,)
        for state in states:
            object_rows.append(
                {
                    "sim_step": sim_step,
                    "sim_time_s": sim_time,
                    "model_tick": index // 2,
                    "env_id": 0,
                    "state": state,
                    "future_object_states": _future_labels(
                        state["instance_id"],
                        state,
                    ),
                }
            )

    if success:
        reason = "none"
        status = "success"
        events = [
            {"kind": "episode_start", "time_s": 0.0, "payload": {}},
            {
                "kind": "object_released",
                "time_s": 0.04,
                "sim_step": 16,
                "object_instance_id": "target",
                "goal_zone_id": "zone-a",
                "payload": {},
            },
            {
                "kind": "object_placed",
                "time_s": 0.56,
                "sim_step": 224,
                "object_instance_id": "target",
                "goal_zone_id": "zone-a",
                "payload": {},
            },
            {
                "kind": "episode_end",
                "time_s": 0.60,
                "sim_step": 240,
                "payload": {"success": True, "failure_reason": "none"},
            },
        ]
        outcome = {
            "status": "sorted_correct",
            "goal_zone_id": "zone-a",
            "ever_held": True,
            "released": True,
            "release_time_s": 0.04,
            "dwell_start_s": 0.06,
            "completion_time_s": 0.56,
            "crossed_exit": False,
            "last_zone_ids": ["zone-a"],
            "last_settled": True,
            "last_seen_time_s": 0.60,
        }
        completed = 1
    else:
        reason = failure_reason
        status = "failure"
        events = [
            {"kind": "episode_start", "time_s": 0.0, "payload": {}},
            {
                "kind": "target_missed",
                "time_s": 0.06,
                "sim_step": 24,
                "object_instance_id": "target",
                "payload": {},
            },
            {
                "kind": "episode_end",
                "time_s": 0.06,
                "sim_step": 24,
                "payload": {
                    "success": False,
                    "failure_reason": failure_reason,
                },
            },
        ]
        outcome = {
            "status": "target_missed",
            "goal_zone_id": "zone-a",
            "ever_held": False,
            "released": False,
            "release_time_s": None,
            "dwell_start_s": None,
            "completion_time_s": None,
            "crossed_exit": True,
            "last_zone_ids": [],
            "last_settled": True,
            "last_seen_time_s": 0.06,
        }
        completed = 0
    metrics = {
        "sample_count": sample_count,
        "object_record_count": len(object_rows),
        "duration_s": sample_count / 50.0,
        "completion_time_s": 0.56 if success else None,
        "scored_object_count": 1,
        "completed_object_count": completed,
        "correct_sort_rate": float(completed),
        "wrong_object_id": None,
        "object_outcomes": {"target": outcome},
    }
    episode_summary = {
        "episode_id": episode_id,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "robot_mode": task["robot_mode"],
        "status": status,
        "success": success,
        "failure_reason": reason,
        "sample_count": sample_count,
        "object_record_count": len(object_rows),
        "action_chunk_count": 0,
        "event_count": len(events),
        "completed_at_utc": "2026-07-30T00:01:00+00:00",
        "metrics": metrics,
    }
    report = {
        "episode_id": episode_id,
        "path": str(episode_dir),
        "success": success,
        "failure_reason": reason,
        "metrics": metrics,
        "camera_frames": sample_count // 2 if with_camera else 0,
        "wall_time_s": 1.0,
    }
    run_summary = {
        "run_id": run_id,
        "protocol_version": "conveyor-bench-v1",
        "task_type": task["task_type"],
        "robot_mode": task["robot_mode"],
        "requested_episodes": 1,
        "successful_episodes": int(success),
        "episodes": [report],
    }

    paths = {
        "run_summary": root / f"{run_id}-summary.json",
        "episode_dir": episode_dir,
        "manifest": episode_dir / "manifest.json",
        "summary": episode_dir / "summary.json",
        "steps": episode_dir / "steps.jsonl",
        "objects": episode_dir / "objects.jsonl",
        "events": episode_dir / "events.jsonl",
        "chunks": episode_dir / "action_chunks.jsonl",
        "camera_index": episode_dir / "camera_frames.jsonl",
        "camera": episode_dir / "cameras" / "head_rgb" / "000000.png",
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["summary"], episode_summary)
    _write_jsonl(paths["steps"], steps)
    _write_jsonl(paths["objects"], object_rows)
    _write_jsonl(paths["events"], events)
    _write_jsonl(paths["chunks"], [])
    _write_json(paths["run_summary"], run_summary)
    if with_camera:
        captures = [
            step for step in steps if step["camera_frames"]
        ]
        _write_jsonl(
            paths["camera_index"],
            [
                {
                    "frame_index": frame_index,
                    "sim_step": step["sim_step"],
                    "capture_time_s": step["sim_time_s"],
                    "frames": {
                        "head_rgb": {
                            "relative_path": step["camera_frames"][0][
                                "relative_path"
                            ],
                            "resolution": [2, 3],
                            "role": "policy_observation",
                        }
                    },
                }
                for frame_index, step in enumerate(captures)
            ],
        )
        for frame_index in range(len(captures)):
            _write_png(
                episode_dir
                / "cameras"
                / "head_rgb"
                / f"{frame_index:06d}.png",
                2,
                3,
            )
    return paths


def _run_cli(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        text=True,
        capture_output=True,
        check=False,
    )


def _declare_train_tasking_metadata(paths: dict[str, Path]) -> dict:
    manifest = _read_json(paths["manifest"])
    task = manifest["episode"]["task"]
    asset_ids = list(TRAIN_OBJECT_IDS[: len(task["objects"])])
    for task_object, asset_id in zip(
        task["objects"],
        asset_ids,
        strict=True,
    ):
        task_object["asset_id"] = asset_id
    task["metadata"] = {
        "tasking_schema_version": TASKING_SCHEMA_VERSION,
        "curriculum_split": "train",
        "active_asset_ids": asset_ids,
    }
    _write_json(paths["manifest"], manifest)
    return manifest


def _declare_stationary_contract(
    paths: dict[str, Path],
    *,
    seed: int = 1101,
) -> dict:
    scenario = STATIONARY_SCENARIOS[seed]
    manifest = _read_json(paths["manifest"])
    episode = manifest["episode"]
    task = episode["task"]
    task.update(
        {
            "task_type": "stationary_sort",
            "belt_speed_mps": 0.0,
            "seed": seed,
            "objects": [
                {
                    "instance_id": "target",
                    "asset_id": STATIONARY_TARGET_ASSET_ID,
                    "class_id": "block",
                    "goal_zone_id": STATIONARY_DESTINATION_ZONE_ID,
                }
            ],
            "goal_zones": [
                {
                    "zone_id": STATIONARY_DESTINATION_ZONE_ID,
                    "min_xyz": [0.0, 0.0, 0.5],
                    "max_xyz": [1.0, 1.0, 1.0],
                }
            ],
            "scored_object_ids": ["target"],
            "metadata": {
                "task_family": "single_target",
                "target_asset_id": STATIONARY_TARGET_ASSET_ID,
                "destination_zone_id": STATIONARY_DESTINATION_ZONE_ID,
                "benchmark_role": "stationary_belt_diagnostic",
                "belt_motion": "stationary",
                "active_object_count": 1,
                "tasking_schema_version": TASKING_SCHEMA_VERSION,
                "curriculum_split": "train",
                "active_asset_ids": [STATIONARY_TARGET_ASSET_ID],
                "spawn_x_by_id": {
                    "target": STATIONARY_SPAWN_ORIGIN_XY_M[0]
                    + scenario.object_xy_offset_m[0]
                },
                "spawn_y_by_id": {
                    "target": STATIONARY_SPAWN_ORIGIN_XY_M[1]
                    + scenario.object_xy_offset_m[1]
                },
                "stationary_scenario": {
                    "scenario_id": seed,
                    "scenario_split": scenario.split,
                    "object_xy_offset_m": list(scenario.object_xy_offset_m),
                    "root_xy_offset_m": list(scenario.root_xy_offset_m),
                    "root_yaw_rad": scenario.root_yaw_rad,
                },
            },
        }
    )
    episode["seeds"] = {"episode": seed, "layout": seed}
    _write_json(paths["manifest"], manifest)

    summary = _read_json(paths["summary"])
    summary["task_type"] = "stationary_sort"
    outcome = summary["metrics"]["object_outcomes"]["target"]
    outcome["goal_zone_id"] = STATIONARY_DESTINATION_ZONE_ID
    outcome["last_zone_ids"] = [STATIONARY_DESTINATION_ZONE_ID]
    _write_json(paths["summary"], summary)

    run_summary = _read_json(paths["run_summary"])
    run_summary["task_type"] = "stationary_sort"
    report_outcome = run_summary["episodes"][0]["metrics"]["object_outcomes"][
        "target"
    ]
    report_outcome["goal_zone_id"] = STATIONARY_DESTINATION_ZONE_ID
    report_outcome["last_zone_ids"] = [STATIONARY_DESTINATION_ZONE_ID]
    _write_json(paths["run_summary"], run_summary)

    events = _read_jsonl(paths["events"])
    events.insert(
        1,
        {
            "kind": "object_spawned",
            "time_s": 0.0,
            "object_instance_id": "target",
            "goal_zone_id": None,
            "payload": {
                "asset_id": STATIONARY_TARGET_ASSET_ID,
                "spawn_xyz": [
                    STATIONARY_SPAWN_ORIGIN_XY_M[0]
                    + scenario.object_xy_offset_m[0],
                    STATIONARY_SPAWN_ORIGIN_XY_M[1]
                    + scenario.object_xy_offset_m[1],
                    0.535,
                ],
            },
        },
    )
    for event in events:
        if event.get("goal_zone_id") == "zone-a":
            event["goal_zone_id"] = STATIONARY_DESTINATION_ZONE_ID
    _write_jsonl(paths["events"], events)
    summary = _read_json(paths["summary"])
    summary["event_count"] = len(events)
    _write_json(paths["summary"], summary)

    steps = _read_jsonl(paths["steps"])
    for step in steps:
        step["belt_measured_speed_mps"] = 0.0
    _write_jsonl(paths["steps"], steps)
    return manifest


def test_accepts_valid_success_from_summary_and_output_root(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True)

    summary_result = validate_v1_dataset(paths["run_summary"])
    root_result = validate_v1_dataset(tmp_path)
    completed = _run_cli(tmp_path)

    assert summary_result.ok, summary_result.errors
    assert root_result.ok, root_result.errors
    assert root_result.run_count == 1
    assert root_result.episode_count == 1
    assert root_result.sample_count == 30
    assert root_result.object_record_count == 60
    assert completed.returncode == 0, completed.stderr
    assert "1 run(s), 1 episode(s), 30 step(s)" in completed.stdout


@pytest.mark.parametrize("seed", sorted(STATIONARY_SCENARIOS))
def test_accepts_registered_stationary_scenarios(tmp_path, seed) -> None:
    paths = _make_dataset(tmp_path, success=True, with_distractor=False)
    _declare_stationary_contract(paths, seed=seed)

    result = validate_v1_dataset(tmp_path)

    assert result.ok, result.errors


def test_accepts_profile_owned_stationary_cola(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True, with_distractor=False)
    manifest = _declare_stationary_contract(paths, seed=1101)
    episode = manifest["episode"]
    task = episode["task"]
    task["objects"][0]["asset_id"] = "cola"
    task["metadata"]["target_asset_id"] = "cola"
    task["metadata"]["active_asset_ids"] = ["cola"]
    episode["metadata"]["scene_profile"] = {
        "backend": "isaac_rtx_native_nurec",
        "object_fixture_contract": {
            "all_rigid_bodies_valid": True,
            "all_visuals_composed": True,
            "objects": [{"object_id": "cola"}],
        },
    }
    _write_json(paths["manifest"], manifest)
    events = _read_jsonl(paths["events"])
    next(
        event for event in events if event["kind"] == "object_spawned"
    )["payload"]["asset_id"] = "cola"
    _write_jsonl(paths["events"], events)

    result = validate_v1_dataset(tmp_path)

    assert result.ok, result.errors


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (
            ("task", "metadata", "stationary_scenario", "scenario_split"),
            "test",
            "scenario_split",
        ),
        (
            ("task", "metadata", "stationary_scenario", "scenario_id"),
            1102,
            "scenario_id",
        ),
        (("seeds", "episode"), 1102, "episode and layout seeds"),
        (
            (
                "task",
                "metadata",
                "stationary_scenario",
                "object_xy_offset_m",
            ),
            [0.01, 0.0],
            "object_xy_offset_m",
        ),
        (
            ("task", "metadata", "stationary_scenario", "root_xy_offset_m"),
            [0.01, 0.0],
            "root_xy_offset_m",
        ),
        (
            ("task", "metadata", "stationary_scenario", "root_yaw_rad"),
            0.01,
            "root_yaw_rad",
        ),
        (
            ("task", "metadata", "spawn_x_by_id", "target"),
            0.123,
            "spawn_x_by_id",
        ),
        (("task", "seed"), 9999, "not a registered scenario"),
    ),
)
def test_rejects_stationary_manifest_outside_registered_contract(
    tmp_path,
    path,
    value,
    message,
) -> None:
    paths = _make_dataset(tmp_path, success=True, with_distractor=False)
    _declare_stationary_contract(paths)
    manifest = _read_json(paths["manifest"])
    target = manifest["episode"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _write_json(paths["manifest"], manifest)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(message in error for error in result.errors), result.errors


def test_rejects_stationary_sort_with_nonzero_measured_belt_speed(
    tmp_path,
) -> None:
    paths = _make_dataset(tmp_path, success=True, with_distractor=False)
    _declare_stationary_contract(paths)
    steps = _read_jsonl(paths["steps"])
    steps[0]["belt_measured_speed_mps"] = 0.1
    _write_jsonl(paths["steps"], steps)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(
        "belt_measured_speed_mps does not match" in error
        for error in result.errors
    )


def test_rejects_stationary_spawn_event_outside_registered_scenario(
    tmp_path,
) -> None:
    paths = _make_dataset(tmp_path, success=True, with_distractor=False)
    _declare_stationary_contract(paths, seed=3101)
    events = _read_jsonl(paths["events"])
    spawned = next(event for event in events if event["kind"] == "object_spawned")
    spawned["payload"]["spawn_xyz"][:2] = [0.65, 0.10]
    _write_jsonl(paths["events"], events)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any("object_spawned position" in error for error in result.errors)


@pytest.mark.parametrize(
    ("task_type", "belt_speed", "message"),
    (
        ("stationary_sort", 0.01, "stationary_sort requires"),
        ("dynamic_sort", 0.0, "require positive"),
        ("unknown_sort", 0.1, "registered V1 sorting task"),
    ),
)
def test_rejects_invalid_task_type_and_speed_pair(
    tmp_path, task_type, belt_speed, message
) -> None:
    paths = _make_dataset(tmp_path, success=True)
    manifest = _read_json(paths["manifest"])
    manifest["episode"]["task"]["task_type"] = task_type
    manifest["episode"]["task"]["belt_speed_mps"] = belt_speed
    _write_json(paths["manifest"], manifest)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(message in error for error in result.errors)


def test_rejects_event_time_that_disagrees_with_sim_step(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True)
    events = _read_jsonl(paths["events"])
    events[1]["time_s"] += 0.01
    _write_jsonl(paths["events"], events)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(
        "event time_s does not match its sim_step clock" in error
        for error in result.errors
    )


def test_rejects_cross_split_asset_even_when_active_metadata_matches(
    tmp_path,
) -> None:
    paths = _make_dataset(tmp_path, success=False)
    manifest = _declare_train_tasking_metadata(paths)
    clean = validate_v1_dataset(tmp_path)
    assert clean.ok, clean.errors

    task = manifest["episode"]["task"]
    task["objects"][1]["asset_id"] = UNSEEN_OBJECT_IDS[0]
    task["metadata"]["active_asset_ids"][1] = UNSEEN_OBJECT_IDS[0]
    _write_json(paths["manifest"], manifest)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any("curriculum split" in error for error in result.errors)


def test_rejects_tasking_active_asset_ids_mismatch(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    manifest = _declare_train_tasking_metadata(paths)
    manifest["episode"]["task"]["metadata"]["active_asset_ids"].reverse()
    _write_json(paths["manifest"], manifest)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any("active_asset_ids" in error for error in result.errors)


def test_task_failure_is_valid_but_runtime_error_is_not(tmp_path) -> None:
    task_failure = _make_dataset(tmp_path / "task", success=False)
    runtime_failure = _make_dataset(
        tmp_path / "runtime",
        success=False,
        failure_reason="runtime_error",
    )

    accepted = validate_v1_dataset(task_failure["run_summary"])
    rejected = validate_v1_dataset(runtime_failure["run_summary"])

    assert accepted.ok, accepted.errors
    assert not rejected.ok
    assert any("runtime_error" in error for error in rejected.errors)


def test_rejects_count_mismatch_and_unfinished_episode(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    summary = _read_json(paths["summary"])
    summary["sample_count"] -= 1
    _write_json(paths["summary"], summary)
    (tmp_path / "episodes" / ".crashed.inprogress").mkdir()

    result = validate_v1_dataset(tmp_path)
    completed = _run_cli(tmp_path)

    assert not result.ok
    assert any("sample_count" in error for error in result.errors)
    assert any(".inprogress" in error for error in result.errors)
    assert completed.returncode == 1
    assert "ERROR:" in completed.stderr


def test_rejects_step_order_and_25_over_50_model_tick_damage(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    steps = _read_jsonl(paths["steps"])
    steps[1]["sim_step"] = steps[0]["sim_step"]
    steps[2]["model_tick"] = 7
    _write_jsonl(paths["steps"], steps)

    result = validate_v1_dataset(tmp_path)

    assert any("sim_step must increase strictly" in error for error in result.errors)
    assert any("25/50 cadence" in error for error in result.errors)


def test_rejects_missing_per_step_object_and_future_horizon(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    rows = _read_jsonl(paths["objects"])
    rows.pop(1)
    rows[0]["future_object_states"].pop()
    _write_jsonl(paths["objects"], rows)
    summary = _read_json(paths["summary"])
    summary["object_record_count"] -= 1
    _write_json(paths["summary"], summary)

    result = validate_v1_dataset(tmp_path)

    assert any("one record per object" in error for error in result.errors)
    assert any("future horizons must be exactly" in error for error in result.errors)


@pytest.mark.parametrize("damage", ("pose", "twist"))
def test_future_labels_must_match_realized_model_tick_state(
    tmp_path,
    damage: str,
) -> None:
    paths = _make_dataset(tmp_path, success=True)
    rows = _read_jsonl(paths["objects"])
    source = next(
        row
        for row in rows
        if row["model_tick"] == 0
        and row["state"]["instance_id"] == "target"
    )
    realized = max(
        (
            row
            for row in rows
            if row["model_tick"] == 2
            and row["state"]["instance_id"] == "target"
        ),
        key=lambda row: row["sim_step"],
    )["state"]
    label = next(
        item
        for item in source["future_object_states"]
        if item["horizon_steps"] == 2
    )
    label.update(
        {
            "valid": True,
            "pose_world": deepcopy(realized["pose_world"]),
            "twist_world": deepcopy(realized["twist_world"]),
            "invalid_reason": None,
        }
    )
    _write_jsonl(paths["objects"], rows)
    clean = validate_v1_dataset(tmp_path)
    assert clean.ok, clean.errors

    if damage == "pose":
        label["pose_world"]["xyz"][0] += 0.1
    else:
        label["twist_world"]["linear_xyz"][0] += 0.1
    _write_jsonl(paths["objects"], rows)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(
        f"future {damage}" in error and "model_tick 2" in error
        for error in result.errors
    )


def test_valid_future_label_cannot_target_episode_tail(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True)
    rows = _read_jsonl(paths["objects"])
    tail_source = max(
        (
            row
            for row in rows
            if row["state"]["instance_id"] == "target"
        ),
        key=lambda row: row["sim_step"],
    )
    tail_label = next(
        item
        for item in tail_source["future_object_states"]
        if item["horizon_steps"] == 20
    )
    tail_label.update(
        {
            "valid": True,
            "pose_world": tail_source["state"]["pose_world"],
            "twist_world": tail_source["state"]["twist_world"],
            "invalid_reason": None,
        }
    )
    _write_jsonl(paths["objects"], rows)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(
        "unavailable model_tick" in error for error in result.errors
    )


def test_valid_future_label_cannot_target_inactive_state(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True)
    rows = _read_jsonl(paths["objects"])
    source = next(
        row
        for row in rows
        if row["model_tick"] == 0
        and row["state"]["instance_id"] == "target"
    )
    future_row = max(
        (
            row
            for row in rows
            if row["model_tick"] == 2
            and row["state"]["instance_id"] == "target"
        ),
        key=lambda row: row["sim_step"],
    )
    future_row["state"]["active"] = False
    label = next(
        item
        for item in source["future_object_states"]
        if item["horizon_steps"] == 2
    )
    label.update(
        {
            "valid": True,
            "pose_world": deepcopy(future_row["state"]["pose_world"]),
            "twist_world": deepcopy(future_row["state"]["twist_world"]),
            "invalid_reason": None,
        }
    )
    _write_jsonl(paths["objects"], rows)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(
        "inactive object at model_tick 2" in error
        for error in result.errors
    )


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("missing", "PNG is missing"),
        ("wrong_size", "PNG dimensions"),
        ("corrupt", "decodable PNG header"),
        ("unsafe", "safe PNG path"),
    ),
)
def test_validates_camera_png_path_header_and_dimensions(
    tmp_path,
    damage: str,
    message: str,
) -> None:
    paths = _make_dataset(tmp_path, success=False, with_camera=True)
    clean = validate_v1_dataset(tmp_path)
    assert clean.ok, clean.errors
    assert clean.camera_frame_count == 2

    if damage == "missing":
        paths["camera"].unlink()
    elif damage == "wrong_size":
        _write_png(paths["camera"], 1, 1)
    elif damage == "corrupt":
        paths["camera"].write_bytes(b"not-png")
    else:
        steps = _read_jsonl(paths["steps"])
        steps[1]["camera_frames"][0]["relative_path"] = "../escape.png"
        _write_jsonl(paths["steps"], steps)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(message in error for error in result.errors)


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("index_time", "capture_time_s does not match step"),
        ("reference_time", "disagrees with step reference"),
        ("missing_row", "one-to-one mapping"),
        ("index_without_capture", "step without a capture"),
        ("frame_index", "disagrees with step reference"),
        ("path", "disagrees with step reference"),
        ("resolution", "resolution disagrees with manifest"),
        ("role", "role disagrees with manifest"),
    ),
)
def test_camera_index_must_exactly_mirror_steps_and_manifest(
    tmp_path,
    damage: str,
    message: str,
) -> None:
    paths = _make_dataset(tmp_path, success=False, with_camera=True)
    index = _read_jsonl(paths["camera_index"])
    steps = _read_jsonl(paths["steps"])
    if damage == "index_time":
        for row in index:
            row["capture_time_s"] += 0.001
    elif damage == "reference_time":
        steps[1]["camera_frames"][0]["capture_time_s"] += 0.001
    elif damage == "missing_row":
        index.pop()
    elif damage == "index_without_capture":
        steps[1]["camera_frames"] = []
    elif damage == "frame_index":
        index[0]["frame_index"] = 3
    elif damage == "path":
        index[0]["frames"]["head_rgb"]["relative_path"] = (
            "cameras/head_rgb/other.png"
        )
    elif damage == "resolution":
        index[0]["frames"]["head_rgb"]["resolution"] = [3, 2]
    else:
        index[0]["frames"]["head_rgb"]["role"] = "observer_only"
    _write_jsonl(paths["camera_index"], index)
    _write_jsonl(paths["steps"], steps)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any(message in error for error in result.errors), result.errors


def test_camera_index_rejects_missing_25_hz_tick(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=True, with_camera=True)
    index = _read_jsonl(paths["camera_index"])
    index[1]["sim_step"] = index[2]["sim_step"]
    index[1]["capture_time_s"] = index[2]["capture_time_s"]
    _write_jsonl(paths["camera_index"], index)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any("exactly 16 physics steps" in error for error in result.errors)


def test_camera_index_is_optional_when_steps_have_no_captures(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False, with_camera=False)

    result = validate_v1_dataset(tmp_path)

    assert result.ok, result.errors
    assert not paths["camera_index"].exists()


@pytest.mark.parametrize("damage", ("no_grasp", "wrong_object", "wrong_zone", "event"))
def test_success_requires_correct_object_zone_dwell_and_event(
    tmp_path,
    damage: str,
) -> None:
    paths = _make_dataset(tmp_path, success=True)
    if damage == "event":
        events = _read_jsonl(paths["events"])
        events.pop(2)
        _write_jsonl(paths["events"], events)
        summary = _read_json(paths["summary"])
        summary["event_count"] -= 1
        _write_json(paths["summary"], summary)
    else:
        rows = _read_jsonl(paths["objects"])
        for row in rows:
            state = row["state"]
            if damage == "no_grasp" and state["instance_id"] == "target":
                state["in_gripper"] = False
            elif (
                damage == "wrong_object"
                and state["instance_id"] == "distractor"
                and row["sim_step"] == 8
            ):
                state["in_gripper"] = True
            elif (
                damage == "wrong_zone"
                and state["instance_id"] == "target"
                and row["sim_step"] >= 24
            ):
                state["pose_world"] = _pose(-0.5)
                row["future_object_states"][0]["pose_world"] = state["pose_world"]
        _write_jsonl(paths["objects"], rows)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    evidence_errors = "\n".join(result.errors)
    assert (
        "correct-object" in evidence_errors
        or "wrong-object" in evidence_errors
        or "correct-zone settled dwell" in evidence_errors
        or "object_placed event" in evidence_errors
    )


def test_rejects_unknown_action_chunk_reference(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    steps = _read_jsonl(paths["steps"])
    steps[0]["action_chunk_id"] = "missing-chunk"
    steps[0]["action_index_in_chunk"] = 0
    _write_jsonl(paths["steps"], steps)

    result = validate_v1_dataset(tmp_path)

    assert any("unknown action chunk" in error for error in result.errors)


def test_malformed_nested_values_return_errors_instead_of_crashing(tmp_path) -> None:
    paths = _make_dataset(tmp_path, success=False)
    steps = _read_jsonl(paths["steps"])
    steps[0]["left_contact_object_ids"] = [{}]
    _write_jsonl(paths["steps"], steps)
    events = _read_jsonl(paths["events"])
    events[0]["kind"] = []
    _write_jsonl(paths["events"], events)

    result = validate_v1_dataset(tmp_path)

    assert not result.ok
    assert any("contact_object_ids" in error for error in result.errors)
    assert any("event kind" in error for error in result.errors)
