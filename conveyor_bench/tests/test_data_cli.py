import hashlib
import json
import struct
import subprocess
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from conveyor_bench.schema.config import BenchmarkConfig
from conveyor_bench.schema.exporters import (
    ExportError,
    validate_episode_for_export,
)
from conveyor_bench.schema.tasking import TASKING_SCHEMA_VERSION, TRAIN_OBJECT_IDS
from conveyor_bench.schema.validation import validate_v1_episode

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def dump_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def dump_png(path: Path, width: int = 2, height: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def pose(x=0.0):
    return {"xyz": [x, 0.0, 0.7], "wxyz": [1.0, 0.0, 0.0, 0.0]}


def twist():
    return {
        "linear_xyz": [0.0, 0.0, 0.0],
        "angular_xyz": [0.0, 0.0, 0.0],
    }


def make_episode(
    path: Path,
    *,
    success: bool = True,
    failure_reason: str = "none",
) -> Path:
    path.mkdir(parents=True)
    config = BenchmarkConfig.v1()
    task = {
        "task_id": "cli-task",
        "task_type": "dynamic_sort",
        "robot_mode": "fixed_base",
        "instruction": "place the can in zone a",
        "objects": [
            {
                "instance_id": "target",
                "asset_id": TRAIN_OBJECT_IDS[0],
                "class_id": "can",
                "goal_zone_id": "zone-a",
            }
        ],
        "goal_zones": [
            {
                "zone_id": "zone-a",
                "min_xyz": [0.0, 0.0, 0.5],
                "max_xyz": [1.0, 1.0, 1.0],
            }
        ],
        "scored_object_ids": ["target"],
        "belt_speed_mps": 0.1,
        "metadata": {
            "tasking_schema_version": TASKING_SCHEMA_VERSION,
            "curriculum_split": "train",
            "active_asset_ids": [TRAIN_OBJECT_IDS[0]],
        },
    }
    dump_json(
        path / "manifest.json",
        {
            "benchmark_config": asdict(config),
            "episode": {
                "episode_id": path.name,
                "run_id": "run-cli",
                "protocol_version": "conveyor-bench-v1",
                "task": task,
                "created_at_utc": "2026-07-30T00:00:00+00:00",
                "env_id": 0,
                "asset_hashes": {},
                "seeds": {"episode": 7},
                "metadata": {
                    "cameras": {
                        "head_rgb": {
                            "resolution": [2, 2],
                            "fps": config.camera_hz,
                            "role": "policy_observation",
                        },
                        "wrist_rgb": {
                            "resolution": [2, 2],
                            "fps": config.camera_hz,
                            "role": "policy_observation",
                        },
                    }
                },
            },
        },
    )
    steps = []
    objects = []
    sample_count = 30
    model_tick_count = sample_count // 2
    physics_steps_per_control = config.physics_hz // config.control_hz
    for control_index in range(sample_count):
        tick = control_index // 2
        sim_step = (control_index + 1) * physics_steps_per_control
        sim_time = (control_index + 1) / config.control_hz
        frames = (
            [
                {
                    "camera_id": camera_id,
                    "frame_index": tick,
                    "capture_time_s": sim_time,
                    "relative_path": (
                        f"cameras/{camera_id}/{tick:06d}.png"
                    ),
                }
                for camera_id in ("head_rgb", "wrist_rgb")
            ]
            if control_index % 2 == 1
            else []
        )
        steps.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_time,
                "model_tick": tick,
                "env_id": 0,
                "robot_root_world": pose(),
                "robot_twist_world": twist(),
                "tcp_base": pose(0.4 + tick * 0.01),
                "joints": {
                    "names": [
                        f"arm_joint{index}" for index in range(1, 9)
                    ],
                    "positions": [0.0] * 8,
                    "velocities": [0.0] * 8,
                },
                "action": {"values": [0.0] * 10},
                "camera_frames": frames,
                "phase": "place",
                "selected_object_id": "target",
                "left_contact_object_ids": (
                    ["target"] if success and control_index < 2 else []
                ),
                "right_contact_object_ids": (
                    ["target"] if success and control_index < 2 else []
                ),
                "action_chunk_id": None,
                "action_index_in_chunk": None,
                "robot_fallen": False,
                "forbidden_collision": False,
                "belt_measured_speed_mps": 0.1,
                "metadata": {},
            }
        )
        state = {
            "instance_id": "target",
            "pose_world": pose(0.5),
            "twist_world": twist(),
            "active": True,
            "in_gripper": success and control_index < 2,
            "crossed_exit": not success and control_index >= 2,
        }
        objects.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_time,
                "model_tick": tick,
                "env_id": 0,
                "state": state,
                "future_object_states": [
                    {
                        "instance_id": "target",
                        "horizon_steps": horizon,
                        "valid": tick + horizon < model_tick_count,
                        "pose_world": (
                            state["pose_world"]
                            if tick + horizon < model_tick_count
                            else None
                        ),
                        "twist_world": (
                            state["twist_world"]
                            if tick + horizon < model_tick_count
                            else None
                        ),
                        "invalid_reason": (
                            None
                            if tick + horizon < model_tick_count
                            else "episode_tail"
                        ),
                    }
                    for horizon in config.future_horizons_steps
                ],
            }
        )
    completion_time = 0.56 if success else None
    if success:
        events = [
            {"kind": "episode_start", "time_s": 0.0, "payload": {}},
            {
                "kind": "object_released",
                "time_s": 0.04,
                "sim_step": round(0.04 * config.physics_hz),
                "object_instance_id": "target",
                "goal_zone_id": "zone-a",
                "payload": {},
            },
            {
                "kind": "object_placed",
                "time_s": completion_time,
                "sim_step": round(completion_time * config.physics_hz),
                "object_instance_id": "target",
                "goal_zone_id": "zone-a",
                "payload": {},
            },
            {
                "kind": "episode_end",
                "time_s": sample_count / config.control_hz,
                "sim_step": sample_count * physics_steps_per_control,
                "payload": {"success": True, "failure_reason": "none"},
            },
        ]
        object_outcome = {
            "status": "sorted_correct",
            "goal_zone_id": "zone-a",
            "completion_time_s": completion_time,
        }
    else:
        events = [
            {"kind": "episode_start", "time_s": 0.0, "payload": {}},
            {
                "kind": "target_missed",
                "time_s": sample_count / config.control_hz,
                "sim_step": sample_count * physics_steps_per_control,
                "object_instance_id": "target",
                "payload": {},
            },
            {
                "kind": "episode_end",
                "time_s": sample_count / config.control_hz,
                "sim_step": sample_count * physics_steps_per_control,
                "payload": {
                    "success": False,
                    "failure_reason": failure_reason,
                },
            },
        ]
        object_outcome = {
            "status": "target_missed",
            "goal_zone_id": "zone-a",
            "completion_time_s": None,
        }
    metrics = {
        "sample_count": len(steps),
        "object_record_count": len(objects),
        "duration_s": sample_count / config.control_hz,
        "completion_time_s": completion_time,
        "scored_object_count": 1,
        "completed_object_count": int(success),
        "correct_sort_rate": float(success),
        "wrong_object_id": None,
        "object_outcomes": {"target": object_outcome},
    }
    dump_json(
        path / "summary.json",
        {
            "episode_id": path.name,
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "robot_mode": task["robot_mode"],
            "status": "success" if success else "failure",
            "success": success,
            "failure_reason": failure_reason,
            "sample_count": len(steps),
            "object_record_count": len(objects),
            "action_chunk_count": 0,
            "event_count": len(events),
            "completed_at_utc": "2026-07-30T00:01:00+00:00",
            "metrics": metrics,
        },
    )
    dump_jsonl(path / "steps.jsonl", steps)
    dump_jsonl(path / "objects.jsonl", objects)
    dump_jsonl(path / "action_chunks.jsonl", [])
    dump_jsonl(path / "events.jsonl", events)
    dump_jsonl(
        path / "camera_frames.jsonl",
        [
            {
                "frame_index": tick,
                "sim_step": (tick + 1) * (
                    config.physics_hz // config.model_hz
                ),
                "capture_time_s": (tick + 1) / config.model_hz,
                "frames": {
                    camera_id: {
                        "relative_path": (
                            f"cameras/{camera_id}/{tick:06d}.png"
                        ),
                        "quality": {
                            "dark_fraction": 0.0,
                            "laplacian_variance": 100.0,
                        },
                        "resolution": [2, 2],
                        "role": "policy_observation",
                    }
                    for camera_id in ("head_rgb", "wrist_rgb")
                },
            }
            for tick in range(model_tick_count)
        ],
    )
    for tick in range(model_tick_count):
        for camera_id in ("head_rgb", "wrist_rgb"):
            dump_png(
                path / "cameras" / camera_id / f"{tick:06d}.png"
            )
    return path


def canonical_digests(episode: Path):
    names = (
        "manifest.json",
        "summary.json",
        "steps.jsonl",
        "objects.jsonl",
        "action_chunks.jsonl",
        "events.jsonl",
        "camera_frames.jsonl",
    )
    return {
        name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
        for name in names
    }


def run_script(name: str, *args: object):
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name), *map(str, args)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_export_cli_handles_output_root_and_requires_force(tmp_path) -> None:
    output_root = tmp_path / "collection"
    episode = make_episode(output_root / "episodes" / "ep-cli")
    before = canonical_digests(episode)

    first = run_script("export.py", output_root, "--profile", "both")
    assert first.returncode == 0, first.stderr
    exports = episode / "exports"
    assert (exports / "dynamicvla.jsonl").is_file()
    assert (exports / "m0.jsonl").is_file()
    export_manifest = json.loads(
        (exports / "export_manifest.json").read_text()
    )
    assert set(export_manifest["profiles"]) == {"dynamicvla", "m0"}
    assert export_manifest["source_task_outcome"] == "success"
    assert export_manifest["source_failure_reason"] == "none"
    assert {
        entry["source_task_outcome"]
        for entry in export_manifest["profiles"].values()
    } == {"success"}
    assert {
        entry["source_failure_reason"]
        for entry in export_manifest["profiles"].values()
    } == {"none"}
    assert export_manifest["canonical_files_modified"] is False
    for profile in ("dynamicvla", "m0"):
        records = [
            json.loads(line)
            for line in (exports / f"{profile}.jsonl").read_text().splitlines()
        ]
        assert records
        assert {
            record["source_task_outcome"] for record in records
        } == {"success"}
        assert {
            record["source_failure_reason"] for record in records
        } == {"none"}
    assert canonical_digests(episode) == before

    refused = run_script("export.py", episode, "--profile", "both")
    assert refused.returncode == 2
    assert "--force" in refused.stderr
    forced = run_script(
        "export.py", episode, "--profile", "both", "--force"
    )
    assert forced.returncode == 0, forced.stderr
    assert canonical_digests(episode) == before


def test_export_cli_writes_causal_m0_mobile_bundle(tmp_path) -> None:
    episode = make_episode(tmp_path / "ep-m0-mobile")
    before = canonical_digests(episode)

    completed = run_script(
        "export.py", episode, "--profile", "m0_mobile"
    )

    assert completed.returncode == 0, completed.stderr
    exports = episode / "exports"
    records = [
        json.loads(line)
        for line in (exports / "m0_mobile.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((exports / "export_manifest.json").read_text())
    assert len(records) == 7
    assert records[0]["observation_sim_step"] == 16
    assert records[0]["label_control_sim_steps"][0] == 24
    assert records[0]["label_control_sim_steps"][-1] == 144
    assert len(records[0]["state28"]) == 28
    assert len(records[0]["model_action10_chunk"]) == 16
    assert records[0]["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]
    assert "phase" not in records[0]
    assert manifest["profiles"]["m0_mobile"]["schema_version"] == (
        "conveyor-bench-m0-mobile-v1"
    )
    assert manifest["canonical_source_hashes"] == before
    assert canonical_digests(episode) == before


def test_export_cli_retains_normal_task_failure_outcome(tmp_path) -> None:
    episode = make_episode(
        tmp_path / "ep-task-failure",
        success=False,
        failure_reason="target_missed",
    )
    before = canonical_digests(episode)

    completed = run_script("export.py", episode, "--profile", "both")
    assert completed.returncode == 0, completed.stderr
    exports = episode / "exports"
    manifest = json.loads((exports / "export_manifest.json").read_text())
    assert manifest["source_task_outcome"] == "failure"
    assert manifest["source_failure_reason"] == "target_missed"
    for profile in ("dynamicvla", "m0"):
        records = [
            json.loads(line)
            for line in (exports / f"{profile}.jsonl").read_text().splitlines()
        ]
        assert records
        assert {
            record["source_task_outcome"] for record in records
        } == {"failure"}
        assert {
            record["source_failure_reason"] for record in records
        } == {"target_missed"}
    assert canonical_digests(episode) == before


def test_export_cli_rejects_runtime_error_without_replacing_force_outputs(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path / "ep-runtime-error")
    first = run_script("export.py", episode, "--profile", "both")
    assert first.returncode == 0, first.stderr
    exports = episode / "exports"
    before_exports = {
        path.name: path.read_bytes()
        for path in exports.iterdir()
        if path.is_file()
    }

    dump_json(
        episode / "summary.json",
        {"success": False, "failure_reason": "runtime_error"},
    )
    refused = run_script(
        "export.py", episode, "--profile", "both", "--force"
    )
    assert refused.returncode == 2
    assert "runtime_error" in refused.stderr
    assert {
        path.name: path.read_bytes()
        for path in exports.iterdir()
        if path.is_file()
    } == before_exports


def test_export_cli_rejects_corruption_and_empty_canonical_stream(
    tmp_path,
) -> None:
    corrupt = make_episode(tmp_path / "ep-corrupt")
    with (corrupt / "objects.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{not-json}\n")
    corrupt_before = canonical_digests(corrupt)

    refused_corrupt = run_script("export.py", corrupt, "--profile", "both")
    assert refused_corrupt.returncode == 2
    assert "data corruption" in refused_corrupt.stderr
    assert canonical_digests(corrupt) == corrupt_before
    corrupt_exports = corrupt / "exports"
    assert not corrupt_exports.exists() or not any(corrupt_exports.iterdir())

    empty = make_episode(tmp_path / "ep-empty")
    (empty / "steps.jsonl").write_text("", encoding="utf-8")
    empty_before = canonical_digests(empty)

    refused_empty = run_script("export.py", empty, "--profile", "both")
    assert refused_empty.returncode == 2
    assert "no valid canonical records" in refused_empty.stderr
    assert canonical_digests(empty) == empty_before
    empty_exports = empty / "exports"
    assert not empty_exports.exists() or not any(empty_exports.iterdir())


def test_audit_cli_reuses_local_camera_quality_without_touching_raw_data(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path / "ep-audit")
    before = canonical_digests(episode)

    completed = run_script("audit_episode.py", episode)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((episode / "quality_report.json").read_text())
    assert report["data_status"] == "clean"
    assert report["frame_stats_source"] == "camera_frames.jsonl"
    assert report["metrics"]["frame_stats_count"] == 30
    assert report["metrics"]["black_frame_count"] == 0
    assert report["metrics"]["blurred_frame_count"] == 0
    assert canonical_digests(episode) == before

    refused = run_script(
        "audit_episode.py",
        episode,
        "--output",
        episode / "steps.jsonl",
    )
    assert refused.returncode == 2
    assert "canonical" in refused.stderr
    assert canonical_digests(episode) == before


def test_export_gate_rejects_realized_future_mismatch(tmp_path) -> None:
    episode = make_episode(tmp_path / "ep-future-mismatch")
    rows = [
        json.loads(line)
        for line in (episode / "objects.jsonl").read_text().splitlines()
    ]
    label = next(
        item
        for item in rows[0]["future_object_states"]
        if item["horizon_steps"] == 2
    )
    label["pose_world"]["xyz"][0] += 0.1
    dump_jsonl(episode / "objects.jsonl", rows)

    validation = validate_v1_episode(episode)

    assert not validation.ok
    assert any("future pose" in error for error in validation.errors)
    with pytest.raises(
        ExportError,
        match=r"strict canonical validation.*future pose.*model_tick 2",
    ):
        validate_episode_for_export(episode)


@pytest.mark.parametrize("camera_id", ("head_rgb", "wrist_rgb"))
def test_export_gate_requires_both_recorded_policy_cameras(
    tmp_path,
    camera_id: str,
) -> None:
    episode = make_episode(tmp_path / f"ep-missing-{camera_id}")
    manifest = json.loads((episode / "manifest.json").read_text())
    del manifest["episode"]["metadata"]["cameras"][camera_id]
    dump_json(episode / "manifest.json", manifest)

    steps = [
        json.loads(line)
        for line in (episode / "steps.jsonl").read_text().splitlines()
    ]
    for step in steps:
        step["camera_frames"] = [
            frame
            for frame in step["camera_frames"]
            if frame["camera_id"] != camera_id
        ]
    dump_jsonl(episode / "steps.jsonl", steps)
    camera_index = [
        json.loads(line)
        for line in (episode / "camera_frames.jsonl").read_text().splitlines()
    ]
    for capture in camera_index:
        del capture["frames"][camera_id]
    dump_jsonl(episode / "camera_frames.jsonl", camera_index)
    clean = validate_v1_episode(episode)
    assert clean.ok, clean.errors

    with pytest.raises(ExportError, match=camera_id):
        validate_episode_for_export(episode)


def test_export_gate_accepts_failure_with_two_policy_views_and_no_overview(
    tmp_path,
) -> None:
    episode = make_episode(
        tmp_path / "ep-valid-task-failure",
        success=False,
        failure_reason="target_missed",
    )

    validation = validate_v1_episode(episode)
    result = validate_episode_for_export(episode)

    assert validation.ok, validation.errors
    assert result.outcome == "failure"
    assert result.failure_reason == "target_missed"


def _load_export_bundle_as_consumer(
    episode: Path,
) -> tuple[dict, dict[str, list[dict]]]:
    """Minimal downstream reader: verify the manifest, then decode JSONL."""

    exports = episode / "exports"
    manifest = json.loads(
        (exports / "export_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != (
        "conveyor-bench-v1-export-manifest-1"
    ):
        raise ValueError("unsupported export manifest schema")
    if manifest.get("canonical_source_hashes") != canonical_digests(episode):
        raise ValueError("canonical source hash mismatch")

    profiles: dict[str, list[dict]] = {}
    for profile, entry in manifest["profiles"].items():
        export_path = exports / entry["relative_path"]
        digest = hashlib.sha256(export_path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{profile} export hash mismatch")
        records = [
            json.loads(line)
            for line in export_path.read_text(encoding="utf-8").splitlines()
        ]
        if len(records) != entry["record_count"]:
            raise ValueError(f"{profile} record count mismatch")
        profiles[profile] = records
    return manifest, profiles


def _prepare_consumer_smoke_episode(episode: Path) -> None:
    """Give the fixture non-zero whole-body actions and an observer view."""

    manifest = json.loads((episode / "manifest.json").read_text())
    manifest["episode"]["task"]["robot_mode"] = "whole_body_policy"
    manifest["episode"]["metadata"]["cameras"]["overview_rgb"] = {
        "resolution": [2, 2],
        "fps": BenchmarkConfig.v1().camera_hz,
        "role": "observer_only",
    }
    dump_json(episode / "manifest.json", manifest)

    summary = json.loads((episode / "summary.json").read_text())
    summary["robot_mode"] = "whole_body_policy"
    dump_json(episode / "summary.json", summary)

    steps = [
        json.loads(line)
        for line in (episode / "steps.jsonl").read_text().splitlines()
    ]
    for step in steps:
        step["action"]["values"] = [
            0.1,
            0.2,
            0.3,
            0.01,
            0.0,
            0.0,
            0.02,
            0.0,
            0.0,
            1.0,
        ]
    for step in steps:
        if step["camera_frames"]:
            tick = step["model_tick"]
            step["camera_frames"].append(
                {
                    "camera_id": "overview_rgb",
                    "frame_index": tick,
                    "capture_time_s": step["sim_time_s"],
                    "relative_path": (
                        f"cameras/overview_rgb/{tick:06d}.png"
                    ),
                }
            )
    dump_jsonl(episode / "steps.jsonl", steps)

    camera_index = [
        json.loads(line)
        for line in (episode / "camera_frames.jsonl").read_text().splitlines()
    ]
    for capture in camera_index:
        tick = capture["frame_index"]
        capture["frames"]["overview_rgb"] = {
            "relative_path": f"cameras/overview_rgb/{tick:06d}.png",
            "quality": {
                "dark_fraction": 0.0,
                "laplacian_variance": 100.0,
            },
            "resolution": [2, 2],
            "role": "observer_only",
        }
    dump_jsonl(episode / "camera_frames.jsonl", camera_index)
    for capture in camera_index:
        dump_png(
            episode
            / "cameras"
            / "overview_rgb"
            / f"{capture['frame_index']:06d}.png"
        )


def test_export_bundle_is_consumable_by_m0_and_dynamicvla_loaders(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path / "ep-consumer")
    _prepare_consumer_smoke_episode(episode)
    before = canonical_digests(episode)

    completed = run_script("export.py", episode, "--profile", "both")
    assert completed.returncode == 0, completed.stderr

    manifest, profiles = _load_export_bundle_as_consumer(episode)
    assert manifest["canonical_source_hashes"] == before
    assert manifest["source"] == {
        "episode_id": "ep-consumer",
        "protocol_version": "conveyor-bench-v1",
        "task_id": "cli-task",
    }
    assert set(profiles) == {"dynamicvla", "m0"}

    dynamic = profiles["dynamicvla"]
    first_dynamic = dynamic[0]
    tail_dynamic = dynamic[-1]
    assert np.asarray(first_dynamic["state6"]).shape == (6,)
    assert np.asarray(first_dynamic["delta_action7_chunk"]).shape == (20, 7)
    assert np.asarray(first_dynamic["base_action3_chunk"]).shape == (20, 3)
    assert np.asarray(first_dynamic["canonical_action10_chunk"]).shape == (
        20,
        10,
    )
    assert np.asarray(first_dynamic["action_valid_mask"]).shape == (20,)
    assert np.asarray(first_dynamic["canonical_valid_mask"]).shape == (20,)
    assert first_dynamic["history_offsets_model_ticks"] == [-2, 0]
    assert first_dynamic["history_valid_mask"] == [False, True]
    assert first_dynamic["history"][0] is None
    assert dynamic[2]["history_valid_mask"] == [True, True]
    assert [
        item["model_tick"] for item in dynamic[2]["history"]
    ] == [0, 2]
    assert first_dynamic["future_ee_offset_model_ticks"] == 5
    assert sum(first_dynamic["action_valid_mask"]) == 10
    assert sum(first_dynamic["canonical_valid_mask"]) == 15
    assert sum(tail_dynamic["action_valid_mask"]) == 0
    assert sum(tail_dynamic["canonical_valid_mask"]) == 1
    np.testing.assert_allclose(
        np.asarray(tail_dynamic["delta_action7_chunk"])[
            ~np.asarray(tail_dynamic["action_valid_mask"], dtype=bool)
        ],
        0.0,
    )
    np.testing.assert_allclose(
        np.asarray(first_dynamic["base_action3_chunk"])[0],
        np.asarray(first_dynamic["canonical_action10_chunk"])[0, :3],
    )

    m0 = profiles["m0"]
    first_m0 = m0[0]
    tail_m0 = m0[-1]
    arm7 = np.asarray(first_m0["world_delta_arm7_chunk"])
    padded14 = np.asarray(first_m0["right_padded_action14_chunk"])
    base3 = np.asarray(first_m0["base_action3_chunk"])
    canonical10 = np.asarray(first_m0["canonical_action10_chunk"])
    assert arm7.shape == (16, 7)
    assert padded14.shape == (16, 14)
    assert base3.shape == (16, 3)
    assert canonical10.shape == (16, 10)
    assert np.asarray(first_m0["action_valid_mask"]).shape == (16,)
    np.testing.assert_allclose(padded14[:, :7], 0.0)
    np.testing.assert_allclose(padded14[:, 7:], arm7)
    np.testing.assert_allclose(base3, canonical10[:, :3])
    assert sum(first_m0["action_valid_mask"]) == 15
    assert sum(tail_m0["action_valid_mask"]) == 1

    for record, frame_key in (
        (first_dynamic["history"][1], "camera_frames"),
        (first_m0, "policy_camera_frames"),
    ):
        assert [frame["camera_id"] for frame in record[frame_key]] == [
            "head_rgb",
            "wrist_rgb",
        ]
        assert all(
            (episode / frame["relative_path"]).is_file()
            for frame in record[frame_key]
        )
    assert first_dynamic["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]
    assert first_m0["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]
    assert "overview_rgb" not in json.dumps(first_dynamic)
    assert "overview_rgb" not in json.dumps(first_m0)
    assert canonical_digests(episode) == before

    m0_path = episode / "exports" / manifest["profiles"]["m0"]["relative_path"]
    with m0_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="m0 export hash mismatch"):
        _load_export_bundle_as_consumer(episode)
