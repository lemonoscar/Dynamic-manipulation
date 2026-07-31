import json
import math
import struct
import zlib
from dataclasses import asdict
from pathlib import Path

import pytest

from conveyor_bench.v1.config import BenchmarkConfig
from conveyor_bench.v1.exporters import (
    ExportError,
    export_dynamicvla_episode,
    export_m0_episode,
    iter_dynamicvla_records,
    iter_m0_records,
    load_model_tick_steps,
)


def write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def write_png(path: Path, width: int = 2, height: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )


def make_episode(tmp_path, model_ticks: int = 30):
    episode = tmp_path / "ep-export"
    episode.mkdir()
    config = BenchmarkConfig.v1()
    task = {
        "task_id": "task-export",
        "task_type": "dynamic_sort",
        "robot_mode": "whole_body_policy",
        "instruction": "put the can in zone a",
        "objects": [
            {
                "instance_id": "target",
                "asset_id": "asset-can",
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
    }
    write_json(
        episode / "manifest.json",
        {
            "benchmark_config": asdict(config),
            "episode": {
                "episode_id": "ep-export",
                "run_id": "run-export",
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
    root_quaternion = (
        math.sqrt(0.5),
        0.0,
        0.0,
        math.sqrt(0.5),
    )
    rows = []
    object_rows = []
    for sim_step in range(model_ticks * 2):
        tick = sim_step // 2
        control_substep = sim_step % 2
        frames = (
            [
                {
                    "camera_id": camera_id,
                    "frame_index": tick,
                    "capture_time_s": tick / config.model_hz,
                    "relative_path": f"cameras/{camera_id}/{tick:06d}.png",
                }
                for camera_id in ("head_rgb", "wrist_rgb")
            ]
            if control_substep == 0
            else []
        )
        rows.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_step / config.control_hz,
                "model_tick": tick,
                "env_id": 0,
                "robot_root_world": {
                    "xyz": [1.0, 2.0, 0.0],
                    "wxyz": root_quaternion,
                },
                "robot_twist_world": {
                    "linear_xyz": [0.0, 0.0, 0.0],
                    "angular_xyz": [0.0, 0.0, 0.0],
                },
                "tcp_base": {
                    "xyz": [tick * 0.01 + control_substep * 0.001, 0.0, 0.5],
                    "wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "joints": {
                    "names": ["joint-1"],
                    "positions": [0.0],
                    "velocities": [0.0],
                },
                "action": {
                    "values": [
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
                },
                "camera_frames": frames,
                "phase": "track",
                "selected_object_id": "target",
                "left_contact_object_ids": (
                    ["target"] if sim_step < 2 else []
                ),
                "right_contact_object_ids": (
                    ["target"] if sim_step < 2 else []
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
            "pose_world": {
                "xyz": [0.5, 0.0, 0.7],
                "wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "twist_world": {
                "linear_xyz": [0.0, 0.0, 0.0],
                "angular_xyz": [0.0, 0.0, 0.0],
            },
            "active": True,
            "in_gripper": sim_step < 2,
            "crossed_exit": False,
        }
        object_rows.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_step / config.control_hz,
                "model_tick": tick,
                "env_id": 0,
                "state": state,
                "future_object_states": [
                    {
                        "instance_id": "target",
                        "horizon_steps": horizon,
                        "valid": tick + horizon < model_ticks,
                        "pose_world": (
                            state["pose_world"]
                            if tick + horizon < model_ticks
                            else None
                        ),
                        "twist_world": (
                            state["twist_world"]
                            if tick + horizon < model_ticks
                            else None
                        ),
                        "invalid_reason": (
                            None
                            if tick + horizon < model_ticks
                            else "episode_tail"
                        ),
                    }
                    for horizon in config.future_horizons_steps
                ],
            }
        )
    completion_time = 0.54
    events = [
        {"kind": "episode_start", "time_s": 0.0, "payload": {}},
        {
            "kind": "object_released",
            "time_s": 0.04,
            "sim_step": 2,
            "object_instance_id": "target",
            "goal_zone_id": "zone-a",
            "payload": {},
        },
        {
            "kind": "object_placed",
            "time_s": completion_time,
            "sim_step": 27,
            "object_instance_id": "target",
            "goal_zone_id": "zone-a",
            "payload": {},
        },
        {
            "kind": "episode_end",
            "time_s": (model_ticks * 2 - 1) / config.control_hz,
            "sim_step": model_ticks * 2 - 1,
            "payload": {"success": True, "failure_reason": "none"},
        },
    ]
    metrics = {
        "sample_count": len(rows),
        "object_record_count": len(object_rows),
        "duration_s": (len(rows) - 1) / config.control_hz,
        "completion_time_s": completion_time,
        "scored_object_count": 1,
        "completed_object_count": 1,
        "correct_sort_rate": 1.0,
        "wrong_object_id": None,
        "object_outcomes": {
            "target": {
                "status": "sorted_correct",
                "goal_zone_id": "zone-a",
                "completion_time_s": completion_time,
            }
        },
    }
    write_json(
        episode / "summary.json",
        {
            "episode_id": "ep-export",
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "robot_mode": task["robot_mode"],
            "status": "success",
            "success": True,
            "failure_reason": "none",
            "sample_count": len(rows),
            "object_record_count": len(object_rows),
            "action_chunk_count": 0,
            "event_count": len(events),
            "completed_at_utc": "2026-07-30T00:01:00+00:00",
            "metrics": metrics,
        },
    )
    write_jsonl(episode / "steps.jsonl", rows)
    write_jsonl(episode / "objects.jsonl", object_rows)
    write_jsonl(episode / "action_chunks.jsonl", [])
    write_jsonl(episode / "events.jsonl", events)
    for tick in range(model_ticks):
        for camera_id in ("head_rgb", "wrist_rgb"):
            write_png(
                episode / "cameras" / camera_id / f"{tick:06d}.png"
            )
    return episode


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_dynamicvla_projection_uses_model_ticks_history_and_future_offset(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path)
    ticks = load_model_tick_steps(episode)
    assert len(ticks) == 30
    assert ticks[0]["sim_step"] == 1
    assert ticks[0]["_source_control_sim_steps"] == [0, 1]
    assert ticks[0]["camera_frames"][0]["frame_index"] == 0

    records = list(iter_dynamicvla_records(episode))
    first = records[0]
    assert first["source_task_outcome"] == "success"
    assert first["source_failure_reason"] == "none"
    assert first["history_offsets_model_ticks"] == (-2, 0)
    assert first["history_valid_mask"] == (False, True)
    assert records[2]["history_valid_mask"] == (True, True)
    assert first["policy_camera_ids"] == ("head_rgb", "wrist_rgb")
    assert first["observer_cameras_excluded"] is True
    assert [
        frame["camera_id"] for frame in first["history"][1]["camera_frames"]
    ] == ["head_rgb", "wrist_rgb"]
    assert len(first["state6"]) == 6
    assert len(first["delta_action7_chunk"]) == 20
    assert len(first["canonical_action10_chunk"][0]) == 10
    assert first["delta_action7_chunk"][0][0] == pytest.approx(0.05)
    assert first["delta_action7_chunk"][0][6] == pytest.approx(1.0)
    assert first["future_ee_offset_model_ticks"] == 5
    assert all(first["action_valid_mask"])
    assert sum(records[10]["action_valid_mask"]) == 15
    assert sum(records[10]["canonical_valid_mask"]) == 20
    assert first["base_action3_chunk"][0] == (0.1, 0.2, 0.3)


def test_m0_projection_right_pads_and_preserves_canonical_source(tmp_path) -> None:
    episode = make_episode(tmp_path)
    before = (episode / "steps.jsonl").read_bytes()
    first = next(iter_m0_records(episode))
    assert first["source_task_outcome"] == "success"
    assert first["source_failure_reason"] == "none"

    arm = first["world_delta_arm7_chunk"][0]
    assert arm[:3] == pytest.approx((0.0, 0.01, 0.0), abs=1.0e-12)
    assert arm[3:6] == pytest.approx((0.0, 0.02, 0.0), abs=1.0e-12)
    assert first["right_padded_action14_chunk"][0][:7] == (0.0,) * 7
    assert first["right_padded_action14_chunk"][0][7:] == arm
    assert first["canonical_action10_chunk"][0][3] == pytest.approx(0.01)
    assert first["base_action3_chunk"][0] == (0.1, 0.2, 0.3)
    assert first["policy_camera_ids"] == ("head_rgb", "wrist_rgb")
    assert first["observer_cameras_excluded"] is True
    assert [
        frame["camera_id"] for frame in first["policy_camera_frames"]
    ] == ["head_rgb", "wrist_rgb"]

    dynamic_path = episode / "exports" / "dynamicvla.jsonl"
    m0_path = episode / "exports" / "m0.jsonl"
    dynamic_summary = export_dynamicvla_episode(episode, dynamic_path)
    m0_summary = export_m0_episode(episode, m0_path)
    assert dynamic_summary.record_count == 30
    assert m0_summary.record_count == 30
    assert dynamic_summary.source_task_outcome == "success"
    assert dynamic_summary.source_failure_reason == "none"
    assert m0_summary.source_task_outcome == "success"
    assert m0_summary.source_failure_reason == "none"
    assert len(read_jsonl(dynamic_path)) == 30
    assert len(read_jsonl(m0_path)) == 30
    assert (episode / "steps.jsonl").read_bytes() == before

    with pytest.raises(ExportError, match="canonical"):
        export_m0_episode(episode, episode / "steps.jsonl")
    with pytest.raises(FileExistsError):
        export_m0_episode(episode, m0_path)
