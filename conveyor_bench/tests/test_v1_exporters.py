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
    iter_conveyorvla_al0_temporal_records,
    iter_m0_records,
    load_model_tick_steps,
    validate_episode_for_export,
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


def make_episode(
    tmp_path,
    model_ticks: int = 30,
    partial_tail_camera_ids: tuple[str, ...] | None = None,
):
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
        "belt_speed_mps": 0.1,
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
    sample_count = model_ticks * 2 + int(partial_tail_camera_ids is not None)
    total_model_ticks = model_ticks + int(partial_tail_camera_ids is not None)
    physics_steps_per_control = config.physics_hz // config.control_hz
    camera_index_rows = []
    for control_index in range(sample_count):
        tick = control_index // 2
        control_substep = control_index % 2
        sim_step = (control_index + 1) * physics_steps_per_control
        sim_time = (control_index + 1) / config.control_hz
        camera_ids = (
            partial_tail_camera_ids
            if tick == model_ticks
            else (
                ("head_rgb", "wrist_rgb")
                if control_substep == 1
                else ()
            )
        )
        frames = [
            {
                "camera_id": camera_id,
                "frame_index": tick,
                "capture_time_s": sim_time,
                "relative_path": f"cameras/{camera_id}/{tick:06d}.png",
            }
            for camera_id in camera_ids
        ]
        rows.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_time,
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
                    "names": [f"arm_joint{index}" for index in range(1, 9)],
                    "positions": [0.0] * 8,
                    "velocities": [0.0] * 8,
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
                    ["target"] if control_index < 2 else []
                ),
                "right_contact_object_ids": (
                    ["target"] if control_index < 2 else []
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
            "in_gripper": control_index < 2,
            "crossed_exit": False,
        }
        object_rows.append(
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
                        "valid": tick + horizon < total_model_ticks,
                        "pose_world": (
                            state["pose_world"]
                            if tick + horizon < total_model_ticks
                            else None
                        ),
                        "twist_world": (
                            state["twist_world"]
                            if tick + horizon < total_model_ticks
                            else None
                        ),
                        "invalid_reason": (
                            None
                            if tick + horizon < total_model_ticks
                            else "episode_tail"
                        ),
                    }
                    for horizon in config.future_horizons_steps
                ],
            }
        )
        if camera_ids:
            camera_index_rows.append(
                {
                    "frame_index": tick,
                    "sim_step": sim_step,
                    "capture_time_s": sim_time,
                    "frames": {
                        camera_id: {
                            "relative_path": (
                                f"cameras/{camera_id}/{tick:06d}.png"
                            ),
                            "resolution": [2, 2],
                            "role": "policy_observation",
                            "quality": {
                                "dark_fraction": 0.0,
                                "laplacian_variance": 100.0,
                            },
                        }
                        for camera_id in camera_ids
                    },
                }
            )
    completion_time = 0.56
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
    metrics = {
        "sample_count": len(rows),
        "object_record_count": len(object_rows),
        "duration_s": len(rows) / config.control_hz,
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
    write_jsonl(episode / "camera_frames.jsonl", camera_index_rows)
    for tick in range(model_ticks):
        for camera_id in ("head_rgb", "wrist_rgb"):
            write_png(
                episode / "cameras" / camera_id / f"{tick:06d}.png"
            )
    if partial_tail_camera_ids is not None:
        for camera_id in partial_tail_camera_ids:
            write_png(
                episode / "cameras" / camera_id / f"{model_ticks:06d}.png"
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
    assert ticks[0]["sim_step"] == 16
    assert ticks[0]["_source_control_sim_steps"] == [8, 16]
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


def test_al0_temporal_projection_has_history_and_random_access_targets(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path, model_ticks=30)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    manifest["episode"]["task"]["metadata"] = {"active_object_count": 1}
    write_json(episode / "manifest.json", manifest)

    records = list(iter_conveyorvla_al0_temporal_records(episode))
    first = records[0]

    assert len(records) == 8
    assert first["profile"] == "conveyorvla_al0_temporal_v2"
    assert first["observation_model_tick"] == 2
    assert first["observation_control_tick"] == 5
    assert first["history_model_ticks"] == (0, 2)
    assert first["history_sim_times_s"][1] - first["history_sim_times_s"][0] == pytest.approx(0.08)
    assert [clip["camera_id"] for clip in first["camera_clips"]] == [
        "head_rgb",
        "wrist_rgb",
    ]
    assert all(len(clip["frames"]) == 2 for clip in first["camera_clips"])
    assert len(first["model_action10_chunk"]) == 20
    assert first["model_action10_chunk"][0][:3] == pytest.approx((0.1, 0.2, 0.3))
    assert first["model_action10_chunk"][0][3] == pytest.approx(0.01)
    assert first["model_action10_chunk"][5][3] == pytest.approx(0.06)
    assert first["model_action10_chunk"][0][9] == pytest.approx(0.0)
    assert first["gripper_action_source"] == (
        "future_measured_joint_open_fraction"
    )
    assert first["future_offsets_model_ticks"] == tuple(range(1, 21))
    assert first["object_state_is_model_input"] is False


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


def test_export_gate_rejects_policy_camera_dropout_on_a_complete_tick(
    tmp_path,
) -> None:
    episode = make_episode(tmp_path)
    rows = read_jsonl(episode / "steps.jsonl")
    for row in rows:
        if row["model_tick"] == 2:
            row["camera_frames"] = []
        elif row["model_tick"] > 2:
            for frame in row["camera_frames"]:
                frame["frame_index"] -= 1
    write_jsonl(episode / "steps.jsonl", rows)

    with pytest.raises(
        ExportError,
        match=r"model_tick 2 is complete.*head_rgb.*wrist_rgb",
    ):
        validate_episode_for_export(episode)


def test_iterators_drop_unframed_partial_tail_but_keep_it_for_chunks(
    tmp_path,
) -> None:
    episode = make_episode(
        tmp_path,
        model_ticks=30,
        partial_tail_camera_ids=(),
    )
    ticks = load_model_tick_steps(episode)
    assert len(ticks) == 31
    assert ticks[-1]["_source_control_sim_steps"] == [488]

    dynamic = list(iter_dynamicvla_records(episode))
    m0 = list(iter_m0_records(episode))

    assert [record["model_tick"] for record in dynamic][-1] == 29
    assert [record["model_tick"] for record in m0][-1] == 29
    assert len(dynamic) == len(m0) == 30
    # Tick 30 remains in by_tick: it supplies tick 25's +5 future label and
    # the second canonical/action element of the final emitted chunks.
    assert dynamic[25]["action_valid_mask"][0] is True
    assert dynamic[-1]["canonical_valid_mask"][:2] == (True, True)
    assert m0[-1]["action_valid_mask"][:2] == (True, True)

    dynamic_summary = export_dynamicvla_episode(
        episode,
        episode / "exports" / "dynamicvla-partial-tail.jsonl",
    )
    m0_summary = export_m0_episode(
        episode,
        episode / "exports" / "m0-partial-tail.jsonl",
    )
    assert dynamic_summary.record_count == 30
    assert m0_summary.record_count == 30


def test_iterators_keep_partial_tail_with_complete_policy_pair(tmp_path) -> None:
    episode = make_episode(
        tmp_path,
        model_ticks=4,
        partial_tail_camera_ids=("head_rgb", "wrist_rgb"),
    )

    dynamic = list(iter_dynamicvla_records(episode))
    m0 = list(iter_m0_records(episode))

    assert [record["model_tick"] for record in dynamic] == list(range(5))
    assert [record["model_tick"] for record in m0] == list(range(5))
    assert [
        frame["camera_id"]
        for frame in dynamic[-1]["history"][-1]["camera_frames"]
    ] == ["head_rgb", "wrist_rgb"]
    assert [
        frame["camera_id"] for frame in m0[-1]["policy_camera_frames"]
    ] == ["head_rgb", "wrist_rgb"]


@pytest.mark.parametrize(
    "iterator",
    (iter_dynamicvla_records, iter_m0_records),
)
def test_iterators_reject_half_observed_partial_tail(tmp_path, iterator) -> None:
    episode = make_episode(
        tmp_path,
        model_ticks=4,
        partial_tail_camera_ids=("head_rgb",),
    )

    with pytest.raises(
        ExportError,
        match=r"final partial model_tick 4.*head_rgb.*wrist_rgb",
    ):
        list(iterator(episode))
