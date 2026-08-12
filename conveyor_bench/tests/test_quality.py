import json
from dataclasses import asdict

import pytest

from conveyor_bench.schema.config import BenchmarkConfig
from conveyor_bench.schema.quality import (
    DataStatus,
    FrameStats,
    TaskOutcome,
    audit_episode,
)


def dump_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def dump_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def pose(x=0.0):
    return {"xyz": [x, 0.0, 0.7], "wxyz": [1.0, 0.0, 0.0, 0.0]}


def twist():
    return {
        "linear_xyz": [0.0, 0.0, 0.0],
        "angular_xyz": [0.0, 0.0, 0.0],
    }


def make_episode(
    tmp_path, *, success: bool = False, step_count: int = 4
):
    episode = tmp_path / "episode"
    episode.mkdir()
    config = BenchmarkConfig.v1()
    task = {
        "task_id": "quality-task",
        "task_type": "dynamic_sort",
        "robot_mode": "fixed_base",
        "instruction": "place the can in zone a",
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
    dump_json(
        episode / "manifest.json",
        {
            "benchmark_config": asdict(config),
            "episode": {
                "episode_id": "ep-quality",
                "protocol_version": "conveyor-bench-v1",
                "task": task,
            },
        },
    )
    dump_json(
        episode / "summary.json",
        {
            "episode_id": "ep-quality",
            "success": success,
            "failure_reason": "none" if success else "target_missed",
        },
    )

    steps = []
    object_rows = []
    for sim_step in range(step_count):
        model_tick = sim_step // 2
        frame = {
            "camera_id": "head",
            "frame_index": model_tick,
            "capture_time_s": model_tick / config.camera_hz,
            "relative_path": f"rgb/head/{model_tick:06d}.png",
        }
        steps.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_step / config.control_hz,
                "model_tick": model_tick,
                "env_id": 0,
                "robot_root_world": pose(0.0),
                "robot_twist_world": twist(),
                "tcp_base": pose(0.4),
                "joints": {
                    "names": ["joint-1"],
                    "positions": [0.0],
                    "velocities": [0.0],
                },
                "action": {"values": [0.0] * 10},
                "left_contact_object_ids": [],
                "right_contact_object_ids": [],
                "camera_frames": [frame],
                "phase": "place",
                "selected_object_id": "target",
                "action_chunk_id": None,
                "action_index_in_chunk": None,
                "robot_fallen": False,
                "forbidden_collision": False,
                "belt_measured_speed_mps": 0.1,
            }
        )
        state = {
            "instance_id": "target",
            "pose_world": pose(0.5),
            "twist_world": twist(),
            "active": True,
            "in_gripper": False,
            "crossed_exit": not success,
        }
        object_rows.append(
            {
                "sim_step": sim_step,
                "sim_time_s": sim_step / config.control_hz,
                "model_tick": model_tick,
                "env_id": 0,
                "state": state,
                "future_object_states": [
                    {
                        "instance_id": "target",
                        "horizon_steps": horizon,
                        "valid": True,
                        "pose_world": state["pose_world"],
                        "twist_world": state["twist_world"],
                        "invalid_reason": None,
                    }
                    for horizon in config.future_horizons_steps
                ],
            }
        )
    dump_jsonl(episode / "steps.jsonl", steps)
    dump_jsonl(episode / "objects.jsonl", object_rows)
    dump_jsonl(episode / "action_chunks.jsonl", [])
    dump_jsonl(
        episode / "events.jsonl",
        [
            {"kind": "episode_start", "time_s": 0.0},
            {
                "kind": "episode_end",
                "time_s": (step_count - 1) / config.control_hz,
            },
        ],
    )
    return episode, steps


def issue_codes(report):
    return {issue.code for issue in report.issues}


def test_task_failure_is_not_data_corruption(tmp_path) -> None:
    episode, _ = make_episode(tmp_path, success=False)
    report = audit_episode(episode)

    assert report.task_outcome is TaskOutcome.FAILURE
    assert report.task_failure_reason == "target_missed"
    assert report.data_status is DataStatus.CLEAN
    assert not report.data_corrupted
    assert report.training_eligible
    assert report.metrics["task_failure_is_data_corruption"] is False
    assert report.metrics["black_frame_fraction"] is None
    assert report.metrics["frame_stats_available"] is False


def test_partial_final_model_tick_is_not_cadence_corruption(tmp_path) -> None:
    episode, _ = make_episode(tmp_path, success=False, step_count=5)

    report = audit_episode(episode)

    assert "model_cadence" not in issue_codes(report)
    assert report.metrics["model_tick_count"] == 3
    assert not report.data_corrupted


def test_injected_frame_stats_and_behavior_checks_are_quality_warnings(
    tmp_path,
) -> None:
    episode, steps = make_episode(tmp_path, success=True)
    steps[1]["action"]["values"][3] = 0.8
    steps[2]["selected_object_id"] = None
    dump_jsonl(episode / "steps.jsonl", steps)
    actions = [{"values": [0.0] * 10} for _ in range(16)]
    dump_jsonl(
        episode / "action_chunks.jsonl",
        [
            {
                "chunk_id": "stale-1",
                "profile": "m0",
                "source_observation_tick": 0,
                "source_observation_time_s": 0.0,
                "valid_from_tick": 0,
                "valid_until_tick": 16,
                "execute_from_tick": 2,
                "execute_until_tick": 16,
                "actions": actions,
                "stale": True,
                "discarded_action_count": 2,
                "discard_reason": "inference_latency",
            }
        ],
    )

    report = audit_episode(
        episode,
        frame_stats_provider=lambda _path, _frame: FrameStats(
            black_fraction=0.99,
            blur_score=1.0,
            object_visibility={"target": 0.0},
        ),
    )

    assert report.task_outcome is TaskOutcome.SUCCESS
    assert report.data_status is DataStatus.WARNING
    assert not report.data_corrupted
    assert {
        "action_jump",
        "language_phase_alignment",
        "camera_black_frame",
        "camera_blur",
        "object_visibility",
        "stale_action_chunk",
    } <= issue_codes(report)
    assert report.metrics["camera_frame_ref_count"] == 4
    assert report.metrics["frame_stats_count"] == 2
    assert report.metrics["black_frame_fraction"] == pytest.approx(1.0)
    assert report.metrics["stale_action_chunk_count"] == 1
    assert report.metrics["discarded_action_count"] == 2


def test_schema_numeric_and_cadence_damage_is_corruption(tmp_path) -> None:
    episode, steps = make_episode(tmp_path, success=False)
    steps[1]["sim_time_s"] = 0.5
    steps[1]["action"]["values"][0] = float("nan")
    steps[1]["action_chunk_id"] = "missing-chunk"
    steps[1]["action_index_in_chunk"] = 0
    dump_jsonl(episode / "steps.jsonl", steps)

    report = audit_episode(episode)

    assert report.task_outcome is TaskOutcome.FAILURE
    assert report.data_status is DataStatus.CORRUPT
    assert report.data_corrupted
    assert not report.training_eligible
    assert {
        "non_finite_numeric",
        "action_schema",
        "control_cadence",
        "chunk_reference",
    } <= issue_codes(report)
