import json
from pathlib import Path

from conveyor_bench.conveyorvla.pct_dataset import (
    audit_pct_episode,
    iter_pct_temporal_records,
)


JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "arm_joint1",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "arm_joint2",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "arm_joint6",
    "arm_joint7",
    "arm_joint8",
]


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, values) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_pct_episode_builds_state_and_future_actions(tmp_path: Path) -> None:
    episode = tmp_path / "liangzhu_0729_n200" / "episode_000000"
    episode.mkdir(parents=True)
    _write_json(
        episode / "task.json",
        {
            "instruction": "Pick up the coke can on box1 and place it on box2.",
            "training_action": {
                "enabled": True,
                "source_gripper_joint_range_m": [0.0, 0.04],
            },
        },
    )
    _write_json(
        episode / "summary.json",
        {
            "success": True,
            "failure_reason": None,
            "execution_provenance_verified": True,
            "training_quality_gate_passed": True,
            "training_visual_source_verified": True,
        },
    )
    _write_json(
        episode / "lerobot_manifest.json",
        {
            "raw_episode_ready": True,
            "camera_keys": ["front", "wrist"],
            "missing_camera_keys": [],
            "camera_state_synchronization": {"verified": True},
            "frequency_report": {"control_hz": 50, "dataset_fps": 5.0},
            "vla_training_action_available": False,
            "vla_training_ineligibility_reason": "lerobot_export_not_training_ready",
        },
    )

    frames = []
    for step in range(1, 72, 10):
        positions = [0.0] * 20
        positions[4] = 0.1
        positions[9] = 0.2
        positions[14:18] = [0.3, 0.4, 0.5, 0.6]
        positions[18:] = [0.04, 0.04]
        x = step * 0.001
        frames.append(
            {
                "action": {
                    "base_velocity": [0.05, 0.0, 0.0],
                    "metadata": {},
                },
                "post_step_observation": {
                    "step_index": step,
                    "robot_root_pose": [x, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                    "robot_root_velocity": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "joint_positions": positions,
                    "joint_velocities": [0.0] * 20,
                    "tcp_pose": [x + 0.3, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
                    "metadata": {"joint_names": JOINT_NAMES},
                },
            }
        )
    _write_jsonl(episode / "frames.jsonl", frames)

    samples = []
    for frame_index, step in enumerate((0, 10, 20, 30)):
        paths = {}
        for key, prefix in (("front", "camera0"), ("wrist", "wrist")):
            relative = Path("images") / key / f"{prefix}_{frame_index:05d}.jpg"
            (episode / relative).parent.mkdir(parents=True, exist_ok=True)
            (episode / relative).write_bytes(b"jpeg")
            paths[key] = {"raw_image_path": relative.as_posix()}
        samples.append(
            {
                "frame_index": frame_index,
                "simulation_step": step,
                "pipeline_state": "exec_nav_to_pick",
                "camera_frames": paths,
            }
        )
    _write_jsonl(episode / "samples.jsonl", samples)

    records = list(iter_pct_temporal_records(episode))
    assert len(records) == 3
    first = records[0]
    assert len(first["state28"]) == 28
    assert len(first["model_action10_chunk"]) == 20
    assert all(len(action) == 10 for action in first["model_action10_chunk"])
    assert first["model_action10_chunk"][0][0:3] == (0.05, 0.0, 0.0)
    assert first["model_action10_chunk"][0][-1] == 1.0
    assert first["source_visual_history_span_s"] == 0.2
    assert first["phase_name"] == "NAV_TO_SOURCE"
    assert first["action_domain_name"] == "NAVIGATION"
    assert first["phase_pure_action_horizon"] is False
    assert first["camera_clips"][0]["frames"][0]["relative_path"].endswith(
        "camera0_00000.jpg"
    )


def test_incomplete_pct_episode_is_ineligible_instead_of_aborting_batch(
    tmp_path: Path,
) -> None:
    episode = tmp_path / "episode_000001"
    episode.mkdir()

    report = audit_pct_episode(episode)

    assert report["eligible"] is False
    assert "cannot read PCT JSON" in report["problems"][0]
