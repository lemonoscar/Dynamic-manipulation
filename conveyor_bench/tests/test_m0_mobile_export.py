import json
import math
from pathlib import Path

import pytest

from conveyor_bench.v1.exporters import (
    ExportError,
    canonical_to_m0_mobile_action,
    iter_m0_mobile_records,
    m0_mobile_to_canonical_action,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode(tmp_path: Path, *, missing_joint: bool = False) -> Path:
    episode = tmp_path / "episode"
    episode.mkdir()
    _write_json(
        episode / "manifest.json",
        {
            "episode": {
                "episode_id": "m0-mobile-test",
                "run_id": "run-test",
                "protocol_version": "conveyor-bench-v1",
                "task": {
                    "task_id": "task-test",
                    "instruction": "pick the moving part",
                },
                "metadata": {
                    "cameras": {
                        "head_rgb": {"role": "policy_observation"},
                        "overview_rgb": {"role": "observer_only"},
                        "wrist_rgb": {"role": "policy_observation"},
                    }
                },
            }
        },
    )
    _write_json(
        episode / "summary.json",
        {"success": True, "failure_reason": "none"},
    )
    joint_values = {
        f"arm_joint{index}": (index / 10.0, float(index))
        for index in range(1, 7)
    }
    joint_values.update(
        {"arm_joint7": (0.022, 0.0), "arm_joint8": (0.022, 0.0)}
    )
    if missing_joint:
        del joint_values["arm_joint6"]
    names = (
        "arm_joint6",
        "other_joint",
        "arm_joint1",
        "arm_joint8",
        "arm_joint3",
        "arm_joint7",
        "arm_joint5",
        "arm_joint2",
        "arm_joint4",
    )
    names = tuple(
        name for name in names if name in joint_values or name == "other_joint"
    )
    joint_values["other_joint"] = (99.0, 99.0)
    root_quaternion = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    rows = []
    for index in range(20):
        camera_frames = []
        if index % 2 == 1:
            camera_frames = [
                {
                    "camera_id": camera_id,
                    "frame_index": index // 2,
                    "capture_time_s": (index + 1) / 50.0,
                    "relative_path": f"cameras/{camera_id}/{index // 2:06d}.png",
                }
                for camera_id in ("overview_rgb", "wrist_rgb", "head_rgb")
            ]
        rows.append(
            {
                "sim_step": (index + 1) * 8,
                "sim_time_s": (index + 1) / 50.0,
                "model_tick": index // 2,
                "robot_root_world": {
                    "xyz": [0.0, 0.0, 0.3],
                    "wxyz": root_quaternion,
                },
                "robot_twist_world": {
                    "linear_xyz": [1.0, 0.0, 0.0],
                    "angular_xyz": [0.0, 1.0, 0.0],
                },
                "tcp_base": {
                    "xyz": [0.3, 0.0, 0.4],
                    "wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "joints": {
                    "names": names,
                    "positions": [joint_values[name][0] for name in names],
                    "velocities": [joint_values[name][1] for name in names],
                },
                "action": {
                    "values": [
                        index / 100.0,
                        0.0,
                        0.01,
                        0.001,
                        0.002,
                        0.003,
                        0.004,
                        0.005,
                        0.006,
                        1.0 if index % 2 else -1.0,
                    ]
                },
                "camera_frames": camera_frames,
                "phase": "privileged-phase",
                "selected_object_id": "privileged-target",
                "future_object_states": [{"privileged": True}],
            }
        )
    _write_jsonl(episode / "steps.jsonl", rows)
    return episode


def test_m0_mobile_records_are_causal_and_exclude_privileged_fields(tmp_path) -> None:
    records = list(iter_m0_mobile_records(_episode(tmp_path)))

    assert len(records) == 2
    first = records[0]
    assert first["observation_sim_step"] == 16
    assert first["label_control_sim_steps"] == tuple(range(24, 145, 8))
    assert first["canonical_action10_chunk"][0][0] == pytest.approx(0.02)
    assert first["canonical_action10_chunk"][-1][0] == pytest.approx(0.17)
    assert first["model_action10_chunk"][0][9] == 0.0
    assert first["action_horizon"] == 16
    assert first["action_rate_hz"] == 50
    assert [
        frame["camera_id"] for frame in first["policy_camera_frames"]
    ] == ["head_rgb", "wrist_rgb"]
    encoded = json.dumps(first)
    for forbidden in (
        "overview_rgb",
        "phase",
        "selected_object_id",
        "future_object_states",
    ):
        assert forbidden not in encoded


def test_m0_mobile_state28_uses_named_joints_and_measured_gripper(tmp_path) -> None:
    state = next(iter_m0_mobile_records(_episode(tmp_path)))["state28"]

    assert len(state) == 28
    assert state[0:3] == pytest.approx((0.0, -1.0, 0.0), abs=1.0e-9)
    assert state[3:6] == pytest.approx((1.0, 0.0, 0.0), abs=1.0e-9)
    assert state[6:9] == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-9)
    assert state[9:15] == pytest.approx(tuple(index / 10 for index in range(1, 7)))
    assert state[15:21] == pytest.approx(tuple(float(index) for index in range(1, 7)))
    assert state[21:27] == pytest.approx((0.3, 0.0, 0.4, 0.0, 0.0, 0.0))
    assert state[27] == pytest.approx(0.5)


def test_m0_mobile_action_gripper_round_trip() -> None:
    canonical = (0.1, 0.0, -0.2, 0.01, 0.02, 0.03, 0.0, 0.0, 0.0, -0.4)
    model = canonical_to_m0_mobile_action(canonical)
    assert model[9] == pytest.approx(0.3)
    assert m0_mobile_to_canonical_action(model) == pytest.approx(canonical)


def test_m0_mobile_rejects_missing_arm_joint(tmp_path) -> None:
    with pytest.raises(ExportError, match="arm_joint6"):
        next(iter_m0_mobile_records(_episode(tmp_path, missing_joint=True)))
