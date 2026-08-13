import json
import math
from pathlib import Path

import pytest

from conveyor_bench.schema.exporters import (
    ExportError,
    canonical_to_m0_mobile_action,
    iter_m0_mobile_records,
    m0_mobile_to_canonical_action,
)
from conveyor_bench.schema.stationary import STATIONARY_SCENARIOS


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
                    "task_type": "dynamic_sort",
                    "instruction": "pick the moving part",
                    "robot_mode": "whole_body_policy",
                    "belt_speed_mps": 0.08,
                    "metadata": {"curriculum_split": "train"},
                },
                "metadata": {
                    "cameras": {
                        "head_rgb": {"role": "policy_observation"},
                        "overview_rgb": {"role": "observer_only"},
                        "wrist_rgb": {"role": "policy_observation"},
                    },
                    "m0_mobile_approach_assist": {"enabled": False},
                    "m0_pregrasp_staging_assist": {"enabled": False},
                    "m0_carry_retract_teacher_executor": {"enabled": False},
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


def _declare_stationary(episode: Path, seed: int) -> dict:
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    task = manifest["episode"]["task"]
    scenario = STATIONARY_SCENARIOS[seed]
    task.update(
        {
            "task_type": "stationary_sort",
            "belt_speed_mps": 0.0,
            "seed": seed,
            "objects": [
                {
                    "instance_id": "part_red_block",
                    "asset_id": "part_red_block",
                    "class_id": "block",
                    "goal_zone_id": "sort_bin_blue",
                }
            ],
            "goal_zones": [
                {
                    "zone_id": "sort_bin_blue",
                    "min_xyz": [0.0, 0.0, 0.0],
                    "max_xyz": [1.0, 1.0, 1.0],
                }
            ],
            "scored_object_ids": ["part_red_block"],
        }
    )
    task["metadata"].update(
        {
            "task_family": "single_target",
            "target_asset_id": "part_red_block",
            "destination_zone_id": "sort_bin_blue",
            "benchmark_role": "stationary_belt_diagnostic",
            "belt_motion": "stationary",
            "active_object_count": 1,
            "spawn_x_by_id": {
                "part_red_block": 0.65 + scenario.object_xy_offset_m[0]
            },
            "spawn_y_by_id": {
                "part_red_block": 0.10 + scenario.object_xy_offset_m[1]
            },
            "stationary_scenario": {
                "scenario_id": seed,
                "scenario_split": scenario.split,
                "object_xy_offset_m": list(scenario.object_xy_offset_m),
                "root_xy_offset_m": list(scenario.root_xy_offset_m),
                "root_yaw_rad": scenario.root_yaw_rad,
            },
        }
    )
    manifest["episode"]["seeds"] = {"episode": seed, "layout": seed}
    _write_json(episode / "manifest.json", manifest)
    return manifest


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
    assert first["split"] == "train"
    assert first["object_curriculum_split"] == "train"
    assert first["robot_mode"] == "whole_body_policy"
    assert first["source_task_type"] == "dynamic_sort"
    assert first["source_assisted"] is False
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


@pytest.mark.parametrize(
    ("assist_key", "flag"),
    (
        (key, flag)
        for key in (
            "m0_mobile_approach_assist",
            "m0_pregrasp_staging_assist",
            "m0_carry_retract_teacher_executor",
        )
        for flag in ("enabled", "assisted")
    ),
)
def test_m0_mobile_export_rejects_declared_diagnostic_assist(
    tmp_path: Path, assist_key: str, flag: str
) -> None:
    episode = _episode(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    manifest["episode"]["metadata"][assist_key][flag] = True
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="diagnostic-assisted"):
        next(iter_m0_mobile_records(episode))


@pytest.mark.parametrize(
    "control_layer",
    (
        "diagnostic_mobile_approach_assist",
        "diagnostic_pregrasp_staging_assist",
        "diagnostic_teacher_via_m0_executor",
    ),
)
def test_m0_mobile_export_rejects_recorded_diagnostic_intervention(
    tmp_path: Path, control_layer: str
) -> None:
    episode = _episode(tmp_path)
    steps = [
        json.loads(line)
        for line in (episode / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    steps[0]["metadata"] = {
        "m0_online_action": {"control_layer": control_layer}
    }
    _write_jsonl(episode / "steps.jsonl", steps)

    with pytest.raises(ExportError, match="recorded control intervention"):
        next(iter_m0_mobile_records(episode))


def test_m0_mobile_export_requires_explicit_curriculum_split(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    del manifest["episode"]["task"]["metadata"]["curriculum_split"]
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="explicitly"):
        next(iter_m0_mobile_records(episode))


def test_m0_mobile_export_rejects_non_mapping_task_metadata(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    manifest = json.loads((episode / "manifest.json").read_text(encoding="utf-8"))
    manifest["episode"]["task"]["metadata"] = None
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="task.metadata must be a JSON object"):
        next(iter_m0_mobile_records(episode))


def test_stationary_export_requires_train_object_curriculum(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    manifest = _declare_stationary(episode, 2101)
    manifest["episode"]["task"]["metadata"]["curriculum_split"] = "val"
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="object curriculum_split must be train"):
        next(iter_m0_mobile_records(episode))


@pytest.mark.parametrize("scenario_split", ("val", "test"))
def test_stationary_scenario_split_cannot_leak_into_training(
    tmp_path, scenario_split
) -> None:
    episode = _episode(tmp_path)
    _declare_stationary(
        episode, 2101 if scenario_split == "val" else 3101
    )

    first = next(iter_m0_mobile_records(episode))

    assert first["split"] == scenario_split
    assert first["object_curriculum_split"] == "train"


def test_stationary_export_rejects_unregistered_task_contract(tmp_path) -> None:
    episode = _episode(tmp_path)
    manifest = _declare_stationary(episode, 1101)
    task = manifest["episode"]["task"]
    task["metadata"]["target_asset_id"] = "part_blue_cylinder"
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="registered diagnostic contract"):
        next(iter_m0_mobile_records(episode))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("scenario_split", "test"),
        ("scenario_id", 3101),
        ("object_xy_offset_m", [0.1, 0.0]),
        ("root_xy_offset_m", [0.1, 0.0]),
        ("root_yaw_rad", 0.1),
    ),
)
def test_stationary_export_rejects_spoofed_scenario_claims(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    episode = _episode(tmp_path)
    manifest = _declare_stationary(episode, 1101)
    manifest["episode"]["task"]["metadata"]["stationary_scenario"][field] = value
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="registered diagnostic contract"):
        next(iter_m0_mobile_records(episode))


def test_stationary_export_rejects_episode_seed_spoof(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    manifest = _declare_stationary(episode, 3101)
    manifest["episode"]["seeds"]["episode"] = 1101
    _write_json(episode / "manifest.json", manifest)

    with pytest.raises(ExportError, match="registered diagnostic contract"):
        next(iter_m0_mobile_records(episode))


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
