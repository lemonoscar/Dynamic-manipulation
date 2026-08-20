import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from conveyor_bench.conveyorvla.waypoint import ACTION_HORIZON, DATASET_SCHEMA_VERSION
from conveyor_bench.conveyorvla.waypoint_data import (
    FORBIDDEN_MODEL_KEYS,
    MODEL_BATCH_KEYS,
    NORMALIZATION_SCHEMA_VERSION,
    ConveyorVLAWaypointDataset,
    iter_waypoint_records,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_episode(tmp_path: Path) -> Path:
    root = tmp_path / "liangzhu_0815_n200" / "episode_000001"
    (root / "images" / "front").mkdir(parents=True)
    (root / "images" / "wrist").mkdir(parents=True)
    _write_json(root / "task.json", {"instruction": "Pick up the Coke can and place it in the other box."})
    phases = (
        ["exec_nav_to_pick"] * 7
        + ["exec_pick"] * 6
        + ["exec_nav_to_place"] * 6
        + ["exec_place"] * 12
    )
    rows = []
    for index, phase in enumerate(phases):
        front = root / "images" / "front" / f"{index:04d}.png"
        wrist = root / "images" / "wrist" / f"{index:04d}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(front)
        Image.new("RGB", (4, 4), (0, index, 0)).save(wrist)
        done_suffix = index >= len(phases) - 6
        rows.append(
            {
                "frame_index": index,
                "timestamp": index * 0.2,
                "simulation_step": index * 10,
                "pipeline_state": phase,
                "base_pose": [index * 0.01, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                "tcp_pose": [0.3 + index * 0.001, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                "gripper_position": 0.04 if done_suffix else 0.02,
                "base_velocity": [0.0, 0.0, 0.0],
                "object_state": [0.0] * 13,
                "camera_frames": {
                    "front": {"raw_image_path": front.relative_to(root).as_posix()},
                    "wrist": {"raw_image_path": wrist.relative_to(root).as_posix()},
                },
                "subtask_signals": {
                    "gripper_command": "hold" if done_suffix else "",
                    "segment_name": "return_home_after_place" if done_suffix else "",
                    "segment_type": "motion" if done_suffix else "",
                },
            }
        )
    with (root / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return root


def test_raw_adapter_builds_typed_state_free_waypoints_and_done(tmp_path: Path) -> None:
    records = list(
        iter_waypoint_records(
            _source_episode(tmp_path),
            split="train",
            require_source_audit=False,
        )
    )
    routes = {record["route"] for record in records}
    assert routes == {"NAV_TO_SOURCE", "PICK", "NAV_TO_TARGET", "PLACE", "DONE"}
    assert not any(FORBIDDEN_MODEL_KEYS.intersection(record) for record in records)

    nav = next(record for record in records if record["route"] == "NAV_TO_SOURCE")
    assert len(nav["nav_waypoints_body"]) == ACTION_HORIZON
    assert nav["nav_waypoints_body"][0] == pytest.approx([0.03, 0.0, 0.0])
    assert nav["arm_targets_base"] is None
    assert nav["action_valid_mask"][0] is True
    assert nav["roundtrip_error"]["navigation_max_m_or_rad"] < 1.0e-12

    arm = next(record for record in records if record["route"] == "PICK")
    assert len(arm["arm_targets_base"]) == ACTION_HORIZON
    assert arm["nav_waypoints_body"] is None
    assert arm["roundtrip_error"]["arm_max_m_or_rad"] < 1.0e-12

    done = [record for record in records if record["route"] == "DONE"]
    assert len(done) == 6
    assert all(record["action_domain"] == "NONE" for record in done)
    assert all(record["nav_waypoints_body"] is None for record in done)
    assert all(not any(record["action_valid_mask"]) for record in done)


def test_loader_returns_exact_model_schema_without_proprioception(tmp_path: Path) -> None:
    source = _source_episode(tmp_path)
    record = next(
        record
        for record in iter_waypoint_records(
            source, split="train", require_source_audit=False
        )
        if record["route"] == "PICK"
    )
    root = tmp_path / "derived"
    root.mkdir()
    records_path = root / "train.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (root / f"{split}.jsonl").write_text("", encoding="utf-8")
    q01_arm = [[-1.0] * 6 for _ in range(ACTION_HORIZON)]
    q99_arm = [[1.0] * 6 for _ in range(ACTION_HORIZON)]
    normalization = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "navigation": {
            "q01": [[-1.0] * 3 for _ in range(ACTION_HORIZON)],
            "q99": [[1.0] * 3 for _ in range(ACTION_HORIZON)],
        },
        "manipulation": {"q01": q01_arm, "q99": q99_arm},
    }
    normalization_path = root / "normalization.json"
    _write_json(normalization_path, normalization)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "records": {
            split: {
                "relative_path": f"{split}.jsonl",
                "sha256": _sha256(root / f"{split}.jsonl"),
            }
            for split in ("train", "val", "test")
        },
        "normalization_relative_path": "normalization.json",
        "normalization_sha256": _sha256(normalization_path),
    }
    _write_json(root / "manifest.json", manifest)

    dataset = ConveyorVLAWaypointDataset(root, split="train")
    example = dataset[0]
    assert set(example) == MODEL_BATCH_KEYS
    assert not FORBIDDEN_MODEL_KEYS.intersection(example)
    assert example["route"] == "PICK"
    assert len(example["video"]) == 2
    assert all(len(clip) == 2 for clip in example["video"])
    assert len(example["action"]) == ACTION_HORIZON
