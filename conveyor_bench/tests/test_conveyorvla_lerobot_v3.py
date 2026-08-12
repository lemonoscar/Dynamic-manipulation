from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conveyor_bench.conveyorvla.lerobot_v3 import (
    VIDEO_FEATURE_KEYS,
    iter_query_records,
    lerobot_features,
    lerobot_frame_from_record,
    lerobot_model_example,
    load_lerobot_v3_config,
    write_lerobot_episodes,
)
from conveyor_bench.conveyorvla.temporal import load_temporal_config
from conveyor_bench.conveyorvla.config import M0MobileError, M0MobileNormalizer


def _record(config: dict, tick: int) -> dict:
    clips = []
    for camera in ("head_rgb", "wrist_rgb"):
        clips.append(
            {
                "camera_id": camera,
                "history_offsets_model_ticks": [-2, 0],
                "frames": [
                    {
                        "camera_id": camera,
                        "relative_path": f"cameras/{camera}/{name}.png",
                    }
                    for name in ("old", "current")
                ],
            }
        )
    return {
        "schema_version": "conveyor-vla-al0-temporal-v3",
        "profile": "conveyorvla_al0_temporal_v3",
        "policy_task_scope": "navigate_grasp_deliver",
        "gripper_action_source": "future_measured_joint_open_fraction",
        "source_task_outcome": "success",
        "source_assisted": False,
        "source_episode_id": "episode-a",
        "sample_id": f"episode-a:model-tick-{tick}",
        "instruction": "Grasp the moving part.",
        "observation_model_tick": tick,
        "observation_control_tick": tick * 2,
        "camera_clips": clips,
        "state28": [float(index) for index in range(28)],
        "state_layout": config["features"]["state"]["names"],
        "model_action10_chunk": [
            [float(row * 10 + column) for column in range(10)]
            for row in range(20)
        ],
        "action_rate_hz": 25,
        "future_offsets_model_ticks": list(range(1, 21)),
    }


def _write_episode(root: Path, config: dict, ticks: range) -> None:
    for camera in ("head_rgb", "wrist_rgb"):
        for name in ("old", "current"):
            path = root / "cameras" / camera / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
    export = root / "exports" / "conveyorvla_al0_temporal.jsonl"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_text(
        "".join(json.dumps(_record(config, tick)) + "\n" for tick in ticks),
        encoding="utf-8",
    )


def test_lerobot_v3_config_freezes_multirate_training_schema() -> None:
    config = load_lerobot_v3_config()
    features = lerobot_features(config)

    assert config["sampling"] == {
        "control_hz": 50,
        "source_model_hz": 25,
        "query_fps": 5,
        "query_stride_model_ticks": 5,
        "query_anchor": "first_eligible_record",
        "history_offsets_model_ticks": [-2, 0],
        "history_span_s": 0.08,
        "action_rate_hz": 25,
        "action_horizon": 20,
        "action_horizon_s": 0.8,
    }
    assert tuple(features) == (*VIDEO_FEATURE_KEYS, "observation.state", "action")
    assert features["action"]["shape"] == (200,)
    assert len(features["action"]["names"]) == 200


def test_query_grid_selects_every_fifth_source_tick(tmp_path: Path) -> None:
    config = load_lerobot_v3_config()
    _write_episode(tmp_path, config, range(6, 19))
    path = tmp_path / "exports" / "conveyorvla_al0_temporal.jsonl"

    selected = list(iter_query_records(path, config))

    assert [record["observation_model_tick"] for record in selected] == [6, 11, 16]


def test_frame_mapping_preserves_two_cameras_and_20x10_actions(tmp_path: Path) -> None:
    config = load_lerobot_v3_config()
    _write_episode(tmp_path, config, range(6, 7))
    record = _record(config, 6)

    frame = lerobot_frame_from_record(
        record,
        tmp_path,
        config,
        image_loader=lambda _path: np.zeros((224, 224, 3), dtype=np.uint8),
    )

    assert tuple(frame) == (*VIDEO_FEATURE_KEYS, "observation.state", "action", "task")
    assert frame["observation.state"].shape == (28,)
    assert frame["action"].shape == (200,)
    assert frame["action"][:11].tolist() == pytest.approx(
        [float(value) for value in range(11)]
    )


def test_pct_d436_raw_frames_are_center_cropped_for_al0(tmp_path: Path) -> None:
    config = load_lerobot_v3_config()
    _write_episode(tmp_path, config, range(6, 7))
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[:, :80] = (255, 0, 0)
    raw[:, 80:560] = (17, 23, 42)
    raw[:, 560:] = (0, 0, 255)

    frame = lerobot_frame_from_record(
        _record(config, 6),
        tmp_path,
        config,
        image_loader=lambda _path: raw,
    )

    for key in VIDEO_FEATURE_KEYS:
        assert frame[key].shape == (224, 224, 3)
        assert frame[key].dtype == np.uint8
        assert np.all(frame[key] == (17, 23, 42))
    assert np.asarray(
        config["image_preprocessing"]["effective_intrinsics_224"]
    ) == pytest.approx(
        np.asarray(
            [
                [178.94150444333334, 0.0, 114.022906032],
                [0.0, 178.97937959066667, 111.48795223066668],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def test_decoded_lerobot_row_maps_to_temporal_policy_example() -> None:
    normalizer = M0MobileNormalizer.from_config(
        load_temporal_config(),
        {"mean": [0.0] * 28, "std": [1.0] * 28},
    )
    images = [object() for _ in VIDEO_FEATURE_KEYS]
    frame = {
        **dict(zip(VIDEO_FEATURE_KEYS, images, strict=True)),
        "observation.state": np.zeros(28, dtype=np.float32),
        "action": np.zeros(200, dtype=np.float32),
        "task": " Grasp the moving part. ",
    }

    example = lerobot_model_example(frame, normalizer)

    assert example["video"] == ((images[0], images[1]), (images[2], images[3]))
    assert example["lang"] == "Grasp the moving part."
    assert len(example["state"][0]) == 28
    assert len(example["action"]) == 20
    assert len(example["action"][0]) == 10


def test_writer_saves_one_lerobot_episode_after_query_downsampling(tmp_path: Path) -> None:
    config = load_lerobot_v3_config()
    _write_episode(tmp_path, config, range(6, 19))

    class FakeDataset:
        def __init__(self) -> None:
            self.frames = []
            self.saved_episodes = 0

        def add_frame(self, frame: dict) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.saved_episodes += 1

    dataset = FakeDataset()
    report = write_lerobot_episodes(
        dataset,
        [tmp_path],
        config,
        image_loader=lambda _path: np.zeros((224, 224, 3), dtype=np.uint8),
    )

    assert dataset.saved_episodes == 1
    assert len(dataset.frames) == report["frame_count"] == 3
    assert report["episodes"][0]["first_observation_model_tick"] == 6
    assert report["episodes"][0]["last_observation_model_tick"] == 16


def test_config_rejects_training_query_rate_that_mismatches_stride(tmp_path: Path) -> None:
    config = load_lerobot_v3_config()
    config["sampling"]["query_fps"] = 10
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(M0MobileError, match="sampling.query_fps"):
        load_lerobot_v3_config(path)
