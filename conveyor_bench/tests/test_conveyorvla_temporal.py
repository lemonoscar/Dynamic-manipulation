from __future__ import annotations

import json
import math
import queue
from pathlib import Path

import pytest

from conveyor_bench.conveyorvla.streaming import (
    ActionStreamBuffer,
    StreamChunk,
    put_latest,
)
from conveyor_bench.conveyorvla.temporal import (
    ACTION_DIMENSION_MASK,
    ACTION_HORIZON,
    TEMPORAL_PROFILE,
    TEMPORAL_SCHEMA_VERSION,
    load_temporal_config,
    reconstruct_tcp_world,
    relative_tcp_target,
    temporal_sample_from_record,
)
from conveyor_bench.m0_mobile import M0MobileError, M0MobileNormalizer
from conveyor_bench.m0_policy import m0_dit_config


def _chunk(
    *,
    observation_control_tick: int = 0,
    episode_id: str = "episode-a",
    generation_id: str = "generation-a",
) -> StreamChunk:
    return StreamChunk(
        episode_id=episode_id,
        generation_id=generation_id,
        observation_model_tick=observation_control_tick // 2,
        observation_control_tick=observation_control_tick,
        actions=tuple(
            tuple(float(row * 10 + column) for column in range(10))
            for row in range(ACTION_HORIZON)
        ),
        inference_started_s=1.0,
        inference_finished_s=1.2,
    )


def test_temporal_config_freezes_motion_and_latency_contract() -> None:
    config = load_temporal_config()

    assert config["model_identity"]["name"] == "ConveyorVLA AL0"
    assert config["data"]["history_offsets_model_ticks"] == [-2, 0]
    assert config["data"]["action_horizon"] == 20
    assert config["data"]["action_rate_hz"] == 25
    assert config["streaming"]["require_episode_generation_id"] is True
    assert ACTION_DIMENSION_MASK[1] is False
    assert m0_dit_config(config).action_horizon == 20


def test_future_tcp_target_round_trips_with_root_motion_and_rotation() -> None:
    yaw_90 = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    source_root = (1.0, 2.0, 0.0)
    source_tcp = (0.4, 0.1, 0.5)
    future_root = (1.2, 2.1, 0.0)
    future_tcp = (0.5, -0.1, 0.6)
    future_tcp_q = (math.cos(0.1), math.sin(0.1), 0.0, 0.0)

    target = relative_tcp_target(
        source_root,
        yaw_90,
        source_tcp,
        (1.0, 0.0, 0.0, 0.0),
        future_root,
        yaw_90,
        future_tcp,
        future_tcp_q,
    )
    reconstructed_xyz, reconstructed_q = reconstruct_tcp_world(
        source_root,
        yaw_90,
        source_tcp,
        (1.0, 0.0, 0.0, 0.0),
        target,
    )
    expected_xyz = (1.3, 2.6, 0.6)
    expected_q = (
        yaw_90[0] * future_tcp_q[0],
        yaw_90[0] * future_tcp_q[1],
        yaw_90[3] * future_tcp_q[1],
        yaw_90[3] * future_tcp_q[0],
    )

    assert reconstructed_xyz == pytest.approx(expected_xyz)
    assert reconstructed_q == pytest.approx(expected_q)


def test_streaming_skips_stale_prefix_and_merges_by_target_tick() -> None:
    buffer = ActionStreamBuffer()
    buffer.reset("episode-a", "generation-a")

    first = buffer.accept(_chunk(), current_control_tick=6)

    assert first.accepted
    assert first.skipped_actions == 3
    assert first.remaining_actions == 17
    assert first.first_target_control_tick == 8
    assert buffer.next_waypoint(6)[0] == 8

    second = buffer.accept(_chunk(observation_control_tick=8), current_control_tick=10)
    ticks = [tick for tick, _ in buffer.waypoints()]

    assert second.accepted
    assert second.skipped_actions == 1
    assert ticks[0] == 12
    assert ticks[-1] == 48
    assert len(ticks) == len(set(ticks)) == 19


def test_streaming_rejects_old_generation_order_and_fully_stale_chunks() -> None:
    buffer = ActionStreamBuffer()
    buffer.reset("episode-a", "generation-new")

    mismatch = buffer.accept(
        _chunk(generation_id="generation-old"), current_control_tick=0
    )
    assert not mismatch.accepted
    assert mismatch.reason == "generation_mismatch"

    accepted = buffer.accept(
        _chunk(generation_id="generation-new", observation_control_tick=4),
        current_control_tick=4,
    )
    assert accepted.accepted
    out_of_order = buffer.accept(
        _chunk(generation_id="generation-new", observation_control_tick=2),
        current_control_tick=4,
    )
    assert not out_of_order.accepted
    assert out_of_order.reason == "out_of_order_observation"

    buffer.reset("episode-a", "generation-newer")
    stale = buffer.accept(
        _chunk(generation_id="generation-newer"), current_control_tick=40
    )
    assert not stale.accepted
    assert stale.reason == "fully_stale"
    assert stale.remaining_actions == 0
    assert buffer.next_waypoint(40) is None


def test_streaming_rejects_fractional_control_ticks() -> None:
    buffer = ActionStreamBuffer()
    buffer.reset("episode-a", "generation-a")

    with pytest.raises(ValueError, match="non-negative integer"):
        buffer.accept(_chunk(), current_control_tick=1.5)  # type: ignore[arg-type]


def test_latest_queue_replaces_old_result() -> None:
    target: queue.Queue[str] = queue.Queue(maxsize=1)
    target.put_nowait("old")

    put_latest(target, "new")

    assert target.get_nowait() == "new"


def test_temporal_record_loader_keeps_only_policy_inputs(tmp_path: Path) -> None:
    for camera in ("head_rgb", "wrist_rgb"):
        for tick in (4, 6):
            path = tmp_path / "cameras" / camera / f"{tick:06d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")
    clips = [
        {
            "camera_id": camera,
            "history_offsets_model_ticks": [-2, 0],
            "frames": [
                {
                    "camera_id": camera,
                    "relative_path": f"cameras/{camera}/{tick:06d}.png",
                }
                for tick in (4, 6)
            ],
        }
        for camera in ("head_rgb", "wrist_rgb")
    ]
    record = {
        "schema_version": TEMPORAL_SCHEMA_VERSION,
        "profile": TEMPORAL_PROFILE,
        "policy_task_scope": "grasp_only",
        "sample_id": "sample-a",
        "source_episode_id": "episode-a",
        "instruction": "grasp the part",
        "camera_clips": clips,
        "state28": [0.0] * 28,
        "model_action10_chunk": [[0.0] * 10] * 20,
        "action_rate_hz": 25,
        "future_offsets_model_ticks": list(range(1, 21)),
        "observation_model_tick": 6,
        "observation_control_tick": 12,
    }
    sample = temporal_sample_from_record(record, tmp_path)
    normalizer = M0MobileNormalizer.from_config(
        load_temporal_config(), {"mean": [0.0] * 28, "std": [1.0] * 28}
    )
    example = sample.as_model_example(normalizer)

    assert tuple(path.name for path in example["video"][0]) == (
        "000004.png",
        "000006.png",
    )
    assert example["action_mask"] == ACTION_DIMENSION_MASK
    assert "observation_reference" not in example

    escaped = json.loads(json.dumps(record))
    escaped["camera_clips"][0]["frames"][0]["relative_path"] = "../escape.png"
    with pytest.raises(M0MobileError, match="escapes episode root"):
        temporal_sample_from_record(escaped, tmp_path, require_images=False)
