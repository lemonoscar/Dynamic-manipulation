"""Synthetic data-plane smoke test for the ConveyorBench V2 pipeline.

This test deliberately does not exercise Isaac Sim, RTX rendering, contact
physics, or controller quality.  It proves only that a structurally complete
two-target trajectory can pass through the canonical recorder, strict V2
validation, and both VLA projections without mutating the source episode.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

from conveyor_bench.v1 import (
    BenchmarkConfig,
    CameraFrameRef,
    CanonicalAction,
    EpisodeManifest,
    EpisodeRecorder,
    Event,
    EventKind,
    FutureObjectState,
    JointState,
    ObjectState,
    Pose,
    StepSample,
    Twist,
)
from conveyor_bench.v1.tasking import CurriculumSplit, TaskFamily
from conveyor_bench.v2.camera_contracts import camera_contract_for_scene
from conveyor_bench.v2.config import SceneId
from conveyor_bench.v2.exporters import (
    EXPORT_SCHEMA_VERSION,
    iter_dynamicvla_records,
    iter_m0_records,
)
from conveyor_bench.v2.tasking import build_task_context
from conveyor_bench.v2.validation import validate_v2_episode


_CANONICAL_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
)
_ZERO_TWIST = Twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _pose(xyz: tuple[float, float, float]) -> Pose:
    return Pose(xyz, _IDENTITY_WXYZ)


def _write_png(path: Path, width: int, height: int) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", checksum)
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(
        b"\x00" + b"\x00\x00\x00" * width for _ in range(height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _unavailable_future_labels(
    objects: tuple[ObjectState, ...],
) -> tuple[FutureObjectState, ...]:
    """Avoid claiming synthetic states are physically realized predictions."""

    return tuple(
        FutureObjectState(
            instance_id=obj.instance_id,
            horizon_steps=horizon,
            valid=False,
            pose_world=None,
            twist_world=None,
            invalid_reason="synthetic_data_plane_smoke",
        )
        for obj in objects
        for horizon in BenchmarkConfig.v1().future_horizons_steps
    )


def _canonical_hashes(episode_path: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((episode_path / name).read_bytes()).hexdigest()
        for name in _CANONICAL_FILES
    }


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _deduplicate(values: list[object]) -> tuple[object, ...]:
    result: list[object] = []
    for value in values:
        if value is not None and (not result or result[-1] != value):
            result.append(value)
    return tuple(result)


def test_synthetic_continuous_episode_crosses_the_v2_data_pipeline(
    tmp_path: Path,
) -> None:
    """Recorder -> validator -> DynamicVLA/M0, with no Isaac/GPU claims."""

    context = build_task_context(
        seed=7,
        scene_id=SceneId.TRANSVERSE_NEAR_SORT_V2,
        family=TaskFamily.CONTINUOUS_MULTI_TARGET,
        mode="fixed_base",
        split=CurriculumSplit.TRAIN,
    )
    target_a, target_b = context.target_sequence_ids
    assert tuple(obj.instance_id for obj in context.task.objects) == (
        target_a,
        target_b,
    )

    suite = context.task.metadata["benchmark_suite"]
    camera_contracts = camera_contract_for_scene(
        SceneId.TRANSVERSE_NEAR_SORT_V2
    )
    manifest = EpisodeManifest(
        episode_id="ep-v2-synthetic-data-plane",
        run_id="run-v2-synthetic-data-plane",
        protocol_version="conveyor-bench-v1",
        task=context.task,
        created_at_utc="2026-07-31T00:00:00+00:00",
        env_id=0,
        asset_hashes={
            obj.asset_id: "sha256:" + hashlib.sha256(
                obj.asset_id.encode("utf-8")
            ).hexdigest()
            for obj in context.task.objects
        },
        seeds={"episode": 7},
        metadata={
            "benchmark_suite": suite,
            "cameras": camera_contracts,
        },
    )
    recorder = EpisodeRecorder(tmp_path, manifest)

    zone_by_id = context.task.goal_zone_by_id
    destination_by_target = context.destination_zone_by_target
    parked_xyz = (3.0, 0.0, -1.0)
    belt_xyz = (0.0, 0.0, context.task.belt_surface_z_m + 0.05)
    sample_index = 0
    camera_index_rows: list[dict] = []

    def sample_time(index: int) -> float:
        return 0.50 + (index + 1) / BenchmarkConfig.v1().control_hz

    def sample_step(index: int) -> int:
        return round(sample_time(index) * BenchmarkConfig.v1().physics_hz)

    def record_sample(
        *,
        selected: str,
        active: frozenset[str],
        placed: frozenset[str] = frozenset(),
        held: str | None = None,
        phase: str,
    ) -> StepSample:
        nonlocal sample_index
        objects: list[ObjectState] = []
        for target_id in context.target_sequence_ids:
            zone_id = destination_by_target[target_id]
            if target_id in placed:
                xyz = tuple(
                    (lower + upper) / 2.0
                    for lower, upper in zip(
                        zone_by_id[zone_id].min_xyz,
                        zone_by_id[zone_id].max_xyz,
                        strict=True,
                    )
                )
            elif target_id in active:
                xyz = belt_xyz
            else:
                xyz = parked_xyz
            objects.append(
                ObjectState(
                    instance_id=target_id,
                    pose_world=_pose(xyz),
                    twist_world=_ZERO_TWIST,
                    active=target_id in active,
                    in_gripper=target_id == held,
                    crossed_exit=False,
                )
            )
        resolved_objects = tuple(objects)
        time_s = sample_time(sample_index)
        camera_frames: tuple[CameraFrameRef, ...] = ()
        if sample_index % 2 == 1:
            model_tick = sample_index // 2
            references: list[CameraFrameRef] = []
            index_entries: dict[str, dict] = {}
            for camera_id, contract in camera_contracts.items():
                relative_path = f"cameras/{camera_id}/{model_tick:06d}.png"
                width, height = contract["resolution"]
                _write_png(
                    recorder.artifact_directory / relative_path,
                    width,
                    height,
                )
                references.append(
                    CameraFrameRef(
                        camera_id=camera_id,
                        frame_index=model_tick,
                        capture_time_s=time_s,
                        relative_path=relative_path,
                    )
                )
                index_entries[camera_id] = {
                    "relative_path": relative_path,
                    "resolution": contract["resolution"],
                    "role": contract["role"],
                    "quality": {
                        "dark_fraction": 0.0,
                        "laplacian_variance": 100.0,
                    },
                }
            camera_frames = tuple(references)
            camera_index_rows.append(
                {
                    "frame_index": model_tick,
                    "sim_step": sample_step(sample_index),
                    "capture_time_s": time_s,
                    "frames": index_entries,
                }
            )
        sample = StepSample(
            sim_step=sample_step(sample_index),
            sim_time_s=time_s,
            model_tick=sample_index // 2,
            env_id=0,
            robot_root_world=_pose((0.0, 0.0, 0.32)),
            robot_twist_world=_ZERO_TWIST,
            tcp_base=_pose((0.38, 0.0, 0.35)),
            joints=JointState(("joint-1",), (0.0,), (0.0,)),
            action=CanonicalAction((0.0,) * 10),
            objects=resolved_objects,
            left_contact_object_ids=((held,) if held is not None else ()),
            right_contact_object_ids=((held,) if held is not None else ()),
            camera_frames=camera_frames,
            future_object_states=_unavailable_future_labels(resolved_objects),
            phase=phase,
            selected_object_id=selected,
            belt_measured_speed_mps=context.task.belt_speed_mps,
            metadata={"synthetic_data_plane_smoke": True},
        )
        recorder.record_step(sample)
        sample_index += 1
        return sample

    recorder.record_event(Event(EventKind.EPISODE_START, 0.0))
    recorder.record_event(
        Event(
            EventKind.TARGET_SELECTED,
            0.50,
            object_instance_id=target_a,
        )
    )
    recorder.record_event(
        Event(
            EventKind.OBJECT_SPAWNED,
            0.50,
            object_instance_id=target_a,
            payload={"synthetic_data_plane_smoke": True},
        )
    )
    record_sample(
        selected=target_a,
        active=frozenset({target_a}),
        held=target_a,
        phase="synthetic_hold",
    )
    released_a = record_sample(
        selected=target_a,
        active=frozenset({target_a}),
        placed=frozenset({target_a}),
        phase="synthetic_dwell",
    )
    recorder.record_event(
        Event(
            EventKind.OBJECT_RELEASED,
            released_a.sim_time_s,
            sim_step=released_a.sim_step,
            object_instance_id=target_a,
            goal_zone_id=destination_by_target[target_a],
        )
    )
    last_a = released_a
    while not recorder.online_metrics.snapshot()["object_outcomes"][target_a][
        "completion_time_s"
    ]:
        last_a = record_sample(
            selected=target_a,
            active=frozenset({target_a}),
            placed=frozenset({target_a}),
            phase="synthetic_dwell",
        )
    recorder.record_event(
        Event(
            EventKind.OBJECT_PLACED,
            last_a.sim_time_s,
            sim_step=last_a.sim_step,
            object_instance_id=target_a,
            goal_zone_id=destination_by_target[target_a],
        )
    )

    second_gate_time = context.service_gates[1].not_before_s
    while sample_time(sample_index) < second_gate_time:
        record_sample(
            selected=target_a,
            active=frozenset({target_a}),
            placed=frozenset({target_a}),
            phase="synthetic_service_wait",
        )
    start_b_time = sample_time(sample_index)
    recorder.record_event(
        Event(
            EventKind.TARGET_SELECTED,
            start_b_time,
            object_instance_id=target_b,
        )
    )
    recorder.record_event(
        Event(
            EventKind.OBJECT_SPAWNED,
            start_b_time,
            object_instance_id=target_b,
            payload={"synthetic_data_plane_smoke": True},
        )
    )
    record_sample(
        selected=target_b,
        active=frozenset({target_a, target_b}),
        placed=frozenset({target_a}),
        held=target_b,
        phase="synthetic_hold",
    )
    released_b = record_sample(
        selected=target_b,
        active=frozenset({target_a, target_b}),
        placed=frozenset({target_a, target_b}),
        phase="synthetic_dwell",
    )
    recorder.record_event(
        Event(
            EventKind.OBJECT_RELEASED,
            released_b.sim_time_s,
            sim_step=released_b.sim_step,
            object_instance_id=target_b,
            goal_zone_id=destination_by_target[target_b],
        )
    )
    last_b = released_b
    while not recorder.online_metrics.success:
        last_b = record_sample(
            selected=target_b,
            active=frozenset({target_a, target_b}),
            placed=frozenset({target_a, target_b}),
            phase="synthetic_dwell",
        )
    recorder.record_event(
        Event(
            EventKind.OBJECT_PLACED,
            last_b.sim_time_s,
            sim_step=last_b.sim_step,
            object_instance_id=target_b,
            goal_zone_id=destination_by_target[target_b],
        )
    )
    (recorder.artifact_directory / "camera_frames.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in camera_index_rows),
        encoding="utf-8",
    )
    episode_path, evaluation = recorder.finalize()
    assert evaluation.success

    validation = validate_v2_episode(episode_path)
    assert validation.ok, "\n".join(validation.errors)

    manifest_json = _read_json(episode_path / "manifest.json")
    assert manifest_json["episode"]["protocol_version"] == "conveyor-bench-v1"
    assert (
        manifest_json["benchmark_config"]["protocol_version"]
        == "conveyor-bench-v1"
    )
    task_suite = manifest_json["episode"]["task"]["metadata"][
        "benchmark_suite"
    ]
    assert task_suite == manifest_json["episode"]["metadata"]["benchmark_suite"]
    assert task_suite["benchmark_suite_version"] == "conveyor-bench-v2"

    steps = _read_jsonl(episode_path / "steps.jsonl")
    object_rows = _read_jsonl(episode_path / "objects.jsonl")
    events = _read_jsonl(episode_path / "events.jsonl")
    expected_registry = set(context.target_sequence_ids)
    registry_by_step: dict[int, set[str]] = {}
    for row in object_rows:
        registry_by_step.setdefault(row["sim_step"], set()).add(
            row["state"]["instance_id"]
        )
    assert set(registry_by_step) == {step["sim_step"] for step in steps}
    assert all(ids == expected_registry for ids in registry_by_step.values())
    assert _deduplicate([step["selected_object_id"] for step in steps]) == (
        target_a,
        target_b,
    )
    assert tuple(
        event["object_instance_id"]
        for event in events
        if event["kind"] == "object_placed"
    ) == (target_a, target_b)

    hashes_before_export = _canonical_hashes(episode_path)
    for iterator in (iter_dynamicvla_records, iter_m0_records):
        records = list(iterator(episode_path))
        assert records
        assert {record["schema_version"] for record in records} == {
            EXPORT_SCHEMA_VERSION
        }
        assert {record["scene_id"] for record in records} == {
            SceneId.TRANSVERSE_NEAR_SORT_V2.value
        }
        assert {record["task_family"] for record in records} == {
            TaskFamily.CONTINUOUS_MULTI_TARGET.value
        }
        assert all(
            record["target_sequence_ids"] == context.target_sequence_ids
            for record in records
        )
        assert _deduplicate(
            [record["current_target_id"] for record in records]
        ) == (target_a, target_b)
        assert _deduplicate(
            [record["current_subtask_index"] for record in records]
        ) == (0, 1)
        assert all(
            record["supervision_only_fields"]
            == ("current_target_id", "current_subtask_index")
            for record in records
        )
        assert all("canonical_action10_chunk" in record for record in records)

    assert _canonical_hashes(episode_path) == hashes_before_export
