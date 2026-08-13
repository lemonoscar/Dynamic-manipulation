import json
from dataclasses import replace

import pytest

from conveyor_bench.schema import (
    ActionChunkProfile,
    ActionChunkTrace,
    BenchmarkConfig,
    CanonicalAction,
    EpisodeManifest,
    EpisodeRecorder,
    Event,
    EventKind,
    FailureReason,
    FutureObjectState,
    GoalZone,
    JointState,
    ObjectInstance,
    ObjectState,
    Pose,
    RobotMode,
    StepSample,
    TaskManifest,
    TaskType,
    Twist,
)


def pose(xyz=(0.0, 0.0, 0.0)) -> Pose:
    return Pose(xyz, (1.0, 0.0, 0.0, 0.0))


def zero_twist() -> Twist:
    return Twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def action() -> CanonicalAction:
    return CanonicalAction((0.0,) * 10)


def manifest(episode_id: str) -> EpisodeManifest:
    task = TaskManifest(
        task_id="sort-task",
        task_type=TaskType.DYNAMIC_SORT,
        robot_mode=RobotMode.FIXED_BASE,
        instruction="put the can in the bin",
        objects=(ObjectInstance("target", "asset-can", "can", "zone-a"),),
        goal_zones=(GoalZone("zone-a", (0.0, 0.0, 0.5), (1.0, 1.0, 1.0)),),
        scored_object_ids=("target",),
        seed=4,
        belt_speed_mps=0.1,
        belt_surface_z_m=0.67,
        transport_direction_xyz=(0.0, -1.0, 0.0),
        exit_plane_point_xyz=(0.0, -0.6, 0.67),
    )
    return EpisodeManifest(
        episode_id=episode_id,
        run_id="run-smoke",
        protocol_version="conveyor-bench-v1",
        task=task,
        created_at_utc="2026-07-30T00:00:00+00:00",
        asset_hashes={"asset-can": "sha256:" + "a" * 64},
        seeds={"task": 4},
    )


def labels(obj: ObjectState) -> tuple[FutureObjectState, ...]:
    return tuple(
        FutureObjectState(
            instance_id=obj.instance_id,
            horizon_steps=horizon,
            valid=True,
            pose_world=obj.pose_world,
            twist_world=obj.twist_world,
        )
        for horizon in BenchmarkConfig.v1().future_horizons_steps
    )


def sample(step: int, time_s: float, *, held: bool) -> StepSample:
    obj = ObjectState(
        "target",
        pose((0.5, 0.5, 0.7)),
        zero_twist(),
        in_gripper=held,
    )
    return StepSample(
        sim_step=step,
        sim_time_s=time_s,
        model_tick=step,
        env_id=0,
        robot_root_world=pose(),
        robot_twist_world=zero_twist(),
        tcp_base=pose((0.4, 0.0, 0.5)),
        joints=JointState(("joint-1",), (0.0,), (0.0,)),
        action=action(),
        objects=(obj,),
        left_contact_object_ids=("target",) if held else (),
        right_contact_object_ids=("target",) if held else (),
        camera_frames=(),
        future_object_states=labels(obj),
        phase="place",
        selected_object_id="target",
        action_chunk_id="chunk-001",
        action_index_in_chunk=step,
    )


def action_chunk() -> ActionChunkTrace:
    return ActionChunkTrace(
        chunk_id="chunk-001",
        profile=ActionChunkProfile.M0,
        source_observation_tick=0,
        source_observation_time_s=0.0,
        valid_from_tick=0,
        valid_until_tick=16,
        execute_from_tick=0,
        execute_until_tick=16,
        actions=tuple(action() for _ in range(16)),
    )


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_recorder_streams_normalized_v1_episode_without_sample_buffer(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-success"))
    assert not hasattr(recorder, "_samples")
    recorder.record_action_chunk(action_chunk())
    recorder.record_event(Event(EventKind.EPISODE_START, 0.0, sim_step=0))
    recorder.record_step(sample(0, 0.0, held=True))
    recorder.record_step(sample(1, 0.1, held=False))
    recorder.record_step(sample(2, 0.6, held=False))
    episode_path, result = recorder.finalize()

    assert result.success
    assert sorted(path.name for path in episode_path.iterdir()) == [
        "action_chunks.jsonl",
        "events.jsonl",
        "manifest.json",
        "objects.jsonl",
        "steps.jsonl",
        "summary.json",
    ]
    summary = read_json(episode_path / "summary.json")
    assert summary["status"] == "success"
    assert summary["sample_count"] == 3
    assert summary["object_record_count"] == 3
    assert summary["action_chunk_count"] == 1

    step_rows = read_jsonl(episode_path / "steps.jsonl")
    object_rows = read_jsonl(episode_path / "objects.jsonl")
    assert "objects" not in step_rows[0]
    assert "future_object_states" not in step_rows[0]
    assert object_rows[0]["state"]["instance_id"] == "target"
    assert len(object_rows[0]["future_object_states"]) == 5
    assert not list((tmp_path / "episodes").glob("*.inprogress"))


def test_abort_publishes_failure_trajectory_and_all_streams(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-runtime-error"))
    recorder.record_action_chunk(action_chunk())
    recorder.record_step(sample(0, 0.0, held=True))
    episode_path, result = recorder.abort(
        FailureReason.RUNTIME_ERROR,
        {"operation": "whole_body_control"},
    )

    assert result.failure_reason is FailureReason.RUNTIME_ERROR
    assert episode_path.exists()
    assert len(read_jsonl(episode_path / "steps.jsonl")) == 1
    summary = read_json(episode_path / "summary.json")
    assert summary["failure_reason"] == "runtime_error"
    assert (
        summary["metrics"]["abort_metadata"]["operation"] == "whole_body_control"
    )


def test_context_exception_is_re_raised_after_atomic_failure_publish(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="controller failed"):
        with EpisodeRecorder(tmp_path, manifest("ep-aborted")) as recorder:
            recorder.record_action_chunk(action_chunk())
            recorder.record_step(sample(0, 0.0, held=True))
            raise RuntimeError("controller failed")

    summary = read_json(tmp_path / "episodes" / "ep-aborted" / "summary.json")
    assert summary["failure_reason"] == "aborted"
    assert summary["sample_count"] == 1


def test_recorder_rejects_unknown_or_mismatched_action_chunk_reference(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-action-ref"))
    with pytest.raises(ValueError, match="unknown action chunk"):
        recorder.record_step(sample(0, 0.0, held=True))

    recorder.record_action_chunk(action_chunk())
    mismatched = sample(0, 0.0, held=True)
    mismatched = StepSample(
        **{
            **mismatched.__dict__,
            "action": CanonicalAction((0.0,) * 8 + (0.1, 0.0)),
        }
    )
    with pytest.raises(ValueError, match="does not match"):
        recorder.record_step(mismatched)
    recorder.abort()


def test_recorder_keeps_only_live_chunks_and_allows_repeated_control_tick(
    tmp_path,
) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-live-chunks"))
    recorder.record_action_chunk(action_chunk())
    first = sample(15, 0.1, held=True)
    recorder.record_step(first)
    recorder.record_step(
        replace(
            sample(16, 0.2, held=True),
            model_tick=15,
            action_index_in_chunk=15,
        )
    )
    assert tuple(recorder._active_action_chunks) == ("chunk-001",)

    recorder.record_step(
        replace(
            sample(17, 0.3, held=True),
            model_tick=16,
            action_chunk_id=None,
            action_index_in_chunk=None,
        )
    )
    assert not recorder._active_action_chunks
    episode_path, _ = recorder.abort()
    assert len(read_jsonl(episode_path / "action_chunks.jsonl")) == 1


def test_episode_end_event_does_not_move_event_time_backwards(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-event-order"))
    recorder.record_event(Event(EventKind.PHASE_CHANGED, 2.0))
    episode_path, _ = recorder.abort()

    event_rows = read_jsonl(episode_path / "events.jsonl")
    assert [row["time_s"] for row in event_rows] == [2.0, 2.0]
