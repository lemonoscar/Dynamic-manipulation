import json

import pytest

from conveyor_bench import (
    BenchmarkConfig,
    EpisodeManifest,
    EpisodeRecorder,
    Event,
    EventKind,
    FailureReason,
    StepSample,
    TaskManifest,
    TaskType,
)


def manifest(episode_id: str) -> EpisodeManifest:
    task = TaskManifest(
        task_id="task-static-001",
        task_type=TaskType.C0_STATIC_PICK,
        instruction="pick the cube",
        target_object_id="cube",
        object_ids=("cube",),
        seed=11,
        belt_speed_mps=0.0,
        belt_surface_z_m=0.7,
        exit_x_m=1.0,
    )
    return EpisodeManifest(
        episode_id=episode_id,
        run_id="smoke-run",
        protocol_version=BenchmarkConfig.v0().protocol_version,
        task=task,
        created_at_utc="2026-07-30T00:00:00+00:00",
        seeds={"task": 11, "physics": 12},
    )


def sample(step: int, time_s: float, *, secure: bool) -> StepSample:
    return StepSample(
        sim_step=step,
        sim_time_s=time_s,
        env_id=0,
        object_xyz=(0.0, 0.0, 0.76 if secure else 0.71),
        object_linear_velocity=(0.0, 0.0, 0.0),
        tcp_xyz=(0.0, 0.0, 0.8),
        belt_command_speed_mps=0.0,
        belt_measured_speed_mps=0.0,
        gripper_closed=secure,
        left_contact=secure,
        right_contact=secure,
        target_in_gripper=secure,
        target_crossed_exit=False,
        robot_fallen=False,
        forbidden_collision=False,
        phase="verify",
        action={"arm": [0.0] * 6, "gripper": float(secure)},
    )


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_recorder_publishes_complete_success_episode(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-success"))
    recorder.record_event(Event(EventKind.EPISODE_START, 0.0, {"seed": 11}))
    recorder.record_step(sample(0, 0.0, secure=True))
    recorder.record_step(sample(1, 0.5, secure=True))
    recorder.record_step(sample(2, 1.0, secure=True))
    episode_path, result = recorder.finalize()

    assert result.success
    assert episode_path == tmp_path / "episodes" / "ep-success"
    assert sorted(path.name for path in episode_path.iterdir()) == [
        "events.jsonl",
        "manifest.json",
        "steps.jsonl",
        "summary.json",
    ]
    assert read_json(episode_path / "summary.json")["status"] == "success"
    assert read_json(episode_path / "summary.json")["sample_count"] == 3
    assert len((episode_path / "steps.jsonl").read_text().splitlines()) == 3
    assert not list((tmp_path / "episodes").glob("*.inprogress"))


def test_failed_episode_is_published_instead_of_discarded(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-failed"))
    recorder.record_step(sample(0, 0.0, secure=False))
    episode_path, result = recorder.finalize()

    assert not result.success
    assert result.failure_reason is FailureReason.GRASP_NOT_SECURED
    assert episode_path.exists()
    summary = read_json(episode_path / "summary.json")
    assert summary["status"] == "failure"
    assert summary["failure_reason"] == "grasp_not_secured"
    assert (episode_path / "steps.jsonl").exists()


def test_exception_aborts_but_retains_episode(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="controller failed"):
        with EpisodeRecorder(tmp_path, manifest("ep-aborted")) as recorder:
            recorder.record_step(sample(0, 0.0, secure=False))
            raise RuntimeError("controller failed")

    episode_path = tmp_path / "episodes" / "ep-aborted"
    assert episode_path.exists()
    summary = read_json(episode_path / "summary.json")
    assert summary["failure_reason"] == "aborted"
    assert summary["metrics"]["abort_metadata"]["exception_type"] == "RuntimeError"


def test_runtime_error_abort_is_not_labeled_as_recorder_error(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-runtime-error"))
    recorder.record_step(sample(0, 0.0, secure=False))
    episode_path, result = recorder.abort(
        FailureReason.RUNTIME_ERROR,
        {"operation": "control_ik", "exception_type": "IKConvergenceError"},
    )

    assert result.failure_reason is FailureReason.RUNTIME_ERROR
    summary = read_json(episode_path / "summary.json")
    assert summary["failure_reason"] == "runtime_error"
    assert summary["metrics"]["abort_metadata"]["operation"] == "control_ik"


def test_recorder_rejects_non_monotonic_samples(tmp_path) -> None:
    recorder = EpisodeRecorder(tmp_path, manifest("ep-order"))
    recorder.record_step(sample(1, 0.1, secure=False))
    with pytest.raises(ValueError, match="sim_step"):
        recorder.record_step(sample(1, 0.2, secure=False))
    recorder.abort()
