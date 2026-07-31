import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_dataset.py"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        text=True,
        capture_output=True,
        check=False,
    )


def _make_valid_run(
    tmp_path: Path,
    *,
    video_enabled: bool = False,
    include_overview: bool = True,
) -> dict[str, Path]:
    run_id = "run-20260730T120000Z"
    episode_id = f"{run_id}-ep0000-seed3"
    episode_dir = tmp_path / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    camera_names = ["head_rgb", "wrist_rgb"]
    if include_overview:
        camera_names.append("overview_rgb")

    manifest = {
        "benchmark_config": {
            "protocol_version": "conveyor-bench-v0",
            "physics_hz": 200,
            "control_hz": 50,
            "camera_hz": 25,
            "evaluation": {
                "hold_time_s": 1.0,
                "lift_height_m": 0.05,
                "static_belt_tolerance_mps": 0.01,
                "dynamic_belt_min_speed_mps": 0.02,
            },
        },
        "episode": {
            "episode_id": episode_id,
            "run_id": run_id,
            "protocol_version": "conveyor-bench-v0",
            "env_id": 0,
            "metadata": {
                "video_enabled": video_enabled,
                "cameras": {name: {} for name in camera_names},
            },
            "task": {
                "task_id": "c0-static-seed-3",
                "task_type": "c0_static_pick",
                "belt_surface_z_m": 0.7,
                "exit_x_m": 1.0,
                "max_duration_s": 10.0,
            },
        },
    }
    steps = [
        {
            "sim_step": index,
            "sim_time_s": time_s,
            "env_id": 0,
            "object_xyz": [0.5, 0.0, 0.76],
            "gripper_closed": True,
            "left_contact": True,
            "right_contact": True,
            "target_in_gripper": True,
            "target_crossed_exit": False,
            "robot_fallen": False,
            "forbidden_collision": False,
            "wrong_object_grasped": False,
        }
        for index, time_s in enumerate((0.0, 0.5, 1.0))
    ]
    events = [
        {"kind": "episode_start", "time_s": 0.0, "payload": {}},
        {"kind": "grasp_verified", "time_s": 1.0, "payload": {}},
        {
            "kind": "episode_end",
            "time_s": 1.0,
            "payload": {"success": True, "failure_reason": "none"},
        },
    ]
    metrics = {
        "sample_count": 3,
        "verification_time_s": 1.0,
        "completion_time_s": 1.0,
        "max_lift_m": 0.06,
        "hold_time_required_s": 1.0,
        "lift_height_required_m": 0.05,
    }
    episode_summary = {
        "episode_id": episode_id,
        "task_id": "c0-static-seed-3",
        "task_type": "c0_static_pick",
        "status": "success",
        "success": True,
        "failure_reason": "none",
        "sample_count": 3,
        "event_count": 3,
        "metrics": metrics,
    }
    video_frames = 2 if video_enabled else 0
    run_summary = {
        "run_id": run_id,
        "protocol_version": "conveyor-bench-v0",
        "task_type": "c0_static_pick",
        "requested_episodes": 1,
        "successful_episodes": 1,
        "episodes": [
            {
                "episode_id": episode_id,
                "path": str(episode_dir),
                "success": True,
                "failure_reason": "none",
                "metrics": metrics,
                "video_frames": video_frames,
            }
        ],
    }

    paths = {
        "run_summary": tmp_path / f"{run_id}-summary.json",
        "manifest": episode_dir / "manifest.json",
        "events": episode_dir / "events.jsonl",
        "steps": episode_dir / "steps.jsonl",
        "episode_summary": episode_dir / "summary.json",
        "head_video": episode_dir / "head_rgb.mp4",
        "wrist_video": episode_dir / "wrist_rgb.mp4",
        "overview_video": episode_dir / "overview_rgb.mp4",
        "frames": episode_dir / "camera_frames.jsonl",
    }
    _write_json(paths["run_summary"], run_summary)
    _write_json(paths["manifest"], manifest)
    _write_jsonl(paths["events"], events)
    _write_jsonl(paths["steps"], steps)
    _write_json(paths["episode_summary"], episode_summary)

    if video_enabled:
        paths["head_video"].write_bytes(b"fake-mp4-head")
        paths["wrist_video"].write_bytes(b"fake-mp4-wrist")
        if include_overview:
            paths["overview_video"].write_bytes(b"fake-mp4-overview")
        _write_jsonl(
            paths["frames"],
            [
                {"frame_index": 0, "sim_step": 0, "sim_time_s": 0.0},
                {"frame_index": 1, "sim_step": 2, "sim_time_s": 1.0},
            ],
        )
    return paths


def test_accepts_complete_success_run_without_video(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 0, completed.stderr
    assert "1 run(s), 1 episode(s), 3 sample(s)" in completed.stdout


def test_accepts_transverse_exit_geometry_and_checks_its_flags(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    manifest = _read_json(paths["manifest"])
    task = manifest["episode"]["task"]
    task.pop("exit_x_m")
    task["transport_direction_xyz"] = [0.0, -1.0, 0.0]
    task["exit_plane_point_xyz"] = [0.70, -0.57, 0.67]
    _write_json(paths["manifest"], manifest)

    completed = _run_validator(paths["run_summary"])
    assert completed.returncode == 0, completed.stderr

    steps = [
        json.loads(line)
        for line in paths["steps"].read_text(encoding="utf-8").splitlines()
    ]
    steps[0]["target_crossed_exit"] = True
    _write_jsonl(paths["steps"], steps)

    completed = _run_validator(paths["run_summary"])
    assert completed.returncode == 1
    assert "disagrees with exit geometry" in completed.stderr


def test_accepts_published_runtime_error_episode(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    run_summary = _read_json(paths["run_summary"])
    run_summary["successful_episodes"] = 0
    run_summary["episodes"][0]["success"] = False
    run_summary["episodes"][0]["failure_reason"] = "runtime_error"
    _write_json(paths["run_summary"], run_summary)

    episode_summary = _read_json(paths["episode_summary"])
    episode_summary["status"] = "failure"
    episode_summary["success"] = False
    episode_summary["failure_reason"] = "runtime_error"
    _write_json(paths["episode_summary"], episode_summary)

    events = [
        json.loads(line)
        for line in paths["events"].read_text(encoding="utf-8").splitlines()
    ]
    events[-1]["payload"] = {
        "success": False,
        "failure_reason": "runtime_error",
    }
    _write_jsonl(paths["events"], events)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 0, completed.stderr


def test_rejects_unreadable_run_summary(tmp_path) -> None:
    summary_path = tmp_path / "run-invalid-summary.json"
    summary_path.write_text("{not-json\n", encoding="utf-8")

    completed = _run_validator(summary_path)

    assert completed.returncode == 1
    assert "cannot read JSON" in completed.stderr


@pytest.mark.parametrize("filename", ("manifest", "events", "steps", "episode_summary"))
def test_rejects_missing_episode_artifact(tmp_path, filename) -> None:
    paths = _make_valid_run(tmp_path)
    paths[filename].unlink()

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "required" in completed.stderr


def test_rejects_unreadable_jsonl(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    paths["steps"].write_text('{"sim_step":0}\nnot-json\n', encoding="utf-8")

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "invalid JSON" in completed.stderr


def test_rejects_sample_count_and_non_monotonic_steps(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    summary = _read_json(paths["episode_summary"])
    summary["sample_count"] = 2
    _write_json(paths["episode_summary"], summary)
    rows = [
        json.loads(line)
        for line in paths["steps"].read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["sim_step"] = rows[0]["sim_step"]
    rows[1]["sim_time_s"] = rows[0]["sim_time_s"]
    _write_jsonl(paths["steps"], rows)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "sample_count 2 does not match 3" in completed.stderr
    assert "sim_step must increase strictly" in completed.stderr
    assert "sim_time_s must increase strictly" in completed.stderr


def test_rejects_non_monotonic_event_time(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    events = [
        {"kind": "episode_start", "time_s": 0.0, "payload": {}},
        {"kind": "grasp_verified", "time_s": 1.0, "payload": {}},
        {
            "kind": "episode_end",
            "time_s": 0.5,
            "payload": {"success": True, "failure_reason": "none"},
        },
    ]
    _write_jsonl(paths["events"], events)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "event time_s must be non-decreasing" in completed.stderr


def test_rejects_success_without_raw_grasp_evidence(tmp_path) -> None:
    paths = _make_valid_run(tmp_path)
    rows = [
        json.loads(line)
        for line in paths["steps"].read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        row["target_in_gripper"] = False
    _write_jsonl(paths["steps"], rows)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "no sustained secure-grasp evidence" in completed.stderr


def test_accepts_enabled_video_with_aligned_frame_index(tmp_path) -> None:
    paths = _make_valid_run(tmp_path, video_enabled=True)

    completed = _run_validator(tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "2 video frame(s)" in completed.stdout


def test_accepts_legacy_two_camera_video_contract(tmp_path) -> None:
    paths = _make_valid_run(
        tmp_path,
        video_enabled=True,
        include_overview=False,
    )

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 0, completed.stderr


def test_rejects_incomplete_video_and_bad_frame_index(tmp_path) -> None:
    paths = _make_valid_run(tmp_path, video_enabled=True)
    paths["overview_video"].unlink()
    frames = [
        {"frame_index": 0, "sim_step": 0, "sim_time_s": 0.0},
        {"frame_index": 2, "sim_step": 1, "sim_time_s": 0.75},
    ]
    _write_jsonl(paths["frames"], frames)

    completed = _run_validator(paths["run_summary"])

    assert completed.returncode == 1
    assert "enabled video artifact is missing" in completed.stderr
    assert "frame_index must be contiguous from zero" in completed.stderr
    assert "time does not match its step" in completed.stderr
