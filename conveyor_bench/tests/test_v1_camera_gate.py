import json
from pathlib import Path

import numpy as np
import pytest

from conveyor_bench.v1.camera_gate import (
    CameraGateError,
    audit_camera_episode,
)


CAMERA_ROLES = {
    "head_rgb": "policy_observation",
    "wrist_rgb": "policy_observation",
    "overview_rgb": "observer_only",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _moving_square(index: int) -> np.ndarray:
    image = np.full((48, 48, 3), 24, dtype=np.uint8)
    x0 = 2 + index * 3
    image[18:30, x0 : x0 + 12] = (220, 30, 30)
    return image


def _make_episode(
    tmp_path: Path,
    *,
    moving_policy: bool,
    moving_overview: bool,
    wrong_overview_role: bool = False,
) -> tuple[Path, dict[Path, np.ndarray]]:
    episode = tmp_path / "episode"
    episode.mkdir()
    images: dict[Path, np.ndarray] = {}
    frame_rows = []
    steps = []
    for frame_index in range(10):
        sim_step = (frame_index + 1) * 16
        frames = {}
        references = []
        for camera_id, role in CAMERA_ROLES.items():
            relative_path = (
                Path("cameras")
                / camera_id
                / f"{frame_index:06d}.png"
            )
            path = episode / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test image supplied by image_loader")
            frames[camera_id] = {
                "relative_path": relative_path.as_posix(),
                "resolution": [48, 48],
                "role": (
                    "policy_observation"
                    if wrong_overview_role and camera_id == "overview_rgb"
                    else role
                ),
            }
            references.append(
                {
                    "camera_id": camera_id,
                    "frame_index": frame_index,
                    "capture_time_s": (frame_index + 1) * 0.04,
                    "relative_path": relative_path.as_posix(),
                }
            )
            moves = (
                moving_overview
                if camera_id == "overview_rgb"
                else moving_policy
            )
            images[path.resolve()] = (
                _moving_square(frame_index)
                if moves
                else np.full((48, 48, 3), 24, dtype=np.uint8)
            )
        frame_rows.append(
            {
                "frame_index": frame_index,
                "sim_step": sim_step,
                "capture_time_s": (frame_index + 1) * 0.04,
                "frames": frames,
            }
        )
        steps.append(
            {
                "sim_step": sim_step,
                "selected_object_id": "target",
                "robot_root_world": {
                    "xyz": [frame_index * 0.01, 0.0, 0.3]
                },
                "metadata": {
                    "tcp_world": {
                        "xyz": [0.3 + frame_index * 0.01, 0.0, 0.7]
                    }
                },
                "camera_frames": references,
            }
        )
    _write_jsonl(episode / "camera_frames.jsonl", frame_rows)
    _write_jsonl(episode / "steps.jsonl", steps)
    _write_jsonl(
        episode / "events.jsonl",
        [
            {
                "kind": "object_spawned",
                "object_instance_id": "target",
                "sim_step": 16,
            },
            {
                "kind": "object_placed",
                "object_instance_id": "target",
                "sim_step": 160,
            },
        ],
    )
    return episode, images


def test_gate_accepts_temporal_policy_and_observer_frames(
    tmp_path: Path,
) -> None:
    episode, images = _make_episode(
        tmp_path,
        moving_policy=True,
        moving_overview=True,
    )

    report = audit_camera_episode(
        episode,
        image_loader=images.__getitem__,
    )

    assert report["passed"] is True
    assert report["issues"] == []
    assert (
        report["metrics"]["observer_only_counts_as_policy_evidence"] is False
    )


def test_gate_rejects_geometrically_frozen_frames(tmp_path: Path) -> None:
    episode, images = _make_episode(
        tmp_path,
        moving_policy=False,
        moving_overview=False,
    )

    report = audit_camera_episode(
        episode,
        image_loader=images.__getitem__,
    )
    issue_codes = [issue["code"] for issue in report["issues"]]

    assert report["passed"] is False
    assert issue_codes.count("camera_geometry_frozen") == 3
    assert "target_not_visible_in_policy_cameras" in issue_codes


def test_observer_motion_cannot_satisfy_policy_target_visibility(
    tmp_path: Path,
) -> None:
    episode, images = _make_episode(
        tmp_path,
        moving_policy=False,
        moving_overview=True,
    )

    report = audit_camera_episode(
        episode,
        image_loader=images.__getitem__,
    )
    issues = report["issues"]

    assert report["passed"] is False
    assert not any(
        issue["code"] == "camera_geometry_frozen"
        and issue.get("camera_id") == "overview_rgb"
        for issue in issues
    )
    assert any(
        issue["code"] == "target_not_visible_in_policy_cameras"
        for issue in issues
    )


def test_gate_fails_closed_when_overview_role_is_changed(
    tmp_path: Path,
) -> None:
    episode, images = _make_episode(
        tmp_path,
        moving_policy=True,
        moving_overview=True,
        wrong_overview_role=True,
    )

    with pytest.raises(CameraGateError, match="observer_only"):
        audit_camera_episode(
            episode,
            image_loader=images.__getitem__,
        )
