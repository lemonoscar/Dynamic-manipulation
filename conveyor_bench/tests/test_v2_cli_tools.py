from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from conveyor_bench.v1.protocol import to_jsonable
from conveyor_bench.v1.tasking import CurriculumSplit, TaskFamily
from conveyor_bench.v2.camera_contracts import camera_contract_for_scene
from conveyor_bench.v2.config import SceneId
from conveyor_bench.v2.tasking import build_task_context
from test_v1_cli import canonical_digests, dump_jsonl, dump_png, make_episode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_v2_episode(path: Path) -> Path:
    episode = make_episode(path)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    context = build_task_context(
        seed=0,
        scene_id=SceneId.TRANSVERSE_NEAR_SORT_V2,
        family=TaskFamily.SINGLE_TARGET,
        mode="fixed_base",
        split=CurriculumSplit.TRAIN,
    )
    task = to_jsonable(context.task)
    target = context.target_sequence_ids[0]
    destination = context.destination_zone_by_target[target]
    zone = context.task.goal_zone_by_id[destination]
    center = [
        (lower + upper) * 0.5
        for lower, upper in zip(zone.min_xyz, zone.max_xyz, strict=True)
    ]
    suite = task["metadata"]["benchmark_suite"]
    manifest["episode"]["task"] = task
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(suite)
    camera_contract = camera_contract_for_scene(
        SceneId.TRANSVERSE_NEAR_SORT_V2
    )
    manifest["episode"]["metadata"]["cameras"] = camera_contract
    _write_json(manifest_path, manifest)

    steps_path = episode / "steps.jsonl"
    steps = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
    ]
    for step in steps:
        step["selected_object_id"] = target
        step["belt_measured_speed_mps"] = task["belt_speed_mps"]
        if step["camera_frames"]:
            tick = step["model_tick"]
            step["camera_frames"].append(
                {
                    "camera_id": "overview_rgb",
                    "frame_index": tick,
                    "capture_time_s": step["sim_time_s"],
                    "relative_path": (
                        f"cameras/overview_rgb/{tick:06d}.png"
                    ),
                }
            )
        for field in (
            "left_contact_object_ids",
            "right_contact_object_ids",
        ):
            step[field] = [
                target if object_id == "target" else object_id
                for object_id in step[field]
            ]
    steps_path.write_text(
        "".join(json.dumps(step) + "\n" for step in steps),
        encoding="utf-8",
    )

    objects_path = episode / "objects.jsonl"
    object_rows = [
        json.loads(line)
        for line in objects_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in object_rows:
        row["state"]["instance_id"] = target
        row["state"]["pose_world"]["xyz"] = center
        for future in row["future_object_states"]:
            future["instance_id"] = target
            if future["pose_world"] is not None:
                future["pose_world"]["xyz"] = center
    objects_path.write_text(
        "".join(json.dumps(row) + "\n" for row in object_rows),
        encoding="utf-8",
    )

    events_path = episode / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    for event in events:
        if event.get("object_instance_id") == "target":
            event["object_instance_id"] = target
        if event.get("goal_zone_id") == "zone-a":
            event["goal_zone_id"] = destination
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    summary_path = episode / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcome = summary["metrics"]["object_outcomes"].pop("target")
    outcome["goal_zone_id"] = destination
    summary["metrics"]["object_outcomes"][target] = outcome
    summary["task_id"] = task["task_id"]
    _write_json(summary_path, summary)

    camera_index_path = episode / "camera_frames.jsonl"
    camera_index = [
        json.loads(line)
        for line in camera_index_path.read_text(encoding="utf-8").splitlines()
    ]
    for capture in camera_index:
        tick = capture["frame_index"]
        capture["frames"]["overview_rgb"] = {
            "relative_path": f"cameras/overview_rgb/{tick:06d}.png",
            "quality": {
                "dark_fraction": 0.0,
                "laplacian_variance": 100.0,
            },
            "resolution": camera_contract["overview_rgb"]["resolution"],
            "role": camera_contract["overview_rgb"]["role"],
        }
        for camera_id, entry in capture["frames"].items():
            entry["resolution"] = camera_contract[camera_id]["resolution"]
            entry["role"] = camera_contract[camera_id]["role"]
    dump_jsonl(camera_index_path, camera_index)
    for capture in camera_index:
        tick = capture["frame_index"]
        dump_png(
            episode / "cameras" / "head_rgb" / f"{tick:06d}.png",
            224,
            224,
        )
        dump_png(
            episode / "cameras" / "wrist_rgb" / f"{tick:06d}.png",
            224,
            224,
        )
        dump_png(
            episode / "cameras" / "overview_rgb" / f"{tick:06d}.png",
            480,
            320,
        )
    return episode


def _run_script(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / name), *map(str, args)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_collection_summary(collection: Path, episode: Path) -> None:
    manifest = json.loads(
        (episode / "manifest.json").read_text(encoding="utf-8")
    )["episode"]
    summary = json.loads(
        (episode / "summary.json").read_text(encoding="utf-8")
    )
    camera_captures = sum(
        bool(json.loads(line)["camera_frames"])
        for line in (episode / "steps.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    report = {
        "episode_id": episode.name,
        "path": str(episode),
        "success": summary["success"],
        "failure_reason": summary["failure_reason"],
        "metrics": summary["metrics"],
        "camera_frames": camera_captures,
        "wall_time_s": 1.0,
    }
    _write_json(
        collection / f"{manifest['run_id']}-summary.json",
        {
            "run_id": manifest["run_id"],
            "protocol_version": manifest["protocol_version"],
            "task_type": manifest["task"]["task_type"],
            "robot_mode": manifest["task"]["robot_mode"],
            "requested_episodes": 1,
            "successful_episodes": int(summary["success"]),
            "episodes": [report],
        },
    )


def test_validate_v2_cli_discovers_episode_and_collection_root(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    episode = _make_v2_episode(collection / "episodes" / "ep-v2-cli")
    _write_collection_summary(collection, episode)

    direct = _run_script("validate_v2_dataset.py", episode)
    assert direct.returncode == 0, direct.stderr
    direct_report = json.loads(direct.stdout)
    assert direct_report["ok"] is True
    assert direct_report["episode_count"] == 1
    assert direct_report["episodes"][0]["episode_directory"] == str(episode)

    discovered = _run_script("validate_v2_dataset.py", collection)
    assert discovered.returncode == 0, discovered.stderr
    report = json.loads(discovered.stdout)
    assert report["ok"] is True
    assert report["valid_episode_count"] == 1
    assert report["invalid_episode_count"] == 0


def test_validate_v2_cli_distinguishes_validation_and_input_failures(
    tmp_path: Path,
) -> None:
    episode = _make_v2_episode(tmp_path / "ep-v2-invalid")
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_suite = manifest["episode"]["task"]["metadata"]["benchmark_suite"]
    task_suite["scene_id"] = "mobile_remote_delivery_v2"
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(task_suite)
    _write_json(manifest_path, manifest)

    invalid = _run_script("validate_v2_dataset.py", episode)
    assert invalid.returncode == 1
    invalid_report = json.loads(invalid.stdout)
    assert invalid_report["ok"] is False
    assert invalid_report["invalid_episode_count"] == 1
    assert any(
        "unsupported scene/task/mode" in error
        for error in invalid_report["episodes"][0]["errors"]
    )

    missing = _run_script("validate_v2_dataset.py", tmp_path / "missing")
    assert missing.returncode == 2
    missing_report = json.loads(missing.stdout)
    assert missing_report["ok"] is False
    assert "does not exist" in missing_report["error"]


@pytest.mark.parametrize(
    ("profile", "expected_profiles"),
    (
        ("dynamicvla", {"dynamicvla"}),
        ("m0", {"m0"}),
        ("both", {"dynamicvla", "m0"}),
    ),
)
def test_export_v2_cli_profiles_are_atomic_and_preserve_canonical_sources(
    tmp_path: Path,
    profile: str,
    expected_profiles: set[str],
) -> None:
    episode = _make_v2_episode(tmp_path / f"ep-v2-{profile}")
    before = canonical_digests(episode)

    completed = _run_script(
        "export_v2.py",
        episode,
        "--profile",
        profile,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    assert set(report["episodes"][0]["profiles"]) == expected_profiles

    exports = episode / "exports"
    manifest = json.loads(
        (exports / "export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == (
        "conveyor-bench-v2-export-manifest-1"
    )
    assert manifest["canonical_source_hashes"] == before
    assert manifest["canonical_files_modified"] is False
    assert manifest["source"]["benchmark_suite_version"] == (
        "conveyor-bench-v2"
    )
    assert set(manifest["profiles"]) == expected_profiles
    for selected_profile in expected_profiles:
        records = [
            json.loads(line)
            for line in (
                exports / f"{selected_profile}.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert records
        assert {record["schema_version"] for record in records} == {
            "conveyor-bench-v2-export-1"
        }
    assert canonical_digests(episode) == before


def test_export_v2_force_replaces_only_exports(tmp_path: Path) -> None:
    episode = _make_v2_episode(tmp_path / "ep-v2-force")
    before = canonical_digests(episode)
    sentinel = episode / "keep-me.txt"
    sentinel.write_text("outside exports\n", encoding="utf-8")

    first = _run_script("export_v2.py", episode, "--profile", "both")
    assert first.returncode == 0, first.stderr
    refused = _run_script("export_v2.py", episode, "--profile", "both")
    assert refused.returncode == 2
    assert "--force" in refused.stderr

    forced = _run_script(
        "export_v2.py",
        episode,
        "--profile",
        "both",
        "--force",
    )
    assert forced.returncode == 0, forced.stderr
    assert sentinel.read_text(encoding="utf-8") == "outside exports\n"
    assert canonical_digests(episode) == before
