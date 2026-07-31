from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from test_v2_cli_tools import _make_v2_episode, _run_script


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_report(episode: Path) -> dict:
    summary = _read_json(episode / "summary.json")
    camera_frames = sum(
        bool(json.loads(line)["camera_frames"])
        for line in (episode / "steps.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    return {
        "episode_id": episode.name,
        "path": str(episode),
        "success": summary["success"],
        "failure_reason": summary["failure_reason"],
        "metrics": summary["metrics"],
        "camera_frames": camera_frames,
        "wall_time_s": 1.0,
    }


def write_collection_summary(
    collection: Path,
    episodes: tuple[Path, ...],
    *,
    extra_reports: tuple[dict, ...] = (),
) -> Path:
    first_manifest = _read_json(episodes[0] / "manifest.json")["episode"]
    reports = [_episode_report(episode) for episode in episodes]
    reports.extend(deepcopy(extra_reports))
    summary = {
        "run_id": first_manifest["run_id"],
        "protocol_version": first_manifest["protocol_version"],
        "task_type": first_manifest["task"]["task_type"],
        "robot_mode": first_manifest["task"]["robot_mode"],
        "requested_episodes": len(reports),
        "successful_episodes": sum(report["success"] for report in reports),
        "episodes": reports,
    }
    path = collection / f"{first_manifest['run_id']}-summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("damage", "message"),
    (
        ("missing_summary", "no run summary"),
        ("inprogress", ".inprogress"),
        ("extra_published", "absent from run summaries"),
        ("missing_published", "missing published episode"),
    ),
)
def test_collection_gate_reports_incomplete_roots_before_export(
    tmp_path: Path,
    damage: str,
    message: str,
) -> None:
    collection = tmp_path / "collection"
    episode = _make_v2_episode(collection / "episodes" / "ep-primary")

    if damage != "missing_summary":
        extra_reports: tuple[dict, ...] = ()
        if damage == "missing_published":
            missing = _episode_report(episode)
            missing["episode_id"] = "ep-missing"
            missing["path"] = str(collection / "episodes" / "ep-missing")
            extra_reports = (missing,)
        write_collection_summary(
            collection,
            (episode,),
            extra_reports=extra_reports,
        )
    if damage == "inprogress":
        (collection / "episodes" / ".ep-crashed.inprogress").mkdir()
    elif damage == "extra_published":
        _make_v2_episode(collection / "episodes" / "ep-extra")

    validated = _run_script("validate_v2_dataset.py", collection)
    assert validated.returncode == 1, validated.stderr
    validation_report = json.loads(validated.stdout)
    assert validation_report["ok"] is False
    assert validation_report["source_kind"] == "collection"
    assert validation_report["collection_error_count"] > 0
    assert any(
        message in error for error in validation_report["collection_errors"]
    )

    exported = _run_script("export_v2.py", collection, "--profile", "m0")
    assert exported.returncode == 2, exported.stderr
    export_report = json.loads(exported.stdout)
    assert export_report["ok"] is False
    assert export_report["collection_errors"]
    assert not (episode / "exports").exists()


def test_collection_export_preflights_every_episode_before_publication(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    first = _make_v2_episode(collection / "episodes" / "ep-first")
    second = _make_v2_episode(collection / "episodes" / "ep-second")
    write_collection_summary(collection, (first, second))

    manifest_path = second / "manifest.json"
    manifest = _read_json(manifest_path)
    suite = manifest["episode"]["task"]["metadata"]["benchmark_suite"]
    suite["scene_id"] = "mobile_remote_delivery_v2"
    manifest["episode"]["metadata"]["benchmark_suite"] = deepcopy(suite)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exported = _run_script("export_v2.py", collection, "--profile", "m0")

    assert exported.returncode == 2, exported.stderr
    report = json.loads(exported.stdout)
    assert report["ok"] is False
    assert "V2 episode validation failed" in report["error"]
    assert not (first / "exports").exists()
    assert not (second / "exports").exists()


def test_collection_export_preflights_later_destination_conflicts(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    first = _make_v2_episode(collection / "episodes" / "ep-first")
    second = _make_v2_episode(collection / "episodes" / "ep-second")
    write_collection_summary(collection, (first, second))
    conflict = second / "exports" / "m0.jsonl"
    conflict.parent.mkdir()
    conflict.write_text("existing\n", encoding="utf-8")

    exported = _run_script("export_v2.py", collection, "--profile", "m0")

    assert exported.returncode == 2, exported.stderr
    report = json.loads(exported.stdout)
    assert report["ok"] is False
    assert "--force" in report["error"]
    assert not (first / "exports").exists()
    assert conflict.read_text(encoding="utf-8") == "existing\n"


def test_collection_gate_accepts_collection_and_episodes_directory(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"
    episode = _make_v2_episode(collection / "episodes" / "ep-complete")
    write_collection_summary(collection, (episode,))

    for source in (collection, collection / "episodes"):
        completed = _run_script("validate_v2_dataset.py", source)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        assert report["ok"] is True
        assert report["run_summary_count"] == 1
        assert report["collection_error_count"] == 0
