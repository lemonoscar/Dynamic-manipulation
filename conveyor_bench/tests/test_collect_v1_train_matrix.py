from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "collect_v1_train_matrix.py"
SPEC = importlib.util.spec_from_file_location("collect_v1_train_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_episode(
    root: Path,
    cell,
    phase: str,
    seed: int,
    *,
    success: bool = True,
    gated: bool = True,
) -> Path:
    episode = root / phase / "cells" / cell.cell_id / "episodes" / f"ep-{seed}"
    episode.mkdir(parents=True)
    manifest = {
        "episode": {
            "episode_id": f"ep-{seed}",
            "metadata": {
                "source_tree": matrix.SOURCE_TREE_FINGERPRINT,
                "asset_lock_sha256": matrix.ASSET_LOCK_SHA256,
            },
            "seeds": {"episode": seed, "layout": seed},
            "task": {
                "task_type": "dynamic_sort",
                "robot_mode": "whole_body_policy",
                "belt_speed_mps": 0.06,
                "metadata": {
                    "target_asset_id": cell.target,
                    "destination_zone_id": cell.destination,
                    "instruction_language": cell.language,
                    "curriculum_split": "train",
                    "task_family": "language_conditioned",
                    "active_object_count": 3,
                },
            },
        }
    }
    (episode / "manifest.json").write_text(json.dumps(manifest))
    summary = {
        "success": success,
        "failure_reason": "none" if success else "target_missed",
    }
    (episode / "summary.json").write_text(json.dumps(summary))
    for name in matrix.REQUIRED_CANONICAL_FILES:
        path = episode / name
        if not path.exists():
            path.write_text("\n")
    if gated:
        (episode / "quality_report.json").write_text(
            json.dumps(
                {
                    "episode_id": f"ep-{seed}",
                    "task_outcome": "success" if success else "failure",
                    "data_status": "clean",
                }
            )
        )
        (episode / "camera_gate_report.json").write_text(
            json.dumps(
                {
                    "schema_version": "conveyor-bench-camera-gate-v1",
                    "episode_directory": str(episode.resolve()),
                    "passed": True,
                }
            )
        )
        exports = episode / "exports"
        exports.mkdir()
        profile_entries = {}
        for profile, schema in matrix.PROFILE_SCHEMAS.items():
            export_file = exports / f"{profile}.jsonl"
            export_file.write_text("{}\n")
            profile_entries[profile] = {
                "relative_path": export_file.name,
                "schema_version": schema,
                "sha256": _sha256(export_file),
                "record_count": 1,
                "source_task_outcome": "success" if success else "failure",
                "source_failure_reason": summary["failure_reason"],
            }
        (exports / "export_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "conveyor-bench-v1-export-manifest-1",
                    "source": {"episode_id": f"ep-{seed}"},
                    "canonical_files_modified": False,
                    "canonical_source_hashes": {
                        name: _sha256(episode / name)
                        for name in matrix.REQUIRED_CANONICAL_FILES
                    },
                    "profiles": profile_entries,
                    "source_task_outcome": "success" if success else "failure",
                    "source_failure_reason": summary["failure_reason"],
                }
            )
        )
    return episode


def test_matrix_is_full_factorial_and_seed_windows_cover_distractors() -> None:
    cells = matrix.cells()
    assert len(cells) == 16
    assert {
        (cell.target, cell.destination, cell.language) for cell in cells
    } == {
        (target, destination, language)
        for target in matrix.TARGETS
        for destination in matrix.DESTINATIONS
        for language in matrix.LANGUAGES
    }
    all_seeds = [seed for cell in cells for seed in cell.seeds("pilot") + cell.seeds("bulk")]
    assert len(all_seeds) == len(set(all_seeds)) == 128
    for cell in cells:
        candidates = [target for target in matrix.TARGETS if target != cell.target]
        permutations = {
            tuple(random.Random(seed).sample(candidates, 2))
            for seed in range(cell.base_seed, cell.base_seed + 8)
        }
        assert len(permutations) == 6


@pytest.mark.parametrize(("phase", "episodes"), (("pilot", "1"), ("bulk", "7")))
def test_dry_run_freezes_collection_contract(tmp_path: Path, phase: str, episodes: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            phase,
            "--output-root",
            str(tmp_path),
            "--renderer-active-gpu",
            "1",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["source_tree"] == matrix.SOURCE_TREE_FINGERPRINT
    assert payload["asset_lock_sha256"] == matrix.ASSET_LOCK_SHA256
    commands = payload["commands"]
    assert len(commands) == 16
    for command in commands:
        assert command[command.index("--episodes") + 1] == episodes
        assert command[command.index("--robot-mode") + 1] == "whole_body_policy"
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--active-objects") + 1] == "3"
        assert command[command.index("--belt-speed") + 1] == "0.06"
        assert "--enable_cameras" in command
        assert "--save-camera-frames" in command
        kit_args = command[command.index("--kit_args") + 1]
        assert "--/renderer/activeGpu=1" in kit_args
        assert f"--/log/file={tmp_path}/{phase}/logs/" in kit_args
        assert ("--require-all-success" in command) == (phase == "pilot")


def test_scan_recovers_exact_contract_and_rejects_orphans(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    episode = _publish_episode(tmp_path, cell, "pilot", cell.base_seed)
    observed = matrix.scan_phase(tmp_path, "pilot")
    assert observed[cell.base_seed].training_eligible

    (episode / "summary.json").unlink()
    with pytest.raises(matrix.MatrixError, match="orphan"):
        matrix.scan_phase(tmp_path, "pilot")


def test_scan_rejects_episode_from_another_source_tree(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    episode = _publish_episode(tmp_path, cell, "pilot", cell.base_seed)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episode"]["metadata"]["source_tree"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(matrix.MatrixError, match="source tree fingerprint"):
        matrix.scan_phase(tmp_path, "pilot")


def test_scan_rejects_episode_from_another_asset_lock(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    episode = _publish_episode(tmp_path, cell, "pilot", cell.base_seed)
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["episode"]["metadata"]["asset_lock_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(matrix.MatrixError, match="asset lock fingerprint"):
        matrix.scan_phase(tmp_path, "pilot")


def test_scan_rejects_stale_inprogress_episode(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    stale = (
        tmp_path
        / "pilot"
        / "cells"
        / cell.cell_id
        / "episodes"
        / ".seed-10011.inprogress"
    )
    stale.mkdir(parents=True)

    with pytest.raises(matrix.MatrixError, match="stale unpublished"):
        matrix.scan_phase(tmp_path, "pilot")


def test_scan_rejects_duplicate_or_conflicting_semantic_seed(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    first = _publish_episode(tmp_path, cell, "pilot", cell.base_seed)
    duplicate = first.with_name("duplicate")
    duplicate.mkdir()
    for source in first.iterdir():
        if source.is_file():
            duplicate.joinpath(source.name).write_bytes(source.read_bytes())
        elif source.name == "exports":
            duplicate.joinpath("exports").mkdir()
            duplicate.joinpath("exports", "export_manifest.json").write_bytes(
                source.joinpath("export_manifest.json").read_bytes()
            )
    with pytest.raises(matrix.MatrixError, match="duplicate semantic seed"):
        matrix.scan_phase(tmp_path, "pilot")

    conflict_root = tmp_path / "conflict"
    conflicting = _publish_episode(
        conflict_root, cell, "pilot", cell.base_seed
    )
    manifest = json.loads((conflicting / "manifest.json").read_text())
    manifest["episode"]["task"]["metadata"]["destination_zone_id"] = "sort_bin_yellow"
    (conflicting / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(matrix.MatrixError, match="frozen matrix contract"):
        matrix.scan_phase(conflict_root, "pilot")


def test_bulk_barrier_requires_all_successful_gated_pilots(tmp_path: Path) -> None:
    cells = matrix.cells()
    for cell in cells:
        _publish_episode(tmp_path, cell, "pilot", cell.base_seed)
    matrix.assert_pilot_ready(tmp_path)

    failed = cells[-1]
    summary_path = (
        tmp_path
        / "pilot"
        / "cells"
        / failed.cell_id
        / "episodes"
        / f"ep-{failed.base_seed}"
        / "summary.json"
    )
    summary_path.write_text(
        json.dumps({"success": False, "failure_reason": "target_missed"})
    )
    with pytest.raises(matrix.MatrixError, match="physically successful"):
        matrix.assert_pilot_ready(tmp_path)


def test_bulk_task_failure_is_retained_but_not_training_eligible(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    seed = cell.base_seed + 1
    _publish_episode(tmp_path, cell, "bulk", seed, success=False)
    observation = matrix.scan_phase(tmp_path, "bulk")[seed]
    assert observation.failure_reason == "target_missed"
    assert observation.gated
    assert not observation.training_eligible


def test_gate_rejects_unknown_quality_and_tampered_export(tmp_path: Path) -> None:
    cell = matrix.cells()[0]
    seed = cell.base_seed
    episode = _publish_episode(tmp_path, cell, "pilot", seed)
    (episode / "quality_report.json").write_text(
        json.dumps({"data_status": "valid"})
    )
    assert not matrix.scan_phase(tmp_path, "pilot")[seed].gated

    tampered_root = tmp_path / "tampered"
    episode = _publish_episode(tampered_root, cell, "pilot", seed)
    (episode / "exports" / "m0_mobile.jsonl").write_text("tampered\n")
    assert not matrix.scan_phase(tampered_root, "pilot")[seed].gated

    canonical_root = tmp_path / "canonical-tamper"
    episode = _publish_episode(canonical_root, cell, "pilot", seed)
    (episode / "steps.jsonl").write_text("tampered\n")
    assert not matrix.scan_phase(canonical_root, "pilot")[seed].gated


def test_export_gate_is_force_recoverable() -> None:
    commands = []

    def fake_run(value, _log):
        commands.append(value)
        return 0

    original = matrix._run
    matrix._run = fake_run
    try:
        matrix._gate_episode(
            Path(sys.executable), Path("cell"), Path("episode"), Path("log")
        )
    finally:
        matrix._run = original
    export = next(item for item in commands if "export_v1.py" in item[2])
    assert "--force" in export


def test_matrix_lock_is_cross_phase(tmp_path: Path) -> None:
    (tmp_path / ".matrix.lock").write_text("pid=1\n")
    with pytest.raises(matrix.MatrixError, match="already locked"):
        matrix.run_phase(tmp_path, "pilot", Path(sys.executable), 1)


@pytest.mark.parametrize(("phase", "expected"), (("pilot", 16), ("bulk", 112)))
def test_two_worker_waves_collect_exactly_once_without_active_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected: int,
) -> None:
    if phase == "bulk":
        for cell in matrix.cells():
            _publish_episode(tmp_path, cell, "pilot", cell.base_seed)

    real_scan = matrix.scan_phase
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}
    commands: list[tuple[str, ...]] = []
    worker_tmpdirs: set[str] = set()
    cells_by_id = {cell.cell_id: cell for cell in matrix.cells()}

    monkeypatch.setenv("TMPDIR", str(tmp_path / "runtime"))

    def fake_run(command, _log, *, env=None):
        output_dir = Path(command[command.index("--output-dir") + 1])
        cell = cells_by_id[output_dir.name]
        seed = int(command[command.index("--seed") + 1])
        count = int(command[command.index("--episodes") + 1])
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            commands.append(tuple(command))
            worker_tmpdirs.add(env["TMPDIR"])
        time.sleep(0.01)
        for episode_seed in range(seed, seed + count):
            _publish_episode(tmp_path, cell, phase, episode_seed)
        with state_lock:
            state["active"] -= 1
        return 0

    def guarded_scan(root, phase):
        with state_lock:
            assert state["active"] == 0
        return real_scan(root, phase)

    monkeypatch.setattr(matrix, "_run", fake_run)
    monkeypatch.setattr(matrix, "_gate_episode", lambda *_args: None)
    monkeypatch.setattr(matrix, "scan_phase", guarded_scan)

    matrix.run_phase(tmp_path, phase, Path(sys.executable), (2, 3), workers=2)

    observed = real_scan(tmp_path, phase)
    assert len(observed) == expected
    assert len(commands) == len(worker_tmpdirs) == 16
    assert state["max_active"] == 2
    assert {
        command[command.index("--kit_args") + 1].split()[0]
        for command in commands
    } == {"--/renderer/activeGpu=2", "--/renderer/activeGpu=3"}
    assert not (tmp_path / ".matrix.lock").exists()


def test_worker_and_renderer_gpu_arguments_are_validated(tmp_path: Path) -> None:
    with pytest.raises(matrix.MatrixError, match="positive"):
        matrix.run_phase(tmp_path, "bulk", Path(sys.executable), 1, workers=0)
    with pytest.raises(matrix.MatrixError, match="GPU indices"):
        matrix.run_phase(tmp_path, "pilot", Path(sys.executable), (), workers=2)
    with pytest.raises(matrix.MatrixError, match="GPU indices"):
        matrix.run_phase(tmp_path, "pilot", Path(sys.executable), (2, -1), workers=2)


def test_dry_run_distributes_repeated_renderer_gpus(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "pilot",
            "--output-root",
            str(tmp_path),
            "--renderer-active-gpu",
            "2",
            "--renderer-active-gpu",
            "3",
            "--workers",
            "2",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    commands = json.loads(completed.stdout)["commands"]
    assert [
        command[command.index("--kit_args") + 1].split()[0]
        for command in commands[:4]
    ] == [
        "--/renderer/activeGpu=2",
        "--/renderer/activeGpu=3",
        "--/renderer/activeGpu=2",
        "--/renderer/activeGpu=3",
    ]
