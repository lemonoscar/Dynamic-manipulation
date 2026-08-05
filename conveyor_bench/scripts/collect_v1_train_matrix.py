#!/usr/bin/env python3
"""Collect the frozen 128-episode V1 train matrix with resumable gates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v1.assets import (  # noqa: E402
    ASSET_LOCK_PATH,
    sha256_file,
    source_tree_fingerprint,
)


SCHEMA_VERSION = "conveyor-bench-v1-train-matrix-1"
SOURCE_TREE_FINGERPRINT = source_tree_fingerprint()
ASSET_LOCK_SHA256 = sha256_file(ASSET_LOCK_PATH)
BELT_SPEED_MPS = 0.06
MAX_DURATION_S = 35.0
ACTIVE_OBJECTS = 3
TARGETS = (
    "part_red_block",
    "part_blue_bar",
    "part_yellow_bushing",
    "part_green_shaft",
)
DESTINATIONS = ("sort_bin_blue", "sort_bin_yellow")
LANGUAGES = ("en", "en_zh")
BASE_SEEDS = (
    10011,
    10122,
    10202,
    10326,
    10426,
    10587,
    10643,
    10715,
    10851,
    10935,
    11024,
    11119,
    11205,
    11325,
    11408,
    11553,
)
REQUIRED_CANONICAL_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
)
PROFILE_SCHEMAS = {
    "dynamicvla": "conveyor-bench-v1-export-1",
    "m0": "conveyor-bench-v1-export-1",
    "m0_mobile": "conveyor-bench-m0-mobile-v1",
}


class MatrixError(ValueError):
    """Raised when collection state is ambiguous or violates the matrix."""


@dataclass(frozen=True)
class Cell:
    cell_id: str
    target: str
    destination: str
    language: str
    base_seed: int

    def seeds(self, phase: str) -> tuple[int, ...]:
        if phase == "pilot":
            return (self.base_seed,)
        if phase == "bulk":
            return tuple(range(self.base_seed + 1, self.base_seed + 8))
        raise MatrixError(f"unknown phase: {phase}")


@dataclass(frozen=True)
class EpisodeObservation:
    path: Path
    cell: Cell
    seed: int
    success: bool
    failure_reason: str
    gated: bool

    @property
    def training_eligible(self) -> bool:
        return self.success and self.gated


def cells() -> tuple[Cell, ...]:
    combinations = (
        (target, destination, language)
        for target in TARGETS
        for destination in DESTINATIONS
        for language in LANGUAGES
    )
    result = tuple(
        Cell(
            cell_id=(
                f"{target.removeprefix('part_')}-{destination.removeprefix('sort_bin_')}-{language}"
            ),
            target=target,
            destination=destination,
            language=language,
            base_seed=seed,
        )
        for (target, destination, language), seed in zip(
            combinations, BASE_SEEDS, strict=True
        )
    )
    if len(result) != 16 or len({cell.cell_id for cell in result}) != 16:
        raise MatrixError("the frozen train matrix must contain 16 unique cells")
    return result


def build_collection_command(
    cell: Cell,
    phase: str,
    seed: int,
    episodes: int,
    output_root: Path,
    python: Path,
    renderer_active_gpu: int,
) -> list[str]:
    if seed not in cell.seeds(phase) or episodes <= 0:
        raise MatrixError("collection range is outside its frozen cell")
    if tuple(range(seed, seed + episodes)) != tuple(
        item for item in cell.seeds(phase) if seed <= item < seed + episodes
    ):
        raise MatrixError("collection range must be contiguous and phase-local")
    kit_log = output_root / phase / "logs" / f"{cell.cell_id}.kit.log"
    kit_args = " ".join(
        (
            f"--/renderer/activeGpu={renderer_active_gpu}",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/maxGpuCount=1",
            f"--/log/file={kit_log}",
        )
    )
    command = [
        str(python),
        "-B",
        str(SCRIPTS / "run_benchmark_v1.py"),
        "--robot-mode",
        "whole_body_policy",
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--split",
        "train",
        "--task-family",
        "language_conditioned",
        "--instruction-language",
        cell.language,
        "--belt-speed",
        str(BELT_SPEED_MPS),
        "--max-duration",
        str(MAX_DURATION_S),
        "--active-objects",
        str(ACTIVE_OBJECTS),
        "--target-asset",
        cell.target,
        "--destination",
        cell.destination,
        "--output-dir",
        str(output_root / phase / "cells" / cell.cell_id),
        "--enable_cameras",
        "--save-camera-frames",
        "--headless",
        "--device",
        "cpu",
        "--kit_args",
        kit_args,
    ]
    if phase == "pilot":
        command.append("--require-all-success")
    return command


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatrixError(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise MatrixError(f"{path} must contain a JSON object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatrixError(f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gates_passed(episode: Path, summary: Mapping[str, Any]) -> bool:
    quality_path = episode / "quality_report.json"
    camera_path = episode / "camera_gate_report.json"
    export_path = episode / "exports" / "export_manifest.json"
    if not all(path.is_file() for path in (quality_path, camera_path, export_path)):
        return False
    quality = _read_json(quality_path)
    camera = _read_json(camera_path)
    export = _read_json(export_path)
    manifest = _read_json(episode / "manifest.json")
    episode_value = _mapping(manifest.get("episode"), "manifest.episode")
    episode_id = episode_value.get("episode_id")
    profiles = export.get("profiles")
    expected_outcome = "success" if summary.get("success") is True else "failure"
    canonical_hashes = export.get("canonical_source_hashes")
    canonical_ok = isinstance(canonical_hashes, Mapping) and set(
        canonical_hashes
    ) == set(REQUIRED_CANONICAL_FILES)
    if canonical_ok:
        canonical_ok = all(
            canonical_hashes.get(name) == _sha256(episode / name)
            for name in REQUIRED_CANONICAL_FILES
        )
    profiles_ok = isinstance(profiles, Mapping) and set(profiles) == set(
        PROFILE_SCHEMAS
    )
    if profiles_ok:
        for profile, schema in PROFILE_SCHEMAS.items():
            entry = profiles.get(profile)
            expected_name = f"{profile}.jsonl"
            if not isinstance(entry, Mapping):
                profiles_ok = False
                break
            profile_path = episode / "exports" / expected_name
            record_count = entry.get("record_count")
            if (
                entry.get("relative_path") != expected_name
                or entry.get("schema_version") != schema
                or isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count <= 0
                or entry.get("source_task_outcome") != expected_outcome
                or entry.get("source_failure_reason")
                != summary.get("failure_reason")
                or not profile_path.is_file()
                or entry.get("sha256") != _sha256(profile_path)
            ):
                profiles_ok = False
                break
    source = export.get("source")
    return (
        quality.get("data_status") in {"clean", "warning"}
        and quality.get("episode_id") == episode_id
        and quality.get("task_outcome") == expected_outcome
        and camera.get("schema_version") == "conveyor-bench-camera-gate-v1"
        and camera.get("passed") is True
        and camera.get("episode_directory") == str(episode.resolve())
        and export.get("schema_version")
        == "conveyor-bench-v1-export-manifest-1"
        and export.get("canonical_files_modified") is False
        and isinstance(source, Mapping)
        and source.get("episode_id") == episode_id
        and export.get("source_task_outcome") == expected_outcome
        and export.get("source_failure_reason") == summary.get("failure_reason")
        and canonical_ok
        and profiles_ok
    )


def scan_phase(output_root: Path, phase: str) -> dict[int, EpisodeObservation]:
    expected_by_seed = {
        seed: cell for cell in cells() for seed in cell.seeds(phase)
    }
    observed: dict[int, EpisodeObservation] = {}
    cells_root = output_root / phase / "cells"
    if not cells_root.exists():
        return observed
    expected_ids = {cell.cell_id for cell in cells()}
    unexpected = sorted(
        child.name
        for child in cells_root.iterdir()
        if child.is_dir() and child.name not in expected_ids
    )
    if unexpected:
        raise MatrixError(f"unexpected matrix cell directories: {unexpected}")

    for cell in cells():
        episodes_root = cells_root / cell.cell_id / "episodes"
        if not episodes_root.exists():
            continue
        for episode in sorted(episodes_root.iterdir()):
            if not episode.is_dir():
                continue
            if episode.name.startswith(".") or episode.name.endswith(".inprogress"):
                raise MatrixError(f"stale unpublished episode: {episode}")
            missing = [
                name for name in REQUIRED_CANONICAL_FILES if not (episode / name).is_file()
            ]
            if missing:
                raise MatrixError(f"orphan episode {episode}: missing {missing}")
            manifest = _read_json(episode / "manifest.json")
            episode_value = _mapping(manifest.get("episode"), "manifest.episode")
            episode_metadata = _mapping(
                episode_value.get("metadata"), "manifest.episode.metadata"
            )
            if episode_metadata.get("source_tree") != SOURCE_TREE_FINGERPRINT:
                raise MatrixError(
                    f"{episode} source tree fingerprint does not match collector"
                )
            if episode_metadata.get("asset_lock_sha256") != ASSET_LOCK_SHA256:
                raise MatrixError(
                    f"{episode} asset lock fingerprint does not match collector"
                )
            task = _mapping(episode_value.get("task"), "manifest.episode.task")
            metadata = _mapping(task.get("metadata"), "manifest.episode.task.metadata")
            seeds = _mapping(episode_value.get("seeds"), "manifest.episode.seeds")
            seed = seeds.get("episode")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise MatrixError(f"{episode} has an invalid episode seed")
            if seed in observed:
                raise MatrixError(
                    f"duplicate semantic seed {seed}: {observed[seed].path} and {episode}"
                )
            expected_cell = expected_by_seed.get(seed)
            if expected_cell is None or expected_cell != cell:
                raise MatrixError(f"{episode} seed {seed} is outside cell {cell.cell_id}")
            exact = {
                "layout_seed": seeds.get("layout"),
                "task_type": task.get("task_type"),
                "robot_mode": task.get("robot_mode"),
                "target": metadata.get("target_asset_id"),
                "destination": metadata.get("destination_zone_id"),
                "language": metadata.get("instruction_language"),
                "split": metadata.get("curriculum_split"),
                "family": metadata.get("task_family"),
                "active_objects": metadata.get("active_object_count"),
            }
            expected = {
                "layout_seed": seed,
                "task_type": "dynamic_sort",
                "robot_mode": "whole_body_policy",
                "target": cell.target,
                "destination": cell.destination,
                "language": cell.language,
                "split": "train",
                "family": "language_conditioned",
                "active_objects": ACTIVE_OBJECTS,
            }
            speed = task.get("belt_speed_mps")
            if exact != expected or (
                isinstance(speed, bool)
                or not isinstance(speed, (int, float))
                or not math.isclose(float(speed), BELT_SPEED_MPS, abs_tol=1e-9)
            ):
                raise MatrixError(
                    f"{episode} does not match its frozen matrix contract"
                )
            summary = _read_json(episode / "summary.json")
            success = summary.get("success")
            reason = summary.get("failure_reason")
            if not isinstance(success, bool) or not isinstance(reason, str):
                raise MatrixError(f"{episode} has an invalid task outcome")
            if reason == "runtime_error":
                raise MatrixError(f"{episode} is an operational runtime failure")
            observed[seed] = EpisodeObservation(
                path=episode.resolve(),
                cell=cell,
                seed=seed,
                success=success,
                failure_reason=reason,
                gated=_gates_passed(episode, summary),
            )
    return observed


def assert_pilot_ready(output_root: Path) -> None:
    observed = scan_phase(output_root, "pilot")
    expected = {cell.base_seed for cell in cells()}
    if set(observed) != expected:
        raise MatrixError(
            f"bulk requires all 16 pilot seeds; missing={sorted(expected - set(observed))}"
        )
    rejected = sorted(
        seed for seed, item in observed.items() if not item.success or not item.gated
    )
    if rejected:
        raise MatrixError(
            f"bulk requires 16 physically successful, fully gated pilots: {rejected}"
        )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_report(output_root: Path) -> None:
    phases: dict[str, Any] = {}
    successful: list[str] = []
    for phase in ("pilot", "bulk"):
        observed = scan_phase(output_root, phase)
        phases[phase] = {
            "expected_episodes": 16 if phase == "pilot" else 112,
            "observed_episodes": len(observed),
            "successful_episodes": sum(item.success for item in observed.values()),
            "fully_gated_episodes": sum(item.gated for item in observed.values()),
            "training_eligible_episodes": sum(
                item.training_eligible for item in observed.values()
            ),
            "failed_task_seeds": sorted(
                seed for seed, item in observed.items() if not item.success
            ),
        }
        successful.extend(
            str(item.path)
            for item in observed.values()
            if item.training_eligible
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_tree": SOURCE_TREE_FINGERPRINT,
        "asset_lock_sha256": ASSET_LOCK_SHA256,
        "belt_speed_mps": BELT_SPEED_MPS,
        "active_objects": ACTIVE_OBJECTS,
        "cells": [cell.__dict__ for cell in cells()],
        "phases": phases,
    }
    _atomic_write(
        output_root / "matrix_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_root / "successful_episode_roots.txt",
        "".join(f"{path}\n" for path in sorted(successful)),
    )


def _run(
    command: Sequence[str],
    log_path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        )
        return completed.returncode


def _gate_episode(python: Path, cell_root: Path, episode: Path, log: Path) -> None:
    commands = (
        [str(python), "-B", str(SCRIPTS / "validate_v1_dataset.py"), str(cell_root)],
        [str(python), "-B", str(SCRIPTS / "audit_v1_episode.py"), str(episode)],
        [
            str(python),
            "-B",
            str(SCRIPTS / "check_v1_camera_gate.py"),
            str(episode),
            "--output",
            str(episode / "camera_gate_report.json"),
        ],
        [
            str(python),
            "-B",
            str(SCRIPTS / "export_v1.py"),
            str(episode),
            "--profile",
            "all",
            "--force",
        ],
    )
    for command in commands:
        if _run(command, log) != 0:
            raise MatrixError(f"episode gate failed: {' '.join(command)}")


def _contiguous_ranges(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(values)
    if not ordered:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append((start, previous - start + 1))
            start = value
        previous = value
    ranges.append((start, previous - start + 1))
    return tuple(ranges)


def _worker_environment(phase: str, cell: Cell) -> dict[str, str]:
    environment = os.environ.copy()
    suffix = Path("matrix-workers") / phase / cell.cell_id
    for name in ("TMPDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        base = environment.get(name)
        if base is None:
            continue
        worker_root = Path(base) / suffix
        worker_root.mkdir(parents=True, exist_ok=True)
        environment[name] = str(worker_root)
    return environment


def _run_cell_ranges(
    cell: Cell,
    ranges: tuple[tuple[int, int], ...],
    output_root: Path,
    python: Path,
    renderer_active_gpu: int,
    phase: str,
) -> int:
    log = output_root / phase / "logs" / f"{cell.cell_id}.log"
    environment = _worker_environment(phase, cell)
    for start, count in ranges:
        command = build_collection_command(
            cell,
            phase,
            start,
            count,
            output_root,
            python,
            renderer_active_gpu,
        )
        returncode = _run(command, log, env=environment)
        if returncode != 0:
            return returncode
    return 0


def _run_phase_parallel(
    output_root: Path,
    phase: str,
    python: Path,
    renderer_active_gpus: tuple[int, ...],
    workers: int,
    observed: Mapping[int, EpisodeObservation],
) -> None:
    pending = [
        (
            cell,
            _contiguous_ranges(set(cell.seeds(phase)) - set(observed)),
        )
        for cell in cells()
        if set(cell.seeds(phase)) - set(observed)
    ]
    for offset in range(0, len(pending), workers):
        wave = pending[offset : offset + workers]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = tuple(
                executor.submit(
                    _run_cell_ranges,
                    cell,
                    ranges,
                    output_root,
                    python,
                    renderer_active_gpus[index % len(renderer_active_gpus)],
                    phase,
                )
                for index, (cell, ranges) in enumerate(wave)
            )
            returncodes = tuple(future.result() for future in futures)

        # Global scans are intentionally delayed until every writer in the
        # wave exits; an active collector owns an unpublished .inprogress dir.
        refreshed = scan_phase(output_root, phase)
        errors: list[str] = []
        for (cell, ranges), returncode in zip(wave, returncodes, strict=True):
            for start, count in ranges:
                for seed in range(start, start + count):
                    item = refreshed.get(seed)
                    if item is None:
                        errors.append(
                            f"{cell.cell_id} did not publish seed {seed}"
                        )
                        continue
                    if item.gated:
                        continue
                    try:
                        _gate_episode(
                            python,
                            output_root / phase / "cells" / cell.cell_id,
                            item.path,
                            output_root / phase / "logs" / f"{cell.cell_id}.log",
                        )
                    except (MatrixError, OSError) as error:
                        errors.append(f"{cell.cell_id} seed {seed}: {error}")
            if returncode != 0:
                errors.append(
                    f"collector failed for {cell.cell_id} with code {returncode}"
                )
            if phase == "pilot" and any(
                refreshed[seed].success is False
                for start, count in ranges
                for seed in range(start, start + count)
                if seed in refreshed
            ):
                errors.append(f"pilot task failed for {cell.cell_id}")
        _write_report(output_root)
        if errors:
            raise MatrixError("; ".join(errors))


def run_phase(
    output_root: Path,
    phase: str,
    python: Path,
    renderer_active_gpu: int | Sequence[int],
    workers: int = 1,
) -> None:
    if workers <= 0:
        raise MatrixError("workers must be positive")
    renderer_active_gpus = (
        (renderer_active_gpu,)
        if isinstance(renderer_active_gpu, int)
        else tuple(renderer_active_gpu)
    )
    if not renderer_active_gpus or any(gpu < 0 for gpu in renderer_active_gpus):
        raise MatrixError("renderer GPU indices must be non-negative")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".matrix.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise MatrixError(f"matrix phase is already locked: {lock_path}") from error
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.close(lock_fd)
        if phase == "bulk":
            for item in scan_phase(output_root, "pilot").values():
                _gate_episode(
                    python,
                    output_root / "pilot" / "cells" / item.cell.cell_id,
                    item.path,
                    output_root / "pilot" / "logs" / f"{item.cell.cell_id}.log",
                )
            assert_pilot_ready(output_root)

        observed = scan_phase(output_root, phase)
        for item in tuple(observed.values()):
            if not item.gated:
                cell_root = output_root / phase / "cells" / item.cell.cell_id
                _gate_episode(
                    python,
                    cell_root,
                    item.path,
                    output_root / phase / "logs" / f"{item.cell.cell_id}.log",
                )
        _write_report(output_root)

        if workers > 1:
            _run_phase_parallel(
                output_root,
                phase,
                python,
                renderer_active_gpus,
                workers,
                observed,
            )
        else:
            for cell in cells():
                observed = scan_phase(output_root, phase)
                missing = set(cell.seeds(phase)) - set(observed)
                for start, count in _contiguous_ranges(missing):
                    command = build_collection_command(
                        cell,
                        phase,
                        start,
                        count,
                        output_root,
                        python,
                        renderer_active_gpus[0],
                    )
                    log = output_root / phase / "logs" / f"{cell.cell_id}.log"
                    returncode = _run(command, log)
                    refreshed = scan_phase(output_root, phase)
                    for seed in range(start, start + count):
                        item = refreshed.get(seed)
                        if item is None:
                            raise MatrixError(
                                f"collector returned without publishing seed {seed}"
                            )
                        if not item.gated:
                            _gate_episode(
                                python,
                                output_root / phase / "cells" / cell.cell_id,
                                item.path,
                                log,
                            )
                    _write_report(output_root)
                    if returncode != 0:
                        raise MatrixError(
                            f"collector failed for {cell.cell_id} with code {returncode}"
                        )
                    if phase == "pilot" and any(
                        not refreshed[seed].success
                        for seed in range(start, start + count)
                    ):
                        raise MatrixError(f"pilot task failed for {cell.cell_id}")
        if phase == "pilot":
            assert_pilot_ready(output_root)
        _write_report(output_root)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "bulk"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--renderer-active-gpu",
        type=int,
        action="append",
        required=True,
        help="Renderer GPU ordinal; repeat to distribute parallel workers.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if any(gpu < 0 for gpu in args.renderer_active_gpu):
        raise SystemExit("--renderer-active-gpu must be non-negative")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    output_root = args.output_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    if args.dry_run:
        commands = [
            build_collection_command(
                cell,
                args.phase,
                cell.seeds(args.phase)[0],
                len(cell.seeds(args.phase)),
                output_root,
                python,
                args.renderer_active_gpu[
                    index % len(args.renderer_active_gpu)
                    if args.workers > 1
                    else 0
                ],
            )
            for index, cell in enumerate(cells())
        ]
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "source_tree": SOURCE_TREE_FINGERPRINT,
                    "asset_lock_sha256": ASSET_LOCK_SHA256,
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0
    try:
        run_phase(
            output_root,
            args.phase,
            python,
            args.renderer_active_gpu,
            args.workers,
        )
    except (MatrixError, OSError) as error:
        print(f"collect_v1_train_matrix: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
