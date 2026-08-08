#!/usr/bin/env python3
"""Collect the resumable ConveyorVLA AL0 low-speed grasp curriculum."""

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

from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    TEMPORAL_SCHEMA_VERSION,
)
from conveyor_bench.v1.assets import (  # noqa: E402
    ASSET_LOCK_PATH,
    sha256_file,
    source_tree_fingerprint,
)


SCHEMA_VERSION = "conveyor-vla-al0-grasp-collection-1"
SOURCE_TREE_FINGERPRINT = source_tree_fingerprint()
ASSET_LOCK_SHA256 = sha256_file(ASSET_LOCK_PATH)
BELT_SPEEDS_MPS = (0.01, 0.02)
TARGETS = (
    "part_red_block",
    "part_blue_bar",
    "part_yellow_bushing",
    "part_green_shaft",
)
DESTINATION = "sort_bin_blue"
INTERCEPT_LEAD_TIME_S = 5.0
MAX_DURATION_S = 40.0
PRODUCTION_SUCCESS_TARGET_PER_CELL = 48
PRODUCTION_MAX_ATTEMPTS_PER_CELL = 72
MAX_EPISODES_PER_PROCESS = 8
BASE_SEEDS = tuple(200_000 + index * 1_000 for index in range(8))
REQUIRED_CANONICAL_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
)
LEGACY_PROFILE_SCHEMAS = {
    "dynamicvla": "conveyor-bench-v1-export-1",
    "m0": "conveyor-bench-v1-export-1",
    "m0_mobile": "conveyor-bench-m0-mobile-v1",
}
TEMPORAL_EXPORT_PROFILE = "conveyorvla_al0_temporal"
TEACHER_PROFILE_ID = "overhead_slow_pick_place_v1"


class CollectionError(ValueError):
    """Raised when collection state is ambiguous or fails a promotion gate."""


@dataclass(frozen=True)
class Cell:
    cell_id: str
    target: str
    belt_speed_mps: float
    base_seed: int

    def seeds(self, phase: str) -> tuple[int, ...]:
        if phase == "pilot":
            return (self.base_seed,)
        if phase == "production":
            return tuple(
                range(
                    self.base_seed + 1,
                    self.base_seed + 1 + PRODUCTION_MAX_ATTEMPTS_PER_CELL,
                )
            )
        raise CollectionError(f"unknown phase: {phase}")


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
        (target, speed) for speed in BELT_SPEEDS_MPS for target in TARGETS
    )
    result = tuple(
        Cell(
            cell_id=(
                f"{target.removeprefix('part_')}-speed-{int(round(speed * 1000)):03d}"
            ),
            target=target,
            belt_speed_mps=speed,
            base_seed=seed,
        )
        for (target, speed), seed in zip(combinations, BASE_SEEDS, strict=True)
    )
    if len(result) != 8 or len({cell.cell_id for cell in result}) != 8:
        raise CollectionError("the AL0 grasp curriculum must contain eight cells")
    return result


def build_collection_command(
    cell: Cell,
    phase: str,
    seed: int,
    episodes: int,
    output_root: Path,
    python: Path,
    physical_gpu: int,
) -> list[str]:
    if physical_gpu not in {2, 3}:
        raise CollectionError("the renderer is restricted to physical GPU 2 or 3")
    allowed = cell.seeds(phase)
    requested = tuple(range(seed, seed + episodes))
    if episodes <= 0 or any(value not in allowed for value in requested):
        raise CollectionError("collection range is outside its frozen cell")
    kit_log = (
        output_root
        / phase
        / "logs"
        / f"{cell.cell_id}-seed-{seed}.kit.log"
    )
    kit_args = " ".join(
        (
            f"--/renderer/activeGpu={physical_gpu}",
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
        "single_target",
        "--instruction-language",
        "en",
        "--belt-speed",
        str(cell.belt_speed_mps),
        "--target-intercept-lead-time",
        str(INTERCEPT_LEAD_TIME_S),
        "--max-duration",
        str(MAX_DURATION_S),
        "--active-objects",
        "1",
        "--target-asset",
        cell.target,
        "--destination",
        DESTINATION,
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
        raise CollectionError(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CollectionError(f"{path} must contain a JSON object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CollectionError(f"{name} must be a JSON object")
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
    expected_outcome = "success" if summary.get("success") is True else "failure"
    profiles = export.get("profiles")
    required_profiles = dict(LEGACY_PROFILE_SCHEMAS)
    if expected_outcome == "success":
        required_profiles[TEMPORAL_EXPORT_PROFILE] = TEMPORAL_SCHEMA_VERSION
    profiles_ok = isinstance(profiles, Mapping) and set(required_profiles) <= set(profiles)
    if profiles_ok:
        for profile, schema in required_profiles.items():
            entry = profiles.get(profile)
            profile_path = episode / "exports" / f"{profile}.jsonl"
            if (
                not isinstance(entry, Mapping)
                or entry.get("schema_version") != schema
                or not profile_path.is_file()
                or entry.get("sha256") != _sha256(profile_path)
                or not isinstance(entry.get("record_count"), int)
                or entry.get("record_count") <= 0
                or entry.get("source_task_outcome") != expected_outcome
                or entry.get("source_failure_reason")
                != summary.get("failure_reason")
            ):
                profiles_ok = False
                break
    canonical_hashes = export.get("canonical_source_hashes")
    canonical_ok = isinstance(canonical_hashes, Mapping) and all(
        canonical_hashes.get(name) == _sha256(episode / name)
        for name in REQUIRED_CANONICAL_FILES
    )
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
    for cell in cells():
        episodes_root = output_root / phase / "cells" / cell.cell_id / "episodes"
        if not episodes_root.exists():
            continue
        for episode in sorted(episodes_root.iterdir()):
            if not episode.is_dir():
                continue
            if episode.name.startswith(".") or episode.name.endswith(".inprogress"):
                raise CollectionError(f"stale unpublished episode: {episode}")
            missing = [
                name for name in REQUIRED_CANONICAL_FILES if not (episode / name).is_file()
            ]
            if missing:
                raise CollectionError(f"orphan episode {episode}: missing {missing}")
            manifest = _read_json(episode / "manifest.json")
            value = _mapping(manifest.get("episode"), "manifest.episode")
            metadata = _mapping(value.get("metadata"), "manifest.episode.metadata")
            if metadata.get("source_tree") != SOURCE_TREE_FINGERPRINT:
                raise CollectionError(f"{episode} source tree fingerprint mismatch")
            if metadata.get("asset_lock_sha256") != ASSET_LOCK_SHA256:
                raise CollectionError(f"{episode} asset lock fingerprint mismatch")
            teacher = _mapping(
                metadata.get("demonstration_teacher"),
                "manifest.episode.metadata.demonstration_teacher",
            )
            if teacher.get("profile_id") != TEACHER_PROFILE_ID:
                raise CollectionError(
                    f"{episode} demonstration teacher profile mismatch"
                )
            seeds = _mapping(value.get("seeds"), "manifest.episode.seeds")
            seed = seeds.get("episode")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise CollectionError(f"{episode} has an invalid episode seed")
            if seed in observed:
                raise CollectionError(f"duplicate semantic seed {seed}")
            expected_cell = expected_by_seed.get(seed)
            if expected_cell != cell:
                raise CollectionError(f"{episode} seed is outside cell {cell.cell_id}")
            task = _mapping(value.get("task"), "manifest.episode.task")
            task_metadata = _mapping(task.get("metadata"), "task.metadata")
            speed = task.get("belt_speed_mps")
            exact = {
                "layout_seed": seeds.get("layout"),
                "task_type": task.get("task_type"),
                "robot_mode": task.get("robot_mode"),
                "family": task_metadata.get("task_family"),
                "language": task_metadata.get("instruction_language"),
                "active_objects": task_metadata.get("active_object_count"),
                "target": task_metadata.get("target_asset_id"),
                "destination": task_metadata.get("destination_zone_id"),
                "lead_time": task_metadata.get("target_intercept_lead_time_s"),
            }
            expected = {
                "layout_seed": seed,
                "task_type": "dynamic_sort",
                "robot_mode": "whole_body_policy",
                "family": "single_target",
                "language": "en",
                "active_objects": 1,
                "target": cell.target,
                "destination": DESTINATION,
                "lead_time": INTERCEPT_LEAD_TIME_S,
            }
            if exact != expected or (
                isinstance(speed, bool)
                or not isinstance(speed, (int, float))
                or not math.isclose(float(speed), cell.belt_speed_mps, abs_tol=1.0e-9)
            ):
                raise CollectionError(f"{episode} violates the grasp curriculum")
            summary = _read_json(episode / "summary.json")
            success = summary.get("success")
            reason = summary.get("failure_reason")
            if not isinstance(success, bool) or not isinstance(reason, str):
                raise CollectionError(f"{episode} has an invalid task outcome")
            if reason == "runtime_error":
                raise CollectionError(f"{episode} is an operational runtime failure")
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
    rejected = sorted(
        seed
        for seed in expected
        if seed not in observed
        or not observed[seed].success
        or not observed[seed].gated
    )
    if rejected:
        raise CollectionError(
            "production requires eight successful, fully gated pilots: "
            f"{rejected}"
        )


def assert_production_complete(output_root: Path) -> None:
    observed = scan_phase(output_root, "production")
    incomplete = {
        cell.cell_id: sum(
            item.training_eligible
            for item in observed.values()
            if item.cell == cell
        )
        for cell in cells()
    }
    incomplete = {
        cell_id: count
        for cell_id, count in incomplete.items()
        if count < PRODUCTION_SUCCESS_TARGET_PER_CELL
    }
    if incomplete:
        raise CollectionError(
            "production success quota is incomplete: "
            + ", ".join(
                f"{cell_id}={count}/{PRODUCTION_SUCCESS_TARGET_PER_CELL}"
                for cell_id, count in sorted(incomplete.items())
            )
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
    phases = {}
    successful = []
    for phase, target, maximum in (
        ("pilot", 8, 8),
        (
            "production",
            len(cells()) * PRODUCTION_SUCCESS_TARGET_PER_CELL,
            len(cells()) * PRODUCTION_MAX_ATTEMPTS_PER_CELL,
        ),
    ):
        observed = scan_phase(output_root, phase)
        eligible = sum(item.training_eligible for item in observed.values())
        phases[phase] = {
            "training_eligible_target": target,
            "maximum_attempts": maximum,
            "observed_episodes": len(observed),
            "successful_episodes": sum(item.success for item in observed.values()),
            "fully_gated_episodes": sum(item.gated for item in observed.values()),
            "training_eligible_episodes": eligible,
            "complete": eligible >= target,
            "failed_task_seeds": sorted(
                item.seed for item in observed.values() if not item.success
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
        "task_scope": "single_object_grasp",
        "demonstration_teacher_profile_id": TEACHER_PROFILE_ID,
        "belt_speeds_mps": BELT_SPEEDS_MPS,
        "target_intercept_lead_time_s": INTERCEPT_LEAD_TIME_S,
        "cells": [cell.__dict__ for cell in cells()],
        "phases": phases,
    }
    _atomic_write(
        output_root / "collection_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_root / "successful_episode_roots.txt",
        "".join(f"{path}\n" for path in sorted(successful)),
    )


def _run(command: Sequence[str], log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=env,
        ).returncode


def _gate_episode(python: Path, cell_root: Path, episode: Path, log: Path) -> None:
    summary = _read_json(episode / "summary.json")
    commands = [
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
    ]
    if summary.get("success") is True:
        commands.append(
            [
                str(python),
                "-B",
                str(SCRIPTS / "export_v1.py"),
                str(episode),
                "--profile",
                TEMPORAL_EXPORT_PROFILE,
                "--force",
            ]
        )
    environment = os.environ.copy()
    for command in commands:
        if _run(command, log, environment) != 0:
            raise CollectionError(f"episode gate failed: {' '.join(command)}")


def _contiguous_batches(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(values)
    batches = []
    while ordered:
        start = ordered.pop(0)
        count = 1
        while (
            ordered
            and ordered[0] == start + count
            and count < MAX_EPISODES_PER_PROCESS
        ):
            ordered.pop(0)
            count += 1
        batches.append((start, count))
    return tuple(batches)


def _worker_environment(
    output_root: Path,
    phase: str,
    cell: Cell,
    seed: int,
    physical_gpu: int,
    isaaclab_source: Path | None,
    kit_cache_root: Path | None,
    runtime_library_dir: Path | None,
) -> dict[str, str]:
    root = output_root / "runtime" / phase / cell.cell_id / f"seed-{seed}"
    paths = {
        "TMPDIR": root / "tmp",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_DATA_HOME": root / "data",
        "XDG_STATE_HOME": root / "state",
        "MPLCONFIGDIR": root / "mpl",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    cache_root = kit_cache_root if kit_cache_root is not None else root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in paths.items()})
    environment["XDG_CACHE_HOME"] = str(cache_root)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if isaaclab_source is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(isaaclab_source) + (
            os.pathsep + existing if existing else ""
        )
    if runtime_library_dir is not None:
        existing = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = str(runtime_library_dir) + (
            os.pathsep + existing if existing else ""
        )
    return environment


def _episodes_by_seed(cell_root: Path) -> dict[int, Path]:
    result = {}
    episodes_root = cell_root / "episodes"
    if not episodes_root.is_dir():
        return result
    for episode in episodes_root.iterdir():
        if not episode.is_dir() or episode.name.startswith("."):
            continue
        manifest = _read_json(episode / "manifest.json")
        value = _mapping(manifest.get("episode"), "manifest.episode")
        seeds = _mapping(value.get("seeds"), "manifest.episode.seeds")
        seed = seeds.get("episode")
        if isinstance(seed, int) and not isinstance(seed, bool):
            if seed in result:
                raise CollectionError(f"duplicate published seed {seed}")
            result[seed] = episode
    return result


def _run_cell(
    cell: Cell,
    phase: str,
    missing: set[int],
    output_root: Path,
    python: Path,
    physical_gpu: int,
    existing_training_eligible: int,
    isaaclab_source: Path | None,
    kit_cache_root: Path | None,
    runtime_library_dir: Path | None,
) -> None:
    cell_root = output_root / phase / "cells" / cell.cell_id
    log = output_root / phase / "logs" / f"{cell.cell_id}.log"
    target = 1 if phase == "pilot" else PRODUCTION_SUCCESS_TARGET_PER_CELL
    training_eligible = existing_training_eligible
    remaining = set(missing)
    while training_eligible < target and remaining:
        seed, available = _contiguous_batches(remaining)[0]
        count = min(available, target - training_eligible)
        remaining.difference_update(range(seed, seed + count))
        command = build_collection_command(
            cell, phase, seed, count, output_root, python, physical_gpu
        )
        environment = _worker_environment(
            output_root,
            phase,
            cell,
            seed,
            physical_gpu,
            isaaclab_source,
            kit_cache_root,
            runtime_library_dir,
        )
        returncode = _run(command, log, environment)
        published = _episodes_by_seed(cell_root)
        for expected_seed in range(seed, seed + count):
            episode = published.get(expected_seed)
            if episode is None:
                raise CollectionError(
                    f"{cell.cell_id} did not publish seed {expected_seed}"
                )
            _gate_episode(python, cell_root, episode, log)
            summary = _read_json(episode / "summary.json")
            if phase == "pilot" and summary.get("success") is not True:
                raise CollectionError(f"pilot task failed for {cell.cell_id}")
            if summary.get("success") is True and _gates_passed(episode, summary):
                training_eligible += 1
        if returncode != 0:
            raise CollectionError(
                f"collector failed for {cell.cell_id} with code {returncode}"
            )
    if training_eligible < target:
        raise CollectionError(
            f"{cell.cell_id} exhausted its seed reserve with "
            f"{training_eligible}/{target} successful gated episodes"
        )


def run_phase(
    output_root: Path,
    phase: str,
    python: Path,
    physical_gpus: Sequence[int],
    workers: int,
    isaaclab_source: Path | None = None,
    kit_cache_root: Path | None = None,
    runtime_library_dir: Path | None = None,
) -> None:
    gpus = tuple(physical_gpus)
    _validate_gpu_assignment(gpus, workers)
    isaaclab_source = _resolve_isaaclab_source(isaaclab_source)
    kit_cache_root = _resolve_existing_directory(
        kit_cache_root, "--kit-cache-root"
    )
    runtime_library_dir = _resolve_existing_directory(
        runtime_library_dir, "--runtime-library-dir"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".collection.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise CollectionError(f"collection is already locked: {lock_path}") from error
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.close(lock_fd)
        if phase == "production":
            assert_pilot_ready(output_root)
        observed = scan_phase(output_root, phase)
        target = 1 if phase == "pilot" else PRODUCTION_SUCCESS_TARGET_PER_CELL
        pending = []
        for cell in cells():
            cell_observations = tuple(
                item for item in observed.values() if item.cell == cell
            )
            eligible = sum(item.training_eligible for item in cell_observations)
            if eligible >= target:
                continue
            used = {item.seed for item in cell_observations}
            pending.append((cell, set(cell.seeds(phase)) - used, eligible))
        for offset in range(0, len(pending), workers):
            wave = pending[offset : offset + workers]
            with ThreadPoolExecutor(max_workers=len(wave)) as executor:
                futures = [
                    executor.submit(
                        _run_cell,
                        cell,
                        phase,
                        missing,
                        output_root,
                        python,
                        gpus[(offset + index) % len(gpus)],
                        eligible,
                        isaaclab_source,
                        kit_cache_root,
                        runtime_library_dir,
                    )
                    for index, (cell, missing, eligible) in enumerate(wave)
                ]
                for future in futures:
                    future.result()
            _write_report(output_root)
        if phase == "pilot":
            assert_pilot_ready(output_root)
        else:
            assert_production_complete(output_root)
        _write_report(output_root)
    finally:
        if lock_path.exists():
            lock_path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "production"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--isaaclab-source",
        type=Path,
        help=(
            "Directory containing isaaclab/__init__.py; prepend it to "
            "worker PYTHONPATH when Isaac Lab is not installed in --python."
        ),
    )
    parser.add_argument(
        "--kit-cache-root",
        type=Path,
        help="Existing prewarmed XDG cache used by the Isaac Sim workers.",
    )
    parser.add_argument(
        "--runtime-library-dir",
        type=Path,
        help="Existing runtime library directory prepended to LD_LIBRARY_PATH.",
    )
    parser.add_argument(
        "--physical-gpu",
        type=int,
        action="append",
        required=True,
        help="Physical GPU index; only 2 and 3 are accepted.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_gpu_assignment(gpus: Sequence[int], workers: int) -> None:
    if workers <= 0:
        raise CollectionError("workers must be positive")
    if not gpus or any(gpu not in {2, 3} for gpu in gpus):
        raise CollectionError("this collection permits only physical GPUs 2 and 3")
    if len(set(gpus)) != len(gpus):
        raise CollectionError("physical GPU indices must be unique")
    if workers > len(gpus):
        raise CollectionError("workers cannot exceed the number of physical GPUs")


def _resolve_isaaclab_source(path: Path | None) -> Path | None:
    if path is None:
        return None
    source = path.expanduser().resolve()
    if not (source / "isaaclab" / "__init__.py").is_file():
        raise CollectionError(
            "--isaaclab-source must contain isaaclab/__init__.py"
        )
    return source


def _resolve_existing_directory(path: Path | None, option: str) -> Path | None:
    if path is None:
        return None
    directory = path.expanduser().resolve()
    if not directory.is_dir():
        raise CollectionError(f"{option} must be an existing directory")
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    try:
        _validate_gpu_assignment(args.physical_gpu, args.workers)
        isaaclab_source = _resolve_isaaclab_source(args.isaaclab_source)
        kit_cache_root = _resolve_existing_directory(
            args.kit_cache_root, "--kit-cache-root"
        )
        runtime_library_dir = _resolve_existing_directory(
            args.runtime_library_dir, "--runtime-library-dir"
        )
    except CollectionError as error:
        raise SystemExit(str(error)) from error
    if args.dry_run:
        commands = [
            build_collection_command(
                cell,
                args.phase,
                cell.seeds(args.phase)[0],
                min(MAX_EPISODES_PER_PROCESS, len(cell.seeds(args.phase))),
                output_root,
                python,
                args.physical_gpu[index % len(args.physical_gpu)],
            )
            for index, cell in enumerate(cells())
        ]
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase": args.phase,
                    "source_tree": SOURCE_TREE_FINGERPRINT,
                    "asset_lock_sha256": ASSET_LOCK_SHA256,
                    "physical_gpus": args.physical_gpu,
                    "isaaclab_source": (
                        str(isaaclab_source)
                        if isaaclab_source is not None
                        else None
                    ),
                    "kit_cache_root": (
                        str(kit_cache_root)
                        if kit_cache_root is not None
                        else None
                    ),
                    "runtime_library_dir": (
                        str(runtime_library_dir)
                        if runtime_library_dir is not None
                        else None
                    ),
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        run_phase(
            output_root,
            args.phase,
            python,
            args.physical_gpu,
            args.workers,
            isaaclab_source,
            kit_cache_root,
            runtime_library_dir,
        )
    except (CollectionError, OSError) as error:
        print(f"collect_conveyorvla_al0_grasp: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
