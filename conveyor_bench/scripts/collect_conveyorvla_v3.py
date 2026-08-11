#!/usr/bin/env python3
"""Collect, gate, and index PCT-scene ConveyorVLA V3 demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SCHEMA_VERSION = "conveyorvla-v3-collection-1"
TARGET = "cola"
DESTINATION = "sort_bin_blue"
MAX_EPISODES_PER_PROCESS = 8
STATIONARY_SEEDS = frozenset((1101, 1102, 1103, 2101, 3101))
REQUIRED_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
)


class CollectionError(ValueError):
    """The collector could not prove a complete, unambiguous result."""


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CollectionError(f"cannot read {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CollectionError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _run(
    command: Sequence[str],
    *,
    log_path: Path,
    environment: Mapping[str, str],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).returncode


def build_collection_command(args: argparse.Namespace) -> list[str]:
    kit_log = args.output_root / "logs" / "isaac.kit.log"
    kit_args = " ".join(
        (
            f"--/renderer/activeGpu={args.physical_gpu}",
            "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/maxGpuCount=1",
            f"--/log/file={kit_log}",
        )
    )
    command = [
        str(args.python),
        "-u",
        "-B",
        str(SCRIPTS / "run_benchmark_v3.py"),
        "--asset-root",
        str(args.asset_root),
        "--robot-mode",
        args.robot_mode,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--belt-speed",
        str(args.belt_speed),
        "--max-duration",
        str(args.max_duration),
        "--active-objects",
        "1",
        "--target-asset",
        TARGET,
        "--split",
        "train",
        "--task-family",
        "single_target",
        "--instruction-language",
        "en",
        "--destination",
        DESTINATION,
        "--output-dir",
        str(args.output_root / "raw"),
        "--enable_cameras",
        "--save-camera-frames",
        "--headless",
        "--device",
        "cpu",
        "--kit_args",
        kit_args,
    ]
    if args.belt_speed > 0.0 and args.target_intercept_lead_time is not None:
        command.extend(
            (
                "--target-intercept-lead-time",
                str(args.target_intercept_lead_time),
            )
        )
    return command


def _worker_environment(args: argparse.Namespace) -> dict[str, str]:
    runtime_root = args.output_root / "runtime"
    paths = {
        "TMPDIR": runtime_root / "tmp",
        "XDG_CONFIG_HOME": runtime_root / "config",
        "XDG_DATA_HOME": runtime_root / "data",
        "XDG_STATE_HOME": runtime_root / "state",
        "XDG_CACHE_HOME": (
            args.kit_cache_root or runtime_root / "cache"
        ),
        "MPLCONFIGDIR": runtime_root / "mpl",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({name: str(path) for name, path in paths.items()})
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.physical_gpu),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CONVEYOR_BENCH_V3_ASSET_ROOT": str(args.asset_root),
        }
    )
    if args.isaaclab_source is not None:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(args.isaaclab_source) + (
            os.pathsep + existing if existing else ""
        )
    return environment


def _episodes_for_seeds(
    raw_root: Path, requested_seeds: set[int]
) -> dict[int, Path]:
    episodes_root = raw_root / "episodes"
    result: dict[int, Path] = {}
    if not episodes_root.is_dir():
        return result
    for episode in sorted(episodes_root.iterdir()):
        if not episode.is_dir() or episode.name.startswith("."):
            continue
        missing = [name for name in REQUIRED_FILES if not (episode / name).is_file()]
        if missing:
            raise CollectionError(f"incomplete episode {episode}: {missing}")
        manifest = _read_json(episode / "manifest.json")
        episode_value = manifest.get("episode")
        if not isinstance(episode_value, Mapping):
            raise CollectionError(f"invalid episode manifest: {episode}")
        seeds = episode_value.get("seeds")
        seed = seeds.get("episode") if isinstance(seeds, Mapping) else None
        if seed not in requested_seeds:
            continue
        if seed in result:
            raise CollectionError(f"duplicate episode seed: {seed}")
        metadata = episode_value.get("metadata")
        scene = metadata.get("scene_profile") if isinstance(metadata, Mapping) else None
        fixture = scene.get("object_fixture_contract") if isinstance(scene, Mapping) else None
        task = episode_value.get("task")
        task_metadata = task.get("metadata") if isinstance(task, Mapping) else None
        if (
            not isinstance(scene, Mapping)
            or scene.get("backend") != "isaac_rtx_native_nurec"
            or not isinstance(fixture, Mapping)
            or fixture.get("all_rigid_bodies_valid") is not True
            or fixture.get("all_visuals_composed") is not True
            or not isinstance(task_metadata, Mapping)
            or task_metadata.get("target_asset_id") != TARGET
        ):
            raise CollectionError(f"V3 scene/object provenance gate failed: {episode}")
        result[int(seed)] = episode.resolve()
    return result


def _gate_episode(
    args: argparse.Namespace, episode: Path, log_path: Path
) -> bool:
    summary = _read_json(episode / "summary.json")
    commands = [
        [
            str(args.python),
            "-B",
            str(SCRIPTS / "validate_v1_dataset.py"),
            str(args.output_root / "raw"),
        ],
        [str(args.python), "-B", str(SCRIPTS / "audit_v1_episode.py"), str(episode)],
        [
            str(args.python),
            "-B",
            str(SCRIPTS / "check_v1_camera_gate.py"),
            str(episode),
            "--output",
            str(episode / "camera_gate_report.json"),
        ],
        [
            str(args.python),
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
                str(args.python),
                "-B",
                str(SCRIPTS / "export_v1.py"),
                str(episode),
                "--profile",
                "conveyorvla_al0_temporal",
                "--force",
            ]
        )
    environment = os.environ.copy()
    for command in commands:
        if _run(command, log_path=log_path, environment=environment) != 0:
            raise CollectionError(f"episode gate failed: {' '.join(command)}")
    camera = _read_json(episode / "camera_gate_report.json")
    export = _read_json(episode / "exports" / "export_manifest.json")
    canonical = export.get("canonical_source_hashes")
    canonical_ok = isinstance(canonical, Mapping) and all(
        canonical.get(name) == _sha256(episode / name)
        for name in REQUIRED_FILES
    )
    return (
        summary.get("success") is True
        and camera.get("passed") is True
        and export.get("canonical_files_modified") is False
        and canonical_ok
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-source", type=Path)
    parser.add_argument("--kit-cache-root", type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument(
        "--robot-mode",
        choices=("fixed_base", "whole_body_policy"),
        default="fixed_base",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1101)
    parser.add_argument("--belt-speed", type=float, default=0.0)
    parser.add_argument("--target-intercept-lead-time", type=float, default=5.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--require-all-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve(args: argparse.Namespace) -> argparse.Namespace:
    args.asset_root = args.asset_root.expanduser().resolve(strict=True)
    args.output_root = args.output_root.expanduser().resolve()
    args.python = args.python.expanduser().resolve(strict=True)
    if args.isaaclab_source is not None:
        args.isaaclab_source = args.isaaclab_source.expanduser().resolve(
            strict=True
        )
        if not (args.isaaclab_source / "isaaclab" / "__init__.py").is_file():
            raise CollectionError("--isaaclab-source is not an Isaac Lab source root")
    if args.kit_cache_root is not None:
        args.kit_cache_root = args.kit_cache_root.expanduser().resolve(
            strict=True
        )
    if args.physical_gpu not in {2, 3}:
        raise CollectionError("V3 collection is restricted to physical GPU 2 or 3")
    if not 1 <= args.episodes <= MAX_EPISODES_PER_PROCESS:
        raise CollectionError("--episodes must be within [1, 8]")
    if args.seed < 0:
        raise CollectionError("--seed cannot be negative")
    if not 0.0 <= args.belt_speed <= 0.01:
        raise CollectionError("V3 pilot belt speed must be within [0, 0.01] m/s")
    requested_seeds = set(range(args.seed, args.seed + args.episodes))
    if args.belt_speed == 0.0 and not requested_seeds <= STATIONARY_SEEDS:
        raise CollectionError(
            "stationary V3 collection requires registered scenario seeds"
        )
    if args.max_duration <= 0.0:
        raise CollectionError("--max-duration must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _resolve(build_parser().parse_args(argv))
        command = build_collection_command(args)
        if args.dry_run:
            print(json.dumps({"schema_version": SCHEMA_VERSION, "command": command}, indent=2))
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        log_path = args.output_root / "logs" / "collection.log"
        returncode = _run(
            command,
            log_path=log_path,
            environment=_worker_environment(args),
        )
        requested = set(range(args.seed, args.seed + args.episodes))
        episodes = _episodes_for_seeds(args.output_root / "raw", requested)
        if set(episodes) != requested:
            raise CollectionError(
                f"collector published seeds {sorted(episodes)}, expected {sorted(requested)}"
            )
        eligible = {
            seed: _gate_episode(args, episode, log_path)
            for seed, episode in episodes.items()
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "asset_root": str(args.asset_root),
            "asset_manifest_sha256": _sha256(
                args.asset_root / "TRANSFER_MANIFEST.sha256"
            ),
            "robot_mode": args.robot_mode,
            "target_asset_id": TARGET,
            "destination_zone_id": DESTINATION,
            "belt_speed_mps": args.belt_speed,
            "requested_seeds": sorted(requested),
            "episode_roots": {
                str(seed): str(path) for seed, path in sorted(episodes.items())
            },
            "training_eligible_by_seed": {
                str(seed): value for seed, value in sorted(eligible.items())
            },
            "training_eligible_count": sum(eligible.values()),
            "collector_returncode": returncode,
        }
        _atomic_write(
            args.output_root / "collection_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(
            args.output_root / "successful_episode_roots.txt",
            "".join(
                f"{episodes[seed]}\n" for seed in sorted(eligible) if eligible[seed]
            ),
        )
        if returncode != 0:
            raise CollectionError(f"raw collector returned {returncode}")
        if args.require_all_success and not all(eligible.values()):
            raise CollectionError("one or more episodes failed the training gate")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (CollectionError, OSError, ValueError) as error:
        print(f"collect_conveyorvla_v3: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
