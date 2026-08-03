#!/usr/bin/env python3
"""Export canonical V1 episodes into local VLA JSONL views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from conveyor_bench.v1.exporters import (  # noqa: E402
    EXPORT_SCHEMA_VERSION,
    M0_MOBILE_SCHEMA_VERSION,
    ExportSummary,
    export_dynamicvla_episode,
    export_m0_episode,
    export_m0_mobile_episode,
)

_CANONICAL_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
)
_EXPORTERS: dict[
    str, Callable[[str | Path, str | Path], ExportSummary]
] = {
    "dynamicvla": export_dynamicvla_episode,
    "m0": export_m0_episode,
    "m0_mobile": export_m0_mobile_episode,
}
_PROFILE_SCHEMA_VERSIONS = {
    "dynamicvla": EXPORT_SCHEMA_VERSION,
    "m0": EXPORT_SCHEMA_VERSION,
    "m0_mobile": M0_MOBILE_SCHEMA_VERSION,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hashes(episode_directory: Path) -> dict[str, str]:
    return {
        name: _sha256(episode_directory / name)
        for name in _CANONICAL_FILES
        if (episode_directory / name).is_file()
    }


def _episode_identity(episode_directory: Path) -> dict[str, Any]:
    with (episode_directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    episode = manifest.get("episode", {}) if isinstance(manifest, Mapping) else {}
    task = episode.get("task", {}) if isinstance(episode, Mapping) else {}
    return {
        "episode_id": episode.get("episode_id"),
        "task_id": task.get("task_id") if isinstance(task, Mapping) else None,
        "protocol_version": episode.get("protocol_version"),
    }


def find_episodes(source: str | Path) -> tuple[Path, ...]:
    """Resolve either one episode directory or a collection output root."""

    path = Path(source)
    if (path / "manifest.json").is_file() and (path / "steps.jsonl").is_file():
        return (path,)
    candidates_root = path / "episodes" if (path / "episodes").is_dir() else path
    if not candidates_root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {path}")
    episodes = tuple(
        child
        for child in sorted(candidates_root.iterdir())
        if child.is_dir()
        and not child.name.startswith(".")
        and (child / "manifest.json").is_file()
        and (child / "steps.jsonl").is_file()
    )
    if not episodes:
        raise ValueError(f"no published V1 episodes found under {path}")
    return episodes


def _read_existing_profiles(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    profiles = manifest.get("profiles") if isinstance(manifest, Mapping) else None
    return dict(profiles) if isinstance(profiles, Mapping) else {}


def _write_json_file(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def export_episode(
    episode_directory: str | Path,
    profiles: tuple[str, ...],
    *,
    force: bool = False,
) -> dict[str, Any]:
    episode = Path(episode_directory)
    exports_directory = episode / "exports"
    manifest_path = exports_directory / "export_manifest.json"
    destinations = {
        profile: exports_directory / f"{profile}.jsonl" for profile in profiles
    }
    conflicts = [
        path for path in (*destinations.values(), manifest_path) if path.exists()
    ]
    if conflicts and not force:
        names = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(
            f"export output already exists ({names}); pass --force to replace it"
        )

    before_hashes = _canonical_hashes(episode)
    if "manifest.json" not in before_hashes or "steps.jsonl" not in before_hashes:
        raise ValueError(f"{episode} is not a canonical V1 episode")
    exports_directory.mkdir(parents=True, exist_ok=True)

    temporary_outputs: dict[str, Path] = {}
    generated: dict[str, ExportSummary] = {}
    manifest_temporary = exports_directory / (
        f".export_manifest.{uuid4().hex}.tmp"
    )
    try:
        for profile in profiles:
            temporary = exports_directory / (
                f".{profile}.{uuid4().hex}.jsonl.tmp"
            )
            temporary_outputs[profile] = temporary
            generated[profile] = _EXPORTERS[profile](episode, temporary)

        if _canonical_hashes(episode) != before_hashes:
            raise RuntimeError(
                "canonical episode changed while exports were being generated"
            )

        profile_entries = _read_existing_profiles(manifest_path) if force else {}
        for profile, result in generated.items():
            temporary = temporary_outputs[profile]
            profile_entries[profile] = {
                "relative_path": f"{profile}.jsonl",
                "record_count": result.record_count,
                "sha256": _sha256(temporary),
                "schema_version": _PROFILE_SCHEMA_VERSIONS[profile],
                "source_task_outcome": result.source_task_outcome,
                "source_failure_reason": result.source_failure_reason,
            }
        source_results = {
            (
                result.source_task_outcome,
                result.source_failure_reason,
            )
            for result in generated.values()
        }
        if len(source_results) != 1:
            raise RuntimeError(
                "export profiles disagree about the canonical task result"
            )
        source_task_outcome, source_failure_reason = source_results.pop()
        export_manifest = {
            "schema_version": "conveyor-bench-v1-export-manifest-1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": _episode_identity(episode),
            "source_task_outcome": source_task_outcome,
            "source_failure_reason": source_failure_reason,
            "canonical_source_hashes": before_hashes,
            "canonical_files_modified": False,
            "profiles": profile_entries,
        }
        _write_json_file(manifest_temporary, export_manifest)

        for profile, temporary in temporary_outputs.items():
            os.replace(temporary, destinations[profile])
        os.replace(manifest_temporary, manifest_path)
        if _canonical_hashes(episode) != before_hashes:
            raise RuntimeError("canonical episode changed during export publication")
    finally:
        for temporary in (*temporary_outputs.values(), manifest_temporary):
            if temporary.exists():
                temporary.unlink()

    return {
        **_episode_identity(episode),
        "episode_directory": str(episode),
        "exports_directory": str(exports_directory),
        "source_task_outcome": source_task_outcome,
        "source_failure_reason": source_failure_reason,
        "profiles": {
            profile: profile_entries[profile] for profile in profiles
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="One episode directory, an episodes directory, or an output root.",
    )
    parser.add_argument(
        "--profile",
        choices=("dynamicvla", "m0", "m0_mobile", "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing export artifacts, never canonical episode files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.profile == "both":
        profiles = ("dynamicvla", "m0")
    elif args.profile == "all":
        profiles = ("dynamicvla", "m0", "m0_mobile")
    else:
        profiles = (args.profile,)
    try:
        results = [
            export_episode(episode, profiles, force=args.force)
            for episode in find_episodes(args.source)
        ]
    except (OSError, ValueError, RuntimeError) as error:
        print(f"export_v1: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "episode_count": len(results),
                "episodes": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
