#!/usr/bin/env python3
"""Export V2 episodes to local DynamicVLA and ABot-M0 JSONL views."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v1.exporters import (  # noqa: E402
    ExportError,
    validate_episode_for_export,
)
from conveyor_bench.v2.collection import (  # noqa: E402
    CollectionIntegrityError,
    require_complete_source,
)
from conveyor_bench.v2.exporters import (  # noqa: E402
    EXPORT_SCHEMA_VERSION,
    iter_dynamicvla_records,
    iter_m0_records,
)
from conveyor_bench.v2.validation import validate_v2_episode  # noqa: E402


EXPORT_MANIFEST_SCHEMA_VERSION = "conveyor-bench-v2-export-manifest-1"
_CANONICAL_FILES = (
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
)
_ITERATORS: dict[
    str,
    Callable[[str | Path], Iterable[Mapping[str, Any]]],
] = {
    "dynamicvla": iter_dynamicvla_records,
    "m0": iter_m0_records,
}


@dataclass(frozen=True)
class _ExportPreflight:
    episode: Path
    canonical_hashes: dict[str, str]
    source_result: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hashes(episode: Path) -> dict[str, str]:
    return {
        name: _sha256(episode / name)
        for name in _CANONICAL_FILES
        if (episode / name).is_file()
    }


def _episode_identity(episode: Path) -> dict[str, Any]:
    with (episode / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    episode_manifest = (
        manifest.get("episode", {}) if isinstance(manifest, Mapping) else {}
    )
    task = (
        episode_manifest.get("task", {})
        if isinstance(episode_manifest, Mapping)
        else {}
    )
    metadata = task.get("metadata", {}) if isinstance(task, Mapping) else {}
    suite = (
        metadata.get("benchmark_suite", {})
        if isinstance(metadata, Mapping)
        else {}
    )
    return {
        "episode_id": episode_manifest.get("episode_id"),
        "task_id": task.get("task_id") if isinstance(task, Mapping) else None,
        "protocol_version": episode_manifest.get("protocol_version"),
        "benchmark_suite_version": (
            suite.get("benchmark_suite_version")
            if isinstance(suite, Mapping)
            else None
        ),
        "scene_id": suite.get("scene_id") if isinstance(suite, Mapping) else None,
        "task_family": (
            suite.get("task_family") if isinstance(suite, Mapping) else None
        ),
    }


def find_episodes(source: str | Path) -> tuple[Path, ...]:
    """Resolve one episode or a complete collection root."""

    return require_complete_source(source).episodes


def _read_existing_profiles(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema_version"
    ) != EXPORT_MANIFEST_SCHEMA_VERSION:
        return {}
    profiles = manifest.get("profiles")
    return dict(profiles) if isinstance(profiles, Mapping) else {}


def _write_jsonl(
    path: Path,
    records: Iterable[Mapping[str, Any]],
) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            json.dump(
                record,
                stream,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            stream.write("\n")
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _preflight_episode(
    episode_directory: str | Path,
    profiles: tuple[str, ...],
    *,
    force: bool,
) -> _ExportPreflight:
    episode = Path(episode_directory)
    if not profiles or any(profile not in _ITERATORS for profile in profiles):
        raise ValueError("at least one supported export profile is required")

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
        raise ValueError(f"{episode} is not a canonical episode")
    validation = validate_v2_episode(episode)
    if not validation.ok:
        details = "; ".join(validation.errors[:5])
        if len(validation.errors) > 5:
            details += f"; ... ({len(validation.errors) - 5} more)"
        raise ExportError(f"V2 episode validation failed: {details}")
    return _ExportPreflight(
        episode=episode,
        canonical_hashes=before_hashes,
        source_result=validate_episode_for_export(episode),
    )


def _export_preflighted_episode(
    preflight: _ExportPreflight,
    profiles: tuple[str, ...],
    *,
    force: bool,
) -> dict[str, Any]:
    episode = preflight.episode
    before_hashes = preflight.canonical_hashes
    source_result = preflight.source_result
    if _canonical_hashes(episode) != before_hashes:
        raise RuntimeError("canonical episode changed after export preflight")

    exports_directory = episode / "exports"
    manifest_path = exports_directory / "export_manifest.json"
    destinations = {
        profile: exports_directory / f"{profile}.jsonl" for profile in profiles
    }
    exports_directory.mkdir(parents=True, exist_ok=True)
    temporary_outputs = {
        profile: exports_directory / f".{profile}.{uuid4().hex}.jsonl.tmp"
        for profile in profiles
    }
    manifest_temporary = exports_directory / (
        f".export_manifest.{uuid4().hex}.tmp"
    )
    profile_entries = _read_existing_profiles(manifest_path) if force else {}
    try:
        for profile in profiles:
            temporary = temporary_outputs[profile]
            count = _write_jsonl(temporary, _ITERATORS[profile](episode))
            profile_entries[profile] = {
                "relative_path": f"{profile}.jsonl",
                "record_count": count,
                "sha256": _sha256(temporary),
                "schema_version": EXPORT_SCHEMA_VERSION,
                "source_task_outcome": source_result.outcome,
                "source_failure_reason": source_result.failure_reason,
            }

        if _canonical_hashes(episode) != before_hashes:
            raise RuntimeError(
                "canonical episode changed while exports were being generated"
            )
        export_manifest = {
            "schema_version": EXPORT_MANIFEST_SCHEMA_VERSION,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": _episode_identity(episode),
            "source_task_outcome": source_result.outcome,
            "source_failure_reason": source_result.failure_reason,
            "canonical_source_hashes": before_hashes,
            "canonical_files_modified": False,
            "profiles": profile_entries,
        }
        _write_json(manifest_temporary, export_manifest)

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
        "source_task_outcome": source_result.outcome,
        "source_failure_reason": source_result.failure_reason,
        "profiles": {
            profile: profile_entries[profile] for profile in profiles
        },
    }


def export_episode(
    episode_directory: str | Path,
    profiles: tuple[str, ...],
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Preflight and atomically publish exports for one episode."""

    return _export_preflighted_episode(
        _preflight_episode(episode_directory, profiles, force=force),
        profiles,
        force=force,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="One episode directory, an episodes directory, or a collection root.",
    )
    parser.add_argument(
        "--profile",
        choices=("dynamicvla", "m0", "both"),
        default="both",
        help="Projection to generate (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing artifacts inside episode exports/ only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = (
        ("dynamicvla", "m0")
        if args.profile == "both"
        else (args.profile,)
    )
    try:
        inventory = require_complete_source(args.source)
        # Validate every source and every destination conflict before creating
        # the first exports/ directory.  An invalid later episode therefore
        # cannot leave a partially exported collection behind.
        preflights = tuple(
            _preflight_episode(episode, profiles, force=args.force)
            for episode in inventory.episodes
        )
        results = [
            _export_preflighted_episode(
                preflight,
                profiles,
                force=args.force,
            )
            for preflight in preflights
        ]
    except CollectionIntegrityError as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "source": str(args.source),
                    "error": str(error),
                    "collection_errors": list(error.errors),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"export_v2: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as error:
        print(
            json.dumps(
                {"ok": False, "source": str(args.source), "error": str(error)},
                indent=2,
                sort_keys=True,
            )
        )
        print(f"export_v2: {error}", file=sys.stderr)
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
