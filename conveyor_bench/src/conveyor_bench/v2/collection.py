"""Read-only source discovery and collection completeness checks for V2 CLIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from conveyor_bench.v1.validation import validate_v1_dataset


@dataclass(frozen=True)
class SourceInventory:
    """Resolved episode input or an exactly accounted collection root."""

    source: Path
    source_kind: str
    episodes: tuple[Path, ...]
    collection_root: Path | None = None
    run_summaries: tuple[Path, ...] = ()
    errors: tuple[str, ...] = ()


class CollectionIntegrityError(ValueError):
    """Raised before mutation when a collection is incomplete or inconsistent."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("collection integrity check failed")


def inspect_source(source: str | Path) -> SourceInventory:
    """Inspect one episode or a collection without silently dropping artifacts."""

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"input must be an episode or collection directory: {path}")

    if _looks_like_episode(path):
        return SourceInventory(
            source=path,
            source_kind="episode",
            episodes=(path,),
        )

    collection_root = path.parent if path.name == "episodes" else path
    episodes_root = collection_root / "episodes"
    run_summaries = tuple(sorted(collection_root.glob("*-summary.json")))
    published = _published_episode_directories(episodes_root)

    # V2 intentionally retains the canonical V1 episode protocol.  Reusing the
    # V1 run validator here keeps run/episode identities, outcome reports,
    # counts, runtime-error rejection, and interrupted recordings consistent.
    v1_result = validate_v1_dataset(collection_root)
    errors = list(v1_result.errors)
    expected_ids, summaries_readable, inventory_errors = _summary_episode_ids(
        run_summaries
    )
    errors.extend(inventory_errors)

    if not episodes_root.is_dir():
        errors.append(f"{episodes_root}: episodes directory is missing")
    if summaries_readable:
        actual_ids = {episode.name for episode in published}
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing:
            errors.append(
                f"{episodes_root}: run summaries reference missing published "
                f"episode directories: {missing}"
            )
        if extra:
            errors.append(
                f"{episodes_root}: published episode directories are absent "
                f"from run summaries: {extra}"
            )
    if not published:
        errors.append(f"{episodes_root}: no published V2 episodes were found")

    return SourceInventory(
        source=path,
        source_kind="collection",
        collection_root=collection_root,
        episodes=published,
        run_summaries=run_summaries,
        errors=_deduplicate(errors),
    )


def require_complete_source(source: str | Path) -> SourceInventory:
    """Resolve ``source`` and reject an incomplete collection before mutation."""

    inventory = inspect_source(source)
    if inventory.errors:
        raise CollectionIntegrityError(inventory.errors)
    return inventory


def _looks_like_episode(path: Path) -> bool:
    canonical_markers = (
        "manifest.json",
        "summary.json",
        "steps.jsonl",
        "objects.jsonl",
    )
    return any((path / marker).exists() for marker in canonical_markers)


def _published_episode_directories(episodes_root: Path) -> tuple[Path, ...]:
    if not episodes_root.is_dir():
        return ()
    return tuple(
        child
        for child in sorted(episodes_root.iterdir())
        if child.is_dir()
        and not child.name.startswith(".")
        and not child.name.endswith(".inprogress")
    )


def _summary_episode_ids(
    summaries: Sequence[Path],
) -> tuple[set[str], bool, list[str]]:
    expected: set[str] = set()
    owners: dict[str, Path] = {}
    errors: list[str] = []
    readable = bool(summaries)
    for summary_path in summaries:
        try:
            with summary_path.open(encoding="utf-8") as stream:
                summary: Any = json.load(stream)
        except (OSError, json.JSONDecodeError):
            readable = False
            continue
        if not isinstance(summary, Mapping):
            readable = False
            continue
        reports = summary.get("episodes")
        if not _is_sequence(reports):
            readable = False
            continue
        for report in reports:
            if not isinstance(report, Mapping):
                readable = False
                continue
            episode_id = report.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                readable = False
                continue
            previous = owners.get(episode_id)
            if previous is not None and previous != summary_path:
                errors.append(
                    f"{summary_path}: episode_id {episode_id!r} is also "
                    f"reported by {previous}"
                )
            owners[episode_id] = summary_path
            expected.add(episode_id)
    return expected, readable, errors


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _deduplicate(errors: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(errors))


__all__ = [
    "CollectionIntegrityError",
    "SourceInventory",
    "inspect_source",
    "require_complete_source",
]
