#!/usr/bin/env python3
"""Detect exact trajectory overlap across PCT source collections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    roots = [value.expanduser().resolve() for value in args.source_root]
    if len(roots) != len(set(roots)) or any(not root.is_dir() for root in roots):
        raise ValueError("source roots must be unique existing directories")
    records = []
    missing = []
    for root in roots:
        for episode in sorted(root.glob("episode_*")):
            if not episode.is_dir():
                continue
            frames = episode / "frames.jsonl"
            samples = episode / "samples.jsonl"
            if not frames.is_file() or not samples.is_file():
                missing.append(str(episode))
                continue
            records.append(
                {
                    "collection": root.name,
                    "episode": episode.name,
                    "episode_root": str(episode),
                    "frames_sha256": _sha256(frames),
                }
            )

    by_fingerprint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_fingerprint[record["frames_sha256"]].append(record)
    duplicates = [
        {"frames_sha256": digest, "episodes": values}
        for digest, values in sorted(by_fingerprint.items())
        if len(values) > 1
    ]
    report = {
        "schema_version": "conveyor-vla-al0-pct-source-overlap-audit-1",
        "ok": not missing and not duplicates,
        "source_roots": [str(root) for root in roots],
        "episode_counts": {
            root.name: sum(record["collection"] == root.name for record in records)
            for root in roots
        },
        "fingerprinted_episodes": len(records),
        "missing_required_sidecars": missing,
        "exact_duplicate_trajectory_groups": duplicates,
        "exact_duplicate_trajectory_count": sum(
            len(group["episodes"]) for group in duplicates
        ),
        "fingerprint_contract": "sha256_of_frames_jsonl_bytes",
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite overlap audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
