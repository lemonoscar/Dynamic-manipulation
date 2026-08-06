#!/usr/bin/env python3
"""Create a traceable AL0 training subset from observation-time windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path


EXPORT_RELATIVE_PATH = Path("exports/m0_mobile.jsonl")


def _window(value: str) -> tuple[float, float]:
    try:
        start_text, end_text = value.split(":", 1)
        start, end = float(start_text), float(end_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
        raise argparse.ArgumentTypeError("window bounds must satisfy 0 <= START <= END")
    return start, end


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--window", required=True, action="append", type=_window)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.episode_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    source_export = source / EXPORT_RELATIVE_PATH
    if not source_export.is_file():
        raise SystemExit(f"missing source export: {source_export}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise SystemExit("output must not equal or be nested inside the source episode")

    selected: list[str] = []
    for line_number, line in enumerate(
        source_export.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
            observation_time = float(record["observation_time_s"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid export record at line {line_number}: {error}") from error
        if any(start <= observation_time <= end for start, end in args.window):
            selected.append(json.dumps(record, allow_nan=False, separators=(",", ":")))
    if not selected:
        raise SystemExit("no records matched the requested windows")

    shutil.copytree(source, output, copy_function=os.link)
    output_export = output / EXPORT_RELATIVE_PATH
    temporary = output_export.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(selected) + "\n", encoding="utf-8")
    os.replace(temporary, output_export)
    manifest = {
        "schema_version": "conveyor-bench-m0-window-booster-1",
        "source_episode_root": str(source),
        "source_export_sha256": _sha256(source_export),
        "output_export_sha256": _sha256(output_export),
        "record_count": len(selected),
        "windows_s": [list(window) for window in args.window],
    }
    (output / "m0_window_booster.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
