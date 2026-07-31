#!/usr/bin/env python3
"""Validate one ConveyorBench V2 episode or a collection of episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v2.collection import (  # noqa: E402
    inspect_source,
    require_complete_source,
)
from conveyor_bench.v2.validation import validate_v2_episode  # noqa: E402


REPORT_SCHEMA_VERSION = "conveyor-bench-v2-validation-report-1"


def find_episodes(source: str | Path) -> tuple[Path, ...]:
    """Resolve one episode or a complete collection root."""

    return require_complete_source(source).episodes


def validate_source(source: str | Path) -> dict[str, Any]:
    inventory = inspect_source(source)
    reports: list[dict[str, Any]] = []
    for episode in inventory.episodes:
        result = validate_v2_episode(episode)
        reports.append(
            {
                "episode_directory": str(episode),
                "ok": result.ok,
                "errors": list(result.errors),
                "sample_count": result.sample_count,
                "object_record_count": result.object_record_count,
                "camera_frame_count": result.camera_frame_count,
            }
        )
    valid_count = sum(report["ok"] for report in reports)
    collection_errors = list(inventory.errors)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "ok": not collection_errors and valid_count == len(reports),
        "source": str(Path(source)),
        "source_kind": inventory.source_kind,
        "collection_root": (
            str(inventory.collection_root)
            if inventory.collection_root is not None
            else None
        ),
        "run_summary_count": len(inventory.run_summaries),
        "collection_error_count": len(collection_errors),
        "collection_errors": collection_errors,
        "episode_count": len(reports),
        "valid_episode_count": valid_count,
        "invalid_episode_count": len(reports) - valid_count,
        "sample_count": sum(report["sample_count"] for report in reports),
        "object_record_count": sum(
            report["object_record_count"] for report in reports
        ),
        "camera_frame_count": sum(
            report["camera_frame_count"] for report in reports
        ),
        "episodes": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="One episode directory, an episodes directory, or a collection root.",
    )
    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_source(args.source)
    except (OSError, ValueError) as error:
        _print_json(
            {
                "schema_version": REPORT_SCHEMA_VERSION,
                "ok": False,
                "source": str(args.source),
                "error": str(error),
            }
        )
        print(f"validate_v2_dataset: {error}", file=sys.stderr)
        return 2
    _print_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
