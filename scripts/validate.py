#!/usr/bin/env python3
"""Validate published ConveyorBench run artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.schema.validation import validate_v1_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a ConveyorBench run summary or output root.",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Path to one *-summary.json file or its output root.",
    )
    args = parser.parse_args(argv)

    result = validate_v1_dataset(args.source)
    if result.ok:
        print(
            "dataset valid: "
            f"{result.run_count} run(s), "
            f"{result.episode_count} episode(s), "
            f"{result.sample_count} step(s), "
            f"{result.object_record_count} object record(s), "
            f"{result.camera_frame_count} PNG frame(s)"
        )
        return 0

    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(
        f"dataset invalid: {len(result.errors)} error(s)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
