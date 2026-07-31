#!/usr/bin/env python3
"""Fail closed when a V1 episode has stale or policy-invisible camera data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v1.camera_gate import (  # noqa: E402
    CameraGateError,
    audit_camera_episode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path; stdout is always written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_camera_episode(args.episode)
    except (CameraGateError, OSError) as error:
        report = {
            "schema_version": "conveyor-bench-camera-gate-v1",
            "episode_directory": str(args.episode.resolve()),
            "passed": False,
            "issues": [
                {
                    "code": "camera_gate_input_invalid",
                    "message": str(error),
                }
            ],
            "metrics": {},
        }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
