#!/usr/bin/env python3
"""Build the immutable state-free ConveyorVLA waypoint-v2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint_v2_data import (  # noqa: E402
    discover_eligible_waypoint_episodes,
    materialize_waypoint_v2_dataset,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    episodes = discover_eligible_waypoint_episodes(args.source_root)
    if not episodes:
        parser.error("no eligible waypoint source episodes were found")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            parser.error("--max-episodes must be positive")
        episodes = episodes[: args.max_episodes]
    payload = (
        {
            "eligible_episode_count": len(episodes),
            "source_roots": [str(path.expanduser().resolve()) for path in args.source_root],
            "first_episode": str(episodes[0]),
            "last_episode": str(episodes[-1]),
        }
        if args.audit_only
        else materialize_waypoint_v2_dataset(episodes, args.output_root)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
