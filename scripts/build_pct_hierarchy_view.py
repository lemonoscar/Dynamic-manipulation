#!/usr/bin/env python3
"""Build the four-phase sidecar view over a PCT LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    materialize_pct_hierarchy_view,
)
from conveyor_bench.conveyorvla.pct_dataset import discover_pct_episodes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--base-lerobot-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        episodes = discover_pct_episodes(args.source_root)
        report = materialize_pct_hierarchy_view(
            episodes,
            args.base_lerobot_root,
            args.output_root,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "dataset_root": report["dataset_root"],
                    "episode_count": report["episode_count"],
                    "selected_frame_count": report["selected_frame_count"],
                    "split_counts": report["split_counts"],
                    "phase_counts": report["phase_counts"],
                    "domain_counts": report["domain_counts"],
                    "annotations_sha256": report["annotations_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
