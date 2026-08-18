#!/usr/bin/env python3
"""Convert successful PCT Liangzhu episodes to ConveyorVLA LeRobot v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    audit_pct_hierarchy_episode,
)
from conveyor_bench.conveyorvla.pct_dataset import (  # noqa: E402
    audit_pct_episode,
    discover_pct_episodes,
    materialize_pct_lerobot_v3,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-id", default="local/conveyorvla-liangzhu-pct")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-episodes-per-source", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--require-hierarchy-eligible",
        action="store_true",
        help="select only episodes passing all four-phase and action-domain gates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.max_episodes is not None and args.max_episodes_per_source is not None:
            raise M0MobileError(
                "--max-episodes and --max-episodes-per-source are mutually exclusive"
            )
        episodes_by_source = [
            discover_pct_episodes([source]) for source in args.source_root
        ]
        episodes = tuple(
            episode for group in episodes_by_source for episode in group
        )
        audit_episode = (
            audit_pct_hierarchy_episode
            if args.require_hierarchy_eligible
            else audit_pct_episode
        )
        audits = [audit_episode(root) for root in episodes]
        eligible = [
            Path(report["episode_root"])
            for report in audits
            if report["eligible"]
        ]
        if args.max_episodes is not None:
            if args.max_episodes <= 0:
                raise M0MobileError("--max-episodes must be positive")
            eligible = eligible[: args.max_episodes]
        elif args.max_episodes_per_source is not None:
            if args.max_episodes_per_source <= 0:
                raise M0MobileError("--max-episodes-per-source must be positive")
            eligible_set = set(eligible)
            eligible = [
                episode
                for group in episodes_by_source
                for episode in list(
                    item for item in group if item in eligible_set
                )[: args.max_episodes_per_source]
            ]
        audit_summary = {
            "eligibility_contract": (
                "four_phase_hierarchy"
                if args.require_hierarchy_eligible
                else "base_pct"
            ),
            "discovered_episodes": len(audits),
            "eligible_episodes": sum(report["eligible"] for report in audits),
            "ineligible_episodes": sum(not report["eligible"] for report in audits),
            "selected_episodes": len(eligible),
            "ineligible": [report for report in audits if not report["eligible"]],
        }
        if args.audit_only:
            print(json.dumps({"ok": True, **audit_summary}, indent=2, sort_keys=True))
            return 0
        if args.output_root is None:
            raise M0MobileError("--output-root is required unless --audit-only is used")
        if not eligible:
            raise M0MobileError("no eligible PCT episodes were selected")
        report = materialize_pct_lerobot_v3(
            eligible,
            args.output_root,
            repo_id=args.repo_id,
        )
        print(json.dumps({"ok": True, **audit_summary, **report}, indent=2, sort_keys=True))
        return 0
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
