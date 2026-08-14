#!/usr/bin/env python3
"""Audit Liangzhu PCT episodes for four-phase ConveyorVLA training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    audit_pct_hierarchy_episode,
)
from conveyor_bench.conveyorvla.pct_dataset import discover_pct_episodes  # noqa: E402
from conveyor_bench.conveyorvla.subtasks import PHASE_ORDER  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-episodes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episodes = discover_pct_episodes(args.source_root)
    reports = [audit_pct_hierarchy_episode(root) for root in episodes]
    summary = _summary(reports)
    payload = {
        "schema_version": "conveyor-vla-al0-pct-hierarchy-audit-1",
        "ok": summary["eligible_episodes"] > 0,
        **summary,
    }
    if args.include_episodes:
        payload["episodes"] = reports
    else:
        payload["ineligible"] = [
            {
                "episode_root": report["episode_root"],
                "problems": report["problems"],
            }
            for report in reports
            if not report["eligible"]
        ]
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")
    return 0 if payload["ok"] else 1


def _summary(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(reports)
    eligible = [report for report in values if report["eligible"]]
    problems = Counter(
        problem
        for report in values
        if not report["eligible"]
        for problem in report["problems"]
    )
    metric_paths = {
        "nav_to_source_tcp_position_drift_m": (
            "NAV_TO_SOURCE",
            "tcp_position_drift_m",
        ),
        "nav_to_target_tcp_position_drift_m": (
            "NAV_TO_TARGET",
            "tcp_position_drift_m",
        ),
        "pick_base_command_abs_max": ("PICK", "base_command_abs_max"),
        "place_base_command_abs_max": ("PLACE", "base_command_abs_max"),
        "nav_to_target_displacement_m": ("NAV_TO_TARGET", "base_displacement_m"),
        "nav_to_target_cumulative_turn_rad": (
            "NAV_TO_TARGET",
            "cumulative_turn_rad",
        ),
    }
    metrics = {}
    for name, (phase, metric) in metric_paths.items():
        series = sorted(
            float(report["hierarchy_metrics"]["phase_metrics"][phase][metric])
            for report in values
            if report["hierarchy_metrics"] is not None
            and math.isfinite(
                float(report["hierarchy_metrics"]["phase_metrics"][phase][metric])
            )
        )
        metrics[name] = _distribution(series)
    phase_queries = {
        phase.name: sum(
            report["hierarchy_metrics"]["phase_query_counts"][phase.name]
            for report in eligible
        )
        for phase in PHASE_ORDER
    }
    return {
        "discovered_episodes": len(values),
        "eligible_episodes": len(eligible),
        "ineligible_episodes": len(values) - len(eligible),
        "eligible_phase_queries": phase_queries,
        "eligible_total_queries": sum(phase_queries.values()),
        "metric_distributions": metrics,
        "problem_counts": dict(problems.most_common()),
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "p50": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(len(values) * 0.95))],
        "max": values[-1],
    }


if __name__ == "__main__":
    raise SystemExit(main())
