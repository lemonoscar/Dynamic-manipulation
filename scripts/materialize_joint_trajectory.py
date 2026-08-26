#!/usr/bin/env python3
"""Materialize immutable Joint-Trajectory v1 data from fresh episode logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.joint_trajectory_data import (  # noqa: E402
    audit_joint_trajectory_dataset,
    materialize_fresh_joint_trajectory_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.episodes_root.expanduser().resolve()
    if not root.is_dir():
        raise M0MobileError(f"fresh episodes root does not exist: {root}")
    output = args.output_root.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise M0MobileError("joint-trajectory data output must stay outside the Git worktree")
    episodes = tuple(
        sorted(
            path.parent
            for path in root.glob("*/joint_commands_50hz.jsonl")
            if (path.parent / "joint_queries_5hz.jsonl").is_file()
            and (path.parent / "summary.json").is_file()
        )
    )
    if not episodes:
        raise M0MobileError("no complete fresh joint-trajectory episodes were found")
    manifest = materialize_fresh_joint_trajectory_dataset(
        episodes, output
    )
    audit = audit_joint_trajectory_dataset(output)
    if not audit["ok"]:
        raise M0MobileError(
            "published joint-trajectory dataset failed audit: "
            + "; ".join(audit["problems"])
        )
    print(json.dumps({"manifest": manifest, "audit": audit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
