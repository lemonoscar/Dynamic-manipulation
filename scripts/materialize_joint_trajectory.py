#!/usr/bin/env python3
"""Materialize the immutable 5 Hz Joint-Trajectory training dataset."""

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
    materialize_modelscope_sampled_5hz_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelscope-dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_root.expanduser().resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise M0MobileError("joint-trajectory data output must stay outside the Git worktree")
    modelscope_root = args.modelscope_dataset_root.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = materialize_modelscope_sampled_5hz_dataset(
        modelscope_root,
        output,
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
