#!/usr/bin/env python3
"""Validate the SSH sidecar assets required by ConveyorBench V3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v3.assets import validate_asset_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        help="Absolute SSH-delivered bundle root; otherwise use the V3 env var.",
    )
    parser.add_argument(
        "--allowed-root",
        type=Path,
        help="Optional writable-root boundary enforced after path resolution.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip file-content hashes; never use this for collection.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    bundle = validate_asset_bundle(
        args.asset_root,
        verify_all_hashes=not args.metadata_only,
        allowed_root=args.allowed_root,
    )
    result = {
        "status": "passed",
        "bundle": bundle.report.to_dict(),
        "scene_usda": str(bundle.scene_usda),
        "nurec_usdz": str(bundle.nurec_usdz),
        "collision_usda": str(bundle.collision_usda),
        "object_usd": {
            object_id: str(bundle.object_usd(object_id))
            for object_id in ("cola", "apple", "orange", "bottle", "box2")
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
