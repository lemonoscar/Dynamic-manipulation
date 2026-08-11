#!/usr/bin/env python3
"""Collect ConveyorBench V3 episodes in the native Liangzhu NuRec scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v3.assets import validate_asset_bundle
from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--robot-mode",
        choices=("fixed_base", "whole_body_policy"),
        default="fixed_base",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--belt-speed", type=float, default=0.0)
    parser.add_argument("--target-intercept-lead-time", type=float)
    parser.add_argument("--max-duration", type=float, default=45.0)
    parser.add_argument("--active-objects", type=int, default=1)
    parser.add_argument("--target-asset", default="cola")
    parser.add_argument(
        "--split", choices=("train", "val", "unseen"), default="train"
    )
    parser.add_argument(
        "--task-family",
        choices=("single_target", "language_conditioned"),
        default="single_target",
    )
    parser.add_argument(
        "--instruction-language", choices=("en", "en_zh"), default="en_zh"
    )
    parser.add_argument("--destination", default="sort_bin_blue")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "v3_3dgs_smoke",
    )
    parser.add_argument("--save-camera-frames", action="store_true")
    parser.add_argument("--require-all-success", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", enable_cameras=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.enable_cameras:
        parser.error("V3 native NuRec collection requires --enable_cameras")

    # Cheap structural preflight before paying Isaac Sim startup cost. The
    # runtime repeats this with full SHA-256 verification before scene load.
    bundle = validate_asset_bundle(args.asset_root, verify_all_hashes=False)
    print(
        f"[V3] asset preflight passed: {bundle.report.file_count} files",
        flush=True,
    )

    app = AppLauncher(args)
    simulation_app = app.app
    print("[V3] Isaac application ready", flush=True)
    try:
        from conveyor_bench.cli import collection_exit_code
        from conveyor_bench.isaac.runtime_v3 import (
            RuntimeOptionsV3,
            run_collection_v3,
        )
        from conveyor_bench.v1.protocol import RobotMode
        from conveyor_bench.v1.tasking import (
            CurriculumSplit,
            InstructionLanguage,
            TaskFamily,
        )

        print("[V3] creating physical scene and collector", flush=True)
        try:
            summary = run_collection_v3(
                RuntimeOptionsV3(
                    output_root=args.output_dir.resolve(),
                    asset_root=args.asset_root,
                    robot_mode=RobotMode(args.robot_mode),
                    episodes=args.episodes,
                    seed=args.seed,
                    belt_speed_mps=args.belt_speed,
                    target_intercept_lead_time_s=(
                        args.target_intercept_lead_time
                    ),
                    max_duration_s=args.max_duration,
                    active_object_count=args.active_objects,
                    target_asset_id=args.target_asset,
                    destination_zone_id=args.destination,
                    device=args.device,
                    enable_cameras=True,
                    save_camera_frames=args.save_camera_frames,
                    curriculum_split=CurriculumSplit(args.split),
                    task_family=TaskFamily(args.task_family),
                    instruction_language=InstructionLanguage(
                        args.instruction_language
                    ),
                )
            )
        except BaseException as error:
            print(
                f"[V3] collector raised {type(error).__name__}: {error!r}",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(json.dumps(summary, indent=2, sort_keys=True))
        return collection_exit_code(
            summary, require_all_success=args.require_all_success
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
