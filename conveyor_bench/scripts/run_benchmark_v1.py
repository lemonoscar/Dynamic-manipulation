#!/usr/bin/env python3
"""Collect ConveyorBench V1 dynamic pick-and-place episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-mode",
        choices=("fixed_base", "whole_body_policy"),
        default="fixed_base",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--belt-speed", type=float, default=0.01)
    parser.add_argument(
        "--target-intercept-lead-time",
        type=float,
        help=(
            "Training curriculum only: spawn the target this many belt-seconds "
            "upstream of the tracking work zone."
        ),
    )
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--active-objects", type=int, default=3)
    parser.add_argument(
        "--target-asset", default="part_red_block"
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "unseen"),
        default="train",
        help="Use only objects from this frozen curriculum split.",
    )
    parser.add_argument(
        "--task-family",
        choices=("single_target", "language_conditioned"),
        help=(
            "Defaults to single_target for one active object and "
            "language_conditioned otherwise."
        ),
    )
    parser.add_argument(
        "--instruction-language",
        choices=("en", "en_zh"),
        default="en_zh",
    )
    parser.add_argument(
        "--destination",
        choices=("sort_bin_blue", "sort_bin_yellow"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "v1_smoke",
    )
    parser.add_argument(
        "--save-camera-frames",
        action="store_true",
        help="Write synchronized lossless PNG observations at 25 Hz.",
    )
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="Return code 3 if any physically completed task fails.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.save_camera_frames and not args.enable_cameras:
        parser.error("--save-camera-frames requires --enable_cameras")
    app = AppLauncher(args)
    simulation_app = app.app
    try:
        from conveyor_bench.cli import collection_exit_code
        from conveyor_bench.isaac.runtime_v1 import (
            RuntimeOptionsV1,
            run_collection_v1,
        )
        from conveyor_bench.v1.protocol import RobotMode
        from conveyor_bench.v1.tasking import (
            CurriculumSplit,
            InstructionLanguage,
            TaskFamily,
        )

        summary = run_collection_v1(
            RuntimeOptionsV1(
                output_root=args.output_dir.resolve(),
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
                enable_cameras=bool(args.enable_cameras),
                save_camera_frames=args.save_camera_frames,
                curriculum_split=CurriculumSplit(args.split),
                task_family=(
                    TaskFamily(args.task_family)
                    if args.task_family is not None
                    else None
                ),
                instruction_language=InstructionLanguage(
                    args.instruction_language
                ),
            )
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return collection_exit_code(
            summary, require_all_success=args.require_all_success
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
