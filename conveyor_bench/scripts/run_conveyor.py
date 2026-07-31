#!/usr/bin/env python3
"""Launch the self-contained ConveyorBench V0 collection loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("c0", "c1"), default="c1")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--belt-speed", type=float, default=0.08)
    parser.add_argument("--max-duration", type=float, default=15.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="Return exit code 3 if any completed episode fails the task.",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.enable_cameras and not args.no_video:
        parser.error("video capture requires --enable_cameras (or pass --no-video)")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        from conveyor_bench.isaac.runtime import RuntimeOptions, run_collection
        from conveyor_bench.cli import collection_exit_code
        from conveyor_bench.protocol import TaskType

        task_type = (
            TaskType.C0_STATIC_PICK
            if args.task == "c0"
            else TaskType.C1_DYNAMIC_PICK
        )
        summary = run_collection(
            RuntimeOptions(
                output_root=args.output_dir.resolve(),
                task_type=task_type,
                episodes=args.episodes,
                seed=args.seed,
                belt_speed_mps=args.belt_speed,
                max_duration_s=args.max_duration,
                device=args.device,
                save_video=not args.no_video,
                enable_cameras=bool(args.enable_cameras),
            )
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return collection_exit_code(
            summary,
            require_all_success=args.require_all_success,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
