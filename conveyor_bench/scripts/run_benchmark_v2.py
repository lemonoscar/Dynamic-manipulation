#!/usr/bin/env python3
"""Collect ConveyorBench V2 near-sort or remote-delivery episodes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.v1.protocol import RobotMode, to_jsonable  # noqa: E402
from conveyor_bench.v1.tasking import (  # noqa: E402
    CurriculumSplit,
    InstructionLanguage,
    TaskFamily,
)
from conveyor_bench.v2.config import DEFAULT_SUITE_CONFIG, SceneId  # noqa: E402
from conveyor_bench.v2.tasking import (  # noqa: E402
    build_task_context,
    validate_task_combination,
)


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scene",
        choices=tuple(scene.value for scene in SceneId),
        default=SceneId.TRANSVERSE_NEAR_SORT_V2.value,
    )
    parser.add_argument(
        "--task-family",
        choices=tuple(family.value for family in TaskFamily),
        default=TaskFamily.SINGLE_TARGET.value,
    )
    parser.add_argument(
        "--robot-mode",
        choices=("fixed_base", "whole_body_policy"),
        default="fixed_base",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--belt-speed",
        type=float,
        choices=DEFAULT_SUITE_CONFIG.belt_speeds_mps,
        default=DEFAULT_SUITE_CONFIG.belt_speeds_mps[0],
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Override the selected scene's frozen smoke-test duration.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(split.value for split in CurriculumSplit),
        default=CurriculumSplit.TRAIN.value,
    )
    parser.add_argument(
        "--instruction-language",
        choices=tuple(language.value for language in InstructionLanguage),
        default=InstructionLanguage.BILINGUAL.value,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "v2_smoke",
    )
    parser.add_argument(
        "--save-camera-frames",
        action="store_true",
        help="Write synchronized head/wrist RGB plus observer overview PNGs.",
    )
    parser.add_argument(
        "--require-all-success",
        action="store_true",
        help="Return code 3 if any physically completed task fails.",
    )
    parser.add_argument(
        "--dry-run-task",
        action="store_true",
        help="Resolve and print the task contract without starting Isaac Sim.",
    )


def _preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    _add_benchmark_arguments(parser)
    return parser


def _resolve_task(args: argparse.Namespace):
    if args.episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if args.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if args.max_duration is not None and (
        not math.isfinite(args.max_duration) or args.max_duration <= 0.0
    ):
        raise ValueError("max-duration must be finite and positive")
    scene, family, mode = validate_task_combination(
        args.scene,
        args.task_family,
        args.robot_mode,
    )
    context = build_task_context(
        seed=args.seed,
        scene_id=scene,
        family=family,
        mode=mode,
        split=args.split,
        instruction_language=args.instruction_language,
    )
    duration = (
        args.max_duration
        if args.max_duration is not None
        else DEFAULT_SUITE_CONFIG.scene(scene).default_max_duration_s
    )
    initialization_end_s = max(
        float(entry["initialization_end_s"])
        for entry in context.task.metadata["spawn_schedule"]
    )
    if duration <= initialization_end_s:
        raise ValueError(
            "max-duration must exceed the task initialization horizon "
            f"({initialization_end_s:g} s)"
        )
    context = replace(
        context,
        task=replace(
            context.task,
            belt_speed_mps=args.belt_speed,
            max_duration_s=duration,
            metadata={
                **dict(context.task.metadata),
                "belt_speed_mps": args.belt_speed,
            },
        ),
    )
    return scene, family, mode, context


def build_parser(app_launcher_type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_benchmark_arguments(parser)
    app_launcher_type.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    preflight, unknown = _preflight_parser().parse_known_args(raw_argv)
    if preflight.dry_run_task and unknown:
        print(
            "run_benchmark_v2: unrecognized dry-run arguments: "
            + " ".join(unknown),
            file=sys.stderr,
        )
        return 2
    try:
        scene, family, mode, context = _resolve_task(preflight)
    except (TypeError, ValueError) as error:
        print(f"run_benchmark_v2: {error}", file=sys.stderr)
        return 2
    if preflight.dry_run_task:
        print(
            json.dumps(
                {
                    "ok": True,
                    "simulator_started": False,
                    "benchmark_suite": "conveyor-bench-v2",
                    "canonical_protocol": "conveyor-bench-v1",
                    "task": to_jsonable(context.task),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    from isaaclab.app import AppLauncher

    parser = build_parser(AppLauncher)
    args = parser.parse_args(raw_argv)
    if args.save_camera_frames and not args.enable_cameras:
        parser.error("--save-camera-frames requires --enable_cameras")
    if args.device != "cpu":
        parser.error("V2 requires --device cpu")

    app = AppLauncher(args)
    simulation_app = app.app
    try:
        from conveyor_bench.cli import collection_exit_code
        from conveyor_bench.isaac.runtime_v2 import (
            RuntimeOptionsV2,
            run_collection_v2,
        )

        summary = run_collection_v2(
            RuntimeOptionsV2(
                output_root=args.output_dir.resolve(),
                scene_id=scene,
                task_family=family,
                robot_mode=mode,
                episodes=args.episodes,
                seed=args.seed,
                belt_speed_mps=args.belt_speed,
                max_duration_s=args.max_duration,
                device=args.device,
                enable_cameras=bool(args.enable_cameras),
                save_camera_frames=args.save_camera_frames,
                curriculum_split=CurriculumSplit(args.split),
                instruction_language=InstructionLanguage(
                    args.instruction_language
                ),
            )
        )
        # App shutdown can tear down simulator-owned logging/output handlers.
        # Flush the machine-readable result before closing Kit so wrappers such
        # as ``conda run`` reliably receive the collection report.
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return collection_exit_code(
            summary, require_all_success=args.require_all_success
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
