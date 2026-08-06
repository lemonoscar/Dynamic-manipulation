#!/usr/bin/env python3
"""Run the ConveyorVLA AL0 online gate on the matched Go2-X5 task."""

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
        "--endpoint",
        default="http://127.0.0.1:18765",
        help="SSH-forwarded localhost ConveyorVLA AL0 inference endpoint.",
    )
    parser.add_argument("--state-statistics", required=True, type=Path)
    parser.add_argument("--policy-timeout", type=float, default=30.0)
    parser.add_argument("--policy-seed", type=int, default=20260803)
    parser.add_argument(
        "--actions-per-replan",
        type=int,
        default=2,
        help="Execute this prefix of each 16-step AL0 chunk before replanning.",
    )
    parser.add_argument(
        "--transition-actions-per-replan",
        type=int,
        default=12,
        help="Longer prefix used only across grasp transition phases.",
    )
    parser.add_argument(
        "--mobile-approach-assist",
        action="store_true",
        help=(
            "Diagnostic only: use the frozen service approach command, then "
            "return base, arm, and gripper control to AL0."
        ),
    )
    pregrasp_diagnostics = parser.add_mutually_exclusive_group()
    pregrasp_diagnostics.add_argument(
        "--pregrasp-workspace-guard",
        action="store_true",
        help="Enable the fixed, diagnostic X5 pregrasp workspace guard.",
    )
    pregrasp_diagnostics.add_argument(
        "--pregrasp-staging-assist",
        action="store_true",
        help=(
            "Diagnostic only: service-hold the fixed world-frame pregrasp "
            "station before handing control back to AL0."
        ),
    )
    parser.add_argument(
        "--carry-retract-teacher-executor",
        action="store_true",
        help=(
            "Diagnostic only: execute the shadow teacher carry-retract "
            "physical action through the same Cartesian IK path as AL0."
        ),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--belt-speed", type=float, default=0.06)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "conveyorvla_al0_closed_loop",
    )
    parser.add_argument(
        "--no-save-camera-frames",
        action="store_true",
        help="Keep policy cameras enabled but omit audit PNG files.",
    )
    parser.add_argument("--require-all-success", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", enable_cameras=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.enable_cameras:
        parser.error(
            "online ConveyorVLA AL0 requires cameras; do not pass --disable_cameras"
        )
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
                robot_mode=RobotMode.WHOLE_BODY_POLICY,
                episodes=args.episodes,
                seed=args.seed,
                belt_speed_mps=args.belt_speed,
                max_duration_s=args.max_duration,
                active_object_count=1,
                target_asset_id="part_red_block",
                destination_zone_id="sort_bin_blue",
                device=args.device,
                enable_cameras=True,
                save_camera_frames=not args.no_save_camera_frames,
                curriculum_split=CurriculumSplit.TRAIN,
                task_family=TaskFamily.SINGLE_TARGET,
                instruction_language=InstructionLanguage.BILINGUAL,
                m0_policy_endpoint=args.endpoint,
                m0_state_statistics=args.state_statistics.resolve(),
                m0_policy_timeout_s=args.policy_timeout,
                m0_policy_seed=args.policy_seed,
                m0_actions_per_replan=args.actions_per_replan,
                m0_transition_actions_per_replan=(
                    args.transition_actions_per_replan
                ),
                m0_mobile_approach_assist=(
                    args.mobile_approach_assist
                ),
                m0_pregrasp_workspace_guard=(
                    args.pregrasp_workspace_guard
                ),
                m0_pregrasp_staging_assist=(
                    args.pregrasp_staging_assist
                ),
                m0_carry_retract_teacher_executor=(
                    args.carry_retract_teacher_executor
                ),
            )
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return collection_exit_code(
            summary, require_all_success=args.require_all_success
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
