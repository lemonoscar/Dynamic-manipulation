#!/usr/bin/env python3
"""Trace target/conveyor state for a few physics steps without running an oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--spawn-y",
    type=float,
    default=0.0,
    help="world-Y spawn coordinate; positive Y is upstream",
)
parser.add_argument(
    "--lane-x",
    type=float,
    default=0.70,
    help="world-X lane coordinate in front of the dog",
)
parser.add_argument("--belt-speed", type=float, default=0.0)
parser.add_argument("--physics-steps", type=int, default=12)
parser.add_argument("--full-trace", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

try:
    from conveyor_bench.isaac.runtime import ConveyorRuntime, RuntimeOptions
    from conveyor_bench.protocol import TaskType

    runtime = ConveyorRuntime(
        RuntimeOptions(
            output_root=PROJECT_ROOT / "outputs" / "probe",
            task_type=(
                TaskType.C1_DYNAMIC_PICK
                if args.belt_speed > 0.0
                else TaskType.C0_STATIC_PICK
            ),
            belt_speed_mps=args.belt_speed if args.belt_speed > 0.0 else 0.08,
            device=args.device,
            save_video=False,
            enable_cameras=False,
        )
    )
    try:
        report = runtime.probe_object_dynamics(
            spawn_y=args.spawn_y,
            lane_x=args.lane_x,
            surface_speed_mps=args.belt_speed,
            physics_steps=args.physics_steps,
        )
        if not args.full_trace and len(report) > 1:
            report = [report[0], report[-1]]
        print(json.dumps(report, indent=2))
    finally:
        runtime.close()
finally:
    simulation_app.close()
