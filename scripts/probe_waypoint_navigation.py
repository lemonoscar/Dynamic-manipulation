#!/usr/bin/env python3
"""Run the real approved PCT/DWA stack on a synthetic collision-free map."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint import ROUTE_TOKENS, WaypointRoute  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_execution import (  # noqa: E402
    NavigationExecutionConfig,
    PCTDWARecedingHorizonExecutor,
)
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (  # noqa: E402
    APPROVED_ARM_VLA_COMMIT,
    ArmVLADWAControllerAdapter,
    ArmVLAPCTPlannerAdapter,
)
from conveyor_bench.conveyorvla.waypoint_protocol import WaypointResponse  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--server-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-control-steps", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = args.report.expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": "conveyorvla-waypoint-navigation-probe-v1",
        "status": "fail",
        "reference_root": str(args.reference_root.expanduser().resolve()),
        "steps": [],
    }
    try:
        report.update(_run_probe(args))
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        _write_json(report_path, report)
        print(json.dumps(_console_summary(report_path, report), indent=2, sort_keys=True), flush=True)
        return 1
    _write_json(report_path, report)
    print(json.dumps(_console_summary(report_path, report), indent=2, sort_keys=True), flush=True)
    return 0


def _run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_control_steps <= 0:
        raise ValueError("max control steps must be positive")
    reference = args.reference_root.expanduser().resolve()
    commit = _clean_commit(reference)
    if commit != APPROVED_ARM_VLA_COMMIT:
        raise RuntimeError(f"approved arm-vla reference is required, got {commit}")
    sys.path.insert(0, str(reference))
    from source.interfaces.navigation import NavGoal
    from source.interfaces.simulation import SimulationState
    from source.navigation.navlib.dwa import DWAConfig, DWAController
    from source.navigation.navlib.grid_map import OccupancyGridMap
    from source.navigation.pct_adapter import PCTNavPlanner, PCTPlannerConfig

    with tempfile.TemporaryDirectory(prefix="conveyorvla-waypoint-pct-") as temporary:
        temporary_root = Path(temporary)
        tomogram_path, walkable_path = _synthetic_pct_assets(temporary_root)
        reference_planner = PCTNavPlanner(
            PCTPlannerConfig(
                enabled=True,
                server_script=reference / "scripts" / "navigation" / "pct_grid_server.py",
                server_python=args.server_python.expanduser().resolve(),
                tomogram_path=tomogram_path,
                walkable_path=walkable_path,
                coord_mode="identity",
                robot_root_to_floor_m=0.45,
                fallback_to_astar=False,
            )
        )
        pct = ArmVLAPCTPlannerAdapter(
            reference_planner,
            simulation_state_factory=SimulationState,
            nav_goal_factory=NavGoal,
            reference_commit=commit,
        )
        dwa = ArmVLADWAControllerAdapter(
            DWAController,
            DWAConfig(
                control_dt=0.05,
                goal_tolerance=0.10,
                max_linear_velocity=0.50,
                max_angular_velocity=1.0,
                max_linear_accel=2.5,
                max_angular_accel=3.0,
            ),
            reference_commit=commit,
        )
        local_map = OccupancyGridMap(
            np.zeros((80, 80), dtype=bool),
            0.05,
            (-2.0, -2.0, 0.0),
        )
        executor = PCTDWARecedingHorizonExecutor(
            pct,
            dwa,
            NavigationExecutionConfig(
                chunk_timeout_s=30.0,
            ),
        )
        response = _known_waypoint_response()
        pose = np.array([0.0, 0.0, 0.45, 0.0], dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        try:
            planned = executor.begin(
                response,
                (pose[0], pose[1], pose[2], 1.0, 0.0, 0.0, 0.0),
                now_s=0.0,
            )
            if planned.failed:
                raise RuntimeError(f"known waypoint PCT plan failed: {planned.reason}")
            traces = []
            terminal = None
            for step in range(args.max_control_steps):
                command = executor.step(
                    (pose[0], pose[1], pose[2], *_yaw_quaternion(pose[3])),
                    tuple(float(value) for value in velocity),
                    local_map,
                    now_s=(step + 1) * 0.05,
                )
                command_trace = dict(command.trace)
                traces.append(
                    {
                        "step": step,
                        "pose_xyzyaw": pose.tolist(),
                        "base_velocity": list(command.base_velocity),
                        "status": command.status,
                        "reason": command.reason,
                        "failed": command.failed,
                        "requires_requery": command.requires_requery,
                        "distance_m": command_trace.get("distance_m"),
                        "yaw_error_rad": command_trace.get("yaw_error_rad"),
                        "dwa_elapsed_ms": command_trace.get("dwa_elapsed_ms"),
                        "dwa_debug": dict(
                            command_trace.get("dwa_adapter_trace") or {}
                        ).get("debug"),
                    }
                )
                if command.requires_requery or command.failed:
                    terminal = command
                    break
                velocity[:] = command.base_velocity
                cosine, sine = math.cos(pose[3]), math.sin(pose[3])
                pose[0] += (velocity[0] * cosine - velocity[1] * sine) * 0.05
                pose[1] += (velocity[0] * sine + velocity[1] * cosine) * 0.05
                pose[3] = _wrap(pose[3] + velocity[2] * 0.05)
            if terminal is None:
                raise RuntimeError("known waypoint DWA execution did not terminate")
            if terminal.failed or not terminal.requires_requery:
                raise RuntimeError(f"known waypoint execution failed: {terminal.reason}")
            commands = [row["base_velocity"] for row in traces]
            if not all(
                all(math.isfinite(float(value)) for value in command)
                and abs(command[0]) <= 0.60
                and abs(command[1]) <= 0.40
                and abs(command[2]) <= 1.20
                for command in commands
            ):
                raise RuntimeError("DWA emitted a non-finite or unbounded command")
            path = planned.trace["pct_path_world"]
            snap = float(planned.trace["pct_snap_distance_m"])
            if len(path) < 2 or snap > 0.10:
                raise RuntimeError("PCT path/snap gate failed")
            if planned.trace["pct_metadata"].get("fallback_allowed") is not False:
                raise RuntimeError("PCT fallback was not explicitly disabled")
            return {
                "arm_vla_reference_commit": commit,
                "known_waypoint_body": [0.60, 0.0, 0.0],
                "predicted_goal_world": planned.trace["predicted_goal_world"],
                "pct_path_world": path,
                "pct_snap_distance_m": snap,
                "pct_metadata": planned.trace["pct_metadata"],
                "control_step_count": len(traces),
                "final_pose_xyzyaw": pose.tolist(),
                "terminal_reason": terminal.reason,
                "requery_after_first_waypoint": terminal.requires_requery,
                "all_dwa_commands_finite_and_bounded": True,
                "fallback_allowed": False,
                "steps": traces,
            }
        finally:
            reference_planner.close()


def _synthetic_pct_assets(root: Path) -> tuple[Path, Path]:
    traversability = np.ones((3, 80, 80), dtype=np.float32)
    zeros = np.zeros_like(traversability)
    tomogram = {
        "data": np.stack((traversability, zeros, zeros, zeros, zeros), axis=0),
        "resolution": 0.05,
        "center": np.array([0.0, 0.0], dtype=np.float64),
        "slice_h0": 0.0,
        "slice_dh": 0.5,
    }
    tomogram_path = root / "synthetic_tomogram.pickle"
    walkable_path = root / "synthetic_walkable.npy"
    with tomogram_path.open("wb") as stream:
        pickle.dump(tomogram, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(walkable_path, np.ones_like(traversability, dtype=bool))
    return tomogram_path, walkable_path


def _known_waypoint_response() -> WaypointResponse:
    route = WaypointRoute.NAV_TO_SOURCE
    return WaypointResponse(
        request_id="planner-probe-1",
        sequence_id=1,
        route=route.value,
        route_token=ROUTE_TOKENS[route],
        action_domain="NAVIGATION",
        subtask="walk to the local source waypoint",
        route_confidence=0.90,
        decision_probs={"ACTION": 0.95, "DONE": 0.05},
        route_probs={
            WaypointRoute.NAV_TO_SOURCE.value: 0.95,
            WaypointRoute.PICK.value: 0.02,
            WaypointRoute.NAV_TO_TARGET.value: 0.02,
            WaypointRoute.PLACE.value: 0.01,
        },
        nav_waypoints_body=((0.60, 0.0, 0.0),),
        arm_targets_base=None,
        action_valid_mask=(True,),
        checkpoint_id="planner-probe-not-a-model",
        normalization_sha256="0" * 64,
        action_units=("m", "m", "rad"),
    )


def _clean_commit(root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("arm-vla reference worktree is dirty")
    return commit


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def _wrap(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _console_summary(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "report": str(path),
        "arm_vla_reference_commit": report.get("arm_vla_reference_commit"),
        "known_waypoint_body": report.get("known_waypoint_body"),
        "predicted_goal_world": report.get("predicted_goal_world"),
        "pct_path_point_count": len(report.get("pct_path_world") or ()),
        "pct_snap_distance_m": report.get("pct_snap_distance_m"),
        "fallback_allowed": report.get("fallback_allowed"),
        "control_step_count": report.get("control_step_count"),
        "all_dwa_commands_finite_and_bounded": report.get(
            "all_dwa_commands_finite_and_bounded"
        ),
        "terminal_reason": report.get("terminal_reason"),
        "requery_after_first_waypoint": report.get("requery_after_first_waypoint"),
        "error": report.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
