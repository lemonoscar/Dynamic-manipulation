#!/usr/bin/env python3
"""Data-free startup check for the joint-trajectory Isaac system wiring."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.joint_trajectory import (  # noqa: E402
    JointTrajectoryRoute,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (
    DirectJointCommand,
    navigation_reference,
)
from conveyor_bench.conveyorvla.joint_trajectory_system import (  # noqa: E402
    IsaacJointActionAdapter,
    PCTDWAJointNavigationExecutor,
    PlacementValidArea,
)
from conveyor_bench.conveyorvla.waypoint_execution import PCTPlan  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (  # noqa: E402
    APPROVED_ARM_VLA_COMMIT,
)


DEFAULT_ANNOTATION = Path("tasks/liangzhu_placement_target.json")


class _PCT:
    def __init__(self) -> None:
        self.goal = None

    def plan(self, current, goal):
        self.goal = tuple(goal)
        return PCTPlan(
            path_world=((current[0], current[1]), (goal[0], goal[1])),
            snapped_goal_world=tuple(goal),
            snap_distance_m=0.0,
            metadata={"planner": "pct", "startup_fixture": True},
        )


class _DWA:
    def __init__(self) -> None:
        self.called = False

    def command(self, _path, _pose, _velocity, _local_map):
        self.called = True
        return (0.22, 0.0, 0.10)


def _git(reference: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(reference), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_reference_types(reference: Path):
    sys.path.insert(0, str(reference))
    try:
        simulation = importlib.import_module("source.interfaces.simulation")
        isaac_runtime = importlib.import_module("source.simulation.isaaclab_runtime")
    finally:
        sys.path.pop(0)
    return simulation.RobotAction, isaac_runtime.IsaacLabNavigationRuntime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    arguments = parser.parse_args(argv)
    reference = arguments.reference_root.expanduser().resolve()
    if not reference.is_dir():
        raise SystemExit(f"approved reference is missing: {reference}")
    head = _git(reference, "rev-parse", "HEAD")
    dirty = _git(reference, "status", "--porcelain")
    if head != APPROVED_ARM_VLA_COMMIT:
        raise SystemExit(f"approved reference commit mismatch: {head}")
    if dirty:
        raise SystemExit("approved reference worktree is dirty")

    RobotAction, IsaacLabNavigationRuntime = _load_reference_types(reference)
    required_methods = {"read", "apply", "step"}
    if any(not callable(getattr(IsaacLabNavigationRuntime, name, None)) for name in required_methods):
        raise SystemExit("approved Isaac runtime interface is incomplete")

    adapter = IsaacJointActionAdapter(RobotAction)
    command = DirectJointCommand(0, (0.0,) * 6, 0.37)
    mani_action = adapter.manipulation(
        command,
        route=JointTrajectoryRoute.PICK,
        sequence_id=2,
    )
    if mani_action.base_velocity != (0.0, 0.0, 0.0):
        raise SystemExit("Mani startup action did not keep base command zero")
    if mani_action.gripper_command != "hold":
        raise SystemExit("continuous gripper startup action became binary")

    pct, dwa = _PCT(), _DWA()
    navigation = PCTDWAJointNavigationExecutor(pct, dwa)
    reference_action = navigation_reference(
        [[0.1 * (index + 1), 0.0, 0.0] for index in range(10)]
    )
    plan = navigation.begin(
        reference_action,
        (0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
        timestamp_s=0.0,
    )
    control = navigation.command(
        (0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        {"startup_fixture": True},
        timestamp_s=0.0,
    )
    if len(plan.reference_world) != 10 or pct.goal is None or not dwa.called:
        raise SystemExit("NAV startup fixture did not reach PCT and DWA")
    if control.base_velocity != (0.22, 0.0, 0.10):
        raise SystemExit("NAV locomotion envelope changed its valid startup command")

    annotation_path = arguments.annotation
    if not annotation_path.is_absolute():
        annotation_path = reference / annotation_path
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    raw_task = annotation.get("task_overrides")
    valid_area = PlacementValidArea.from_raw_task(raw_task)

    report = {
        "status": "startup_wiring_ready",
        "approved_reference": {
            "root": str(reference),
            "head": head,
            "clean": True,
        },
        "checks": {
            "reference_robot_action_import": True,
            "reference_isaac_runtime_import": True,
            "direct_arm_joint_target": len(mani_action.arm_joint_positions) == 6,
            "continuous_gripper_metadata": list(
                mani_action.metadata["gripper_joint_positions"]
            ),
            "mani_base_exact_zero": True,
            "ik_or_curobo_used": False,
            "nav_reference_points": len(plan.reference_world),
            "pct_endpoint_index": plan.trace["pct_endpoint_reference_index"],
            "dwa_command_guarded": list(control.base_velocity),
            "placement_valid_area_loaded": [
                valid_area.x_min,
                valid_area.x_max,
                valid_area.y_min,
                valid_area.y_max,
            ],
        },
        "data_available": False,
        "dataset_materialized": False,
        "training_started": False,
        "actual_isaac_stage_started": False,
        "next_gate": "approved Isaac stage boot plus one hold/NAV/Mani control smoke",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
