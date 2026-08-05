from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_V1_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "runtime_v1.py"
)
RUNTIME_V2_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "runtime_v2.py"
)
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_benchmark_v2.py"


def _class_source(path: Path, class_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.unparse(node)


def test_shared_runtime_sequences_service_gated_targets_without_v1_fork() -> None:
    source = RUNTIME_V1_PATH.read_text(encoding="utf-8")

    assert "SequentialTargetCoordinator" in source
    assert 'approach_stage = "sequential_rearm"' in source
    assert "_spawn_not_before_s" in source
    assert "resolved.select_target" in source
    assert "EventKind.TARGET_SELECTED" in source
    assert "EventKind.OBJECT_PLACED" in source


def test_v2_runtime_wires_offline_scenes_and_remote_navigation_contract() -> None:
    source = _class_source(RUNTIME_V2_PATH, "ConveyorRuntimeV2")

    assert "ConveyorNearSortV2SceneCfg" in source
    assert "ConveyorRemoteDeliverySceneCfg" in source
    assert "V2_ASSET_LOCK_PATH" in source
    assert "max(-0.2, vy)" in source
    assert "return 'navigate'" in source
    assert "_mobile_continue_carry_before_place" in source
    assert "return 0.008" in source
    assert "_object_crossed_task_exit" in source
    assert "self._ever_held_target" in source
    assert "delivery_root_goal_xy_m" in source
    assert "delivery_goal_yaw_rad" in source
    assert "return 0.0" in source
    assert "_mobile_navigation_drive_heading_tolerance_rad" in source
    assert "return 0.18" in source
    assert "_mobile_navigation_yaw_command" in source
    assert "return super()._mobile_navigation_yaw_command" in source
    assert "_remote_navigation_errors" in source
    assert "math.copysign(0.3, along_track)" in source
    assert "1.5 * cross_track" in source
    assert "min(0.2" in source
    assert "1.5 * yaw_error_rad" in source
    assert "min(0.35" in source
    assert "return 0.09" in source
    assert "return 0.12" in source
    assert source.count("return 0.21") == 2
    assert "return 50.0" in source
    assert "'turn': 10.0" in source


def test_shared_runtime_reseeds_ik_between_service_gated_targets() -> None:
    source = RUNTIME_V1_PATH.read_text(encoding="utf-8")

    assert 'if approach_stage == "sequential_rearm"' in source
    assert "measured_arm = self.robot.data.joint_pos" in source
    assert "self._arm_ik_seed = tuple(" in source
    assert "self._last_ik_iterations = 0" in source


def test_v2_dry_run_resolves_two_target_task_without_starting_isaac() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            "--scene",
            "transverse_near_sort_v2",
            "--task-family",
            "continuous_multi_target",
            "--robot-mode",
            "fixed_base",
            "--seed",
            "7",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["simulator_started"] is False
    assert payload["canonical_protocol"] == "conveyor-bench-v1"
    assert payload["task"]["task_type"] == "continuous_sort"
    assert len(payload["task"]["scored_object_ids"]) == 2


def test_v2_preflight_rejects_invalid_remote_fixed_base_before_isaac() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            "--scene",
            "mobile_remote_delivery_v2",
            "--task-family",
            "single_target",
            "--robot-mode",
            "fixed_base",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "remote delivery requires whole_body_policy" in completed.stderr


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("--episodes", "0", "episodes must be a positive integer"),
        ("--seed", "-1", "seed must be a non-negative integer"),
        ("--max-duration", "0", "max-duration must be finite and positive"),
        ("--max-duration", "nan", "max-duration must be finite and positive"),
    ),
)
def test_v2_dry_run_rejects_invalid_collection_scalars_before_isaac(
    option: str,
    value: str,
    message: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            option,
            value,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert completed.stdout == ""


def test_v2_dry_run_rejects_unknown_options_and_too_short_horizon() -> None:
    typo = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            "--episdoes",
            "99",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert typo.returncode == 2
    assert "unrecognized dry-run arguments" in typo.stderr

    too_short = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            "--max-duration",
            "0.001",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert too_short.returncode == 2
    assert "task initialization horizon" in too_short.stderr

    boundary = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--dry-run-task",
            "--max-duration",
            "1.25",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert boundary.returncode == 2
    assert "exceed the task initialization horizon" in boundary.stderr
