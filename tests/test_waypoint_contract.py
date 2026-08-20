import math

import pytest

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    PRED_DONE_TOKEN,
    ArmTargetSafety,
    WaypointRoute,
    arm_target_base,
    arm_target_world,
    canonical_solution,
    nav_waypoint_body,
    nav_waypoint_world,
    rpy_to_quaternion,
    select_navigation_waypoint,
    validate_arm_targets,
    waypoint_prompt,
    wrap_to_pi,
)


def _quaternion_distance(left, right):
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, dot))


def test_waypoint_route_syntax_and_prompt_are_frozen() -> None:
    prompt = waypoint_prompt("Pick and place the Coke can.")
    assert "state" not in prompt.lower()
    assert "previous" not in prompt.lower()
    assert prompt.count("Task:") == 1
    assert canonical_solution(WaypointRoute.DONE) == PRED_DONE_TOKEN
    assert canonical_solution(WaypointRoute.PICK) == (
        "<|pred_action|><|route_pick|><|subtask|>"
        "Move the gripper to grasp, lift, and retract the Coke can."
        "<|end_subtask|>"
    )


def test_nav_world_body_round_trip_is_exact() -> None:
    query = (1.2, -0.4, 0.3, *rpy_to_quaternion(0.02, -0.03, 1.1))
    target = (1.7, 0.2, 0.3, *rpy_to_quaternion(0.0, 0.0, -2.8))
    waypoint = nav_waypoint_body(query, target)
    reconstructed = nav_waypoint_world(query, waypoint)
    assert reconstructed[:2] == pytest.approx(target[:2], abs=1.0e-12)
    assert wrap_to_pi(reconstructed[2] + 2.8) == pytest.approx(0.0, abs=1.0e-12)


def test_absolute_tcp_target_round_trip_uses_query_base_frame() -> None:
    query = (0.2, 1.0, 0.3, *rpy_to_quaternion(0.04, -0.02, 0.7))
    target = (0.55, 1.1, 0.72, *rpy_to_quaternion(0.4, -0.3, 1.2))
    target_base = arm_target_base(query, target, 0.75)
    reconstructed = arm_target_world(query, target_base)
    assert reconstructed[:3] == pytest.approx(target[:3], abs=1.0e-12)
    assert _quaternion_distance(reconstructed[3:], target[3:]) < 1.0e-12
    assert target_base[-1] == 0.75


def test_runtime_selects_first_non_degenerate_nav_waypoint_and_fails_closed() -> None:
    chunk = [[0.0, 0.0, 0.0] for _ in range(ACTION_HORIZON)]
    chunk[1] = [0.04, 0.0, 0.0]
    chunk[2] = [0.10, 0.0, 0.0]
    index, selected = select_navigation_waypoint(chunk, [True, True, True] + [False] * 17)
    assert index == 1
    assert selected == (0.04, 0.0, 0.0)

    chunk[2] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="translation limit"):
        select_navigation_waypoint(chunk, [True, True, True] + [False] * 17)
    index, selected = select_navigation_waypoint(
        chunk,
        [True, True, True] + [False] * 17,
        validate_full_horizon=False,
    )
    assert index == 1
    assert selected == (0.04, 0.0, 0.0)


def test_arm_gate_validates_absolute_targets_without_state_input() -> None:
    chunk = [[0.30, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0] for _ in range(ACTION_HORIZON)]
    accepted = validate_arm_targets(
        chunk,
        [True] + [False] * 19,
        (0.28, 0.0, 0.35, 0.0, 0.0, 0.0),
    )
    assert accepted == (tuple(chunk[0]),)
    with pytest.raises(ValueError, match="workspace"):
        validate_arm_targets(
            [[0.90, *row[1:]] for row in chunk],
            [True] + [False] * 19,
            (0.28, 0.0, 0.35, 0.0, 0.0, 0.0),
            safety=ArmTargetSafety(),
        )
