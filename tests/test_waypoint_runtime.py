from types import SimpleNamespace

import pytest

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    CAMERA_CALIBRATION_ID,
    ROUTE_TOKENS,
    WaypointRoute,
    waypoint_prompt,
)
from conveyor_bench.conveyorvla.waypoint_execution import (
    ArmPlan,
    CuRoboIKRecedingHorizonExecutor,
    NAVIGATION_SAFETY_PROFILE_EXECUTABLE_PREFIX,
    NavigationExecutionConfig,
    PCTDWARecedingHorizonExecutor,
    PCTPlan,
)
from conveyor_bench.conveyorvla.waypoint_model import (
    ConstrainedRouteDecision,
    WaypointPrediction,
)
from conveyor_bench.conveyorvla.waypoint_protocol import (
    RECOVER_ROUTE,
    WaypointProtocolError,
    WaypointRequest,
    WaypointResponse,
)
from conveyor_bench.conveyorvla.waypoint_runtime import WaypointInferenceSession


DECISION_PROBS = {"ACTION": 0.9, "DONE": 0.1}
ROUTE_PROBS = {
    "NAV_TO_SOURCE": 0.7,
    "PICK": 0.1,
    "NAV_TO_TARGET": 0.1,
    "PLACE": 0.1,
}


def _request(sequence=1):
    return {
        "protocol_version": "conveyorvla-waypoint-runtime/v1",
        "request_id": f"request-{sequence}",
        "episode_id": "episode-1",
        "sequence_id": sequence,
        "instruction": "Pick and place the Coke can.",
        "images": {"head": ["h0", "h1"], "wrist": ["w0", "w1"]},
        "camera_calibration_id": CAMERA_CALIBRATION_ID,
    }


def _response(route, rows, *, sequence=1):
    nav = route in {WaypointRoute.NAV_TO_SOURCE, WaypointRoute.NAV_TO_TARGET}
    return WaypointResponse(
        request_id=f"request-{sequence}",
        sequence_id=sequence,
        route=route.value,
        route_token=ROUTE_TOKENS[route],
        action_domain="NAVIGATION" if nav else "MANIPULATION",
        subtask="move now",
        route_confidence=0.63,
        decision_probs=DECISION_PROBS,
        route_probs=ROUTE_PROBS,
        nav_waypoints_body=tuple(rows) if nav else None,
        arm_targets_base=tuple(rows) if not nav else None,
        action_valid_mask=(True,) * len(rows),
        checkpoint_id="step-20",
        normalization_sha256="a" * 64,
        action_units=(
            ("m", "m", "rad")
            if nav
            else ("m", "m", "m", "rad", "rad", "rad", "fraction")
        ),
    )


def test_request_rejects_state_phase_history_and_wrong_temporal_shape():
    request = WaypointRequest.from_mapping(_request())
    assert request.head_images == ("h0", "h1")
    for key in ("state28", "phase", "locked_route", "target_pose", "subtask_history"):
        leaked = _request()
        leaked["metadata"] = {key: [0.0]}
        with pytest.raises(WaypointProtocolError, match="forbidden"):
            WaypointRequest.from_mapping(leaked)
    malformed = _request()
    malformed["images"]["head"] = ["only-current"]
    with pytest.raises(WaypointProtocolError, match="exactly"):
        WaypointRequest.from_mapping(malformed)


def test_response_schema_is_typed_finite_and_expert_exclusive():
    response = _response(
        WaypointRoute.NAV_TO_SOURCE,
        [(0.2, 0.0, 0.0)] * ACTION_HORIZON,
    )
    assert WaypointResponse.from_mapping(response.to_mapping()) == response
    wrong = response.to_mapping()
    wrong["arm_targets_base"] = [[0.2] * 7]
    with pytest.raises(WaypointProtocolError, match="only body"):
        WaypointResponse.from_mapping(wrong)
    nonfinite = response.to_mapping()
    nonfinite["nav_waypoints_body"][0][0] = float("nan")
    with pytest.raises(WaypointProtocolError, match="non-finite"):
        WaypointResponse.from_mapping(nonfinite)


class _Policy:
    def __init__(self, prediction):
        self.prediction = prediction
        self.examples = []

    def predict(self, examples):
        self.examples.extend(examples)
        return (self.prediction,)


class _Normalizer:
    def denormalize(self, _route, action):
        return action


def test_inference_session_uses_only_images_task_and_rejects_replay():
    decision = ConstrainedRouteDecision(
        route=WaypointRoute.NAV_TO_SOURCE,
        assistant_prefix="prefix",
        subtask_text="move now",
        route_confidence=0.63,
        decision_probs=DECISION_PROBS,
        route_probs=ROUTE_PROBS,
        valid=True,
    )
    action = tuple((0.2, 0.0, 0.0) for _ in range(ACTION_HORIZON))
    policy = _Policy(WaypointPrediction(decision, action))
    session = WaypointInferenceSession(
        policy,
        _Normalizer(),
        checkpoint_id="step-20",
        normalization_sha256="a" * 64,
        camera_calibration_id=CAMERA_CALIBRATION_ID,
    )
    result = session.infer(_request())
    assert result.response.route == "NAV_TO_SOURCE"
    assert result.response.nav_waypoints_body == action
    assert set(policy.examples[0]) == {"video", "lang"}
    assert policy.examples[0]["lang"] == waypoint_prompt(_request()["instruction"])
    replay = session.infer(_request())
    assert replay.response.route == RECOVER_ROUTE
    assert replay.response.recover_reason == "stale_or_replayed_sequence"
    assert len(policy.examples) == 1


class _PCT:
    def __init__(self, *, fail=False, snap=0.0):
        self.calls = []
        self.fail = fail
        self.snap = snap

    def plan(self, current, goal):
        self.calls.append((current, goal))
        if self.fail:
            raise RuntimeError("no PCT path")
        return PCTPlan(
            path_world=((current[0], current[1]), (goal[0], goal[1])),
            snapped_goal_world=goal,
            snap_distance_m=self.snap,
        )


class _DWA:
    def __init__(self, command=(0.2, 0.05, 0.3)):
        self.value = command
        self.calls = []

    def command(self, path, pose, velocity, local_map):
        self.calls.append((path, pose, velocity, local_map))
        return self.value


def test_nav_runs_body_to_world_pct_dwa_and_requeries_after_first_waypoint():
    pct, dwa = _PCT(), _DWA()
    executor = PCTDWARecedingHorizonExecutor(pct, dwa)
    response = _response(
        WaypointRoute.NAV_TO_SOURCE,
        [(0.2, 0.0, 0.0)] * ACTION_HORIZON,
    )
    planned = executor.begin(response, (1.0, 2.0, 1.57079632679), now_s=0.0)
    assert not planned.failed
    assert planned.gripper_target == 1.0
    assert pct.calls[0][1] == pytest.approx((1.0, 2.2, 0.0, 1.57079632679))
    command = executor.step((1.0, 2.0, 1.57079632679), (0.0, 0.0, 0.0), object(), now_s=0.1)
    assert command.base_velocity == (0.2, 0.05, 0.3)
    assert command.gripper_target == 1.0
    assert len(dwa.calls) == 1
    reached = executor.step((1.0, 2.2, 1.57079632679), (0.0, 0.0, 0.0), object(), now_s=0.2)
    assert reached.base_velocity == (0.0, 0.0, 0.0)
    assert reached.requires_requery
    assert reached.reason == "first_waypoint_reached"


def test_nav_preserves_base_height_for_pct_local_goal():
    pct = _PCT()
    response = _response(
        WaypointRoute.NAV_TO_TARGET,
        [(0.2, 0.0, 0.0)] * ACTION_HORIZON,
    )
    executor = PCTDWARecedingHorizonExecutor(pct, _DWA())
    planned = executor.begin(
        response,
        (1.0, 2.0, 1.25, 1.0, 0.0, 0.0, 0.0),
        now_s=0.0,
    )
    assert not planned.failed
    assert planned.gripper_target == 0.0
    assert pct.calls[0][0] == (1.0, 2.0, 1.25, 0.0)
    assert pct.calls[0][1] == pytest.approx((1.2, 2.0, 1.25, 0.0))


def test_nav_pct_failure_and_unsafe_dwa_are_zero_speed_fail_closed():
    response = _response(
        WaypointRoute.NAV_TO_TARGET,
        [(0.2, 0.0, 0.0)] * ACTION_HORIZON,
    )
    failed = PCTDWARecedingHorizonExecutor(_PCT(fail=True), _DWA()).begin(
        response, (0.0, 0.0, 0.0), now_s=0.0
    )
    assert failed.failed and failed.base_velocity == (0.0, 0.0, 0.0)

    executor = PCTDWARecedingHorizonExecutor(_PCT(), _DWA((99.0, 0.0, 0.0)))
    assert not executor.begin(response, (0.0, 0.0, 0.0), now_s=0.0).failed
    unsafe = executor.step((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), object(), now_s=0.1)
    assert unsafe.failed and unsafe.base_velocity == (0.0, 0.0, 0.0)


def test_nav_diagnostic_profile_executes_only_a_legal_prefix_and_reports_bad_tail():
    rows = [(0.2, 0.0, 0.0)] * ACTION_HORIZON
    rows[5] = (1.2, 0.0, 0.0)
    response = _response(WaypointRoute.NAV_TO_SOURCE, rows)
    strict = PCTDWARecedingHorizonExecutor(_PCT(), _DWA()).begin(
        response, (0.0, 0.0, 0.0), now_s=0.0
    )
    assert strict.failed
    assert "segment 5 exceeds translation limit" in strict.reason

    diagnostic = PCTDWARecedingHorizonExecutor(
        _PCT(),
        _DWA(),
        NavigationExecutionConfig(
            safety_profile=NAVIGATION_SAFETY_PROFILE_EXECUTABLE_PREFIX
        ),
    ).begin(response, (0.0, 0.0, 0.0), now_s=0.0)
    assert not diagnostic.failed
    assert diagnostic.trace["selected_waypoint_index"] == 0
    assert diagnostic.trace["full_horizon_contract_passed"] is False
    assert "segment 5 exceeds translation limit" in diagnostic.trace[
        "full_horizon_violation"
    ]


def test_pure_rotation_bypasses_pct_and_uses_bounded_terminal_yaw():
    pct = _PCT()
    response = _response(
        WaypointRoute.NAV_TO_SOURCE,
        [(0.0, 0.0, 0.2)] * ACTION_HORIZON,
    )
    executor = PCTDWARecedingHorizonExecutor(pct, _DWA())
    executor.begin(response, (0.0, 0.0, 0.0), now_s=0.0)
    command = executor.step((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), object(), now_s=0.1)
    assert pct.calls == []
    assert command.base_velocity[:2] == (0.0, 0.0)
    assert 0.0 < command.base_velocity[2] <= 0.6


def test_nav_bypasses_pct_inside_position_tolerance_and_uses_terminal_yaw():
    pct, dwa = _PCT(), _DWA()
    response = _response(
        WaypointRoute.NAV_TO_SOURCE,
        [(0.04, 0.0, 0.3)] * ACTION_HORIZON,
    )
    executor = PCTDWARecedingHorizonExecutor(pct, dwa)
    planned = executor.begin(
        response, (0.0, 0.0, 0.0), now_s=0.0
    )
    assert not planned.failed
    assert planned.trace["planner"] == "terminal_yaw"
    assert pct.calls == []
    command = executor.step(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        object(),
        now_s=4.0,
    )
    assert not command.failed
    assert command.base_velocity[:2] == (0.0, 0.0)
    assert 0.0 < command.base_velocity[2] <= 0.6
    assert command.trace["control_mode"] == "terminal_yaw"
    assert dwa.calls == []


class _ArmPlanner:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def plan(self, joints, target, scene):
        self.calls.append((joints, target, scene))
        if self.fail:
            raise RuntimeError("unreachable")
        return ArmPlan(
            joint_path=((0.0, 0.0), (0.1, -0.1)),
            planner="curobo",
            reachable=True,
            collision_free=True,
            target_position_error_m=0.001,
            target_orientation_error_rad=0.01,
        )


class _ArmController:
    def __init__(self):
        self.plan = None
        self.gripper = None
        self.done = False

    def reset(self, plan, gripper):
        self.plan, self.gripper = plan, gripper

    def command(self, _measured):
        return (0.1, -0.1), self.done


def test_arm_plans_first_absolute_tcp_target_holds_base_and_requeries():
    rows = [(0.31, 0.0, 0.2, 0.0, 0.0, 0.0, 0.25)] * ACTION_HORIZON
    response = _response(WaypointRoute.PICK, rows)
    planner, controller = _ArmPlanner(), _ArmController()
    executor = CuRoboIKRecedingHorizonExecutor(planner, controller)
    planned = executor.begin(
        response,
        (0.30, 0.0, 0.2, 0.0, 0.0, 0.0),
        (0.0, 0.0),
        {"collision": "scene"},
        now_s=0.0,
    )
    assert not planned.failed and planned.base_velocity == (0.0, 0.0, 0.0)
    assert planner.calls[0][1] == rows[0][:6]
    moving = executor.step((0.0, 0.0), now_s=0.1)
    assert moving.arm_joint_target == (0.1, -0.1)
    assert moving.gripper_target == 0.25
    assert moving.base_velocity == (0.0, 0.0, 0.0)
    controller.done = True
    reached = executor.step((0.1, -0.1), now_s=0.2)
    assert reached.requires_requery and not reached.failed


def test_arm_unreachable_is_zero_action_fail_closed():
    rows = [(0.31, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)] * ACTION_HORIZON
    executor = CuRoboIKRecedingHorizonExecutor(_ArmPlanner(fail=True), _ArmController())
    failed = executor.begin(
        _response(WaypointRoute.PLACE, rows),
        (0.30, 0.0, 0.2, 0.0, 0.0, 0.0),
        (0.0, 0.0),
        object(),
        now_s=0.0,
    )
    assert failed.failed
    assert failed.base_velocity == (0.0, 0.0, 0.0)
    assert failed.arm_joint_target is None
