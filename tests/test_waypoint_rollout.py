import base64
import io

import numpy as np
import pytest
from PIL import Image

from scripts.run_waypoint_rollout import (
    _ExternalWaypointCuRoboLifecycle,
    _initial_source_front_sector_report,
    _simulation_curobo_safety_gate,
    build_parser,
)

from conveyor_bench.conveyorvla.waypoint import CAMERA_CALIBRATION_ID
from conveyor_bench.conveyorvla.waypoint_rollout import (
    TemporalJPEGBuffer,
    WaypointHTTPClient,
    measured_arm_joints,
    measured_body_velocity,
    planner_base_from_query_base,
    tcp_pose_in_query_base,
    waypoint_request_from_frames,
)


def _image(value):
    return np.full((8, 12, 3), value, dtype=np.uint8)


def test_rollout_defaults_to_the_full_horizon_contract_safety_profile():
    parser = build_parser()
    assert parser.get_default("navigation_safety_profile") == "contract"
    choices = next(
        action.choices
        for action in parser._actions
        if action.dest == "navigation_safety_profile"
    )
    assert tuple(choices) == (
        "contract",
        "arm-vla-reference",
        "lookahead-arm-vla-reference",
        "executable-prefix-diagnostic",
        "unbounded-translation-diagnostic",
    )
    assert parser.get_default("require_initial_source_visible") is True
    assert parser.get_default("initial_source_max_bearing_deg") == 30.0


def test_initial_source_visibility_requires_front_rgb_and_frontal_bearing():
    report = _initial_source_front_sector_report(
        (0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0),
        (1.0, 0.2, 0.2, 1.0, 0.0, 0.0, 0.0),
        {"front": _image(1)},
        max_bearing_deg=30.0,
    )
    assert report["passed"]
    assert report["source_truth_sent_to_model"] is False

    side = _initial_source_front_sector_report(
        (0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.0),
        {"front": _image(1)},
        max_bearing_deg=30.0,
    )
    assert not side["passed"]


def test_temporal_buffer_requires_exact_synchronized_point_two_second_pair():
    buffer = TemporalJPEGBuffer(separation_steps=10, jpeg_quality=90)
    assert not buffer.add(0, {"front": _image(1)})
    assert buffer.add(0, {"front": _image(1), "wrist": _image(2)})
    assert buffer.add(9, {"front": _image(3), "wrist": _image(4)})
    assert buffer.pair_after(None) is None
    assert buffer.add(10, {"front": _image(5), "wrist": _image(6)})
    pair = buffer.pair_after(None)
    assert pair is not None and (pair[0].step_index, pair[1].step_index) == (0, 10)
    assert buffer.pair_after(10) is None
    for frame in pair:
        for encoded in (frame.head_jpeg_base64, frame.wrist_jpeg_base64):
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                assert image.format == "JPEG" and image.size == (12, 8)


def test_rollout_request_contains_only_task_and_four_images():
    buffer = TemporalJPEGBuffer(separation_steps=10)
    buffer.add(0, {"front": _image(1), "wrist": _image(2)})
    buffer.add(10, {"front": _image(3), "wrist": _image(4)})
    pair = buffer.pair_after(None)
    assert pair is not None
    request = waypoint_request_from_frames(
        episode_id="episode-7",
        sequence_id=3,
        instruction="Pick and place the can.",
        frames=pair,
    )
    payload = request.to_mapping()
    assert set(payload) == {
        "protocol_version",
        "request_id",
        "episode_id",
        "sequence_id",
        "instruction",
        "images",
        "camera_calibration_id",
    }
    assert set(payload["images"]) == {"head", "wrist"}
    assert len(payload["images"]["head"]) == len(payload["images"]["wrist"]) == 2
    assert payload["camera_calibration_id"] == CAMERA_CALIBRATION_ID
    forbidden = {"state", "phase", "operation", "history", "locked_route"}
    assert forbidden.isdisjoint(payload)


def test_query_tcp_pose_and_planner_transform_use_live_executor_frames():
    half_sqrt = 2.0**-0.5
    root = (1.0, 2.0, 3.0, half_sqrt, 0.0, 0.0, half_sqrt)
    tcp = (0.9, 2.3, 3.2, half_sqrt, 0.0, 0.0, half_sqrt)
    assert tcp_pose_in_query_base(root, tcp) == pytest.approx(
        (0.3, 0.1, 0.2, 0.0, 0.0, 0.0)
    )
    world_from_planner = np.eye(4)
    world_from_planner[:3, 3] = (0.5, 1.0, 2.0)
    transform = planner_base_from_query_base(world_from_planner, root)
    assert transform["position_xyz"] == pytest.approx([0.5, 1.0, 1.0])
    assert transform["quaternion_wxyz"] == pytest.approx(
        [half_sqrt, 0.0, 0.0, half_sqrt]
    )


def test_executor_state_helpers_are_named_finite_and_body_relative():
    assert measured_arm_joints(
        ("leg", "arm_joint2", "arm_joint1"),
        (9.0, 0.2, 0.1),
        ("arm_joint1", "arm_joint2"),
    ) == (0.1, 0.2)
    state = type(
        "State",
        (),
        {
            "metadata": {},
            "robot_root_pose": (0.0, 0.0, 0.0, 2.0**-0.5, 0.0, 0.0, 2.0**-0.5),
            "robot_root_velocity": (0.0, 1.0, 0.0, 0.0, 0.0, 0.25),
        },
    )()
    assert measured_body_velocity(state) == pytest.approx((1.0, 0.0, 0.25))


def test_model_client_rejects_non_loopback_or_non_http_endpoints():
    for endpoint in (
        "https://127.0.0.1:18081",
        "http://example.com:18081",
        "ws://127.0.0.1:18081",
    ):
        with pytest.raises(ValueError, match="loopback HTTP"):
            WaypointHTTPClient(endpoint)


def test_simulation_curobo_gate_is_independent_and_fail_closed():
    request = {"deployment": "simulation", "target_frame": "query-base-B_t"}
    response = {
        "reachable": True,
        "collision_free": True,
        "target_position_error_m": 0.001,
        "target_orientation_error_rad": 0.01,
    }
    assert _simulation_curobo_safety_gate(request, response)
    assert not _simulation_curobo_safety_gate(
        request, {**response, "collision_free": False}
    )


def test_reference_lifecycle_reuses_only_the_gated_waypoint_curobo_service():
    requests = []

    def transport(request):
        requests.append(dict(request))
        if request["command"] == "ping":
            return {
                "ok": True,
                "arm_vla_reference_commit": (
                    "388b6818f4c605a707d13c519fbb58b1d07acd92"
                ),
            }
        return {
            "ok": True,
            "arm_vla_reference_commit": (
                "388b6818f4c605a707d13c519fbb58b1d07acd92"
            ),
            "features": {
                "direct_absolute_tcp_target": True,
                "input_target_frame": "query-base-B_t",
                "planner_target_frame": "curobo-planner-base",
                "orientation_fallback": False,
                "world_collision": True,
            },
        }

    lifecycle = _ExternalWaypointCuRoboLifecycle(
        object(), port=8766, timeout_s=1.0, transport=transport
    )
    lifecycle.start()
    assert lifecycle.wait_until_ready()
    assert requests == [{"command": "ping"}, {"command": "capabilities"}]
    assert lifecycle.start_report["started"] is False
    assert lifecycle.start_report["reused_existing"] is True
    lifecycle.close()
    assert lifecycle.start_report["external_server_preserved"] is True


def test_reference_lifecycle_fails_closed_on_legacy_curobo_capabilities():
    def transport(request):
        return {
            "ok": True,
            "arm_vla_reference_commit": (
                "388b6818f4c605a707d13c519fbb58b1d07acd92"
            ),
            "features": {} if request["command"] == "capabilities" else None,
        }

    lifecycle = _ExternalWaypointCuRoboLifecycle(
        object(), port=8766, timeout_s=1.0, transport=transport
    )
    with pytest.raises(RuntimeError, match="capability gate"):
        lifecycle.start()
    assert not lifecycle.wait_until_ready()
