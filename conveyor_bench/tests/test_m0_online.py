from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import threading
from pathlib import Path

import numpy as np
import pytest

from conveyor_bench.m0_mobile import M0MobileNormalizer, load_m0_mobile_config
from conveyor_bench.m0_online import (
    ONLINE_SCHEMA_VERSION,
    M0OnlineClient,
    M0OnlineError,
    build_live_state28,
    encode_rgb_jpeg,
    guard_pregrasp_tcp_target,
    health_payload,
    make_infer_response,
    parse_infer_request,
    project_action_chunk,
    quantize_go2_forward_intent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = PROJECT_ROOT / "scripts" / "serve_m0_mobile.py"
SPEC = importlib.util.spec_from_file_location("serve_m0_mobile", SERVER_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)

MODEL_IDENTITY = {
    "action_model_sha256": "a" * 64,
    "state_statistics_sha256": "b" * 64,
    "training_report_sha256": "c" * 64,
    "training_steps": 10,
    "dataset_records": 3,
}


def _normalizer() -> M0MobileNormalizer:
    return M0MobileNormalizer.from_config(
        load_m0_mobile_config(), {"mean": [0.0] * 28, "std": [1.0] * 28}
    )


def _jpeg() -> bytes:
    image_module = pytest.importorskip("PIL.Image")
    return encode_rgb_jpeg(image_module.new("RGB", (8, 6), (20, 40, 60)))


def _payload() -> dict:
    encoded = base64.b64encode(_jpeg()).decode("ascii")
    return {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "request_id": "episode-1:tick-2",
        "sequence_id": 2,
        "instruction": "pick the moving red block",
        "state28": [0.0] * 28,
        "images": [
            {"camera_id": camera, "encoding": "jpeg", "data_base64": encoded}
            for camera in ("head_rgb", "wrist_rgb")
        ],
        "seed": 17,
    }


def test_live_state_builder_matches_export_layout_and_quaternion_rotation() -> None:
    half = math.sqrt(0.5)
    state = build_live_state28(
        np.asarray((1, 2, 3), dtype=np.float32),
        (4, 5, 6),
        (0, 0, -1),
        range(6),
        range(6, 12),
        (0.3, 0.0, 0.4),
        (half, 0.0, 0.0, half),
        0.25,
    )

    assert len(state) == 28
    assert state[:9] == pytest.approx((1, 2, 3, 4, 5, 6, 0, 0, -1))
    assert state[21:27] == pytest.approx((0.3, 0.0, 0.4, 0.0, 0.0, math.pi / 2))
    assert state[27] == pytest.approx(0.25)
    with pytest.raises(M0OnlineError, match="unit length"):
        build_live_state28((0,) * 3, (0,) * 3, (0, 0, -1), (0,) * 6, (0,) * 6, (0,) * 3, (2, 0, 0, 0), 1.0)


def test_request_parser_is_strict_about_schema_order_and_finite_state() -> None:
    request = parse_infer_request(_payload())
    assert request.request_id == "episode-1:tick-2"
    assert request.sequence_id == 2
    assert len(request.jpeg_images) == 2

    wrong_order = _payload()
    wrong_order["images"] = list(reversed(wrong_order["images"]))
    with pytest.raises(M0OnlineError, match="ordered"):
        parse_infer_request(wrong_order)

    nonfinite = _payload()
    nonfinite["state28"][0] = math.nan
    with pytest.raises(M0OnlineError, match="finite"):
        parse_infer_request(nonfinite)

    extra = _payload()
    extra["observer_image"] = "forbidden"
    with pytest.raises(M0OnlineError, match="fields must be exactly"):
        parse_infer_request(extra)


def test_action_projection_clamps_hard_zeros_and_binarizes_gripper() -> None:
    row = [2.0, -0.7, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49]
    normalized, physical = project_action_chunk([row] * 16, _normalizer())

    assert normalized[0] == pytest.approx((1.0, 0.0, -1.0, 0.5, 0, 0, 0, 0, 0, 0))
    assert physical[0] == pytest.approx((0.3, 0.0, -0.35, 0.0125, 0, 0, 0, 0, 0, 0))
    open_row = list(row)
    open_row[9] = 0.5
    _, open_actions = project_action_chunk([open_row] * 16, _normalizer())
    assert open_actions[0][9] == 1.0

    bad = [list(row) for _ in range(16)]
    bad[3][4] = math.inf
    with pytest.raises(M0OnlineError, match="finite"):
        project_action_chunk(bad, _normalizer())


def test_go2_forward_intent_uses_only_the_audited_speed_primitive() -> None:
    assert quantize_go2_forward_intent((0.079, 0.2, 0.1)) == pytest.approx(
        (0.0, 0.0, 0.1)
    )
    assert quantize_go2_forward_intent((0.12, -0.2, 0.1)) == pytest.approx(
        (0.16, 0.0, 0.1)
    )
    assert quantize_go2_forward_intent((0.2, 0.0, 0.1)) == pytest.approx(
        (0.2, 0.0, 0.1)
    )


def test_pregrasp_workspace_guard_only_clamps_audited_drift_directions() -> None:
    inside, inside_axes = guard_pregrasp_tcp_target((0.60, -0.04, 0.27))
    assert inside == pytest.approx((0.60, -0.04, 0.27))
    assert inside_axes == ()

    guarded, axes = guard_pregrasp_tcp_target((0.70, -0.10, 0.20))
    assert guarded == pytest.approx((0.622, -0.060, 0.250))
    assert axes == ("x", "y", "z")

    unbounded, unbounded_axes = guard_pregrasp_tcp_target((-1.0, 1.0, 1.0))
    assert unbounded == pytest.approx((-1.0, 1.0, 1.0))
    assert unbounded_axes == ()


class _FakeService:
    def health(self):
        return health_payload(MODEL_IDENTITY)

    def infer(self, request):
        return make_infer_response(request, [[2.0] * 10] * 16, 12.5)


def test_localhost_server_and_client_round_trip() -> None:
    server = SERVER.create_server(0, _FakeService())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        client = M0OnlineClient(
            f"http://127.0.0.1:{server.server_port}",
            2.0,
            normalizer=_normalizer(),
        )
        assert client.health()["status"] == "ready"
        assert client.health()["model"] == MODEL_IDENTITY
        result = client.infer(
            np.zeros((6, 8, 4), dtype=np.uint8),
            _jpeg(),
            "pick the moving red block",
            [0.0] * 28,
            sequence_id=9,
            request_id="smoke-9",
            seed=5,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert result.request_id == "smoke-9"
    assert result.sequence_id == 9
    assert result.server_inference_ms == pytest.approx(12.5)
    assert result.round_trip_ms >= 0.0
    assert len(result.normalized_actions) == 16
    assert result.normalized_actions[0][1] == 0.0
    assert result.physical_actions[0][0] == pytest.approx(0.3)
    assert result.physical_actions[0][9] == 1.0


def test_client_rejects_non_loopback_endpoint() -> None:
    with pytest.raises(M0OnlineError, match="127.0.0.1"):
        M0OnlineClient("http://0.0.0.0:18080")


def test_server_pins_action_and_state_artifacts_to_training_report(
    tmp_path,
) -> None:
    action = tmp_path / "action.safetensors"
    statistics = tmp_path / "state.json"
    report = tmp_path / "training_report.json"
    action.write_bytes(b"trained-action")
    statistics.write_bytes(b"trained-state")
    report.write_text(
        json.dumps(
            {
                "ok": True,
                "action_model_sha256": hashlib.sha256(
                    action.read_bytes()
                ).hexdigest(),
                "state_statistics_sha256": hashlib.sha256(
                    statistics.read_bytes()
                ).hexdigest(),
                "max_steps": 10,
                "dataset_records": 3,
            }
        ),
        encoding="utf-8",
    )

    verified = SERVER._verify_training_artifacts(
        action, statistics, report
    )

    assert verified["training_steps"] == 10
    assert verified["training_report_sha256"] == hashlib.sha256(
        report.read_bytes()
    ).hexdigest()
    action.write_bytes(b"tampered")
    with pytest.raises(M0OnlineError, match="action checkpoint SHA-256"):
        SERVER._verify_training_artifacts(action, statistics, report)
