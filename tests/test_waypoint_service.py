import base64
import io

import pytest

Image = pytest.importorskip("PIL.Image")

from scripts import export_waypoint_inference as exporter
from scripts import serve_waypoint as service

from conveyor_bench.conveyorvla.waypoint import CAMERA_CALIBRATION_ID
from conveyor_bench.conveyorvla.waypoint_protocol import WaypointProtocolError


def _encoded_image(*, fmt="JPEG"):
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), (30, 60, 90)).save(stream, format=fmt)
    return base64.b64encode(stream.getvalue()).decode()


def _payload():
    image = _encoded_image()
    return {
        "protocol_version": "conveyorvla-waypoint-runtime/v1",
        "request_id": "request-1",
        "episode_id": "episode-1",
        "sequence_id": 1,
        "instruction": "Pick and place the Coke can.",
        "images": {"head": [image, image], "wrist": [image, image]},
        "camera_calibration_id": CAMERA_CALIBRATION_ID,
    }


def test_waypoint_http_decoder_accepts_only_four_jpegs_and_no_state():
    request = service._decode_request(_payload())
    assert request.head_images[0].size == (8, 6)
    assert request.head_images[0].mode == "RGB"

    leaked = _payload()
    leaked["state28"] = [0.0] * 28
    with pytest.raises(WaypointProtocolError, match="keys are not exact"):
        service._decode_request(leaked)

    png = _payload()
    png["images"]["wrist"][1] = _encoded_image(fmt="PNG")
    with pytest.raises(WaypointProtocolError, match="JPEG"):
        service._decode_request(png)


def test_inference_export_is_outside_git_and_requires_sharded_safe_weights(tmp_path):
    output = tmp_path / "export"
    exporter._reserve_export(output)
    with pytest.raises(Exception, match="already exists"):
        exporter._reserve_export(output)

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (weights / "model-00001-of-00001.safetensors").write_bytes(b"safe")
    identity = exporter._safe_weight_identity(weights)
    assert set(identity) == {
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    }
