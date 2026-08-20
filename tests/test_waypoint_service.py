import base64
import io
import json

import pytest

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")

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


def test_inference_export_clones_tied_weights_before_safetensors(tmp_path):
    from safetensors.torch import load_file

    source = tmp_path / "pytorch"
    destination = tmp_path / "safe"
    source.mkdir()
    tied = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    torch.save(
        {"embed_tokens.weight": tied, "lm_head.weight": tied},
        source / "pytorch_model-00001-of-00001.bin",
    )
    (source / "pytorch_model.bin.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": tied.numel() * tied.element_size() * 2},
                "weight_map": {
                    "embed_tokens.weight": "pytorch_model-00001-of-00001.bin",
                    "lm_head.weight": "pytorch_model-00001-of-00001.bin",
                },
            }
        ),
        encoding="utf-8",
    )

    exporter._convert_pytorch_shards_to_safetensors(source, destination)
    loaded = load_file(destination / "model-00001-of-00001.safetensors")
    torch.testing.assert_close(loaded["embed_tokens.weight"], tied)
    torch.testing.assert_close(loaded["lm_head.weight"], tied)
    assert loaded["embed_tokens.weight"].data_ptr() != loaded["lm_head.weight"].data_ptr()
