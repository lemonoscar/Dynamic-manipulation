import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def test_v3_3dgs_preserves_v1_timing_and_camera_contract() -> None:
    v1 = _load("configs/v1.json")
    v3 = _load("configs/v3_3dgs.json")

    assert v3["status"] == "design"
    assert v3["collection_ready"] is False
    assert v3["version_boundary"]["canonical_protocol"] == v1["protocol_version"]
    assert v3["rates_hz"] == v1["rates_hz"]
    for camera_id, v1_camera in v1["cameras"].items():
        if not isinstance(v1_camera, dict) or "resolution" not in v1_camera:
            continue
        v3_camera = v3["cameras"][camera_id]
        assert v3_camera["resolution"] == v1_camera["resolution"]
        assert v3_camera["fps"] == v1_camera["fps"]
        assert v3_camera["role"] == v1_camera["role"]


def test_v3_3dgs_separates_physics_and_visual_layers() -> None:
    config = _load("configs/v3_3dgs.json")
    partition = config["visual_partition"]

    assert set(partition["static_in_3dgs"]).isdisjoint(partition["dynamic_in_isaac"])
    assert set(partition["dynamic_in_isaac"]).issubset(
        partition["excluded_from_static_capture"]
    )
    assert config["physics_layer"]["gaussian_splats_have_collision"] is False
    assert "depth_m" in config["render_products"]["isaac_foreground"]
    assert "depth_m" in config["render_products"]["gaussian_background"]


def test_v3_3dgs_has_no_external_runtime_reference() -> None:
    config = _load("configs/v3_3dgs.json")
    runtime = config["runtime"]

    assert runtime["network_required"] is False
    assert runtime["runtime_downloads_allowed"] is False
    assert runtime["external_project_files_required"] is False
    assert all("://" not in value for value in _strings(config))
    for key in (
        "asset_manifest",
        "visual_ply",
        "calibration",
        "capture_masks",
        "photometric_calibration",
    ):
        path = Path(config["gaussian_layer"][key])
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_v3_3dgs_keeps_overview_out_of_policy_inputs() -> None:
    config = _load("configs/v3_3dgs.json")
    recording = config["recording"]

    assert recording["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]
    assert recording["observer_camera_ids"] == ["overview_rgb"]
    assert config["acceptance_gates"]["data"]["overview_as_policy_input_allowed"] is False
