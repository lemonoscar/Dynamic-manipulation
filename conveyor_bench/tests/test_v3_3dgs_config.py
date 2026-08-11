import json
import math
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


def test_v3_preserves_v1_timing_and_camera_contract() -> None:
    v1 = _load("configs/v1.json")
    v3 = _load("configs/v3_3dgs.json")

    assert v3["status"] == "integration"
    assert v3["collection_ready"] is False
    assert "native_nurec_remote_render_gate" not in v3["collection_blockers"]
    assert "nurec_workcell_visibility_clearance_gate" in v3["collection_blockers"]
    assert "liangzhu_collision_clearance_gate" in v3["collection_blockers"]
    assert "carry_to_preplace_teacher_transition" in v3["collection_blockers"]
    assert v3["version_boundary"]["canonical_protocol"] == v1["protocol_version"]
    assert v3["rates_hz"] == v1["rates_hz"]
    for camera_id, v1_camera in v1["cameras"].items():
        if not isinstance(v1_camera, dict) or "resolution" not in v1_camera:
            continue
        v3_camera = v3["cameras"][camera_id]
        assert v3_camera["resolution"] == v1_camera["resolution"]
        assert v3_camera["fps"] == v1_camera["fps"]
        assert v3_camera["role"] == v1_camera["role"]


def test_v3_uses_native_nurec_and_separate_collision() -> None:
    config = _load("configs/v3_3dgs.json")

    assert config["runtime"]["backend"] == "isaac_rtx_native_nurec"
    assert config["runtime"]["composition"] == (
        "single_pass_registered_compositing"
    )
    assert config["physics_layer"]["gaussian_splats_have_collision"] is False
    assert config["physics_layer"]["procedural_ground_enabled"] is False
    assert config["gaussian_layer"]["asset_container"].endswith(".usdz")
    assert config["gaussian_layer"]["nurec_archive_member"].endswith(".nurec")
    assert config["visual_partition"]["procedural_v1_room_enabled"] is False


def test_v3_asset_delivery_is_ssh_sidecar_and_never_runtime_network() -> None:
    config = _load("configs/v3_3dgs.json")
    runtime = config["runtime"]
    delivery = config["asset_delivery"]

    assert runtime["network_required"] is False
    assert runtime["runtime_downloads_allowed"] is False
    assert runtime["repository_embedded_large_assets"] is False
    assert runtime["ssh_sidecar_asset_bundle_required"] is True
    assert delivery["method"] == "ssh_sidecar"
    assert delivery["manifest"] == "TRANSFER_MANIFEST.sha256"
    assert delivery["symlinks_allowed"] is False
    assert all("://" not in value for value in _strings(config))

    path_keys = (
        config["physics_layer"]["static_collision_asset"],
        config["gaussian_layer"]["asset_container"],
        config["gaussian_layer"]["source_scene"],
        config["gaussian_layer"]["source_runtime_manifest"],
        *config["object_assets"]["usd_paths"].values(),
    )
    for raw_path in path_keys:
        path = Path(raw_path)
        assert not path.is_absolute()
        assert ".." not in path.parts


def test_v3_task_frame_maps_pct_robot_and_conveyor_consistently() -> None:
    config = _load("configs/v3_3dgs.json")
    task = config["task_frame"]
    translation = config["gaussian_layer"]["sim_translation_xyz_m"]
    workcell_ground = task["open_room_workcell_ground_xyz_m"]

    assert translation == [0.0, 0.0, 0.0]
    assert workcell_ground == [-12.0, 14.0, -0.0993]
    assert task["sim_robot_anchor_xy_m"] == workcell_ground[:2]
    assert task["placement_method"] == "collision_mesh_raycast_open_room"
    assert task["nurec_parent_transform_policy"].startswith("identity")

    source_conveyor = task["conveyor_center_source_xy_m"]
    sim_conveyor = task["conveyor_center_xy_m"]
    assert math.isclose(
        source_conveyor[0] + translation[0], sim_conveyor[0], abs_tol=1e-12
    )
    assert math.isclose(
        source_conveyor[1] + translation[1], sim_conveyor[1], abs_tol=1e-12
    )
    assert task["conveyor_long_axis"] == "world_y"
    assert task["transport_direction_world"] == [0.0, -1.0, 0.0]
    assert task["candidate_lateral_offsets_m"] == [-0.25, 0.0, 0.25]


def test_v3_first_object_pilot_excludes_deformable_blanket() -> None:
    objects = _load("configs/v3_3dgs.json")["object_assets"]

    assert objects["first_pilot_ids"] == ["cola", "apple", "orange"]
    assert objects["unseen_gate_ids"] == ["bottle"]
    assert objects["deferred_ids"] == ["blanket"]
    assert objects["destination_id"] == "box2"
    assert set(objects["usd_paths"]) == {
        "cola",
        "apple",
        "orange",
        "bottle",
        "box2",
    }


def test_v3_keeps_overview_out_of_policy_inputs() -> None:
    config = _load("configs/v3_3dgs.json")
    recording = config["recording"]

    assert recording["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]
    assert recording["observer_camera_ids"] == ["overview_rgb"]
    assert config["acceptance_gates"]["data"][
        "overview_as_policy_input_allowed"
    ] is False
