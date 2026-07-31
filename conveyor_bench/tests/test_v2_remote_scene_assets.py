from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from conveyor_bench.v1.assets import load_receptacles
from conveyor_bench.v2.camera_contracts import camera_contract_for_scene
from conveyor_bench.v2.config import SceneId


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_V1_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "scene_v1.py"
)
REMOTE_SCENE_PATH = (
    PROJECT_ROOT
    / "src"
    / "conveyor_bench"
    / "isaac"
    / "scene_remote_delivery.py"
)
REMOTE_RECEPTACLES_PATH = (
    PROJECT_ROOT / "assets" / "receptacles" / "remote_delivery_v2.json"
)
REMOTE_WORKCELL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "workcells"
    / "remote_delivery_v2"
    / "ASSET_MANIFEST.json"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _constants(tree: ast.Module) -> dict[str, object]:
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return result


def test_remote_receptacles_are_symmetric_and_force_delivery_motion() -> None:
    zones = {
        zone.zone_id: zone
        for zone in load_receptacles(REMOTE_RECEPTACLES_PATH)
    }
    assert set(zones) == {"delivery_bin_blue", "delivery_bin_yellow"}

    blue = zones["delivery_bin_blue"]
    yellow = zones["delivery_bin_yellow"]
    assert blue.center_xyz_m == pytest.approx((-0.16, 1.20, 0.46))
    assert yellow.center_xyz_m == pytest.approx((-0.16, -1.20, 0.46))
    assert blue.goal_half_extents_xyz_m == pytest.approx(
        (0.105, 0.125, 0.075)
    )
    assert yellow.goal_half_extents_xyz_m == pytest.approx(
        blue.goal_half_extents_xyz_m
    )
    assert blue.floor_top_z_m == yellow.floor_top_z_m == pytest.approx(0.405)
    assert blue.wall_height_m == yellow.wall_height_m == pytest.approx(0.10)
    assert blue.settle_dwell_s == yellow.settle_dwell_s == pytest.approx(0.50)


def test_remote_workcell_manifest_freezes_route_and_v1_reuse() -> None:
    manifest = json.loads(REMOTE_WORKCELL_PATH.read_text(encoding="utf-8"))
    assert manifest["scene_id"] == "mobile_remote_delivery_v2"
    assert manifest["runtime_dependency"] == "none"
    assert manifest["base_workcell"] == "conveyor_station_v1"

    design = manifest["design"]
    assert design["include_local_sort_trays"] is False
    assert design["include_reject_catch"] is True
    assert design["delivery_standoff_m"] == pytest.approx(0.42)
    assert design["minimum_loaded_base_displacement_m"] == pytest.approx(0.65)
    assert design["delivery_root_goals"]["delivery_bin_blue"] == pytest.approx(
        [-0.16, 0.78, math.pi / 2.0]
    )
    assert design["delivery_root_goals"]["delivery_bin_yellow"] == pytest.approx(
        [-0.16, -0.78, -math.pi / 2.0]
    )


def test_v1_workcell_switch_defaults_to_existing_local_trays() -> None:
    tree = _tree(SCENE_V1_PATH)
    config = _class(tree, "ProceduralWorkcellCfg")
    switch = next(
        node
        for node in config.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "include_local_sort_trays"
    )
    assert ast.literal_eval(switch.value) is True

    spawner = _function(tree, "spawn_conveyor_workcell")
    source = ast.unparse(spawner)
    assert "if cfg.include_local_sort_trays:" in source
    assert "sort_bin_blue" in source
    assert "sort_bin_yellow" in source


def test_remote_scene_inherits_v1_and_disables_only_local_sort_trays() -> None:
    tree = _tree(REMOTE_SCENE_PATH)
    constants = _constants(tree)
    assert constants["SCENE_ID"] == "mobile_remote_delivery_v2"
    assert constants["REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ"] == pytest.approx(
        (-2.80, -2.60, 3.20)
    )
    assert constants["REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ"] == pytest.approx(
        (0.89625224, -0.10238193, 0.27791988, 0.33016722)
    )

    scene = _class(tree, "ConveyorRemoteDeliverySceneCfg")
    assert [ast.unparse(base) for base in scene.bases] == ["ConveyorSceneV1Cfg"]
    source = ast.unparse(scene)
    assert "ground = AssetBaseCfg" in source
    assert "sim_utils.CuboidCfg" in source
    assert "GroundPlaneCfg" not in source
    assert "include_local_sort_trays=False" in source
    assert "spawn_conveyor_workcell" in source
    assert "spawn_remote_delivery_extension" in source
    assert "height=320" in source
    assert "width=480" in source
    assert "clipping_range=(0.05, 10.0)" in source


def test_remote_scene_assets_have_no_external_runtime_references() -> None:
    payload = "\n".join(
        (
            REMOTE_SCENE_PATH.read_text(encoding="utf-8"),
            REMOTE_RECEPTACLES_PATH.read_text(encoding="utf-8"),
            REMOTE_WORKCELL_PATH.read_text(encoding="utf-8"),
        )
    ).lower()
    for marker in ("http:", "https:", "omniverse:", "s3:"):
        assert marker not in payload


def test_frozen_camera_contract_matches_scene_mount_constants() -> None:
    near_constants = _constants(_tree(SCENE_V1_PATH))
    remote_constants = _constants(_tree(REMOTE_SCENE_PATH))
    near = camera_contract_for_scene(SceneId.TRANSVERSE_NEAR_SORT_V2)
    remote = camera_contract_for_scene(SceneId.MOBILE_REMOTE_DELIVERY_V2)

    assert near["head_rgb"]["mount"]["xyz_m"] == pytest.approx(
        near_constants["HEAD_CAMERA_OFFSET_XYZ"]
    )
    assert near["head_rgb"]["mount"]["wxyz"] == pytest.approx(
        near_constants["HEAD_CAMERA_OFFSET_WXYZ"]
    )
    assert near["wrist_rgb"]["mount"]["xyz_m"] == pytest.approx(
        near_constants["WRIST_CAMERA_OFFSET_XYZ"]
    )
    assert near["wrist_rgb"]["mount"]["wxyz"] == pytest.approx(
        near_constants["WRIST_CAMERA_OFFSET_WXYZ"]
    )
    assert near["overview_rgb"]["mount"]["xyz_m"] == pytest.approx(
        near_constants["OVERVIEW_CAMERA_OFFSET_XYZ"]
    )
    assert near["overview_rgb"]["mount"]["wxyz"] == pytest.approx(
        near_constants["OVERVIEW_CAMERA_OFFSET_WXYZ"]
    )
    assert remote["overview_rgb"]["mount"]["xyz_m"] == pytest.approx(
        remote_constants["REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ"]
    )
    assert remote["overview_rgb"]["mount"]["wxyz"] == pytest.approx(
        remote_constants["REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ"]
    )
