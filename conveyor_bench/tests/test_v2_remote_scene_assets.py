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
ASSET_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "asset_config.py"
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
    assert near["head_rgb"]["mount"]["orientation_convention"] == (
        near_constants["HEAD_CAMERA_ORIENTATION_CONVENTION"]
    )
    assert near["wrist_rgb"]["mount"]["xyz_m"] == pytest.approx(
        near_constants["WRIST_CAMERA_OFFSET_XYZ"]
    )
    assert near["wrist_rgb"]["mount"]["wxyz"] == pytest.approx(
        near_constants["WRIST_CAMERA_OFFSET_WXYZ"]
    )
    assert near["wrist_rgb"]["mount"]["orientation_convention"] == (
        near_constants["WRIST_CAMERA_ORIENTATION_CONVENTION"]
    )
    assert near["overview_rgb"]["mount"]["xyz_m"] == pytest.approx(
        near_constants["OVERVIEW_CAMERA_OFFSET_XYZ"]
    )
    assert near["overview_rgb"]["mount"]["wxyz"] == pytest.approx(
        near_constants["OVERVIEW_CAMERA_OFFSET_WXYZ"]
    )
    assert near["overview_rgb"]["mount"]["orientation_convention"] == (
        near_constants["OVERVIEW_CAMERA_ORIENTATION_CONVENTION"]
    )
    assert remote["overview_rgb"]["mount"]["xyz_m"] == pytest.approx(
        remote_constants["REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ"]
    )
    assert remote["overview_rgb"]["mount"]["wxyz"] == pytest.approx(
        remote_constants["REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ"]
    )
    for contract in (near, remote):
        for camera_id in ("head_rgb", "wrist_rgb"):
            camera = contract[camera_id]
            assert camera["resolution"] == [640, 480]
            assert camera["model"] == "opencv_pinhole"
            expected_intrinsics = (
                (383.44608095, 0.0, 324.33479864),
                (0.0, 383.52724198, 238.90275478),
                (0.0, 0.0, 1.0),
            )
            for actual, expected in zip(
                camera["intrinsics"],
                expected_intrinsics,
                strict=True,
            ):
                assert actual == pytest.approx(expected)
            assert camera["distortion_coefficients"] == [0.0] * 12


def test_go2_camera_mounts_match_the_audited_arm_vla_reference() -> None:
    constants = _constants(_tree(SCENE_V1_PATH))
    assert constants["HEAD_CAMERA_OFFSET_XYZ"] == pytest.approx((0.28, 0.0, 0.07))
    assert constants["HEAD_CAMERA_OFFSET_WXYZ"] == pytest.approx(
        (0.5, -0.5, 0.5, -0.5)
    )
    assert constants["HEAD_CAMERA_ORIENTATION_CONVENTION"] == "ros"
    assert constants["FRONT_CAMERA_PRIM_PATH"] == (
        "{ENV_REGEX_NS}/Robot/base/head_cam"
    )
    assert constants["WRIST_CAMERA_PRIM_PATH"] == (
        "{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera"
    )
    assert constants["WRIST_CAMERA_OFFSET_XYZ"] == pytest.approx(
        (0.0666580792, 0.0028071889, 0.0935779972)
    )
    assert constants["WRIST_CAMERA_OFFSET_WXYZ"] == pytest.approx(
        (0.3377891849, -0.6214992221, 0.6185057335, -0.3421810063)
    )
    assert constants["WRIST_CAMERA_ORIENTATION_CONVENTION"] == "ros"
    assert constants["D436_CAMERA_RESOLUTION_WH"] == (640, 480)
    assert constants["D436_CAMERA_FX_PX"] == pytest.approx(383.44608095)
    assert constants["D436_CAMERA_FY_PX"] == pytest.approx(383.52724198)
    assert constants["D436_CAMERA_CX_PX"] == pytest.approx(324.33479864)
    assert constants["D436_CAMERA_CY_PX"] == pytest.approx(238.90275478)
    assert constants["WRIST_CAMERA_NEAR_CLIPPING_M"] == pytest.approx(0.03)
    assert constants["BELT_DARK_GREEN_RGB"] == pytest.approx((0.015, 0.10, 0.035))

    scene_source = ast.unparse(_class(_tree(SCENE_V1_PATH), "ConveyorSceneV1Cfg"))
    assert "height=D436_CAMERA_RESOLUTION_WH[1]" in scene_source
    assert "width=D436_CAMERA_RESOLUTION_WH[0]" in scene_source
    assert "func=make_d436_camera_spawn_function()" in scene_source
    assert "convention=HEAD_CAMERA_ORIENTATION_CONVENTION" in scene_source
    assert "convention=WRIST_CAMERA_ORIENTATION_CONVENTION" in scene_source


def test_mobile_runtime_uses_pct_urdf_and_original_finray_colliders() -> None:
    asset_tree = _tree(ASSET_CONFIG_PATH)
    constants = _constants(asset_tree)
    assert constants["TCP_OFFSET_X_M"] == pytest.approx(0.15757)
    asset_source = ast.unparse(asset_tree)
    policy_source = ast.unparse(
        _function(asset_tree, "make_go2_x5_policy_cfg")
    )
    assert "UrdfFileCfg" in asset_source
    assert "UsdFileCfg" not in policy_source

    collision_source = ast.unparse(
        _function(_tree(SCENE_V1_PATH), "apply_pct_gripper_collision_patch")
    )
    assert "convexDecomposition" in SCENE_V1_PATH.read_text(encoding="utf-8")
    assert "PhysicsMeshCollisionAPI" in collision_source
    assert "geometry_replaced" in collision_source
    assert "SetActive" not in collision_source
    assert "prim_type='Cube'" not in collision_source
