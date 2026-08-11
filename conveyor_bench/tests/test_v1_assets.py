import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest

from conveyor_bench.v1.assets import (
    ASSET_LOCK_PATH,
    ASSET_ROOT,
    ObjectAsset,
    geometry_half_extents,
    load_object_registry,
    load_receptacles,
    load_workcell_manifest,
    sha256_file,
    source_tree_fingerprint,
    verify_asset_lock,
)


def test_object_registry_has_eight_distinct_geometry_assets() -> None:
    assets = load_object_registry()
    assert len(assets) == 8
    assert len({asset.object_id for asset in assets}) == 8
    assert {asset.split for asset in assets} == {"seen", "unseen"}
    assert sum(asset.split == "seen" for asset in assets) == 6
    assert sum(asset.split == "unseen" for asset in assets) == 2
    assert {asset.geometry["kind"] for asset in assets} == {
        "box",
        "cylinder",
        "compound",
    }
    assert all(asset.nominal_height_m > 0.0 for asset in assets)


def test_round_parts_have_calibrated_rolling_resistance() -> None:
    assets = {asset.object_id: asset for asset in load_object_registry()}

    calibrated = {"part_yellow_bushing", "part_green_shaft"}
    assert all(
        assets[object_id].angular_damping == pytest.approx(5.0)
        for object_id in calibrated
    )
    assert all(
        asset.angular_damping == 0.0
        for object_id, asset in assets.items()
        if object_id not in calibrated
    )


def test_grasp_curriculum_parts_are_fixture_aligned_for_one_wrist_branch(
) -> None:
    assets = {asset.object_id: asset for asset in load_object_registry()}
    curriculum = {
        "part_red_block",
        "part_blue_bar",
        "part_yellow_bushing",
        "part_green_shaft",
    }

    assert all(
        assets[object_id].grasp_affordances[0].finger_closing_axis == "y"
        for object_id in curriculum
    )
    assert assets["part_blue_bar"].stable_poses_wxyz[0][3] > 0.70
    assert assets["part_green_shaft"].stable_poses_wxyz[0][3] > 0.70
    assert all(
        0.34
        + geometry_half_extents(assets[object_id].geometry)[2]
        + assets[object_id].grasp_affordances[0].tcp_offset_xyz[2]
        == pytest.approx(0.376)
        for object_id in curriculum
    )


def test_compound_geometry_extent_includes_offsets() -> None:
    assets = {asset.object_id: asset for asset in load_object_registry()}
    flange = assets["part_orange_flange"]
    half = geometry_half_extents(flange.geometry)
    assert half[0] == pytest.approx(0.035)
    assert half[2] == pytest.approx(0.028)


def test_receptacles_have_two_sort_bins_and_post_exit_catch() -> None:
    zones = {zone.zone_id: zone for zone in load_receptacles()}
    assert {"sort_bin_blue", "sort_bin_yellow", "reject_catch"} == set(zones)
    assert zones["sort_bin_blue"].center_xyz_m[1] == pytest.approx(0.40)
    assert zones["sort_bin_yellow"].center_xyz_m[1] == pytest.approx(-0.40)
    assert zones["reject_catch"].center_xyz_m[1] == pytest.approx(-0.93)


def test_workcell_is_project_local_procedural_asset() -> None:
    manifest = load_workcell_manifest()
    assert manifest["runtime_dependency"] == "none"
    design = manifest["design"]
    assert design["belt_top_z_m"] == pytest.approx(0.34)
    assert design["belt_center_xyz_m"] == pytest.approx((0.70, 0.0, 0.31))
    assert design["belt_size_xyz_m"] == pytest.approx((0.252, 1.56, 0.06))
    assert design["recommended_low_speed_mps"] == pytest.approx(0.01)
    assert design["belt_top_z_m"] < design[
        "nominal_mobile_head_camera_axis_z_m"
    ]
    assert design["belt_clearance_below_nominal_head_axis_m"] == pytest.approx(
        0.03
    )
    assert design["belt_diffuse_color_linear_rgb"] == pytest.approx(
        (0.015, 0.10, 0.035)
    )
    assert design["belt_appearance"] == "dark green PVC"
    assert "transport_surface" in manifest["components"]
    assert "motor_and_guard" in manifest["components"]
    assert "catch_tray" in manifest["components"]


def test_registry_rejects_external_reference(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "conveyor-bench-object-registry-v1",
                "units": "m-kg-s",
                "objects": [{"geometry": {"kind": "https://invalid.example/asset"}}],
            }
        ),
        encoding="utf-8",
    )
    from conveyor_bench.v1.assets import load_object_registry

    with pytest.raises(ValueError, match="external reference"):
        load_object_registry(registry_path)


def test_asset_lock_matches_files() -> None:
    locked = verify_asset_lock()
    assert "objects/registry.json" in locked
    assert "robots/go2_x5/go2_x5.urdf" in locked
    assert "robots/go2_x5/ASSET_PROVENANCE.md" in locked
    for relative_name, digest in locked.items():
        assert sha256_file(ASSET_ROOT / relative_name) == digest


def test_go2_x5_composed_asset_dependencies_are_fully_locked() -> None:
    raw = json.loads(ASSET_LOCK_PATH.read_text(encoding="utf-8"))
    locked_names = set(raw["files"])
    robot_root = ASSET_ROOT / "robots" / "go2_x5"
    resolved_asset_root = ASSET_ROOT.resolve()

    mesh_names: set[str] = set()
    for mesh in ElementTree.parse(robot_root / "go2_x5.urdf").iterfind(
        ".//mesh"
    ):
        filename = mesh.get("filename")
        assert filename
        path = (robot_root / filename).resolve()
        assert path.is_relative_to(resolved_asset_root)
        assert path.is_file()
        mesh_names.add(path.relative_to(resolved_asset_root).as_posix())

    locked_mesh_names = {
        name
        for name in locked_names
        if name.startswith("robots/go2_x5/meshes/")
    }
    assert locked_mesh_names == mesh_names

    configuration_names = {
        path.relative_to(ASSET_ROOT).as_posix()
        for path in (robot_root / "configuration").glob("*.usd")
    }
    locked_configuration_names = {
        name
        for name in locked_names
        if name.startswith("robots/go2_x5/configuration/")
    }
    assert locked_configuration_names == configuration_names


def test_go2_x5_urdf_and_mesh_snapshot_match_pct_scene() -> None:
    robot_root = ASSET_ROOT / "robots" / "go2_x5"
    urdf_path = robot_root / "go2_x5.urdf"
    assert sha256_file(urdf_path) == (
        "d52f9690ec3828692e1bcafaea14f08ae0b790126bc55c3619288d508cb1e23e"
    )

    mesh_root = robot_root / "meshes"
    rows = []
    mesh_paths = (
        candidate for candidate in mesh_root.rglob("*") if candidate.is_file()
    )
    for path in sorted(mesh_paths):
        rows.append(
            f"{path.relative_to(mesh_root).as_posix()}\0{sha256_file(path)}\n"
        )
    assert len(rows) == 18
    assert hashlib.sha256("".join(rows).encode()).hexdigest() == (
        "3b19b8340431661e8107c14cc4310cee2ddf02efaa30a956a1430be91f959015"
    )

    root = ElementTree.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    arm_mount = joints["arm_base_joint"].find("origin")
    tcp = joints["grasp_tcp_fixed_joint"].find("origin")
    assert arm_mount is not None and arm_mount.get("xyz") == "0.12 0 0.05"
    assert tcp is not None and tcp.get("xyz") == "0.15757 0.0 0.0"
    assert "front_camera_joint" in joints

    expected_usd_sha256 = {
        "configuration/go2_x5_base.usd": (
            "68400cedea2c9548b556a238c386e2bf9f1b6729f7eb956cf0a04102cd9cdacc"
        ),
        "configuration/go2_x5_physics.usd": (
            "c9e7616d4747f8c85e0872ab1c2db2cb702b7b79bf855aff2863fea2b096c1c5"
        ),
        "configuration/go2_x5_robot.usd": (
            "045b0ea9586b937280ae6850f392d7dbad5508b48ec57e082ff1f2acdcb2de29"
        ),
        "configuration/go2_x5_sensor.usd": (
            "ba7935f81bd54b6cabddf0ac2cf30c28f7d66caf9282b458e368570c917abe24"
        ),
        "go2_x5.usd": (
            "b2936b1c65d20b608aa2b05985e73c79e8a59ac97c5d57d18741f7715250b1bc"
        ),
    }
    assert {
        relative_path: sha256_file(robot_root / relative_path)
        for relative_path in expected_usd_sha256
    } == expected_usd_sha256


def test_verify_asset_lock_remains_compatible_with_flat_v1_schema(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_path = asset_root / "asset.bin"
    asset_path.write_bytes(b"locked asset")
    digest = sha256_file(asset_path)
    lock_path = tmp_path / "asset_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "conveyor-bench-asset-lock-v1",
                "files": {"asset.bin": digest},
            }
        ),
        encoding="utf-8",
    )

    assert verify_asset_lock(lock_path, asset_root) == {"asset.bin": digest}


def test_source_tree_fingerprint_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    for directory in ("src", "scripts", "configs"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (tmp_path / "scripts" / "run.py").write_text(
        "print('run')\n", encoding="utf-8"
    )
    (tmp_path / "configs" / "v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'fixture'\n", encoding="utf-8"
    )
    (tmp_path / "src" / "ignored.pyc").write_bytes(b"cache")

    first = source_tree_fingerprint(tmp_path)
    second = source_tree_fingerprint(tmp_path)
    assert first == second
    assert first["file_count"] == 4

    (tmp_path / "src" / "module.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    assert source_tree_fingerprint(tmp_path)["sha256"] != first["sha256"]
