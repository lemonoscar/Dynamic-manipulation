from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/scene_v3.py"
RUNTIME_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/runtime_v3.py"
RUNTIME_V1_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/runtime_v1.py"
SCENE_V1_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/scene_v1.py"
PROBE_PATH = PROJECT_ROOT / "scripts/probe_v1_scene.py"
RUN_PATH = PROJECT_ROOT / "scripts/run_benchmark_v3.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path))


def _class(path: Path, name: str) -> ast.ClassDef:
    return next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def test_v3_scene_inherits_v1_but_replaces_only_static_context() -> None:
    scene = _class(SCENE_PATH, "ConveyorSceneV3Cfg")
    source = ast.unparse(scene)

    assert [ast.unparse(base) for base in scene.bases] == ["ConveyorSceneV1Cfg"]
    assert "ground = None" in source
    assert "include_room_context=False" in source
    assert "include_local_sort_trays=True" in source
    assert "liangzhu_scene = None" in source

    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in _tree(SCENE_PATH).body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert constants["V3_OVERVIEW_CAMERA_OFFSET_XYZ"] == (
        0.3849319648,
        -3.6261365028,
        2.138194105,
    )
    assert constants["V3_OVERVIEW_CAMERA_OFFSET_WXYZ"] == (
        0.6972261796,
        -0.1484619933,
        0.1511499216,
        0.6848272718,
    )
    assert constants["V3_OVERVIEW_CAMERA_FAR_CLIPPING_M"] == 50.0
    assert constants["V3_PCT_COKE_TASK_WORKCELL_GROUND_XYZ_M"] == (
        -1.4849319648,
        5.1261365028,
        -0.138194105,
    )

    v1_source = _source(SCENE_V1_PATH)
    assert "include_room_context: bool = True" in v1_source
    assert "if cfg.include_room_context:" in v1_source


def test_v3_runtime_reuses_v1_collector_and_adds_scene_provenance() -> None:
    runtime = _class(RUNTIME_PATH, "ConveyorRuntimeV3")
    source = ast.unparse(runtime)

    assert [ast.unparse(base) for base in runtime.bases] == ["ConveyorRuntimeV1"]
    assert "verify_all_hashes=True" in source
    assert "make_conveyor_scene_v3_cfg" in source
    assert "place_workcell_in_liangzhu_task_area" in source
    assert "validate_liangzhu_stage" in source
    assert "disable_liangzhu_background_collision" in source
    assert "background_collision_contract" in source
    assert "isaac_rtx_native_nurec" in source
    assert "ssh_sidecar_bundle" in source
    assert "def _task_world_origin_xyz" in source
    assert "V3_PCT_COKE_TASK_WORKCELL_GROUND_XYZ_M" in source

    scene_source = _source(SCENE_PATH)
    assert "attribute.Set(False)" in scene_source
    assert '"validated_then_disabled_for_collection"' in scene_source
    assert "V3_LOCAL_FLOOR_PATCH_PRIM_PATH" in scene_source


def test_v3_task_frame_is_translated_before_teacher_control() -> None:
    runtime_v1 = _source(RUNTIME_V1_PATH)

    assert "origin_y = self._task_world_origin_xyz()[1]" in runtime_v1
    assert "task_origin = self._task_world_origin_xyz()" in runtime_v1
    assert '"task_world_origin_xyz_m": task_origin' in runtime_v1
    assert "resolved.manifest.belt_surface_z_m" in runtime_v1
    assert "task_origin_world_xyz=self._task_world_origin_xyz()" in runtime_v1


def test_v3_probe_and_collection_have_explicit_asset_roots() -> None:
    probe = _source(PROBE_PATH)
    runner = _source(RUN_PATH)

    assert "v3_nurec" in probe
    assert "--v3-asset-root" in probe
    assert "--v3-workcell-ground-xyz" in probe
    assert "--overview-resolution" in probe
    assert "--overview-eye-world-xyz" in probe
    assert "--overview-target-world-xyz" in probe
    assert "--antialiasing-mode" in probe
    assert "validate_asset_bundle" in probe
    assert "place_workcell_in_liangzhu_task_area" in probe
    assert "--asset-root" in runner
    assert "RuntimeOptionsV3" in runner
    assert "run_collection_v3" in runner
