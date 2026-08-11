from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/scene_v3.py"
RUNTIME_PATH = PROJECT_ROOT / "src/conveyor_bench/isaac/runtime_v3.py"
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
    assert constants["V3_OVERVIEW_CAMERA_OFFSET_XYZ"] == (-5.1, -3.5, 4.7)
    assert constants["V3_OVERVIEW_CAMERA_FAR_CLIPPING_M"] == 50.0

    v1_source = _source(SCENE_V1_PATH)
    assert "include_room_context: bool = True" in v1_source
    assert "if cfg.include_room_context:" in v1_source


def test_v3_runtime_reuses_v1_collector_and_adds_scene_provenance() -> None:
    runtime = _class(RUNTIME_PATH, "ConveyorRuntimeV3")
    source = ast.unparse(runtime)

    assert [ast.unparse(base) for base in runtime.bases] == ["ConveyorRuntimeV1"]
    assert "verify_all_hashes=True" in source
    assert "make_conveyor_scene_v3_cfg" in source
    assert "validate_liangzhu_stage" in source
    assert "isaac_rtx_native_nurec" in source
    assert "ssh_sidecar_bundle" in source


def test_v3_probe_and_collection_have_explicit_asset_roots() -> None:
    probe = _source(PROBE_PATH)
    runner = _source(RUN_PATH)

    assert "v3_nurec" in probe
    assert "--v3-asset-root" in probe
    assert "--v3-scene-translation-xyz" in probe
    assert "validate_asset_bundle" in probe
    assert "--asset-root" in runner
    assert "RuntimeOptionsV3" in runner
    assert "run_collection_v3" in runner
