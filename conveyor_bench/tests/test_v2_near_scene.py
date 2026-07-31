from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "scene_v2.py"
)


def test_v2_near_scene_keeps_v1_geometry_with_offline_ground() -> None:
    source = SCENE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scene = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ConveyorNearSortV2SceneCfg"
    )

    assert [ast.unparse(base) for base in scene.bases] == [
        "ConveyorSceneV1Cfg"
    ]
    class_source = ast.unparse(scene)
    assert "sim_utils.CuboidCfg" in class_source
    assert "GroundPlaneCfg" not in class_source
    assert "size=(6.0, 6.0, 0.1)" in class_source
    assert "pos=(0.0, 0.0, -0.05)" in class_source

    lowered = source.lower()
    for marker in ("http:", "https:", "omniverse:", "s3:"):
        assert marker not in lowered
