from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from conveyor_bench.v3.assets import (
    ASSET_ROOT_ENV,
    LIANGZHU_SCENE_TRANSLATION_XYZ_M,
    OBJECT_USD_RELATIVE_PATHS,
    resolve_asset_root,
    validate_asset_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str | bytes = b"asset") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(value)


def _make_bundle(root: Path) -> Path:
    _write(
        root / "liangzhu/liangzhu.usda",
        "\n".join(
            (
                'def Xform "VisualScene"',
                'def Xform "GaussianScene"',
                'def Xform "PhysicsScene"',
                'def Xform "CollisionScene"',
                'over "LiangzhuCollision"',
                "./usdz/liangzhu.usdz[gauss.usda]",
                "./usd/liangzhu_collision.usda",
            )
        ),
    )
    _write(
        root / "liangzhu/runtime_asset_manifest.json",
        json.dumps(
            {
                "scene_profile": "liangzhu_single_floor",
                "scene_runtime": {
                    "visual_prim_path": "/World/VisualScene/GaussianScene",
                    "collision_prim_path": (
                        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
                    ),
                },
            }
        ),
    )
    _write(root / "liangzhu/usd/liangzhu_collision.usda", "collision")
    archive_path = root / "liangzhu/usdz/liangzhu.usdz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("default.usda", "#usda 1.0")
        archive.writestr(
            "gauss.usda",
            "\n".join(
                (
                    "OmniNuRecFieldAsset",
                    "omni:nurec:isNuRecVolume = 1",
                    "@./liangzhu_cropped_raw.nurec@",
                )
            ),
        )
        archive.writestr("liangzhu_cropped_raw.nurec", b"nurec")
    for relative in OBJECT_USD_RELATIVE_PATHS.values():
        _write(root / relative)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    _write(root / "TRANSFER_MANIFEST.sha256", manifest)
    return root


def test_validate_asset_bundle_and_write_native_runtime_layer(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    bundle = validate_asset_bundle(root, allowed_root=tmp_path)

    assert bundle.report.hashes_verified is True
    assert bundle.report.file_count == 9
    assert bundle.report.total_bytes > 0
    assert bundle.object_usd("cola").is_file()

    runtime_layer = bundle.write_runtime_layer(tmp_path / "runtime/scene.usda")
    payload = runtime_layer.read_text(encoding="utf-8")
    assert "single_pass" not in payload
    assert "registeredCompositing" in payload
    assert str(bundle.scene_usda) in payload
    assert "</World/VisualScene/GaussianScene>" in payload
    assert "</World/PhysicsScene/CollisionScene>" in payload
    visual_block, collision_block = payload.split('def Xform "Collision"', 1)
    assert 'token visibility = "inherited"' in visual_block
    assert 'token visibility = "invisible"' in collision_block
    for component in LIANGZHU_SCENE_TRANSLATION_XYZ_M:
        assert f"{component:.16g}" in payload

    assert "double3 xformOp:translate = (0, 0, 0)" in payload


def test_asset_root_can_come_from_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_bundle(tmp_path / "assets")
    monkeypatch.setenv(ASSET_ROOT_ENV, str(root))
    assert resolve_asset_root() == root.resolve()


def test_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    bundle = validate_asset_bundle(root)
    bundle.object_usd("apple").write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_asset_bundle(root)


def test_bundle_rejects_symlinks(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    target = root / "extra.txt"
    target.write_text("extra", encoding="utf-8")
    os.symlink(target, root / "linked.txt")

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        validate_asset_bundle(root, verify_all_hashes=False)


def test_bundle_rejects_path_outside_allowed_root(tmp_path: Path) -> None:
    root = _make_bundle(tmp_path / "assets")
    allowed = tmp_path / "different"
    allowed.mkdir()

    with pytest.raises(ValueError, match="is outside"):
        validate_asset_bundle(
            root,
            verify_all_hashes=False,
            allowed_root=allowed,
        )
