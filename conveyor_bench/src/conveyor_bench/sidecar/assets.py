"""Validate the SSH-delivered scene bundle and compose its runtime USD.

The large NuRec and object files are deliberately not part of Git.  A server
run receives one immutable sidecar directory over SSH and proves its contents
against ``TRANSFER_MANIFEST.sha256`` before Isaac Sim opens any asset.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


ASSET_ROOT_ENV = "CONVEYOR_BENCH_ASSET_ROOT"
LEGACY_ASSET_ROOT_ENV = "CONVEYOR_BENCH_V3_ASSET_ROOT"
TRANSFER_MANIFEST_NAME = "TRANSFER_MANIFEST.sha256"

LIANGZHU_SCENE_RELATIVE_PATH = Path("liangzhu/liangzhu.usda")
LIANGZHU_RUNTIME_MANIFEST_RELATIVE_PATH = Path(
    "liangzhu/runtime_asset_manifest.json"
)
LIANGZHU_NUREC_USDZ_RELATIVE_PATH = Path(
    "liangzhu/usdz/liangzhu.usdz"
)
LIANGZHU_COLLISION_USD_RELATIVE_PATH = Path(
    "liangzhu/usd/liangzhu_collision.usda"
)

# The NuRec scene must remain in its authored world frame.  Registered
# compositing uses that frame for camera calibration, so moving this parent
# Xform produces a plausible USD stage but incorrect RGB images.
LIANGZHU_SOURCE_ROBOT_XYZ_M = (
    -1.4849319648011197,
    5.126136502764003,
    0.29281728532721385,
)
LIANGZHU_SOURCE_ROOT_TO_GROUND_M = 0.43101139033367283
LIANGZHU_SOURCE_GROUND_Z_M = (
    LIANGZHU_SOURCE_ROBOT_XYZ_M[2] - LIANGZHU_SOURCE_ROOT_TO_GROUND_M
)
LIANGZHU_SCENE_TRANSLATION_XYZ_M = (0.0, 0.0, 0.0)

OBJECT_USD_RELATIVE_PATHS = {
    "cola": Path(
        "objects/cola/MesaTask-10K/MesaTask_Assets/can/"
        "0364ab96f338493c972248102b462aa4/usd/"
        "0364ab96f338493c972248102b462aa4.usd"
    ),
    "apple": Path(
        "objects/apple/MesaTask-10K/MesaTask_Assets/apple/"
        "0176be079c2449e7aaebfb652910a854/usd/"
        "0176be079c2449e7aaebfb652910a854.usd"
    ),
    "orange": Path(
        "objects/orange/MesaTask-10K/MesaTask_Assets/orange/"
        "0896dc31d5154c97aa3f24e8ec1277aa/usd/"
        "0896dc31d5154c97aa3f24e8ec1277aa.usd"
    ),
    "bottle": Path(
        "objects/bottle/MesaTask-10K/MesaTask_Assets/body-care_products/"
        "ec67d7141333464ca1061320452f06a2/usd/"
        "ec67d7141333464ca1061320452f06a2.usd"
    ),
    "box2": Path("objects/box2/box2.usd"),
}

_REQUIRED_RELATIVE_PATHS = frozenset(
    {
        LIANGZHU_SCENE_RELATIVE_PATH,
        LIANGZHU_RUNTIME_MANIFEST_RELATIVE_PATH,
        LIANGZHU_NUREC_USDZ_RELATIVE_PATH,
        LIANGZHU_COLLISION_USD_RELATIVE_PATH,
        *OBJECT_USD_RELATIVE_PATHS.values(),
    }
)
_MANIFEST_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, line_number: int) -> Path:
    if "\\" in value:
        raise ValueError(
            f"manifest line {line_number} must use POSIX separators"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(
            f"manifest line {line_number} has an unsafe path: {value!r}"
        )
    if pure.parts[0] == ".":
        pure = PurePosixPath(*pure.parts[1:])
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(
            f"manifest line {line_number} has an unsafe path: {value!r}"
        )
    return Path(*pure.parts)


def _load_transfer_manifest(path: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        match = _MANIFEST_LINE.fullmatch(raw_line)
        if match is None:
            raise ValueError(
                f"invalid SHA-256 manifest line {line_number}: {raw_line!r}"
            )
        digest, raw_relative = match.groups()
        relative = _safe_relative_path(
            raw_relative, line_number=line_number
        )
        if relative in entries:
            raise ValueError(f"duplicate manifest path: {relative.as_posix()}")
        entries[relative] = digest.lower()
    if not entries:
        raise ValueError("transfer manifest contains no files")
    return entries


def _assert_regular_file(root: Path, relative: Path) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"asset is missing: {relative.as_posix()}")
    if path.is_symlink():
        raise ValueError(f"asset cannot be a symlink: {relative.as_posix()}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"asset escapes its bundle: {relative.as_posix()}")
    return resolved


def _assert_bundle_has_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"asset bundle cannot contain symlinks: "
                f"{path.relative_to(root).as_posix()}"
            )


def _validate_scene_contract(root: Path) -> None:
    scene = (root / LIANGZHU_SCENE_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    for marker in (
        'def Xform "VisualScene"',
        'def Xform "GaussianScene"',
        'def Xform "PhysicsScene"',
        'def Xform "CollisionScene"',
        'over "LiangzhuCollision"',
        "./usdz/liangzhu.usdz[gauss.usda]",
        "./usd/liangzhu_collision.usda",
    ):
        if marker not in scene:
            raise ValueError(f"Liangzhu scene is missing contract marker: {marker}")

    runtime_manifest = json.loads(
        (root / LIANGZHU_RUNTIME_MANIFEST_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
    )
    if runtime_manifest.get("scene_profile") != "liangzhu_single_floor":
        raise ValueError("unexpected Liangzhu scene profile")
    runtime = runtime_manifest.get("scene_runtime", {})
    if runtime.get("visual_prim_path") != "/World/VisualScene/GaussianScene":
        raise ValueError("unexpected Liangzhu visual prim path")
    if runtime.get("collision_prim_path") != (
        "/World/PhysicsScene/CollisionScene/LiangzhuCollision"
    ):
        raise ValueError("unexpected Liangzhu collision prim path")

    with zipfile.ZipFile(root / LIANGZHU_NUREC_USDZ_RELATIVE_PATH) as archive:
        names = set(archive.namelist())
        required_members = {
            "default.usda",
            "gauss.usda",
            "liangzhu_cropped_raw.nurec",
        }
        if not required_members.issubset(names):
            missing = sorted(required_members - names)
            raise ValueError(f"NuRec USDZ is missing members: {missing}")
        gauss = archive.read("gauss.usda").decode("utf-8")
    for marker in (
        "OmniNuRecFieldAsset",
        "omni:nurec:isNuRecVolume = 1",
        "@./liangzhu_cropped_raw.nurec@",
    ):
        if marker not in gauss:
            raise ValueError(f"NuRec layer is missing contract marker: {marker}")


@dataclass(frozen=True)
class AssetBundleReport:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    file_count: int
    total_bytes: int
    hashes_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "hashes_verified": self.hashes_verified,
        }


@dataclass(frozen=True)
class AssetBundle:
    root: Path
    report: AssetBundleReport

    @property
    def scene_usda(self) -> Path:
        return self.root / LIANGZHU_SCENE_RELATIVE_PATH

    @property
    def collision_usda(self) -> Path:
        return self.root / LIANGZHU_COLLISION_USD_RELATIVE_PATH

    @property
    def nurec_usdz(self) -> Path:
        return self.root / LIANGZHU_NUREC_USDZ_RELATIVE_PATH

    def object_usd(self, object_id: str) -> Path:
        try:
            relative = OBJECT_USD_RELATIVE_PATHS[object_id]
        except KeyError as exc:
            raise KeyError(f"unknown object asset: {object_id}") from exc
        return self.root / relative

    def write_runtime_layer(
        self,
        output_path: Path,
    ) -> Path:
        """Write a deterministic native NuRec/collision composition layer."""

        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene_path = self.scene_usda.as_posix()
        if "@" in scene_path or "\n" in scene_path:
            raise ValueError("scene asset path cannot be represented in USDA")
        tx, ty, tz = LIANGZHU_SCENE_TRANSLATION_XYZ_M
        payload = f'''#usda 1.0
(
    defaultPrim = "LiangzhuScene"
    metersPerUnit = 1
    upAxis = "Z"
    customLayerData = {{
        dictionary renderSettings = {{
            int "rtx:directLighting:sampledLighting:samplesPerPixel" = 8
            bool "rtx:material:enableRefraction" = 0
            bool "rtx:matteObject:visibility:secondaryRays" = 1
            bool "rtx:post:histogram:enabled" = 0
            bool "rtx:post:registeredCompositing:invertColorCorrection" = 1
            bool "rtx:post:registeredCompositing:invertToneMap" = 1
            int "rtx:post:tonemap:op" = 2
            bool "rtx:raytracing:fractionalCutoutOpacity" = 0
            string "rtx:rendermode" = "RaytracedLighting"
            bool "rtx:useViewLightingMode" = 1
        }}
    }}
)

def Xform "LiangzhuScene"
{{
    double3 xformOp:translate = ({tx:.16g}, {ty:.16g}, {tz:.16g})
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Xform "Visual" (
        prepend references = @{scene_path}@</World/VisualScene/GaussianScene>
    )
    {{
        token visibility = "inherited"
    }}

    def Xform "Collision" (
        prepend references = @{scene_path}@</World/PhysicsScene/CollisionScene>
    )
    {{
        token visibility = "invisible"
    }}
}}
'''
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            temporary_path = Path(stream.name)
        os.replace(temporary_path, output_path)
        return output_path


def resolve_asset_root(value: str | Path | None = None) -> Path:
    """Resolve an explicit root or the one exported by the launch shell."""

    raw_value = value if value is not None else (
        os.environ.get(ASSET_ROOT_ENV) or os.environ.get(LEGACY_ASSET_ROOT_ENV)
    )
    if raw_value is None or not str(raw_value).strip():
        raise ValueError(
            f"assets require --asset-root or {ASSET_ROOT_ENV}"
        )
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        raise ValueError("asset root must be an absolute path")
    if path.is_symlink():
        raise ValueError("asset root cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"asset root is not a directory: {resolved}")
    return resolved


def validate_asset_bundle(
    root: str | Path | None = None,
    *,
    verify_all_hashes: bool = True,
    allowed_root: str | Path | None = None,
) -> AssetBundle:
    """Validate one complete, immutable SSH-delivered asset directory."""

    resolved_root = resolve_asset_root(root)
    if allowed_root is not None:
        resolved_allowed = Path(allowed_root).expanduser().resolve(strict=True)
        if not resolved_root.is_relative_to(resolved_allowed):
            raise ValueError(
                f"asset root {resolved_root} is outside {resolved_allowed}"
            )
    _assert_bundle_has_no_symlinks(resolved_root)
    manifest_path = _assert_regular_file(
        resolved_root, Path(TRANSFER_MANIFEST_NAME)
    )
    entries = _load_transfer_manifest(manifest_path)
    missing_contract_paths = _REQUIRED_RELATIVE_PATHS - entries.keys()
    if missing_contract_paths:
        missing = sorted(path.as_posix() for path in missing_contract_paths)
        raise ValueError(f"transfer manifest is missing required assets: {missing}")

    total_bytes = 0
    for relative, expected_digest in entries.items():
        path = _assert_regular_file(resolved_root, relative)
        total_bytes += path.stat().st_size
        if verify_all_hashes:
            actual_digest = _sha256(path)
            if actual_digest != expected_digest:
                raise ValueError(
                    f"SHA-256 mismatch for {relative.as_posix()}: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
    _validate_scene_contract(resolved_root)
    report = AssetBundleReport(
        root=resolved_root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        file_count=len(entries),
        total_bytes=total_bytes,
        hashes_verified=verify_all_hashes,
    )
    return AssetBundle(root=resolved_root, report=report)
