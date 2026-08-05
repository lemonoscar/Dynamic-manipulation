"""Project-local V1 digital-asset registry and integrity helpers.

The V1 scene is intentionally procedural: geometry recipes in JSON are the
canonical assets, while Isaac Sim builds collision and visual primitives from
those recipes.  This keeps the benchmark self-contained and makes every random
choice and physical property auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSET_ROOT = PROJECT_ROOT / "assets"
OBJECT_REGISTRY_PATH = ASSET_ROOT / "objects" / "registry.json"
RECEPTACLE_MANIFEST_PATH = ASSET_ROOT / "receptacles" / "ASSET_MANIFEST.json"
WORKCELL_MANIFEST_PATH = (
    ASSET_ROOT / "workcells" / "conveyor_station_v1" / "ASSET_MANIFEST.json"
)
ASSET_LOCK_PATH = ASSET_ROOT / "asset_lock.json"

_ALLOWED_SPLITS = {"seen", "unseen"}
_ALLOWED_KINDS = {"box", "cylinder", "compound"}
_URL_MARKERS = ("://", "http:", "https:", "omniverse:", "s3:")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _vec(
    value: Any,
    length: int,
    name: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    if positive and any(component <= 0.0 for component in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _assert_no_external_references(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in _URL_MARKERS):
            raise ValueError(f"external reference is forbidden at {path}: {value}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_external_references(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _assert_no_external_references(item, f"{path}[{index}]")


@dataclass(frozen=True)
class GraspAffordance:
    affordance_id: str
    approach_axis: str
    finger_closing_axis: str
    tcp_offset_xyz: tuple[float, float, float]
    required_opening_m: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GraspAffordance":
        affordance = cls(
            affordance_id=str(value.get("id", "")),
            approach_axis=str(value.get("approach_axis", "")),
            finger_closing_axis=str(value.get("finger_closing_axis", "")),
            tcp_offset_xyz=_vec(value.get("tcp_offset_xyz"), 3, "tcp_offset_xyz"),
            required_opening_m=_finite_positive(
                value.get("required_opening_m"), "required_opening_m"
            ),
        )
        if not affordance.affordance_id:
            raise ValueError("grasp affordance id cannot be empty")
        if affordance.approach_axis not in {"+x", "-x", "+y", "-y", "+z", "-z"}:
            raise ValueError("unsupported approach_axis")
        if affordance.finger_closing_axis not in {"x", "y", "z"}:
            raise ValueError("unsupported finger_closing_axis")
        return affordance


@dataclass(frozen=True)
class ObjectAsset:
    object_id: str
    display_name: str
    category: str
    attributes: Mapping[str, str]
    language_aliases: Mapping[str, tuple[str, ...]]
    geometry: Mapping[str, Any]
    mass_kg: float
    static_friction: float
    dynamic_friction: float
    restitution: float
    angular_damping: float
    stable_poses_wxyz: tuple[tuple[float, float, float, float], ...]
    grasp_affordances: tuple[GraspAffordance, ...]
    split: str
    real_twin_id: str
    license: str
    provenance: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectAsset":
        geometry = value.get("geometry")
        physics = value.get("physics")
        aliases = value.get("language_aliases")
        attributes = value.get("attributes")
        if not isinstance(geometry, Mapping):
            raise ValueError("geometry must be an object")
        if not isinstance(physics, Mapping):
            raise ValueError("physics must be an object")
        if not isinstance(aliases, Mapping) or not aliases:
            raise ValueError("language_aliases must be a non-empty object")
        if not isinstance(attributes, Mapping):
            raise ValueError("attributes must be an object")
        _validate_geometry(geometry)

        stable_poses = value.get("stable_poses_wxyz")
        if not isinstance(stable_poses, list) or not stable_poses:
            raise ValueError("stable_poses_wxyz must be non-empty")
        resolved_poses = tuple(
            _normalized_quaternion(pose, "stable_poses_wxyz")
            for pose in stable_poses
        )
        affordances = value.get("grasp_affordances")
        if not isinstance(affordances, list) or not affordances:
            raise ValueError("grasp_affordances must be non-empty")

        asset = cls(
            object_id=str(value.get("object_id", "")),
            display_name=str(value.get("display_name", "")),
            category=str(value.get("category", "")),
            attributes={str(key): str(item) for key, item in attributes.items()},
            language_aliases={
                str(language): tuple(str(alias) for alias in language_alias_list)
                for language, language_alias_list in aliases.items()
            },
            geometry=dict(geometry),
            mass_kg=_finite_positive(physics.get("mass_kg"), "mass_kg"),
            static_friction=_finite_positive(
                physics.get("static_friction"), "static_friction"
            ),
            dynamic_friction=_finite_positive(
                physics.get("dynamic_friction"), "dynamic_friction"
            ),
            restitution=float(physics.get("restitution", 0.0)),
            angular_damping=float(physics.get("angular_damping", 0.0)),
            stable_poses_wxyz=resolved_poses,
            grasp_affordances=tuple(
                GraspAffordance.from_dict(affordance)
                for affordance in affordances
            ),
            split=str(value.get("split", "")),
            real_twin_id=str(value.get("real_twin_id", "")),
            license=str(value.get("license", "")),
            provenance=str(value.get("provenance", "")),
        )
        for name in (
            "object_id",
            "display_name",
            "category",
            "real_twin_id",
            "license",
            "provenance",
        ):
            if not getattr(asset, name):
                raise ValueError(f"{name} cannot be empty")
        if asset.split not in _ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {asset.split}")
        if not math.isfinite(asset.restitution) or not 0.0 <= asset.restitution <= 1.0:
            raise ValueError("restitution must be in [0, 1]")
        if (
            not math.isfinite(asset.angular_damping)
            or asset.angular_damping < 0.0
        ):
            raise ValueError("angular_damping must be finite and non-negative")
        if not all(asset.language_aliases.values()):
            raise ValueError("each language must provide at least one alias")
        return asset

    @property
    def half_extents_xyz(self) -> tuple[float, float, float]:
        return geometry_half_extents(self.geometry)

    @property
    def nominal_height_m(self) -> float:
        return 2.0 * self.half_extents_xyz[2]


@dataclass(frozen=True)
class ReceptacleAsset:
    zone_id: str
    display_name: str
    color_rgb: tuple[float, float, float]
    center_xyz_m: tuple[float, float, float]
    goal_half_extents_xyz_m: tuple[float, float, float]
    floor_top_z_m: float
    wall_height_m: float
    settle_dwell_s: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReceptacleAsset":
        result = cls(
            zone_id=str(value.get("zone_id", "")),
            display_name=str(value.get("display_name", "")),
            color_rgb=_vec(value.get("color_rgb"), 3, "color_rgb"),
            center_xyz_m=_vec(value.get("center_xyz_m"), 3, "center_xyz_m"),
            goal_half_extents_xyz_m=_vec(
                value.get("goal_half_extents_xyz_m"),
                3,
                "goal_half_extents_xyz_m",
                positive=True,
            ),
            floor_top_z_m=_finite_positive(
                value.get("floor_top_z_m"), "floor_top_z_m"
            ),
            wall_height_m=_finite_positive(
                value.get("wall_height_m"), "wall_height_m"
            ),
            settle_dwell_s=_finite_positive(
                value.get("settle_dwell_s"), "settle_dwell_s"
            ),
        )
        if not result.zone_id or not result.display_name:
            raise ValueError("zone_id and display_name cannot be empty")
        if any(not 0.0 <= channel <= 1.0 for channel in result.color_rgb):
            raise ValueError("color_rgb channels must be in [0, 1]")
        return result


def _normalized_quaternion(value: Any, name: str) -> tuple[float, float, float, float]:
    quaternion = _vec(value, 4, name)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise ValueError(f"{name} must contain unit quaternions")
    return quaternion


def _validate_geometry(geometry: Mapping[str, Any]) -> None:
    kind = geometry.get("kind")
    if kind not in _ALLOWED_KINDS:
        raise ValueError(f"unsupported geometry kind: {kind}")
    if kind == "box":
        _vec(geometry.get("size_xyz"), 3, "geometry.size_xyz", positive=True)
        return
    if kind == "cylinder":
        _finite_positive(geometry.get("radius_m"), "geometry.radius_m")
        _finite_positive(geometry.get("height_m"), "geometry.height_m")
        if geometry.get("axis") not in {"x", "y", "z"}:
            raise ValueError("cylinder axis must be x, y, or z")
        sides = geometry.get("sides")
        if not isinstance(sides, int) or isinstance(sides, bool) or sides < 3:
            raise ValueError("cylinder sides must be an integer >= 3")
        return
    parts = geometry.get("parts")
    if not isinstance(parts, list) or len(parts) < 2:
        raise ValueError("compound geometry requires at least two parts")
    names: set[str] = set()
    for part in parts:
        if not isinstance(part, Mapping):
            raise ValueError("compound part must be an object")
        name = str(part.get("name", ""))
        if not name or name in names:
            raise ValueError("compound part names must be non-empty and unique")
        names.add(name)
        shape = part.get("shape")
        if shape == "box":
            _vec(part.get("size_xyz"), 3, "part.size_xyz", positive=True)
        elif shape == "cylinder":
            _finite_positive(part.get("radius_m"), "part.radius_m")
            _finite_positive(part.get("height_m"), "part.height_m")
            if part.get("axis") not in {"x", "y", "z"}:
                raise ValueError("compound cylinder axis must be x, y, or z")
            sides = part.get("sides")
            if not isinstance(sides, int) or isinstance(sides, bool) or sides < 3:
                raise ValueError("compound cylinder sides must be >= 3")
        else:
            raise ValueError(f"unsupported compound part shape: {shape}")
        _vec(part.get("offset_xyz"), 3, "part.offset_xyz")


def geometry_half_extents(geometry: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return an axis-aligned local bounding half-extent for a recipe."""

    kind = geometry["kind"]
    if kind == "box":
        size = _vec(geometry["size_xyz"], 3, "geometry.size_xyz", positive=True)
        return tuple(component * 0.5 for component in size)
    if kind == "cylinder":
        return _cylinder_half_extents(geometry)

    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    for part in geometry["parts"]:
        offset = _vec(part["offset_xyz"], 3, "part.offset_xyz")
        if part["shape"] == "box":
            size = _vec(part["size_xyz"], 3, "part.size_xyz", positive=True)
            half = tuple(component * 0.5 for component in size)
        else:
            half = _cylinder_half_extents(part)
        for axis in range(3):
            minima[axis] = min(minima[axis], offset[axis] - half[axis])
            maxima[axis] = max(maxima[axis], offset[axis] + half[axis])
    return tuple(
        max(abs(minimum), abs(maximum))
        for minimum, maximum in zip(minima, maxima, strict=True)
    )


def _cylinder_half_extents(value: Mapping[str, Any]) -> tuple[float, float, float]:
    radius = float(value["radius_m"])
    half_height = float(value["height_m"]) * 0.5
    axis = value["axis"]
    if axis == "x":
        return (half_height, radius, radius)
    if axis == "y":
        return (radius, half_height, radius)
    return (radius, radius, half_height)


def load_object_registry(path: Path = OBJECT_REGISTRY_PATH) -> tuple[ObjectAsset, ...]:
    raw = _read_json(path)
    _assert_no_external_references(raw)
    if raw.get("schema_version") != "conveyor-bench-object-registry-v1":
        raise ValueError("unsupported object registry schema")
    if raw.get("units") != "m-kg-s":
        raise ValueError("object registry units must be m-kg-s")
    objects = raw.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("object registry must contain objects")
    result = tuple(ObjectAsset.from_dict(value) for value in objects)
    ids = [asset.object_id for asset in result]
    if len(ids) != len(set(ids)):
        raise ValueError("object_id values must be unique")
    return result


def load_receptacles(
    path: Path = RECEPTACLE_MANIFEST_PATH,
) -> tuple[ReceptacleAsset, ...]:
    raw = _read_json(path)
    _assert_no_external_references(raw)
    if raw.get("schema_version") != "conveyor-bench-receptacles-v1":
        raise ValueError("unsupported receptacle manifest schema")
    values = raw.get("receptacles")
    if not isinstance(values, list) or not values:
        raise ValueError("receptacle manifest must contain receptacles")
    result = tuple(ReceptacleAsset.from_dict(value) for value in values)
    ids = [item.zone_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("zone_id values must be unique")
    return result


def load_workcell_manifest(path: Path = WORKCELL_MANIFEST_PATH) -> dict[str, Any]:
    value = _read_json(path)
    _assert_no_external_references(value)
    if value.get("schema_version") != "conveyor-bench-procedural-workcell-v1":
        raise ValueError("unsupported workcell manifest schema")
    if value.get("runtime_dependency") != "none":
        raise ValueError("workcell may not require an external runtime asset")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_fingerprint(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Hash the executable benchmark source and configuration tree."""

    root = project_root.resolve()
    paths: list[Path] = []
    for relative_root, suffixes in (
        ("src", {".py"}),
        ("scripts", {".py"}),
        ("configs", {".json"}),
    ):
        directory = root / relative_root
        if not directory.is_dir():
            raise FileNotFoundError(
                f"source provenance directory is missing: {directory}"
            )
        paths.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )
    package_config = root / "pyproject.toml"
    if not package_config.is_file():
        raise FileNotFoundError(
            f"source provenance file is missing: {package_config}"
        )
    paths.append(package_config)

    entries = {
        path.resolve().relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(paths)
    }
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": "conveyor-bench-source-tree-v1",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": len(entries),
    }


def verify_asset_lock(
    lock_path: Path = ASSET_LOCK_PATH,
    asset_root: Path = ASSET_ROOT,
) -> dict[str, str]:
    """Validate locked asset hashes and return them keyed by relative path."""

    raw = _read_json(lock_path)
    if raw.get("schema_version") != "conveyor-bench-asset-lock-v1":
        raise ValueError("unsupported asset lock schema")
    files = raw.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("asset lock must contain files")
    result: dict[str, str] = {}
    resolved_root = asset_root.resolve()
    for relative_name, expected_digest in files.items():
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe asset lock path: {relative_name}")
        path = (resolved_root / relative_path).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"asset lock path escapes asset root: {relative_name}")
        if not path.is_file():
            raise FileNotFoundError(f"locked asset is missing: {path}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(f"asset hash mismatch: {relative_name}")
        result[relative_path.as_posix()] = actual_digest
    return result
