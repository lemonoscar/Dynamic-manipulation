"""ConveyorBench V1 workcell, procedural parts and three-camera scene.

The contact-critical belt remains the simple, already validated surface-
velocity rigid body.  Realistic appearance and static workcell collision are
layered around it so visual fidelity cannot silently change conveyor physics.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Any, Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg, SpawnerCfg
from isaaclab.sim.utils import (
    bind_physics_material,
    bind_visual_material,
    clone,
    create_prim,
    get_current_stage,
)
from isaaclab.utils import configclass
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from conveyor_bench.v1.assets import ObjectAsset, load_object_registry, load_receptacles

from .asset_config import TCP_OFFSET_X_M, make_go2_x5_cfg
from .scene import _collision, _spawn_direct_cuboid


LAYOUT_ID = "transverse_dynamic_sort_station_v1"
BELT_CENTER_X_M = 0.70
BELT_CENTER_Y_M = 0.0
BELT_LENGTH_M = 1.56
BELT_WIDTH_M = 0.252
BELT_THICKNESS_M = 0.06
BELT_TOP_Z_M = 0.34
BELT_CENTER_Z_M = BELT_TOP_Z_M - BELT_THICKNESS_M * 0.5
TRANSPORT_DIRECTION_WORLD = (0.0, -1.0, 0.0)
OBJECT_SPAWN_Y_M = BELT_CENTER_Y_M + BELT_LENGTH_M * 0.5 - 0.12
OBJECT_INTERCEPT_Y_M = 0.0
OBJECT_EXIT_Y_M = BELT_CENTER_Y_M - BELT_LENGTH_M * 0.5 + 0.03
# Use the near-side lane of the narrowed belt.  The 76 mm center-to-edge
# clearance keeps every registered part inside the belt while retaining the
# calibrated X5 top-down workspace.
OBJECT_LANE_X_M = 0.65
EXIT_PLANE_POINT_WORLD = (OBJECT_LANE_X_M, OBJECT_EXIT_Y_M, BELT_TOP_Z_M)

# Exact robot-camera contract from arm-vla-grasp-sim pct_scene@c7fe62c7.
FRONT_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/base/head_cam"
HEAD_CAMERA_OFFSET_XYZ = (0.28, 0.0, 0.07)
HEAD_CAMERA_OFFSET_WXYZ = (0.5, -0.5, 0.5, -0.5)
HEAD_CAMERA_ORIENTATION_CONVENTION = "ros"
WRIST_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera"
WRIST_CAMERA_CALIBRATION_FRAME = "arm_link6_T_camera_color_optical"
WRIST_CAMERA_HAND_EYE_POS_XYZ_M = (
    0.0559054476,
    0.0026732239,
    0.0767149320,
)
WRIST_CAMERA_VISUAL_ALIGNMENT_OFFSET_CAMERA_XYZ_M = (0.0, -0.02, 0.0)
WRIST_CAMERA_OFFSET_XYZ = (
    0.0666580792,
    0.0028071889,
    0.0935779972,
)
WRIST_CAMERA_OFFSET_WXYZ = (
    0.3377891849,
    -0.6214992221,
    0.6185057335,
    -0.3421810063,
)
WRIST_CAMERA_ORIENTATION_CONVENTION = "ros"
D436_CAMERA_RESOLUTION_WH = (640, 480)
D436_CAMERA_FX_PX = 383.44608095
D436_CAMERA_FY_PX = 383.52724198
D436_CAMERA_CX_PX = 324.33479864
D436_CAMERA_CY_PX = 238.90275478
D436_CAMERA_DISTORTION_COEFFICIENTS = (0.0,) * 12
D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM = 18.0
D436_CAMERA_FALLBACK_FX_FY_PX = 383.486661465
D436_CAMERA_FALLBACK_CX_PX = 320.0
D436_CAMERA_FALLBACK_CY_PX = 240.0
D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM = 30.040158257372415
D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM = 22.530118693029312
WRIST_CAMERA_NEAR_CLIPPING_M = 0.03
# Observer-only view, deliberately farther away than V0.
OVERVIEW_CAMERA_OFFSET_XYZ = (-2.10, -1.60, 2.40)
OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.92554193,
    -0.07441319,
    0.26721596,
    0.25774106,
)
OVERVIEW_CAMERA_ORIENTATION_CONVENTION = "world"

# Linear RGB chosen to render as the dark green PVC belt used by the target
# workcell under the benchmark lights.  The contact surface and visual skin
# intentionally share this value so no grey edge leaks into camera frames.
BELT_DARK_GREEN_RGB = (0.015, 0.10, 0.035)


def enable_d436_lens_distortion_schema() -> dict[str, Any]:
    """Enable the renderer schema used by the PCT D436 calibration."""

    extension_name = "omni.usd.schema.omni_lens_distortion"
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        enabled_before = bool(manager.is_extension_enabled(extension_name))
        if not enabled_before:
            manager.set_extension_enabled_immediate(extension_name, True)
        enabled_after = bool(manager.is_extension_enabled(extension_name))
    except Exception as exc:
        return {
            "requested": True,
            "extension": extension_name,
            "enabled": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "requested": True,
        "extension": extension_name,
        "enabled_before": enabled_before,
        "enabled": enabled_after,
    }


def _apply_d436_opencv_pinhole_schema(prim: Any) -> bool:
    try:
        schema_applied = prim.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI")
    except Exception:
        return False
    if not schema_applied:
        return False
    attributes: tuple[tuple[str, Any], ...] = (
        ("omni:lensdistortion:model", "opencvPinhole"),
        (
            "omni:lensdistortion:opencvPinhole:imageSize",
            Gf.Vec2i(*D436_CAMERA_RESOLUTION_WH),
        ),
        ("omni:lensdistortion:opencvPinhole:fx", D436_CAMERA_FX_PX),
        ("omni:lensdistortion:opencvPinhole:fy", D436_CAMERA_FY_PX),
        ("omni:lensdistortion:opencvPinhole:cx", D436_CAMERA_CX_PX),
        ("omni:lensdistortion:opencvPinhole:cy", D436_CAMERA_CY_PX),
    )
    coefficient_names = (
        "k1", "k2", "p1", "p2", "k3", "k4",
        "k5", "k6", "s1", "s2", "s3", "s4",
    )
    attributes += tuple(
        (f"omni:lensdistortion:opencvPinhole:{name}", value)
        for name, value in zip(
            coefficient_names,
            D436_CAMERA_DISTORTION_COEFFICIENTS,
            strict=True,
        )
    )
    for attribute_name, value in attributes:
        attribute = prim.GetAttribute(attribute_name)
        if not attribute.IsValid() or not attribute.Set(value):
            return False
    return True


def make_d436_camera_spawn_function() -> Any:
    """Apply PCT's OpenCV schema before Isaac Lab clones the camera."""

    from isaaclab.sim.spawners.sensors.sensors import spawn_camera

    @clone
    def spawn_calibrated_d436_camera(
        prim_path: str,
        cfg: Any,
        translation: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        prim = spawn_camera.__wrapped__(
            prim_path,
            cfg,
            translation=translation,
            orientation=orientation,
            **kwargs,
        )
        _apply_d436_opencv_pinhole_schema(prim)
        return prim

    return spawn_calibrated_d436_camera


def apply_d436_runtime_intrinsics(sensor: Any) -> dict[str, Any]:
    """Keep Isaac Lab's exposed K equal to the renderer's effective K."""

    matrices = sensor._data.intrinsic_matrices
    camera_prim = sensor._sensor_prims[0].GetPrim()
    model = camera_prim.GetAttribute("omni:lensdistortion:model")
    schema_applied = bool(model.IsValid() and model.Get() == "opencvPinhole")
    fx = D436_CAMERA_FX_PX if schema_applied else D436_CAMERA_FALLBACK_FX_FY_PX
    fy = D436_CAMERA_FY_PX if schema_applied else D436_CAMERA_FALLBACK_FX_FY_PX
    cx = D436_CAMERA_CX_PX if schema_applied else D436_CAMERA_FALLBACK_CX_PX
    cy = D436_CAMERA_CY_PX if schema_applied else D436_CAMERA_FALLBACK_CY_PX
    matrices[..., :, :] = 0.0
    matrices[..., 0, 0] = fx
    matrices[..., 0, 2] = cx
    matrices[..., 1, 1] = fy
    matrices[..., 1, 2] = cy
    matrices[..., 2, 2] = 1.0
    return {
        "renderer_schema_applied": schema_applied,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }


GRIPPER_COLLISION_MODEL_ID = "pct_finray_convex_decomposition_v1"
GRIPPER_COLLISION_APPROXIMATION = "convexDecomposition"
GRIPPER_PAD_CONTACT_OFFSET_M = 0.002
GRIPPER_PAD_REST_OFFSET_M = 0.0
GRIPPER_COLLISION_LINKS = ("arm_link7", "arm_link8")

OBJECT_ASSETS = load_object_registry()
RECEPTACLE_ASSETS = load_receptacles()
OBJECT_ENTITY_NAMES = tuple(
    f"object_{index:02d}" for index in range(len(OBJECT_ASSETS))
)
OBJECT_PRIM_BASENAMES = tuple(
    f"Object{index:02d}" for index in range(len(OBJECT_ASSETS))
)

_COLORS = {
    "red": (0.84, 0.08, 0.05),
    "blue": (0.05, 0.22, 0.82),
    "yellow": (0.95, 0.68, 0.04),
    "green": (0.06, 0.56, 0.18),
    "silver": (0.62, 0.66, 0.70),
    "orange": (0.95, 0.31, 0.04),
    "purple": (0.47, 0.10, 0.72),
    "cyan": (0.04, 0.65, 0.75),
}


def apply_pct_gripper_collision_patch(
    stage: Any,
    robot_prim_path: str,
) -> dict[str, Any]:
    """Apply PCT's PhysX settings to the original FinRay collision meshes.

    The patch runs after the URDF is composed and before the first reset.  It
    never replaces or disables robot geometry, so visual and physical assets
    remain the exact PCT files.
    """

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"robot prim is missing: {robot_prim_path}")

    deinstanced: list[str] = []
    for _ in range(8):
        candidates: dict[str, Any] = {}
        try:
            prims = Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies())
        except (AttributeError, TypeError):
            prims = Usd.PrimRange(robot_prim)
        for prim in prims:
            current = prim
            while (
                current
                and current.IsValid()
                and not current.IsPseudoRoot()
            ):
                if current.IsInstance() or current.IsInstanceable():
                    candidates[str(current.GetPath())] = current
                    break
                current = current.GetParent()
        pending = [
            candidates[path]
            for path in sorted(
                candidates,
                key=lambda value: (value.count("/"), value),
            )
            if path not in deinstanced
        ]
        if not pending:
            break
        progress = False
        for prim in pending:
            path = str(prim.GetPath())
            prim.SetInstanceable(False)
            deinstanced.append(path)
            progress = True
        if not progress:
            break

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    patched: list[dict[str, Any]] = []
    patched_links: set[str] = set()
    for prim in Usd.PrimRange(robot_prim):
        path = str(prim.GetPath())
        path_segments = tuple(value for value in path.split("/") if value)
        link_name = next(
            (
                name
                for name in GRIPPER_COLLISION_LINKS
                if name in path_segments
            ),
            None,
        )
        if link_name is None:
            continue
        applied_schemas = tuple(str(value) for value in prim.GetAppliedSchemas())
        mesh_like = bool(
            str(prim.GetTypeName()) == "Mesh"
            or prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            or any(
                "PhysicsMeshCollisionAPI" in value
                for value in applied_schemas
            )
            or prim.GetAttribute("physics:approximation").IsValid()
        )
        collision_like = bool(
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)
            or any(
                marker in value
                for value in applied_schemas
                for marker in ("PhysicsCollisionAPI", "PhysxCollisionAPI")
            )
            or prim.GetAttribute("physics:collisionEnabled").IsValid()
        )
        if not (mesh_like and collision_like):
            continue

        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(prim)
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)

        attributes = (
            (
                "physics:approximation",
                Sdf.ValueTypeNames.Token,
                GRIPPER_COLLISION_APPROXIMATION,
            ),
            (
                "physxCollision:contactOffset",
                Sdf.ValueTypeNames.Float,
                GRIPPER_PAD_CONTACT_OFFSET_M,
            ),
            (
                "physxCollision:restOffset",
                Sdf.ValueTypeNames.Float,
                GRIPPER_PAD_REST_OFFSET_M,
            ),
            (
                "physics:restOffset",
                Sdf.ValueTypeNames.Float,
                GRIPPER_PAD_REST_OFFSET_M,
            ),
        )
        for name, type_name, value in attributes:
            attribute = prim.GetAttribute(name)
            if not attribute.IsValid():
                attribute = prim.CreateAttribute(
                    name,
                    type_name,
                    custom=False,
                )
            attribute.Set(value)
        patched_links.add(link_name)
        patched.append(
            {
                "prim_path": path,
                "link": link_name,
                "approximation": GRIPPER_COLLISION_APPROXIMATION,
                "contact_offset_m": GRIPPER_PAD_CONTACT_OFFSET_M,
                "rest_offset_m": GRIPPER_PAD_REST_OFFSET_M,
            }
        )

    missing_links = sorted(set(GRIPPER_COLLISION_LINKS) - patched_links)
    if missing_links:
        raise RuntimeError(
            "PCT FinRay collision mesh patch did not cover links: "
            f"{missing_links}"
        )
    return {
        "model_id": GRIPPER_COLLISION_MODEL_ID,
        "tcp_offset_x_m": TCP_OFFSET_X_M,
        "approximation": GRIPPER_COLLISION_APPROXIMATION,
        "contact_offset_m": GRIPPER_PAD_CONTACT_OFFSET_M,
        "rest_offset_m": GRIPPER_PAD_REST_OFFSET_M,
        "deinstanced_prim_paths": deinstanced,
        "patch_count": len(patched),
        "patched_prims": patched,
        "geometry_replaced": False,
    }


@configclass
class ProceduralRigidObjectCfg(RigidObjectSpawnerCfg):
    """Spawner configuration for one registry-backed rigid part."""

    func: Callable = MISSING
    object_id: str = MISSING
    geometry: dict[str, Any] = MISSING
    color_rgb: tuple[float, float, float] = MISSING
    physics_material: sim_utils.RigidBodyMaterialCfg = MISSING
    visual_material: sim_utils.PreviewSurfaceCfg = MISSING


@configclass
class ProceduralWorkcellCfg(SpawnerCfg):
    """Spawner configuration for the static station shell."""

    func: Callable = MISSING
    include_local_sort_trays: bool = True


def _make_materials(
    prim_path: str,
    visual_cfg: sim_utils.PreviewSurfaceCfg,
    physics_cfg: sim_utils.RigidBodyMaterialCfg,
) -> tuple[str, str]:
    visual_path = f"{prim_path}/Looks/visual"
    physics_path = f"{prim_path}/Looks/physics"
    visual_cfg.func(visual_path, visual_cfg)
    physics_cfg.func(physics_path, physics_cfg)
    return visual_path, physics_path


def _spawn_box_child(
    path: str,
    *,
    size_xyz: tuple[float, float, float],
    offset_xyz: tuple[float, float, float],
    collision_cfg: sim_utils.CollisionPropertiesCfg,
    visual_material_path: str,
    physics_material_path: str,
) -> None:
    cube_size = min(size_xyz)
    create_prim(
        path,
        prim_type="Cube",
        translation=offset_xyz,
        scale=tuple(component / cube_size for component in size_xyz),
        attributes={"size": cube_size},
    )
    schemas.define_collision_properties(path, collision_cfg)
    bind_visual_material(path, visual_material_path)
    bind_physics_material(path, physics_material_path)


def _spawn_cylinder_child(
    path: str,
    *,
    radius_m: float,
    height_m: float,
    axis: str,
    sides: int,
    offset_xyz: tuple[float, float, float],
    collision_cfg: sim_utils.CollisionPropertiesCfg,
    visual_material_path: str,
    physics_material_path: str,
) -> None:
    if sides >= 24:
        create_prim(
            path,
            prim_type="Cylinder",
            translation=offset_xyz,
            attributes={
                "radius": radius_m,
                "height": height_m,
                "axis": axis.upper(),
            },
        )
    else:
        if axis != "z":
            raise ValueError("polygonal procedural prisms currently require z axis")
        _spawn_polygonal_prism(
            path,
            radius_m=radius_m,
            height_m=height_m,
            sides=sides,
            offset_xyz=offset_xyz,
        )
    schemas.define_collision_properties(path, collision_cfg)
    prim = get_current_stage().GetPrimAtPath(path)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "convexHull"
        )
    bind_visual_material(path, visual_material_path)
    bind_physics_material(path, physics_material_path)


def _spawn_polygonal_prism(
    path: str,
    *,
    radius_m: float,
    height_m: float,
    sides: int,
    offset_xyz: tuple[float, float, float],
) -> None:
    create_prim(path, prim_type="Mesh", translation=offset_xyz)
    mesh = UsdGeom.Mesh(get_current_stage().GetPrimAtPath(path))
    half_height = height_m * 0.5
    lower = [
        Gf.Vec3f(
            radius_m * math.cos(2.0 * math.pi * index / sides),
            radius_m * math.sin(2.0 * math.pi * index / sides),
            -half_height,
        )
        for index in range(sides)
    ]
    upper = [Gf.Vec3f(point[0], point[1], half_height) for point in lower]
    points = lower + upper
    face_counts = [sides, sides] + [4] * sides
    face_indices: list[int] = [
        *reversed(range(sides)),
        *range(sides, 2 * sides),
    ]
    for index in range(sides):
        next_index = (index + 1) % sides
        face_indices.extend(
            (index, next_index, sides + next_index, sides + index)
        )
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")


@clone
def spawn_procedural_object(
    prim_path: str,
    cfg: ProceduralRigidObjectCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
):
    """Build one compound rigid body from its frozen registry recipe."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: {prim_path}")
    create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    visual_path, physics_path = _make_materials(
        prim_path,
        cfg.visual_material,
        cfg.physics_material,
    )
    collision_cfg = cfg.collision_props or _collision()

    geometry = cfg.geometry
    kind = geometry["kind"]
    if kind == "compound":
        parts = list(geometry["parts"])
    else:
        part = dict(geometry)
        part["shape"] = kind
        part["name"] = "main"
        part["offset_xyz"] = [0.0, 0.0, 0.0]
        parts = [part]

    for part in parts:
        child_path = f"{prim_path}/geometry/{part['name']}"
        offset = tuple(float(value) for value in part["offset_xyz"])
        if part["shape"] == "box":
            _spawn_box_child(
                child_path,
                size_xyz=tuple(float(value) for value in part["size_xyz"]),
                offset_xyz=offset,
                collision_cfg=collision_cfg,
                visual_material_path=visual_path,
                physics_material_path=physics_path,
            )
        elif part["shape"] == "cylinder":
            _spawn_cylinder_child(
                child_path,
                radius_m=float(part["radius_m"]),
                height_m=float(part["height_m"]),
                axis=str(part["axis"]),
                sides=int(part["sides"]),
                offset_xyz=offset,
                collision_cfg=collision_cfg,
                visual_material_path=visual_path,
                physics_material_path=physics_path,
            )
        else:
            raise ValueError(f"unsupported part shape: {part['shape']}")

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    return stage.GetPrimAtPath(prim_path)


def _static_visual_material(path: str, color: tuple[float, float, float]) -> str:
    material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color,
        roughness=0.64,
        metallic=0.05,
    )
    material.func(path, material)
    return path


def _static_box(
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material_path: str,
    *,
    collision: bool = True,
) -> None:
    cube_size = min(size)
    create_prim(
        path,
        prim_type="Cube",
        translation=center,
        scale=tuple(component / cube_size for component in size),
        attributes={"size": cube_size},
    )
    if collision:
        schemas.define_collision_properties(path, _collision())
    bind_visual_material(path, material_path)


def _static_cylinder(
    path: str,
    center: tuple[float, float, float],
    *,
    radius: float,
    height: float,
    axis: str,
    material_path: str,
) -> None:
    create_prim(
        path,
        prim_type="Cylinder",
        translation=center,
        attributes={"radius": radius, "height": height, "axis": axis.upper()},
    )
    bind_visual_material(path, material_path)


def _spawn_sort_tray(
    root: str,
    *,
    center_xyz: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    material = _static_visual_material(f"{root}/Looks/tray", color)
    center_x, center_y, center_z = center_xyz
    # Receptacle centers are defined 55 mm above the tray floor.  Computing
    # the floor from the manifest-driven center preserves the V1 value
    # (0.40 -> 0.345 m) while allowing the remote mobile workcell to use a
    # higher, well-conditioned X5 release surface.
    floor_top = center_z - 0.055
    floor_thickness = 0.020
    outer_x = 0.22
    outer_y = 0.27
    wall_t = 0.018
    wall_h = 0.10
    _static_box(
        f"{root}/floor",
        (center_x, center_y, floor_top - floor_thickness * 0.5),
        (outer_x, outer_y, floor_thickness),
        material,
    )
    wall_z = floor_top + wall_h * 0.5
    for name, center, size in (
        (
            "wall_near",
            (center_x - outer_x * 0.5 + wall_t * 0.5, center_y, wall_z),
            (wall_t, outer_y, wall_h),
        ),
        (
            "wall_far",
            (center_x + outer_x * 0.5 - wall_t * 0.5, center_y, wall_z),
            (wall_t, outer_y, wall_h),
        ),
        (
            "wall_left",
            (center_x, center_y + outer_y * 0.5 - wall_t * 0.5, wall_z),
            (outer_x, wall_t, wall_h),
        ),
        (
            "wall_right",
            (center_x, center_y - outer_y * 0.5 + wall_t * 0.5, wall_z),
            (outer_x, wall_t, wall_h),
        ),
    ):
        _static_box(f"{root}/{name}", center, size, material)


@clone
def spawn_conveyor_workcell(
    prim_path: str,
    cfg: ProceduralWorkcellCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
):
    """Build frame, rollers, guards, bins and industrial context locally."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: {prim_path}")
    create_prim(
        prim_path,
        prim_type="Xform",
        translation=translation,
        orientation=orientation,
        stage=stage,
    )
    steel = _static_visual_material(
        f"{prim_path}/Looks/steel", (0.20, 0.23, 0.27)
    )
    aluminum = _static_visual_material(
        f"{prim_path}/Looks/aluminum", (0.48, 0.52, 0.56)
    )
    rubber = _static_visual_material(
        f"{prim_path}/Looks/rubber", BELT_DARK_GREEN_RGB
    )
    yellow = _static_visual_material(
        f"{prim_path}/Looks/safety_yellow", (0.95, 0.62, 0.04)
    )
    red = _static_visual_material(
        f"{prim_path}/Looks/safety_red", (0.82, 0.04, 0.025)
    )

    half_length = BELT_LENGTH_M * 0.5
    half_width = BELT_WIDTH_M * 0.5
    near_edge_x = BELT_CENTER_X_M - half_width
    far_edge_x = BELT_CENTER_X_M + half_width
    beam_x_by_name = (
        ("near_beam", near_edge_x - 0.020),
        ("far_beam", far_edge_x + 0.020),
    )
    frame_beam_z = BELT_TOP_Z_M - 0.095
    support_leg_height = frame_beam_z - 0.005
    support_y = half_length - 0.13

    # Load-bearing frame below the independently simulated transport surface.
    for name, x in beam_x_by_name:
        _static_box(
            f"{prim_path}/frame/{name}",
            (x, 0.0, frame_beam_z),
            (0.060, BELT_LENGTH_M + 0.060, 0.090),
            steel,
        )
    for x_name, x in (
        ("near", near_edge_x - 0.020),
        ("far", far_edge_x + 0.020),
    ):
        for y_name, y in (
            ("upstream", support_y),
            ("downstream", -support_y),
        ):
            _static_box(
                f"{prim_path}/frame/leg_{x_name}_{y_name}",
                (x, y, support_leg_height * 0.5),
                (0.060, 0.060, support_leg_height),
                steel,
            )
    for name, y in (
        ("upstream", support_y),
        ("downstream", -support_y),
    ):
        _static_box(
            f"{prim_path}/frame/cross_brace_{name}",
            (BELT_CENTER_X_M, y, support_leg_height * 0.66),
            (BELT_WIDTH_M + 0.10, 0.050, 0.050),
            aluminum,
        )

    # Visual-only belt skin, seam markers and rollers. The collider under them
    # remains the direct rigid cube with surface velocity.
    _static_box(
        f"{prim_path}/belt/visual_skin",
        (BELT_CENTER_X_M, BELT_CENTER_Y_M, BELT_TOP_Z_M - 0.004),
        (BELT_WIDTH_M - 0.008, BELT_LENGTH_M - 0.012, 0.006),
        rubber,
        collision=False,
    )
    for index, y in enumerate(
        (-BELT_LENGTH_M * 0.25, 0.0, BELT_LENGTH_M * 0.25)
    ):
        _static_box(
            f"{prim_path}/belt/seam_{index}",
            (BELT_CENTER_X_M, y, BELT_TOP_Z_M + 0.001),
            (BELT_WIDTH_M - 0.015, 0.006, 0.002),
            aluminum,
            collision=False,
        )
    for name, y in (
        ("drive", -half_length + 0.035),
        ("idler", half_length - 0.035),
    ):
        _static_cylinder(
            f"{prim_path}/rollers/{name}",
            (BELT_CENTER_X_M, y, BELT_CENTER_Z_M - 0.005),
            radius=0.055,
            height=BELT_WIDTH_M + 0.045,
            axis="X",
            material_path=aluminum,
        )

    # Far-side rail protects the workcell without blocking the robot-facing edge.
    _static_box(
        f"{prim_path}/guards/far_rail",
        (far_edge_x + 0.045, 0.0, BELT_TOP_Z_M + 0.055),
        (0.035, BELT_LENGTH_M - 0.020, 0.055),
        yellow,
    )
    _static_box(
        f"{prim_path}/motor/guard",
        (
            far_edge_x + 0.090,
            half_length - 0.17,
            BELT_TOP_Z_M - 0.11,
        ),
        (0.15, 0.22, 0.22),
        steel,
    )
    _static_cylinder(
        f"{prim_path}/motor/shaft",
        (
            far_edge_x + 0.030,
            half_length - 0.11,
            BELT_TOP_Z_M - 0.05,
        ),
        radius=0.030,
        height=0.10,
        axis="Y",
        material_path=aluminum,
    )
    _static_box(
        f"{prim_path}/controls/emergency_stop_base",
        (
            far_edge_x + 0.095,
            half_length - 0.28,
            BELT_TOP_Z_M + 0.035,
        ),
        (0.065, 0.055, 0.055),
        yellow,
        collision=False,
    )
    _static_cylinder(
        f"{prim_path}/controls/emergency_stop_button",
        (
            far_edge_x + 0.060,
            half_length - 0.28,
            BELT_TOP_Z_M + 0.035,
        ),
        radius=0.027,
        height=0.025,
        axis="X",
        material_path=red,
    )
    for name, y in (("entry", support_y), ("exit", -support_y)):
        _static_box(
            f"{prim_path}/sensors/{name}_photoeye",
            (near_edge_x - 0.035, y, BELT_TOP_Z_M + 0.045),
            (0.025, 0.035, 0.065),
            aluminum,
            collision=False,
        )
    _static_box(
        f"{prim_path}/markers/exit_line",
        (BELT_CENTER_X_M, OBJECT_EXIT_Y_M, BELT_TOP_Z_M + 0.002),
        (BELT_WIDTH_M - 0.012, 0.010, 0.003),
        red,
        collision=False,
    )

    # Two semantically distinct sorting trays on the robot side. Scene variants
    # may omit them to leave a clear mobile-delivery corridor; V1 keeps them by
    # default.
    if cfg.include_local_sort_trays:
        sort_zones = {
            zone.zone_id: zone
            for zone in RECEPTACLE_ASSETS
            if zone.zone_id.startswith("sort_bin_")
        }
        for zone_id in ("sort_bin_blue", "sort_bin_yellow"):
            zone = sort_zones[zone_id]
            _spawn_sort_tray(
                f"{prim_path}/receptacles/{zone_id}",
                center_xyz=zone.center_xyz_m,
                color=zone.color_rgb,
            )

    # Downstream catch tray is post-exit and therefore cannot turn a missed
    # target into a successful placement.
    catch = next(
        zone for zone in RECEPTACLE_ASSETS if zone.zone_id == "reject_catch"
    )
    catch_material = _static_visual_material(
        f"{prim_path}/Looks/catch", catch.color_rgb
    )
    _static_box(
        f"{prim_path}/receptacles/reject_catch/floor",
        (
            catch.center_xyz_m[0],
            catch.center_xyz_m[1],
            catch.floor_top_z_m - 0.0125,
        ),
        (0.42, 0.25, 0.025),
        catch_material,
    )
    for name, x in (
        ("near", catch.center_xyz_m[0] - 0.20),
        ("far", catch.center_xyz_m[0] + 0.20),
    ):
        _static_box(
            f"{prim_path}/receptacles/reject_catch/wall_x_{name}",
            (
                x,
                catch.center_xyz_m[1],
                catch.floor_top_z_m + 0.060,
            ),
            (0.025, 0.25, 0.15),
            catch_material,
        )
    _static_box(
        f"{prim_path}/receptacles/reject_catch/end_wall",
        (
            catch.center_xyz_m[0],
            catch.center_xyz_m[1] - 0.11,
            catch.floor_top_z_m + 0.060,
        ),
        (0.42, 0.025, 0.15),
        catch_material,
    )

    # Simple industrial room context; these are deliberately outside robot reach.
    wall = _static_visual_material(
        f"{prim_path}/Looks/wall", (0.63, 0.66, 0.70)
    )
    _static_box(
        f"{prim_path}/room/back_wall",
        (2.15, 0.0, 1.25),
        (0.08, 3.40, 2.50),
        wall,
    )
    _static_box(
        f"{prim_path}/room/side_wall",
        (0.75, 1.70, 1.25),
        (2.85, 0.08, 2.50),
        wall,
    )
    return stage.GetPrimAtPath(prim_path)


def _object_cfg(index: int, asset: ObjectAsset) -> RigidObjectCfg:
    color_name = asset.attributes["color"]
    color = _COLORS[color_name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{OBJECT_PRIM_BASENAMES[index]}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, -0.70 + index * 0.20, 0.20),
            rot=asset.stable_poses_wxyz[0],
        ),
        spawn=ProceduralRigidObjectCfg(
            func=spawn_procedural_object,
            object_id=asset.object_id,
            geometry=dict(asset.geometry),
            color_rgb=color,
            mass_props=sim_utils.MassPropertiesCfg(mass=asset.mass_kg),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                angular_damping=asset.angular_damping,
                max_linear_velocity=5.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=2.0,
            ),
            collision_props=_collision(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=asset.static_friction,
                dynamic_friction=asset.dynamic_friction,
                restitution=asset.restitution,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.38,
                metallic=0.22
                if asset.attributes["material"] != "polymer"
                else 0.0,
            ),
            semantic_tags=[
                ("class", asset.category),
                ("asset_id", asset.object_id),
                ("color", color_name),
            ],
        ),
    )


@configclass
class ConveyorSceneV1Cfg(InteractiveSceneCfg):
    """Complete single-workcell V1 scene with an eight-object reusable pool."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.24, 0.25, 0.27),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.85,
                restitution=0.0,
            ),
        ),
    )

    robot = make_go2_x5_cfg()

    conveyor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TransportSurface",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BELT_CENTER_X_M, BELT_CENTER_Y_M, BELT_CENTER_Z_M)
        ),
        spawn=sim_utils.CuboidCfg(
            func=_spawn_direct_cuboid,
            size=(BELT_WIDTH_M, BELT_LENGTH_M, BELT_THICKNESS_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=2.0,
            ),
            collision_props=_collision(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=BELT_DARK_GREEN_RGB,
                roughness=0.78,
            ),
        ),
    )

    workcell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorStation",
        spawn=ProceduralWorkcellCfg(func=spawn_conveyor_workcell),
    )

    object_00 = _object_cfg(0, OBJECT_ASSETS[0])
    object_01 = _object_cfg(1, OBJECT_ASSETS[1])
    object_02 = _object_cfg(2, OBJECT_ASSETS[2])
    object_03 = _object_cfg(3, OBJECT_ASSETS[3])
    object_04 = _object_cfg(4, OBJECT_ASSETS[4])
    object_05 = _object_cfg(5, OBJECT_ASSETS[5])
    object_06 = _object_cfg(6, OBJECT_ASSETS[6])
    object_07 = _object_cfg(7, OBJECT_ASSETS[7])

    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link7",
        update_period=0.0,
        history_length=2,
        force_threshold=0.2,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/{name}" for name in OBJECT_PRIM_BASENAMES
        ],
    )
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link8",
        update_period=0.0,
        history_length=2,
        force_threshold=0.2,
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/{name}" for name in OBJECT_PRIM_BASENAMES
        ],
    )

    head_camera = CameraCfg(
        prim_path=FRONT_CAMERA_PRIM_PATH,
        update_period=0.0,
        height=D436_CAMERA_RESOLUTION_WH[1],
        width=D436_CAMERA_RESOLUTION_WH[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            func=make_d436_camera_spawn_function(),
            focal_length=D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM,
            vertical_aperture=D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=HEAD_CAMERA_OFFSET_XYZ,
            rot=HEAD_CAMERA_OFFSET_WXYZ,
            convention=HEAD_CAMERA_ORIENTATION_CONVENTION,
        ),
    )

    wrist_camera = CameraCfg(
        prim_path=WRIST_CAMERA_PRIM_PATH,
        update_period=0.0,
        height=D436_CAMERA_RESOLUTION_WH[1],
        width=D436_CAMERA_RESOLUTION_WH[0],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            func=make_d436_camera_spawn_function(),
            focal_length=D436_CAMERA_FALLBACK_FOCAL_LENGTH_MM,
            focus_distance=400.0,
            horizontal_aperture=D436_CAMERA_FALLBACK_HORIZONTAL_APERTURE_MM,
            vertical_aperture=D436_CAMERA_FALLBACK_VERTICAL_APERTURE_MM,
            clipping_range=(WRIST_CAMERA_NEAR_CLIPPING_M, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=WRIST_CAMERA_OFFSET_XYZ,
            rot=WRIST_CAMERA_OFFSET_WXYZ,
            convention=WRIST_CAMERA_ORIENTATION_CONVENTION,
        ),
    )

    overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCameraV1",
        update_period=1.0 / 25.0,
        height=320,
        width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=3.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 8.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=OVERVIEW_CAMERA_OFFSET_XYZ,
            rot=OVERVIEW_CAMERA_OFFSET_WXYZ,
            convention=OVERVIEW_CAMERA_ORIENTATION_CONVENTION,
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.89, 0.92, 1.0),
            intensity=1450.0,
        ),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4, -1.2, 2.4),
            rot=(0.9239, 0.0, 0.3827, 0.0),
        ),
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.93, 0.83),
            intensity=850.0,
            angle=2.0,
        ),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 1.1, 1.8),
            rot=(0.9239, 0.0, -0.3827, 0.0),
        ),
        spawn=sim_utils.DistantLightCfg(
            color=(0.72, 0.84, 1.0),
            intensity=420.0,
            angle=3.0,
        ),
    )
