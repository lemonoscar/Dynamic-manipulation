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
from pxr import Gf, UsdGeom, UsdPhysics

from conveyor_bench.v1.assets import ObjectAsset, load_object_registry, load_receptacles

from .asset_config import make_go2_x5_cfg
from .scene import _collision, _spawn_direct_cuboid


LAYOUT_ID = "transverse_dynamic_sort_station_v1"
BELT_CENTER_X_M = 0.70
BELT_CENTER_Y_M = 0.0
BELT_LENGTH_M = 1.20
BELT_WIDTH_M = 0.42
BELT_THICKNESS_M = 0.06
BELT_TOP_Z_M = 0.50
BELT_CENTER_Z_M = BELT_TOP_Z_M - BELT_THICKNESS_M * 0.5
TRANSPORT_DIRECTION_WORLD = (0.0, -1.0, 0.0)
OBJECT_SPAWN_Y_M = 0.48
OBJECT_INTERCEPT_Y_M = 0.0
OBJECT_EXIT_Y_M = -0.57
# Use the near-side lane of the 0.42 m belt.  This leaves 0.16 m of belt
# between the part center and the near rail while keeping the part inside the
# stable X5 workspace with its corrected, pad-centered TCP.
OBJECT_LANE_X_M = 0.65
EXIT_PLANE_POINT_WORLD = (OBJECT_LANE_X_M, OBJECT_EXIT_Y_M, BELT_TOP_Z_M)

# The dog-head camera remains straight ahead as requested, but a short local
# bracket raises it above the 0.50 m belt so it can actually see candidate
# objects.  It is still rigidly attached to the robot base.
HEAD_CAMERA_OFFSET_XYZ = (0.355, 0.0, 0.18)
HEAD_CAMERA_OFFSET_WXYZ = (1.0, 0.0, 0.0, 0.0)
# Wrist camera: centered above the gripper.  The local +X optical axis is
# pitched down by 42.5 degrees so it intersects the benchmark TCP at
# ``arm_link6 + (0.125, 0, 0)`` instead of looking over the belt.
WRIST_CAMERA_OFFSET_XYZ = (0.025, 0.0, 0.115)
WRIST_CAMERA_OFFSET_WXYZ = (0.93200787, 0.0, 0.36243804, 0.0)
# Observer-only view, deliberately farther away than V0.
OVERVIEW_CAMERA_OFFSET_XYZ = (-2.10, -1.60, 2.40)
OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.92554193,
    -0.07441319,
    0.26721596,
    0.25774106,
)

# The vendored FinRay finger meshes contain a wide mounting flange and a
# curved, open finger. Isaac Sim imports each complete mesh as one convex
# collision hull, filling the open space between the flange and contact pad.
# That artificial wedge strikes a part while the gripper is still open and
# roughly 50 mm above the pad-centred TCP. Keep the detailed mesh for
# rendering, but replace only its collision with thin boxes measured at the
# usable parallel contact-pad section of link7/link8.
GRIPPER_COLLISION_MODEL_ID = "x5_finray_parallel_pad_proxy_v1"
GRIPPER_PAD_STATIC_FRICTION = 1.35
GRIPPER_PAD_DYNAMIC_FRICTION = 1.15
GRIPPER_PAD_RESTITUTION = 0.0
GRIPPER_PAD_CONTACT_OFFSET_M = 0.002
GRIPPER_PAD_REST_OFFSET_M = 0.0
GRIPPER_PAD_PROXY_BY_LINK = {
    # Link-local xyz; the joint origins are at y=+/-0.0249 m and the two
    # prismatic axes move outward. Mirrored centres therefore preserve the
    # real closed-jaw gap and the 0.125 m link6-to-TCP calibration.
    "arm_link7": {
        "center_xyz_m": (0.03843, -0.01195, -0.00279),
        "size_xyz_m": (0.040, 0.004, 0.034),
    },
    "arm_link8": {
        "center_xyz_m": (0.03843, 0.01195, -0.00279),
        "size_xyz_m": (0.040, 0.004, 0.034),
    },
}

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


def install_gripper_collision_proxies(
    stage: Any,
    robot_prim_path: str,
) -> dict[str, Any]:
    """Replace convex FinRay finger hulls with measured thin pad colliders.

    This must run after the robot USD has been composed and before the first
    simulation reset, when PhysX builds the articulation. The collision
    proxies remain children of the original finger rigid bodies, so contact
    sensors attached to ``arm_link7`` and ``arm_link8`` continue to report
    filtered object contacts.
    """

    material_path = "/World/Looks/BenchmarkGripperPadPhysicsV1"
    if not stage.GetPrimAtPath(material_path).IsValid():
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=GRIPPER_PAD_STATIC_FRICTION,
            dynamic_friction=GRIPPER_PAD_DYNAMIC_FRICTION,
            restitution=GRIPPER_PAD_RESTITUTION,
        )
        material_cfg.func(material_path, material_cfg)

    installed: dict[str, dict[str, Any]] = {}
    for link_name, spec in GRIPPER_PAD_PROXY_BY_LINK.items():
        link_path = f"{robot_prim_path}/{link_name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            raise RuntimeError(
                f"gripper collision proxy parent is missing: {link_path}"
            )
        if not link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "gripper collision proxy parent is not a rigid body: "
                f"{link_path}"
            )

        original_path = f"{link_path}/collisions"
        original_prim = stage.GetPrimAtPath(original_path)
        if not original_prim.IsValid():
            raise RuntimeError(
                f"original gripper collision prim is missing: {original_path}"
            )
        original_prim.SetActive(False)

        proxy_path = f"{link_path}/benchmark_pad_collision_v1"
        if stage.GetPrimAtPath(proxy_path).IsValid():
            raise RuntimeError(
                f"gripper collision proxy already exists: {proxy_path}"
            )
        size_xyz = tuple(float(value) for value in spec["size_xyz_m"])
        cube_size = min(size_xyz)
        create_prim(
            proxy_path,
            prim_type="Cube",
            translation=tuple(
                float(value) for value in spec["center_xyz_m"]
            ),
            scale=tuple(value / cube_size for value in size_xyz),
            attributes={"size": cube_size},
            stage=stage,
        )
        schemas.define_collision_properties(
            proxy_path,
            sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=GRIPPER_PAD_CONTACT_OFFSET_M,
                rest_offset=GRIPPER_PAD_REST_OFFSET_M,
            ),
            stage=stage,
        )
        bind_physics_material(proxy_path, material_path, stage=stage)
        proxy_prim = stage.GetPrimAtPath(proxy_path)
        if proxy_prim.HasAPI(UsdPhysics.RigidBodyAPI) or proxy_prim.HasAPI(
            UsdPhysics.MassAPI
        ):
            raise RuntimeError(
                "gripper pad proxy must remain a compound shape on its "
                f"finger link: {proxy_path}"
            )
        UsdGeom.Imageable(proxy_prim).MakeInvisible()
        installed[link_name] = {
            "original_collision_path": original_path,
            "proxy_path": proxy_path,
            "center_xyz_m": list(spec["center_xyz_m"]),
            "size_xyz_m": list(size_xyz),
        }

    return {
        "model_id": GRIPPER_COLLISION_MODEL_ID,
        "tcp_offset_x_m": 0.125,
        "collision": {
            "contact_offset_m": GRIPPER_PAD_CONTACT_OFFSET_M,
            "rest_offset_m": GRIPPER_PAD_REST_OFFSET_M,
        },
        "physics_material": {
            "static_friction": GRIPPER_PAD_STATIC_FRICTION,
            "dynamic_friction": GRIPPER_PAD_DYNAMIC_FRICTION,
            "restitution": GRIPPER_PAD_RESTITUTION,
        },
        "topology": {
            "parent_link_is_rigid_body": True,
            "proxy_is_compound_shape": True,
            "proxy_has_rigid_body_api": False,
            "proxy_has_mass_api": False,
        },
        "links": installed,
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
        f"{prim_path}/Looks/rubber", (0.035, 0.045, 0.055)
    )
    yellow = _static_visual_material(
        f"{prim_path}/Looks/safety_yellow", (0.95, 0.62, 0.04)
    )
    red = _static_visual_material(
        f"{prim_path}/Looks/safety_red", (0.82, 0.04, 0.025)
    )

    # Load-bearing frame below the independently simulated transport surface.
    for name, x in (("near_beam", 0.47), ("far_beam", 0.93)):
        _static_box(
            f"{prim_path}/frame/{name}",
            (x, 0.0, 0.405),
            (0.060, 1.26, 0.090),
            steel,
        )
    for x_name, x in (("near", 0.47), ("far", 0.93)):
        for y_name, y in (("upstream", 0.47), ("downstream", -0.47)):
            _static_box(
                f"{prim_path}/frame/leg_{x_name}_{y_name}",
                (x, y, 0.20),
                (0.060, 0.060, 0.40),
                steel,
            )
    for name, y in (("upstream", 0.47), ("downstream", -0.47)):
        _static_box(
            f"{prim_path}/frame/cross_brace_{name}",
            (BELT_CENTER_X_M, y, 0.265),
            (0.52, 0.050, 0.050),
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
    for index, y in enumerate((-0.30, 0.0, 0.30)):
        _static_box(
            f"{prim_path}/belt/seam_{index}",
            (BELT_CENTER_X_M, y, BELT_TOP_Z_M + 0.001),
            (BELT_WIDTH_M - 0.015, 0.006, 0.002),
            aluminum,
            collision=False,
        )
    for name, y in (("drive", -0.565), ("idler", 0.565)):
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
        (0.955, 0.0, 0.555),
        (0.035, 1.18, 0.055),
        yellow,
    )
    _static_box(
        f"{prim_path}/motor/guard",
        (1.00, 0.43, 0.39),
        (0.15, 0.22, 0.22),
        steel,
    )
    _static_cylinder(
        f"{prim_path}/motor/shaft",
        (0.94, 0.49, 0.45),
        radius=0.030,
        height=0.10,
        axis="Y",
        material_path=aluminum,
    )
    _static_box(
        f"{prim_path}/controls/emergency_stop_base",
        (1.005, 0.32, 0.535),
        (0.065, 0.055, 0.055),
        yellow,
        collision=False,
    )
    _static_cylinder(
        f"{prim_path}/controls/emergency_stop_button",
        (0.97, 0.32, 0.535),
        radius=0.027,
        height=0.025,
        axis="X",
        material_path=red,
    )
    for name, y in (("entry", 0.47), ("exit", -0.47)):
        _static_box(
            f"{prim_path}/sensors/{name}_photoeye",
            (0.455, y, 0.545),
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
        (catch.center_xyz_m[0], catch.center_xyz_m[1], 0.33),
        (0.42, 0.25, 0.025),
        catch_material,
    )
    for name, x in (("near", 0.50), ("far", 0.90)):
        _static_box(
            f"{prim_path}/receptacles/reject_catch/wall_x_{name}",
            (x, catch.center_xyz_m[1], 0.40),
            (0.025, 0.25, 0.15),
            catch_material,
        )
    _static_box(
        f"{prim_path}/receptacles/reject_catch/end_wall",
        (catch.center_xyz_m[0], -0.86, 0.40),
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
                diffuse_color=(0.045, 0.055, 0.065),
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
        prim_path="{ENV_REGEX_NS}/Robot/base/conveyor_head_camera_v1",
        update_period=1.0 / 25.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=1.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=HEAD_CAMERA_OFFSET_XYZ,
            rot=HEAD_CAMERA_OFFSET_WXYZ,
            convention="world",
        ),
    )

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link6/conveyor_wrist_camera_v1",
        update_period=1.0 / 25.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.8,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 3.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=WRIST_CAMERA_OFFSET_XYZ,
            rot=WRIST_CAMERA_OFFSET_WXYZ,
            convention="world",
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
            convention="world",
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
