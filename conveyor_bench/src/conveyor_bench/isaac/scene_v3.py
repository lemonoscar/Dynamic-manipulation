"""Native NuRec Liangzhu scene layered around the canonical V1 task."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import (
    bind_physics_material,
    clone,
    create_prim,
    get_current_stage,
)
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from conveyor_bench.v1.assets import ObjectAsset

from .scene import _collision
from .scene_v1 import (
    ConveyorSceneV1Cfg,
    ProceduralWorkcellCfg,
    spawn_conveyor_workcell,
)


SCENE_ID = "transverse_dynamic_sort_liangzhu_nurec_v3"
LIANGZHU_STAGE_PRIM_PATH = "/World/LiangzhuScene"
# Observer-only side view selected inside the well-reconstructed PCT coke
# grasp area.  It keeps the complete robot and transverse conveyor visible.
V3_OVERVIEW_CAMERA_OFFSET_XYZ = (
    0.3849319648,
    -3.6261365028,
    2.1381941050,
)
V3_OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.6972261796,
    -0.1484619933,
    0.1511499216,
    0.6848272718,
)
V3_OVERVIEW_CAMERA_FAR_CLIPPING_M = 50.0
# This is the authored robot ground anchor of PCT's Liangzhu coke-grasp task,
# where the NuRec reconstruction and scanned collision are both strongest.
V3_PCT_COKE_TASK_WORKCELL_GROUND_XYZ_M = (
    -1.4849319648,
    5.1261365028,
    -0.1381941050,
)
V3_LOCAL_FLOOR_PATCH_PRIM_PATH = "/World/envs/env_0/LocalFloorPatch"
V3_OBJECT_PRIM_BASENAMES = (
    "Object00",
    "Object01",
    "Object02",
    "Object03",
    "Object04",
    "Object05",
    "Object06",
    "Object07",
)


@configclass
class V3RigidObjectCfg(RigidObjectSpawnerCfg):
    """A real USD visual with one deterministic analytic rigid fixture."""

    func: Callable = MISSING
    object_id: str = MISSING
    visual_usd_path: str = MISSING
    geometry: dict[str, Any] = MISSING
    physics_material: sim_utils.RigidBodyMaterialCfg = MISSING


@clone
def spawn_v3_rigid_object(
    prim_path: str,
    cfg: V3RigidObjectCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
):
    """Compose a sidecar visual and a stable collision proxy as one body."""

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
    create_prim(
        f"{prim_path}/Visual",
        usd_path=cfg.visual_usd_path,
        stage=stage,
    )

    collision_path = f"{prim_path}/Collision"
    geometry = cfg.geometry
    if geometry["kind"] == "cylinder":
        create_prim(
            collision_path,
            prim_type="Cylinder",
            attributes={
                "radius": float(geometry["radius_m"]),
                "height": float(geometry["height_m"]),
                "axis": str(geometry["axis"]).upper(),
            },
            stage=stage,
        )
    elif geometry["kind"] == "box":
        size = tuple(float(value) for value in geometry["size_xyz"])
        cube_size = min(size)
        create_prim(
            collision_path,
            prim_type="Cube",
            scale=tuple(value / cube_size for value in size),
            attributes={"size": cube_size},
            stage=stage,
        )
    else:
        raise ValueError(
            f"V3 fixture does not support {geometry['kind']!r} geometry"
        )
    schemas.define_collision_properties(
        collision_path, cfg.collision_props or _collision(), stage=stage
    )
    UsdGeom.Imageable(stage.GetPrimAtPath(collision_path)).MakeInvisible()
    physics_material_path = f"{prim_path}/Looks/Physics"
    cfg.physics_material.func(physics_material_path, cfg.physics_material)
    bind_physics_material(
        collision_path, physics_material_path, stage=stage
    )
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(
            prim_path, cfg.rigid_props, stage=stage
        )
    return stage.GetPrimAtPath(prim_path)


def _v3_object_cfg(
    index: int,
    asset: ObjectAsset,
    visual_usd_path: Path,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{V3_OBJECT_PRIM_BASENAMES[index]}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, -0.70 + index * 0.20, 0.20),
            rot=asset.stable_poses_wxyz[0],
        ),
        spawn=V3RigidObjectCfg(
            func=spawn_v3_rigid_object,
            object_id=asset.object_id,
            visual_usd_path=str(visual_usd_path),
            geometry=dict(asset.geometry),
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
            semantic_tags=[
                ("class", asset.category),
                ("asset_id", asset.object_id),
                ("real_twin_id", asset.real_twin_id),
            ],
        ),
    )


@configclass
class ConveyorSceneV3Cfg(ConveyorSceneV1Cfg):
    """V1 physics/cameras with Liangzhu native NuRec visual and collision."""

    # The Liangzhu collision layer owns the floor. Keeping V1's ground plane
    # would create two contact surfaces at nearly the same height.
    ground = None

    local_floor_patch = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/LocalFloorPatch",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            visible=False,
            size=(3.0, 3.4, 0.02),
            collision_props=_collision(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.85,
                restitution=0.0,
            ),
        ),
    )

    workcell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorStation",
        spawn=ProceduralWorkcellCfg(
            func=spawn_conveyor_workcell,
            include_local_sort_trays=True,
            include_room_context=False,
        ),
    )

    # Filled by ``make_conveyor_scene_v3_cfg`` only after the SSH bundle and
    # generated composition layer have passed their preflight.
    liangzhu_scene = None


def make_conveyor_scene_v3_cfg(
    runtime_layer: Path,
    *,
    object_assets: Sequence[ObjectAsset],
    object_usd_paths: Mapping[str, Path],
    num_envs: int = 1,
    env_spacing: float = 3.0,
) -> ConveyorSceneV3Cfg:
    """Build the one-environment V3 scene from a verified runtime layer."""

    runtime_layer = Path(runtime_layer).expanduser().resolve(strict=True)
    if runtime_layer.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError("V3 runtime layer must be an OpenUSD file")
    cfg = ConveyorSceneV3Cfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        replicate_physics=True,
        clone_in_fabric=False,
        lazy_sensor_update=True,
    )
    cfg.liangzhu_scene = AssetBaseCfg(
        prim_path=LIANGZHU_STAGE_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(usd_path=str(runtime_layer)),
    )
    if not object_assets or len(object_assets) > len(V3_OBJECT_PRIM_BASENAMES):
        raise ValueError("V3 object pool must contain between one and eight assets")
    for index in range(len(V3_OBJECT_PRIM_BASENAMES)):
        setattr(cfg, f"object_{index:02d}", None)
    for index, asset in enumerate(object_assets):
        try:
            usd_path = Path(object_usd_paths[asset.object_id]).resolve(
                strict=True
            )
        except KeyError as exc:
            raise KeyError(
                f"V3 object USD is missing for {asset.object_id!r}"
            ) from exc
        setattr(cfg, f"object_{index:02d}", _v3_object_cfg(index, asset, usd_path))
    filter_paths = [
        f"{{ENV_REGEX_NS}}/{V3_OBJECT_PRIM_BASENAMES[index]}"
        for index in range(len(object_assets))
    ]
    cfg.left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link7",
        update_period=0.0,
        history_length=2,
        force_threshold=0.2,
        filter_prim_paths_expr=filter_paths,
    )
    cfg.right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link8",
        update_period=0.0,
        history_length=2,
        force_threshold=0.2,
        filter_prim_paths_expr=filter_paths,
    )
    cfg.overview_camera.offset.pos = V3_OVERVIEW_CAMERA_OFFSET_XYZ
    cfg.overview_camera.offset.rot = V3_OVERVIEW_CAMERA_OFFSET_WXYZ
    cfg.overview_camera.spawn.clipping_range = (
        0.05,
        V3_OVERVIEW_CAMERA_FAR_CLIPPING_M,
    )
    return cfg


def validate_v3_object_fixtures(
    stage: Any,
    object_assets: Sequence[ObjectAsset],
    object_usd_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Fail before reset if a real visual or its rigid fixture is absent."""

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    reports: list[dict[str, Any]] = []
    for index, asset in enumerate(object_assets):
        root_path = (
            f"/World/envs/env_0/{V3_OBJECT_PRIM_BASENAMES[index]}"
        )
        root = stage.GetPrimAtPath(root_path)
        visual = stage.GetPrimAtPath(f"{root_path}/Visual")
        collision = stage.GetPrimAtPath(f"{root_path}/Collision")
        if not root.IsValid() or not visual.IsValid() or not collision.IsValid():
            raise RuntimeError(
                f"V3 real-object fixture is incomplete: {root_path}"
            )
        if not root.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"V3 object has no rigid body: {root_path}")
        if not root.HasAPI(UsdPhysics.MassAPI):
            raise RuntimeError(f"V3 object has no mass API: {root_path}")
        if not collision.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"V3 object has no collision: {root_path}")
        bounds = cache.ComputeWorldBound(visual).ComputeAlignedRange()
        size = tuple(
            float(value)
            for value in (bounds.GetMax() - bounds.GetMin())
        )
        if any(value <= 1.0e-4 or value >= 2.0 for value in size):
            raise RuntimeError(
                f"V3 object visual has implausible bounds: {asset.object_id}={size}"
            )
        reports.append(
            {
                "object_id": asset.object_id,
                "real_twin_id": asset.real_twin_id,
                "root_prim": root_path,
                "visual_prim": str(visual.GetPath()),
                "collision_prim": str(collision.GetPath()),
                "visual_usd": str(object_usd_paths[asset.object_id]),
                "visual_world_aabb_size_m": list(size),
                "collision_geometry": dict(asset.geometry),
                "mass_kg": asset.mass_kg,
                "analytic_collision_fixture": True,
            }
        )
    return {
        "fixture_count": len(reports),
        "all_rigid_bodies_valid": True,
        "all_visuals_composed": True,
        "objects": reports,
    }


def place_workcell_in_liangzhu_task_area(
    scene: Any,
    stage: Any,
    ground_xyz_m: tuple[float, float, float] = (
        V3_PCT_COKE_TASK_WORKCELL_GROUND_XYZ_M
    ),
) -> dict[str, Any]:
    """Move the task environment into PCT's calibrated coke-grasp area."""

    if len(scene.env_prim_paths) != 1:
        raise ValueError(
            "V3 NuRec placement currently requires exactly one env"
        )
    env_prim_path = scene.env_prim_paths[0]
    prim = stage.GetPrimAtPath(env_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"environment prim is missing: {env_prim_path}")

    xformable = UsdGeom.Xformable(prim)
    translate_op = next(
        (
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ),
        None,
    )
    if translate_op is None:
        translate_op = xformable.AddTranslateOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        )
    vector_type = (
        Gf.Vec3d
        if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
        else Gf.Vec3f
    )
    ground_xyz_m = tuple(float(value) for value in ground_xyz_m)
    translate_op.Set(vector_type(*ground_xyz_m))

    # Runtime reset/spawn code consumes env_origins when converting local task
    # coordinates into simulation-world poses.
    scene.env_origins[0] = scene.env_origins.new_tensor(ground_xyz_m)
    return {
        "environment_prim": env_prim_path,
        "workcell_ground_world_xyz_m": list(ground_xyz_m),
        "local_floor_patch_prim": V3_LOCAL_FLOOR_PATCH_PRIM_PATH,
        "nurec_scene_translation_xyz_m": [0.0, 0.0, 0.0],
    }


def validate_liangzhu_stage(stage: Any) -> dict[str, Any]:
    """Fail before simulation reset if NuRec or collision did not compose."""

    root = stage.GetPrimAtPath(LIANGZHU_STAGE_PRIM_PATH)
    if not root.IsValid():
        raise RuntimeError(
            f"Liangzhu runtime prim is missing: {LIANGZHU_STAGE_PRIM_PATH}"
        )

    nurec_volumes: list[str] = []
    nurec_fields: list[str] = []
    collision_meshes: list[str] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(f"{LIANGZHU_STAGE_PRIM_PATH}/"):
            continue
        marker = prim.GetAttribute("omni:nurec:isNuRecVolume")
        if marker.IsValid() and bool(marker.Get()):
            nurec_volumes.append(prim_path)
        if prim.GetTypeName() == "OmniNuRecFieldAsset":
            file_path = prim.GetAttribute("filePath")
            if not file_path.IsValid() or not str(file_path.Get()):
                raise RuntimeError(f"NuRec field has no filePath: {prim_path}")
            nurec_fields.append(prim_path)
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_meshes.append(prim_path)

    if not nurec_volumes or len(nurec_fields) < 2:
        raise RuntimeError(
            "native NuRec volume did not compose; check Isaac Sim NuRec support "
            "and the transferred USDZ"
        )
    if not collision_meshes:
        raise RuntimeError("Liangzhu collision layer contains no collision mesh")
    return {
        "scene_id": SCENE_ID,
        "root_prim": LIANGZHU_STAGE_PRIM_PATH,
        "nurec_volume_prims": nurec_volumes,
        "nurec_field_prims": nurec_fields,
        "collision_mesh_prims": collision_meshes,
        "native_registered_compositing": True,
    }


__all__ = [
    "ConveyorSceneV3Cfg",
    "LIANGZHU_STAGE_PRIM_PATH",
    "SCENE_ID",
    "V3_OVERVIEW_CAMERA_FAR_CLIPPING_M",
    "V3_OVERVIEW_CAMERA_OFFSET_XYZ",
    "V3_OVERVIEW_CAMERA_OFFSET_WXYZ",
    "V3_PCT_COKE_TASK_WORKCELL_GROUND_XYZ_M",
    "V3_LOCAL_FLOOR_PATCH_PRIM_PATH",
    "V3_OBJECT_PRIM_BASENAMES",
    "make_conveyor_scene_v3_cfg",
    "place_workcell_in_liangzhu_task_area",
    "spawn_v3_rigid_object",
    "validate_liangzhu_stage",
    "validate_v3_object_fixtures",
]
