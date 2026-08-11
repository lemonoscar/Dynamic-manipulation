"""Native NuRec Liangzhu scene layered around the canonical V1 task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom, UsdPhysics

from .scene import _collision
from .scene_v1 import (
    BELT_CENTER_X_M,
    BELT_CENTER_Y_M,
    BELT_CENTER_Z_M,
    BELT_DARK_GREEN_RGB,
    BELT_LENGTH_M,
    BELT_THICKNESS_M,
    BELT_WIDTH_M,
    ConveyorSceneV1Cfg,
    ProceduralWorkcellCfg,
    spawn_conveyor_workcell,
)


SCENE_ID = "transverse_dynamic_sort_liangzhu_nurec_v3"
LIANGZHU_STAGE_PRIM_PATH = "/World/LiangzhuScene"
# Observer-only view at about 1.5 times the V1 target distance.  It remains
# below the lowest candidate-room ceiling while keeping the complete workcell
# centered in view.
V3_OVERVIEW_CAMERA_OFFSET_XYZ = (-4.00, -3.00, 2.10)
V3_OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.9491926869,
    -0.0407995283,
    0.1384975349,
    0.2796195172,
)
V3_OVERVIEW_CAMERA_FAR_CLIPPING_M = 50.0
# Horizontal collision slices identify this as the center of the large empty
# rectangular room.  Its scanned floor has holes, so a small invisible local
# collision patch supports the task without altering the NuRec RGB scene.
V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M = (0.0, 14.5, -0.14)
V3_LOCAL_FLOOR_PATCH_PRIM_PATH = "/World/envs/env_0/LocalFloorPatch"
V3_CONVEYOR_PRIM_PATH = "/World/TransportSurfaceV3"


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

    # Keep the kinematic surface outside the translated environment parent.
    # Its world pose is explicit, avoiding PhysX's nested-rigid-body frame
    # ambiguity while preserving the canonical surface-velocity mechanism.
    conveyor = RigidObjectCfg(
        prim_path=V3_CONVEYOR_PRIM_PATH,
        collision_group=-1,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M[0] + BELT_CENTER_X_M,
                V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M[1] + BELT_CENTER_Y_M,
                V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M[2] + BELT_CENTER_Z_M,
            )
        ),
        spawn=sim_utils.CuboidCfg(
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
    cfg.overview_camera.offset.pos = V3_OVERVIEW_CAMERA_OFFSET_XYZ
    cfg.overview_camera.offset.rot = V3_OVERVIEW_CAMERA_OFFSET_WXYZ
    cfg.overview_camera.spawn.clipping_range = (
        0.05,
        V3_OVERVIEW_CAMERA_FAR_CLIPPING_M,
    )
    return cfg


def place_workcell_in_liangzhu_open_room(
    scene: Any,
    stage: Any,
    ground_xyz_m: tuple[float, float, float] = (
        V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M
    ),
) -> dict[str, Any]:
    """Move the one task environment into the calibrated NuRec open room."""

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


def describe_v3_conveyor_world_pose(scene: Any) -> dict[str, Any]:
    """Report the global belt pose kept outside the translated env parent."""

    position = scene["conveyor"].data.default_root_state[0, :3]
    return {
        "position_world_xyz_m": [
            float(value) for value in position.detach().cpu().tolist()
        ],
        "source": "global_kinematic_surface",
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
    "V3_OPEN_ROOM_WORKCELL_GROUND_XYZ_M",
    "V3_LOCAL_FLOOR_PATCH_PRIM_PATH",
    "V3_CONVEYOR_PRIM_PATH",
    "make_conveyor_scene_v3_cfg",
    "place_workcell_in_liangzhu_open_room",
    "describe_v3_conveyor_world_pose",
    "validate_liangzhu_stage",
]
