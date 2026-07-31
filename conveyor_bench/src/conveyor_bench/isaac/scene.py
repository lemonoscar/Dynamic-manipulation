"""V0 conveyor scene and PhysX surface-velocity setup."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import (
    bind_physics_material,
    bind_visual_material,
    clone,
    create_prim,
    get_current_stage,
)
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from pxr import Gf, PhysxSchema, UsdPhysics

from .asset_config import make_go2_x5_cfg


LAYOUT_ID = "transverse_y_negative_low_v2"
BELT_CENTER_X_M = 0.70
BELT_CENTER_Y_M = 0.0
BELT_LENGTH_M = 1.20
BELT_WIDTH_M = 0.42
BELT_THICKNESS_M = 0.06
BELT_TOP_Z_M = 0.50
BELT_CENTER_Z_M = BELT_TOP_Z_M - BELT_THICKNESS_M * 0.5
OBJECT_SIZE_M = (0.05, 0.05, 0.08)
OBJECT_CENTER_Z_M = BELT_TOP_Z_M + OBJECT_SIZE_M[2] * 0.5 + 0.002
OBJECT_INTERCEPT_X_M = BELT_CENTER_X_M
OBJECT_INTERCEPT_Y_M = BELT_CENTER_Y_M
OBJECT_SPAWN_Y_M = BELT_CENTER_Y_M + BELT_LENGTH_M * 0.5 - 0.12
OBJECT_EXIT_Y_M = BELT_CENTER_Y_M - BELT_LENGTH_M * 0.5 + 0.03
TRANSPORT_DIRECTION_WORLD = (0.0, -1.0, 0.0)
EXIT_PLANE_POINT_WORLD = (
    BELT_CENTER_X_M,
    OBJECT_EXIT_Y_M,
    BELT_TOP_Z_M,
)
# Just outside the URDF's front-camera housing, looking along the dog's +X.
HEAD_CAMERA_OFFSET_XYZ = (0.355, 0.0, 0.06)
HEAD_CAMERA_OFFSET_WXYZ = (1.0, 0.0, 0.0, 0.0)
# Centered above the gripper housing and pitched 25 degrees toward the fingers.
WRIST_CAMERA_OFFSET_XYZ = (0.02, 0.0, 0.125)
WRIST_CAMERA_OFFSET_WXYZ = (0.97629601, 0.0, 0.21643961, 0.0)
# Fixed third-person observer view; never used as a robot policy observation.
OVERVIEW_CAMERA_OFFSET_XYZ = (-1.20, -0.70, 1.80)
OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.93761823,
    -0.05877532,
    0.28098172,
    0.19612953,
)


def _collision() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.003,
        rest_offset=0.0,
    )


@clone
def _spawn_direct_cuboid(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
):
    """Spawn a Cube whose collision and rigid-body APIs share one prim.

    PhysX surface velocity is defined on the rigid actor.  Keeping the collider
    on that same actor matches Isaac Sim's conveyor implementation and avoids a
    contact-modification bug seen with an Xform rigid root plus child collider.
    """

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: {prim_path}")

    size = min(cfg.size)
    scale = tuple(dimension / size for dimension in cfg.size)
    create_prim(
        prim_path,
        prim_type="Cube",
        translation=translation,
        orientation=orientation,
        scale=scale,
        attributes={"size": size},
        stage=stage,
    )

    if cfg.collision_props is not None:
        schemas.define_collision_properties(
            prim_path,
            cfg.collision_props,
            stage=stage,
        )
    if cfg.visual_material is not None:
        material_path = cfg.visual_material_path
        if not material_path.startswith("/"):
            material_path = f"{prim_path}/{material_path}"
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(prim_path, material_path, stage=stage)
    if cfg.physics_material is not None:
        material_path = cfg.physics_material_path
        if not material_path.startswith("/"):
            material_path = f"{prim_path}/{material_path}"
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(prim_path, material_path, stage=stage)
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(
            prim_path,
            cfg.rigid_props,
            stage=stage,
        )
    return stage.GetPrimAtPath(prim_path)


@configclass
class ConveyorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.30, 0.31, 0.32),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
        ),
    )

    robot = make_go2_x5_cfg()

    conveyor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Conveyor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BELT_CENTER_X_M, BELT_CENTER_Y_M, BELT_CENTER_Z_M)
        ),
        spawn=sim_utils.CuboidCfg(
            func=_spawn_direct_cuboid,
            # Cube dimensions are world aligned: the long axis is robot-left
            # to robot-right (world Y), across the dog's forward +X axis.
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
                diffuse_color=(0.08, 0.11, 0.14),
                roughness=0.72,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBJECT_INTERCEPT_X_M, OBJECT_SPAWN_Y_M, OBJECT_CENTER_Z_M),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            func=_spawn_direct_cuboid,
            size=OBJECT_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=5.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=_collision(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.86, 0.18, 0.12),
                roughness=0.45,
            ),
        ),
    )

    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link[78]",
        update_period=0.0,
        history_length=2,
        force_threshold=0.2,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/TargetObject"],
    )

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/conveyor_head_camera",
        update_period=1.0 / 25.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=2.0,
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
        prim_path="{ENV_REGEX_NS}/Robot/arm_link6/conveyor_wrist_camera",
        update_period=1.0 / 25.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.0,
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
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        update_period=1.0 / 25.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
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
            color=(0.95, 0.96, 1.0),
            intensity=1800.0,
        ),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, -1.0, 2.2),
            rot=(0.9239, 0.0, 0.3827, 0.0),
        ),
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.94, 0.86),
            intensity=700.0,
            angle=2.5,
        ),
    )


def apply_surface_velocity(stage, belt_prim_path: str, speed_mps: float):
    """Apply the conveyor API before the first simulation reset."""

    prim = stage.GetPrimAtPath(belt_prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Conveyor prim does not exist: {belt_prim_path}")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Conveyor prim is not a rigid body: {belt_prim_path}")

    api = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(prim)
    api.CreateSurfaceVelocityEnabledAttr().Set(bool(speed_mps))
    api.CreateSurfaceVelocityLocalSpaceAttr().Set(False)
    api.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(0.0, -float(speed_mps), 0.0))
    return api
