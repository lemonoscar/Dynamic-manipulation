"""Remote-delivery scene extension built on the validated V1 conveyor."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from conveyor_bench.v1.assets import load_receptacles

from .scene_v1 import (
    ConveyorSceneV1Cfg,
    ProceduralWorkcellCfg,
    _spawn_sort_tray,
    _static_box,
    _static_visual_material,
    spawn_conveyor_workcell,
)


SCENE_ID = "mobile_remote_delivery_v2"
REMOTE_RECEPTACLE_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "receptacles"
    / "remote_delivery_v2.json"
)
REMOTE_RECEPTACLE_ASSETS = load_receptacles(REMOTE_RECEPTACLE_MANIFEST_PATH)

REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ = (-2.80, -2.60, 3.20)
REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ = (
    0.89625224,
    -0.10238193,
    0.27791988,
    0.33016722,
)


@configclass
class RemoteDeliveryExtensionCfg(SpawnerCfg):
    """Spawner configuration for the two remote delivery stations."""

    func: Callable = MISSING


@clone
def spawn_remote_delivery_extension(
    prim_path: str,
    cfg: RemoteDeliveryExtensionCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
):
    """Build two supported trays and a visual-only flat route marker."""

    del cfg
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

    pedestal = _static_visual_material(
        f"{prim_path}/Looks/pedestal", (0.18, 0.21, 0.25)
    )
    route = _static_visual_material(
        f"{prim_path}/Looks/route", (0.12, 0.30, 0.25)
    )
    _static_box(
        f"{prim_path}/navigation/flat_corridor",
        (-0.16, 0.0, 0.002),
        (0.44, 1.75, 0.004),
        route,
        collision=False,
    )

    for zone in REMOTE_RECEPTACLE_ASSETS:
        pedestal_height = zone.floor_top_z_m - 0.020
        _static_box(
            f"{prim_path}/pedestals/{zone.zone_id}",
            (
                zone.center_xyz_m[0],
                zone.center_xyz_m[1],
                pedestal_height * 0.5,
            ),
            (0.18, 0.22, pedestal_height),
            pedestal,
        )
        _spawn_sort_tray(
            f"{prim_path}/receptacles/{zone.zone_id}",
            center_xyz=zone.center_xyz_m,
            color=zone.color_rgb,
        )

    return stage.GetPrimAtPath(prim_path)


@configclass
class ConveyorRemoteDeliverySceneCfg(ConveyorSceneV1Cfg):
    """V1 conveyor with a clear corridor and two remote delivery trays."""

    # GroundPlaneCfg composes Isaac's default environment USD. Keep V2 fully
    # offline with an equivalent project-authored static cuboid whose top is z=0.
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.85,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.24, 0.25, 0.27),
                roughness=0.82,
            ),
        ),
    )

    workcell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ConveyorStation",
        spawn=ProceduralWorkcellCfg(
            func=spawn_conveyor_workcell,
            include_local_sort_trays=False,
        ),
    )

    remote_delivery = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RemoteDeliveryStation",
        spawn=RemoteDeliveryExtensionCfg(func=spawn_remote_delivery_extension),
    )

    overview_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCameraRemoteDeliveryV2",
        update_period=1.0 / 25.0,
        height=320,
        width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=4.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ,
            rot=REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ,
            convention="world",
        ),
    )


__all__ = [
    "ConveyorRemoteDeliverySceneCfg",
    "REMOTE_OVERVIEW_CAMERA_OFFSET_WXYZ",
    "REMOTE_OVERVIEW_CAMERA_OFFSET_XYZ",
    "REMOTE_RECEPTACLE_ASSETS",
    "REMOTE_RECEPTACLE_MANIFEST_PATH",
    "SCENE_ID",
    "spawn_remote_delivery_extension",
]
