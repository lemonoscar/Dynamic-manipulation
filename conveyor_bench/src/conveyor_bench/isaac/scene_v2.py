"""Offline near-sort scene for the ConveyorBench V2 suite."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from .scene_v1 import ConveyorSceneV1Cfg


SCENE_ID = "transverse_near_sort_v2"


@configclass
class ConveyorNearSortV2SceneCfg(ConveyorSceneV1Cfg):
    """Frozen V1 workcell with a project-local procedural ground."""

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


__all__ = ["ConveyorNearSortV2SceneCfg", "SCENE_ID"]
