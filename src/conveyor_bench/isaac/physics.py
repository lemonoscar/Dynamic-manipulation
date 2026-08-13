"""Shared PhysX helpers for the conveyor workcell."""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import (
    bind_physics_material,
    bind_visual_material,
    clone,
    create_prim,
    get_current_stage,
)
from pxr import Gf, PhysxSchema, UsdPhysics


def collision_properties() -> sim_utils.CollisionPropertiesCfg:
    """Return the contact settings shared by task rigid bodies."""

    return sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.003,
        rest_offset=0.0,
    )


@clone
def spawn_direct_cuboid(
    prim_path: str,
    cfg: sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **_: object,
) -> Any:
    """Spawn one cube with rigid-body and collision APIs on the same prim."""

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: {prim_path}")

    size = min(cfg.size)
    create_prim(
        prim_path,
        prim_type="Cube",
        translation=translation,
        orientation=orientation,
        scale=tuple(dimension / size for dimension in cfg.size),
        attributes={"size": size},
        stage=stage,
    )
    if cfg.collision_props is not None:
        schemas.define_collision_properties(
            prim_path, cfg.collision_props, stage=stage
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
            prim_path, cfg.rigid_props, stage=stage
        )
    return stage.GetPrimAtPath(prim_path)


def apply_surface_velocity(stage: Any, belt_prim_path: str, speed_mps: float):
    """Apply world-space belt velocity before the first simulation reset."""

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


__all__ = ["apply_surface_velocity", "collision_properties", "spawn_direct_cuboid"]
