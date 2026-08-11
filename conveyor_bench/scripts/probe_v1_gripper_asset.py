#!/usr/bin/env python3
"""Print the live PCT-URDF collision hierarchy for the V1 gripper."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_URDF = PROJECT_ROOT / "assets" / "robots" / "go2_x5" / "go2_x5.urdf"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    app = AppLauncher(args)
    simulation_app = app.app
    simulation = None
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        from conveyor_bench.isaac.asset_config import make_go2_x5_cfg
        from conveyor_bench.isaac.scene_v1 import (
            apply_pct_gripper_collision_patch,
        )

        print(
            json.dumps({"probe": "spawning_robot_urdf", "path": str(ROBOT_URDF)}),
            flush=True,
        )
        simulation = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=0.0025, device=args.device)
        )
        robot_cfg = make_go2_x5_cfg(fix_base=True)
        robot_cfg.prim_path = "/World/Robot"
        robot = Articulation(robot_cfg)
        stage = get_current_stage()
        patch_report = apply_pct_gripper_collision_patch(stage, "/World/Robot")
        simulation.reset()
        robot.update(simulation.get_physics_dt())
        print(
            json.dumps(
                {
                    "probe": "robot_urdf_spawned",
                    "robot_prim": "/World/Robot",
                    "collision_patch": patch_report,
                }
            ),
            flush=True,
        )
        time_code = Usd.TimeCode.Default()
        xform_cache = UsdGeom.XformCache(time_code)
        bbox_cache = UsdGeom.BBoxCache(
            time_code,
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
            ],
            useExtentsHint=True,
        )
        link_by_name = {
            prim.GetName(): prim
            for prim in stage.TraverseAll()
            if prim.GetName() in {"arm_link7", "arm_link8"}
        }
        if set(link_by_name) != {"arm_link7", "arm_link8"}:
            raise RuntimeError(
                f"could not resolve gripper links in {ROBOT_USD}: "
                f"{sorted(link_by_name)}"
            )

        def vector(value) -> list[float]:
            return [float(component) for component in value]

        def matrix(value) -> list[list[float]]:
            return [
                [float(value[row][column]) for column in range(4)]
                for row in range(4)
            ]

        def json_value(value):
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            try:
                return [json_value(item) for item in value]
            except TypeError:
                return str(value)

        def range_record(value) -> dict[str, list[float]] | None:
            if value is None or value.IsEmpty():
                return None
            minimum = vector(value.GetMin())
            maximum = vector(value.GetMax())
            return {
                "min": minimum,
                "max": maximum,
                "center": [
                    (lower + upper) * 0.5
                    for lower, upper in zip(minimum, maximum, strict=True)
                ],
                "size": [
                    upper - lower
                    for lower, upper in zip(minimum, maximum, strict=True)
                ],
            }

        def extent_range(prim):
            if not prim.IsA(UsdGeom.Boundable):
                return None
            extent = UsdGeom.Boundable(prim).GetExtentAttr().Get(time_code)
            if extent is None or len(extent) != 2:
                return None
            return Gf.Range3d(
                Gf.Vec3d(*extent[0]),
                Gf.Vec3d(*extent[1]),
            )

        def transform_range_to_link(value, prim, link):
            if value is None or value.IsEmpty():
                return None
            prim_to_world = xform_cache.GetLocalToWorldTransform(prim)
            world_to_link = xform_cache.GetLocalToWorldTransform(link).GetInverse()
            minimum = value.GetMin()
            maximum = value.GetMax()
            output = Gf.Range3d()
            for x in (minimum[0], maximum[0]):
                for y in (minimum[1], maximum[1]):
                    for z in (minimum[2], maximum[2]):
                        point_world = prim_to_world.Transform(
                            Gf.Vec3d(x, y, z)
                        )
                        output.UnionWith(world_to_link.Transform(point_world))
            return output

        def local_matrix(prim):
            xformable = UsdGeom.Xformable(prim)
            if not xformable:
                return None
            value = xformable.GetLocalTransformation(time_code)
            if isinstance(value, tuple):
                value = value[0]
            return matrix(value)

        def link_local_frame(prim, link):
            prim_to_world = xform_cache.GetLocalToWorldTransform(prim)
            world_to_link = xform_cache.GetLocalToWorldTransform(link).GetInverse()

            def point_in_link(x: float, y: float, z: float):
                return world_to_link.Transform(
                    prim_to_world.Transform(Gf.Vec3d(x, y, z))
                )

            origin = point_in_link(0.0, 0.0, 0.0)
            return {
                "origin": vector(origin),
                "basis_x": vector(point_in_link(1.0, 0.0, 0.0) - origin),
                "basis_y": vector(point_in_link(0.0, 1.0, 0.0) - origin),
                "basis_z": vector(point_in_link(0.0, 0.0, 1.0) - origin),
            }

        def collision_owner(prim, link):
            current = prim
            while current and current.GetPath().HasPrefix(link.GetPath()):
                if current.HasAPI(UsdPhysics.CollisionAPI):
                    return current
                current = current.GetParent()
            return None

        def approximation_attributes(prim) -> dict[str, object]:
            result = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                lowered = name.lower()
                if "approximation" in lowered or "collision" in lowered:
                    result[name] = json_value(attribute.Get(time_code))
            approximation = (
                UsdPhysics.MeshCollisionAPI(prim)
                .GetApproximationAttr()
                .Get(time_code)
            )
            if approximation is not None:
                result["resolved_mesh_approximation"] = str(approximation)
            return result

        def prim_record(prim, link, *, prototype_path=None):
            owner = collision_owner(prim, link)
            authored_extent = extent_range(prim)
            local_bound = None
            if prim.IsA(UsdGeom.Boundable):
                local_bound = (
                    bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()
                )
            bounds_source = authored_extent or local_bound
            prim_in_prototype = (
                prim.GetPrimInPrototype()
                if prim.IsInstanceProxy()
                else None
            )
            return {
                "path": str(prim.GetPath()),
                "prototype_path": (
                    str(prototype_path) if prototype_path is not None else None
                ),
                "prim_in_prototype": (
                    str(prim_in_prototype.GetPath())
                    if prim_in_prototype
                    else None
                ),
                "type": prim.GetTypeName(),
                "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                "collision_owner": (
                    str(owner.GetPath()) if owner is not None else None
                ),
                "instance": prim.IsInstance(),
                "instance_proxy": prim.IsInstanceProxy(),
                "prototype": (
                    str(prim.GetPrototype().GetPath())
                    if prim.IsInstance()
                    else None
                ),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "local_xform_matrix": local_matrix(prim),
                "link_local_frame": link_local_frame(prim, link),
                "authored_extent": (
                    range_record(authored_extent)
                    if authored_extent is not None
                    else None
                ),
                "computed_local_bound": (
                    range_record(local_bound)
                    if local_bound is not None
                    else None
                ),
                "link_local_bounds": range_record(
                    transform_range_to_link(bounds_source, prim, link)
                ),
                "approximation_attributes": approximation_attributes(prim),
                "prim_stack": [
                    f"{spec.layer.identifier}:{spec.path}"
                    for spec in prim.GetPrimStack()
                ],
            }

        def traverse(root):
            return Usd.PrimRange(root, Usd.TraverseInstanceProxies())

        for link_name in ("arm_link7", "arm_link8"):
            link = link_by_name[link_name]
            prefix = str(link.GetPath())
            live_prims = list(traverse(link))
            live_records = [
                prim_record(prim, link)
                for prim in live_prims
                if (
                    prim.GetTypeName() in {"Mesh", "Cube"}
                    or prim.HasAPI(UsdPhysics.CollisionAPI)
                    or prim.IsInstance()
                )
            ]
            prototype_records = []
            for instance in (prim for prim in live_prims if prim.IsInstance()):
                prototype = instance.GetPrototype()
                if not prototype:
                    continue
                for prototype_prim in traverse(prototype):
                    if prototype_prim.GetTypeName() not in {"Mesh", "Cube"}:
                        continue
                    relative = prototype_prim.GetPath().MakeRelativePath(
                        prototype.GetPath()
                    )
                    proxy = stage.GetPrimAtPath(
                        instance.GetPath().AppendPath(relative)
                    )
                    if not proxy or collision_owner(proxy, link) is None:
                        continue
                    prototype_records.append(
                        prim_record(
                            proxy,
                            link,
                            prototype_path=prototype_prim.GetPath(),
                        )
                    )
            print(
                json.dumps(
                    {
                        "link": prefix,
                        "live_subtree": live_records,
                        "instance_prototype_geometry": prototype_records,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        print(
            json.dumps(
                {
                    "all_collision_prims": [
                        str(prim.GetPath())
                        for prim in traverse(stage.GetPseudoRoot())
                        if prim.HasAPI(UsdPhysics.CollisionAPI)
                    ]
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
