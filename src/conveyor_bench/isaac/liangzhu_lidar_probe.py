"""Isaac adapter for the Liangzhu coke-grasp LiDAR diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import RayCasterCfg, patterns
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from conveyor_bench.perception import (
    LidarScan,
    UnitreeL2ProvisionalConfig,
    quaternion_wxyz_to_matrix,
    transform_points,
)
from conveyor_bench.sidecar.objects import COLA_OBJECT

from .asset_config import make_go2_x5_cfg
from .scene import (
    LIANGZHU_STAGE_PRIM_PATH,
    TASK_AREA_GROUND_XYZ_M,
    make_conveyor_scene_cfg,
)


DIAGNOSTIC_BACKEND_ID = "isaaclab_warp_background_analytic_foreground_v4"
BOX1_ROOT_WORLD_POSITION_XYZ_M = (
    0.3291217764175582,
    5.589614570976296,
    -0.1288772646929709,
)
BOX1_SUPPORT_CENTER_WORLD_XYZ_M = (
    -0.5526548518474443,
    6.59816551497957,
    0.14424825618208506,
)
BOX1_SUPPORT_SIZE_XYZ_M = (
    0.4075790024217219,
    0.3794805321318073,
    0.27312554689830346,
)
BOX2_ROOT_WORLD_POSITION_XYZ_M = (
    -0.5395068138980879,
    5.756616506575413,
    -0.13822619360037947,
)
BOX2_SUPPORT_CENTER_WORLD_XYZ_M = (
    -0.5410275620365157,
    3.5065302396019398,
    0.22902583028887316,
)
BOX2_SUPPORT_SIZE_XYZ_M = (
    0.4005865097288545,
    0.26878706140284336,
    0.36725204272785263,
)
COLA_WORLD_POSITION_XYZ_M = (
    -0.5526548518474443,
    6.59816551497957,
    0.19789323636730362,
)
ROBOT_ROOT_HEIGHT_ABOVE_TASK_GROUND_M = 0.431011105
BOX_ASSET_SCALE_XYZ = (0.005, 0.005, 0.005)
BOX_ASSET_ORIENTATION_WXYZ = (0.0, 1.0, 0.0, 0.0)
BOX_INTERNAL_TRANSLATION_XYZ = {
    "/World/box1/node_0": (
        -176.23772616569585,
        -200.92293270482264,
        -0.00002699577611409154,
    ),
    "/World/box2/node_0": (
        0.0,
        451.0569588371643,
        0.00006060349709358093,
    ),
}
RAYCAST_MESH_ID_AUDIT = {
    1: "liangzhu_background",
    2: "box1_aligned_proxy",
    3: "box2_aligned_proxy",
    4: "cola_aligned_proxy",
}


def make_liangzhu_lidar_probe_scene_cfg(
    runtime_layer: Path,
    *,
    cola_usd_path: Path,
    box1_usd_path: Path,
    box2_usd_path: Path,
    lidar_config: UnitreeL2ProvisionalConfig,
    head_camera_depth: bool = False,
) -> Any:
    """Build the existing Liangzhu scene without conveyor-only entities."""

    cfg = make_conveyor_scene_cfg(
        runtime_layer,
        object_assets=(COLA_OBJECT,),
        object_usd_paths={COLA_OBJECT.object_id: cola_usd_path},
    )
    cfg.conveyor = None
    cfg.workcell = None
    cfg.wrist_camera = None
    cfg.left_finger_contact = None
    cfg.right_finger_contact = None
    cfg.robot = make_go2_x5_cfg(fix_base=False)
    if head_camera_depth:
        cfg.head_camera.data_types = ["rgb", "distance_to_image_plane"]
    cfg.robot.init_state.pos = (
        0.0,
        0.0,
        ROBOT_ROOT_HEIGHT_ABOVE_TASK_GROUND_M,
    )
    cfg.object_00.init_state.pos = tuple(
        world - ground
        for world, ground in zip(
            COLA_WORLD_POSITION_XYZ_M,
            TASK_AREA_GROUND_XYZ_M,
            strict=True,
        )
    )
    cfg.object_00.init_state.rot = COLA_OBJECT.stable_poses_wxyz[0]
    cfg.box1 = AssetBaseCfg(
        prim_path="/World/box1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BOX1_ROOT_WORLD_POSITION_XYZ_M,
            rot=BOX_ASSET_ORIENTATION_WXYZ,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(box1_usd_path),
            scale=BOX_ASSET_SCALE_XYZ,
        ),
    )
    cfg.box2 = AssetBaseCfg(
        prim_path="/World/box2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BOX2_ROOT_WORLD_POSITION_XYZ_M,
            rot=BOX_ASSET_ORIENTATION_WXYZ,
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(box2_usd_path),
            scale=BOX_ASSET_SCALE_XYZ,
        ),
    )
    cfg.lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=lidar_config.scan_period_s,
        mesh_prim_paths=[f"{LIANGZHU_STAGE_PRIM_PATH}/Collision"],
        offset=RayCasterCfg.OffsetCfg(
            pos=lidar_config.mount_position_xyz_m,
            rot=lidar_config.mount_orientation_wxyz,
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=lidar_config.channels,
            vertical_fov_range=(
                -lidar_config.vertical_fov_deg / 2.0,
                lidar_config.vertical_fov_deg / 2.0,
            ),
            horizontal_fov_range=(
                -lidar_config.horizontal_fov_deg / 2.0,
                lidar_config.horizontal_fov_deg / 2.0,
            ),
            horizontal_res=lidar_config.horizontal_resolution_deg,
        ),
        max_distance=lidar_config.raw_max_range_m,
        debug_vis=False,
    )
    return cfg


def disable_box_physics_collisions(stage: Any) -> dict[str, Any]:
    """Keep both high-detail boxes visible without simulating triangle meshes."""

    disabled: list[str] = []
    roots = ("/World/box1", "/World/box2")
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(roots) or not prim.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        attribute = collision.GetCollisionEnabledAttr()
        if not attribute.IsValid():
            attribute = collision.CreateCollisionEnabledAttr()
        attribute.Set(False)
        disabled.append(prim_path)
    return {
        "policy": "box_visuals_enabled_physics_collisions_disabled",
        "disabled_collision_prims": disabled,
        "lidar_collision_source": "aligned_low_polygon_proxies",
    }


def apply_and_validate_box_visual_alignment(stage: Any) -> dict[str, Any]:
    """Restore original child overrides and verify task-space box bounds."""

    for prim_path, translation in BOX_INTERNAL_TRANSLATION_XYZ.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"box visual prim is missing: {prim_path}")
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
        translate_op.Set(Gf.Vec3d(*translation))

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    reports: list[dict[str, Any]] = []
    specifications = (
        ("box1", BOX1_SUPPORT_CENTER_WORLD_XYZ_M, BOX1_SUPPORT_SIZE_XYZ_M),
        ("box2", BOX2_SUPPORT_CENTER_WORLD_XYZ_M, BOX2_SUPPORT_SIZE_XYZ_M),
    )
    for name, support_center, expected_size in specifications:
        prim = stage.GetPrimAtPath(f"/World/{name}")
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
        maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
        actual_center = (minimum + maximum) / 2.0
        actual_size = maximum - minimum
        expected_center = np.asarray(
            (
                support_center[0],
                support_center[1],
                support_center[2] - expected_size[2] / 2.0,
            ),
            dtype=np.float64,
        )
        center_error = float(np.linalg.norm(actual_center - expected_center))
        size_error = float(np.max(np.abs(actual_size - expected_size)))
        if center_error > 1.0e-4 or size_error > 1.0e-4:
            raise RuntimeError(
                f"{name} visual alignment mismatch: center_error={center_error}, "
                f"size_error={size_error}, center={actual_center}, size={actual_size}"
            )
        reports.append(
            {
                "name": name,
                "visual_aabb_center_world_xyz_m": actual_center.tolist(),
                "visual_aabb_size_xyz_m": actual_size.tolist(),
                "support_surface_z_m": float(maximum[2]),
                "center_error_m": center_error,
                "max_size_error_m": size_error,
            }
        )
    return {"all_aligned": True, "boxes": reports}


def _ray_aabb_distance(
    origins: np.ndarray,
    directions: np.ndarray,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> np.ndarray:
    minimum = np.asarray(center, dtype=np.float64) - np.asarray(size) / 2.0
    maximum = np.asarray(center, dtype=np.float64) + np.asarray(size) / 2.0
    parallel = np.abs(directions) < 1.0e-10
    outside_parallel = parallel & ((origins < minimum) | (origins > maximum))
    safe_directions = np.where(parallel, 1.0, directions)
    first = (minimum - origins) / safe_directions
    second = (maximum - origins) / safe_directions
    near = np.max(np.where(parallel, -np.inf, np.minimum(first, second)), axis=1)
    far = np.min(np.where(parallel, np.inf, np.maximum(first, second)), axis=1)
    distance = np.where(near >= 0.0, near, far)
    valid = (~outside_parallel.any(axis=1)) & (far >= np.maximum(near, 0.0))
    return np.where(valid, distance, np.inf)


def _ray_upright_cylinder_distance(
    origins: np.ndarray,
    directions: np.ndarray,
    center: tuple[float, float, float],
    radius: float,
    height: float,
) -> np.ndarray:
    center_array = np.asarray(center, dtype=np.float64)
    relative = origins - center_array
    lower_z = -height / 2.0
    upper_z = height / 2.0
    candidates = np.full((len(origins), 4), np.inf, dtype=np.float64)

    a = np.sum(directions[:, :2] ** 2, axis=1)
    b = 2.0 * np.sum(relative[:, :2] * directions[:, :2], axis=1)
    c = np.sum(relative[:, :2] ** 2, axis=1) - radius**2
    discriminant = b**2 - 4.0 * a * c
    has_side = (a > 1.0e-12) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    for column, sign in enumerate((-1.0, 1.0)):
        distance = (-b + sign * root) / np.where(a > 1.0e-12, 2.0 * a, 1.0)
        hit_z = relative[:, 2] + distance * directions[:, 2]
        valid = has_side & (distance >= 0.0) & (hit_z >= lower_z) & (hit_z <= upper_z)
        candidates[:, column] = np.where(valid, distance, np.inf)

    has_cap = np.abs(directions[:, 2]) > 1.0e-12
    for column, cap_z in enumerate((lower_z, upper_z), start=2):
        distance = (cap_z - relative[:, 2]) / np.where(
            has_cap, directions[:, 2], 1.0
        )
        hit_xy = relative[:, :2] + distance[:, None] * directions[:, :2]
        valid = has_cap & (distance >= 0.0) & (
            np.sum(hit_xy**2, axis=1) <= radius**2
        )
        candidates[:, column] = np.where(valid, distance, np.inf)
    return np.min(candidates, axis=1)


def lidar_scan_from_ray_caster(
    sensor: Any,
    *,
    scan_index: int,
    sim_time_s: float,
    config: UnitreeL2ProvisionalConfig,
) -> LidarScan:
    """Convert one ideal Warp ray-cast buffer into the raw scan contract."""

    data = sensor.data
    hits_world = data.ray_hits_w[0].detach().cpu().numpy().astype(np.float64)
    ray_origins_world = sensor._ray_starts_w[0].detach().cpu().numpy().astype(np.float64)
    ray_directions_world = sensor._ray_directions_w[0].detach().cpu().numpy().astype(np.float64)
    sensor_position = data.pos_w[0].detach().cpu().numpy().astype(np.float64)
    sensor_orientation = data.quat_w[0].detach().cpu().numpy().astype(np.float64)
    object_ids = np.ones(len(hits_world), dtype=np.uint32)
    best_distance = np.linalg.norm(hits_world - ray_origins_world, axis=1)
    best_distance[~np.isfinite(hits_world).all(axis=1)] = np.inf

    box1_center = (
        BOX1_SUPPORT_CENTER_WORLD_XYZ_M[0],
        BOX1_SUPPORT_CENTER_WORLD_XYZ_M[1],
        BOX1_SUPPORT_CENTER_WORLD_XYZ_M[2] - BOX1_SUPPORT_SIZE_XYZ_M[2] / 2.0,
    )
    box2_center = (
        BOX2_SUPPORT_CENTER_WORLD_XYZ_M[0],
        BOX2_SUPPORT_CENTER_WORLD_XYZ_M[1],
        BOX2_SUPPORT_CENTER_WORLD_XYZ_M[2] - BOX2_SUPPORT_SIZE_XYZ_M[2] / 2.0,
    )
    foreground_distances = (
        (2, _ray_aabb_distance(ray_origins_world, ray_directions_world, box1_center, BOX1_SUPPORT_SIZE_XYZ_M)),
        (3, _ray_aabb_distance(ray_origins_world, ray_directions_world, box2_center, BOX2_SUPPORT_SIZE_XYZ_M)),
        (
            4,
            _ray_upright_cylinder_distance(
                ray_origins_world,
                ray_directions_world,
                COLA_WORLD_POSITION_XYZ_M,
                float(COLA_OBJECT.geometry["radius_m"]),
                float(COLA_OBJECT.geometry["height_m"]),
            ),
        ),
    )
    for object_id, distances in foreground_distances:
        closer = distances < best_distance
        hits_world[closer] = (
            ray_origins_world[closer]
            + distances[closer, None] * ray_directions_world[closer]
        )
        best_distance[closer] = distances[closer]
        object_ids[closer] = object_id

    sensor_to_world = np.eye(4, dtype=np.float64)
    sensor_to_world[:3, :3] = quaternion_wxyz_to_matrix(sensor_orientation)
    sensor_to_world[:3, 3] = sensor_position

    finite = np.isfinite(hits_world).all(axis=1)
    ranges = np.linalg.norm(hits_world - sensor_position, axis=1)
    valid = finite & (ranges >= config.raw_min_range_m) & (ranges <= config.raw_max_range_m)
    points_world = hits_world[valid]
    world_to_sensor = np.linalg.inv(sensor_to_world)
    points_sensor = transform_points(points_world, world_to_sensor)

    emitted_indices = np.arange(config.emitted_points_per_scan, dtype=np.int64)
    columns = emitted_indices % config.columns_per_revolution
    rings = emitted_indices // config.columns_per_revolution
    relative_times = columns / config.columns_per_revolution * config.scan_period_s
    return LidarScan(
        scan_index=scan_index,
        sim_time_s=sim_time_s,
        xyz_sensor_m=points_sensor.astype(np.float32),
        xyz_world_m=points_world.astype(np.float32),
        intensity=np.ones(int(valid.sum()), dtype=np.float32),
        relative_time_s=relative_times[valid].astype(np.float32),
        ring=rings[valid].astype(np.uint16),
        sensor_to_world=sensor_to_world,
        emitted_point_count=config.emitted_points_per_scan,
        backend=DIAGNOSTIC_BACKEND_ID,
        object_id_audit=object_ids[valid],
        intensity_synthetic=True,
    )


__all__ = [
    "BOX1_ROOT_WORLD_POSITION_XYZ_M",
    "BOX_ASSET_ORIENTATION_WXYZ",
    "BOX_ASSET_SCALE_XYZ",
    "BOX1_SUPPORT_CENTER_WORLD_XYZ_M",
    "BOX1_SUPPORT_SIZE_XYZ_M",
    "BOX2_ROOT_WORLD_POSITION_XYZ_M",
    "BOX2_SUPPORT_CENTER_WORLD_XYZ_M",
    "BOX2_SUPPORT_SIZE_XYZ_M",
    "COLA_WORLD_POSITION_XYZ_M",
    "DIAGNOSTIC_BACKEND_ID",
    "RAYCAST_MESH_ID_AUDIT",
    "apply_and_validate_box_visual_alignment",
    "disable_box_physics_collisions",
    "lidar_scan_from_ray_caster",
    "make_liangzhu_lidar_probe_scene_cfg",
]
