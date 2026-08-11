#!/usr/bin/env python3
"""Render and inspect the complete ConveyorBench V1 workcell."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-profile",
        choices=("v1", "v3_nurec"),
        default="v1",
    )
    parser.add_argument("--v3-asset-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualization" / "v1_scene_probe",
    )
    parser.add_argument("--belt-speed", type=float, default=0.01)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", enable_cameras=True)
    return parser


def _as_bgr(image):
    import cv2
    import numpy as np

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., ::-1])


def _letterbox(image, *, width: int, height: int):
    import cv2
    import numpy as np

    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def main() -> int:
    args = build_parser().parse_args()
    v3_bundle = None
    v3_runtime_layer = None
    if args.scene_profile == "v3_nurec":
        from conveyor_bench.v3.assets import validate_asset_bundle

        v3_bundle = validate_asset_bundle(
            args.v3_asset_root,
            verify_all_hashes=True,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        v3_runtime_layer = v3_bundle.write_runtime_layer(
            args.output_dir / "liangzhu_conveyorvla_v3.usda",
        )
    app = AppLauncher(args)
    simulation_app = app.app
    try:
        import cv2
        import numpy as np
        import omni.usd
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        from conveyor_bench.isaac.asset_config import (
            ARM_JOINT_NAMES,
            GRIPPER_JOINT_NAMES,
            LEG_JOINT_NAMES,
        )
        from conveyor_bench.isaac.scene import apply_surface_velocity
        from conveyor_bench.isaac.scene_v1 import (
            BELT_TOP_Z_M,
            OBJECT_ASSETS,
            OBJECT_ENTITY_NAMES,
            OBJECT_LANE_X_M,
            TRANSPORT_DIRECTION_WORLD,
            ConveyorSceneV1Cfg,
            apply_pct_gripper_collision_patch,
        )

        if not args.enable_cameras:
            raise ValueError("probe_v1_scene requires --enable_cameras")
        if args.belt_speed < 0.0:
            raise ValueError("--belt-speed cannot be negative")
        if args.settle_seconds <= 0.0:
            raise ValueError("--settle-seconds must be positive")

        physics_hz = 400
        camera_hz = 25
        camera_render_stride = physics_hz // camera_hz
        simulation = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / physics_hz,
                render_interval=1,
                device=args.device,
                use_fabric=True,
                physx=sim_utils.PhysxCfg(
                    enable_enhanced_determinism=True,
                    bounce_threshold_velocity=0.2,
                    friction_correlation_distance=0.00625,
                ),
            )
        )
        if args.scene_profile == "v3_nurec":
            from conveyor_bench.isaac.scene_v3 import (
                V3_CONVEYOR_PRIM_PATH,
                describe_v3_conveyor_world_pose,
                make_conveyor_scene_v3_cfg,
                place_workcell_in_liangzhu_open_room,
                validate_liangzhu_stage,
            )

            assert v3_runtime_layer is not None
            scene_cfg = make_conveyor_scene_v3_cfg(v3_runtime_layer)
        else:
            scene_cfg = ConveyorSceneV1Cfg(
                num_envs=1,
                env_spacing=3.0,
                replicate_physics=True,
                clone_in_fabric=False,
                lazy_sensor_update=True,
            )
        scene = InteractiveScene(scene_cfg)
        workcell_placement = (
            place_workcell_in_liangzhu_open_room(
                scene,
                omni.usd.get_context().get_stage(),
            )
            if args.scene_profile == "v3_nurec"
            else None
        )
        scene_stage_contract = (
            validate_liangzhu_stage(omni.usd.get_context().get_stage())
            if args.scene_profile == "v3_nurec"
            else {}
        )
        apply_pct_gripper_collision_patch(
            omni.usd.get_context().get_stage(),
            "/World/envs/env_0/Robot",
        )
        surface_api = apply_surface_velocity(
            omni.usd.get_context().get_stage(),
            (
                V3_CONVEYOR_PRIM_PATH
                if args.scene_profile == "v3_nurec"
                else "/World/envs/env_0/TransportSurface"
            ),
            args.belt_speed,
        )
        simulation.reset()
        scene.reset()
        conveyor_placement = (
            describe_v3_conveyor_world_pose(scene)
            if args.scene_profile == "v3_nurec"
            else None
        )

        robot = scene["robot"]
        arm_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        leg_ids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
        gripper_ids, _ = robot.find_joints(
            list(GRIPPER_JOINT_NAMES), preserve_order=True
        )
        arm_pose = torch.tensor(
            [[0.0, 1.0, 1.5, 0.0, 0.0, 0.0]],
            device=simulation.device,
            dtype=torch.float32,
        )
        joint_positions = robot.data.default_joint_pos.clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, arm_ids] = arm_pose
        joint_positions[:, gripper_ids] = 0.044
        robot.write_joint_state_to_sim(joint_positions, joint_velocities)

        # Put four geometrically different parts on distinct portions of the
        # belt; park the rest locally but outside the rendered workcell.
        active_indices = (0, 2, 5, 6)
        active_y = (0.43, 0.16, -0.12, -0.38)
        initial_positions: dict[str, tuple[float, float, float]] = {}
        for index, (entity_name, asset) in enumerate(
            zip(OBJECT_ENTITY_NAMES, OBJECT_ASSETS, strict=True)
        ):
            rigid_object = scene[entity_name]
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            if index in active_indices:
                slot = active_indices.index(index)
                root_state[:, 0] = OBJECT_LANE_X_M + scene.env_origins[:, 0]
                root_state[:, 1] = active_y[slot] + scene.env_origins[:, 1]
                root_state[:, 2] = (
                    BELT_TOP_Z_M
                    + asset.half_extents_xyz[2]
                    + 0.003
                    + scene.env_origins[:, 2]
                )
                initial_positions[asset.object_id] = (
                    OBJECT_LANE_X_M + float(scene.env_origins[0, 0].item()),
                    active_y[slot] + float(scene.env_origins[0, 1].item()),
                    float(root_state[0, 2].item()),
                )
            else:
                root_state[:, 0] = 3.0 + scene.env_origins[:, 0]
                root_state[:, 1] = (
                    -0.7 + index * 0.2 + scene.env_origins[:, 1]
                )
                root_state[:, 2] = 0.15 + scene.env_origins[:, 2]
            root_state[:, 3:7] = torch.tensor(
                [asset.stable_poses_wxyz[0]],
                device=simulation.device,
                dtype=torch.float32,
            )
            root_state[:, 7:] = 0.0
            rigid_object.write_root_pose_to_sim(root_state[:, :7])
            rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

        scene.reset()
        total_steps = max(1, round(args.settle_seconds * physics_hz))
        for step in range(total_steps):
            robot.set_joint_position_target(
                robot.data.default_joint_pos[:, leg_ids],
                joint_ids=leg_ids,
            )
            robot.set_joint_position_target(arm_pose, joint_ids=arm_ids)
            robot.set_joint_position_target(
                torch.full(
                    (1, len(gripper_ids)),
                    0.044,
                    device=simulation.device,
                ),
                joint_ids=gripper_ids,
            )
            scene.write_data_to_sim()
            simulation.step(
                render=(step + 1) % camera_render_stride == 0
                or step + 1 == total_steps
            )
            scene.update(1.0 / physics_hz)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        images = {
            "head_rgb": _as_bgr(scene["head_camera"].data.output["rgb"]),
            "wrist_rgb": _as_bgr(scene["wrist_camera"].data.output["rgb"]),
            "overview_rgb": _as_bgr(scene["overview_camera"].data.output["rgb"]),
        }
        object_states = {
            OBJECT_ASSETS[index].object_id: [
                float(value)
                for value in scene[OBJECT_ENTITY_NAMES[index]]
                .data.root_pos_w[0]
                .detach()
                .cpu()
                .tolist()
            ]
            for index in active_indices
        }
        transport_displacement_m = {
            object_id: sum(
                (final[index] - initial[index])
                * TRANSPORT_DIRECTION_WORLD[index]
                for index in range(3)
            )
            for object_id, initial in initial_positions.items()
            for final in (object_states[object_id],)
        }
        minimum_transport_m = max(
            0.005,
            args.belt_speed * args.settle_seconds * 0.20,
        )
        passed = all(
            displacement >= minimum_transport_m
            for displacement in transport_displacement_m.values()
        )
        for name, image in images.items():
            if not cv2.imwrite(str(args.output_dir / f"{name}.png"), image):
                raise RuntimeError(f"failed to write {name}.png")

        tile_width = 480
        image_height = 300
        tiles = []
        for name in ("head_rgb", "wrist_rgb", "overview_rgb"):
            image = images[name]
            canvas = np.full((340, tile_width, 3), 28, dtype=np.uint8)
            canvas[:image_height] = _letterbox(
                image,
                width=tile_width,
                height=image_height,
            )
            cv2.putText(
                canvas,
                name,
                (14, 326),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            tiles.append(canvas)
        mosaic = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(args.output_dir / "three_camera_scene.png"), mosaic)

        report = {
            "layout_id": (
                "transverse_dynamic_sort_liangzhu_nurec_v3"
                if args.scene_profile == "v3_nurec"
                else "transverse_dynamic_sort_station_v1"
            ),
            "scene_profile": args.scene_profile,
            "scene_stage_contract": scene_stage_contract,
            "workcell_placement": workcell_placement,
            "conveyor_placement": conveyor_placement,
            "v3_asset_bundle": (
                v3_bundle.report.to_dict() if v3_bundle is not None else None
            ),
            "belt_speed_mps": args.belt_speed,
            "belt_surface_velocity": list(
                surface_api.GetSurfaceVelocityAttr().Get()
            ),
            "active_assets": [
                OBJECT_ASSETS[index].object_id for index in active_indices
            ],
            "object_states": object_states,
            "transport_displacement_m": transport_displacement_m,
            "minimum_transport_displacement_m": minimum_transport_m,
            "passed": passed,
            "camera_files": {
                name: str((args.output_dir / f"{name}.png").resolve())
                for name in images
            },
            "camera_pose_world": {
                name: {
                    "position_xyz": [
                        float(value)
                        for value in scene[camera_name]
                        .data.pos_w[0]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "orientation_wxyz": [
                        float(value)
                        for value in scene[camera_name]
                        .data.quat_w_world[0]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                }
                for name, camera_name in (
                    ("head_rgb", "head_camera"),
                    ("wrist_rgb", "wrist_camera"),
                    ("overview_rgb", "overview_camera"),
                )
            },
            "mosaic": str(
                (args.output_dir / "three_camera_scene.png").resolve()
            ),
        }
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        scene = None
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        return 0 if passed else 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
