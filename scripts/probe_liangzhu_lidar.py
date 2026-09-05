#!/usr/bin/env python3
"""Record a headless three-panel LiDAR probe in Liangzhu coke-grasp."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=34.0)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--physics-hz", type=int, default=100)
    parser.add_argument(
        "--sync-capture-time",
        type=float,
        help=(
            "save one synchronized head RGB/depth/camera-pose/LiDAR bundle at "
            "the first rendered scan at or after this simulation time"
        ),
    )
    parser.add_argument(
        "--antialiasing-mode",
        choices=("Off", "FXAA", "DLSS", "TAA", "DLAA"),
        default="FXAA",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cuda:0", headless=True, enable_cameras=True)
    return parser


def _default_output_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "artifacts" / "runs" / f"liangzhu-lidar-probe-{stamp}"


def _scripted_root_offset(sim_time_s: float) -> tuple[float, float, str]:
    turn_end = 18.0 + math.pi / (2.0 * 0.20)
    return_end = turn_end + 3.0 + 5.0
    if sim_time_s < 10.0:
        return 0.0, 0.0, "static_10s"
    if sim_time_s < 15.0:
        return 0.10 * (sim_time_s - 10.0), 0.0, "forward_0.10mps"
    if sim_time_s < 18.0:
        return 0.50, 0.0, "stop_after_forward"
    if sim_time_s < turn_end:
        return 0.50, 0.20 * (sim_time_s - 18.0), "turn_0.20radps"
    if sim_time_s < turn_end + 3.0:
        return 0.50, math.pi / 2.0, "stop_after_turn"
    if sim_time_s < return_end:
        alpha = (sim_time_s - turn_end - 3.0) / 5.0
        return 0.50 * (1.0 - alpha), math.pi / 2.0 * (1.0 - alpha), "scripted_return"
    return 0.0, 0.0, "final_static"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _transcode_h264(source: Path, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.asset_root is None:
        parser.error("--asset-root is required")
    if args.duration_seconds <= 0.0:
        parser.error("--duration-seconds must be positive")
    if args.video_fps <= 0.0:
        parser.error("--video-fps must be positive")
    if args.physics_hz < 20:
        parser.error("--physics-hz must be at least 20")
    if not args.enable_cameras:
        parser.error("this probe requires --enable_cameras")
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    from conveyor_bench.sidecar.assets import validate_asset_bundle

    asset_bundle = validate_asset_bundle(
        args.asset_root.expanduser().resolve(),
        verify_all_hashes=True,
    )
    box1_usd_path = (
        asset_bundle.root / "objects" / "box" / "box.usd"
    ).resolve(strict=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}-staging"
    if staging_dir.exists():
        parser.error(f"staging directory already exists: {staging_dir}")
    staging_dir.mkdir()
    runtime_layer = asset_bundle.write_runtime_layer(staging_dir / "liangzhu_lidar.usda")

    app = AppLauncher(args)
    simulation_app = app.app
    recorder = None
    video = None
    try:
        import omni.usd
        import torch
        from pxr import Usd, UsdGeom

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        from conveyor_bench.isaac.liangzhu_lidar_probe import (
            BOX1_SUPPORT_CENTER_WORLD_XYZ_M,
            BOX2_SUPPORT_CENTER_WORLD_XYZ_M,
            COLA_WORLD_POSITION_XYZ_M,
            DIAGNOSTIC_BACKEND_ID,
            RAYCAST_MESH_ID_AUDIT,
            apply_and_validate_box_visual_alignment,
            disable_box_physics_collisions,
            lidar_scan_from_ray_caster,
            make_liangzhu_lidar_probe_scene_cfg,
        )
        from conveyor_bench.isaac.scene import (
            TASK_AREA_GROUND_XYZ_M,
            disable_liangzhu_background_collision,
            place_workcell_in_liangzhu_task_area,
            validate_liangzhu_stage,
            validate_object_fixtures,
        )
        from conveyor_bench.isaac.workcell import apply_d436_runtime_intrinsics
        from conveyor_bench.perception import (
            ProbeRecorder,
            ThreePanelVideoWriter,
            UnitreeL2ProvisionalConfig,
        )
        from conveyor_bench.sidecar.objects import COLA_OBJECT

        lidar_config = UnitreeL2ProvisionalConfig()
        simulation = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / args.physics_hz,
                render_interval=1,
                device=args.device,
                use_fabric=True,
                physx=sim_utils.PhysxCfg(enable_enhanced_determinism=True),
                render=sim_utils.RenderCfg(antialiasing_mode=args.antialiasing_mode),
            )
        )
        scene_cfg = make_liangzhu_lidar_probe_scene_cfg(
            runtime_layer,
            cola_usd_path=asset_bundle.object_usd(COLA_OBJECT.object_id),
            box1_usd_path=box1_usd_path,
            box2_usd_path=asset_bundle.object_usd("box2"),
            lidar_config=lidar_config,
            head_camera_depth=args.sync_capture_time is not None,
        )
        scene = InteractiveScene(scene_cfg)
        stage = omni.usd.get_context().get_stage()
        placement = place_workcell_in_liangzhu_task_area(scene, stage)
        box_visual_alignment = apply_and_validate_box_visual_alignment(stage)
        stage_contract = validate_liangzhu_stage(stage)
        collision_policy = disable_liangzhu_background_collision(
            stage,
            stage_contract["collision_mesh_prims"],
        )
        box_collision_policy = disable_box_physics_collisions(stage)
        object_contract = validate_object_fixtures(
            stage,
            (COLA_OBJECT,),
            {COLA_OBJECT.object_id: asset_bundle.object_usd(COLA_OBJECT.object_id)},
        )
        simulation.reset()
        scene.reset()
        camera_contract = apply_d436_runtime_intrinsics(scene["head_camera"])

        robot = scene["robot"]
        initial_root_state = robot.data.default_root_state.clone()
        initial_root_state[:, :3] += scene.env_origins
        initial_root_state[:, 7:] = 0.0
        initial_robot_yaw_rad = math.atan2(
            BOX1_SUPPORT_CENTER_WORLD_XYZ_M[1]
            - float(initial_root_state[0, 1].item()),
            BOX1_SUPPORT_CENTER_WORLD_XYZ_M[0]
            - float(initial_root_state[0, 0].item()),
        )
        initial_joint_positions = robot.data.default_joint_pos.clone()
        initial_joint_velocities = torch.zeros_like(initial_joint_positions)
        robot.write_joint_state_to_sim(initial_joint_positions, initial_joint_velocities)

        cola = scene["object_00"]
        cola_state = cola.data.default_root_state.clone()
        cola_state[:, :3] = torch.tensor(
            [COLA_WORLD_POSITION_XYZ_M],
            device=simulation.device,
            dtype=torch.float32,
        )
        cola_state[:, 3:7] = torch.tensor(
            [COLA_OBJECT.stable_poses_wxyz[0]],
            device=simulation.device,
            dtype=torch.float32,
        )
        cola_state[:, 7:] = 0.0
        cola.write_root_pose_to_sim(cola_state[:, :7])
        cola.write_root_velocity_to_sim(cola_state[:, 7:])

        resolved_config = {
            "contract_id": "liangzhu_l2_three_panel_probe_v1",
            "scene": {
                "name": "liangzhu_box1_coke_to_box2_no_conveyor",
                "task_ground_world_xyz_m": TASK_AREA_GROUND_XYZ_M,
                "cola_world_xyz_m": COLA_WORLD_POSITION_XYZ_M,
                "box1_support_center_world_xyz_m": (
                    BOX1_SUPPORT_CENTER_WORLD_XYZ_M
                ),
                "box2_support_center_world_xyz_m": (
                    BOX2_SUPPORT_CENTER_WORLD_XYZ_M
                ),
                "initial_robot_yaw_rad": initial_robot_yaw_rad,
                "arm_state": "robot_default",
                "scripted_root_motion": True,
                "cola_pose_held_for_scan_diagnostic": True,
            },
            "backend": {
                "id": DIAGNOSTIC_BACKEND_ID,
                "formal_unitree_l2_rtx_result": False,
                "background_raycast": "isaaclab_warp_single_mesh",
                "foreground_lidar_geometry": (
                    "analytic_ray_intersections_aligned_to_visual_support_aabbs"
                ),
                "foreground_proxy_visible_in_rgb": False,
                "robot_self_returns": False,
                "raycast_mesh_id_audit": RAYCAST_MESH_ID_AUDIT,
                "intensity_synthetic": True,
                "reason": (
                    "Stable Warp background ray casting is merged ray-by-ray "
                    "with explicitly audited, visual-aligned foreground geometry"
                ),
            },
            "lidar": lidar_config.to_dict(),
            "video": {
                "layout": [
                    "head_rgb_with_overview_inset",
                    "gravity_aligned_current_raw_oblique_3d_range_color",
                    "world_raw_last_1s_oblique_3d_recency_color",
                ],
                "fps": args.video_fps,
                "resolution_wh": [1920, 720],
                "display_clip_m": [lidar_config.raw_min_range_m, lidar_config.display_max_range_m],
            },
            "simulation": {
                "duration_s": args.duration_seconds,
                "physics_hz": args.physics_hz,
                "device": args.device,
                "headless": args.headless,
            },
            "assets": {
                "root": str(asset_bundle.root),
                "runtime_layer_relative_to_evidence": "../liangzhu_lidar.usda",
                "box1_usd": str(box1_usd_path),
                "box2_usd": str(asset_bundle.object_usd("box2")),
                "transfer_manifest_sha256": _sha256(asset_bundle.root / "TRANSFER_MANIFEST.sha256"),
            },
            "stage_contract": stage_contract,
            "object_contract": object_contract,
            "placement": placement,
            "collision_policy": collision_policy,
            "box_collision_policy": box_collision_policy,
            "box_visual_alignment": box_visual_alignment,
            "camera_contract": camera_contract,
        }
        recorder = ProbeRecorder(staging_dir / "evidence", resolved_config)
        video_path = staging_dir / "three_panel_lidar_probe.mp4"
        video = ThreePanelVideoWriter(video_path, lidar_config, fps=args.video_fps)

        total_steps = max(1, round(args.duration_seconds * args.physics_hz))
        render_stride = max(1, round(args.physics_hz / 25.0))
        video_period = 1.0 / args.video_fps
        next_video_time = 0.0
        next_scan_time = 0.0
        scan_index = 0
        latest_scan = None
        sync_capture_written = False
        for step in range(total_steps):
            sim_time_s = (step + 1) / args.physics_hz
            x_offset, yaw, phase = _scripted_root_offset(sim_time_s)
            root_state = initial_root_state.clone()
            root_state[:, 0] += x_offset * math.cos(initial_robot_yaw_rad)
            root_state[:, 1] += x_offset * math.sin(initial_robot_yaw_rad)
            world_yaw = initial_robot_yaw_rad + yaw
            root_state[:, 3:7] = torch.tensor(
                [
                    [
                        math.cos(world_yaw / 2.0),
                        0.0,
                        0.0,
                        math.sin(world_yaw / 2.0),
                    ]
                ],
                device=simulation.device,
                dtype=torch.float32,
            )
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.set_joint_position_target(initial_joint_positions)
            cola.write_root_pose_to_sim(cola_state[:, :7])
            cola.write_root_velocity_to_sim(cola_state[:, 7:])
            scene.write_data_to_sim()
            should_render = (step + 1) % render_stride == 0
            simulation.step(render=should_render)
            scene.update(1.0 / args.physics_hz)

            if sim_time_s + 1.0e-9 >= next_scan_time and (
                args.sync_capture_time is None or should_render
            ):
                latest_scan = lidar_scan_from_ray_caster(
                    scene["lidar"],
                    scan_index=scan_index,
                    sim_time_s=sim_time_s,
                    config=lidar_config,
                )
                recorder.record_scan(latest_scan)
                if (
                    args.sync_capture_time is not None
                    and not sync_capture_written
                    and sim_time_s + 1.0e-9 >= args.sync_capture_time
                ):
                    capture_dir = staging_dir / "evidence" / "raw" / "sync_capture"
                    capture_dir.mkdir(parents=True, exist_ok=False)
                    head_camera = scene["head_camera"]
                    rgb = head_camera.data.output["rgb"][0].detach().cpu().numpy()
                    if rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                    Image.fromarray(rgb).save(capture_dir / "head_rgb.png")
                    depth = (
                        head_camera.data.output["distance_to_image_plane"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    np.savez_compressed(capture_dir / "head_depth.npz", depth=depth)
                    camera_payload = {
                        "scan_index": latest_scan.scan_index,
                        "sim_time_s": latest_scan.sim_time_s,
                        "image_width": int(rgb.shape[1]),
                        "image_height": int(rgb.shape[0]),
                        "intrinsics_row_major": (
                            head_camera.data.intrinsic_matrices[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1)
                            .tolist()
                        ),
                        "camera_position_world_xyz_m": (
                            head_camera.data.pos_w[0].detach().cpu().numpy().tolist()
                        ),
                        "camera_orientation_world_from_ros_wxyz": (
                            head_camera.data.quat_w_ros[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .tolist()
                        ),
                        "camera_prim_local_to_world_row_major": np.asarray(
                            UsdGeom.Xformable(
                                stage.GetPrimAtPath(
                                    "/World/envs/env_0/Robot/base/head_cam"
                                )
                            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                        )
                        .reshape(-1)
                        .tolist(),
                        "robot_root_position_world_xyz_m": (
                            robot.data.root_pos_w[0].detach().cpu().numpy().tolist()
                        ),
                        "robot_root_orientation_world_wxyz": (
                            robot.data.root_quat_w[0].detach().cpu().numpy().tolist()
                        ),
                        "cola_root_position_world_xyz_m": (
                            cola.data.root_pos_w[0].detach().cpu().numpy().tolist()
                        ),
                        "depth_type": "distance_to_image_plane",
                        "lidar_raw_path": f"../scans/scan_{latest_scan.scan_index:06d}.npz",
                        "lidar_object_id_audit_path": (
                            f"../../audit/object_ids/scan_{latest_scan.scan_index:06d}.npz"
                        ),
                    }
                    (capture_dir / "camera.json").write_text(
                        json.dumps(camera_payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    sync_capture_written = True
                scan_index += 1
                next_scan_time += lidar_config.scan_period_s

            if should_render and latest_scan is not None and sim_time_s + 1.0e-9 >= next_video_time:
                video.write(
                    head_rgb=scene["head_camera"].data.output["rgb"],
                    overview_rgb=scene["overview_camera"].data.output["rgb"],
                    scan=latest_scan,
                    sim_time_s=sim_time_s,
                    phase=phase,
                )
                next_video_time += video_period

        video.close()
        video = None
        compatible_video_path = staging_dir / "three_panel_lidar_probe_h264.mp4"
        _transcode_h264(video_path, compatible_video_path)
        video_probe = _ffprobe(compatible_video_path)
        recorder.close(
            {
                "video_path": "../three_panel_lidar_probe_h264.mp4",
                "video_sha256": _sha256(compatible_video_path),
                "video_probe": video_probe,
                "source_video_path": "../three_panel_lidar_probe.mp4",
                "source_video_sha256": _sha256(video_path),
                "diagnostic_backend_only": True,
                "completed_sim_time_s": total_steps / args.physics_hz,
                "sync_capture_written": sync_capture_written,
            }
        )
        recorder = None
        staging_dir.rename(output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "video": str(output_dir / compatible_video_path.name),
                    "ffprobe": video_probe,
                },
                indent=2,
            )
        )
        return 0
    finally:
        if video is not None:
            video.close()
        if recorder is not None:
            recorder.close({"incomplete": True})
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
