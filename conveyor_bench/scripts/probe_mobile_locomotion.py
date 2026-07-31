#!/usr/bin/env python3
"""Run the vendored Go2-X5 policy on a floating-root flat-ground robot."""

from __future__ import annotations

import argparse
import json
from math import asin, atan2
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.isaac.locomotion import (
    ACTION_JOINT_ORDER,
    DEFAULT_ARM_POSE,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_LEG_POSE,
    DEFAULT_POLICY_PATH,
    FLAT_HEIGHT_SCAN_VALUE,
    POLICY_SHA256,
    STATE_JOINT_ORDER,
    build_observation,
    infer,
    leg_target,
    load_contract,
    load_policy,
)


PHYSICS_HZ = 400
POLICY_HZ = 50
DECIMATION = PHYSICS_HZ // POLICY_HZ
REPORT_SCHEMA = "conveyor-bench-mobile-locomotion-smoke-v1"


def roll_pitch_from_quaternion(
    quaternion_wxyz: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[float, float]:
    """Return roll and pitch for a finite scalar-first quaternion."""

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("quaternion_wxyz cannot have zero norm")
    w, x, y, z = quaternion / norm
    roll = atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, asin(pitch_sine)


def evaluate_samples(
    samples: list[dict[str, Any]],
    command: tuple[float, float, float],
) -> dict[str, Any]:
    """Evaluate velocity tracking, posture, stopping, and fall criteria."""

    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    track_samples = [sample for sample in samples if sample["phase"] == "track"]
    if not track_samples:
        raise ValueError("track samples are required")
    stop_samples = [sample for sample in samples if sample["phase"] == "stop"]

    state_values = np.asarray(
        [
            [
                *sample["root_position_m"],
                *sample["base_linear_velocity_mps"],
                *sample["base_angular_velocity_radps"],
                sample["roll_rad"],
                sample["pitch_rad"],
            ]
            for sample in samples
        ],
        dtype=np.float64,
    )
    if not np.isfinite(state_values).all():
        return {
            "passed": False,
            "failures": ["nonfinite_state"],
            "fall_detected": True,
            "sample_count": len(samples),
        }

    positions = state_values[:, 0:3]
    rolls = state_values[:, 9]
    pitches = state_values[:, 10]
    root_heights = positions[:, 2]
    max_abs_tilt = float(
        max(np.max(np.abs(rolls)), np.max(np.abs(pitches)))
    )
    minimum_root_height = float(np.min(root_heights))
    fall_detected = minimum_root_height < 0.18 or max_abs_tilt > 0.70

    steady_start = len(track_samples) // 2
    steady_samples = track_samples[steady_start:]
    steady_linear = np.asarray(
        [
            sample["base_linear_velocity_mps"]
            for sample in steady_samples
        ],
        dtype=np.float64,
    )
    steady_angular = np.asarray(
        [
            sample["base_angular_velocity_radps"]
            for sample in steady_samples
        ],
        dtype=np.float64,
    )
    measured = np.column_stack(
        (steady_linear[:, 0], steady_linear[:, 1], steady_angular[:, 2])
    )
    commanded = np.asarray(command, dtype=np.float64)
    error = measured - commanded
    means = np.mean(measured, axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    vx_gain = (
        float(means[0] / commanded[0])
        if abs(commanded[0]) > 1.0e-9
        else None
    )
    wz_gain = (
        float(means[2] / commanded[2])
        if abs(commanded[2]) > 1.0e-9
        else None
    )

    stop_speed = None
    stop_yaw_rate = None
    if stop_samples:
        stop_steady = stop_samples[len(stop_samples) // 2 :]
        stop_linear = np.asarray(
            [
                sample["base_linear_velocity_mps"]
                for sample in stop_steady
            ],
            dtype=np.float64,
        )
        stop_angular = np.asarray(
            [
                sample["base_angular_velocity_radps"]
                for sample in stop_steady
            ],
            dtype=np.float64,
        )
        stop_speed = float(np.mean(np.linalg.norm(stop_linear[:, :2], axis=1)))
        stop_yaw_rate = float(np.mean(np.abs(stop_angular[:, 2])))

    failures: list[str] = []
    if fall_detected:
        failures.append("fall_detected")
    if max_abs_tilt > 0.35:
        failures.append("posture_tilt")
    if float(np.std(root_heights)) > 0.05:
        failures.append("root_height_stability")
    if vx_gain is not None and not 0.70 <= vx_gain <= 1.30:
        failures.append("vx_gain")
    if float(rmse[0]) > max(0.04, 0.30 * abs(commanded[0])):
        failures.append("vx_rmse")
    if abs(commanded[1]) < 1.0e-9 and float(rmse[1]) > 0.08:
        failures.append("vy_cross_axis_rmse")
    if abs(commanded[2]) < 1.0e-9:
        if float(rmse[2]) > 0.10:
            failures.append("wz_cross_axis_rmse")
    else:
        if wz_gain is not None and not 0.70 <= wz_gain <= 1.30:
            failures.append("wz_gain")
        if float(rmse[2]) > max(0.08, 0.30 * abs(commanded[2])):
            failures.append("wz_rmse")
    if stop_speed is not None and stop_speed > 0.08:
        failures.append("stop_linear_speed")
    if stop_yaw_rate is not None and stop_yaw_rate > 0.10:
        failures.append("stop_yaw_rate")

    track_positions = np.asarray(
        [sample["root_position_m"] for sample in track_samples],
        dtype=np.float64,
    )
    return {
        "passed": not failures,
        "failures": failures,
        "fall_detected": fall_detected,
        "sample_count": len(samples),
        "phase_sample_counts": {
            phase: sum(sample["phase"] == phase for sample in samples)
            for phase in ("initial", "settle", "track", "stop")
        },
        "root_displacement_total_m": (
            positions[-1] - positions[0]
        ).tolist(),
        "root_displacement_track_m": (
            track_positions[-1] - track_positions[0]
        ).tolist(),
        "minimum_root_height_m": minimum_root_height,
        "root_height_std_m": float(np.std(root_heights)),
        "max_abs_roll_pitch_rad": max_abs_tilt,
        "steady_tracking": {
            "window_samples": len(steady_samples),
            "command_vx_vy_wz": list(command),
            "mean_vx_vy_wz": means.tolist(),
            "rmse_vx_vy_wz": rmse.tolist(),
            "vx_gain": vx_gain,
            "wz_gain": wz_gain,
        },
        "stop": {
            "mean_planar_speed_mps": stop_speed,
            "mean_abs_yaw_rate_radps": stop_yaw_rate,
        },
        "thresholds": {
            "fall_minimum_root_height_m": 0.18,
            "fall_max_abs_roll_pitch_rad": 0.70,
            "acceptance_max_abs_roll_pitch_rad": 0.35,
            "acceptance_root_height_std_m": 0.05,
            "tracking_gain": [0.70, 1.30],
            "zero_axis_linear_rmse_mps": 0.08,
            "zero_axis_yaw_rmse_radps": 0.10,
            "stop_planar_speed_mps": 0.08,
            "stop_yaw_rate_radps": 0.10,
        },
    }


def _build_parser() -> tuple[argparse.ArgumentParser, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "gate"
        / "mobile_locomotion"
        / "report.json",
    )
    parser.add_argument("--vx", type=float, default=0.20)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--stop-seconds", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu", headless=True, enable_cameras=False)
    return parser, AppLauncher


def _validate_args(args: argparse.Namespace) -> None:
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    guardrail = contract["command_limits"]["benchmark_v1_guardrail"]
    vx_lower, vx_upper = guardrail["linear_x_mps"]
    wz_lower, wz_upper = guardrail["angular_z_radps"]
    if not vx_lower <= args.vx <= vx_upper:
        raise ValueError(f"--vx must be in [{vx_lower}, {vx_upper}]")
    minimum = guardrail["minimum_nonzero_forward_mps"]
    if 0.0 < args.vx < minimum:
        raise ValueError(f"nonzero --vx must be at least {minimum}")
    if not wz_lower <= args.wz <= wz_upper:
        raise ValueError(f"--wz must be in [{wz_lower}, {wz_upper}]")
    if args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds cannot be negative")
    if args.hold_seconds <= 0.0:
        raise ValueError("--hold-seconds must be positive")
    if args.stop_seconds < 0.0:
        raise ValueError("--stop-seconds cannot be negative")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")
    if args.device != "cpu":
        raise ValueError("this reproducible smoke requires --device cpu")


def _sample_robot(robot: Any, phase: str, time_seconds: float) -> dict[str, Any]:
    position = robot.data.root_pos_w[0].detach().cpu().tolist()
    quaternion = robot.data.root_quat_w[0].detach().cpu().tolist()
    linear_velocity = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
    angular_velocity = robot.data.root_ang_vel_b[0].detach().cpu().tolist()
    roll, pitch = roll_pitch_from_quaternion(quaternion)
    return {
        "phase": phase,
        "time_seconds": float(time_seconds),
        "root_position_m": [float(value) for value in position],
        "root_quaternion_wxyz": [float(value) for value in quaternion],
        "base_linear_velocity_mps": [
            float(value) for value in linear_velocity
        ],
        "base_angular_velocity_radps": [
            float(value) for value in angular_velocity
        ],
        "roll_rad": roll,
        "pitch_rad": pitch,
    }


def _run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    from conveyor_bench.isaac.asset_config import (
        ARM_JOINT_NAMES,
        GRIPPER_JOINT_NAMES,
        make_go2_x5_policy_cfg,
    )

    robot_cfg = make_go2_x5_policy_cfg()

    @configclass
    class MobileLocomotionSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )
        robot = robot_cfg

    simulation = None
    scene = None
    try:
        simulation = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1.0 / PHYSICS_HZ,
                render_interval=DECIMATION,
                device=args.device,
                use_fabric=False,
                physx=sim_utils.PhysxCfg(
                    enable_enhanced_determinism=True,
                    bounce_threshold_velocity=0.5,
                    friction_correlation_distance=0.025,
                ),
            )
        )
        scene = InteractiveScene(
            MobileLocomotionSceneCfg(
                num_envs=1,
                env_spacing=2.5,
                replicate_physics=True,
                clone_in_fabric=False,
                lazy_sensor_update=True,
            )
        )
        simulation.reset()
        scene.reset()

        robot = scene["robot"]
        if robot.is_fixed_base:
            raise RuntimeError("spawned Go2-X5 root is fixed")
        state_joint_ids, state_joint_names = robot.find_joints(
            list(STATE_JOINT_ORDER),
            preserve_order=True,
        )
        leg_joint_ids, action_joint_names = robot.find_joints(
            list(ACTION_JOINT_ORDER),
            preserve_order=True,
        )
        arm_joint_ids, _ = robot.find_joints(
            list(ARM_JOINT_NAMES),
            preserve_order=True,
        )
        gripper_joint_ids, _ = robot.find_joints(
            list(GRIPPER_JOINT_NAMES),
            preserve_order=True,
        )
        if tuple(state_joint_names) != STATE_JOINT_ORDER:
            raise RuntimeError(
                f"state joint order mismatch: {state_joint_names}"
            )
        if tuple(action_joint_names) != ACTION_JOINT_ORDER:
            raise RuntimeError(
                f"action joint order mismatch: {action_joint_names}"
            )

        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        root_state[:, 7:] = 0.0
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.clone(),
            robot.data.default_joint_vel.clone(),
        )
        scene.reset()

        policy = load_policy(DEFAULT_POLICY_PATH, device=args.device)
        command_tensor = torch.zeros(
            (1, 3),
            device=simulation.device,
            dtype=torch.float32,
        )
        last_action = torch.zeros(
            (1, len(ACTION_JOINT_ORDER)),
            device=simulation.device,
            dtype=torch.float32,
        )
        arm_target = torch.tensor(
            [DEFAULT_ARM_POSE],
            device=simulation.device,
            dtype=torch.float32,
        )
        gripper_observation = torch.zeros(
            (1, 1),
            device=simulation.device,
            dtype=torch.float32,
        )
        gripper_target = torch.full(
            (1, len(GRIPPER_JOINT_NAMES)),
            0.044,
            device=simulation.device,
            dtype=torch.float32,
        )

        samples = [_sample_robot(robot, "initial", 0.0)]
        physics_steps = 0
        policy_steps = 0
        phases = (
            ("settle", 0.0, 0.0, args.settle_seconds),
            ("track", args.vx, args.wz, args.hold_seconds),
            ("stop", 0.0, 0.0, args.stop_seconds),
        )
        for phase, vx, wz, duration_seconds in phases:
            phase_steps = round(duration_seconds * POLICY_HZ)
            command_tensor[:, :] = torch.tensor(
                (vx, 0.0, wz),
                device=simulation.device,
                dtype=torch.float32,
            )
            for _ in range(phase_steps):
                observation = build_observation(
                    {
                        "root_lin_vel_b": robot.data.root_lin_vel_b,
                        "root_ang_vel_b": robot.data.root_ang_vel_b,
                        "projected_gravity_b": robot.data.projected_gravity_b,
                        "joint_pos": robot.data.joint_pos[:, state_joint_ids],
                        "default_joint_pos": (
                            robot.data.default_joint_pos[:, state_joint_ids]
                        ),
                        "joint_vel": robot.data.joint_vel[:, state_joint_ids],
                    },
                    command_tensor,
                    last_action,
                    arm_target,
                    gripper_observation,
                )
                warmup_scale = (
                    1.0
                    if args.warmup_steps == 0
                    else min(1.0, (policy_steps + 1) / args.warmup_steps)
                )
                last_action = infer(
                    policy,
                    observation,
                    warmup_scale=warmup_scale,
                )
                robot.set_joint_position_target(
                    leg_target(last_action),
                    joint_ids=leg_joint_ids,
                )
                robot.set_joint_position_target(
                    arm_target,
                    joint_ids=arm_joint_ids,
                )
                robot.set_joint_position_target(
                    gripper_target,
                    joint_ids=gripper_joint_ids,
                )
                for _ in range(DECIMATION):
                    scene.write_data_to_sim()
                    simulation.step(render=False)
                    scene.update(1.0 / PHYSICS_HZ)
                    physics_steps += 1
                policy_steps += 1
                samples.append(
                    _sample_robot(
                        robot,
                        phase,
                        physics_steps / PHYSICS_HZ,
                    )
                )

        metrics = evaluate_samples(
            samples,
            (args.vx, 0.0, args.wz),
        )
        return {
            "schema_version": REPORT_SCHEMA,
            "status": "completed",
            "passed": metrics["passed"],
            "runtime": {
                "device": args.device,
                "physics_backend": "PhysX",
                "use_fabric": False,
                "fabric_note": (
                    "disabled explicitly because Isaac Lab Warp fabric-array "
                    "operations require CUDA"
                ),
                "headless": bool(args.headless),
                "physics_hz": PHYSICS_HZ,
                "policy_hz": POLICY_HZ,
                "decimation": DECIMATION,
                "robot_asset": "assets/robots/go2_x5/go2_x5.usd",
                "fix_base": False,
                "spawned_is_fixed_base": bool(robot.is_fixed_base),
                "observation_dimension": 260,
                "action_dimension": 12,
                "height_scan": {
                    "mode": "constant_direct_flat_approximation",
                    "value": FLAT_HEIGHT_SCAN_VALUE,
                    "live_raycaster": False,
                },
            },
            "policy": {
                "asset": "assets/policies/go2_x5_pct_dog_only/policy.pt",
                "sha256": POLICY_SHA256,
            },
            "schedule": {
                "settle_seconds": args.settle_seconds,
                "hold_seconds": args.hold_seconds,
                "stop_seconds": args.stop_seconds,
                "warmup_policy_steps": args.warmup_steps,
                "command_vx_vy_wz": [args.vx, 0.0, args.wz],
            },
            "metrics": metrics,
        }
    finally:
        scene = None
        if simulation is not None:
            simulation.clear_all_callbacks()
            simulation.clear_instance()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser, app_launcher_type = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    simulation_app = None
    try:
        launcher = app_launcher_type(args)
        simulation_app = launcher.app
        report = _run_smoke(args)
        exit_code = 0 if report["passed"] else 1
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "error",
            "passed": False,
            "runtime": {
                "device": args.device,
                "physics_backend": "PhysX",
                "headless": bool(args.headless),
                "physics_hz": PHYSICS_HZ,
                "policy_hz": POLICY_HZ,
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        exit_code = 2
    try:
        _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    finally:
        if simulation_app is not None:
            simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
