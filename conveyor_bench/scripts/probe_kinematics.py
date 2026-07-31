#!/usr/bin/env python3
"""Compare live fixed-URDF and policy-USD TCP poses with calibrated FK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


FIXED_JOINTS = (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)
POLICY_JOINTS = (
    -0.0002719297,
    0.29707828,
    0.43148327,
    -0.03687394,
    -0.00008331,
    0.000047826,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.utils.math import quat_apply, quat_inv

        from conveyor_bench.isaac.arm_kinematics import (
            CalibratedArmKinematics,
        )
        from conveyor_bench.isaac.asset_config import (
            ARM_JOINT_NAMES,
            TCP_OFFSET_X_M,
            make_go2_x5_cfg,
            make_go2_x5_policy_cfg,
        )

        simulation = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=0.0025,
                device=args.device,
                use_fabric=False,
            )
        )

        fixed_cfg = make_go2_x5_cfg(fix_base=True)
        fixed_cfg.prim_path = "/World/FixedRobot"
        fixed_cfg.init_state.pos = (0.0, -1.0, 0.38)
        fixed_robot = Articulation(fixed_cfg)

        policy_cfg = make_go2_x5_policy_cfg()
        policy_cfg.prim_path = "/World/PolicyRobot"
        policy_cfg.init_state.pos = (0.0, 1.0, 0.30)
        policy_cfg.spawn.rigid_props.disable_gravity = True
        policy_robot = Articulation(policy_cfg)

        simulation.reset()
        tcp_offset = torch.tensor(
            [[TCP_OFFSET_X_M, 0.0, 0.0]],
            dtype=torch.float32,
            device=simulation.device,
        )

        probes = (
            (
                "fixed_urdf",
                fixed_robot,
                FIXED_JOINTS,
                CalibratedArmKinematics.in_robot_root_frame(),
            ),
            (
                "policy_usd",
                policy_robot,
                POLICY_JOINTS,
                CalibratedArmKinematics.in_policy_usd_root_frame(),
            ),
        )
        resolved = []
        for _, robot, joints, _ in probes:
            arm_ids, arm_names = robot.find_joints(
                list(ARM_JOINT_NAMES), preserve_order=True
            )
            if arm_names != list(ARM_JOINT_NAMES):
                raise RuntimeError(f"unexpected arm joint order: {arm_names}")
            link6_ids, link_names = robot.find_bodies(
                ["arm_link6"], preserve_order=True
            )
            if link_names != ["arm_link6"]:
                raise RuntimeError(f"could not resolve arm_link6: {link_names}")

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(joint_pos)
            target = torch.tensor(
                [joints], dtype=torch.float32, device=simulation.device
            )
            joint_pos[:, arm_ids] = target
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            resolved.append((arm_ids, link6_ids[0]))

        for _ in range(2):
            fixed_robot.write_data_to_sim()
            policy_robot.write_data_to_sim()
            simulation.step(render=False)
            fixed_robot.update(simulation.get_physics_dt())
            policy_robot.update(simulation.get_physics_dt())

        report = {"tcp_offset_x_m": TCP_OFFSET_X_M, "probes": []}
        for (name, robot, requested_joints, solver), (
            arm_ids,
            link6_id,
        ) in zip(probes, resolved, strict=True):
            measured_joints = tuple(
                float(value)
                for value in robot.data.joint_pos[0, arm_ids]
                .detach()
                .cpu()
                .tolist()
            )
            link6_pose = robot.data.body_pose_w[0, link6_id]
            tcp_world = link6_pose[:3] + quat_apply(
                link6_pose[3:7].unsqueeze(0), tcp_offset
            )[0]
            tcp_root = quat_apply(
                quat_inv(robot.data.root_quat_w[0].unsqueeze(0)),
                (tcp_world - robot.data.root_pos_w[0]).unsqueeze(0),
            )[0]
            solver_position, _ = solver.forward(measured_joints)
            live = tcp_root.detach().cpu().numpy()
            report["probes"].append(
                {
                    "name": name,
                    "requested_joints": list(requested_joints),
                    "measured_joints": list(measured_joints),
                    "live_tcp_root_xyz": [
                        float(value) for value in live.tolist()
                    ],
                    "solver_tcp_root_xyz": [
                        float(value) for value in solver_position.tolist()
                    ],
                    "root_frame_error_m": float(
                        ((live - solver_position) ** 2).sum() ** 0.5
                    ),
                }
            )

        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
