"""Self-contained Go2-X5 articulation configuration."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GO2_X5_URDF = PROJECT_ROOT / "assets" / "robots" / "go2_x5" / "go2_x5.urdf"

LEG_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
ARM_JOINT_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")
# arm-vla-grasp-sim pct_scene uses the FinRay tip frame as its calibrated TCP.
TCP_OFFSET_X_M = 0.15757


def make_go2_x5_cfg(*, fix_base: bool = True) -> ArticulationCfg:
    """Return a project-local Go2-X5 configuration.

    The actuator, solver and reset contracts match PCT. ``fix_base=True`` only
    pins the root for the stationary diagnostic; it does not select a second
    robot asset or alter the X5 model.
    """

    if not GO2_X5_URDF.is_file():
        raise FileNotFoundError(f"Go2-X5 URDF is missing: {GO2_X5_URDF}")

    root_height_m = 0.38
    leg_joint_positions = {
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.8,
        "FR_calf_joint": -1.5,
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FL_calf_joint": -1.5,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 0.8,
        "RR_calf_joint": -1.5,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.8,
        "RL_calf_joint": -1.5,
    }
    arm_joint_positions = {name: 0.0 for name in ARM_JOINT_NAMES}
    leg_actuators = {
        "legs": DCMotorCfg(
            joint_names_expr=list(LEG_JOINT_NAMES),
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        )
    }
    if not fix_base:
        # Exact reset pose and leg drive limits used to train the vendored
        # 260-D -> 12-D TorchScript locomotion actor.
        root_height_m = 0.30
        leg_joint_positions = {
            name: value
            for name, value in zip(
                LEG_JOINT_NAMES,
                (
                    0.1,
                    0.8,
                    -1.5,
                    -0.1,
                    0.8,
                    -1.5,
                    0.1,
                    1.0,
                    -1.5,
                    -0.1,
                    1.0,
                    -1.5,
                ),
                strict=True,
            )
        }
        arm_joint_positions = {
            name: value
            for name, value in zip(
                ARM_JOINT_NAMES,
                (0.0, 0.3, 0.5, 0.0, 0.0, 0.0),
                strict=True,
            )
        }
        leg_actuators = {
            "hip_and_thigh": DCMotorCfg(
                joint_names_expr=[
                    name
                    for name in LEG_JOINT_NAMES
                    if not name.endswith("_calf_joint")
                ],
                effort_limit=35.278,
                saturation_effort=35.278,
                velocity_limit=30.0,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
            ),
            "calf": DCMotorCfg(
                joint_names_expr=[
                    name
                    for name in LEG_JOINT_NAMES
                    if name.endswith("_calf_joint")
                ],
                effort_limit=44.4,
                saturation_effort=44.4,
                velocity_limit=30.0,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
            ),
        }

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(GO2_X5_URDF),
            fix_base=fix_base,
            merge_fixed_joints=True,
            replace_cylinders_with_capsules=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                fix_root_link=fix_base,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, root_height_m),
            joint_pos={
                **leg_joint_positions,
                **arm_joint_positions,
                "arm_joint7": 0.044,
                "arm_joint8": 0.044,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.90,
        actuators={
            **leg_actuators,
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit_sim=100.0,
                velocity_limit_sim=10.0,
                stiffness=1000.0,
                damping=50.0,
                friction=0.0,
            ),
            "gripper": DCMotorCfg(
                joint_names_expr=list(GRIPPER_JOINT_NAMES),
                effort_limit=20.0,
                saturation_effort=20.0,
                velocity_limit=1.0,
                stiffness=1000.0,
                damping=50.0,
                friction=0.0,
            ),
        },
    )


def make_go2_x5_policy_cfg() -> ArticulationCfg:
    """Return the checkpoint-matched floating-base locomotion articulation.

    This is the single source of truth shared by the standalone locomotion
    gate and the V1 whole-body runtime.  The PCT reference deliberately uses
    the canonical URDF here so stale USD layers cannot replace its robot
    geometry, joints, materials, or collision meshes.
    """

    cfg = make_go2_x5_cfg(fix_base=False)
    cfg.actuators = {
        "legs_hip_thigh": DCMotorCfg(
            joint_names_expr=[
                name
                for name in LEG_JOINT_NAMES
                if "hip" in name or "thigh" in name
            ],
            effort_limit=35.278,
            saturation_effort=35.278,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "legs_calf": DCMotorCfg(
            joint_names_expr=[
                name for name in LEG_JOINT_NAMES if "calf" in name
            ],
            effort_limit=44.4,
            saturation_effort=44.4,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=list(ARM_JOINT_NAMES),
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
            friction=0.0,
        ),
        "gripper": cfg.actuators["gripper"],
    }
    return cfg
