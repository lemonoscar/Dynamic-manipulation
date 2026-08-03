"""Single-owner Isaac Sim loop for ConveyorBench V1 data collection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import numpy as np
import omni.usd
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

from conveyor_bench.m0_online import (
    PREGRASP_WORKSPACE_LIMITS_BASE,
    guard_pregrasp_tcp_target,
)
from conveyor_bench.v1.assets import (
    ASSET_LOCK_PATH,
    ObjectAsset,
    ReceptacleAsset,
    load_object_registry,
    load_receptacles,
    sha256_file,
    source_tree_fingerprint,
    verify_asset_lock,
)
from conveyor_bench.v1.camera_io import (
    CameraSpec,
    MultiCameraFrameWriter,
)
from conveyor_bench.v1.config import BenchmarkConfig
from conveyor_bench.v1.future_labels import with_realized_future_labels
from conveyor_bench.v1.metrics import EpisodeEvaluation
from conveyor_bench.v1.oracle import (
    DynamicSortOracle,
    OracleConfig,
    OracleObservation,
)
from conveyor_bench.v1.protocol import (
    CanonicalAction,
    EpisodeManifest,
    Event,
    EventKind,
    FailureReason,
    FutureObjectState,
    GoalZone,
    JointState,
    ObjectInstance,
    ObjectState,
    Pose,
    RobotMode,
    StepSample,
    TaskManifest,
    TaskType,
    Twist,
    make_run_id,
)
from conveyor_bench.v1.recorder import EpisodeRecorder
from conveyor_bench.v1.tasking import (
    TASKING_SCHEMA_VERSION,
    CurriculumSplit,
    InstructionLanguage,
    TaskFamily,
    split_object_ids,
)

from .arm_kinematics import (
    CalibratedArmKinematics,
    IKConvergenceError,
)
from .asset_config import (
    ARM_JOINT_NAMES,
    GO2_X5_URDF,
    GRIPPER_JOINT_NAMES,
    LEG_JOINT_NAMES,
    TCP_OFFSET_X_M,
    make_go2_x5_cfg,
    make_go2_x5_policy_cfg,
)
from .locomotion import (
    ACTION_JOINT_ORDER,
    DEFAULT_POLICY_PATH,
    POLICY_SHA256,
    STATE_JOINT_ORDER,
    build_observation,
    infer,
    leg_target,
    load_policy,
)
from .scene import apply_surface_velocity
from .scene_v1 import (
    BELT_CENTER_X_M,
    BELT_CENTER_Y_M,
    BELT_LENGTH_M,
    BELT_TOP_Z_M,
    BELT_WIDTH_M,
    EXIT_PLANE_POINT_WORLD,
    HEAD_CAMERA_OFFSET_WXYZ,
    HEAD_CAMERA_OFFSET_XYZ,
    LAYOUT_ID,
    OBJECT_ASSETS,
    OBJECT_ENTITY_NAMES,
    OBJECT_LANE_X_M,
    OBJECT_SPAWN_Y_M,
    OVERVIEW_CAMERA_OFFSET_WXYZ,
    OVERVIEW_CAMERA_OFFSET_XYZ,
    TRANSPORT_DIRECTION_WORLD,
    WRIST_CAMERA_OFFSET_WXYZ,
    WRIST_CAMERA_OFFSET_XYZ,
    ConveyorSceneV1Cfg,
    install_gripper_collision_proxies,
)


# Collision-free intercept-ready posture.  With the policy-USD mount and the
# pad-centered TCP this places the TCP at approximately [0.55, 0.0, 0.30] in
# the robot-root frame, just above the near-side conveyor lane.
_PREGRASP_ARM = (0.002, 1.431, 0.746, 0.686, 0.002, 0.0)
_LOCOMOTION_PHYSICS_HZ = 400
_LOCOMOTION_POLICY_HZ = 50
_LOCOMOTION_DECIMATION = 8
_LOCOMOTION_WARMUP_POLICY_STEPS = 50
# Stop the mobile base far enough forward that both the transverse belt
# and the near-side object lane are inside the calibrated arm workspace.
# The earlier -0.035 m target left the moving belt object roughly 0.74 m from
# the arm root and forced the Cartesian controller through an unreachable
# descent, which correctly tripped the forbidden-collision gate.
_LOCOMOTION_APPROACH_TARGET_X_M = 0.080
_LOCOMOTION_APPROACH_POSITION_TOLERANCE_M = 0.04
_LOCOMOTION_STOP_PLANAR_SPEED_MPS = 0.08
_LOCOMOTION_STABLE_DWELL_S = 0.50
_LOCOMOTION_APPROACH_TIMEOUT_S = 3.0
_LOCOMOTION_STABILIZE_TIMEOUT_S = 2.0
_ARM_PREPOSITION_TIMEOUT_S = 5.0
# Deterministic joint-space carry posture.  It keeps the payload high and
# close to the trunk while avoiding the low-reach Cartesian IK branch used at
# the conveyor.  Values and TCP pose are frozen together.
_MOBILE_COMPACT_ARM = (0.0, 1.0, 0.8, 0.0, 0.0, 0.0)
_MOBILE_COMPACT_TCP_BASE = (0.46590005, -0.00050342, 0.33512429)
_MOBILE_GOAL_STANDOFF_M = 0.38
_MOBILE_CARRY_SETTLE_S = 0.40
_MOBILE_CARRY_ARC_YAW_RAD = 0.45
_MOBILE_TURN_RATE_RADPS = 0.35
_MOBILE_NAVIGATE_SPEED_MPS = 0.16
# With the X5 shoulder mount offset, end-effector planar bearing is about
# 0.74 * arm_joint1 around the compact carry posture.
_MOBILE_ARM_Q1_PLANAR_GAIN = 0.74
_MOBILE_PLACE_CARTESIAN_STEP_M = 0.008
_MOBILE_PLACE_DESCEND_STEP_M = 0.003
_MOBILE_PLACE_HOLD_STEP_M = 0.003
_MOBILE_ROOT_HOLD_MIN_X_M = 0.025
_MOBILE_INTERCEPT_Y_WORLD_M = 0.10
_M0_SERVICE_PREPOSITION_PHASES = frozenset(
    {"arm_preposition", "sequential_rearm"}
)
_M0_SERVICE_HOLD_PHASES = frozenset(
    {"mobile_settle", "mobile_stabilize"}
)
_M0_ACTION_HORIZON = 16
_M0_TRANSITION_CHUNK_PHASES = frozenset(
    {"track", "descend", "close", "lift"}
)
_CAMERA_SPECS = (
    CameraSpec("head_rgb", 224, 224, "policy_observation"),
    CameraSpec("wrist_rgb", 224, 224, "policy_observation"),
    CameraSpec("overview_rgb", 480, 320, "observer_only"),
)


class _MobilePreconditionFailure(Exception):
    """Task-level failure before the dynamic-sort oracle is initialized."""

    def __init__(
        self,
        reason: FailureReason,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.code = code


@dataclass(frozen=True)
class RuntimeOptionsV1:
    output_root: Path
    robot_mode: RobotMode = RobotMode.FIXED_BASE
    episodes: int = 1
    seed: int = 0
    belt_speed_mps: float = 0.06
    max_duration_s: float = 20.0
    active_object_count: int = 3
    target_asset_id: str = "part_red_block"
    destination_zone_id: str | None = None
    device: str = "cpu"
    enable_cameras: bool = False
    save_camera_frames: bool = False
    curriculum_split: CurriculumSplit = CurriculumSplit.TRAIN
    task_family: TaskFamily | None = None
    instruction_language: InstructionLanguage = (
        InstructionLanguage.BILINGUAL
    )
    m0_policy_endpoint: str | None = None
    m0_state_statistics: Path | None = None
    m0_policy_timeout_s: float = 30.0
    m0_policy_seed: int = 20260803
    m0_actions_per_replan: int = 2
    m0_transition_actions_per_replan: int = 12
    m0_pregrasp_workspace_guard: bool = False

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if not 1 <= self.active_object_count <= len(OBJECT_ASSETS):
            raise ValueError("active_object_count is outside the object pool")
        if self.belt_speed_mps <= 0.0:
            raise ValueError("belt_speed_mps must be positive")
        if self.max_duration_s <= 0.0:
            raise ValueError("max_duration_s must be positive")
        if self.device != "cpu":
            raise ValueError(
                "V1 currently requires --device cpu: its validated "
                "PhysxSurfaceVelocityAPI contact path is CPU PhysX"
            )
        if self.save_camera_frames and not self.enable_cameras:
            raise ValueError(
                "save_camera_frames requires enable_cameras"
            )
        if (
            isinstance(self.m0_policy_timeout_s, bool)
            or not math.isfinite(self.m0_policy_timeout_s)
            or self.m0_policy_timeout_s <= 0.0
        ):
            raise ValueError("m0_policy_timeout_s must be finite and positive")
        if (
            isinstance(self.m0_policy_seed, bool)
            or not isinstance(self.m0_policy_seed, int)
            or self.m0_policy_seed < 0
        ):
            raise ValueError("m0_policy_seed must be a non-negative integer")
        if (
            isinstance(self.m0_actions_per_replan, bool)
            or not isinstance(self.m0_actions_per_replan, int)
            or not 1 <= self.m0_actions_per_replan <= _M0_ACTION_HORIZON
        ):
            raise ValueError("m0_actions_per_replan must be within [1, 16]")
        if (
            isinstance(self.m0_transition_actions_per_replan, bool)
            or not isinstance(self.m0_transition_actions_per_replan, int)
            or not 1
            <= self.m0_transition_actions_per_replan
            <= _M0_ACTION_HORIZON
        ):
            raise ValueError(
                "m0_transition_actions_per_replan must be within [1, 16]"
            )
        if not isinstance(self.m0_pregrasp_workspace_guard, bool):
            raise TypeError("m0_pregrasp_workspace_guard must be a bool")
        if self.robot_mode not in {
            RobotMode.FIXED_BASE,
            RobotMode.WHOLE_BODY_POLICY,
        }:
            raise ValueError(
                "runtime supports fixed_base or whole_body_policy"
            )
        object_ids = {asset.object_id for asset in OBJECT_ASSETS}
        if self.target_asset_id not in object_ids:
            raise ValueError(
                f"unknown target asset: {self.target_asset_id!r}"
            )
        if self.destination_zone_id is not None and self.destination_zone_id not in {
            "sort_bin_blue",
            "sort_bin_yellow",
        }:
            raise ValueError("destination must be one of the two sorting bins")
        if not isinstance(self.curriculum_split, CurriculumSplit):
            raise TypeError("curriculum_split must be a CurriculumSplit")
        resolved_family = self.task_family or (
            TaskFamily.SINGLE_TARGET
            if self.active_object_count == 1
            else TaskFamily.LANGUAGE_CONDITIONED
        )
        object.__setattr__(self, "task_family", resolved_family)
        if not isinstance(resolved_family, TaskFamily):
            raise TypeError("task_family must be a TaskFamily")
        if not isinstance(self.instruction_language, InstructionLanguage):
            raise TypeError(
                "instruction_language must be an InstructionLanguage"
            )
        if resolved_family is TaskFamily.CONTINUOUS_MULTI_TARGET:
            raise ValueError(
                "the physical collector supports one scored target per "
                "episode; continuous multi-target curricula are generated "
                "offline by v1.tasking"
            )
        if (
            resolved_family is TaskFamily.SINGLE_TARGET
            and self.active_object_count != 1
        ):
            raise ValueError("single_target requires active_object_count=1")
        if (
            resolved_family is TaskFamily.LANGUAGE_CONDITIONED
            and self.active_object_count < 2
        ):
            raise ValueError(
                "language_conditioned requires at least two active objects"
            )
        split_pool = split_object_ids()[self.curriculum_split]
        if self.target_asset_id not in split_pool:
            raise ValueError(
                f"target {self.target_asset_id!r} is not in "
                f"{self.curriculum_split.value!r} split"
            )
        if self.active_object_count > len(split_pool):
            raise ValueError(
                "active_object_count exceeds the selected split-local pool"
            )
        if self.m0_policy_endpoint is not None:
            if not isinstance(self.m0_policy_endpoint, str):
                raise TypeError("m0_policy_endpoint must be a string")
            if self.robot_mode is not RobotMode.WHOLE_BODY_POLICY:
                raise ValueError("online M0 requires whole_body_policy")
            if not self.enable_cameras:
                raise ValueError("online M0 requires enable_cameras")
            if resolved_family is not TaskFamily.SINGLE_TARGET:
                raise ValueError("the first online M0 gate requires single_target")
            if self.m0_state_statistics is None:
                raise ValueError("online M0 requires m0_state_statistics")
            object.__setattr__(
                self, "m0_state_statistics", Path(self.m0_state_statistics)
            )
        elif self.m0_pregrasp_workspace_guard:
            raise ValueError(
                "m0_pregrasp_workspace_guard requires online M0"
            )


@dataclass(frozen=True)
class _ResolvedTarget:
    instance_id: str
    asset: ObjectAsset
    zone: ReceptacleAsset


@dataclass(frozen=True)
class _ResolvedTask:
    manifest: TaskManifest
    assets: tuple[ObjectAsset, ...]
    targets: tuple[_ResolvedTarget, ...]
    spawn_y_by_id: dict[str, float]
    current_target_index: int = 0
    service_gated_spawn: bool = False

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("resolved task requires at least one target")
        if not 0 <= self.current_target_index < len(self.targets):
            raise ValueError("current_target_index is outside target sequence")

    @property
    def current_target(self) -> _ResolvedTarget:
        return self.targets[self.current_target_index]

    @property
    def target_asset(self) -> ObjectAsset:
        return self.current_target.asset

    @property
    def target_zone(self) -> ReceptacleAsset:
        return self.current_target.zone

    @property
    def target_instance_id(self) -> str:
        return self.current_target.instance_id

    def select_target(self, instance_id: str) -> "_ResolvedTask":
        for index, target in enumerate(self.targets):
            if target.instance_id == instance_id:
                return replace(self, current_target_index=index)
        raise ValueError(f"unknown resolved target: {instance_id!r}")


class ConveyorRuntimeV1:
    """Own physics, sensors, controllers, recording, and teardown."""

    def __init__(self, options: RuntimeOptionsV1):
        self.options = options
        self.benchmark = BenchmarkConfig.v1()
        actual_timing = (
            self.benchmark.physics_hz,
            self.benchmark.control_hz,
        )
        expected_timing = (
            _LOCOMOTION_PHYSICS_HZ,
            _LOCOMOTION_POLICY_HZ,
        )
        if actual_timing != expected_timing:
            raise RuntimeError(
                "unexpected locomotion timing: "
                f"expected {expected_timing}, got {actual_timing}"
            )
        self.physics_dt = 1.0 / self.benchmark.physics_hz
        self.control_dt = 1.0 / self.benchmark.control_hz
        self.physics_decimation = (
            self.benchmark.physics_hz // self.benchmark.control_hz
        )
        if self.physics_decimation != _LOCOMOTION_DECIMATION:
            raise RuntimeError(
                "unexpected locomotion decimation: "
                f"expected {_LOCOMOTION_DECIMATION}, "
                f"got {self.physics_decimation}"
            )
        self.camera_physics_stride = (
            self.benchmark.physics_hz // self.benchmark.camera_hz
        )
        self.model_control_stride = (
            self.benchmark.control_hz // self.benchmark.model_hz
        )
        self._physics_step_count = 0
        self._closed = False
        self._objects_by_asset_id = {
            asset.object_id: OBJECT_ENTITY_NAMES[index]
            for index, asset in enumerate(OBJECT_ASSETS)
        }
        self._m0_client = None
        self._m0_health: dict[str, Any] | None = None
        if self.options.m0_policy_endpoint is not None:
            from conveyor_bench.m0_online import M0OnlineClient

            assert self.options.m0_state_statistics is not None
            self._m0_client = M0OnlineClient.from_files(
                self.options.m0_policy_endpoint,
                self.options.m0_state_statistics,
                timeout_s=self.options.m0_policy_timeout_s,
            )
            self._m0_health = dict(self._m0_client.health())

        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=self.physics_dt,
                render_interval=1,
                device=options.device,
                # RTX cameras read rendered transforms through Fabric.  With
                # Fabric disabled, physics still advances but Hydra keeps the
                # initial robot/object poses, producing plausible-looking
                # yet geometrically frozen observations.
                use_fabric=True,
                physx=sim_utils.PhysxCfg(
                    enable_enhanced_determinism=True,
                    bounce_threshold_velocity=0.2,
                    friction_correlation_distance=0.00625,
                ),
            )
        )
        try:
            eye, target = self._viewer_camera_view()
            self.sim.set_camera_view(eye=eye, target=target)
            scene_cfg = self._make_scene_cfg()
            scene_cfg.robot = (
                make_go2_x5_policy_cfg()
                if options.robot_mode is RobotMode.WHOLE_BODY_POLICY
                else make_go2_x5_cfg(fix_base=True)
            )
            if not options.enable_cameras:
                scene_cfg.head_camera = None
                scene_cfg.wrist_camera = None
                scene_cfg.overview_camera = None
            self.scene = InteractiveScene(scene_cfg)
            stage = omni.usd.get_context().get_stage()
            self.gripper_collision_contract = (
                install_gripper_collision_proxies(
                    stage,
                    "/World/envs/env_0/Robot",
                )
            )
            self.surface_velocity_api = apply_surface_velocity(
                stage,
                "/World/envs/env_0/TransportSurface",
                options.belt_speed_mps,
            )
            self.sim.reset()
            self.scene.reset()
            self._resolve_entities()
            self._warm_up()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        simulation = getattr(self, "sim", None)
        for name in (
            "robot",
            "objects",
            "left_contact_sensor",
            "right_contact_sensor",
            "head_camera",
            "wrist_camera",
            "overview_camera",
            "surface_velocity_api",
            "locomotion_policy",
            "_tcp_offset",
            "gripper_collision_contract",
        ):
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, "scene"):
            del self.scene
        if simulation is None:
            return
        simulation.clear_all_callbacks()
        simulation.clear_instance()
        del self.sim

    def run(self) -> dict[str, Any]:
        run_id = make_run_id()
        reports: list[dict[str, Any]] = []
        for episode_index in range(self.options.episodes):
            reports.append(
                self._run_episode(
                    run_id=run_id,
                    episode_index=episode_index,
                    episode_seed=self.options.seed + episode_index,
                )
            )
        summary = {
            "run_id": run_id,
            "protocol_version": self.benchmark.protocol_version,
            "task_type": self._summary_task_type().value,
            "robot_mode": self.options.robot_mode.value,
            "requested_episodes": self.options.episodes,
            "successful_episodes": sum(
                bool(report["success"]) for report in reports
            ),
            "episodes": reports,
        }
        self.options.output_root.mkdir(parents=True, exist_ok=True)
        path = self.options.output_root / f"{run_id}-summary.json"
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        summary["summary_path"] = str(path)
        return summary

    def _make_scene_cfg(self) -> ConveyorSceneV1Cfg:
        """Build the scene configuration while preserving the frozen V1 default."""

        scene_cfg = ConveyorSceneV1Cfg(
            num_envs=1,
            env_spacing=3.0,
            replicate_physics=True,
            clone_in_fabric=False,
            lazy_sensor_update=True,
        )
        return scene_cfg

    def _viewer_camera_view(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (-2.10, -1.60, 2.40), (0.62, 0.0, 0.45)

    def _asset_lock_path(self) -> Path:
        return ASSET_LOCK_PATH

    def _layout_id(self) -> str:
        return LAYOUT_ID

    def _camera_contract(self) -> dict[str, Any]:
        return _camera_contract_v1()

    def _guard_locomotion_command(
        self, command: Sequence[float]
    ) -> tuple[float, float, float]:
        """Apply the frozen V1 forward-only locomotion envelope."""

        return _guard_locomotion_command(command)

    def _summary_task_type(self) -> TaskType:
        return TaskType.DYNAMIC_SORT

    def _extra_episode_metadata(
        self, resolved: _ResolvedTask
    ) -> dict[str, Any]:
        del resolved
        return {}

    def _resolve_entities(self) -> None:
        self.robot = self.scene["robot"]
        expected_fixed_base = (
            self.options.robot_mode is RobotMode.FIXED_BASE
        )
        if bool(self.robot.is_fixed_base) != expected_fixed_base:
            raise RuntimeError(
                "spawned robot fixed-base state does not match robot_mode: "
                f"expected {expected_fixed_base}, "
                f"got {bool(self.robot.is_fixed_base)}"
            )
        self.objects = {
            asset.object_id: self.scene[entity_name]
            for asset, entity_name in zip(
                OBJECT_ASSETS, OBJECT_ENTITY_NAMES, strict=True
            )
        }
        self.left_contact_sensor = self.scene["left_finger_contact"]
        self.right_contact_sensor = self.scene["right_finger_contact"]
        self.head_camera = (
            self.scene["head_camera"]
            if self.options.enable_cameras
            else None
        )
        self.wrist_camera = (
            self.scene["wrist_camera"]
            if self.options.enable_cameras
            else None
        )
        self.overview_camera = (
            self.scene["overview_camera"]
            if self.options.enable_cameras
            else None
        )

        self.leg_joint_ids, leg_names = self.robot.find_joints(
            list(LEG_JOINT_NAMES), preserve_order=True
        )
        self.arm_joint_ids, arm_names = self.robot.find_joints(
            list(ARM_JOINT_NAMES), preserve_order=True
        )
        self.gripper_joint_ids, gripper_names = self.robot.find_joints(
            list(GRIPPER_JOINT_NAMES), preserve_order=True
        )
        self.locomotion_state_joint_ids, state_names = self.robot.find_joints(
            list(STATE_JOINT_ORDER), preserve_order=True
        )
        self.locomotion_action_joint_ids, action_names = self.robot.find_joints(
            list(ACTION_JOINT_ORDER), preserve_order=True
        )
        self.link6_body_ids, link_names = self.robot.find_bodies(
            ["arm_link6"], preserve_order=True
        )
        if leg_names != list(LEG_JOINT_NAMES):
            raise RuntimeError(f"unexpected leg joint order: {leg_names}")
        if arm_names != list(ARM_JOINT_NAMES):
            raise RuntimeError(f"unexpected arm joint order: {arm_names}")
        if gripper_names != list(GRIPPER_JOINT_NAMES):
            raise RuntimeError(
                f"unexpected gripper joint order: {gripper_names}"
            )
        if state_names != list(STATE_JOINT_ORDER):
            raise RuntimeError(
                f"unexpected locomotion state order: {state_names}"
            )
        if action_names != list(ACTION_JOINT_ORDER):
            raise RuntimeError(
                f"unexpected locomotion action order: {action_names}"
            )
        if link_names != ["arm_link6"]:
            raise RuntimeError("could not resolve arm_link6")

        kinematics_factory = (
            CalibratedArmKinematics.in_policy_usd_root_frame
            if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY
            else CalibratedArmKinematics.in_robot_root_frame
        )
        self.arm_kinematics = kinematics_factory(
            # The placement evaluator uses a 20 mm Cartesian gate.  Keep IK
            # tighter than that gate while allowing small calibration error
            # near the lateral sorting trays.
            position_tolerance_m=0.015,
            orientation_tolerance=0.08,
        )
        self._tcp_offset = torch.tensor(
            [[TCP_OFFSET_X_M, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self._arm_target = self.robot.data.joint_pos[
            :, self.arm_joint_ids
        ].clone()
        self._arm_ik_seed = tuple(
            float(value)
            for value in self._arm_target[0].detach().cpu().tolist()
        )
        self._last_ik_error_m = 0.0
        self._last_ik_iterations = 0
        self._last_policy_action = torch.zeros(
            (1, 12), dtype=torch.float32, device=self.sim.device
        )
        self._locomotion_policy_step_count = 0
        self._mobile_stable_since_s: float | None = None
        self.locomotion_policy = (
            load_policy(device=self.options.device)
            if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY
            else None
        )

    def _warm_up(self) -> None:
        self._reset_robot_state()
        for _ in range(self.camera_physics_stride):
            self._hold_all_joints()
            self.scene.write_data_to_sim()
            self._step_physics()
        if self.options.enable_cameras:
            _ = self.head_camera.data.output["rgb"]
            _ = self.wrist_camera.data.output["rgb"]
            _ = self.overview_camera.data.output["rgb"]

    def _run_episode(
        self,
        *,
        run_id: str,
        episode_index: int,
        episode_seed: int,
    ) -> dict[str, Any]:
        resolved = self._make_task(episode_seed)
        coordinator = None
        if len(resolved.targets) > 1:
            from conveyor_bench.v2.coordinator import (
                SequentialTargetCoordinator,
            )

            coordinator = SequentialTargetCoordinator(
                tuple(target.instance_id for target in resolved.targets),
                episode_start_time_s=0.0,
                episode_timeout_s=self.options.max_duration_s,
            )
        episode_id = (
            f"{run_id}-ep{episode_index:04d}-seed{episode_seed}-"
            f"{self.options.robot_mode.value}"
        )
        manifest = EpisodeManifest(
            episode_id=episode_id,
            run_id=run_id,
            protocol_version=self.benchmark.protocol_version,
            task=resolved.manifest,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            env_id=0,
            asset_hashes={
                asset.object_id: sha256_file(
                    Path(__file__).resolve().parents[3]
                    / "assets"
                    / "objects"
                    / "registry.json"
                )
                for asset in resolved.assets
            },
            seeds={"episode": episode_seed, "layout": episode_seed},
            metadata={
                "isaac_sim": _package_version("isaacsim"),
                "isaac_lab": _package_version("isaaclab"),
                "device": self.options.device,
                "use_fabric": True,
                "timing_hz": {
                    "physics": self.benchmark.physics_hz,
                    "control": self.benchmark.control_hz,
                    "camera": self.benchmark.camera_hz,
                    "model": self.benchmark.model_hz,
                },
                "asset_lock": verify_asset_lock(self._asset_lock_path()),
                "asset_lock_sha256": sha256_file(
                    self._asset_lock_path()
                ),
                "source_tree": source_tree_fingerprint(),
                "robot_urdf_sha256": sha256_file(GO2_X5_URDF),
                "gripper_collision": self.gripper_collision_contract,
                "locomotion_policy_sha256": (
                    POLICY_SHA256
                    if self.options.robot_mode
                    is RobotMode.WHOLE_BODY_POLICY
                    else None
                ),
                "layout_id": self._layout_id(),
                "cameras": self._camera_contract(),
                "canonical_action": {
                    "layout": [
                        "base_vx_body_mps",
                        "base_vy_body_mps",
                        "base_wz_body_radps",
                        "tcp_dx_base_m",
                        "tcp_dy_base_m",
                        "tcp_dz_base_m",
                        "tcp_drx_base_rad",
                        "tcp_dry_base_rad",
                        "tcp_drz_base_rad",
                        "gripper",
                    ],
                    "gripper": "1=open,-1=close",
                    "quaternion": "wxyz",
                    "units": "m-rad-s",
                },
                "controller": (
                    "m0_mobile_online"
                    if self._m0_client is not None
                    else "privileged_oracle"
                ),
                "m0_online_contract": self._m0_health,
                "m0_pregrasp_workspace_guard": {
                    "enabled": self.options.m0_pregrasp_workspace_guard,
                    "scope": "diagnostic_only",
                    "phase": "pregrasp",
                    "frame": "robot_base",
                    "bounds": {
                        axis: {"min": limits[0], "max": limits[1]}
                        for axis, limits in zip(
                            "xyz",
                            PREGRASP_WORKSPACE_LIMITS_BASE,
                            strict=True,
                        )
                    },
                },
                **self._extra_episode_metadata(resolved),
            },
        )

        recorder = EpisodeRecorder(
            self.options.output_root, manifest, self.benchmark
        )
        camera_writer: MultiCameraFrameWriter | None = None
        oracle: DynamicSortOracle | None = None
        oracle_terminal_reason: str | None = None
        phase = "reset"
        previous_phase: str | None = None
        target_spawned = False
        initial_spawn_assets: tuple[ObjectAsset, ...] = ()
        placed_event_ids: set[str] = set()
        approach_stage = (
            "mobile_settle"
            if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY
            else "fixed_ready"
        )
        stage_started_at = 0.0
        self._held_instance_id: str | None = None
        self._ever_held_target = False
        self._gripper_open = True
        self._last_gripper_open = True
        self._physics_step_count = 0
        wall_started = time.perf_counter()
        buffered_samples: list[StepSample] = []
        samples_flushed = False
        m0_chunk: tuple[tuple[float, ...], ...] = ()
        m0_chunk_sequence: int | None = None
        m0_chunk_server_ms: float | None = None
        m0_chunk_round_trip_ms: float | None = None
        m0_action_index = 0
        m0_next_sequence = 0
        m0_inference_ms: list[float] = []
        m0_round_trip_ms: list[float] = []
        m0_consumed_actions = 0
        m0_full_action_steps = 0
        m0_transition_action_steps = 0
        m0_base_only_action_steps = 0
        m0_service_preposition_steps = 0
        m0_service_hold_steps = 0
        m0_terminal_hold_steps = 0
        m0_safe_hold_steps = 0
        m0_workspace_guard_evaluated_steps = 0
        m0_workspace_guard_active_steps = 0
        m0_workspace_guard_axis_counts = {axis: 0 for axis in "xyz"}
        m0_workspace_guard_correction_norms: list[float] = []
        m0_workspace_guard_tracking_error_norms: list[float] = []
        m0_trace_stream = (
            (recorder.artifact_directory / "m0_online_trace.jsonl").open(
                "x", encoding="utf-8"
            )
            if self._m0_client is not None
            else None
        )

        def flush_realized_samples() -> None:
            nonlocal samples_flushed
            if samples_flushed:
                return
            realized = with_realized_future_labels(
                buffered_samples, self.benchmark
            )
            # Mark before writing so a recorder failure cannot retry a
            # partially written sequence and duplicate canonical rows.
            samples_flushed = True
            for buffered_sample in realized:
                recorder.record_step(buffered_sample)

        try:
            self._reset_episode(resolved)
            if self.options.robot_mode is RobotMode.FIXED_BASE:
                self._preposition_fixed_arm()
                if self._spawn_not_before_s(resolved) > 0.0:
                    approach_stage = "sequential_rearm"
                    stage_started_at = 0.0
                else:
                    initial_spawn_assets = (
                        self._spawn_assets_for_current_target(resolved)
                    )
                    self._spawn_task_objects(resolved, initial_spawn_assets)
                    target_spawned = True
                    oracle = self._make_oracle(resolved, sim_time_s=0.0)

            recorder.record_event(
                Event(
                    kind=EventKind.EPISODE_START,
                    time_s=0.0,
                    payload={
                        "instruction": resolved.manifest.instruction,
                        "robot_mode": self.options.robot_mode.value,
                    },
                )
            )
            if coordinator is not None:
                recorder.record_event(
                    Event(
                        kind=EventKind.TARGET_SELECTED,
                        time_s=0.0,
                        sim_step=self._physics_step_count,
                        object_instance_id=resolved.target_instance_id,
                        goal_zone_id=resolved.target_zone.zone_id,
                        payload={"target_index": 0},
                    )
                )
            if target_spawned:
                self._record_spawn_events(
                    recorder,
                    resolved,
                    time_s=0.0,
                    assets=initial_spawn_assets,
                )
            if self.options.save_camera_frames:
                camera_writer = MultiCameraFrameWriter(
                    recorder.artifact_directory, _CAMERA_SPECS
                )

            max_control_steps = round(
                self.options.max_duration_s * self.benchmark.control_hz
            )
            last_command_target_base: tuple[float, float, float] | None = None
            terminal = False
            for control_index in range(max_control_steps):
                sim_time_s = control_index * self.control_dt
                state_before = self._read_state(resolved)
                base_command = (0.0, 0.0, 0.0)
                gripper_open = self._gripper_open
                canonical_ee_delta = (0.0, 0.0, 0.0)
                canonical_rotvec = (0.0, 0.0, 0.0)
                selected_object_id: str | None = None
                oracle_command = None
                target_command_terminal = False
                target_command_success = False
                m0_step_metadata: dict[str, Any] | None = None
                m0_workspace_guard_metadata: dict[str, Any] | None = None
                shadow_arm_target = (
                    self._arm_target.clone()
                    if self._m0_client is not None
                    else None
                )
                shadow_arm_ik_seed = self._arm_ik_seed
                shadow_ik_error = self._last_ik_error_m
                shadow_ik_iterations = self._last_ik_iterations

                if oracle is None:
                    if approach_stage == "sequential_rearm":
                        phase = approach_stage
                        selected_object_id = resolved.target_instance_id
                        (
                            canonical_ee_delta,
                            canonical_rotvec,
                            last_command_target_base,
                        ) = self._command_pregrasp_joint_target(
                            state_before["tcp_base"]
                        )
                        gripper_open = True
                        arm_error = float(
                            torch.max(
                                torch.abs(
                                    self.robot.data.joint_pos[
                                        :, self.arm_joint_ids
                                    ]
                                    - torch.tensor(
                                        [_PREGRASP_ARM],
                                        dtype=torch.float32,
                                        device=self.sim.device,
                                    )
                                )
                            ).item()
                        )
                        ready_to_spawn = (
                            arm_error < 0.080
                            and sim_time_s - stage_started_at >= 0.30
                            and sim_time_s
                            >= self._spawn_not_before_s(resolved)
                        )
                        if (
                            not ready_to_spawn
                            and sim_time_s - stage_started_at
                            >= _ARM_PREPOSITION_TIMEOUT_S
                        ):
                            raise _MobilePreconditionFailure(
                                FailureReason.TIMEOUT,
                                "sequential_rearm_timeout",
                                "arm did not return to pregrasp before the "
                                "next service-gated target",
                            )
                    else:
                        phase = approach_stage
                        (
                            approach_stage,
                            stage_started_at,
                            base_command,
                            ready_to_spawn,
                        ) = self._mobile_preoracle_command(
                            stage=approach_stage,
                            stage_started_at=stage_started_at,
                            sim_time_s=sim_time_s,
                            root_x=state_before["root_pose"].xyz[0],
                            root_planar_speed_mps=math.hypot(
                                *state_before["root_twist"].linear_xyz[:2]
                            ),
                            robot_fallen=state_before["robot_fallen"],
                        )
                        phase = approach_stage
                        if approach_stage == "arm_preposition":
                            (
                                canonical_ee_delta,
                                canonical_rotvec,
                                last_command_target_base,
                            ) = self._command_pregrasp_joint_target(
                                state_before["tcp_base"]
                            )
                        else:
                            self._hold_arm_target()
                        gripper_open = True
                        ready_to_spawn = (
                            ready_to_spawn
                            and sim_time_s
                            >= self._spawn_not_before_s(resolved)
                        )
                    if ready_to_spawn:
                        if approach_stage == "sequential_rearm":
                            # Start each service-gated subtask from the arm
                            # branch that was physically reached during
                            # re-arm.  Keeping the previous target's lateral
                            # placement solution as the IK seed can select a
                            # discontinuous branch on the next interception.
                            measured_arm = self.robot.data.joint_pos[
                                :, self.arm_joint_ids
                            ]
                            self._arm_ik_seed = tuple(
                                float(value)
                                for value in measured_arm[0]
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            self._last_ik_error_m = 0.0
                            self._last_ik_iterations = 0
                        spawn_assets = self._spawn_assets_for_current_target(
                            resolved
                        )
                        self._spawn_task_objects(resolved, spawn_assets)
                        target_spawned = True
                        self._record_spawn_events(
                            recorder,
                            resolved,
                            time_s=sim_time_s,
                            assets=spawn_assets,
                        )
                        oracle = self._make_oracle(
                            resolved, sim_time_s=sim_time_s
                        )
                        phase = "settle"
                        if self._m0_client is not None:
                            # Never apply a chunk inferred from an image in
                            # which the service-gated object was still absent.
                            m0_chunk = ()
                            m0_chunk_sequence = None
                            m0_chunk_server_ms = None
                            m0_chunk_round_trip_ms = None
                            m0_action_index = 0
                else:
                    observation = self._oracle_observation(
                        resolved, state_before, sim_time_s
                    )
                    oracle_command = oracle.step(observation)
                    phase = oracle_command.phase.value
                    selected_object_id = oracle_command.selected_object_id
                    target_tcp_pose_world = (
                        oracle_command.target_tcp_pose_world
                    )
                    requested_base_command = (
                        oracle_command.base_command_body
                    )
                    if (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase in {"descend", "close"}
                        and not oracle_command.terminal
                    ):
                        # Descend on the interception line and let the moving
                        # part enter the gripper.  Chasing it sideways couples
                        # an unsupported lateral base motion into the Go2.
                        target_tcp_pose_world = Pose(
                            (
                                target_tcp_pose_world.xyz[0],
                                _MOBILE_INTERCEPT_Y_WORLD_M,
                                target_tcp_pose_world.xyz[2],
                            ),
                            target_tcp_pose_world.wxyz,
                        )
                    if (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase in {"track", "descend", "close"}
                        and state_before["root_pose"].xyz[0]
                        < _MOBILE_ROOT_HOLD_MIN_X_M
                    ):
                        # The locomotion actor has a tested forward dead-zone
                        # and no audited reverse/lateral command.  Apply its
                        # smallest valid forward command only after the arm
                        # load has pushed the root behind the hold threshold.
                        requested_base_command = (0.16, 0.0, 0.0)
                    if (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase in {"close", "lift"}
                        and not oracle_command.terminal
                    ):
                        # Preserve the orientation that actually established
                        # bilateral contact.  Correcting the small floating-
                        # base attitude error while lifting couples a large
                        # wrist/shoulder motion into the fragile new grasp.
                        target_tcp_pose_world = Pose(
                            target_tcp_pose_world.xyz,
                            state_before["tcp_world"].wxyz,
                        )
                    if (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and (
                            phase == "carry"
                            or self._mobile_continue_carry_before_place(
                                resolved, phase
                            )
                        )
                        and not oracle_command.terminal
                    ):
                        (
                            target_tcp_pose_world,
                            requested_base_command,
                            phase,
                        ) = self._mobile_carry_command(
                            resolved=resolved,
                            state=state_before,
                            oracle_target=target_tcp_pose_world,
                            sim_time_s=sim_time_s,
                        )
                    elif (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase
                        in {
                            "preplace",
                            "place_descend",
                            "open",
                            "retreat",
                            "verify_place",
                        }
                        and not oracle_command.terminal
                    ):
                        placement_step = (
                            math.inf
                            if phase in {"retreat", "verify_place"}
                            else _MOBILE_PLACE_CARTESIAN_STEP_M
                            if phase == "preplace"
                            else self._mobile_place_descend_step_m(resolved)
                            if phase == "place_descend"
                            else _MOBILE_PLACE_HOLD_STEP_M
                        )
                        target_tcp_pose_world = self._mobile_place_target(
                            target_tcp_pose_world,
                            state_before,
                            waypoint_step_m=placement_step,
                        )
                        requested_base_command = (0.0, 0.0, 0.0)
                    elif (
                        self.options.robot_mode is RobotMode.FIXED_BASE
                        and phase
                        in {
                            "carry",
                            "preplace",
                            "place_descend",
                            "open",
                            "retreat",
                            "verify_place",
                        }
                        and not oracle_command.terminal
                    ):
                        # The identity wrist attitude used over the belt is
                        # outside the X5's lateral tray workspace.  Choose the
                        # same position-coupled, reachable attitude used by
                        # the whole-body placement path; the base command
                        # remains zero for this ablation.
                        target_tcp_pose_world = self._mobile_place_target(
                            target_tcp_pose_world,
                            state_before,
                            waypoint_step_m=(
                                _MOBILE_PLACE_DESCEND_STEP_M
                                if phase in {"place_descend", "open"}
                                else _MOBILE_PLACE_CARTESIAN_STEP_M
                            ),
                        )
                    # Commands below the policy's audited forward dead-zone
                    # are explicitly treated as stationary.
                    base_command = self._guard_locomotion_command(
                        requested_base_command
                    )
                    gripper_open = oracle_command.gripper_command > 0.5
                    if (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase in {"retreat", "verify_place"}
                    ):
                        (
                            canonical_ee_delta,
                            canonical_rotvec,
                            last_command_target_base,
                        ) = self._command_mobile_retreat_joint_target(
                            target_tcp_pose_world,
                            state_before["tcp_base"],
                        )
                    elif (
                        self.options.robot_mode
                        is RobotMode.WHOLE_BODY_POLICY
                        and phase
                        in {
                            "carry_retract",
                            "carry_turn",
                            "carry_navigate",
                            "carry_settle",
                            "carry_recover",
                        }
                    ):
                        (
                            canonical_ee_delta,
                            canonical_rotvec,
                            last_command_target_base,
                        ) = self._command_mobile_compact_joint_target(
                            state_before["tcp_base"]
                        )
                    else:
                        (
                            canonical_ee_delta,
                            canonical_rotvec,
                            last_command_target_base,
                        ) = self._apply_tcp_command(
                            target_tcp_pose_world,
                            state_before["tcp_base"],
                            max_translation_m=(
                                0.015
                                if (
                                    self.options.robot_mode
                                    is RobotMode.WHOLE_BODY_POLICY
                                    and phase == "lift"
                                )
                                else _MOBILE_PLACE_CARTESIAN_STEP_M
                                if (
                                    self.options.robot_mode
                                    is RobotMode.WHOLE_BODY_POLICY
                                    and phase == "carry"
                                )
                                else self._mobile_place_descend_step_m(resolved)
                                if (
                                    self.options.robot_mode
                                    is RobotMode.WHOLE_BODY_POLICY
                                    and phase == "place_descend"
                                )
                                else _MOBILE_PLACE_HOLD_STEP_M
                                if (
                                    self.options.robot_mode
                                    is RobotMode.WHOLE_BODY_POLICY
                                    and phase == "open"
                                )
                                else _MOBILE_PLACE_CARTESIAN_STEP_M
                                if (
                                    self.options.robot_mode
                                    is RobotMode.WHOLE_BODY_POLICY
                                    and phase
                                    in {"preplace", "retreat", "verify_place"}
                                )
                                else _MOBILE_PLACE_DESCEND_STEP_M
                                if (
                                    self.options.robot_mode
                                    is RobotMode.FIXED_BASE
                                    and phase in {"place_descend", "open"}
                                )
                                else _MOBILE_PLACE_CARTESIAN_STEP_M
                                if (
                                    self.options.robot_mode
                                    is RobotMode.FIXED_BASE
                                    and phase
                                    in {
                                        "carry",
                                        "preplace",
                                        "retreat",
                                        "verify_place",
                                    }
                                )
                                else 0.002
                                if phase
                                in {"carry", "preplace", "place_descend"}
                                else 0.008
                                if phase in {"descend", "close", "lift"}
                                else 0.020
                            ),
                        )
                    if oracle_command.terminal:
                        target_command_terminal = True
                        target_command_success = oracle_command.success
                        oracle_terminal_reason = (
                            None
                            if oracle_command.success
                            else oracle_command.failure_reason
                        )

                m0_execution_prefix = (
                    self.options.m0_transition_actions_per_replan
                    if phase in _M0_TRANSITION_CHUNK_PHASES
                    else self.options.m0_actions_per_replan
                )
                if self._m0_client is not None:
                    assert shadow_arm_target is not None
                    scripted_preposition = (
                        oracle is None
                        and phase in _M0_SERVICE_PREPOSITION_PHASES
                    )
                    scripted_hold = (
                        oracle is None and phase in _M0_SERVICE_HOLD_PHASES
                    )
                    if not scripted_preposition:
                        # The oracle remains a shadow task evaluator.  Its
                        # candidate actuation is discarded before physics;
                        # only the service-gated M0 preposition below is kept.
                        self._arm_target = shadow_arm_target
                        self._arm_ik_seed = shadow_arm_ik_seed
                        self._last_ik_error_m = shadow_ik_error
                        self._last_ik_iterations = shadow_ik_iterations
                    if target_command_terminal:
                        base_command = (0.0, 0.0, 0.0)
                        canonical_ee_delta = (0.0, 0.0, 0.0)
                        canonical_rotvec = (0.0, 0.0, 0.0)
                        last_command_target_base = state_before[
                            "tcp_base"
                        ].xyz
                        self._hold_arm_target()
                        gripper_open = self._gripper_open
                        m0_terminal_hold_steps += 1
                        m0_step_metadata = {
                            "sequence_id": m0_chunk_sequence,
                            "action_index_control": None,
                            "policy_proposed_action10": None,
                            "server_inference_ms": m0_chunk_server_ms,
                            "round_trip_ms": m0_chunk_round_trip_ms,
                            "control_layer": "terminal_hold",
                            "dimension_sources": {
                                "base": "service",
                                "arm": "service",
                                "gripper": "service",
                            },
                            "execution_prefix": m0_execution_prefix,
                        }
                    elif (
                        m0_chunk
                        and m0_action_index < m0_execution_prefix
                    ):
                        model_action = m0_chunk[m0_action_index]
                        if scripted_preposition:
                            base_command = (0.0, 0.0, 0.0)
                            gripper_open = True
                            control_layer = "service_preposition"
                            dimension_sources = {
                                "base": "service",
                                "arm": "service",
                                "gripper": "service",
                            }
                            m0_service_preposition_steps += 1
                        elif scripted_hold:
                            base_command = (0.0, 0.0, 0.0)
                            canonical_ee_delta = (0.0, 0.0, 0.0)
                            canonical_rotvec = (0.0, 0.0, 0.0)
                            last_command_target_base = state_before[
                                "tcp_base"
                            ].xyz
                            self._hold_arm_target()
                            gripper_open = True
                            control_layer = "service_hold"
                            dimension_sources = {
                                "base": "service",
                                "arm": "service",
                                "gripper": "service",
                            }
                            m0_service_hold_steps += 1
                        elif phase == "mobile_approach":
                            from conveyor_bench.m0_online import (
                                quantize_go2_forward_intent,
                            )

                            base_command = self._guard_locomotion_command(
                                quantize_go2_forward_intent(model_action[:3])
                            )
                            canonical_ee_delta = (0.0, 0.0, 0.0)
                            canonical_rotvec = (0.0, 0.0, 0.0)
                            last_command_target_base = state_before[
                                "tcp_base"
                            ].xyz
                            self._hold_arm_target()
                            gripper_open = True
                            control_layer = "m0_base_only"
                            dimension_sources = {
                                "base": "m0",
                                "arm": "service",
                                "gripper": "service",
                            }
                            m0_base_only_action_steps += 1
                        else:
                            (
                                base_command,
                                canonical_ee_delta,
                                canonical_rotvec,
                                last_command_target_base,
                                gripper_open,
                                m0_workspace_guard_metadata,
                            ) = self._apply_m0_mobile_action(
                                model_action,
                                state_before,
                                guard_pregrasp_workspace=(
                                    self.options.m0_pregrasp_workspace_guard
                                    and phase == "pregrasp"
                                ),
                            )
                            control_layer = (
                                "m0_full_with_workspace_guard"
                                if m0_workspace_guard_metadata is not None
                                else "m0_full"
                            )
                            dimension_sources = {
                                "base": "m0",
                                "arm": (
                                    "m0+fixed_workspace_guard"
                                    if m0_workspace_guard_metadata is not None
                                    else "m0"
                                ),
                                "gripper": "m0",
                            }
                            m0_full_action_steps += 1
                            if phase in _M0_TRANSITION_CHUNK_PHASES:
                                m0_transition_action_steps += 1
                            if m0_workspace_guard_metadata is not None:
                                m0_workspace_guard_evaluated_steps += 1
                                correction_norm = float(
                                    m0_workspace_guard_metadata[
                                        "correction_norm_m"
                                    ]
                                )
                                m0_workspace_guard_correction_norms.append(
                                    correction_norm
                                )
                                if m0_workspace_guard_metadata["active"]:
                                    m0_workspace_guard_active_steps += 1
                                for axis in m0_workspace_guard_metadata[
                                    "clipped_axes"
                                ]:
                                    m0_workspace_guard_axis_counts[axis] += 1
                        m0_step_metadata = {
                            "sequence_id": m0_chunk_sequence,
                            "action_index_control": m0_action_index,
                            "policy_proposed_action10": model_action,
                            "server_inference_ms": m0_chunk_server_ms,
                            "round_trip_ms": m0_chunk_round_trip_ms,
                            "control_layer": control_layer,
                            "dimension_sources": dimension_sources,
                            "execution_prefix": m0_execution_prefix,
                        }
                        if m0_workspace_guard_metadata is not None:
                            m0_step_metadata["workspace_guard"] = (
                                m0_workspace_guard_metadata
                            )
                        m0_action_index += 1
                        m0_consumed_actions += 1
                    else:
                        base_command = (0.0, 0.0, 0.0)
                        if scripted_preposition:
                            gripper_open = True
                            control_layer = "service_preposition"
                            m0_service_preposition_steps += 1
                        elif scripted_hold:
                            canonical_ee_delta = (0.0, 0.0, 0.0)
                            canonical_rotvec = (0.0, 0.0, 0.0)
                            last_command_target_base = state_before[
                                "tcp_base"
                            ].xyz
                            self._hold_arm_target()
                            gripper_open = True
                            control_layer = "service_hold"
                            m0_service_hold_steps += 1
                        else:
                            canonical_ee_delta = (0.0, 0.0, 0.0)
                            canonical_rotvec = (0.0, 0.0, 0.0)
                            last_command_target_base = state_before[
                                "tcp_base"
                            ].xyz
                            self._hold_arm_target()
                            gripper_open = self._gripper_open
                            control_layer = "safe_hold"
                            m0_safe_hold_steps += 1
                        m0_step_metadata = {
                            "sequence_id": m0_chunk_sequence,
                            "action_index_control": None,
                            "policy_proposed_action10": None,
                            "server_inference_ms": m0_chunk_server_ms,
                            "round_trip_ms": m0_chunk_round_trip_ms,
                            "control_layer": control_layer,
                            "dimension_sources": {
                                "base": "service",
                                "arm": "service",
                                "gripper": "service",
                            },
                            "execution_prefix": m0_execution_prefix,
                        }

                self._apply_gripper(gripper_open)
                applied_policy_action = self._apply_base_command(base_command)
                for _ in range(self.physics_decimation):
                    self.scene.write_data_to_sim()
                    self._step_physics()

                sample_time_s = (control_index + 1) * self.control_dt
                state_after = self._read_state(resolved)
                if (
                    m0_step_metadata is not None
                    and "workspace_guard" in m0_step_metadata
                ):
                    guard_metadata = m0_step_metadata["workspace_guard"]
                    realized_position = tuple(
                        float(value) for value in state_after["tcp_base"].xyz
                    )
                    applied_target = tuple(
                        float(value)
                        for value in guard_metadata[
                            "applied_tcp_target_xyz"
                        ]
                    )
                    tracking_error = tuple(
                        realized - target
                        for realized, target in zip(
                            realized_position, applied_target, strict=True
                        )
                    )
                    tracking_error_norm = float(
                        np.linalg.norm(tracking_error)
                    )
                    guard_metadata["realized_tcp_after_xyz"] = list(
                        realized_position
                    )
                    guard_metadata["realized_tracking_error_xyz"] = list(
                        tracking_error
                    )
                    guard_metadata["realized_tracking_error_norm_m"] = (
                        tracking_error_norm
                    )
                    m0_workspace_guard_tracking_error_norms.append(
                        tracking_error_norm
                    )
                camera_frames = ()
                if (
                    camera_writer is not None
                    and self._physics_step_count
                    % self.camera_physics_stride
                    == 0
                ):
                    camera_frames = camera_writer.add(
                        sim_step=self._physics_step_count,
                        capture_time_s=sample_time_s,
                        images={
                            "head_rgb": self.head_camera.data.output["rgb"],
                            "wrist_rgb": self.wrist_camera.data.output["rgb"],
                            "overview_rgb": self.overview_camera.data.output[
                                "rgb"
                            ],
                        },
                    )
                if (
                    self._m0_client is not None
                    and self._physics_step_count
                    % self.camera_physics_stride
                    == 0
                    and (
                        not m0_chunk
                        or m0_action_index >= m0_execution_prefix
                    )
                    and not target_command_terminal
                ):
                    online_observation = self._m0_live_state28(state_after)
                    policy_result = self._m0_client.infer(
                        self._camera_rgb_numpy(
                            self.head_camera.data.output["rgb"]
                        ),
                        self._camera_rgb_numpy(
                            self.wrist_camera.data.output["rgb"]
                        ),
                        resolved.manifest.instruction,
                        online_observation,
                        sequence_id=m0_next_sequence,
                        request_id=(
                            f"{episode_id}:seq-{m0_next_sequence}"
                        ),
                        seed=(
                            self.options.m0_policy_seed + m0_next_sequence
                        )
                        % (2**31),
                    )
                    m0_chunk = policy_result.physical_actions
                    m0_chunk_sequence = policy_result.sequence_id
                    m0_chunk_server_ms = (
                        policy_result.server_inference_ms
                    )
                    m0_chunk_round_trip_ms = policy_result.round_trip_ms
                    m0_action_index = 0
                    m0_next_sequence += 1
                    m0_inference_ms.append(policy_result.server_inference_ms)
                    m0_round_trip_ms.append(policy_result.round_trip_ms)
                    assert m0_trace_stream is not None
                    json.dump(
                        {
                            "schema_version": "conveyor-bench-m0-runtime-trace-1",
                            "sequence_id": policy_result.sequence_id,
                            "request_id": policy_result.request_id,
                            "observation_sim_step": self._physics_step_count,
                            "observation_time_s": sample_time_s,
                            "source_model_tick": control_index
                            // self.model_control_stride,
                            "execution_prefix": m0_execution_prefix,
                            "server_inference_ms": (
                                policy_result.server_inference_ms
                            ),
                            "round_trip_ms": policy_result.round_trip_ms,
                            "normalized_actions": (
                                policy_result.normalized_actions
                            ),
                            "physical_actions": (
                                policy_result.physical_actions
                            ),
                        },
                        m0_trace_stream,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    m0_trace_stream.write("\n")
                    m0_trace_stream.flush()

                if phase != previous_phase:
                    recorder.record_event(
                        Event(
                            kind=EventKind.PHASE_CHANGED,
                            time_s=sample_time_s,
                            sim_step=self._physics_step_count,
                            object_instance_id=selected_object_id,
                            payload={
                                "from": previous_phase,
                                "to": phase,
                            },
                        )
                    )
                    previous_phase = phase
                if self._last_gripper_open and not gripper_open:
                    recorder.record_event(
                        Event(
                            kind=EventKind.GRASP_ATTEMPT,
                            time_s=sample_time_s,
                            sim_step=self._physics_step_count,
                            object_instance_id=resolved.target_asset.object_id,
                        )
                    )
                if (
                    not self._last_gripper_open
                    and gripper_open
                    and self._ever_held_target
                ):
                    recorder.record_event(
                        Event(
                            kind=EventKind.OBJECT_RELEASED,
                            time_s=sample_time_s,
                            sim_step=self._physics_step_count,
                            object_instance_id=resolved.target_asset.object_id,
                            goal_zone_id=resolved.target_zone.zone_id,
                        )
                    )
                self._last_gripper_open = gripper_open

                action = CanonicalAction(
                    (
                        *tuple(float(value) for value in base_command),
                        *canonical_ee_delta,
                        *canonical_rotvec,
                        1.0 if gripper_open else -1.0,
                    )
                )
                if m0_step_metadata is not None:
                    m0_step_metadata["applied_canonical_action10"] = list(
                        action.values
                    )
                sample = self._make_sample(
                    resolved=resolved,
                    state=state_after,
                    action=action,
                    phase=phase,
                    selected_object_id=selected_object_id,
                    sim_time_s=sample_time_s,
                    model_tick=control_index
                    // self.model_control_stride,
                    camera_frames=camera_frames,
                    base_command=base_command,
                    policy_action=applied_policy_action,
                    oracle_target_base=last_command_target_base,
                    m0_step_metadata=m0_step_metadata,
                )
                buffered_samples.append(sample)

                if target_command_terminal and coordinator is not None:
                    completed_target = resolved.current_target
                    if target_command_success:
                        recorder.record_event(
                            Event(
                                kind=EventKind.OBJECT_PLACED,
                                time_s=sample_time_s,
                                sim_step=self._physics_step_count,
                                object_instance_id=(
                                    completed_target.instance_id
                                ),
                                goal_zone_id=completed_target.zone.zone_id,
                                payload={
                                    "target_index": (
                                        resolved.current_target_index
                                    )
                                },
                            )
                        )
                        placed_event_ids.add(completed_target.instance_id)
                        transition = coordinator.mark_success(
                            completed_target.instance_id,
                            sim_time_s=sample_time_s,
                        )
                        if transition.episode_terminal:
                            terminal = True
                        else:
                            assert transition.next_target_id is not None
                            resolved = resolved.select_target(
                                transition.next_target_id
                            )
                            recorder.record_event(
                                Event(
                                    kind=EventKind.TARGET_SELECTED,
                                    time_s=sample_time_s,
                                    sim_step=self._physics_step_count,
                                    object_instance_id=(
                                        resolved.target_instance_id
                                    ),
                                    goal_zone_id=(
                                        resolved.target_zone.zone_id
                                    ),
                                    payload={
                                        "target_index": (
                                            resolved.current_target_index
                                        ),
                                        "after_target_instance_id": (
                                            completed_target.instance_id
                                        ),
                                    },
                                )
                            )
                            oracle = None
                            oracle_terminal_reason = None
                            approach_stage = "sequential_rearm"
                            stage_started_at = sample_time_s
                            target_spawned = False
                            self._held_instance_id = None
                            self._ever_held_target = False
                            self._gripper_open = True
                            self._last_gripper_open = True
                            previous_phase = phase
                            if self._m0_client is not None:
                                m0_chunk = ()
                                m0_chunk_sequence = None
                                m0_chunk_server_ms = None
                                m0_chunk_round_trip_ms = None
                                m0_action_index = 0
                    else:
                        failure = oracle_terminal_reason or "oracle_failure"
                        coordinator.mark_failure(
                            completed_target.instance_id,
                            sim_time_s=sample_time_s,
                            reason=failure,
                        )
                        terminal = True
                elif target_command_terminal:
                    terminal = True

                if terminal:
                    break

            if m0_trace_stream is not None:
                m0_trace_stream.flush()
                os.fsync(m0_trace_stream.fileno())
                m0_trace_stream.close()
            if camera_writer is not None:
                camera_writer.close()
            flush_realized_samples()
            evaluation = recorder.online_metrics.finalize()
            if terminal and oracle_terminal_reason and evaluation.success:
                raise RuntimeError(
                    "oracle failed while metric evaluator reported success"
                )
            if oracle_terminal_reason and not evaluation.success:
                metrics = dict(evaluation.metrics)
                metrics["oracle_failure_reason"] = oracle_terminal_reason
                evaluation = EpisodeEvaluation(
                    success=False,
                    failure_reason=evaluation.failure_reason,
                    metrics=metrics,
                )
            if evaluation.success:
                for target in resolved.targets:
                    if target.instance_id in placed_event_ids:
                        continue
                    recorder.record_event(
                        Event(
                            kind=EventKind.OBJECT_PLACED,
                            time_s=recorder.online_metrics.snapshot()[
                                "duration_s"
                            ],
                            sim_step=self._physics_step_count,
                            object_instance_id=target.instance_id,
                            goal_zone_id=target.zone.zone_id,
                        )
                    )
            episode_path, evaluation = recorder.finalize(evaluation)
        except Exception as error:
            if m0_trace_stream is not None and not m0_trace_stream.closed:
                try:
                    m0_trace_stream.close()
                except Exception:
                    pass
            if camera_writer is not None:
                try:
                    camera_writer.close()
                except Exception:
                    pass
            if recorder.finalized:
                raise
            if not samples_flushed:
                try:
                    flush_realized_samples()
                except Exception as flush_error:
                    error = RuntimeError(
                        "episode failed and buffered samples could not be "
                        f"published: {flush_error}"
                    )
            failure_reason = FailureReason.RUNTIME_ERROR
            abort_metadata = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "phase": phase,
            }
            if isinstance(error, _MobilePreconditionFailure):
                failure_reason = error.reason
                abort_metadata["precondition_failure"] = error.code
            episode_path, evaluation = recorder.abort(
                failure_reason,
                abort_metadata,
            )

        report = {
            "episode_id": episode_id,
            "path": str(episode_path),
            "success": evaluation.success,
            "failure_reason": evaluation.failure_reason.value,
            "metrics": evaluation.metrics,
            "camera_frames": (
                camera_writer.frame_count
                if camera_writer is not None
                else 0
            ),
            "wall_time_s": time.perf_counter() - wall_started,
        }
        if self._m0_client is not None:
            report["m0_online"] = {
                "request_count": len(m0_inference_ms),
                "consumed_action_count": m0_consumed_actions,
                "applied_action_count": (
                    m0_full_action_steps + m0_base_only_action_steps
                ),
                "full_action_control_steps": m0_full_action_steps,
                "transition_action_control_steps": (
                    m0_transition_action_steps
                ),
                "base_only_action_control_steps": m0_base_only_action_steps,
                "service_preposition_control_steps": (
                    m0_service_preposition_steps
                ),
                "service_hold_control_steps": m0_service_hold_steps,
                "terminal_hold_control_steps": m0_terminal_hold_steps,
                "safe_hold_control_steps": m0_safe_hold_steps,
                "server_inference_ms": _latency_summary(m0_inference_ms),
                "round_trip_ms": _latency_summary(m0_round_trip_ms),
                "observation_hz_simulated": self.benchmark.camera_hz,
                "default_executed_actions_per_chunk": (
                    self.options.m0_actions_per_replan
                ),
                "transition_executed_actions_per_chunk": (
                    self.options.m0_transition_actions_per_replan
                ),
                "transition_chunk_phases": sorted(
                    _M0_TRANSITION_CHUNK_PHASES
                ),
                "pregrasp_workspace_guard": {
                    "enabled": self.options.m0_pregrasp_workspace_guard,
                    "scope": "diagnostic_only",
                    "phase": "pregrasp",
                    "frame": "robot_base",
                    "bounds": {
                        axis: {"min": limits[0], "max": limits[1]}
                        for axis, limits in zip(
                            "xyz",
                            PREGRASP_WORKSPACE_LIMITS_BASE,
                            strict=True,
                        )
                    },
                    "evaluated_control_steps": (
                        m0_workspace_guard_evaluated_steps
                    ),
                    "intervention_control_steps": (
                        m0_workspace_guard_active_steps
                    ),
                    "intervention_rate": (
                        m0_workspace_guard_active_steps
                        / m0_workspace_guard_evaluated_steps
                        if m0_workspace_guard_evaluated_steps
                        else 0.0
                    ),
                    "clipped_axis_control_steps": (
                        m0_workspace_guard_axis_counts
                    ),
                    "correction_norm_m": _latency_summary(
                        m0_workspace_guard_correction_norms
                    ),
                    "realized_tracking_error_norm_m": _latency_summary(
                        m0_workspace_guard_tracking_error_norms
                    ),
                },
            }
        return report

    def _make_task(self, seed: int) -> _ResolvedTask:
        rng = random.Random(seed)
        asset_by_id = {asset.object_id: asset for asset in OBJECT_ASSETS}
        target = asset_by_id[self.options.target_asset_id]
        split_pool = split_object_ids()[self.options.curriculum_split]
        candidates = [
            asset_by_id[asset_id]
            for asset_id in split_pool
            if asset_id != target.object_id
        ]
        distractors = rng.sample(
            candidates, self.options.active_object_count - 1
        )
        assets = (target, *distractors)
        receptacles = {
            item.zone_id: item for item in load_receptacles()
        }
        zone_id = self.options.destination_zone_id or (
            "sort_bin_blue" if seed % 2 == 0 else "sort_bin_yellow"
        )
        target_zone = receptacles[zone_id]
        zones = tuple(
            _goal_zone(receptacles[item])
            for item in ("sort_bin_blue", "sort_bin_yellow")
        )
        objects = tuple(
            ObjectInstance(
                instance_id=asset.object_id,
                asset_id=asset.object_id,
                class_id=asset.category,
                goal_zone_id=(
                    target_zone.zone_id
                    if asset.object_id == target.object_id
                    else None
                ),
            )
            for asset in assets
        )
        target_alias = target.language_aliases["en"][0]
        instruction_en = (
            f"Pick the {target_alias} moving from left to right and place it "
            f"in the {target_zone.display_name}."
        )
        instruction_zh = (
            f"抓取从左向右移动的{target.language_aliases['zh'][0]}，"
            f"并将它放入"
            f"{'蓝色' if zone_id == 'sort_bin_blue' else '黄色'}分拣盘。"
        )
        instruction = (
            instruction_en
            if self.options.instruction_language
            is InstructionLanguage.ENGLISH
            else f"[EN] {instruction_en} [ZH] {instruction_zh}"
        )
        spawn_slots = (0.34, 0.48, 0.10, -0.14, -0.34, 0.25, -0.02, 0.42)
        spawn_y_by_id = {
            asset.object_id: spawn_slots[index]
            for index, asset in enumerate(assets)
        }
        manifest = TaskManifest(
            task_id=(
                f"dynamic-sort-{self.options.curriculum_split.value}-"
                f"{self.options.task_family.value}-"
                f"{self.options.robot_mode.value}-seed-{seed}"
            ),
            task_type=TaskType.DYNAMIC_SORT,
            robot_mode=self.options.robot_mode,
            instruction=instruction,
            objects=objects,
            goal_zones=zones,
            scored_object_ids=(target.object_id,),
            seed=seed,
            belt_speed_mps=self.options.belt_speed_mps,
            belt_surface_z_m=BELT_TOP_Z_M,
            transport_direction_xyz=TRANSPORT_DIRECTION_WORLD,
            exit_plane_point_xyz=EXIT_PLANE_POINT_WORLD,
            max_duration_s=self.options.max_duration_s,
            metadata={
                "tasking_schema_version": TASKING_SCHEMA_VERSION,
                "curriculum_split": self.options.curriculum_split.value,
                "registry_split": (
                    "unseen"
                    if self.options.curriculum_split
                    is CurriculumSplit.UNSEEN
                    else "seen"
                ),
                "task_family": self.options.task_family.value,
                "instruction_language": (
                    self.options.instruction_language.value
                ),
                "canonical_instruction": instruction,
                "canonical_instruction_en": instruction_en,
                "canonical_instruction_zh": instruction_zh,
                "active_object_ids": tuple(
                    asset.object_id for asset in assets
                ),
                "active_asset_ids": tuple(
                    asset.object_id for asset in assets
                ),
                "target_id": target.object_id,
                "target_ids": (target.object_id,),
                "destination_zone": target_zone.zone_id,
                "destination_zone_by_target": {
                    target.object_id: target_zone.zone_id
                },
                "distractors": tuple(
                    asset.object_id for asset in distractors
                ),
                "distractor_asset_ids": tuple(
                    asset.object_id for asset in distractors
                ),
                "layout_id": LAYOUT_ID,
                "target_asset_id": target.object_id,
                "destination_zone_id": target_zone.zone_id,
                "active_object_count": len(assets),
                "spawn_y_by_id": spawn_y_by_id,
                "spawn_policy": (
                    "after_mobile_approach_and_arm_preposition"
                    if self.options.robot_mode
                    is RobotMode.WHOLE_BODY_POLICY
                    else "after_fixed_arm_preposition"
                ),
                "object_split": {
                    asset.object_id: asset.split for asset in assets
                },
                "instruction_zh": instruction_zh,
            },
        )
        return _ResolvedTask(
            manifest=manifest,
            assets=assets,
            targets=(
                _ResolvedTarget(
                    instance_id=target.object_id,
                    asset=target,
                    zone=target_zone,
                ),
            ),
            spawn_y_by_id=spawn_y_by_id,
        )

    def _reset_episode(self, resolved: _ResolvedTask) -> None:
        self._reset_robot_state()
        for index, asset in enumerate(OBJECT_ASSETS):
            rigid_object = self.objects[asset.object_id]
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            root_state[:, 0] = 3.0
            root_state[:, 1] = -0.8 + index * 0.18
            root_state[:, 2] = 0.15
            root_state[:, 7:] = 0.0
            rigid_object.write_root_pose_to_sim(root_state[:, :7])
            rigid_object.write_root_velocity_to_sim(root_state[:, 7:])
        self.scene.reset()
        self._physics_step_count = 0
        self._arm_target = self.robot.data.joint_pos[
            :, self.arm_joint_ids
        ].clone()
        self._arm_ik_seed = tuple(
            float(value)
            for value in self._arm_target[0].detach().cpu().tolist()
        )
        self._last_policy_action.zero_()
        self._locomotion_policy_step_count = 0
        self._mobile_stable_since_s = None
        self._mobile_carry_stage: str | None = None
        self._mobile_carry_stage_started_s = 0.0
        self._mobile_carry_stable_since_s: float | None = None
        self._mobile_goal_yaw_rad: float | None = None
        self._mobile_goal_root_xy: tuple[float, float] | None = None
        self._mobile_carry_orientation_base_wxyz: (
            tuple[float, float, float, float] | None
        ) = None
        self._mobile_retreat_arm_target: (
            tuple[float, float, float, float, float, float] | None
        ) = None

    def _reset_robot_state(self) -> None:
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY:
            root_state[:, 0] = -0.22
            root_state[:, 2] = 0.30
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
        )
        self._hold_all_joints()

    def _preposition_fixed_arm(self) -> None:
        target = torch.tensor(
            [_PREGRASP_ARM],
            dtype=torch.float32,
            device=self.sim.device,
        )
        joint_positions = self.robot.data.joint_pos.clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, self.arm_joint_ids] = target
        joint_positions[:, self.gripper_joint_ids] = 0.044
        self.robot.write_joint_state_to_sim(
            joint_positions, joint_velocities
        )
        self._arm_target = target
        self.robot.set_joint_position_target(
            target, joint_ids=self.arm_joint_ids
        )
        self._apply_gripper(True)
        for _ in range(round(0.12 * self.benchmark.physics_hz)):
            self.robot.set_joint_position_target(
                self.robot.data.default_joint_pos[:, self.leg_joint_ids],
                joint_ids=self.leg_joint_ids,
            )
            self.scene.write_data_to_sim()
            self._step_physics()
        self._arm_ik_seed = _PREGRASP_ARM
        self._physics_step_count = 0

    def _spawn_assets_for_current_target(
        self, resolved: _ResolvedTask
    ) -> tuple[ObjectAsset, ...]:
        if resolved.service_gated_spawn:
            return (resolved.target_asset,)
        return resolved.assets

    def _spawn_not_before_s(self, resolved: _ResolvedTask) -> float:
        suite = resolved.manifest.metadata.get("benchmark_suite")
        if not isinstance(suite, dict):
            return 0.0
        if resolved.service_gated_spawn:
            gates = suite.get("service_gates", ())
            for gate in gates if isinstance(gates, (tuple, list)) else ():
                if (
                    isinstance(gate, dict)
                    and gate.get("target_instance_id")
                    == resolved.target_instance_id
                ):
                    return float(gate.get("not_before_s", 0.0))
            return 0.0
        schedule = resolved.manifest.metadata.get("spawn_schedule", ())
        return max(
            (
                float(entry.get("spawn_time_s", 0.0))
                for entry in schedule
                if isinstance(entry, dict)
            ),
            default=0.0,
        )

    def _spawn_task_objects(
        self,
        resolved: _ResolvedTask,
        assets: Sequence[ObjectAsset] | None = None,
    ) -> None:
        for asset in assets or resolved.assets:
            rigid_object = self.objects[asset.object_id]
            root_state = rigid_object.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            root_state[:, 0] = OBJECT_LANE_X_M
            root_state[:, 1] = resolved.spawn_y_by_id[asset.object_id]
            root_state[:, 2] = (
                BELT_TOP_Z_M + asset.half_extents_xyz[2] + 0.003
            )
            root_state[:, 3:7] = torch.tensor(
                [asset.stable_poses_wxyz[0]],
                dtype=torch.float32,
                device=self.sim.device,
            )
            root_state[:, 7:] = 0.0
            rigid_object.write_root_pose_to_sim(root_state[:, :7])
            rigid_object.write_root_velocity_to_sim(root_state[:, 7:])
        self.scene.reset()

    def _record_spawn_events(
        self,
        recorder: EpisodeRecorder,
        resolved: _ResolvedTask,
        *,
        time_s: float,
        assets: Sequence[ObjectAsset] | None = None,
    ) -> None:
        for asset in assets or resolved.assets:
            recorder.record_event(
                Event(
                    kind=EventKind.OBJECT_SPAWNED,
                    time_s=time_s,
                    sim_step=self._physics_step_count,
                    object_instance_id=asset.object_id,
                    payload={
                        "asset_id": asset.object_id,
                        "spawn_xyz": [
                            OBJECT_LANE_X_M,
                            resolved.spawn_y_by_id[asset.object_id],
                            BELT_TOP_Z_M
                            + asset.half_extents_xyz[2]
                            + 0.003,
                        ],
                    },
                )
            )

    def _make_oracle(
        self, resolved: _ResolvedTask, *, sim_time_s: float
    ) -> DynamicSortOracle:
        asset = resolved.target_asset
        affordance = asset.grasp_affordances[0]
        # Release above the tray rim and let gravity produce the final settled
        # pose. Commanding the TCP to the tray floor is both physically wrong
        # and outside the low corner of the X5 workspace.
        release_object_center_z = (
            resolved.target_zone.floor_top_z_m
            + resolved.target_zone.wall_height_m
            + asset.half_extents_xyz[2]
            + 0.015
        )
        zone_x = resolved.target_zone.center_xyz_m[0]
        zone_y = resolved.target_zone.center_xyz_m[1]
        # Bias toward the tray's robot-facing inner quadrant while retaining
        # enough wall clearance for the full 48 mm target, not merely its
        # center point. This avoids the X5 lateral workspace boundary after
        # the short mobile carry arc for both mirrored sorting trays.
        reachable_release_x = zone_x + 0.06
        reachable_release_y = zone_y - math.copysign(0.09, zone_y)
        oracle = DynamicSortOracle(
            OracleConfig(
                target_object_id=asset.object_id,
                goal_center_world=(
                    reachable_release_x,
                    reachable_release_y,
                    release_object_center_z,
                ),
                robot_mode=self.options.robot_mode,
                object_height_m=asset.nominal_height_m,
                grasp_offset_world=affordance.tcp_offset_xyz,
                tcp_orientation_wxyz=(-1.0, 0.0, 0.0, 0.0),
                intercept_horizon_s=0.12,
                # Wait above a fixed intercept point instead of sweeping the
                # floating arm laterally toward an upstream part.  Tracking
                # begins only once the predicted part enters this window.
                intercept_staging_y_world=_MOBILE_INTERCEPT_Y_WORLD_M,
                # Enter a narrow prediction window so the open 114 mm jaw is
                # centered before its lower collision envelope reaches the
                # 48 mm part.  A broad window closes above the moving part.
                intercept_entry_tolerance_m=0.005,
                # Contact is recorded as a learning signal, but closure waits
                # for the Cartesian grasp gate.  The open finger bottoms touch
                # the part about 50 mm above the pad center and must be allowed
                # to slide around it before the jaw closes.
                close_on_target_contact=False,
                pregrasp_clearance_m=0.10,
                # Clears the 0.555 m far rail with the tallest registered
                # object while staying inside the well-conditioned X5 volume.
                safe_carry_clearance_m=0.10,
                position_tolerance_m=0.020,
                # The TCP is the center of the usable finger grasp region.
                # Closing too high catches only the top edges and the part
                # slips during lift; collision-noise extrapolation is handled
                # separately by the transport-axis velocity below.
                grasp_tolerance_m=0.015,
                grasp_contact_dwell_s=0.04,
                # Lateral placement is deliberately rate-limited while the
                # arm carries a part.  Both robot modes need the same timeout
                # envelope to reach and settle over the side trays.
                phase_timeout_s=self._oracle_phase_timeout_s(resolved),
                episode_timeout_s=self.options.max_duration_s - sim_time_s,
                mobile_select_base_command_body=(0.0, 0.0, 0.0),
                mobile_carry_base_command_body=(0.0, 0.0, 0.0),
                # On the floating platform, first lift vertically at the
                # grasp XY.  Retraction to the compact body-relative carry
                # pose is a distinct stage in ``_mobile_carry_command``.
                # Combining both motions produces a large transient pitch
                # moment while the object is still just above the belt.
                mobile_lift_target_world=None,
            )
        )
        oracle.reset(sim_time_s=sim_time_s)
        return oracle

    def _oracle_phase_timeout_s(self, resolved: _ResolvedTask) -> float:
        del resolved
        return 15.0

    def _mobile_preoracle_command(
        self,
        *,
        stage: str,
        stage_started_at: float,
        sim_time_s: float,
        root_x: float,
        root_planar_speed_mps: float,
        robot_fallen: bool,
    ) -> tuple[str, float, tuple[float, float, float], bool]:
        if robot_fallen:
            raise _MobilePreconditionFailure(
                FailureReason.ROBOT_FALLEN,
                "robot_fallen",
                f"robot fell during mobile precondition stage {stage!r}",
            )
        elapsed = sim_time_s - stage_started_at
        if stage == "mobile_settle":
            warmup_duration_s = (
                _LOCOMOTION_WARMUP_POLICY_STEPS / _LOCOMOTION_POLICY_HZ
            )
            if elapsed >= warmup_duration_s:
                self._mobile_stable_since_s = None
                return (
                    "mobile_approach",
                    sim_time_s,
                    (0.20, 0.0, 0.0),
                    False,
                )
            return stage, stage_started_at, (0.0, 0.0, 0.0), False
        if stage == "mobile_approach":
            if root_x >= _LOCOMOTION_APPROACH_TARGET_X_M:
                self._mobile_stable_since_s = None
                return (
                    "mobile_stabilize",
                    sim_time_s,
                    (0.0, 0.0, 0.0),
                    False,
                )
            if elapsed >= _LOCOMOTION_APPROACH_TIMEOUT_S:
                raise _MobilePreconditionFailure(
                    FailureReason.TIMEOUT,
                    "approach_timeout",
                    "mobile base did not reach the approach target before timeout",
                )
            return stage, stage_started_at, (0.20, 0.0, 0.0), False
        if stage == "mobile_stabilize":
            position_ready = (
                abs(root_x - _LOCOMOTION_APPROACH_TARGET_X_M)
                <= _LOCOMOTION_APPROACH_POSITION_TOLERANCE_M
            )
            speed_ready = (
                root_planar_speed_mps
                <= _LOCOMOTION_STOP_PLANAR_SPEED_MPS
            )
            if position_ready and speed_ready:
                if self._mobile_stable_since_s is None:
                    self._mobile_stable_since_s = sim_time_s
                stable_duration_s = (
                    sim_time_s - self._mobile_stable_since_s
                )
            else:
                self._mobile_stable_since_s = None
                stable_duration_s = 0.0
            if stable_duration_s >= _LOCOMOTION_STABLE_DWELL_S:
                return (
                    "arm_preposition",
                    sim_time_s,
                    (0.0, 0.0, 0.0),
                    False,
                )
            if elapsed >= _LOCOMOTION_STABILIZE_TIMEOUT_S:
                raise _MobilePreconditionFailure(
                    FailureReason.TIMEOUT,
                    "stabilize_timeout",
                    "mobile base did not satisfy the position and speed dwell",
                )
            return stage, stage_started_at, (0.0, 0.0, 0.0), False
        if stage == "arm_preposition":
            position_ready = (
                abs(root_x - _LOCOMOTION_APPROACH_TARGET_X_M)
                <= _LOCOMOTION_APPROACH_POSITION_TOLERANCE_M
            )
            speed_ready = (
                root_planar_speed_mps
                <= _LOCOMOTION_STOP_PLANAR_SPEED_MPS
            )
            if position_ready and speed_ready:
                if self._mobile_stable_since_s is None:
                    self._mobile_stable_since_s = sim_time_s
            else:
                self._mobile_stable_since_s = None
            base_stable = (
                self._mobile_stable_since_s is not None
                and sim_time_s - self._mobile_stable_since_s
                >= _LOCOMOTION_STABLE_DWELL_S
            )
            arm_error = float(
                torch.max(
                    torch.abs(
                        self.robot.data.joint_pos[:, self.arm_joint_ids]
                        - torch.tensor(
                            [_PREGRASP_ARM],
                            dtype=torch.float32,
                            device=self.sim.device,
                        )
                    )
                ).item()
            )
            ready = (
                base_stable
                # The rated-effort floating-base drive can retain a small
                # gravity residual; the Cartesian oracle closes it after the
                # slow, collision-free joint preposition.
                and arm_error < 0.080
                and elapsed >= 0.30
            )
            if not ready and elapsed >= _ARM_PREPOSITION_TIMEOUT_S:
                raise _MobilePreconditionFailure(
                    FailureReason.TIMEOUT,
                    "arm_preposition_timeout",
                    "arm did not reach the collision-free pregrasp posture",
                )
            return stage, stage_started_at, (0.0, 0.0, 0.0), ready
        raise RuntimeError(f"unknown mobile pre-oracle stage: {stage}")

    def _mobile_carry_command(
        self,
        *,
        resolved: _ResolvedTask,
        state: dict[str, Any],
        oracle_target: Pose,
        sim_time_s: float,
    ) -> tuple[Pose, tuple[float, float, float], str]:
        """Coordinate retract, navigation and placement on the floating base."""

        root_pose: Pose = state["root_pose"]
        root_twist: Twist = state["root_twist"]
        if self._mobile_carry_stage is None:
            goal_yaw, goal_root_xy = self._plan_mobile_carry_goal(
                resolved, root_pose
            )
            self._mobile_goal_yaw_rad = goal_yaw
            self._mobile_goal_root_xy = goal_root_xy
            # Preserve the base-frame wrist attitude that established and
            # lifted the grasp. A hard-coded world quaternion produced a
            # roughly 0.8 rad IK jump during retraction, driving the arm down
            # while pitching the floating base.
            self._mobile_carry_orientation_base_wxyz = tuple(
                float(value) for value in state["tcp_base"].wxyz
            )
            self._transition_mobile_carry("retract", sim_time_s)

        assert self._mobile_goal_yaw_rad is not None
        assert self._mobile_goal_root_xy is not None
        assert self._mobile_carry_orientation_base_wxyz is not None
        # Retraction and locomotion constrain the compact TCP position but do
        # not servo a fixed wrist attitude against floating-base motion.
        # Reusing the measured base-frame attitude each control tick removes
        # a large shoulder/wrist correction that otherwise pitches the Go2
        # even though gripper orientation is irrelevant for transport.
        carry_orientation_base = tuple(
            float(value) for value in state["tcp_base"].wxyz
        )
        compact_target = self._root_pose_to_world(
            Pose(
                _MOBILE_COMPACT_TCP_BASE,
                carry_orientation_base,
            )
        )
        tilt = float(state["root_tilt_rad"])
        if tilt > 0.30:
            raise _MobilePreconditionFailure(
                FailureReason.ROBOT_FALLEN,
                "mobile_carry_unstable",
                "root tilt exceeded 0.30 rad during whole-body carry",
            )
        if tilt > 0.20:
            self._mobile_carry_stable_since_s = None
            return compact_target, (0.0, 0.0, 0.0), "carry_recover"

        stage = self._mobile_carry_stage
        elapsed = sim_time_s - self._mobile_carry_stage_started_s
        stage_timeout_s = self._mobile_carry_stage_timeout_s(stage)
        if elapsed > stage_timeout_s:
            raise _MobilePreconditionFailure(
                FailureReason.TIMEOUT,
                f"mobile_carry_{stage}_timeout",
                f"mobile carry stage {stage!r} exceeded its timeout",
            )

        if stage == "retract":
            compact_error = float(
                torch.max(
                    torch.abs(
                        self.robot.data.joint_pos[:, self.arm_joint_ids]
                        - torch.tensor(
                            [_MOBILE_COMPACT_ARM],
                            dtype=torch.float32,
                            device=self.sim.device,
                        )
                    )
                ).item()
            )
            arm_speed = float(
                torch.max(
                    torch.abs(
                        self.robot.data.joint_vel[:, self.arm_joint_ids]
                    )
                ).item()
            )
            if self._mobile_carry_dwell(
                compact_error <= 0.060 and arm_speed <= 0.35,
                sim_time_s,
                0.30,
            ):
                self._transition_mobile_carry("turn", sim_time_s)
                stage = "turn"
            return compact_target, (0.0, 0.0, 0.0), f"carry_{stage}"

        current_yaw = _yaw_from_wxyz(root_pose.wxyz)
        yaw_error = _wrap_angle(
            self._mobile_goal_yaw_rad - current_yaw
        )
        if stage == "turn":
            angular_speed = abs(root_twist.angular_xyz[2])
            if self._mobile_carry_dwell(
                abs(yaw_error) <= 0.08
                and angular_speed
                <= self._mobile_turn_angular_speed_tolerance_radps(resolved),
                sim_time_s,
                0.30,
            ):
                next_stage = self._mobile_post_turn_stage(resolved)
                self._transition_mobile_carry(next_stage, sim_time_s)
                return (
                    compact_target,
                    (0.0, 0.0, 0.0),
                    f"carry_{next_stage}",
                )
            yaw_command = (
                0.0
                if abs(yaw_error) <= 0.06
                else math.copysign(_MOBILE_TURN_RATE_RADPS, yaw_error)
            )
            forward_command = (
                0.0
                if abs(yaw_error) <= 0.06
                else self._mobile_turn_forward_speed_mps(resolved)
            )
            return (
                compact_target,
                (forward_command, 0.0, yaw_command),
                "carry_turn",
            )

        goal_x, goal_y = self._mobile_goal_root_xy
        position_error = math.hypot(
            goal_x - root_pose.xyz[0], goal_y - root_pose.xyz[1]
        )
        planar_speed = math.hypot(*root_twist.linear_xyz[:2])
        if stage == "navigate":
            if position_error <= self._mobile_navigation_position_tolerance_m(
                resolved
            ):
                self._transition_mobile_carry("settle", sim_time_s)
                return (
                    compact_target,
                    (0.0, 0.0, 0.0),
                    "carry_settle",
                )
            navigation_yaw_error = self._mobile_navigation_yaw_error(
                root_pose
            )
            yaw_command = self._mobile_navigation_yaw_command(
                resolved, navigation_yaw_error
            )
            forward_command = (
                0.0
                if abs(navigation_yaw_error) > 0.18
                else self._mobile_navigate_forward_speed_mps(
                    resolved, root_pose
                )
            )
            lateral_command = (
                0.0
                if abs(navigation_yaw_error) > 0.18
                else self._mobile_navigate_lateral_speed_mps(
                    resolved, root_pose
                )
            )
            return (
                compact_target,
                (forward_command, lateral_command, yaw_command),
                "carry_navigate",
            )

        if stage == "settle":
            position_tolerance = self._mobile_settle_position_tolerance_m(
                resolved
            )
            if (
                position_tolerance is not None
                and position_error > position_tolerance
            ):
                self._transition_mobile_carry("navigate", sim_time_s)
                return (
                    compact_target,
                    (0.0, 0.0, 0.0),
                    "carry_navigate",
                )
            if abs(yaw_error) > 0.15:
                self._transition_mobile_carry("turn", sim_time_s)
                return (
                    compact_target,
                    (0.0, 0.0, 0.0),
                    "carry_turn",
                )
            if self._mobile_carry_dwell(
                planar_speed <= 0.07
                and abs(root_twist.angular_xyz[2])
                <= self._mobile_settle_angular_speed_tolerance_radps(resolved),
                sim_time_s,
                _MOBILE_CARRY_SETTLE_S,
            ):
                self._transition_mobile_carry("place", sim_time_s)
                return (
                    self._mobile_place_target(oracle_target, state),
                    (0.0, 0.0, 0.0),
                    "carry",
                )
            return compact_target, (0.0, 0.0, 0.0), "carry_settle"

        assert stage == "place"
        return (
            self._mobile_place_target(oracle_target, state),
            (0.0, 0.0, 0.0),
            "carry",
        )

    def _plan_mobile_carry_goal(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> tuple[float, tuple[float, float]]:
        """Return the validated short-arc goal used by the V1 mobile task."""

        current_yaw = _yaw_from_wxyz(root_pose.wxyz)
        turn_sign = math.copysign(
            1.0,
            resolved.target_zone.center_xyz_m[1] - root_pose.xyz[1],
        )
        goal_yaw = _wrap_angle(
            current_yaw + turn_sign * _MOBILE_CARRY_ARC_YAW_RAD
        )
        signed_rate = turn_sign * _MOBILE_TURN_RATE_RADPS
        radius = _MOBILE_NAVIGATE_SPEED_MPS / signed_rate
        goal_root_xy = (
            root_pose.xyz[0]
            + radius * (math.sin(goal_yaw) - math.sin(current_yaw)),
            root_pose.xyz[1]
            - radius * (math.cos(goal_yaw) - math.cos(current_yaw)),
        )
        return goal_yaw, goal_root_xy

    def _mobile_post_turn_stage(self, resolved: _ResolvedTask) -> str:
        del resolved
        return "settle"

    def _mobile_continue_carry_before_place(
        self, resolved: _ResolvedTask, oracle_phase: str
    ) -> bool:
        del resolved, oracle_phase
        return False

    def _mobile_place_descend_step_m(self, resolved: _ResolvedTask) -> float:
        del resolved
        return _MOBILE_PLACE_DESCEND_STEP_M

    def _object_crossed_task_exit(
        self,
        resolved: _ResolvedTask,
        asset: ObjectAsset,
        position_world: Sequence[float],
        *,
        active: bool,
    ) -> bool:
        return active and _crossed_exit(
            position_world,
            resolved.manifest.exit_plane_point_xyz,
            resolved.manifest.transport_direction_xyz,
        )

    def _mobile_turn_forward_speed_mps(
        self, resolved: _ResolvedTask
    ) -> float:
        del resolved
        return _MOBILE_NAVIGATE_SPEED_MPS

    def _mobile_navigation_yaw_error(self, root_pose: Pose) -> float:
        """Return the heading error used while translating.

        V1 follows its audited constant-curvature arc and therefore keeps the
        final carry yaw as the translation heading.  Remote V2 overrides this
        hook with a closed-loop bearing to its explicit root goal.
        """

        assert self._mobile_goal_yaw_rad is not None
        return _wrap_angle(
            self._mobile_goal_yaw_rad - _yaw_from_wxyz(root_pose.wxyz)
        )

    def _mobile_navigate_forward_speed_mps(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> float:
        del resolved, root_pose
        return _MOBILE_NAVIGATE_SPEED_MPS

    def _mobile_navigate_lateral_speed_mps(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> float:
        del resolved, root_pose
        return 0.0

    def _mobile_navigation_yaw_command(
        self,
        resolved: _ResolvedTask,
        yaw_error_rad: float,
    ) -> float:
        del resolved
        return max(-0.15, min(0.15, 0.8 * yaw_error_rad))

    def _mobile_turn_angular_speed_tolerance_radps(
        self, resolved: _ResolvedTask
    ) -> float:
        del resolved
        return 0.12

    def _mobile_settle_angular_speed_tolerance_radps(
        self, resolved: _ResolvedTask
    ) -> float:
        del resolved
        return 0.12

    def _mobile_navigation_position_tolerance_m(
        self, resolved: _ResolvedTask
    ) -> float:
        del resolved
        return 0.045

    def _mobile_settle_position_tolerance_m(
        self, resolved: _ResolvedTask
    ) -> float | None:
        del resolved
        return None

    def _mobile_carry_stage_timeout_s(self, stage: str) -> float:
        return {
            "retract": 6.0,
            "turn": 5.0,
            "navigate": 6.0,
            "settle": 4.0,
            "place": 15.0,
        }[stage]

    def _mobile_place_target(
        self,
        oracle_target: Pose,
        state: dict[str, Any],
        *,
        waypoint_step_m: float = _MOBILE_PLACE_CARTESIAN_STEP_M,
    ) -> Pose:
        """Choose a reachable wrist attitude for the lateral tray target."""

        target_base = self._world_pose_to_root(
            Pose(oracle_target.xyz, state["tcp_world"].wxyz)
        )
        current_base = np.asarray(
            state["tcp_base"].xyz, dtype=np.float64
        )
        waypoint_base = np.asarray(
            target_base.xyz, dtype=np.float64
        )
        waypoint_delta = waypoint_base - current_base
        waypoint_distance = float(np.linalg.norm(waypoint_delta))
        if waypoint_distance > waypoint_step_m:
            waypoint_delta *= waypoint_step_m / waypoint_distance
        orientation_waypoint = current_base + waypoint_delta
        # Couple wrist yaw to the same incremental Cartesian waypoint. Asking
        # for the final lateral orientation while position is still centred
        # makes the redundant IK branch rotate q1/q5 in the opposite
        # direction and destabilizes the base.
        planar_bearing = math.atan2(
            orientation_waypoint[1], orientation_waypoint[0]
        )
        shoulder_yaw = max(
            -1.2,
            min(
                1.2,
                planar_bearing / _MOBILE_ARM_Q1_PLANAR_GAIN,
            ),
        )
        _, reference_rotation = self.arm_kinematics.forward(
            (
                shoulder_yaw,
                _MOBILE_COMPACT_ARM[1],
                _MOBILE_COMPACT_ARM[2],
                0.0,
                0.0,
                0.0,
            )
        )
        reference_base = Pose(
            target_base.xyz,
            _quaternion_from_rotation(reference_rotation),
        )
        target_world = self._root_pose_to_world(reference_base)
        return Pose(oracle_target.xyz, target_world.wxyz)

    def _transition_mobile_carry(
        self, stage: str, sim_time_s: float
    ) -> None:
        self._mobile_carry_stage = stage
        self._mobile_carry_stage_started_s = sim_time_s
        self._mobile_carry_stable_since_s = None

    def _mobile_carry_dwell(
        self, condition: bool, sim_time_s: float, duration_s: float
    ) -> bool:
        if not condition:
            self._mobile_carry_stable_since_s = None
            return False
        if self._mobile_carry_stable_since_s is None:
            self._mobile_carry_stable_since_s = sim_time_s
            return False
        return (
            sim_time_s - self._mobile_carry_stable_since_s >= duration_s
        )

    def _command_pregrasp_joint_target(
        self, current_tcp_base: Pose
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        target_position, target_rotation = self.arm_kinematics.forward(
            _PREGRASP_ARM
        )
        target_pose = Pose(
            tuple(float(value) for value in target_position),
            _quaternion_from_rotation(target_rotation),
        )
        raw_delta = np.asarray(target_pose.xyz) - np.asarray(
            current_tcp_base.xyz
        )
        distance = float(np.linalg.norm(raw_delta))
        if distance > 0.025:
            raw_delta *= 0.025 / distance
        rotation_delta = _rotation_vector_between(
            current_tcp_base.wxyz, target_pose.wxyz
        )
        angle = float(np.linalg.norm(rotation_delta))
        if angle > 0.12:
            rotation_delta *= 0.12 / angle
        planned = torch.tensor(
            [_PREGRASP_ARM],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self._arm_target = self._slew_arm_target(
            planned, carrying_object=False
        )
        self.robot.set_joint_position_target(
            self._arm_target, joint_ids=self.arm_joint_ids
        )
        return (
            tuple(float(value) for value in raw_delta),
            tuple(float(value) for value in rotation_delta),
            tuple(
                float(value)
                for value in (
                    np.asarray(current_tcp_base.xyz) + raw_delta
                )
            ),
        )

    def _command_mobile_compact_joint_target(
        self, current_tcp_base: Pose
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Slew to the stable whole-body carry posture in joint space."""

        planned = torch.tensor(
            [_MOBILE_COMPACT_ARM],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self._arm_target = self._slew_arm_target(
            planned, carrying_object=True
        )
        self.robot.set_joint_position_target(
            self._arm_target, joint_ids=self.arm_joint_ids
        )
        commanded_joints = tuple(
            float(value)
            for value in self._arm_target[0].detach().cpu().tolist()
        )
        commanded_position, commanded_rotation = (
            self.arm_kinematics.forward(commanded_joints)
        )
        commanded_pose = Pose(
            tuple(float(value) for value in commanded_position),
            _quaternion_from_rotation(commanded_rotation),
        )
        translation_delta = tuple(
            float(target - current)
            for target, current in zip(
                commanded_pose.xyz,
                current_tcp_base.xyz,
                strict=True,
            )
        )
        rotation_delta = _rotation_vector_between(
            current_tcp_base.wxyz,
            commanded_pose.wxyz,
        )
        return (
            translation_delta,
            tuple(float(value) for value in rotation_delta),
            commanded_pose.xyz,
        )

    def _command_mobile_retreat_joint_target(
        self,
        target_world: Pose,
        current_tcp_base: Pose,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Solve the post-release high pose once, then slew to it safely."""

        if self._mobile_retreat_arm_target is None:
            target_base = self._world_pose_to_root(target_world)
            seed = tuple(
                float(value)
                for value in self.robot.data.joint_pos[
                    0, self.arm_joint_ids
                ]
                .detach()
                .cpu()
                .tolist()
            )
            solution = self.arm_kinematics.solve(
                target_base.xyz,
                target_base.wxyz,
                seed=seed,
            )
            self._mobile_retreat_arm_target = solution.joint_positions
            self._arm_ik_seed = solution.joint_positions
            self._last_ik_error_m = solution.position_error_m
            self._last_ik_iterations = solution.iterations

        planned = torch.tensor(
            [self._mobile_retreat_arm_target],
            dtype=torch.float32,
            device=self.sim.device,
        )
        # Keep the loaded-arm slew envelope for the first retreat ticks even
        # though the gripper has opened; this bounded motion is safer for the
        # floating base than the unloaded 0.02 rad control step.
        self._arm_target = self._slew_arm_target(
            planned, carrying_object=True
        )
        self.robot.set_joint_position_target(
            self._arm_target, joint_ids=self.arm_joint_ids
        )
        commanded_joints = tuple(
            float(value)
            for value in self._arm_target[0].detach().cpu().tolist()
        )
        commanded_position, commanded_rotation = (
            self.arm_kinematics.forward(commanded_joints)
        )
        commanded_pose = Pose(
            tuple(float(value) for value in commanded_position),
            _quaternion_from_rotation(commanded_rotation),
        )
        translation_delta = tuple(
            float(target - current)
            for target, current in zip(
                commanded_pose.xyz,
                current_tcp_base.xyz,
                strict=True,
            )
        )
        rotation_delta = _rotation_vector_between(
            current_tcp_base.wxyz,
            commanded_pose.wxyz,
        )
        return (
            translation_delta,
            tuple(float(value) for value in rotation_delta),
            commanded_pose.xyz,
        )

    def _apply_tcp_command(
        self,
        target_world: Pose,
        current_tcp_base: Pose,
        *,
        max_translation_m: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        target_base = self._world_pose_to_root(target_world)
        return self._apply_tcp_target_base(
            target_base,
            current_tcp_base,
            max_translation_m=max_translation_m,
        )

    def _apply_tcp_target_base(
        self,
        target_base: Pose,
        current_tcp_base: Pose,
        *,
        max_translation_m: float = 0.025,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        raw_delta = np.asarray(target_base.xyz) - np.asarray(
            current_tcp_base.xyz
        )
        distance = float(np.linalg.norm(raw_delta))
        if distance > max_translation_m:
            raw_delta *= max_translation_m / distance
        next_position = np.asarray(current_tcp_base.xyz) + raw_delta
        rotation_delta = _rotation_vector_between(
            current_tcp_base.wxyz, target_base.wxyz
        )
        angle = float(np.linalg.norm(rotation_delta))
        if angle > 0.12:
            rotation_delta *= 0.12 / angle
        next_orientation = _apply_rotation_vector(
            current_tcp_base.wxyz, rotation_delta
        )
        try:
            solution = self.arm_kinematics.solve(
                next_position,
                next_orientation,
                seed=self._arm_ik_seed,
            )
        except IKConvergenceError:
            # Retry the final oracle orientation at the bounded position.  If
            # it is unreachable the error must remain visible to the episode
            # abort path rather than silently fabricating a command.
            solution = self.arm_kinematics.solve(
                next_position,
                target_base.wxyz,
                seed=self._arm_ik_seed,
            )
        self._arm_ik_seed = solution.joint_positions
        self._last_ik_error_m = solution.position_error_m
        self._last_ik_iterations = solution.iterations
        planned = torch.tensor(
            [solution.joint_positions],
            dtype=torch.float32,
            device=self.sim.device,
        )
        self._arm_target = self._slew_arm_target(
            planned,
            carrying_object=self._held_instance_id is not None,
        )
        limits = self.robot.data.soft_joint_pos_limits[
            :, self.arm_joint_ids
        ]
        self._arm_target = torch.maximum(
            torch.minimum(self._arm_target, limits[..., 1]),
            limits[..., 0],
        )
        self.robot.set_joint_position_target(
            self._arm_target, joint_ids=self.arm_joint_ids
        )
        return (
            tuple(float(value) for value in raw_delta),
            tuple(float(value) for value in rotation_delta),
            tuple(float(value) for value in next_position),
        )

    def _m0_live_state28(self, state: dict[str, Any]) -> tuple[float, ...]:
        """Build only the 28 policy-visible proprioceptive values."""

        from conveyor_bench.m0_online import build_live_state28

        gripper_position = float(
            self.robot.data.joint_pos[0, self.gripper_joint_ids]
            .mean()
            .item()
        )
        return build_live_state28(
            _tensor_tuple(self.robot.data.root_lin_vel_b[0]),
            _tensor_tuple(self.robot.data.root_ang_vel_b[0]),
            _tensor_tuple(self.robot.data.projected_gravity_b[0]),
            _tensor_tuple(
                self.robot.data.joint_pos[0, self.arm_joint_ids]
            ),
            _tensor_tuple(
                self.robot.data.joint_vel[0, self.arm_joint_ids]
            ),
            state["tcp_base"].xyz,
            state["tcp_base"].wxyz,
            min(1.0, max(0.0, gripper_position / 0.044)),
        )

    @staticmethod
    def _camera_rgb_numpy(image: Any) -> np.ndarray:
        """Copy one Isaac RGB/RGBA tensor into a bounded uint8 RGB array."""

        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        array = np.asarray(image)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"invalid online policy camera shape: {array.shape}")
        rgb = array[..., :3]
        if np.issubdtype(rgb.dtype, np.floating):
            maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
            if maximum <= 1.0:
                rgb = rgb * 255.0
        return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))

    def _apply_m0_mobile_action(
        self,
        action: Sequence[float],
        state: dict[str, Any],
        *,
        guard_pregrasp_workspace: bool = False,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        bool,
        dict[str, Any] | None,
    ]:
        """Project one physical M0 action through existing robot guards."""

        if len(action) != 10 or any(
            not math.isfinite(float(value)) for value in action
        ):
            raise ValueError("online M0 action must be finite canonical10")
        values = tuple(float(value) for value in action)
        base_command = self._guard_locomotion_command(values[:3])
        current_tcp: Pose = state["tcp_base"]
        translation = np.asarray(values[3:6], dtype=np.float64)
        rotation = np.asarray(values[6:9], dtype=np.float64)
        proposed_position = tuple(
            float(current + delta)
            for current, delta in zip(
                current_tcp.xyz, translation, strict=True
            )
        )
        workspace_guard = None
        guarded_position = proposed_position
        if guard_pregrasp_workspace:
            guarded_position, clipped_axes = guard_pregrasp_tcp_target(
                proposed_position
            )
            correction = tuple(
                guarded - proposed
                for guarded, proposed in zip(
                    guarded_position, proposed_position, strict=True
                )
            )
            workspace_guard = {
                "enabled": True,
                "active": bool(clipped_axes),
                "phase": "pregrasp",
                "frame": "robot_base",
                "bounds": {
                    axis: {"min": limits[0], "max": limits[1]}
                    for axis, limits in zip(
                        "xyz", PREGRASP_WORKSPACE_LIMITS_BASE, strict=True
                    )
                },
                "current_tcp_before_xyz": list(current_tcp.xyz),
                "policy_proposed_tcp_target_xyz": list(proposed_position),
                "guarded_tcp_target_xyz": list(guarded_position),
                "correction_xyz": list(correction),
                "correction_norm_m": float(np.linalg.norm(correction)),
                "clipped_axes": list(clipped_axes),
            }
        guarded_delta = np.asarray(guarded_position) - np.asarray(
            current_tcp.xyz
        )
        if (
            float(np.linalg.norm(guarded_delta)) < 1.0e-10
            and float(np.linalg.norm(rotation)) < 1.0e-10
        ):
            self._hold_arm_target()
            ee_delta = (0.0, 0.0, 0.0)
            rotvec = (0.0, 0.0, 0.0)
            target_position = current_tcp.xyz
        else:
            target = Pose(
                guarded_position,
                _apply_rotation_vector(current_tcp.wxyz, rotation),
            )
            ee_delta, rotvec, target_position = (
                self._apply_tcp_target_base(
                    target,
                    current_tcp,
                    max_translation_m=0.025,
                )
            )
        if workspace_guard is not None:
            workspace_guard["applied_tcp_target_xyz"] = list(
                target_position
            )
        return (
            base_command,
            ee_delta,
            rotvec,
            target_position,
            values[9] >= 0.5,
            workspace_guard,
        )

    def _slew_arm_target(
        self,
        planned: torch.Tensor,
        *,
        carrying_object: bool,
    ) -> torch.Tensor:
        """Rate-limit every arm target write on the floating platform."""

        current = self.robot.data.joint_pos[:, self.arm_joint_ids]
        commanded = self._arm_target
        if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY:
            per_joint = (
                (0.008, 0.010, 0.010, 0.010, 0.008, 0.010)
                if carrying_object
                else (0.020,) * len(ARM_JOINT_NAMES)
            )
            delta_limit = torch.tensor(
                [per_joint],
                dtype=torch.float32,
                device=self.sim.device,
            )
        else:
            delta_limit = torch.full_like(commanded, 0.08)
        # Project the stored command onto the segment between the measured
        # joint and the new IK solution.  Without this anti-windup step, a
        # target accumulated during descent can remain on the wrong side of
        # the measured joint for several control ticks after lift begins,
        # briefly driving the gripper back into the belt.
        lower = torch.minimum(current, planned)
        upper = torch.maximum(current, planned)
        commanded = torch.maximum(torch.minimum(commanded, upper), lower)
        target = commanded + torch.maximum(
            torch.minimum(planned - commanded, delta_limit),
            -delta_limit,
        )
        # A command may ramp ahead of a gravity-loaded joint, but never by
        # enough to demand more than the rated actuator effort.
        tracking_window = torch.tensor(
            [[0.12, 0.12, 0.12, 0.10, 0.10, 0.10]],
            dtype=torch.float32,
            device=self.sim.device,
        )
        if self.options.robot_mode is RobotMode.WHOLE_BODY_POLICY:
            target = torch.maximum(
                torch.minimum(target, current + tracking_window),
                current - tracking_window,
            )
        limits = self.robot.data.soft_joint_pos_limits[
            :, self.arm_joint_ids
        ]
        return torch.maximum(
            torch.minimum(target, limits[..., 1]), limits[..., 0]
        )

    def _hold_arm_target(self) -> None:
        self.robot.set_joint_position_target(
            self._arm_target, joint_ids=self.arm_joint_ids
        )

    def _apply_gripper(self, open_gripper: bool) -> None:
        opening = 0.044 if open_gripper else 0.0
        self.robot.set_joint_position_target(
            torch.full(
                (1, len(self.gripper_joint_ids)),
                opening,
                dtype=torch.float32,
                device=self.sim.device,
            ),
            joint_ids=self.gripper_joint_ids,
        )
        self._gripper_open = open_gripper

    def _apply_base_command(
        self,
        command: tuple[float, float, float],
    ) -> tuple[float, ...]:
        if self.options.robot_mode is RobotMode.FIXED_BASE:
            self.robot.set_joint_position_target(
                self.robot.data.default_joint_pos[:, self.leg_joint_ids],
                joint_ids=self.leg_joint_ids,
            )
            return (0.0,) * 12
        assert self.locomotion_policy is not None
        command_tensor = torch.tensor(
            [command], dtype=torch.float32, device=self.sim.device
        )
        state = {
            "root_lin_vel_b": self.robot.data.root_lin_vel_b,
            "root_ang_vel_b": self.robot.data.root_ang_vel_b,
            "projected_gravity_b": self.robot.data.projected_gravity_b,
            "joint_pos": self.robot.data.joint_pos[
                :, self.locomotion_state_joint_ids
            ],
            "default_joint_pos": self.robot.data.default_joint_pos[
                :, self.locomotion_state_joint_ids
            ],
            "joint_vel": self.robot.data.joint_vel[
                :, self.locomotion_state_joint_ids
            ],
        }
        observation = build_observation(
            state,
            command_tensor,
            self._last_policy_action,
            self._arm_target,
            torch.zeros(
                (1, 1), dtype=torch.float32, device=self.sim.device
            ),
        )
        warmup_scale = min(
            1.0,
            (self._locomotion_policy_step_count + 1)
            / _LOCOMOTION_WARMUP_POLICY_STEPS,
        )
        applied_action = infer(
            self.locomotion_policy,
            observation,
            warmup_scale=warmup_scale,
        )
        self._locomotion_policy_step_count += 1
        target = leg_target(applied_action)
        self.robot.set_joint_position_target(
            target, joint_ids=self.locomotion_action_joint_ids
        )
        self._last_policy_action = applied_action
        return tuple(
            float(value)
            for value in applied_action[0].detach().cpu().tolist()
        )

    def _read_state(self, resolved: _ResolvedTask) -> dict[str, Any]:
        root_pose = Pose(
            _tensor_tuple(self.robot.data.root_pos_w[0]),
            _tensor_tuple(self.robot.data.root_quat_w[0]),
        )
        root_twist = Twist(
            _tensor_tuple(self.robot.data.root_lin_vel_w[0]),
            _tensor_tuple(self.robot.data.root_ang_vel_w[0]),
        )
        link6_pose = self.robot.data.body_pose_w[
            0, self.link6_body_ids[0]
        ]
        tcp_position_world = (
            link6_pose[:3]
            + quat_apply(
                link6_pose[3:7].unsqueeze(0), self._tcp_offset
            )[0]
        )
        tcp_world = Pose(
            _tensor_tuple(tcp_position_world),
            _tensor_tuple(link6_pose[3:7]),
        )
        tcp_base = self._world_pose_to_root(tcp_world)
        left_contacts = self._contact_ids(
            self.left_contact_sensor, resolved.assets
        )
        right_contacts = self._contact_ids(
            self.right_contact_sensor, resolved.assets
        )
        bilateral = set(left_contacts).intersection(right_contacts)

        distances = {
            asset.object_id: float(
                torch.linalg.vector_norm(
                    self.objects[asset.object_id].data.root_pos_w[0]
                    - tcp_position_world
                ).item()
            )
            for asset in resolved.assets
        }
        if not self._gripper_open:
            candidates = [
                object_id
                for object_id in bilateral
                if distances.get(object_id, math.inf) < 0.10
            ]
            if candidates:
                self._held_instance_id = min(
                    candidates, key=lambda item: distances[item]
                )
            elif (
                self._held_instance_id is not None
                and distances.get(self._held_instance_id, math.inf) > 0.12
            ):
                self._held_instance_id = None
        else:
            self._held_instance_id = None
        if self._held_instance_id == resolved.target_asset.object_id:
            self._ever_held_target = True

        object_states: list[ObjectState] = []
        object_active: dict[str, bool] = {}
        for asset in resolved.assets:
            rigid_object = self.objects[asset.object_id]
            position = _tensor_tuple(rigid_object.data.root_pos_w[0])
            active = position[0] < 2.0
            object_active[asset.object_id] = active
            object_states.append(
                ObjectState(
                    instance_id=asset.object_id,
                    pose_world=Pose(
                        position,
                        _tensor_tuple(rigid_object.data.root_quat_w[0]),
                    ),
                    twist_world=Twist(
                        _tensor_tuple(rigid_object.data.root_lin_vel_w[0]),
                        _tensor_tuple(rigid_object.data.root_ang_vel_w[0]),
                    ),
                    active=active,
                    in_gripper=(
                        self._held_instance_id == asset.object_id
                    ),
                    crossed_exit=self._object_crossed_task_exit(
                        resolved,
                        asset,
                        position,
                        active=active,
                    ),
                )
            )
        root_up = quat_apply(
            self.robot.data.root_quat_w,
            torch.tensor(
                [[0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.sim.device,
            ),
        )[0]
        root_tilt_rad = math.acos(
            max(-1.0, min(1.0, float(root_up[2].item())))
        )
        return {
            "root_pose": root_pose,
            "root_twist": root_twist,
            "tcp_world": tcp_world,
            "tcp_base": tcp_base,
            "objects": tuple(object_states),
            "left_contacts": left_contacts,
            "right_contacts": right_contacts,
            "robot_fallen": (
                root_pose.xyz[2] < 0.20 or float(root_up[2].item()) < 0.55
            ),
            "root_tilt_rad": root_tilt_rad,
            "forbidden_collision": _tcp_intrudes_belt(tcp_world.xyz),
            "object_active": object_active,
        }

    def _contact_ids(
        self,
        sensor: Any,
        active_assets: Sequence[ObjectAsset],
    ) -> tuple[str, ...]:
        matrix = sensor.data.force_matrix_w
        if matrix is None:
            return ()
        magnitudes = torch.linalg.vector_norm(matrix[0, 0], dim=-1)
        threshold = float(sensor.cfg.force_threshold)
        # Filter columns follow OBJECT_PRIM_BASENAMES/OBJECT_ASSETS order.
        active_ids = {asset.object_id for asset in active_assets}
        return tuple(
            OBJECT_ASSETS[index].object_id
            for index, magnitude in enumerate(magnitudes)
            if index < len(OBJECT_ASSETS)
            and OBJECT_ASSETS[index].object_id in active_ids
            and float(magnitude.item()) >= threshold
        )

    def _oracle_observation(
        self,
        resolved: _ResolvedTask,
        state: dict[str, Any],
        sim_time_s: float,
    ) -> OracleObservation:
        target = next(
            item
            for item in state["objects"]
            if item.instance_id == resolved.target_asset.object_id
        )
        speed = math.sqrt(
            sum(value * value for value in target.twist_world.linear_xyz)
        )
        angular_speed = math.sqrt(
            sum(value * value for value in target.twist_world.angular_xyz)
        )
        target_in_goal = _goal_zone(resolved.target_zone).contains(
            target.pose_world.xyz
        )
        return OracleObservation(
            sim_time_s=sim_time_s,
            target_position_world=target.pose_world.xyz,
            # The privileged intercept oracle uses the known transport motion,
            # not lateral/vertical collision impulses from a first finger
            # touch. Rich measured twist remains in the recorded object state.
            target_velocity_world=tuple(
                float(axis * self.options.belt_speed_mps)
                for axis in TRANSPORT_DIRECTION_WORLD
            ),
            tcp_position_world=state["tcp_world"].xyz,
            left_contact_object_ids=state["left_contacts"],
            right_contact_object_ids=state["right_contacts"],
            target_held=target.in_gripper,
            target_lifted=(
                target.pose_world.xyz[2] - BELT_TOP_Z_M >= 0.04
            ),
            target_in_goal=target_in_goal,
            target_released=(
                self._ever_held_target
                and not target.in_gripper
                and self._gripper_open
            ),
            target_settled=(
                speed
                <= self.benchmark.evaluation.settled_linear_speed_mps
                and angular_speed
                <= self.benchmark.evaluation.settled_angular_speed_radps
            ),
            wrong_object_grasped=(
                self._held_instance_id is not None
                and self._held_instance_id
                != resolved.target_asset.object_id
            ),
            robot_fallen=state["robot_fallen"],
            forbidden_collision=state["forbidden_collision"],
            target_crossed_exit=target.crossed_exit,
        )

    def _make_sample(
        self,
        *,
        resolved: _ResolvedTask,
        state: dict[str, Any],
        action: CanonicalAction,
        phase: str,
        selected_object_id: str | None,
        sim_time_s: float,
        model_tick: int,
        camera_frames: tuple[Any, ...],
        base_command: tuple[float, float, float],
        policy_action: tuple[float, ...],
        oracle_target_base: tuple[float, float, float] | None,
        m0_step_metadata: dict[str, Any] | None,
    ) -> StepSample:
        future_labels: list[FutureObjectState] = []
        for obj in state["objects"]:
            for horizon in self.benchmark.future_horizons_steps:
                if not obj.active:
                    future_labels.append(
                        FutureObjectState(
                            instance_id=obj.instance_id,
                            horizon_steps=horizon,
                            valid=False,
                            pose_world=None,
                            twist_world=None,
                            invalid_reason="inactive_not_spawned",
                        )
                    )
                    continue
                if horizon == 0:
                    future_labels.append(
                        FutureObjectState(
                            instance_id=obj.instance_id,
                            horizon_steps=0,
                            valid=True,
                            pose_world=obj.pose_world,
                            twist_world=obj.twist_world,
                        )
                    )
                else:
                    # Positive horizons are deliberately pending here.  The
                    # complete episode buffer replaces them with states that
                    # actually occurred at model_tick+h; they are never
                    # extrapolated from current velocity.
                    future_labels.append(
                        FutureObjectState(
                            instance_id=obj.instance_id,
                            horizon_steps=horizon,
                            valid=False,
                            pose_world=None,
                            twist_world=None,
                            invalid_reason="future_pending",
                        )
                    )
        return StepSample(
            sim_step=self._physics_step_count,
            sim_time_s=sim_time_s,
            model_tick=model_tick,
            env_id=0,
            robot_root_world=state["root_pose"],
            robot_twist_world=state["root_twist"],
            tcp_base=state["tcp_base"],
            joints=JointState(
                names=tuple(self.robot.joint_names),
                positions=_tensor_tuple(self.robot.data.joint_pos[0]),
                velocities=_tensor_tuple(self.robot.data.joint_vel[0]),
            ),
            action=action,
            objects=state["objects"],
            left_contact_object_ids=state["left_contacts"],
            right_contact_object_ids=state["right_contacts"],
            camera_frames=tuple(camera_frames),
            future_object_states=tuple(future_labels),
            phase=phase,
            selected_object_id=selected_object_id,
            robot_fallen=state["robot_fallen"],
            forbidden_collision=state["forbidden_collision"],
            belt_measured_speed_mps=self._surface_speed(),
            metadata={
                "tcp_world": {
                    "xyz": state["tcp_world"].xyz,
                    "wxyz": state["tcp_world"].wxyz,
                },
                "base_command_body": base_command,
                "locomotion_policy_action": policy_action,
                "oracle_next_tcp_target_base_xyz": oracle_target_base,
                "held_instance_id": self._held_instance_id,
                "root_tilt_rad": state["root_tilt_rad"],
                "mobile_carry_stage": self._mobile_carry_stage,
                "ik_position_error_m": self._last_ik_error_m,
                "ik_iterations": self._last_ik_iterations,
                "m0_online_action": m0_step_metadata,
                **self._extra_step_metadata(resolved, state),
            },
        )

    def _extra_step_metadata(
        self,
        resolved: _ResolvedTask,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        del resolved, state
        return {}

    def _world_pose_to_root(self, pose_world: Pose) -> Pose:
        root_position = self.robot.data.root_pos_w[0]
        root_quaternion = self.robot.data.root_quat_w[0]
        inverse = quat_inv(root_quaternion.unsqueeze(0))[0]
        position = quat_apply(
            inverse.unsqueeze(0),
            torch.tensor(
                [pose_world.xyz],
                dtype=torch.float32,
                device=self.sim.device,
            )
            - root_position.unsqueeze(0),
        )[0]
        orientation = quat_mul(
            inverse.unsqueeze(0),
            torch.tensor(
                [pose_world.wxyz],
                dtype=torch.float32,
                device=self.sim.device,
            ),
        )[0]
        return Pose(_tensor_tuple(position), _tensor_tuple(orientation))

    def _root_pose_to_world(self, pose_root: Pose) -> Pose:
        root_position = self.robot.data.root_pos_w[0]
        root_quaternion = self.robot.data.root_quat_w[0]
        position = root_position + quat_apply(
            root_quaternion.unsqueeze(0),
            torch.tensor(
                [pose_root.xyz],
                dtype=torch.float32,
                device=self.sim.device,
            ),
        )[0]
        orientation = quat_mul(
            root_quaternion.unsqueeze(0),
            torch.tensor(
                [pose_root.wxyz],
                dtype=torch.float32,
                device=self.sim.device,
            ),
        )[0]
        return Pose(_tensor_tuple(position), _tensor_tuple(orientation))

    def _surface_speed(self) -> float:
        velocity = self.surface_velocity_api.GetSurfaceVelocityAttr().Get()
        return float(
            sum(
                float(velocity[index]) * direction
                for index, direction in enumerate(
                    TRANSPORT_DIRECTION_WORLD
                )
            )
        )

    def _hold_all_joints(self) -> None:
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos
        )

    def _step_physics(self) -> None:
        self._physics_step_count += 1
        render = (
            self.options.enable_cameras
            and self._physics_step_count % self.camera_physics_stride == 0
        )
        self.sim.step(render=render)
        self.scene.update(self.physics_dt)


def run_collection_v1(options: RuntimeOptionsV1) -> dict[str, Any]:
    runtime = ConveyorRuntimeV1(options)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _goal_zone(asset: ReceptacleAsset) -> GoalZone:
    return GoalZone(
        zone_id=asset.zone_id,
        min_xyz=tuple(
            center - half
            for center, half in zip(
                asset.center_xyz_m,
                asset.goal_half_extents_xyz_m,
                strict=True,
            )
        ),
        max_xyz=tuple(
            center + half
            for center, half in zip(
                asset.center_xyz_m,
                asset.goal_half_extents_xyz_m,
                strict=True,
            )
        ),
    )


def _guard_locomotion_command(
    command: Sequence[float],
) -> tuple[float, float, float]:
    vx, vy, wz = (float(value) for value in command)
    if abs(vy) > 1.0e-9:
        raise ValueError("V1 locomotion guardrail forbids lateral command")
    if 0.0 < abs(vx) < 0.16:
        vx = 0.0
    return (
        min(0.30, max(0.0, vx)),
        0.0,
        min(0.35, max(-0.35, wz)),
    )


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _tcp_intrudes_belt(position_world: Sequence[float]) -> bool:
    """Return whether the TCP penetrates the conveyor's occupied prism.

    A global height threshold incorrectly labelled a low TCP over the robot
    side or a sorting tray as a conveyor collision.  Keep a small XY safety
    margin around the physical belt and rails, then apply the penetration
    threshold only inside that footprint.
    """

    x, y, z = (float(value) for value in position_world)
    safety_margin_m = 0.04
    inside_x = (
        abs(x - BELT_CENTER_X_M)
        <= BELT_WIDTH_M * 0.5 + safety_margin_m
    )
    inside_y = (
        abs(y - BELT_CENTER_Y_M)
        <= BELT_LENGTH_M * 0.5 + safety_margin_m
    )
    return inside_x and inside_y and z < BELT_TOP_Z_M - 0.025


def _crossed_exit(
    position: Sequence[float],
    plane_point: Sequence[float],
    direction: Sequence[float],
) -> bool:
    return (
        sum(
            (float(value) - float(point)) * float(axis)
            for value, point, axis in zip(
                position, plane_point, direction, strict=True
            )
        )
        >= 0.0
    )


def _tensor_tuple(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(
        float(value) for value in tensor.detach().cpu().tolist()
    )


def _quaternion_from_rotation(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(float(value / norm) for value in quaternion)


def _rotation_vector_between(
    current_wxyz: Sequence[float],
    target_wxyz: Sequence[float],
) -> np.ndarray:
    current = np.asarray(current_wxyz, dtype=np.float64)
    target = np.asarray(target_wxyz, dtype=np.float64)
    current /= np.linalg.norm(current)
    target /= np.linalg.norm(target)
    inverse = current * np.asarray((1.0, -1.0, -1.0, -1.0))
    delta = _quaternion_multiply(target, inverse)
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(delta[0]))
    return delta[1:] / vector_norm * angle


def _apply_rotation_vector(
    current_wxyz: Sequence[float], rotation_vector: np.ndarray
) -> tuple[float, float, float, float]:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-10:
        return tuple(float(value) for value in current_wxyz)
    axis = rotation_vector / angle
    half = angle * 0.5
    delta = np.concatenate(
        ([math.cos(half)], axis * math.sin(half))
    )
    result = _quaternion_multiply(
        delta, np.asarray(current_wxyz, dtype=np.float64)
    )
    result /= np.linalg.norm(result)
    return tuple(float(value) for value in result)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _camera_contract_v1() -> dict[str, Any]:
    horizontal_aperture = 20.955

    def spec(
        camera: CameraSpec,
        focal_length: float,
        parent: str,
        xyz: tuple[float, float, float],
        wxyz: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        focal_x = (
            focal_length / horizontal_aperture * camera.width
        )
        return {
            "resolution": [camera.width, camera.height],
            "fps": 25,
            "role": camera.role,
            "model": "pinhole",
            "intrinsics": [
                [focal_x, 0.0, camera.width / 2.0],
                [0.0, focal_x, camera.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "mount": {
                "parent": parent,
                "xyz_m": list(xyz),
                "wxyz": list(wxyz),
                "orientation_convention": "world",
            },
        }

    return {
        "head_rgb": spec(
            _CAMERA_SPECS[0],
            16.0,
            "base",
            HEAD_CAMERA_OFFSET_XYZ,
            HEAD_CAMERA_OFFSET_WXYZ,
        ),
        "wrist_rgb": spec(
            _CAMERA_SPECS[1],
            18.0,
            "arm_link6",
            WRIST_CAMERA_OFFSET_XYZ,
            WRIST_CAMERA_OFFSET_WXYZ,
        ),
        "overview_rgb": spec(
            _CAMERA_SPECS[2],
            18.0,
            "environment_origin",
            OVERVIEW_CAMERA_OFFSET_XYZ,
            OVERVIEW_CAMERA_OFFSET_WXYZ,
        ),
    }
