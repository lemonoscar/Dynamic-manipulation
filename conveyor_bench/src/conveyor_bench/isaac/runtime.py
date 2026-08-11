"""Single-owner Isaac Sim loop for V0 data collection."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import omni.usd
import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from conveyor_bench.config import BenchmarkConfig
from conveyor_bench.metrics import evaluate_episode
from conveyor_bench.oracle import (
    DynamicPickOracle,
    OracleConfig,
    OracleObservation,
)
from conveyor_bench.protocol import (
    EpisodeManifest,
    Event,
    EventKind,
    FailureReason,
    StepSample,
    TaskManifest,
    TaskType,
    TimingTrace,
    make_run_id,
)
from conveyor_bench.recorder import EpisodeRecorder
from conveyor_bench.video import EpisodeVideoWriter

from .asset_config import (
    ARM_JOINT_NAMES,
    GO2_X5_URDF,
    GRIPPER_JOINT_NAMES,
    LEG_JOINT_NAMES,
    TCP_OFFSET_X_M,
)
from .arm_kinematics import CalibratedArmKinematics
from .scene import (
    BELT_TOP_Z_M,
    EXIT_PLANE_POINT_WORLD,
    LAYOUT_ID,
    OBJECT_CENTER_Z_M,
    OBJECT_INTERCEPT_X_M,
    OBJECT_INTERCEPT_Y_M,
    OBJECT_SPAWN_Y_M,
    TRANSPORT_DIRECTION_WORLD,
    HEAD_CAMERA_OFFSET_WXYZ,
    HEAD_CAMERA_OFFSET_XYZ,
    OVERVIEW_CAMERA_OFFSET_WXYZ,
    OVERVIEW_CAMERA_OFFSET_XYZ,
    WRIST_CAMERA_OFFSET_WXYZ,
    WRIST_CAMERA_OFFSET_XYZ,
    ConveyorSceneCfg,
    apply_surface_velocity,
)
from .scene_v1 import apply_pct_gripper_collision_patch


@dataclass(frozen=True)
class RuntimeOptions:
    output_root: Path
    task_type: TaskType
    episodes: int = 1
    seed: int = 0
    belt_speed_mps: float = 0.08
    max_duration_s: float = 15.0
    device: str = "cpu"
    save_video: bool = True
    enable_cameras: bool = True

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive")
        if self.task_type is TaskType.C1_DYNAMIC_PICK and self.belt_speed_mps <= 0:
            raise ValueError("C1 requires a positive belt speed")
        if self.task_type is TaskType.C1_DYNAMIC_PICK and self.device != "cpu":
            raise ValueError(
                "C1 requires --device cpu: Isaac Sim 5.1 GPU PhysX does not "
                "preserve normal contacts for PhysxSurfaceVelocityAPI"
            )
        if self.save_video and not self.enable_cameras:
            raise ValueError("save_video requires enable_cameras")


class ConveyorRuntime:
    def __init__(self, options: RuntimeOptions):
        self._is_closed = False
        self.options = options
        self.benchmark = BenchmarkConfig.v0()
        self.physics_dt = 1.0 / self.benchmark.physics_hz
        self.control_dt = 1.0 / self.benchmark.control_hz
        self.decimation = self.benchmark.physics_hz // self.benchmark.control_hz
        self.camera_stride = self.benchmark.control_hz // self.benchmark.camera_hz
        self.camera_render_stride = (
            self.benchmark.physics_hz // self.benchmark.camera_hz
        )
        self._physics_step_count = 0

        physx_cfg = sim_utils.PhysxCfg(
            enable_enhanced_determinism=True,
            bounce_threshold_velocity=0.2,
            friction_correlation_distance=0.00625,
        )
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.physics_dt,
            # Direct workflows decide explicitly which physics steps render.
            # A larger rendering_dt would make one sim.step() advance multiple
            # physics substeps and break the 200/50/25 Hz contract.
            render_interval=1,
            device=options.device,
            physx=physx_cfg,
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        try:
            self.sim.set_camera_view(
                eye=(2.4, -2.0, 1.8),
                target=(0.85, 0.0, 0.65),
            )

            scene_cfg = ConveyorSceneCfg(
                num_envs=1,
                env_spacing=2.5,
                replicate_physics=True,
                clone_in_fabric=False,
                lazy_sensor_update=True,
            )
            if not options.enable_cameras:
                scene_cfg.head_camera = None
                scene_cfg.wrist_camera = None
                scene_cfg.overview_camera = None
            self.scene = InteractiveScene(scene_cfg)
            stage = omni.usd.get_context().get_stage()
            self.gripper_collision_contract = (
                apply_pct_gripper_collision_patch(
                    stage,
                    "/World/envs/env_0/Robot",
                )
            )
            initial_speed = self._task_belt_speed()
            self.surface_velocity_api = apply_surface_velocity(
                stage,
                "/World/envs/env_0/Conveyor",
                initial_speed,
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
        if getattr(self, "_is_closed", False):
            return
        self._is_closed = True
        simulation = getattr(self, "sim", None)
        # Release tensor views and sensors before clearing the singleton.  This
        # mirrors Isaac Lab's own simulation-context teardown contract.
        for name in (
            "head_camera",
            "wrist_camera",
            "overview_camera",
            "contact_sensor",
            "target_object",
            "robot",
            "surface_velocity_api",
            "gripper_collision_contract",
            "_tcp_offset",
        ):
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, "scene"):
            del self.scene
        if simulation is None:
            return

        cleanup_error: Exception | None = None
        try:
            simulation.clear_all_callbacks()
        except Exception as error:
            cleanup_error = error
        try:
            simulation.clear_instance()
        except Exception as error:
            if cleanup_error is None:
                cleanup_error = error
        del self.sim
        if cleanup_error is not None:
            raise cleanup_error

    def run(self) -> dict:
        run_id = make_run_id()
        episode_reports: list[dict] = []
        for episode_index in range(self.options.episodes):
            episode_seed = self.options.seed + episode_index
            report = self._run_episode(
                run_id=run_id,
                episode_index=episode_index,
                episode_seed=episode_seed,
            )
            episode_reports.append(report)

        summary = {
            "run_id": run_id,
            "protocol_version": self.benchmark.protocol_version,
            "task_type": self.options.task_type.value,
            "requested_episodes": self.options.episodes,
            "successful_episodes": sum(report["success"] for report in episode_reports),
            "episodes": episode_reports,
        }
        self.options.output_root.mkdir(parents=True, exist_ok=True)
        summary_path = self.options.output_root / f"{run_id}-summary.json"
        temporary = summary_path.with_name(f".{summary_path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(summary_path)
        summary["summary_path"] = str(summary_path)
        return summary

    def probe_arm_candidates(self) -> list[dict]:
        """Return TCP poses for a small set of safe joint-space candidates."""

        candidates = (
            ("default", (0.0, 1.6, 1.2, 0.0, 0.0, 0.0)),
            ("raised_a", (0.0, 1.2, 1.2, 0.0, 0.0, 0.0)),
            ("raised_b", (0.0, 1.0, 1.5, 0.0, 0.0, 0.0)),
            ("forward_a", (0.0, 2.0, 1.0, 0.0, 0.0, 0.0)),
            ("raised_c", (0.0, 1.2, 1.8, 0.0, 0.0, 0.0)),
            ("wrist_down", (0.0, 1.6, 1.2, -0.5, 0.0, 0.0)),
        )
        self._write_object_state(x=1.40, y=0.0)
        reports: list[dict] = []
        for name, values in candidates:
            joint_positions = self.robot.data.joint_pos.clone()
            joint_velocities = torch.zeros_like(joint_positions)
            target = torch.tensor(
                [values],
                device=self.sim.device,
                dtype=torch.float32,
            )
            joint_positions[:, self.arm_joint_ids] = target
            self.robot.write_joint_state_to_sim(joint_positions, joint_velocities)
            self.robot.set_joint_position_target(target, joint_ids=self.arm_joint_ids)
            for _ in range(4):
                self.scene.write_data_to_sim()
                self._step_simulation()
            link6_pose_w = self.robot.data.body_pose_w[0, self.link6_body_ids[0]]
            reports.append(
                {
                    "name": name,
                    "arm_joint_positions": list(values),
                    "tcp_xyz": _vec3(self._tcp_position()),
                    "link6_wxyz": [
                        float(value)
                        for value in link6_pose_w[3:7].detach().cpu().tolist()
                    ],
                }
            )
        return reports

    def probe_object_dynamics(
        self,
        *,
        spawn_y: float = OBJECT_INTERCEPT_Y_M,
        lane_x: float = OBJECT_INTERCEPT_X_M,
        surface_speed_mps: float = 0.0,
        physics_steps: int = 12,
    ) -> list[dict]:
        """Trace a freshly spawned target without running the oracle."""

        if physics_steps <= 0:
            raise ValueError("physics_steps must be positive")
        configured_speed = self._task_belt_speed()
        if abs(surface_speed_mps - configured_speed) > 1.0e-9:
            raise ValueError(
                "surface_speed_mps must match RuntimeOptions; surface velocity "
                "is configured before the first simulation reset"
            )
        self._reset_episode_state(
            task_type=TaskType.C0_STATIC_PICK,
            lane_x=lane_x,
        )
        if spawn_y != OBJECT_INTERCEPT_Y_M:
            self._write_object_state(x=lane_x, y=spawn_y)
            self.scene.reset()

        conveyor = self.scene["conveyor"]
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=False,
        )
        usd_bounds = {}
        for name, path in {
            "object": "/World/envs/env_0/TargetObject",
            "conveyor": "/World/envs/env_0/Conveyor",
        }.items():
            prim = stage.GetPrimAtPath(path)
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            usd_bounds[name] = {
                "min_xyz": list(aligned.GetMin()),
                "max_xyz": list(aligned.GetMax()),
                "applied_schemas": list(prim.GetAppliedSchemas()),
                "children": [
                    {
                        "path": str(child.GetPath()),
                        "type": child.GetTypeName(),
                        "applied_schemas": list(child.GetAppliedSchemas()),
                    }
                    for child in Usd.PrimRange(prim)
                ],
            }
        reports: list[dict] = []
        for physics_step in range(physics_steps + 1):
            object_position = self.target_object.data.root_pos_w[0, :3]
            body_positions = self.robot.data.body_pos_w[0]
            distances = torch.linalg.vector_norm(
                body_positions - object_position.unsqueeze(0),
                dim=-1,
            )
            nearest = torch.argsort(distances)[:5].detach().cpu().tolist()
            reports.append(
                {
                    "physics_step": physics_step,
                    "time_s": physics_step * self.physics_dt,
                    "object_xyz": _vec3(object_position),
                    "object_velocity": _vec3(
                        self.target_object.data.root_lin_vel_w[0, :3]
                    ),
                    "conveyor_xyz": _vec3(conveyor.data.root_pos_w[0, :3]),
                    "conveyor_velocity": _vec3(
                        conveyor.data.root_lin_vel_w[0, :3]
                    ),
                    "usd_bounds": usd_bounds if physics_step == 0 else None,
                    "nearest_robot_bodies": (
                        [
                            {
                                "name": self.robot.body_names[index],
                                "origin_distance_m": float(distances[index].item()),
                                "origin_xyz": _vec3(body_positions[index]),
                            }
                            for index in nearest
                        ]
                        if physics_step in {0, physics_steps}
                        else []
                    ),
                }
            )
            if physics_step == physics_steps:
                break
            self.robot.set_joint_position_target(self.robot.data.joint_pos)
            self.scene.write_data_to_sim()
            self._step_simulation()
        return reports

    def _resolve_entities(self) -> None:
        self.robot = self.scene["robot"]
        self.target_object = self.scene["object"]
        self.contact_sensor = self.scene["finger_contact"]
        self.head_camera = (
            self.scene["head_camera"] if self.options.enable_cameras else None
        )
        self.wrist_camera = (
            self.scene["wrist_camera"] if self.options.enable_cameras else None
        )
        self.overview_camera = (
            self.scene["overview_camera"] if self.options.enable_cameras else None
        )

        self.arm_joint_ids, arm_names = self.robot.find_joints(
            list(ARM_JOINT_NAMES),
            preserve_order=True,
        )
        self.gripper_joint_ids, gripper_names = self.robot.find_joints(
            list(GRIPPER_JOINT_NAMES),
            preserve_order=True,
        )
        self.leg_joint_ids, leg_names = self.robot.find_joints(
            list(LEG_JOINT_NAMES),
            preserve_order=True,
        )
        self.link6_body_ids, link_names = self.robot.find_bodies(
            ["arm_link6"],
            preserve_order=True,
        )
        if arm_names != list(ARM_JOINT_NAMES):
            raise RuntimeError(f"Unexpected arm joint order: {arm_names}")
        if gripper_names != list(GRIPPER_JOINT_NAMES):
            raise RuntimeError(f"Unexpected gripper joint order: {gripper_names}")
        if leg_names != list(LEG_JOINT_NAMES):
            raise RuntimeError(f"Unexpected leg joint order: {leg_names}")
        if link_names != ["arm_link6"]:
            raise RuntimeError(f"Could not resolve arm_link6: {link_names}")
        if len(self.contact_sensor.body_names) != 2:
            raise RuntimeError(
                f"Expected two finger contact bodies, got {self.contact_sensor.body_names}"
            )

        self.arm_kinematics = CalibratedArmKinematics()
        self._arm_ik_seed = tuple(
            float(value)
            for value in self.robot.data.joint_pos[0, self.arm_joint_ids]
            .detach()
            .cpu()
            .tolist()
        )
        self._last_ik_position_error_m = 0.0
        self._last_ik_iterations = 0
        self._gripper_close_commanded = False
        self._tcp_offset = torch.tensor(
            [[TCP_OFFSET_X_M, 0.0, 0.0]],
            device=self.sim.device,
            dtype=torch.float32,
        )

    def _warm_up(self) -> None:
        self._reset_robot()
        for _ in range(12):
            self._hold_default_joints()
            self.scene.write_data_to_sim()
            self._step_simulation()
            # Access the tensors so render products and lazy sensors initialize.
            if (
                self.options.enable_cameras
                and self._physics_step_count % self.camera_render_stride == 0
            ):
                _ = self.head_camera.data.output["rgb"]
                _ = self.wrist_camera.data.output["rgb"]
                _ = self.overview_camera.data.output["rgb"]

    def _run_episode(
        self,
        *,
        run_id: str,
        episode_index: int,
        episode_seed: int,
    ) -> dict:
        rng = random.Random(episode_seed)
        lane_offset = rng.uniform(-0.025, 0.025)
        lane_x = OBJECT_INTERCEPT_X_M + lane_offset
        belt_speed = self._task_belt_speed()

        task = TaskManifest(
            task_id=f"{self.options.task_type.value}-seed-{episode_seed}",
            task_type=self.options.task_type,
            instruction="Pick the red block from the conveyor before it exits.",
            target_object_id="target-red-cuboid",
            object_ids=("target-red-cuboid",),
            seed=episode_seed,
            belt_speed_mps=belt_speed,
            belt_surface_z_m=BELT_TOP_Z_M,
            transport_direction_xyz=TRANSPORT_DIRECTION_WORLD,
            exit_plane_point_xyz=EXIT_PLANE_POINT_WORLD,
            max_duration_s=self.options.max_duration_s,
            metadata={
                "layout_id": LAYOUT_ID,
                "lane_axis_xyz": [1.0, 0.0, 0.0],
                "lane_offset_m": lane_offset,
                "intercept_xyz_m": [
                    lane_x,
                    OBJECT_INTERCEPT_Y_M,
                    OBJECT_CENTER_Z_M,
                ],
                "spawn_xyz_m": [
                    lane_x,
                    (
                        OBJECT_INTERCEPT_Y_M
                        if self.options.task_type is TaskType.C0_STATIC_PICK
                        else OBJECT_SPAWN_Y_M
                    ),
                    OBJECT_CENTER_Z_M,
                ],
            },
        )
        episode_id = f"{run_id}-ep{episode_index:04d}-seed{episode_seed}"
        manifest = EpisodeManifest(
            episode_id=episode_id,
            run_id=run_id,
            protocol_version=self.benchmark.protocol_version,
            task=task,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            env_id=0,
            asset_hashes={"go2_x5_urdf_sha256": _sha256(GO2_X5_URDF)},
            seeds={"episode": episode_seed, "layout": episode_seed},
            metadata={
                "isaac_sim": _package_version("isaacsim"),
                "isaac_lab": _package_version("isaaclab"),
                "device": self.options.device,
                "physics_hz": self.benchmark.physics_hz,
                "control_hz": self.benchmark.control_hz,
                "camera_hz": self.benchmark.camera_hz,
                "layout_id": LAYOUT_ID,
                "cameras_enabled": self.options.enable_cameras,
                "video_enabled": self.options.save_video,
                "joint_names": list(self.robot.joint_names),
                "cameras": _camera_contract(),
            },
        )

        recorder = EpisodeRecorder(self.options.output_root, manifest, self.benchmark)
        video: EpisodeVideoWriter | None = None
        samples: list[StepSample] = []
        previous_phase = None
        emitted_lift = False
        emitted_exit = False
        wall_started = time.perf_counter()
        max_steps = int(self.options.max_duration_s * self.benchmark.control_hz)
        current_operation = "episode_reset"

        try:
            self._reset_episode_state(
                task_type=self.options.task_type,
                lane_x=lane_x,
            )
            current_operation = "oracle_initialization"
            oracle = DynamicPickOracle(
                OracleConfig(
                    belt_top_z=BELT_TOP_Z_M,
                    intercept_xy=(lane_x, OBJECT_INTERCEPT_Y_M),
                    transport_direction_xy=TRANSPORT_DIRECTION_WORLD[:2],
                    # The benchmark TCP lies near the finger tips.  During a
                    # transverse moving grasp, placing the TCP 2 cm farther
                    # forward seats the block deeper between the fingers
                    # instead of letting it squeeze out along local +X.
                    grasp_target_offset_xy=(
                        0.02
                        if self.options.task_type is TaskType.C1_DYNAMIC_PICK
                        else 0.0,
                        0.0,
                    ),
                    expected_transport_speed=belt_speed,
                    hold_duration=(
                        self.benchmark.evaluation.hold_time_s + self.control_dt
                    ),
                )
            )
            oracle.reset(
                sim_time=0.0,
                object_position=(
                    lane_x,
                    (
                        OBJECT_INTERCEPT_Y_M
                        if self.options.task_type is TaskType.C0_STATIC_PICK
                        else OBJECT_SPAWN_Y_M
                    ),
                    OBJECT_CENTER_Z_M,
                ),
            )
            if self.options.save_video:
                current_operation = "video_initialization"
                video = EpisodeVideoWriter(
                    recorder.artifact_directory,
                    fps=float(self.benchmark.camera_hz),
                    frame_size=(224, 224),
                )
            current_operation = "recorder_initial_events"
            recorder.record_event(
                Event(
                    kind=EventKind.EPISODE_START,
                    time_s=0.0,
                    payload={"seed": episode_seed},
                )
            )
            recorder.record_event(
                Event(
                    kind=EventKind.OBJECT_SPAWNED,
                    time_s=0.0,
                    payload={"object_id": task.target_object_id},
                )
            )

            for control_step in range(max_steps):
                current_operation = "observation_read"
                observation_capture_s = time.perf_counter() - wall_started
                state = self._read_state(task)
                sim_time_s = control_step * self.control_dt
                inference_start_s = time.perf_counter() - wall_started
                current_operation = "oracle_step"
                command = oracle.step(
                    OracleObservation(
                        sim_time=sim_time_s,
                        object_position=state["object_xyz"],
                        object_velocity=state["object_linear_velocity"],
                        gripper_position=state["tcp_xyz"],
                        object_crossed_exit=state["target_crossed_exit"],
                        robot_fallen=state["robot_fallen"],
                        object_lifted=state["object_lifted"],
                        left_contact=state["left_contact"],
                        right_contact=state["right_contact"],
                        secure_grasp=(
                            state["target_in_gripper"] and state["object_lifted"]
                        ),
                        forbidden_collision=state["forbidden_collision"],
                    )
                )
                inference_end_s = time.perf_counter() - wall_started

                if command.phase.value != previous_phase:
                    current_operation = "recorder_phase_event"
                    recorder.record_event(
                        Event(
                            kind=EventKind.PHASE_CHANGED,
                            time_s=sim_time_s,
                            payload={
                                "from": previous_phase,
                                "to": command.phase.value,
                            },
                        )
                    )
                    if command.phase.value == "close":
                        recorder.record_event(
                            Event(
                                kind=EventKind.GRIPPER_CLOSED,
                                time_s=sim_time_s,
                                payload={"commanded": True},
                            )
                        )
                    previous_phase = command.phase.value

                current_operation = "control_ik"
                arm_targets = self._apply_command(command)
                action_enqueue_s = time.perf_counter() - wall_started
                action_execute_start_s = time.perf_counter() - wall_started
                current_operation = "simulation_step"
                for _ in range(self.decimation):
                    self.scene.write_data_to_sim()
                    self._step_simulation()
                action_execute_end_s = time.perf_counter() - wall_started

                sample_time_s = (control_step + 1) * self.control_dt
                current_operation = "observation_read_after_step"
                state = self._read_state(task)
                camera_frame_index = (
                    video.frame_count
                    if video is not None
                    and (control_step + 1) % self.camera_stride == 0
                    else None
                )
                sample = StepSample(
                    sim_step=control_step,
                    sim_time_s=sample_time_s,
                    env_id=0,
                    object_xyz=state["object_xyz"],
                    object_linear_velocity=state["object_linear_velocity"],
                    tcp_xyz=state["tcp_xyz"],
                    tcp_wxyz=state["tcp_wxyz"],
                    joint_positions=state["joint_positions"],
                    joint_velocities=state["joint_velocities"],
                    camera_frame_index=camera_frame_index,
                    belt_command_speed_mps=belt_speed,
                    belt_measured_speed_mps=self._surface_speed(),
                    gripper_closed=state["gripper_closed"],
                    left_contact=state["left_contact"],
                    right_contact=state["right_contact"],
                    target_in_gripper=state["target_in_gripper"],
                    target_crossed_exit=state["target_crossed_exit"],
                    robot_fallen=state["robot_fallen"],
                    forbidden_collision=state["forbidden_collision"],
                    phase=command.phase.value,
                    action={
                        "target_tcp_xyz": command.target_position,
                        "target_tcp_wxyz": command.target_orientation,
                        "arm_joint_position_target": arm_targets,
                        "gripper_opening_target": command.gripper_opening,
                        "source": "privileged_constant_velocity_oracle",
                        "observation_sim_time_s": sim_time_s,
                        "execution_end_sim_time_s": sample_time_s,
                        "prediction_horizon_s": 0.12,
                    },
                    timing=TimingTrace(
                        observation_capture_s=observation_capture_s,
                        inference_start_s=inference_start_s,
                        inference_end_s=inference_end_s,
                        action_enqueue_s=action_enqueue_s,
                        action_execute_start_s=action_execute_start_s,
                        action_execute_end_s=action_execute_end_s,
                    ),
                    metadata={
                        "gripper_opening_m": state["gripper_opening_m"],
                        "finger_contact_forces_n": state["finger_contact_forces_n"],
                        "object_lifted": state["object_lifted"],
                        "object_transport_speed_mps": state[
                            "object_transport_speed_mps"
                        ],
                        "future_state_labels": _future_state_labels(
                            state,
                            task=task,
                            intercept_xyz=(
                                lane_x,
                                OBJECT_INTERCEPT_Y_M,
                                OBJECT_CENTER_Z_M,
                            ),
                            prediction_horizon_s=0.12,
                        ),
                        "ik_position_error_m": self._last_ik_position_error_m,
                        "ik_iterations": self._last_ik_iterations,
                    },
                )
                current_operation = "recorder_step"
                recorder.record_step(sample)
                samples.append(sample)

                if state["object_lifted"] and not emitted_lift:
                    emitted_lift = True
                    current_operation = "recorder_lift_event"
                    recorder.record_event(
                        Event(
                            kind=EventKind.TARGET_LIFTED,
                            time_s=sample_time_s,
                            payload={"height_m": state["object_xyz"][2] - BELT_TOP_Z_M},
                        )
                    )
                if state["target_crossed_exit"] and not emitted_exit:
                    emitted_exit = True
                    current_operation = "recorder_exit_event"
                    recorder.record_event(
                        Event(
                            kind=EventKind.TARGET_CROSSED_EXIT,
                            time_s=sample_time_s,
                        )
                    )

                if (
                    video is not None
                    and (control_step + 1) % self.camera_stride == 0
                ):
                    current_operation = "video_add_frame"
                    video.add(
                        sim_step=control_step,
                        sim_time_s=sample_time_s,
                        head_rgb=self.head_camera.data.output["rgb"],
                        wrist_rgb=self.wrist_camera.data.output["rgb"],
                        overview_rgb=self.overview_camera.data.output["rgb"],
                    )
                    current_operation = "recorder_camera_event"
                    recorder.record_event(
                        Event(
                            kind=EventKind.CAMERA_FRAME,
                            time_s=sample_time_s,
                            payload={
                                "frame_index": camera_frame_index,
                                "sim_step": control_step,
                            },
                        )
                    )

                if command.terminal:
                    break

            current_operation = "episode_evaluation"
            evaluation = evaluate_episode(self.benchmark, task, samples)
            current_operation = "recorder_result_event"
            if evaluation.success:
                recorder.record_event(
                    Event(
                        kind=EventKind.GRASP_VERIFIED,
                        time_s=samples[-1].sim_time_s,
                    )
                )
            else:
                recorder.record_event(
                    Event(
                        kind=EventKind.FAILURE,
                        time_s=samples[-1].sim_time_s if samples else 0.0,
                        payload={"reason": evaluation.failure_reason.value},
                    )
                )
            if video is not None:
                current_operation = "video_close"
                video.close()
            current_operation = "recorder_finalize"
            episode_path, evaluation = recorder.finalize(evaluation)
        except Exception as error:
            failure_metadata = {
                "operation": current_operation,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            if video is not None:
                try:
                    video.close()
                except Exception as close_error:
                    failure_metadata["video_close_error"] = {
                        "exception_type": type(close_error).__name__,
                        "message": str(close_error),
                    }
            failure_reason = (
                FailureReason.RECORDER_ERROR
                if current_operation.startswith("recorder_")
                else FailureReason.RUNTIME_ERROR
            )
            if not recorder.finalized:
                episode_path, evaluation = recorder.abort(
                    failure_reason,
                    failure_metadata,
                )
            else:
                raise

        return {
            "episode_id": episode_id,
            "path": str(episode_path),
            "success": evaluation.success,
            "failure_reason": evaluation.failure_reason.value,
            "metrics": evaluation.metrics,
            "video_frames": video.frame_count if video is not None else 0,
        }

    def _reset_episode_state(self, *, task_type: TaskType, lane_x: float) -> None:
        self._reset_robot()
        # Keep the target away from the arm while the arm moves from its URDF
        # default pose to the interception pose.  The default TCP is close to the
        # belt and would otherwise strike the freshly spawned object.
        self._write_object_state(x=1.40, y=0.0)
        self.scene.reset()
        self._preposition_arm()

        spawn_y = (
            OBJECT_INTERCEPT_Y_M
            if task_type is TaskType.C0_STATIC_PICK
            else OBJECT_SPAWN_Y_M
        )
        self._write_object_state(x=lane_x, y=spawn_y)
        self.scene.reset()
        self._physics_step_count = 0

    def _write_object_state(self, *, x: float, y: float) -> None:
        root_state = self.target_object.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        root_state[:, 0] = x
        root_state[:, 1] = y
        root_state[:, 2] = OBJECT_CENTER_Z_M
        root_state[:, 3:7] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=self.sim.device,
        )
        root_state[:, 7:] = 0.0
        self.target_object.write_root_pose_to_sim(root_state[:, :7])
        self.target_object.write_root_velocity_to_sim(root_state[:, 7:])

    def _preposition_arm(self) -> None:
        # This calibrated configuration is tucked, collision-free and gives a
        # high TCP pose at roughly (0.37, 0.0, 1.08).  Writing it before
        # spawning the target avoids asking Cartesian IK to cross the belt from
        # the URDF's low default pose.
        arm_pose = torch.tensor(
            [[0.0, 1.0, 1.5, 0.0, 0.0, 0.0]],
            device=self.sim.device,
            dtype=torch.float32,
        )
        joint_positions = self.robot.data.joint_pos.clone()
        joint_velocities = torch.zeros_like(joint_positions)
        joint_positions[:, self.arm_joint_ids] = arm_pose
        joint_positions[:, self.gripper_joint_ids] = 0.044
        self.robot.write_joint_state_to_sim(joint_positions, joint_velocities)
        self._arm_ik_seed = tuple(
            float(value)
            for value in arm_pose[0].detach().cpu().tolist()
        )
        self.robot.set_joint_position_target(arm_pose, joint_ids=self.arm_joint_ids)
        self.robot.set_joint_position_target(
            torch.full(
                (1, len(self.gripper_joint_ids)),
                0.044,
                device=self.sim.device,
            ),
            joint_ids=self.gripper_joint_ids,
        )
        for _ in range(20):
            self.scene.write_data_to_sim()
            self._step_simulation()

    def _reset_robot(self) -> None:
        self._gripper_close_commanded = False
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
        )
        self._hold_default_joints()

    def _hold_default_joints(self) -> None:
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.leg_joint_ids],
            joint_ids=self.leg_joint_ids,
        )
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.arm_joint_ids],
            joint_ids=self.arm_joint_ids,
        )
        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.gripper_joint_ids],
            joint_ids=self.gripper_joint_ids,
        )

    def _apply_command(self, command) -> list[float]:
        solution = self.arm_kinematics.solve(
            command.target_position,
            command.target_orientation,
            seed=self._arm_ik_seed,
        )
        self._arm_ik_seed = solution.joint_positions
        self._last_ik_position_error_m = solution.position_error_m
        self._last_ik_iterations = solution.iterations
        planned_targets = torch.tensor(
            [solution.joint_positions],
            device=self.sim.device,
            dtype=torch.float32,
        )
        current_arm = self.robot.data.joint_pos[:, self.arm_joint_ids]
        max_delta = 0.08
        arm_targets = current_arm + torch.clamp(
            planned_targets - current_arm,
            min=-max_delta,
            max=max_delta,
        )
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        arm_targets = torch.maximum(
            torch.minimum(arm_targets, limits[..., 1]),
            limits[..., 0],
        )

        self.robot.set_joint_position_target(
            self.robot.data.default_joint_pos[:, self.leg_joint_ids],
            joint_ids=self.leg_joint_ids,
        )
        self.robot.set_joint_position_target(
            arm_targets,
            joint_ids=self.arm_joint_ids,
        )
        gripper_targets = torch.full(
            (1, len(self.gripper_joint_ids)),
            float(command.gripper_opening),
            device=self.sim.device,
        )
        self.robot.set_joint_position_target(
            gripper_targets,
            joint_ids=self.gripper_joint_ids,
        )
        self._gripper_close_commanded = float(command.gripper_opening) <= 0.001
        return [float(value) for value in arm_targets[0].detach().cpu().tolist()]

    def _read_state(self, task: TaskManifest) -> dict:
        object_xyz_tensor = self.target_object.data.root_pos_w[0, :3]
        object_velocity_tensor = self.target_object.data.root_lin_vel_w[0, :3]
        link6_pose_w = self.robot.data.body_pose_w[:, self.link6_body_ids[0]]
        tcp_xyz_tensor = link6_pose_w[:, :3] + quat_apply(
            link6_pose_w[:, 3:7],
            self._tcp_offset,
        )

        force_matrix = self.contact_sensor.data.force_matrix_w
        forces_by_name: dict[str, float] = {}
        if force_matrix is not None:
            magnitudes = torch.linalg.vector_norm(force_matrix[0, :, 0, :], dim=-1)
            forces_by_name = {
                name: float(magnitudes[index].item())
                for index, name in enumerate(self.contact_sensor.body_names)
            }
        left_force = forces_by_name.get("arm_link7", 0.0)
        right_force = forces_by_name.get("arm_link8", 0.0)
        left_contact = left_force >= self.contact_sensor.cfg.force_threshold
        right_contact = right_force >= self.contact_sensor.cfg.force_threshold

        gripper_positions = self.robot.data.joint_pos[0, self.gripper_joint_ids]
        gripper_opening = float(gripper_positions.mean().item())
        # A finger cannot reach its empty-gripper joint limit while an object is
        # held between the pads.  "Closed" therefore records that a close
        # command is actively being executed; physical security remains a
        # separate conjunction of bilateral target contact and geometry.
        gripper_closed = self._gripper_close_commanded
        tcp_object_distance = torch.linalg.vector_norm(
            tcp_xyz_tensor[0] - object_xyz_tensor
        ).item()
        target_in_gripper = (
            left_contact
            and right_contact
            and tcp_object_distance < 0.08
        )
        object_lifted = float(object_xyz_tensor[2].item()) - BELT_TOP_Z_M >= (
            self.benchmark.evaluation.lift_height_m
        )
        root_z = float(self.robot.data.root_pos_w[0, 2].item())
        forbidden_collision = float(tcp_xyz_tensor[0, 2].item()) < BELT_TOP_Z_M - 0.01
        object_xyz = _vec3(object_xyz_tensor)
        object_linear_velocity = _vec3(object_velocity_tensor)

        return {
            "object_xyz": object_xyz,
            "object_linear_velocity": object_linear_velocity,
            "object_transport_speed_mps": task.forward_speed(
                object_linear_velocity
            ),
            "tcp_xyz": _vec3(tcp_xyz_tensor[0]),
            "tcp_wxyz": _vecn(link6_pose_w[0, 3:7]),
            "joint_positions": _vecn(self.robot.data.joint_pos[0]),
            "joint_velocities": _vecn(self.robot.data.joint_vel[0]),
            "gripper_opening_m": gripper_opening,
            "gripper_closed": gripper_closed,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "finger_contact_forces_n": [left_force, right_force],
            "target_in_gripper": target_in_gripper,
            "target_crossed_exit": task.has_crossed_exit(object_xyz),
            "object_lifted": object_lifted,
            "robot_fallen": root_z < 0.25,
            "forbidden_collision": forbidden_collision,
        }

    def _tcp_position(self) -> torch.Tensor:
        link6_pose_w = self.robot.data.body_pose_w[:, self.link6_body_ids[0]]
        return (
            link6_pose_w[:, :3]
            + quat_apply(link6_pose_w[:, 3:7], self._tcp_offset)
        )[0]

    def _surface_speed(self) -> float:
        value = self.surface_velocity_api.GetSurfaceVelocityAttr().Get()
        return sum(
            float(value[index]) * direction
            for index, direction in enumerate(TRANSPORT_DIRECTION_WORLD)
        )

    def _task_belt_speed(self) -> float:
        if self.options.task_type is TaskType.C0_STATIC_PICK:
            return 0.0
        return float(self.options.belt_speed_mps)

    def _step_simulation(self) -> None:
        self._physics_step_count += 1
        should_render = (
            self.options.enable_cameras
            and self._physics_step_count % self.camera_render_stride == 0
        )
        self.sim.step(render=should_render)
        self.scene.update(self.physics_dt)


def run_collection(options: RuntimeOptions) -> dict:
    runtime = ConveyorRuntime(options)
    try:
        return runtime.run()
    finally:
        runtime.close()


def _vec3(tensor: torch.Tensor) -> tuple[float, float, float]:
    values = tensor.detach().cpu().tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


def _vecn(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().cpu().tolist())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _camera_contract() -> dict:
    width = 224
    height = 224
    horizontal_aperture = 20.955

    def camera(
        *,
        focal_length: float,
        parent: str,
        offset_xyz: tuple[float, float, float],
        offset_wxyz: tuple[float, float, float, float],
        orientation_convention: str,
        role: str,
    ) -> dict:
        focal_pixels = focal_length / horizontal_aperture * width
        return {
            "resolution": [width, height],
            "fps": 25.0,
            "model": "pinhole",
            "role": role,
            "intrinsics": [
                [focal_pixels, 0.0, width / 2.0],
                [0.0, focal_pixels, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            "mount": {
                "parent": parent,
                "xyz_m": list(offset_xyz),
                "wxyz": list(offset_wxyz),
                "orientation_convention": orientation_convention,
            },
        }

    return {
        "head_rgb": camera(
            focal_length=18.0,
            parent="base",
            offset_xyz=HEAD_CAMERA_OFFSET_XYZ,
            offset_wxyz=HEAD_CAMERA_OFFSET_WXYZ,
            orientation_convention="world",
            role="policy_observation",
        ),
        "wrist_rgb": camera(
            focal_length=18.0,
            parent="arm_link6",
            offset_xyz=WRIST_CAMERA_OFFSET_XYZ,
            offset_wxyz=WRIST_CAMERA_OFFSET_WXYZ,
            orientation_convention="world",
            role="policy_observation",
        ),
        "overview_rgb": camera(
            focal_length=18.0,
            parent="environment_origin",
            offset_xyz=OVERVIEW_CAMERA_OFFSET_XYZ,
            offset_wxyz=OVERVIEW_CAMERA_OFFSET_WXYZ,
            orientation_convention="world",
            role="observer_only",
        ),
    }


def _future_state_labels(
    state: dict,
    *,
    task: TaskManifest,
    intercept_xyz: tuple[float, float, float],
    prediction_horizon_s: float,
) -> dict:
    position = state["object_xyz"]
    velocity = state["object_linear_velocity"]
    predicted = [
        float(position[index] + velocity[index] * prediction_horizon_s)
        for index in range(3)
    ]
    forward_speed = task.forward_speed(velocity)
    remaining_distance = task.remaining_distance_to_exit(position)
    time_to_exit_s = (
        max(0.0, remaining_distance / forward_speed)
        if forward_speed > 1.0e-4
        else None
    )
    transport_progress = task.transport_progress(position)
    intercept_progress = task.transport_progress(intercept_xyz)
    return {
        "prediction_horizon_s": prediction_horizon_s,
        "constant_velocity_object_xyz": predicted,
        "transport_progress_m": transport_progress,
        "transport_speed_mps": forward_speed,
        "remaining_distance_to_exit_m": remaining_distance,
        "time_to_exit_s": time_to_exit_s,
        "inside_nominal_grasp_window": (
            abs(transport_progress - intercept_progress) <= 0.15
        ),
    }
