#!/usr/bin/env python3
"""Run model-owned Waypoint routing in the approved arm-vla Isaac runtime."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint import (  # noqa: E402
    CAMERA_CALIBRATION_ID,
    WaypointActionDomain,
    WaypointRoute,
)
from conveyor_bench.conveyorvla.waypoint_execution import (  # noqa: E402
    CuRoboIKRecedingHorizonExecutor,
    ManipulationExecutionConfig,
    NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE,
    NAVIGATION_SAFETY_PROFILE_CONTRACT,
    NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
    NAVIGATION_SAFETY_PROFILES,
    NavigationExecutionConfig,
    PCTDWARecedingHorizonExecutor,
)
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (  # noqa: E402
    APPROVED_ARM_VLA_COMMIT,
    ArmVLADWAControllerAdapter,
    ArmVLAPCTPlannerAdapter,
    JointPathController,
    JsonLineCuRoboTransport,
    WaypointCuRoboPlannerAdapter,
)
from conveyor_bench.conveyorvla.waypoint_protocol import (  # noqa: E402
    RECOVER_ROUTE,
)
from conveyor_bench.conveyorvla.waypoint_rollout import (  # noqa: E402
    TemporalJPEGBuffer,
    WaypointHTTPClient,
    measured_arm_joints,
    measured_body_velocity,
    planner_base_from_query_base,
    tcp_pose_in_query_base,
    waypoint_request_from_frames,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:18081")
    parser.add_argument("--model-timeout-s", type=float, default=180.0)
    parser.add_argument("--curobo-port", type=int, default=8766)
    parser.add_argument("--curobo-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--max-control-steps", type=int, default=24_000)
    parser.add_argument(
        "--navigation-safety-profile",
        choices=NAVIGATION_SAFETY_PROFILES,
        default=NAVIGATION_SAFETY_PROFILE_CONTRACT,
        help=(
            "contract ranks a checkpoint-calibrated trusted prefix by useful "
            "lookahead and selects the first PCT-planable model goal; "
            "lookahead-arm-vla-reference uses that selector with arm-vla's "
            "original downstream tolerances, DWA, stall, and requery behavior; "
            "arm-vla-reference applies only arm-vla 388b681's first-waypoint "
            "limits, tolerances, DWA bounds, stall detector, and requery behavior; "
            "executable-prefix-diagnostic still audits all 20 but permits only "
            "a legal first non-degenerate waypoint to reach PCT/DWA; "
            "unbounded-translation-diagnostic additionally disables the "
            "translation cap for that executed waypoint"
        ),
    )
    parser.add_argument(
        "--stop-after-route",
        choices=tuple(route.value for route in WaypointRoute if route is not WaypointRoute.DONE),
    )
    parser.add_argument(
        "--required-first-route",
        choices=tuple(route.value for route in WaypointRoute),
    )
    parser.add_argument(
        "--require-initial-source-visible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require settled source object to lie in the front camera sector",
    )
    parser.add_argument(
        "--initial-source-max-bearing-deg",
        type=float,
        default=30.0,
        help="maximum absolute settled source bearing for the visibility preflight",
    )
    return parser


class _ExternalWaypointCuRoboLifecycle:
    """Bind the reference pipeline lifecycle to the already-gated waypoint service."""

    def __init__(
        self,
        _legacy_config: Any,
        *,
        port: int,
        timeout_s: float,
        transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self._transport = transport or JsonLineCuRoboTransport(
            port=int(port), timeout_s=float(timeout_s)
        )
        self.start_report: dict[str, Any] = {
            "requested": False,
            "external_waypoint_service": True,
            "port": int(port),
        }

    def start(self) -> None:
        ping = dict(self._transport({"command": "ping"}))
        capabilities = dict(self._transport({"command": "capabilities"}))
        features = capabilities.get("features")
        valid = bool(
            ping.get("ok") is True
            and ping.get("arm_vla_reference_commit") == APPROVED_ARM_VLA_COMMIT
            and capabilities.get("ok") is True
            and capabilities.get("arm_vla_reference_commit")
            == APPROVED_ARM_VLA_COMMIT
            and isinstance(features, Mapping)
            and features.get("direct_absolute_tcp_target") is True
            and features.get("input_target_frame") == "query-base-B_t"
            and features.get("planner_target_frame") == "curobo-planner-base"
            and features.get("orientation_fallback") is False
            and features.get("world_collision") is True
        )
        self.start_report = {
            "requested": True,
            "started": False,
            "reused_existing": valid,
            "ready": valid,
            "external_waypoint_service": True,
            "arm_vla_reference_commit": ping.get("arm_vla_reference_commit"),
            "capabilities": capabilities,
        }
        if not valid:
            raise RuntimeError("external waypoint cuRobo service failed capability gate")

    def wait_until_ready(self) -> bool:
        return self.start_report.get("ready") is True

    def close(self) -> None:
        self.start_report["external_server_preserved"] = True


class WaypointRolloutPipeline:
    """Own the physics loop; every semantic dispatch comes from the latest Qwen route."""

    def __init__(
        self,
        *,
        config: Any,
        episode_spec: Any,
        episode_seed: int,
        episode_dir: str | Path,
        simulation: Any,
        model_endpoint: str,
        model_timeout_s: float,
        jpeg_quality: int,
        curobo_port: int,
        curobo_timeout_s: float,
        max_queries: int,
        max_control_steps: int,
        stop_after_route: str | None,
        required_first_route: str | None,
        navigation_safety_profile: str,
        navigation_max_chunk_steps: int,
        require_initial_source_visible: bool,
        initial_source_max_bearing_deg: float,
        close_simulation_on_exit: bool,
    ) -> None:
        from source.interfaces import NavGoal, RobotAction, SimulationState
        from source.navigation.navlib import DWAController
        from source.pipeline.navigation_smoke import create_navigation_components

        self.config = config
        self.episode_spec = episode_spec
        self.episode_seed = int(episode_seed)
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.simulation = simulation
        self.client = WaypointHTTPClient(model_endpoint, timeout_s=model_timeout_s)
        self.max_queries = int(max_queries)
        self.max_control_steps = int(max_control_steps)
        self.stop_after_route = stop_after_route
        self.required_first_route = required_first_route
        self.require_initial_source_visible = bool(require_initial_source_visible)
        self.initial_source_max_bearing_deg = float(initial_source_max_bearing_deg)
        self.close_simulation_on_exit = bool(close_simulation_on_exit)
        self.RobotAction = RobotAction
        self.SimulationState = SimulationState

        if self.max_queries <= 0 or self.max_control_steps <= 0:
            raise ValueError("waypoint rollout limits must be positive")
        if (
            not math.isfinite(self.initial_source_max_bearing_deg)
            or not 0.0 < self.initial_source_max_bearing_deg < 90.0
        ):
            raise ValueError("initial source bearing limit must be within (0, 90) deg")
        planner, reference_executor, navigation_verifier = create_navigation_components(
            config=config,
            episode_spec=episode_spec,
        )
        self.reference_navigation_executor = reference_executor
        self.navigation_verifier = navigation_verifier
        self.pct_adapter = ArmVLAPCTPlannerAdapter(
            planner,
            simulation_state_factory=SimulationState,
            nav_goal_factory=NavGoal,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )
        self.dwa_adapter = ArmVLADWAControllerAdapter(
            DWAController,
            reference_executor.dwa_config,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )
        arm_vla_reference = (
            navigation_safety_profile
            in (
                NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE,
                NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
            )
        )
        self.navigation = PCTDWARecedingHorizonExecutor(
            self.pct_adapter,
            self.dwa_adapter,
            NavigationExecutionConfig(
                safety_profile=navigation_safety_profile,
                goal_tolerance_m=(
                    float(reference_executor.position_tolerance)
                    if arm_vla_reference
                    else 0.12
                ),
                yaw_tolerance_rad=(
                    float(reference_executor.yaw_tolerance)
                    if arm_vla_reference
                    else 0.14
                ),
                terminal_yaw_max_radps=(
                    float(reference_executor.dwa_config.max_angular_velocity)
                    if arm_vla_reference
                    else 0.60
                ),
                max_chunk_execution_steps=int(navigation_max_chunk_steps),
                stow_joint_target=None,
                carry_joint_target=None,
            ),
            stall_detector=(
                reference_executor.stall_detector if arm_vla_reference else None
            ),
        )
        transport = JsonLineCuRoboTransport(
            port=int(curobo_port), timeout_s=float(curobo_timeout_s)
        )
        self.curobo = WaypointCuRoboPlannerAdapter(
            transport,
            deployment="simulation",
            safety_gate=_simulation_curobo_safety_gate,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )
        self.manipulation = CuRoboIKRecedingHorizonExecutor(
            self.curobo,
            JointPathController(),
            ManipulationExecutionConfig(),
        )
        separation_steps = int(round(0.20 / float(config.navigation.control_dt)))
        self.frames = TemporalJPEGBuffer(
            separation_steps=separation_steps,
            jpeg_quality=int(jpeg_quality),
        )
        self._camera_states: dict[int, Any] = {}
        self._last_query_camera_step: int | None = None
        self._control_steps = 0
        self._query_count = 0
        self._active_route = "INITIALIZE"
        self._state_trace: list[str] = []
        self._held_arm_target: tuple[float, ...] | None = None
        self._held_gripper_command: str | None = None
        self._prepared_for_pick = False
        self._last_settled_arm_route: str | None = None
        self._navigation_since_arm_settle = True
        self._trace_stream: Any | None = None
        self._video: Any | None = None
        self._initial_source_visibility: dict[str, Any] | None = None

    def run_episode(self) -> dict[str, Any]:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.episode_dir / "waypoint_trace.jsonl"
        summary_path = self.episode_dir / "summary.json"
        success = False
        pure_physics_success = False
        failure_reason = "episode_not_started"
        started = time.time()
        self._trace_stream = trace_path.open("x", encoding="utf-8")
        try:
            self._start_video()
            self._prepare_episode()
            failure_reason = "query_limit_exhausted"
            while self._query_count < self.max_queries:
                request, query_state = self._next_request()
                wire = self.client.infer(request)
                response = wire.response
                self._query_count += 1
                self._record(
                    "model_query",
                    {
                        "request": {
                            "protocol_version": request.protocol_version,
                            "request_id": request.request_id,
                            "episode_id": request.episode_id,
                            "sequence_id": request.sequence_id,
                            "instruction": request.instruction,
                            "image_count": 4,
                            "camera_calibration_id": request.camera_calibration_id,
                            "model_state_fields": 0,
                        },
                        "response": response.to_mapping(),
                        "model_trace": dict(wire.trace),
                        "diffusion_seed": wire.diffusion_seed,
                        "query_anchor": _state_snapshot(query_state),
                    },
                )
                if self._query_count == 1 and self.required_first_route is not None:
                    if response.route != self.required_first_route:
                        failure_reason = (
                            "required_first_route_mismatch:"
                            f"expected={self.required_first_route}:actual={response.route}"
                        )
                        break
                self._active_route = response.route
                self._state_trace.append(response.route)
                if response.route == RECOVER_ROUTE:
                    failure_reason = f"model_recover:{response.recover_reason}"
                    break
                if response.route == WaypointRoute.DONE.value:
                    success, verification = self._verify_done()
                    pure_physics_success = success
                    failure_reason = "" if success else "done_verification_failed"
                    self._record("done_verification", verification)
                    break
                if response.action_domain == WaypointActionDomain.NAVIGATION.value:
                    self._navigation_since_arm_settle = True
                    completed, reason = self._execute_navigation(response, query_state)
                elif response.action_domain == WaypointActionDomain.MANIPULATION.value:
                    completed, reason = self._execute_manipulation(response, query_state)
                else:
                    completed, reason = False, "active_route_has_invalid_domain"
                if not completed:
                    failure_reason = reason
                    break
                if self.stop_after_route == response.route:
                    success = True
                    failure_reason = ""
                    self._record(
                        "staged_route_gate_passed",
                        {"route": response.route, "sequence_id": response.sequence_id},
                    )
                    break
            if not success and not failure_reason:
                failure_reason = "episode_failed_without_reason"
        except BaseException as error:
            failure_reason = f"{type(error).__name__}:{error}"
            self._record(
                "rollout_exception",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        finally:
            try:
                self._apply_final_hold()
            except Exception as error:
                self._record(
                    "final_hold_failed",
                    {"type": type(error).__name__, "message": str(error)},
                )
            try:
                video_summary = self._close_video(
                    "success" if success else "failed"
                )
            except Exception as error:
                video_summary = {
                    "closed": False,
                    "error": f"{type(error).__name__}:{error}",
                }
                self._record("video_close_failed", video_summary)
            final_state = self._safe_read()
            summary = {
                "schema_version": "conveyorvla-waypoint-rollout-summary-v2",
                "execution_mode": (
                    "waypoint_staged" if self.stop_after_route else "waypoint_autonomous"
                ),
                "success": success,
                "pure_physics_success": pure_physics_success,
                "failure_reason": failure_reason,
                "final_state": "DONE" if success else "FAILED",
                "state_trace": self._state_trace,
                "query_count": self._query_count,
                "control_steps": self._control_steps,
                "duration_s": time.time() - started,
                "trace_path": str(trace_path),
                "video": video_summary,
                "final_observation": (
                    None if final_state is None else _state_snapshot(final_state)
                ),
                "model_inputs": ["instruction", "head[t-0.20,t]", "wrist[t-0.20,t]"],
                "model_state_fields": 0,
                "route_owner": "Qwen Pass 1",
                "external_fsm_used": False,
                "navigation_safety_profile": self.navigation.config.safety_profile,
                "initial_source_visibility": self._initial_source_visibility,
            }
            summary_path.write_text(
                json.dumps(_jsonable(summary), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._record("episode_summary", summary)
            if self._trace_stream is not None:
                self._trace_stream.close()
                self._trace_stream = None
            if self.close_simulation_on_exit:
                self.simulation.close()
        return summary

    def _prepare_episode(self) -> None:
        prepare = getattr(self.simulation, "prepare_episode", None)
        if callable(prepare):
            report = dict(prepare(self.episode_spec))
        else:
            self.simulation.build(self.episode_spec)
            report = {"stage_built": True}
        self._record("stage_prepared", report)
        self.simulation.reset(self.episode_spec, seed=self.episode_seed)
        self._record("episode_reset", {"seed": self.episode_seed})
        settings = self.config.manipulation
        if not settings.settle_object_before_navigation:
            for _ in range(self.frames.separation_steps + 1):
                self._physical_step(self._hold_action("initial_visual_warmup"))
            self._check_initial_source_visibility()
            return
        begin = getattr(self.simulation, "begin_object_settle", None)
        finalize = getattr(self.simulation, "finalize_object_settle", None)
        if not callable(begin) or not callable(finalize):
            raise RuntimeError("approved runtime does not support object settling")
        begin_report = dict(begin(self.episode_spec))
        if begin_report.get("applied") is not True:
            raise RuntimeError(f"object settle did not start: {begin_report}")
        self._record("object_settle_started", begin_report)
        self.simulation.apply(
            self.RobotAction(
                source="waypoint_initialization_audit",
                metadata={
                    "skip_physics_step": True,
                    "skip_reason": "audit_post_reset_before_waypoint_rollout",
                },
            )
        )
        stable_steps = 0
        for elapsed in range(1, int(settings.object_settle_max_steps) + 1):
            state = self.simulation.read()
            object_velocity = state.object_velocity
            if object_velocity is None or len(object_velocity) < 6:
                raise RuntimeError("object settle velocity is unavailable")
            object_linear = _norm(object_velocity[:3])
            object_angular = _norm(object_velocity[3:6])
            expected_pose = self.episode_spec.object_initial_pose
            displacement = (
                0.0
                if expected_pose is None or state.object_pose is None
                else _norm(
                    tuple(
                        float(state.object_pose[index]) - float(expected_pose[index])
                        for index in range(3)
                    )
                )
            )
            if displacement > settings.object_settle_max_displacement_m:
                raise RuntimeError(
                    "object settle exceeded its displacement gate: "
                    f"{displacement:.6f} m"
                )
            base_linear = _norm(state.robot_root_velocity[:3])
            base_angular = _norm(state.robot_root_velocity[3:6])
            roll, pitch = _roll_pitch(state.robot_root_pose[3:7])
            base_stable = bool(
                base_linear <= settings.base_settle_linear_velocity_mps
                and base_angular <= settings.base_settle_angular_velocity_rps
                and abs(roll) <= settings.base_settle_max_tilt_rad
                and abs(pitch) <= settings.base_settle_max_tilt_rad
            )
            locked = bool(
                settings.settle_base_before_navigation
                and elapsed <= settings.initialization_base_lock_steps
            )
            stable = bool(
                object_linear <= settings.object_settle_linear_velocity_mps
                and object_angular <= settings.object_settle_angular_velocity_rps
                and (not settings.settle_base_before_navigation or base_stable)
                and not locked
            )
            stable_steps = stable_steps + 1 if stable else 0
            if stable_steps >= settings.object_settle_required_stable_steps:
                final_report = dict(finalize(self.episode_spec))
                if final_report.get("applied") is not True:
                    raise RuntimeError(f"object settle did not finalize: {final_report}")
                self._record(
                    "object_settle_completed",
                    {
                        "elapsed_steps": elapsed,
                        "stable_steps": stable_steps,
                        "displacement_m": displacement,
                        "finalize": final_report,
                    },
                )
                self._check_initial_source_visibility()
                return
            self._physical_step(
                self.RobotAction(
                    source="waypoint_object_settle",
                    metadata={
                        "object_settle_active": True,
                        "manipulation_base_lock": locked,
                        "manipulation_support_joint_lock": bool(
                            locked or settings.initialization_base_lock_steps <= 0
                        ),
                    },
                )
            )
        raise RuntimeError("object/base settle gate timed out")

    def _check_initial_source_visibility(self) -> None:
        state = self.simulation.read()
        report = _initial_source_front_sector_report(
            state.robot_root_pose,
            state.object_pose,
            state.camera_images,
            max_bearing_deg=self.initial_source_max_bearing_deg,
        )
        report["required"] = self.require_initial_source_visible
        self._initial_source_visibility = report
        self._record("initial_source_visibility_preflight", report)
        if self.require_initial_source_visible and not report["passed"]:
            raise RuntimeError(
                "initial_source_not_front_visible:"
                f"bearing_deg={report['bearing_deg']:.3f}:"
                f"front_rgb_present={report['front_rgb_present']}"
            )

    def _next_request(self) -> tuple[Any, Any]:
        for _ in range(max(100, 4 * self.frames.separation_steps)):
            pair = self.frames.pair_after(self._last_query_camera_step)
            if pair is not None:
                current_step = pair[1].step_index
                state = self._camera_states.get(current_step)
                if state is None:
                    raise RuntimeError("camera pair is not bound to executor state")
                self._last_query_camera_step = current_step
                request = waypoint_request_from_frames(
                    episode_id=str(self.episode_spec.episode_id),
                    sequence_id=self._query_count,
                    instruction=str(self.episode_spec.instruction),
                    frames=pair,
                    camera_calibration_id=CAMERA_CALIBRATION_ID,
                )
                return request, state
            self._physical_step(self._hold_action("visual_query_wait"))
        raise RuntimeError("could not acquire exact t-0.20,t head/wrist frames")

    def _execute_navigation(self, response: Any, query_state: Any) -> tuple[bool, str]:
        planned = self.navigation.begin(
            response,
            query_state.robot_root_pose,
            now_s=float(query_state.timestamp),
        )
        self._record("navigation_begin", asdict(planned))
        if planned.failed:
            return False, planned.reason or "navigation_begin_failed"
        self._physical_step(self._robot_action(planned, response.route, response.sequence_id))
        local_map = self._local_map(response.route)
        while True:
            state = self.simulation.read()
            command = self.navigation.step(
                state.robot_root_pose,
                measured_body_velocity(state),
                local_map,
                now_s=float(state.timestamp),
            )
            if command.failed:
                self._record("navigation_failed", asdict(command))
                return False, command.reason or "navigation_execution_failed"
            if command.requires_requery:
                self._record("navigation_requery", asdict(command))
                return True, command.reason or "navigation_requery"
            self._physical_step(
                self._robot_action(command, response.route, response.sequence_id)
            )

    def _execute_manipulation(self, response: Any, query_state: Any) -> tuple[bool, str]:
        self._settle_base_for_model_route(response.route)
        if response.route == WaypointRoute.PICK.value and not self._prepared_for_pick:
            prepare = getattr(self.simulation, "prepare_object_for_pick", None)
            if not callable(prepare):
                return False, "runtime_missing_prepare_object_for_pick"
            report = dict(prepare(self.episode_spec))
            self._record("object_prepared_for_model_pick", report)
            if report.get("applied") is not True:
                return False, "object_prepare_for_pick_failed"
            self._prepared_for_pick = True
        state = self.simulation.read()
        if state.tcp_pose is None:
            return False, "current_tcp_pose_unavailable"
        expected_names = tuple(str(name) for name in self.simulation._config.arm_joint_names)
        joints = measured_arm_joints(
            state.metadata.get("joint_names", ()),
            state.joint_positions,
            expected_names,
        )
        current_tcp_query = tcp_pose_in_query_base(
            query_state.robot_root_pose,
            state.tcp_pose,
        )
        scene = self._collision_scene(response.route, query_state.robot_root_pose, state)
        planned = self.manipulation.begin(
            response,
            current_tcp_query,
            joints,
            scene,
            now_s=float(state.timestamp),
        )
        self._record(
            "manipulation_begin",
            {**asdict(planned), "collision_scene": scene},
        )
        if planned.failed:
            return False, planned.reason or "manipulation_begin_failed"
        self._physical_step(self._robot_action(planned, response.route, response.sequence_id))
        while True:
            state = self.simulation.read()
            joints = measured_arm_joints(
                state.metadata.get("joint_names", ()),
                state.joint_positions,
                expected_names,
            )
            command = self.manipulation.step(joints, now_s=float(state.timestamp))
            if command.failed:
                self._record("manipulation_failed", asdict(command))
                return False, command.reason or "manipulation_execution_failed"
            if command.requires_requery:
                self._record("manipulation_requery", asdict(command))
                return True, command.reason or "manipulation_requery"
            self._physical_step(
                self._robot_action(command, response.route, response.sequence_id)
            )

    def _settle_base_for_model_route(self, route: str) -> None:
        if (
            not self._navigation_since_arm_settle
            and self._last_settled_arm_route == route
        ):
            return
        settings = self.config.manipulation
        steps = (
            settings.base_lock_settle_steps
            if route == WaypointRoute.PICK.value
            else settings.place_base_lock_settle_steps
        )
        for _ in range(int(steps)):
            self._physical_step(
                self.RobotAction(
                    arm_joint_positions=self._held_arm_target,
                    gripper_command=self._held_gripper_command,
                    source="waypoint_model_route_base_settle",
                    metadata={
                        "manipulation_base_lock": True,
                        "manipulation_support_joint_lock": bool(
                            settings.lock_support_joints_during_manipulation
                        ),
                        "model_route": route,
                    },
                )
            )
        self._last_settled_arm_route = route
        self._navigation_since_arm_settle = False

    def _collision_scene(
        self,
        route: str,
        query_root_pose: Sequence[float],
        state: Any,
    ) -> dict[str, Any]:
        import omni.usd
        import numpy as np

        object_path = self.episode_spec.object_prim_path
        if not object_path:
            raise RuntimeError("episode has no object prim for cuRobo collision export")
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage is unavailable for cuRobo collision export")
        world_from_planner, planner_source = self.simulation._read_body_matrix(
            "arm_base_link"
        )
        bbox = self.simulation._compute_object_bbox(stage, object_path)
        # This collision profile is selected only from the current model route;
        # no simulator phase or external state machine participates in dispatch.
        collision_profile = (
            "pick" if route == WaypointRoute.PICK.value else "place"
        )
        cuboids = self.simulation._export_current_world_collision_cuboids(
            stage=stage,
            episode_spec=self.episode_spec,
            phase=collision_profile,
            robot_root_path=self.simulation._robot_prim_path(),
            object_prim_path=object_path,
            T_world_base=world_from_planner,
            object_bbox_center=np.asarray(bbox["center_xyz"], dtype=float),
        )
        return {
            "frame": "curobo-planner-base",
            "planner_base_from_query_base": planner_base_from_query_base(
                world_from_planner, query_root_pose
            ),
            "cuboids_base": cuboids,
            "executor_provenance": {
                "planner_frame_source": str(planner_source),
                "model_route": route,
                "collision_profile_from_model_route": collision_profile,
                "query_step_index": int(state.step_index),
                "ground_truth_target_used": False,
            },
        }

    def _local_map(self, route: str) -> dict[str, Any]:
        executor = self.reference_navigation_executor
        raw = (
            executor._carry_single_floor_raw_map
            if route == WaypointRoute.NAV_TO_TARGET.value
            else executor._single_floor_raw_map
        )
        if raw is None:
            raise RuntimeError("approved DWA executor has no route-local PCT map")
        grid = raw.inflate(float(executor.local_clearance_radius))
        return {"grid_map": grid, "raw_grid_map": raw}

    def _robot_action(self, command: Any, route: str, sequence_id: int) -> Any:
        if command.arm_joint_target is not None:
            self._held_arm_target = tuple(float(value) for value in command.arm_joint_target)
        if command.gripper_target is not None:
            self._held_gripper_command = (
                "open" if float(command.gripper_target) >= 0.5 else "close"
            )
        manipulation = route in {WaypointRoute.PICK.value, WaypointRoute.PLACE.value}
        return self.RobotAction(
            base_velocity=tuple(float(value) for value in command.base_velocity),
            arm_joint_positions=self._held_arm_target,
            gripper_command=self._held_gripper_command,
            source=f"waypoint_{route.lower()}",
            metadata={
                "waypoint_policy": True,
                "model_route": route,
                "model_sequence_id": int(sequence_id),
                "executor_status": command.status,
                "manipulation_base_lock": manipulation,
                "manipulation_support_joint_lock": bool(
                    manipulation
                    and self.config.manipulation.lock_support_joints_during_manipulation
                ),
            },
        )

    def _hold_action(self, source: str) -> Any:
        manipulation = self._active_route in {
            WaypointRoute.PICK.value,
            WaypointRoute.PLACE.value,
        }
        return self.RobotAction(
            arm_joint_positions=self._held_arm_target,
            gripper_command=self._held_gripper_command,
            source=source,
            metadata={
                "waypoint_policy": True,
                "model_route": self._active_route,
                "manipulation_base_lock": manipulation,
                "manipulation_support_joint_lock": bool(
                    manipulation
                    and self.config.manipulation.lock_support_joints_during_manipulation
                ),
            },
        )

    def _physical_step(self, action: Any) -> Any:
        if self._control_steps >= self.max_control_steps:
            raise RuntimeError("waypoint rollout control-step watchdog expired")
        self.simulation.apply(action)
        self.simulation.step(render=self.config.render)
        state = self.simulation.read()
        self._control_steps += 1
        if self.frames.add(int(state.step_index), state.camera_images):
            self._camera_states[int(state.step_index)] = state
            valid_steps = set(self.frames.step_indices)
            self._camera_states = {
                step: value
                for step, value in self._camera_states.items()
                if step in valid_steps
            }
        if self._video is not None:
            self._video.add_frame(
                state=self._active_route,
                timestamp=state.timestamp,
                step_index=self._control_steps,
                camera_images=state.camera_images,
                robot_root_pose=state.robot_root_pose,
            )
        self._record(
            "control_step",
            {
                "control_step": self._control_steps,
                "model_route": self._active_route,
                "action": {
                    "source": action.source,
                    "base_velocity": action.base_velocity,
                    "arm_joint_positions": action.arm_joint_positions,
                    "gripper_command": action.gripper_command,
                    "metadata": action.metadata,
                },
                "state": _state_snapshot(state),
            },
        )
        return state

    def _verify_done(self) -> tuple[bool, dict[str, Any]]:
        stable = 0
        latest = self.simulation.read()
        for _ in range(25):
            latest = self._physical_step(self._hold_action("waypoint_done_verification"))
            velocity = latest.object_velocity
            if (
                velocity is not None
                and _norm(velocity[:3]) <= 0.05
                and _norm(velocity[3:6]) <= 0.50
            ):
                stable += 1
            else:
                stable = 0
        target = self.episode_spec.place_target_pose
        if target is None or latest.object_pose is None or latest.object_velocity is None:
            return False, {
                "verified": False,
                "reason": "place target/object state unavailable",
            }
        xy_error = math.hypot(
            float(latest.object_pose[0]) - float(target[0]),
            float(latest.object_pose[1]) - float(target[1]),
        )
        z_error = abs(float(latest.object_pose[2]) - float(target[2]))
        linear_speed = _norm(latest.object_velocity[:3])
        angular_speed = _norm(latest.object_velocity[3:6])
        verified = bool(
            xy_error <= 0.06
            and z_error <= 0.03
            and linear_speed <= 0.05
            and angular_speed <= 0.50
            and stable >= 20
        )
        return verified, {
            "verified": verified,
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "object_linear_speed_mps": linear_speed,
            "object_angular_speed_rps": angular_speed,
            "stable_steps": stable,
            "target_source": "executor_only_episode_place_target",
            "target_sent_to_model": False,
        }

    def _start_video(self) -> None:
        if not self.config.video.enabled:
            return
        from source.recording.overview_video_recorder import OverviewVideoRecorder

        self._video = OverviewVideoRecorder(
            settings=self.config.video,
            episode_dir=self.episode_dir,
            episode_id=self.episode_spec.episode_id,
            auto_switch_camera=self.config.video.overview_camera_mode in {"auto", "fixed"},
            save_overview_images=bool(
                self.config.recording.enabled and self.config.recording.save_raw_images
            ),
            overview_image_fps=float(self.config.recording.dataset_fps),
            overview_jpeg_quality=int(self.config.recording.jpeg_quality),
        )
        self._video.start_episode()

    def _close_video(self, status: str) -> Any:
        if self._video is None:
            return None
        try:
            return self._video.close(status=status)
        finally:
            self._video = None

    def _apply_final_hold(self) -> None:
        state = self._safe_read()
        if state is None:
            return
        self.simulation.apply(self._hold_action("waypoint_final_hold"))

    def _safe_read(self) -> Any | None:
        try:
            return self.simulation.read()
        except Exception:
            return None

    def _record(self, event: str, payload: Mapping[str, Any]) -> None:
        if self._trace_stream is None:
            return
        row = {
            "event": event,
            "timestamp_unix_s": time.time(),
            **dict(payload),
        }
        self._trace_stream.write(
            json.dumps(_jsonable(row), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        self._trace_stream.flush()


def _simulation_curobo_safety_gate(
    request: Mapping[str, Any], response: Mapping[str, Any]
) -> bool:
    try:
        position_error = float(response.get("target_position_error_m", math.inf))
        orientation_error = float(
            response.get("target_orientation_error_rad", math.inf)
        )
    except (TypeError, ValueError):
        return False
    return bool(
        request.get("deployment") == "simulation"
        and request.get("target_frame") == "query-base-B_t"
        and response.get("reachable") is True
        and response.get("collision_free") is True
        and math.isfinite(position_error)
        and position_error <= 0.02
        and math.isfinite(orientation_error)
        and orientation_error <= 0.10
    )


def _initial_source_front_sector_report(
    robot_root_pose: Sequence[Any],
    object_pose: Sequence[Any] | None,
    camera_images: Mapping[str, Any],
    *,
    max_bearing_deg: float,
) -> dict[str, Any]:
    if object_pose is None or len(robot_root_pose) != 7 or len(object_pose) < 3:
        raise RuntimeError("initial source visibility requires robot and object poses")
    root = tuple(float(value) for value in robot_root_pose)
    source = tuple(float(value) for value in object_pose[:3])
    if not all(math.isfinite(value) for value in (*root, *source)):
        raise RuntimeError("initial source visibility poses are non-finite")
    w, x, y, z = root[3:7]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise RuntimeError("initial source visibility base quaternion is invalid")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    dx, dy = source[0] - root[0], source[1] - root[1]
    body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    bearing_deg = math.degrees(math.atan2(body_y, body_x))
    front = camera_images.get("front")
    shape = tuple(int(value) for value in getattr(front, "shape", ()))
    front_rgb_present = bool(front is not None and len(shape) >= 2 and min(shape[:2]) > 0)
    passed = bool(
        front_rgb_present
        and body_x > 0.0
        and abs(bearing_deg) <= float(max_bearing_deg)
    )
    return {
        "passed": passed,
        "bearing_deg": bearing_deg,
        "max_bearing_deg": float(max_bearing_deg),
        "distance_m": math.hypot(body_x, body_y),
        "source_body_xy_m": [body_x, body_y],
        "front_rgb_present": front_rgb_present,
        "front_rgb_shape": list(shape),
        "gate_basis": "settled_gt_bearing_plus_front_rgb_present",
        "source_truth_sent_to_model": False,
    }


def _state_snapshot(state: Any) -> dict[str, Any]:
    return {
        "step_index": int(state.step_index),
        "timestamp": float(state.timestamp),
        "robot_root_pose": list(state.robot_root_pose),
        "robot_root_velocity": list(state.robot_root_velocity),
        "tcp_pose": None if state.tcp_pose is None else list(state.tcp_pose),
        "object_pose": None if state.object_pose is None else list(state.object_pose),
        "object_velocity": (
            None if state.object_velocity is None else list(state.object_velocity)
        ),
        "joint_positions": list(state.joint_positions),
        "joint_velocities": list(state.joint_velocities),
        "joint_names": list(state.metadata.get("joint_names", ())),
        "camera_keys": sorted(state.camera_images),
    }


def _norm(values: Sequence[Any]) -> float:
    result = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isfinite(result):
        raise RuntimeError("executor state norm is non-finite")
    return result


def _roll_pitch(quaternion: Sequence[Any]) -> tuple[float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise RuntimeError("base quaternion is invalid")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _reference_identity(root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("approved arm-vla reference worktree must be clean")
    if commit != APPROVED_ARM_VLA_COMMIT:
        raise RuntimeError(
            f"arm-vla reference must be {APPROVED_ARM_VLA_COMMIT}, got {commit}"
        )
    return commit


def _load_reference_main(reference_root: Path) -> Any:
    path = reference_root / "scripts" / "pipeline" / "run_full_physics_pipeline.py"
    spec = importlib.util.spec_from_file_location("waypoint_arm_vla_pipeline_main", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load approved arm-vla pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw == ["--help"]:
        build_parser().print_help()
        print("\nPass approved arm-vla arguments after a literal -- separator.")
        return 0
    if "--" not in raw:
        raise SystemExit("separate waypoint and approved arm-vla arguments with --")
    separator = raw.index("--")
    args = build_parser().parse_args(raw[:separator])
    reference_args = raw[separator + 1 :]
    reference_root = args.reference_root.expanduser().resolve()
    _reference_identity(reference_root)
    sys.path.insert(0, str(reference_root))
    parsed_endpoint = urllib.parse.urlparse(args.model_endpoint)
    if parsed_endpoint.port is None:
        raise SystemExit("--model-endpoint must include an explicit loopback port")
    if any(
        flag in reference_args
        for flag in (
            "--remote-vla-eval",
            "--dry-run",
            "--simulation-smoke",
            "--navigation-smoke",
            "--full-physics",
        )
    ):
        raise SystemExit("rollout wrapper owns the approved arm-vla execution mode")

    import source.evaluation.factory as evaluation_factory
    import source.manipulation as reference_manipulation

    reference_manipulation.CuroboPlannerServerProcess = functools.partial(
        _ExternalWaypointCuRoboLifecycle,
        port=args.curobo_port,
        timeout_s=args.curobo_timeout_s,
    )

    def create_waypoint_pipeline(
        *,
        config: Any,
        episode_spec: Any,
        episode_seed: int,
        episode_dir: str | Path,
        simulation: Any,
        endpoint: str,
        connect_timeout_s: float,
        response_timeout_s: float,
        jpeg_quality: int,
        max_replans_per_navigation: int,
        max_chunk_execution_steps: int,
        arm_mode: str,
        close_simulation_on_exit: bool = True,
        first_waypoint_only: bool = False,
        arm_gate: Any = None,
    ) -> WaypointRolloutPipeline:
        del (
            connect_timeout_s,
            response_timeout_s,
            max_replans_per_navigation,
            arm_mode,
            first_waypoint_only,
        )
        if endpoint != f"ws://127.0.0.1:{parsed_endpoint.port}":
            raise RuntimeError("approved arm-vla endpoint binding was changed")
        if arm_gate is not None:
            raise RuntimeError("waypoint rollout does not permit an external route gate")
        return WaypointRolloutPipeline(
            config=config,
            episode_spec=episode_spec,
            episode_seed=episode_seed,
            episode_dir=episode_dir,
            simulation=simulation,
            model_endpoint=args.model_endpoint,
            model_timeout_s=args.model_timeout_s,
            jpeg_quality=jpeg_quality,
            curobo_port=args.curobo_port,
            curobo_timeout_s=args.curobo_timeout_s,
            max_queries=args.max_queries,
            max_control_steps=args.max_control_steps,
            stop_after_route=args.stop_after_route,
            required_first_route=args.required_first_route,
            navigation_safety_profile=args.navigation_safety_profile,
            navigation_max_chunk_steps=max_chunk_execution_steps,
            require_initial_source_visible=args.require_initial_source_visible,
            initial_source_max_bearing_deg=args.initial_source_max_bearing_deg,
            close_simulation_on_exit=close_simulation_on_exit,
        )

    evaluation_factory.create_remote_vla_evaluation_pipeline = create_waypoint_pipeline
    reference_main = _load_reference_main(reference_root)
    return int(
        reference_main.main(
            [
                *reference_args,
                "--remote-vla-eval",
                "--vla-endpoint",
                f"ws://127.0.0.1:{parsed_endpoint.port}",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
