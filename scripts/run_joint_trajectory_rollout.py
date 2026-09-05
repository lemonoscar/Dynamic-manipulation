#!/usr/bin/env python3
"""Play Joint-Trajectory v1 in the approved arm-vla Isaac runtime."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import run_waypoint_rollout as waypoint_runner  # noqa: E402
from conveyor_bench.conveyorvla.formal_physics import FormalPhysics  # noqa: E402
from conveyor_bench.conveyorvla.joint_trajectory import (  # noqa: E402
    JointTrajectoryDomain,
    JointTrajectoryRoute,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (  # noqa: E402
    DirectJointChunk,
    DirectJointCommand,
    JointTrajectoryRuntimeStep,
    NavigationReference,
    RouteCommitStatus,
)
from conveyor_bench.conveyorvla.joint_trajectory_system import (  # noqa: E402
    IsaacJointActionAdapter,
    IsaacJointTrajectorySystemExecutor,
    IsaacTransferTruthAdapter,
    PCTDWAJointNavigationExecutor,
    PlacementValidArea,
    measured_named_joint_state,
)
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (  # noqa: E402
    APPROVED_ARM_VLA_COMMIT,
    ArmVLADWAControllerAdapter,
    ArmVLAPCTPlannerAdapter,
)
from conveyor_bench.conveyorvla.waypoint_rollout import (  # noqa: E402
    TemporalJPEGBuffer,
)


PROTOCOL = "conveyorvla-joint-trajectory-runtime/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--expected-identity", type=Path, help="frozen formal service identity JSON")
    parser.add_argument("--physics-profile", choices=("source_assisted", "no_grasp_assist"), default="source_assisted")
    parser.add_argument("--diffusion-seed", type=int, default=17)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:18082")
    parser.add_argument("--model-timeout-s", type=float, default=180.0)
    parser.add_argument("--isaac-device", default="cuda:3")
    parser.add_argument("--max-queries", type=int, default=96)
    parser.add_argument("--max-control-steps", type=int, default=12_000)
    parser.add_argument("--exact-replay-template-task", type=Path)
    parser.add_argument("--exact-replay-dataset-summary", type=Path)
    parser.add_argument("--require-initial-source-visible", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-source-max-bearing-deg", type=float, default=30.0)
    return parser


class _RecordedPCTPlanner:
    def __init__(self, planner, record):
        self.planner, self.record = planner, record

    def plan(self, current_world_pose, predicted_world_goal):
        plan = self.planner.plan(current_world_pose, predicted_world_goal)
        self.record("pct_plan_evidence", {"current_world_pose": current_world_pose,
                    "predicted_world_goal": predicted_world_goal,
                    "snapped_goal_world": plan.snapped_goal_world,
                    "snap_distance_m": plan.snap_distance_m, "path_world": plan.path_world})
        return plan


class JointTrajectoryHTTPClient:
    def __init__(self, endpoint: str, *, timeout_s: float) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("joint-trajectory endpoint must be loopback HTTP")
        if parsed.port is None or not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("joint-trajectory endpoint or timeout is invalid")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)

    def health(self) -> Mapping[str, Any]:
        return self._request("GET", "/health", None)

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        wire = self._request("POST", "/infer", payload)
        if wire.get("ok") is not True or not isinstance(wire.get("response"), Mapping):
            raise RuntimeError(f"joint-trajectory model returned an invalid response: {wire}")
        return wire["response"]

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None
    ) -> Mapping[str, Any]:
        raw = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.endpoint + path,
            data=raw,
            method=method,
            headers={} if raw is None else {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"joint-trajectory HTTP {error.code}: {error.read().decode()}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"joint-trajectory request failed: {error}") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("joint-trajectory response must be an object")
        return value


class _NoCuRoboLifecycle:
    """Satisfy the approved remote-pipeline lifecycle without starting IK."""

    def __init__(self, _legacy_config: Any) -> None:
        self.start_report = {
            "requested": False,
            "ready": True,
            "joint_trajectory_direct_joint_mode": True,
            "ik_used": False,
            "curobo_used": False,
        }

    def start(self) -> None:
        self.start_report["started"] = False

    def wait_until_ready(self) -> bool:
        return True

    def close(self) -> None:
        self.start_report["closed"] = True


class JointTrajectoryRolloutPipeline:
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
        max_queries: int,
        max_control_steps: int,
        require_initial_source_visible: bool,
        initial_source_max_bearing_deg: float,
        close_simulation_on_exit: bool,
        expected_identity: Mapping[str, Any] | None = None,
        physics_profile: str = "source_assisted",
        diffusion_seed: int = 17,
    ) -> None:
        from source.interfaces import NavGoal, RobotAction, SimulationState
        from source.navigation.navlib import DWAController
        from source.pipeline.navigation_smoke import create_navigation_components

        self.config = config
        self.episode_spec = episode_spec
        self.episode_seed = int(episode_seed)
        self.episode_dir = Path(episode_dir).expanduser().resolve()
        self.simulation = simulation
        self.expected_identity = expected_identity
        self.diffusion_seed = diffusion_seed
        self.physics = None
        if expected_identity is not None:
            self.physics = FormalPhysics(simulation, physics_profile, self._record)
            self.simulation = simulation = self.physics
        self._inference_wall_s: list[float] = []
        self._saturation_events = [0, 0, 0]
        self._manipulation_chunks = 0
        self.client = JointTrajectoryHTTPClient(model_endpoint, timeout_s=model_timeout_s)
        self.max_queries = int(max_queries)
        self.max_control_steps = int(max_control_steps)
        self.require_initial_source_visible = bool(require_initial_source_visible)
        self.initial_source_max_bearing_deg = float(initial_source_max_bearing_deg)
        self.close_simulation_on_exit = bool(close_simulation_on_exit)
        self.jpeg_quality = int(jpeg_quality)
        self.RobotAction = RobotAction
        if self.max_queries <= 0 or self.max_control_steps <= 0:
            raise ValueError("joint-trajectory rollout limits must be positive")

        planner, reference_executor, _navigation_verifier = create_navigation_components(
            config=config, episode_spec=episode_spec
        )
        self.reference_navigation_executor = reference_executor
        pct = ArmVLAPCTPlannerAdapter(
            planner,
            simulation_state_factory=SimulationState,
            nav_goal_factory=NavGoal,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )
        if expected_identity is not None:
            pct = _RecordedPCTPlanner(pct, self._record)
        dwa = ArmVLADWAControllerAdapter(
            DWAController,
            reference_executor.dwa_config,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )
        self.action_adapter = IsaacJointActionAdapter(RobotAction)
        self.system = IsaacJointTrajectorySystemExecutor(
            simulation,
            self.action_adapter,
            PCTDWAJointNavigationExecutor(pct, dwa),
            render=bool(config.render),
            on_control_tick=self._on_control_tick,
        )
        self.truth = IsaacTransferTruthAdapter(
            PlacementValidArea.from_raw_task(episode_spec.raw_task)
        )
        separation_steps = int(round(0.20 / float(config.navigation.control_dt)))
        self.frames = TemporalJPEGBuffer(
            separation_steps=separation_steps,
            jpeg_quality=self.jpeg_quality,
        )
        self._camera_states: dict[int, Any] = {}
        self._last_query_camera_step: int | None = None
        self._control_steps = 0
        self._query_count = 0
        self._state_trace: list[str] = []
        self._trace_stream: Any | None = None
        self._video: Any | None = None
        self._latest_truth: Any | None = None
        self._last_runtime_step: JointTrajectoryRuntimeStep | None = None
        self._prepared_for_pick = False
        self._initial_source_visibility: Mapping[str, Any] | None = None
        self._model_health: Mapping[str, Any] | None = None

    def run_episode(self) -> Mapping[str, Any]:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.episode_dir / "joint_trajectory_trace.jsonl"
        summary_path = self.episode_dir / "summary.json"
        started = time.time()
        success = False
        failure_reason = "episode_not_started"
        self._trace_stream = trace_path.open("x", encoding="utf-8")
        try:
            self._model_health = self.client.health()
            if (
                self._model_health.get("ok") is not True
                or self._model_health.get("protocol_version") != PROTOCOL
                or (self.expected_identity is None and self._model_health.get("global_step") != 250)
            ):
                raise RuntimeError(f"model health gate failed: {self._model_health}")
            if self.expected_identity is not None:
                for key in ("checkpoint_id", "global_step", "run_kind", "model_contract_id", "weights_sha256",
                            "dataset_manifest_sha256", "normalization_sha256", "normalizer_id", "policy_config_sha256",
                            "source_sha256"):
                    if self.expected_identity.get(key) is None or self._model_health.get(key) != self.expected_identity[key]:
                        raise RuntimeError(f"formal service identity mismatch: {key}")
            self._record("model_health", self._model_health)
            self._start_video()
            self._prepare_episode()
            if self.physics is not None:
                self.physics.arm()
            failure_reason = "query_limit_exhausted"
            while self._query_count < self.max_queries:
                payload, query_state = self._next_request()
                inference_started = time.perf_counter()
                response = self.client.infer(payload)
                self._inference_wall_s.append(time.perf_counter() - inference_started)
                if response.get("request_id") != payload["request_id"] or response.get("sequence_id") != payload["sequence_id"]:
                    raise RuntimeError("model response request/sequence binding differs")
                if self.expected_identity is not None and (response.get("checkpoint_id") != self.expected_identity["checkpoint_id"]
                        or response.get("normalization_sha256") != self.expected_identity["normalization_sha256"]):
                    raise RuntimeError("model response identity changed")
                self._query_count += 1
                step = _runtime_step(response)
                self._last_runtime_step = step
                if step.manipulation is not None:
                    self._manipulation_chunks += 1
                    for axis, count in enumerate((step.manipulation.position_saturation_count,
                                                 step.manipulation.rate_saturation_count,
                                                 step.manipulation.gripper_saturation_count)):
                        self._saturation_events[axis] += count
                self._state_trace.append(
                    "PENDING" if step.committed_route is None else step.committed_route.value
                )
                self._record(
                    "model_query",
                    {
                        "request": {
                            "request_id": payload["request_id"],
                            "episode_id": payload["episode_id"],
                            "sequence_id": payload["sequence_id"],
                            "image_count": 4,
                            "joint_state_dimensions": [6, 6, 1],
                            "truth_fields": 0,
                        },
                        "response": response,
                        "query_anchor": waypoint_runner._state_snapshot(query_state),
                    },
                )
                if (
                    self.expected_identity is None
                    and step.committed_route is JointTrajectoryRoute.PICK
                    and step.pass2_executed
                    and not self._prepared_for_pick
                ):
                    prepare = getattr(self.simulation, "prepare_object_for_pick", None)
                    if not callable(prepare):
                        raise RuntimeError("approved runtime lacks prepare_object_for_pick")
                    report = dict(prepare(self.episode_spec))
                    self._record("object_prepared_for_model_pick", report)
                    if report.get("applied") is not True:
                        raise RuntimeError(f"object prepare for PICK failed: {report}")
                    self._prepared_for_pick = True
                local_map = (
                    self._local_map(step.committed_route)
                    if step.action_domain is JointTrajectoryDomain.NAVIGATION
                    else None
                )
                result = self.system.execute(step, local_map=local_map)
                self._record("system_execution", asdict(result))
                if result.failed:
                    failure_reason = result.reason or "joint_system_execution_failed"
                    break
                if self._latest_truth is not None and self._latest_truth.success.success:
                    success = True
                    failure_reason = ""
                    self._record("transfer_success", asdict(self._latest_truth))
                    break
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
                self._record("final_hold_failed", {"error": f"{type(error).__name__}:{error}"})
            try:
                video = self._close_video("success" if success else "failed")
            except Exception as error:
                video = {"closed": False, "error": f"{type(error).__name__}:{error}"}
            final_state = self._safe_read()
            summary = {
                "schema_version": "conveyorvla-joint-trajectory-rollout-summary-v1",
                "execution_mode": "joint_trajectory_autonomous",
                "success": success,
                "pure_physics_success": False,
                "failure_reason": failure_reason,
                "global_step": None if self._model_health is None else self._model_health.get("global_step"),
                "episode_seed": self.episode_seed,
                "query_count": self._query_count,
                "control_steps": self._control_steps,
                "duration_s": time.time() - started,
                "state_trace": self._state_trace,
                "route_owner": "Qwen Pass 1 with two-observation commit",
                "model_inputs": [
                    "instruction",
                    "head[t-0.20,t]",
                    "wrist[t-0.20,t]",
                    "joint_position/velocity/gripper for Mani Pass 2 only",
                ],
                "navigation": "all 10 points retained; point 10 to approved PCT/DWA; 2.0s window",
                "manipulation": "10 direct joint/gripper points; 0.20s each, ten 50Hz ticks; base exact zero",
                "physics_evidence": None if self.physics is None else self.physics.evidence(),
                "transfer_chain_success": bool(success and self.physics is not None and self.physics.pick
                                               and self.physics.carry and self.physics.release and not self.physics.drop),
                "environment_class": "legacy" if self.physics is None else "IsaacSim5.1_migration_from_Sim6_source",
                "diffusion_seed": self.diffusion_seed,
                "timing": {"inference_freezes_simulation": True, "observation_hz": 5., "action_point_hz": 5.,
                           "control_hz": 50., "inference_wall_s": self._inference_wall_s,
                           "simulation_control_time_s": self._control_steps * .02,
                           "query_rate_per_simulation_s": self._query_count / max(.02, self._control_steps * .02)},
                "predicted_chunk_saturation": {"position_events": self._saturation_events[0],
                      "rate_events": self._saturation_events[1], "gripper_events": self._saturation_events[2],
                      "denominator": self._manipulation_chunks * 70,
                      "rate": None if not self._manipulation_chunks else sum(self._saturation_events)/(self._manipulation_chunks*70)},
                "ik_used": False,
                "curobo_used": False,
                "prefix_selected": False,
                "model_health": self._model_health,
                "latest_truth": None if self._latest_truth is None else asdict(self._latest_truth),
                "initial_source_visibility": self._initial_source_visibility,
                "video": video,
                "trace_path": str(trace_path),
                "final_observation": (
                    None if final_state is None else waypoint_runner._state_snapshot(final_state)
                ),
            }
            summary_path.write_text(
                json.dumps(waypoint_runner._jsonable(summary), ensure_ascii=False, indent=2) + "\n",
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
        report = (
            dict(prepare(self.episode_spec))
            if callable(prepare)
            else {"stage_built": bool(self.simulation.build(self.episode_spec) is None)}
        )
        self._record("stage_prepared", report)
        self.simulation.reset(self.episode_spec, seed=self.episode_seed)
        self._record("episode_reset", {"seed": self.episode_seed})
        settings = self.config.manipulation
        begin = getattr(self.simulation, "begin_object_settle", None)
        finalize = getattr(self.simulation, "finalize_object_settle", None)
        if settings.settle_object_before_navigation:
            if not callable(begin) or not callable(finalize):
                raise RuntimeError("approved runtime does not support object settling")
            begin_report = dict(begin(self.episode_spec))
            if begin_report.get("applied") is not True:
                raise RuntimeError(f"object settle did not start: {begin_report}")
            self._record("object_settle_started", begin_report)
            stable_steps = 0
            for elapsed in range(1, int(settings.object_settle_max_steps) + 1):
                state = self.simulation.read()
                velocity = state.object_velocity
                if velocity is None:
                    raise RuntimeError("object settle velocity is unavailable")
                object_stable = bool(
                    waypoint_runner._norm(velocity[:3])
                    <= settings.object_settle_linear_velocity_mps
                    and waypoint_runner._norm(velocity[3:6])
                    <= settings.object_settle_angular_velocity_rps
                )
                base_stable = bool(
                    waypoint_runner._norm(state.robot_root_velocity[:3])
                    <= settings.base_settle_linear_velocity_mps
                    and waypoint_runner._norm(state.robot_root_velocity[3:6])
                    <= settings.base_settle_angular_velocity_rps
                )
                locked = bool(
                    settings.settle_base_before_navigation
                    and elapsed <= settings.initialization_base_lock_steps
                )
                stable_steps = stable_steps + 1 if object_stable and base_stable and not locked else 0
                if stable_steps >= settings.object_settle_required_stable_steps:
                    final_report = dict(finalize(self.episode_spec))
                    if final_report.get("applied") is not True:
                        raise RuntimeError(f"object settle did not finalize: {final_report}")
                    self._record(
                        "object_settle_completed",
                        {"elapsed_steps": elapsed, "stable_steps": stable_steps, "finalize": final_report},
                    )
                    break
                self._physical_step(
                    self.RobotAction(
                        source="joint_trajectory_object_settle",
                        metadata={
                            "object_settle_active": True,
                            "manipulation_base_lock": locked,
                            "manipulation_support_joint_lock": bool(
                                locked or settings.initialization_base_lock_steps <= 0
                            ),
                        },
                    ),
                    route=None,
                    command_index=None,
                )
            else:
                raise RuntimeError("object/base settle gate timed out")
        for _ in range(self.frames.separation_steps + 1):
            self._physical_step(self._measured_hold("joint_trajectory_initial_visual_warmup"), route=None, command_index=None)
        # With 5Hz rendering, most 50Hz states intentionally have no images.
        # Finish warmup on a real capture tick, not by relabeling an older frame.
        for _ in range(2 * self.frames.separation_steps + 1):
            state = self.simulation.read()
            if {"front", "wrist"}.issubset(state.camera_images) and self.frames.pair_after(None) is not None:
                break
            self._physical_step(self._measured_hold("joint_trajectory_wait_for_capture_tick"), route=None, command_index=None)
        else:
            raise RuntimeError("initial synchronized front/wrist capture unavailable")
        visibility = waypoint_runner._initial_source_front_sector_report(
            state.robot_root_pose,
            state.object_pose,
            state.camera_images,
            max_bearing_deg=self.initial_source_max_bearing_deg,
        )
        visibility["required"] = self.require_initial_source_visible
        visibility["head_view_evidence"] = waypoint_runner._persist_initial_head_view(
            self.episode_dir, state.camera_images["front"], quality=self.jpeg_quality
        )
        self._initial_source_visibility = visibility
        self._record("initial_source_visibility_preflight", visibility)
        if self.require_initial_source_visible and not visibility["passed"]:
            raise RuntimeError(f"initial_source_not_front_visible:{visibility}")

    def _next_request(self) -> tuple[Mapping[str, Any], Any]:
        for _ in range(max(100, 4 * self.frames.separation_steps)):
            pair = self.frames.pair_after(self._last_query_camera_step)
            if pair is not None:
                current_step = pair[1].step_index
                state = self._camera_states.get(current_step)
                if state is None:
                    raise RuntimeError("camera pair is not bound to executor state")
                self._last_query_camera_step = current_step
                joints = measured_named_joint_state(state)
                episode = (f"train-seed-{self.episode_seed:09d}" if self.expected_identity is None
                           else f"formal-{hashlib.sha256(str(self.episode_dir).encode()).hexdigest()[:12]}-{self.episode_seed}-{self.diffusion_seed}")
                payload = {
                    "protocol_version": PROTOCOL,
                    "request_id": f"{episode}-joint-{self._query_count:06d}",
                    "episode_id": episode,
                    "sequence_id": self._query_count,
                    "instruction": str(self.episode_spec.instruction),
                    "head_images": [pair[0].head_jpeg_base64, pair[1].head_jpeg_base64],
                    "wrist_images": [pair[0].wrist_jpeg_base64, pair[1].wrist_jpeg_base64],
                    "joint_position": list(joints.joint_position),
                    "joint_velocity": list(joints.joint_velocity),
                    "gripper_open_fraction": joints.gripper_open_fraction,
                }
                if self.expected_identity is not None:
                    payload["diffusion_seed"] = self.diffusion_seed
                    directory = self.episode_dir / "observations"
                    directory.mkdir(exist_ok=True)
                    artifacts = {}
                    for camera, values in (("head", payload["head_images"]), ("wrist", payload["wrist_images"])):
                        paths = []
                        for frame, value in zip(pair, values, strict=True):
                            path = directory / f"{frame.step_index:08d}_{camera}.jpg"
                            raw = base64.b64decode(value, validate=True)
                            if not path.exists():
                                path.write_bytes(raw)
                            paths.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                                          "step_index": frame.step_index})
                        artifacts[camera] = paths
                    self._record("model_input_artifacts", {"request": {k: v for k, v in payload.items()
                                 if k not in {"head_images", "wrist_images"}}, "images": artifacts})
                return payload, state
            self._physical_step(
                self._measured_hold("joint_trajectory_visual_query_wait"),
                route=None,
                command_index=None,
            )
        raise RuntimeError("could not acquire exact t-0.20,t head/wrist frames")

    def _local_map(self, route: JointTrajectoryRoute | None) -> Mapping[str, Any]:
        executor = self.reference_navigation_executor
        raw = (
            executor._carry_single_floor_raw_map
            if route is JointTrajectoryRoute.NAV_TO_TARGET
            else executor._single_floor_raw_map
        )
        if raw is None:
            raise RuntimeError("approved DWA executor has no route-local PCT map")
        return {
            "grid_map": raw.inflate(float(executor.local_clearance_radius)),
            "raw_grid_map": raw,
        }

    def _measured_hold(self, source: str) -> Any:
        state = self.simulation.read()
        joints = measured_named_joint_state(state)
        command = DirectJointCommand(
            index=0,
            joint_position=joints.joint_position,
            gripper_open_fraction=joints.gripper_open_fraction,
        )
        route = None if self._last_runtime_step is None else self._last_runtime_step.committed_route
        return self.action_adapter.hold(command, route=route, sequence_id=self._query_count, source=source)

    def _physical_step(
        self,
        action: Any,
        *,
        route: JointTrajectoryRoute | None,
        command_index: int | None,
    ) -> Any:
        before = self.simulation.read()
        self.simulation.apply(action)
        self.simulation.step(render=bool(self.config.render))
        after = self.simulation.read()
        self._capture_tick(before, after, action, route, command_index)
        return after

    def _on_control_tick(self, tick: Any) -> None:
        self._capture_tick(
            tick.state_before,
            tick.state_after,
            tick.action,
            tick.route,
            tick.command_index,
        )

    def _capture_tick(
        self,
        before: Any,
        after: Any,
        action: Any,
        route: JointTrajectoryRoute | None,
        command_index: int | None,
    ) -> None:
        if self._control_steps >= self.max_control_steps:
            raise RuntimeError("joint-trajectory rollout control-step watchdog expired")
        self._control_steps += 1
        if self.frames.add(int(after.step_index), after.camera_images):
            self._camera_states[int(after.step_index)] = after
            valid = set(self.frames.step_indices)
            self._camera_states = {key: value for key, value in self._camera_states.items() if key in valid}
        self._latest_truth = self.truth.update(after)
        if self._video is not None:
            self._video.add_frame(
                state="PENDING" if route is None else route.value,
                timestamp=after.timestamp,
                step_index=self._control_steps,
                camera_images=after.camera_images,
                robot_root_pose=after.robot_root_pose,
            )
        self._record(
            "control_step",
            {
                "control_step": self._control_steps,
                "route": None if route is None else route.value,
                "command_index": command_index,
                "action": {
                    "source": action.source,
                    "base_velocity": action.base_velocity,
                    "arm_joint_positions": action.arm_joint_positions,
                    "gripper_command": action.gripper_command,
                    "metadata": action.metadata,
                },
                "truth": asdict(self._latest_truth),
                "state_before_step": int(before.step_index),
                "state_after": waypoint_runner._state_snapshot(after),
            },
        )

    def _start_video(self) -> None:
        if not self.config.video.enabled:
            return
        from source.recording.overview_video_recorder import OverviewVideoRecorder

        self._video = OverviewVideoRecorder(
            settings=(replace(self.config.video, fps=5.) if self.expected_identity is not None
                      else self.config.video),
            episode_dir=self.episode_dir,
            episode_id=self.episode_spec.episode_id,
            auto_switch_camera=self.config.video.overview_camera_mode in {"auto", "fixed"},
            save_overview_images=False,
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
        if self._safe_read() is not None:
            self.simulation.apply(self._measured_hold("joint_trajectory_final_hold"))

    def _safe_read(self) -> Any | None:
        try:
            return self.simulation.read()
        except Exception:
            return None

    def _record(self, event: str, payload: Mapping[str, Any]) -> None:
        if self._trace_stream is None:
            return
        self._trace_stream.write(
            json.dumps(
                waypoint_runner._jsonable(
                    {"event": event, "timestamp_unix_s": time.time(), **dict(payload)}
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._trace_stream.flush()


def _runtime_step(value: Mapping[str, Any]) -> JointTrajectoryRuntimeStep:
    predicted = _route(value.get("predicted_route"))
    committed = _route(value.get("committed_route"))
    domain = None if value.get("action_domain") is None else JointTrajectoryDomain(str(value["action_domain"]))
    status = None if value.get("commit_status") is None else RouteCommitStatus(str(value["commit_status"]))
    navigation = None
    if isinstance(value.get("navigation"), Mapping):
        raw = value["navigation"]
        navigation = NavigationReference(
            points_query_body=tuple(tuple(float(item) for item in row) for row in raw["points_query_body"]),
            local_goal_query_body=tuple(float(item) for item in raw["local_goal_query_body"]),
            stride_s=float(raw["stride_s"]),
        )
    manipulation = None
    if isinstance(value.get("manipulation"), Mapping):
        raw = value["manipulation"]
        manipulation = DirectJointChunk(
            commands=tuple(_command(item) for item in raw["commands"]),
            position_saturation_count=int(raw["position_saturation_count"]),
            rate_saturation_count=int(raw["rate_saturation_count"]),
            gripper_saturation_count=int(raw["gripper_saturation_count"]),
        )
    hold = _command(value["hold"]) if isinstance(value.get("hold"), Mapping) else None
    return JointTrajectoryRuntimeStep(
        request_id=str(value["request_id"]),
        sequence_id=int(value["sequence_id"]),
        predicted_route=predicted,
        committed_route=committed,
        commit_status=status,
        route_probs={str(key): float(item) for key, item in value["route_probs"].items()},
        subtask=str(value["subtask"]),
        action_domain=domain,
        navigation=navigation,
        manipulation=manipulation,
        hold=hold,
        pass2_executed=bool(value["pass2_executed"]),
        checkpoint_id=str(value["checkpoint_id"]),
        normalization_sha256=str(value["normalization_sha256"]),
        elapsed_ms=float(value["elapsed_ms"]),
        recover_reason=value.get("recover_reason"),
    )


def _command(value: Mapping[str, Any]) -> DirectJointCommand:
    return DirectJointCommand(
        index=int(value["index"]),
        joint_position=tuple(float(item) for item in value["joint_position"]),
        gripper_open_fraction=float(value["gripper_open_fraction"]),
        base_velocity=tuple(float(item) for item in value["base_velocity"]),
        duration_s=float(value["duration_s"]),
    )


def _route(value: Any) -> JointTrajectoryRoute | None:
    return None if value is None else JointTrajectoryRoute(str(value))


def _reference_arg_value(arguments: list[str], flag: str) -> str:
    try:
        index = arguments.index(flag)
    except ValueError as error:
        raise SystemExit(f"exact replay requires approved runtime argument {flag}") from error
    if index + 1 >= len(arguments):
        raise SystemExit(f"approved runtime argument {flag} has no value")
    return arguments[index + 1]


def _replace_exact_numbers(value: Any, replacements: tuple[tuple[float, float], ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_exact_numbers(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_exact_numbers(item, replacements) for item in value]
    if isinstance(value, float):
        for old, new in replacements:
            if math.isclose(value, old, rel_tol=0.0, abs_tol=1e-12):
                return new
    return value


def _materialize_exact_replay_task(
    *,
    template_path: Path,
    dataset_summary_path: Path,
    source_task_path: Path,
    output_dir: Path,
) -> Path:
    template = json.loads(template_path.expanduser().resolve().read_text(encoding="utf-8"))
    dataset_summary = json.loads(
        dataset_summary_path.expanduser().resolve().read_text(encoding="utf-8")
    )
    source_task = json.loads(source_task_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not all(isinstance(item, dict) for item in (template, dataset_summary, source_task)):
        raise RuntimeError("exact replay inputs must be JSON objects")
    metadata = dataset_summary.get("episode_metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError("dataset summary has no episode_metadata")
    randomization = metadata.get("randomization")
    requested = (
        randomization.get("requested_resolved")
        if isinstance(randomization, Mapping)
        else None
    )
    sample = requested.get("sample") if isinstance(requested, Mapping) else None
    if not isinstance(sample, Mapping):
        raise RuntimeError("dataset summary has no realized randomization sample")
    if int(metadata.get("seed", -1)) != 170007 or dataset_summary.get("success") is not True:
        raise RuntimeError("exact replay is bound to successful training seed 170007")
    instruction = str(metadata.get("instruction", ""))
    if instruction != str(template.get("instruction", "")):
        raise RuntimeError("template and training episode instructions differ")

    robot = sample.get("robot")
    pick_goal = sample.get("pick_base_goal")
    place_goal = sample.get("place_base_goal")
    if not all(isinstance(item, Mapping) for item in (robot, pick_goal, place_goal)):
        raise RuntimeError("training randomization lacks robot or base-goal truth")
    old_start = template["start"]
    old_pick = template["pick"]["base_goal"]
    old_place = template["place"]["base_goal"]
    replacements = (
        (float(old_start["x"]), float(robot["xyz"][0])),
        (float(old_start["y"]), float(robot["xyz"][1])),
        (float(old_start["z"]), float(robot["xyz"][2])),
        (float(old_start["yaw"]), float(robot["yaw_rad"])),
        (float(old_pick["x"]), float(pick_goal["x"])),
        (float(old_pick["y"]), float(pick_goal["y"])),
        (float(old_pick["z"]), float(pick_goal["z"])),
        (float(old_pick["yaw"]), float(pick_goal["yaw"])),
        (float(old_place["x"]), float(place_goal["x"])),
        (float(old_place["y"]), float(place_goal["y"])),
        (float(old_place["z"]), float(place_goal["z"])),
        (float(old_place["yaw"]), float(place_goal["yaw"])),
    )
    replay = _replace_exact_numbers(template, replacements)
    replay["scene_usd"] = source_task["scene_usd"]
    replay.pop("annotation_config", None)
    replay.pop("annotation_config_report", None)
    replay.pop("scene_asset_binding_runtime", None)
    replay["episode_id"] = 7
    replay.setdefault("notes", {})["joint_trajectory_exact_replay"] = {
        "source_episode_id": dataset_summary["episode_id"],
        "seed": 170007,
        "randomization_reapplied": False,
        "source_summary": str(dataset_summary_path.expanduser().resolve()),
    }
    replay_dir = output_dir.expanduser().resolve() / ".runtime_inputs"
    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_path = replay_dir / "seed170007_exact_training_replay_task.json"
    raw = json.dumps(replay, ensure_ascii=False, indent=2) + "\n"
    replay_path.write_text(raw, encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "exact_training_replay_materialized",
                "path": str(replay_path),
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "seed": 170007,
                "instruction": instruction,
                "start": replay["start"],
                "pick_base_goal": replay["pick"]["base_goal"],
                "place_base_goal": replay["place"]["base_goal"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return replay_path


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw == ["--help"]:
        build_parser().print_help()
        print("\nPass approved arm-vla arguments after a literal -- separator.")
        return 0
    if "--" not in raw:
        raise SystemExit("separate joint-trajectory and approved arm-vla arguments with --")
    separator = raw.index("--")
    args = build_parser().parse_args(raw[:separator])
    reference_args = raw[separator + 1 :]
    reference_root = args.reference_root.expanduser().resolve()
    waypoint_runner._reference_identity(reference_root)
    sys.path.insert(0, str(reference_root))
    if not args.isaac_device.startswith("cuda:"):
        raise SystemExit("--isaac-device must use an explicit cuda:N device")
    try:
        int(args.isaac_device.split(":", 1)[1])
    except ValueError as error:
        raise SystemExit("--isaac-device must use an explicit cuda:N device") from error
    parsed_endpoint = urllib.parse.urlparse(args.model_endpoint)
    if parsed_endpoint.port is None:
        raise SystemExit("--model-endpoint must include an explicit loopback port")
    if any(
        flag in reference_args
        for flag in ("--remote-vla-eval", "--dry-run", "--simulation-smoke", "--navigation-smoke", "--full-physics")
    ):
        raise SystemExit("joint-trajectory wrapper owns the approved arm-vla execution mode")
    replay_flags = (
        args.exact_replay_template_task is not None,
        args.exact_replay_dataset_summary is not None,
    )
    if any(replay_flags) and not all(replay_flags):
        raise SystemExit("exact replay requires both template-task and dataset-summary")
    if all(replay_flags):
        task_index = reference_args.index("--task-json") + 1
        source_task_path = Path(_reference_arg_value(reference_args, "--task-json"))
        output_dir = Path(_reference_arg_value(reference_args, "--output-dir"))
        replay_path = _materialize_exact_replay_task(
            template_path=args.exact_replay_template_task,
            dataset_summary_path=args.exact_replay_dataset_summary,
            source_task_path=source_task_path,
            output_dir=output_dir,
        )
        reference_args[task_index] = str(replay_path)
        for flag in ("--no-randomize-task", "--no-randomize-base-goal"):
            if flag not in reference_args:
                reference_args.append(flag)

    import source.evaluation.factory as evaluation_factory
    import source.manipulation as reference_manipulation
    import source.simulation as reference_simulation
    import isaaclab.app as isaac_app

    reference_manipulation.CuroboPlannerServerProcess = _NoCuRoboLifecycle
    approved_runtime_config = reference_simulation.IsaacLabNavigationRuntimeConfig

    def _device_bound_runtime_config(*config_args: Any, **config_kwargs: Any) -> Any:
        config_kwargs["device"] = args.isaac_device
        if args.expected_identity is not None:
            config_kwargs["enable_verified_grasp_fixed_joint"] = args.physics_profile == "source_assisted"
            config_kwargs["camera_render_interval_control_steps"] = 10
        return approved_runtime_config(*config_args, **config_kwargs)

    reference_simulation.IsaacLabNavigationRuntimeConfig = _device_bound_runtime_config
    approved_app_launcher = isaac_app.AppLauncher

    class _DeviceBoundAppLauncher:
        def __new__(cls, launcher_args: Mapping[str, Any]) -> Any:
            resolved = dict(launcher_args)
            resolved["device"] = args.isaac_device
            return approved_app_launcher(resolved)

    isaac_app.AppLauncher = _DeviceBoundAppLauncher

    def create_joint_pipeline(
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
    ) -> JointTrajectoryRolloutPipeline:
        import isaaclab_tasks.utils.hydra as hydra_utils

        del (
            connect_timeout_s,
            response_timeout_s,
            max_replans_per_navigation,
            max_chunk_execution_steps,
            arm_mode,
            first_waypoint_only,
        )
        if endpoint != f"ws://127.0.0.1:{parsed_endpoint.port}" or arm_gate is not None:
            raise RuntimeError("approved arm-vla endpoint/factory binding changed")
        if not getattr(hydra_utils.hydra_task_config, "_joint_device_bound", False):
            approved_hydra_task_config = hydra_utils.hydra_task_config

            def _device_bound_hydra_task_config(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
                approved_decorator = approved_hydra_task_config(
                    *decorator_args, **decorator_kwargs
                )

                def _decorate(function: Any) -> Any:
                    def _invoke(env_cfg: Any, agent_cfg: Any, *call_args: Any, **call_kwargs: Any) -> Any:
                        env_cfg.sim.device = args.isaac_device
                        agent_cfg.device = args.isaac_device
                        return function(env_cfg, agent_cfg, *call_args, **call_kwargs)

                    _invoke.__name__ = function.__name__
                    return approved_decorator(_invoke)

                return _decorate

            _device_bound_hydra_task_config._joint_device_bound = True  # type: ignore[attr-defined]
            hydra_utils.hydra_task_config = _device_bound_hydra_task_config
        return JointTrajectoryRolloutPipeline(
            config=config,
            episode_spec=episode_spec,
            episode_seed=episode_seed,
            episode_dir=episode_dir,
            simulation=simulation,
            model_endpoint=args.model_endpoint,
            model_timeout_s=args.model_timeout_s,
            jpeg_quality=jpeg_quality,
            max_queries=args.max_queries,
            max_control_steps=args.max_control_steps,
            require_initial_source_visible=args.require_initial_source_visible,
            initial_source_max_bearing_deg=args.initial_source_max_bearing_deg,
            close_simulation_on_exit=close_simulation_on_exit,
            expected_identity=None if args.expected_identity is None else json.loads(args.expected_identity.read_text()),
            physics_profile=args.physics_profile,
            diffusion_seed=args.diffusion_seed,
        )

    evaluation_factory.create_remote_vla_evaluation_pipeline = create_joint_pipeline
    reference_main = waypoint_runner._load_reference_main(reference_root)
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
