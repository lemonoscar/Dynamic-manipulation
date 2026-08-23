"""Fail-closed PCT/DWA and cuRobo/IK receding-horizon executors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    ArmTargetSafety,
    NavWaypointSafety,
    WaypointActionDomain,
    WaypointRoute,
    nav_waypoint_world,
    rank_navigation_waypoints,
    select_navigation_waypoint,
    validate_arm_targets,
    wrap_to_pi,
)
from conveyor_bench.conveyorvla.waypoint_protocol import WaypointResponse


PlanarWorldPose = tuple[float, float, float]
PCTWorldPose = tuple[float, float, float, float]
BaseVelocity = tuple[float, float, float]

NAVIGATION_SAFETY_PROFILE_CONTRACT = "contract"
NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE = "arm-vla-reference"
NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA = "lookahead-arm-vla-reference"
NAVIGATION_SAFETY_PROFILE_EXECUTABLE_PREFIX = "executable-prefix-diagnostic"
NAVIGATION_SAFETY_PROFILE_UNBOUNDED_TRANSLATION = (
    "unbounded-translation-diagnostic"
)
NAVIGATION_SAFETY_PROFILES = (
    NAVIGATION_SAFETY_PROFILE_CONTRACT,
    NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE,
    NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
    NAVIGATION_SAFETY_PROFILE_EXECUTABLE_PREFIX,
    NAVIGATION_SAFETY_PROFILE_UNBOUNDED_TRANSLATION,
)

_RANKED_LOOKAHEAD_PROFILES = (
    NAVIGATION_SAFETY_PROFILE_CONTRACT,
    NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
)
_ARM_VLA_CONTROL_PROFILES = (
    NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE,
    NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
)


@dataclass(frozen=True)
class PCTPlan:
    path_world: tuple[tuple[float, float], ...]
    snapped_goal_world: PCTWorldPose
    snap_distance_m: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PCTPlanner(Protocol):
    def plan(
        self,
        current_world_pose: PCTWorldPose,
        predicted_world_goal: PCTWorldPose,
    ) -> PCTPlan: ...


class DWAController(Protocol):
    def command(
        self,
        path_world: Sequence[Sequence[float]],
        current_world_pose: PlanarWorldPose,
        measured_body_velocity: BaseVelocity,
        local_map: Any,
    ) -> Sequence[float]: ...


@dataclass(frozen=True)
class ArmPlan:
    joint_path: tuple[tuple[float, ...], ...]
    planner: str
    reachable: bool
    collision_free: bool
    target_position_error_m: float
    target_orientation_error_rad: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class CuRoboIKPlanner(Protocol):
    def plan(
        self,
        current_joints: Sequence[float],
        target_tcp_base: Sequence[float],
        scene_collision: Any,
    ) -> ArmPlan: ...


class ArmTrajectoryController(Protocol):
    def reset(self, plan: ArmPlan, gripper_target: float) -> None: ...

    def command(self, measured_joints: Sequence[float]) -> tuple[Sequence[float], bool]: ...


@dataclass(frozen=True)
class ExecutionCommand:
    base_velocity: BaseVelocity = (0.0, 0.0, 0.0)
    arm_joint_target: tuple[float, ...] | None = None
    gripper_target: float | None = None
    status: str = "hold"
    requires_requery: bool = False
    failed: bool = False
    reason: str | None = None
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationExecutionConfig:
    waypoint_safety: NavWaypointSafety = NavWaypointSafety()
    safety_profile: str = NAVIGATION_SAFETY_PROFILE_CONTRACT
    pct_snap_max_m: float = 0.10
    goal_tolerance_m: float = 0.12
    yaw_tolerance_rad: float = 0.14
    trusted_horizon_points: int = 10
    minimum_lookahead_tolerance_factor: float = 2.0
    target_lookahead_m: float = 0.50
    max_abs_vx_mps: float = 0.60
    max_abs_vy_mps: float = 0.40
    max_abs_wz_radps: float = 1.20
    terminal_yaw_kp: float = 1.5
    terminal_yaw_max_radps: float = 0.60
    chunk_timeout_s: float = 12.0
    max_chunk_execution_steps: int = 250
    stow_joint_target: tuple[float, ...] | None = None
    carry_joint_target: tuple[float, ...] | None = None
    open_gripper_target: float = 1.0
    closed_gripper_target: float = 0.0

    def __post_init__(self) -> None:
        if self.safety_profile not in NAVIGATION_SAFETY_PROFILES:
            raise ValueError(
                "navigation safety profile must be one of "
                + ", ".join(NAVIGATION_SAFETY_PROFILES)
            )
        values = (
            self.pct_snap_max_m,
            self.goal_tolerance_m,
            self.yaw_tolerance_rad,
            self.minimum_lookahead_tolerance_factor,
            self.target_lookahead_m,
            self.max_abs_vx_mps,
            self.max_abs_vy_mps,
            self.max_abs_wz_radps,
            self.terminal_yaw_kp,
            self.terminal_yaw_max_radps,
            self.chunk_timeout_s,
            self.open_gripper_target,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("navigation execution limits must be finite and positive")
        if self.max_chunk_execution_steps <= 0:
            raise ValueError("navigation chunk step limit must be positive")
        if (
            isinstance(self.trusted_horizon_points, bool)
            or not 1 <= self.trusted_horizon_points <= ACTION_HORIZON
        ):
            raise ValueError(
                f"trusted_horizon_points must be within [1, {ACTION_HORIZON}]"
            )
        if self.target_lookahead_m < (
            self.minimum_lookahead_tolerance_factor * self.goal_tolerance_m
        ):
            raise ValueError("target lookahead must be at least the minimum lookahead")
        if not math.isfinite(self.closed_gripper_target):
            raise ValueError("closed gripper target must be finite")
        if not 0.0 <= self.closed_gripper_target <= self.open_gripper_target <= 1.0:
            raise ValueError("navigation gripper targets must be ordered within [0,1]")
        for name, target in (
            ("stow_joint_target", self.stow_joint_target),
            ("carry_joint_target", self.carry_joint_target),
        ):
            if target is not None and (
                not target or not all(math.isfinite(float(value)) for value in target)
            ):
                raise ValueError(f"{name} must be a non-empty finite vector")


@dataclass(frozen=True)
class ManipulationExecutionConfig:
    target_safety: ArmTargetSafety = ArmTargetSafety()
    enforce_target_step_limits: bool = True
    chunk_timeout_s: float = 12.0
    max_target_position_error_m: float = 0.02
    max_target_orientation_error_rad: float = 0.10
    max_joint_command_step_rad: float = 0.15
    joint_position_limits: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enforce_target_step_limits, bool):
            raise ValueError("enforce_target_step_limits must be a boolean")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.chunk_timeout_s,
                self.max_target_position_error_m,
                self.max_target_orientation_error_rad,
                self.max_joint_command_step_rad,
            )
        ):
            raise ValueError("manipulation execution limits must be finite and positive")
        if any(
            len(limit) != 2
            or not all(math.isfinite(float(value)) for value in limit)
            or float(limit[0]) >= float(limit[1])
            for limit in self.joint_position_limits
        ):
            raise ValueError("joint position limits must be finite ordered pairs")


class PCTDWARecedingHorizonExecutor:
    """Execute one selected model local goal, then force a new visual query."""

    def __init__(
        self,
        pct_planner: PCTPlanner,
        dwa_controller: DWAController,
        config: NavigationExecutionConfig = NavigationExecutionConfig(),
        stall_detector: Any | None = None,
    ) -> None:
        self.pct_planner = pct_planner
        self.dwa_controller = dwa_controller
        self.config = config
        self.stall_detector = stall_detector
        if (
            self.config.safety_profile in _ARM_VLA_CONTROL_PROFILES
            and self.stall_detector is None
        ):
            raise ValueError("arm-vla control profiles require arm-vla's stall detector")
        self._active: dict[str, Any] | None = None
        self._last_sequence = -1

    def begin(
        self,
        response: WaypointResponse,
        current_base_world: Sequence[float],
        *,
        now_s: float | None = None,
    ) -> ExecutionCommand:
        self._active = None
        now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s")
        if response.sequence_id <= self._last_sequence:
            return _stop("stale_navigation_response", failed=True)
        self._last_sequence = response.sequence_id
        if response.terminal:
            return _stop(
                f"navigation_terminal_route:{response.route}",
                failed=response.route != WaypointRoute.DONE.value,
            )
        if response.action_domain != WaypointActionDomain.NAVIGATION.value:
            return _stop("response_is_not_navigation", failed=True)
        full_horizon_violation: str | None = None
        minimum_lookahead_m: float | None = None
        trusted_horizon_points: int | None = None
        selection_policy = "first-nondegenerate-v1"
        candidate_rejections: list[dict[str, Any]] = []
        try:
            current = _world_pose(current_base_world)
            waypoints, mask = _fixed_navigation_chunk(response)
            if self.config.safety_profile in _RANKED_LOOKAHEAD_PROFILES:
                trusted_horizon_points = (
                    min(
                        response.trusted_prefix_k,
                        self.config.trusted_horizon_points,
                    )
                    if response.trusted_prefix_k is not None
                    else self.config.trusted_horizon_points
                )
                minimum_lookahead_m = (
                    self.config.minimum_lookahead_tolerance_factor
                    * self.config.goal_tolerance_m
                )
                candidates = rank_navigation_waypoints(
                    waypoints,
                    mask,
                    trusted_horizon_points=trusted_horizon_points,
                    minimum_lookahead_m=minimum_lookahead_m,
                    target_lookahead_m=self.config.target_lookahead_m,
                    safety=self.config.waypoint_safety,
                )
                selection_policy = "trusted-prefix-target-lookahead-pct-v1"
            elif (
                self.config.safety_profile
                == NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE
            ):
                original_safety = replace(
                    self.config.waypoint_safety,
                    minimum_translation_m=1.0e-3,
                    minimum_yaw_rad=1.0e-3,
                )
                candidates = (
                    select_navigation_waypoint(
                        waypoints,
                        (mask[0],) + (False,) * (ACTION_HORIZON - 1),
                        safety=original_safety,
                        validate_full_horizon=False,
                    ),
                )
                selection_policy = "arm-vla-reference-first-v1"
            else:
                try:
                    candidate = select_navigation_waypoint(
                        waypoints,
                        mask,
                        safety=self.config.waypoint_safety,
                    )
                except (ValueError, TypeError) as error:
                    full_horizon_violation = str(error)
                    retry_safety = self.config.waypoint_safety
                    if (
                        self.config.safety_profile
                        == NAVIGATION_SAFETY_PROFILE_UNBOUNDED_TRANSLATION
                    ):
                        retry_safety = replace(
                            retry_safety, max_segment_translation_m=math.inf
                        )
                    candidate = select_navigation_waypoint(
                        waypoints,
                        mask,
                        safety=retry_safety,
                        validate_full_horizon=False,
                    )
                candidates = (candidate,)
                selection_policy = "diagnostic-first-nondegenerate-v1"
        except (ValueError, TypeError) as error:
            return _stop(f"navigation_waypoint_rejected:{error}", failed=True)

        selected_index: int | None = None
        selected_body: tuple[float, float, float] | None = None
        predicted_goal: PCTWorldPose | None = None
        plan: PCTPlan | None = None
        mode: str | None = None
        pct_elapsed_ms = 0.0
        for candidate_index, candidate_body in candidates:
            candidate_goal_raw = nav_waypoint_world(
                (current[0], current[1], current[2], *_yaw_quaternion(current[3])),
                candidate_body,
            )
            candidate_goal: PCTWorldPose = (
                candidate_goal_raw[0],
                candidate_goal_raw[1],
                current[2],
                candidate_goal_raw[2],
            )
            translation = math.hypot(candidate_body[0], candidate_body[1])
            candidate_mode = (
                "terminal_yaw"
                if (
                    translation < self.config.waypoint_safety.minimum_translation_m
                    or (
                        self.config.safety_profile
                        != NAVIGATION_SAFETY_PROFILE_CONTRACT
                        and translation <= self.config.goal_tolerance_m
                    )
                )
                else "pct_dwa"
            )
            candidate_plan: PCTPlan | None = None
            candidate_elapsed_ms = 0.0
            if candidate_mode == "pct_dwa":
                try:
                    planned_at = time.perf_counter()
                    candidate_plan = self.pct_planner.plan(current, candidate_goal)
                    _validate_pct_plan(
                        candidate_plan,
                        candidate_goal,
                        (
                            math.inf
                            if self.config.safety_profile
                            in _ARM_VLA_CONTROL_PROFILES
                            else self.config.pct_snap_max_m
                        ),
                    )
                    candidate_elapsed_ms = (
                        time.perf_counter() - planned_at
                    ) * 1000.0
                except Exception as error:
                    if self.config.safety_profile not in _RANKED_LOOKAHEAD_PROFILES:
                        return _stop(
                            f"pct_plan_failed:{type(error).__name__}:{error}",
                            failed=True,
                        )
                    candidate_rejections.append(
                        {
                            "index": candidate_index,
                            "reason": f"{type(error).__name__}:{error}",
                        }
                    )
                    continue
            selected_index = candidate_index
            selected_body = candidate_body
            predicted_goal = candidate_goal
            plan = candidate_plan
            mode = candidate_mode
            pct_elapsed_ms = candidate_elapsed_ms
            break
        if (
            selected_index is None
            or selected_body is None
            or predicted_goal is None
            or mode is None
        ):
            return _stop(
                "pct_candidates_exhausted",
                failed=True,
                trace={
                    "selection_policy": selection_policy,
                    "ranked_waypoint_indices": [index for index, _row in candidates],
                    "pct_candidate_rejections": candidate_rejections,
                },
            )
        self._active = {
            "request_id": response.request_id,
            "sequence_id": response.sequence_id,
            "route": response.route,
            "execution_mode": (
                "stow_open"
                if response.route == WaypointRoute.NAV_TO_SOURCE.value
                else "carry_closed"
            ),
            "safety_profile": self.config.safety_profile,
            "arm_vla_reference_rules": (
                self.config.safety_profile in _ARM_VLA_CONTROL_PROFILES
            ),
            "translation_limit_disabled": (
                self.config.safety_profile
                == NAVIGATION_SAFETY_PROFILE_UNBOUNDED_TRANSLATION
            ),
            "full_horizon_contract_passed": (
                None
                if self.config.safety_profile
                in (
                    NAVIGATION_SAFETY_PROFILE_CONTRACT,
                    NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE,
                    NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA,
                )
                else full_horizon_violation is None
            ),
            "full_horizon_violation": full_horizon_violation,
            "selection_policy": selection_policy,
            "trusted_horizon_points": trusted_horizon_points,
            "model_trusted_prefix_k": response.trusted_prefix_k,
            "minimum_lookahead_m": minimum_lookahead_m,
            "target_lookahead_m": (
                self.config.target_lookahead_m
                if self.config.safety_profile in _RANKED_LOOKAHEAD_PROFILES
                else None
            ),
            "ranked_indices": [index for index, _row in candidates],
            "candidate_rejections": candidate_rejections,
            "selected_index": selected_index,
            "selected_body": selected_body,
            "predicted_goal": predicted_goal,
            "plan": plan,
            "pct_elapsed_ms": pct_elapsed_ms,
            "mode": mode,
            "started_s": now,
            "chunk_step_count": 0,
            "stall_diagnostics": None,
        }
        if self.config.safety_profile in _ARM_VLA_CONTROL_PROFILES:
            reset_dwa = getattr(self.dwa_controller, "reset", None)
            if callable(reset_dwa):
                reset_dwa()
        if self.stall_detector is not None:
            self.stall_detector.reset()
        arm_target, gripper_target = self._posture()
        return ExecutionCommand(
            arm_joint_target=arm_target,
            gripper_target=gripper_target,
            status="navigation_chunk_planned",
            trace=self.status(),
        )

    def step(
        self,
        current_base_world: Sequence[float],
        measured_body_velocity: Sequence[float],
        local_map: Any,
        *,
        now_s: float | None = None,
    ) -> ExecutionCommand:
        if self._active is None:
            return _stop("no_active_navigation_chunk", failed=True)
        now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s")
        active = self._active
        current = _world_pose(current_base_world)
        velocity = _velocity(measured_body_velocity)
        goal: PCTWorldPose = active["predicted_goal"]
        distance = math.hypot(goal[0] - current[0], goal[1] - current[1])
        yaw_error = abs(wrap_to_pi(goal[3] - current[3]))
        active["chunk_step_count"] += 1
        arm_vla_reference = self.config.safety_profile in _ARM_VLA_CONTROL_PROFILES
        if (
            arm_vla_reference
            and active["chunk_step_count"] > self.config.max_chunk_execution_steps
        ):
            return self._finish("navigation_chunk_timeout", failed=False)
        if (
            not arm_vla_reference
            and now - active["started_s"] > self.config.chunk_timeout_s
        ):
            return self._finish("navigation_chunk_timeout", failed=True)
        if distance <= self.config.goal_tolerance_m and yaw_error <= self.config.yaw_tolerance_rad:
            return self._finish(
                (
                    "first_waypoint_reached"
                    if active["selection_policy"]
                    == "arm-vla-reference-first-v1"
                    else "selected_waypoint_reached"
                ),
                failed=False,
            )
        terminal_yaw_active = bool(
            active["mode"] == "terminal_yaw"
            or distance <= self.config.goal_tolerance_m
        )
        if terminal_yaw_active:
            command = (
                0.0,
                0.0,
                _clamp(
                    self.config.terminal_yaw_kp * wrap_to_pi(goal[3] - current[3]),
                    self.config.terminal_yaw_max_radps,
                ),
            )
        else:
            try:
                commanded_at = time.perf_counter()
                raw = self.dwa_controller.command(
                    active["plan"].path_world,
                    (current[0], current[1], current[3]),
                    velocity,
                    local_map,
                )
                command = (
                    _velocity(raw)
                    if arm_vla_reference
                    else _bounded_velocity(raw, self.config)
                )
                dwa_elapsed_ms = (time.perf_counter() - commanded_at) * 1000.0
            except Exception as error:
                return self._finish(
                    f"dwa_command_failed:{type(error).__name__}:{error}", failed=True
                )
        if terminal_yaw_active:
            raw = command
            dwa_elapsed_ms = 0.0
        elif arm_vla_reference:
            stalled, diagnostics = self.stall_detector.update(
                current[0],
                current[1],
                command[0],
                current[3],
                command[2],
            )
            active["stall_diagnostics"] = dict(vars(diagnostics))
            if stalled:
                return self._finish("navigation_stall", failed=False)
        arm_target, gripper_target = self._posture()
        return ExecutionCommand(
            base_velocity=command,
            arm_joint_target=arm_target,
            gripper_target=gripper_target,
            status="navigation_executing",
            trace={
                **self.status(),
                "distance_m": distance,
                "yaw_error_rad": yaw_error,
                "control_mode": (
                    "terminal_yaw" if terminal_yaw_active else "pct_dwa"
                ),
                "dwa_raw_command": [float(value) for value in raw],
                "bounded_base_velocity": list(command),
                "dwa_elapsed_ms": dwa_elapsed_ms,
                "dwa_adapter_trace": dict(
                    getattr(self.dwa_controller, "last_trace", {}) or {}
                ),
            },
        )

    def cancel_for_new_query(self) -> ExecutionCommand:
        self._active = None
        return _stop("navigation_cancelled_for_new_query", failed=False, requery=True)

    def status(self) -> dict[str, Any]:
        if self._active is None:
            return {"active": False}
        active = self._active
        plan: PCTPlan | None = active["plan"]
        return {
            "active": True,
            "request_id": active["request_id"],
            "sequence_id": active["sequence_id"],
            "route": active["route"],
            "execution_mode": active["execution_mode"],
            "safety_profile": active["safety_profile"],
            "arm_vla_reference_rules": active["arm_vla_reference_rules"],
            "translation_limit_disabled": active["translation_limit_disabled"],
            "full_horizon_contract_passed": active[
                "full_horizon_contract_passed"
            ],
            "full_horizon_violation": active["full_horizon_violation"],
            "selection_policy": active["selection_policy"],
            "trusted_horizon_points": active["trusted_horizon_points"],
            "model_trusted_prefix_k": active["model_trusted_prefix_k"],
            "minimum_lookahead_m": active["minimum_lookahead_m"],
            "target_lookahead_m": active["target_lookahead_m"],
            "ranked_waypoint_indices": active["ranked_indices"],
            "pct_candidate_rejections": active["candidate_rejections"],
            "chunk_step_count": active["chunk_step_count"],
            "stall_diagnostics": active["stall_diagnostics"],
            "goal_tolerance_m": self.config.goal_tolerance_m,
            "yaw_tolerance_rad": self.config.yaw_tolerance_rad,
            "max_chunk_execution_steps": self.config.max_chunk_execution_steps,
            "selected_waypoint_index": active["selected_index"],
            "selected_waypoint_body": list(active["selected_body"]),
            "predicted_goal_world": list(active["predicted_goal"]),
            "planner": active["mode"],
            "pct_path_world": None if plan is None else [list(row) for row in plan.path_world],
            "pct_snap_distance_m": None if plan is None else plan.snap_distance_m,
            "pct_elapsed_ms": active["pct_elapsed_ms"],
            "pct_metadata": None if plan is None else dict(plan.metadata),
        }

    def _posture(self) -> tuple[tuple[float, ...] | None, float]:
        if self._active is None:
            raise RuntimeError("navigation posture requested without an active chunk")
        source = self._active["route"] == WaypointRoute.NAV_TO_SOURCE.value
        return (
            self.config.stow_joint_target if source else self.config.carry_joint_target,
            self.config.open_gripper_target
            if source
            else self.config.closed_gripper_target,
        )

    def _finish(self, reason: str, *, failed: bool) -> ExecutionCommand:
        trace = self.status()
        self._active = None
        return _stop(reason, failed=failed, requery=True, trace=trace)


class CuRoboIKRecedingHorizonExecutor:
    """Plan/execute one absolute TCP target, hold base, then requery."""

    def __init__(
        self,
        planner: CuRoboIKPlanner,
        controller: ArmTrajectoryController,
        config: ManipulationExecutionConfig = ManipulationExecutionConfig(),
    ) -> None:
        self.planner = planner
        self.controller = controller
        self.config = config
        self._active: dict[str, Any] | None = None
        self._last_sequence = -1

    def begin(
        self,
        response: WaypointResponse,
        current_tcp_base: Sequence[float],
        current_joints: Sequence[float],
        scene_collision: Any,
        *,
        now_s: float | None = None,
    ) -> ExecutionCommand:
        self._active = None
        now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s")
        if response.sequence_id <= self._last_sequence:
            return _stop("stale_manipulation_response", failed=True)
        self._last_sequence = response.sequence_id
        if response.terminal:
            return _stop(
                f"manipulation_terminal_route:{response.route}",
                failed=response.route != WaypointRoute.DONE.value,
            )
        if response.action_domain != WaypointActionDomain.MANIPULATION.value:
            return _stop("response_is_not_manipulation", failed=True)
        gate_trace = {
            "target_step_limits_enforced": self.config.enforce_target_step_limits,
            "configured_max_translation_step_m": (
                self.config.target_safety.max_translation_step_m
            ),
            "configured_max_axis_rotation_step_rad": (
                self.config.target_safety.max_axis_rotation_step_rad
            ),
        }
        try:
            targets, mask = _fixed_arm_chunk(response)
            target_safety = self.config.target_safety
            if not self.config.enforce_target_step_limits:
                target_safety = ArmTargetSafety(
                    workspace_min_xyz=target_safety.workspace_min_xyz,
                    workspace_max_xyz=target_safety.workspace_max_xyz,
                    max_translation_step_m=math.inf,
                    max_axis_rotation_step_rad=math.inf,
                )
            accepted = validate_arm_targets(
                targets,
                mask,
                current_tcp_base,
                safety=target_safety,
            )
            target = accepted[0]
            joints = _finite_vector(current_joints, None, "current_joints")
            planned_at = time.perf_counter()
            plan = self.planner.plan(joints, target[:6], scene_collision)
            planning_elapsed_ms = (time.perf_counter() - planned_at) * 1000.0
            _validate_arm_plan(plan, joints, self.config)
            self.controller.reset(plan, target[6])
        except Exception as error:
            return _stop(
                f"arm_plan_failed:{type(error).__name__}:{error}",
                failed=True,
                trace=gate_trace,
            )
        self._active = {
            "request_id": response.request_id,
            "sequence_id": response.sequence_id,
            "route": response.route,
            "model_trusted_prefix_k": response.trusted_prefix_k,
            "selected_target_index": 0,
            "target_tcp_base": target,
            "plan": plan,
            "planning_elapsed_ms": planning_elapsed_ms,
            "started_s": now,
            **gate_trace,
        }
        return ExecutionCommand(
            base_velocity=(0.0, 0.0, 0.0),
            gripper_target=target[6],
            status="manipulation_chunk_planned",
            trace=self.status(),
        )

    def step(
        self,
        measured_joints: Sequence[float],
        *,
        now_s: float | None = None,
    ) -> ExecutionCommand:
        if self._active is None:
            return _stop("no_active_manipulation_chunk", failed=True)
        now = time.monotonic() if now_s is None else _finite_scalar(now_s, "now_s")
        if now - self._active["started_s"] > self.config.chunk_timeout_s:
            return self._finish("manipulation_chunk_timeout", failed=True)
        try:
            commanded_at = time.perf_counter()
            target, done = self.controller.command(measured_joints)
            joints = _finite_vector(target, None, "arm_joint_target")
            measured = _finite_vector(measured_joints, len(joints), "measured_joints")
            _validate_joint_command(joints, measured, self.config)
            controller_elapsed_ms = (time.perf_counter() - commanded_at) * 1000.0
        except Exception as error:
            return self._finish(
                f"arm_controller_failed:{type(error).__name__}:{error}", failed=True
            )
        if done:
            return self._finish("first_tcp_target_reached", failed=False)
        return ExecutionCommand(
            base_velocity=(0.0, 0.0, 0.0),
            arm_joint_target=joints,
            gripper_target=self._active["target_tcp_base"][6],
            status="manipulation_executing",
            trace={
                **self.status(),
                "arm_joint_target": list(joints),
                "gripper_target": self._active["target_tcp_base"][6],
                "controller_elapsed_ms": controller_elapsed_ms,
            },
        )

    def cancel_for_new_query(self) -> ExecutionCommand:
        self._active = None
        return _stop("manipulation_cancelled_for_new_query", failed=False, requery=True)

    def status(self) -> dict[str, Any]:
        if self._active is None:
            return {"active": False}
        active = self._active
        plan: ArmPlan = active["plan"]
        return {
            "active": True,
            "request_id": active["request_id"],
            "sequence_id": active["sequence_id"],
            "route": active["route"],
            "model_trusted_prefix_k": active["model_trusted_prefix_k"],
            "base_hold": [0.0, 0.0, 0.0],
            "selected_target_index": active["selected_target_index"],
            "target_tcp_base": list(active["target_tcp_base"]),
            "planner": plan.planner,
            "joint_path_length": len(plan.joint_path),
            "target_position_error_m": plan.target_position_error_m,
            "target_orientation_error_rad": plan.target_orientation_error_rad,
            "collision_free": plan.collision_free,
            "planning_elapsed_ms": active["planning_elapsed_ms"],
            "target_step_limits_enforced": active[
                "target_step_limits_enforced"
            ],
            "configured_max_translation_step_m": active[
                "configured_max_translation_step_m"
            ],
            "configured_max_axis_rotation_step_rad": active[
                "configured_max_axis_rotation_step_rad"
            ],
            "planner_metadata": dict(plan.metadata),
        }

    def _finish(self, reason: str, *, failed: bool) -> ExecutionCommand:
        trace = self.status()
        self._active = None
        return _stop(reason, failed=failed, requery=True, trace=trace)


def _fixed_navigation_chunk(
    response: WaypointResponse,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[bool, ...]]:
    if response.nav_waypoints_body is None:
        raise ValueError("NAV response has no waypoints")
    rows = list(response.nav_waypoints_body)
    mask = list(response.action_valid_mask)
    if len(rows) > ACTION_HORIZON or len(rows) != len(mask):
        raise ValueError("NAV wire suffix has an invalid length")
    rows.extend([(0.0, 0.0, 0.0)] * (ACTION_HORIZON - len(rows)))
    mask.extend([False] * (ACTION_HORIZON - len(mask)))
    return tuple(rows), tuple(mask)


def _fixed_arm_chunk(
    response: WaypointResponse,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    if response.arm_targets_base is None:
        raise ValueError("ARM response has no TCP targets")
    rows = list(response.arm_targets_base)
    mask = list(response.action_valid_mask)
    if len(rows) > ACTION_HORIZON or len(rows) != len(mask):
        raise ValueError("ARM wire suffix has an invalid length")
    rows.extend([(0.0,) * 7] * (ACTION_HORIZON - len(rows)))
    mask.extend([False] * (ACTION_HORIZON - len(mask)))
    return tuple(rows), tuple(mask)


def _validate_pct_plan(plan: PCTPlan, goal: PCTWorldPose, max_snap_m: float) -> None:
    if not isinstance(plan, PCTPlan) or len(plan.path_world) < 2:
        raise ValueError("PCT must return at least two path points")
    if not all(len(point) == 2 and all(math.isfinite(float(item)) for item in point) for point in plan.path_world):
        raise ValueError("PCT path contains invalid points")
    if (
        not math.isfinite(plan.snap_distance_m)
        or plan.snap_distance_m < 0.0
        or plan.snap_distance_m > max_snap_m
    ):
        raise ValueError("PCT endpoint snap exceeds the contract")
    if len(plan.snapped_goal_world) != 4 or not all(
        math.isfinite(float(value)) for value in plan.snapped_goal_world
    ):
        raise ValueError("PCT snapped goal is invalid")
    endpoint_error = math.hypot(
        plan.snapped_goal_world[0] - goal[0],
        plan.snapped_goal_world[1] - goal[1],
    )
    if endpoint_error > max_snap_m + 1.0e-9:
        raise ValueError("PCT snapped goal differs from model goal")


def _validate_arm_plan(
    plan: ArmPlan,
    current_joints: Sequence[float],
    config: ManipulationExecutionConfig,
) -> None:
    if not isinstance(plan, ArmPlan) or plan.planner.lower() not in {"curobo", "ik", "curobo+ik"}:
        raise ValueError("arm planner must report cuRobo/IK provenance")
    if not plan.reachable or not plan.collision_free or not plan.joint_path:
        raise ValueError("cuRobo/IK target is unreachable or colliding")
    joint_count = len(current_joints)
    if any(len(row) != joint_count or not all(math.isfinite(item) for item in row) for row in plan.joint_path):
        raise ValueError("cuRobo/IK joint path is invalid")
    previous = tuple(float(value) for value in current_joints)
    for row in plan.joint_path:
        _validate_joint_command(row, previous, config)
        previous = row
    if (
        not math.isfinite(plan.target_position_error_m)
        or plan.target_position_error_m > config.max_target_position_error_m
        or not math.isfinite(plan.target_orientation_error_rad)
        or plan.target_orientation_error_rad > config.max_target_orientation_error_rad
    ):
        raise ValueError("cuRobo/IK terminal target error exceeds the contract")


def _validate_joint_command(
    target: Sequence[float],
    measured: Sequence[float],
    config: ManipulationExecutionConfig,
) -> None:
    if len(target) != len(measured) or any(
        abs(float(goal) - float(current)) > config.max_joint_command_step_rad
        for goal, current in zip(target, measured, strict=True)
    ):
        raise ValueError("arm joint command exceeds the per-cycle rate limit")
    if config.joint_position_limits:
        if len(config.joint_position_limits) != len(target):
            raise ValueError("arm joint limit count does not match the plan")
        if any(
            not lower <= float(value) <= upper
            for value, (lower, upper) in zip(
                target, config.joint_position_limits, strict=True
            )
        ):
            raise ValueError("arm joint command exceeds a position limit")


def _world_pose(value: Sequence[float]) -> PCTWorldPose:
    values = _finite_vector(value, None, "current_base_world")
    if len(values) == 3:
        return values[0], values[1], 0.0, wrap_to_pi(values[2])
    if len(values) == 7:
        from conveyor_bench.conveyorvla.waypoint import yaw_from_quaternion

        return values[0], values[1], values[2], yaw_from_quaternion(values[3:])
    raise ValueError("current_base_world must contain [x,y,yaw] or a 7D pose")


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def _velocity(value: Sequence[float]) -> BaseVelocity:
    values = _finite_vector(value, 3, "measured_body_velocity")
    return values[0], values[1], values[2]


def _bounded_velocity(
    value: Sequence[float], config: NavigationExecutionConfig
) -> BaseVelocity:
    values = _finite_vector(value, 3, "DWA command")
    limits = (config.max_abs_vx_mps, config.max_abs_vy_mps, config.max_abs_wz_radps)
    if any(abs(item) > limit + 1.0e-9 for item, limit in zip(values, limits, strict=True)):
        raise ValueError("DWA command exceeds configured limits")
    return values[0], values[1], values[2]


def _finite_vector(
    value: Sequence[float], length: int | None, name: str
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(float(item) for item in value)
    if (length is not None and len(result) != length) or not result or not all(
        math.isfinite(item) for item in result
    ):
        raise ValueError(f"{name} has invalid shape or values")
    return result


def _finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _stop(
    reason: str,
    *,
    failed: bool,
    requery: bool = False,
    trace: Mapping[str, Any] | None = None,
) -> ExecutionCommand:
    return ExecutionCommand(
        base_velocity=(0.0, 0.0, 0.0),
        status="recover" if failed else "hold",
        requires_requery=requery,
        failed=failed,
        reason=reason,
        trace=dict(trace or {}),
    )


__all__ = [
    "ArmPlan",
    "ArmTrajectoryController",
    "CuRoboIKPlanner",
    "CuRoboIKRecedingHorizonExecutor",
    "DWAController",
    "ExecutionCommand",
    "ManipulationExecutionConfig",
    "NAVIGATION_SAFETY_PROFILE_ARM_VLA_REFERENCE",
    "NAVIGATION_SAFETY_PROFILE_CONTRACT",
    "NAVIGATION_SAFETY_PROFILE_EXECUTABLE_PREFIX",
    "NAVIGATION_SAFETY_PROFILE_LOOKAHEAD_ARM_VLA",
    "NAVIGATION_SAFETY_PROFILE_UNBOUNDED_TRANSLATION",
    "NAVIGATION_SAFETY_PROFILES",
    "NavigationExecutionConfig",
    "PCTDWARecedingHorizonExecutor",
    "PCTPlan",
    "PCTPlanner",
]
