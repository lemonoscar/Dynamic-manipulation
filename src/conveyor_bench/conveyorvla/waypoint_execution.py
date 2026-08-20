"""Fail-closed PCT/DWA and cuRobo/IK receding-horizon executors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    ArmTargetSafety,
    NavWaypointSafety,
    WaypointActionDomain,
    WaypointRoute,
    nav_waypoint_world,
    select_navigation_waypoint,
    validate_arm_targets,
    wrap_to_pi,
)
from conveyor_bench.conveyorvla.waypoint_protocol import WaypointResponse


WorldPose = tuple[float, float, float]
BaseVelocity = tuple[float, float, float]


@dataclass(frozen=True)
class PCTPlan:
    path_world: tuple[tuple[float, float], ...]
    snapped_goal_world: WorldPose
    snap_distance_m: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class PCTPlanner(Protocol):
    def plan(self, current_world_pose: WorldPose, predicted_world_goal: WorldPose) -> PCTPlan: ...


class DWAController(Protocol):
    def command(
        self,
        path_world: Sequence[Sequence[float]],
        current_world_pose: WorldPose,
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
    pct_snap_max_m: float = 0.10
    goal_tolerance_m: float = 0.12
    yaw_tolerance_rad: float = 0.14
    max_abs_vx_mps: float = 0.60
    max_abs_vy_mps: float = 0.40
    max_abs_wz_radps: float = 1.20
    terminal_yaw_kp: float = 1.5
    terminal_yaw_max_radps: float = 0.60
    chunk_timeout_s: float = 12.0
    stall_timeout_s: float = 3.0
    stall_progress_m: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.pct_snap_max_m,
            self.goal_tolerance_m,
            self.yaw_tolerance_rad,
            self.max_abs_vx_mps,
            self.max_abs_vy_mps,
            self.max_abs_wz_radps,
            self.terminal_yaw_kp,
            self.terminal_yaw_max_radps,
            self.chunk_timeout_s,
            self.stall_timeout_s,
            self.stall_progress_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("navigation execution limits must be finite and positive")


@dataclass(frozen=True)
class ManipulationExecutionConfig:
    target_safety: ArmTargetSafety = ArmTargetSafety()
    chunk_timeout_s: float = 12.0
    max_target_position_error_m: float = 0.02
    max_target_orientation_error_rad: float = 0.10

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.chunk_timeout_s,
                self.max_target_position_error_m,
                self.max_target_orientation_error_rad,
            )
        ):
            raise ValueError("manipulation execution limits must be finite and positive")


class PCTDWARecedingHorizonExecutor:
    """Execute only the first model waypoint, then force a new visual query."""

    def __init__(
        self,
        pct_planner: PCTPlanner,
        dwa_controller: DWAController,
        config: NavigationExecutionConfig = NavigationExecutionConfig(),
    ) -> None:
        self.pct_planner = pct_planner
        self.dwa_controller = dwa_controller
        self.config = config
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
        try:
            current = _world_pose(current_base_world)
            waypoints, mask = _fixed_navigation_chunk(response)
            selected_index, selected_body = select_navigation_waypoint(
                waypoints,
                mask,
                safety=self.config.waypoint_safety,
            )
            predicted_goal = nav_waypoint_world(
                (current[0], current[1], 0.0, *_yaw_quaternion(current[2])),
                selected_body,
            )
        except (ValueError, TypeError) as error:
            return _stop(f"navigation_waypoint_rejected:{error}", failed=True)
        translation = math.hypot(selected_body[0], selected_body[1])
        mode = "terminal_yaw" if translation < self.config.waypoint_safety.minimum_translation_m else "pct_dwa"
        plan: PCTPlan | None = None
        if mode == "pct_dwa":
            try:
                plan = self.pct_planner.plan(current, predicted_goal)
                _validate_pct_plan(plan, predicted_goal, self.config.pct_snap_max_m)
            except Exception as error:
                return _stop(f"pct_plan_failed:{type(error).__name__}:{error}", failed=True)
        distance = math.hypot(predicted_goal[0] - current[0], predicted_goal[1] - current[1])
        self._active = {
            "request_id": response.request_id,
            "sequence_id": response.sequence_id,
            "route": response.route,
            "execution_mode": (
                "stow_open"
                if response.route == WaypointRoute.NAV_TO_SOURCE.value
                else "carry_closed"
            ),
            "selected_index": selected_index,
            "selected_body": selected_body,
            "predicted_goal": predicted_goal,
            "plan": plan,
            "mode": mode,
            "started_s": now,
            "last_progress_s": now,
            "best_distance_m": distance,
        }
        return ExecutionCommand(
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
        goal: WorldPose = active["predicted_goal"]
        distance = math.hypot(goal[0] - current[0], goal[1] - current[1])
        yaw_error = abs(wrap_to_pi(goal[2] - current[2]))
        if now - active["started_s"] > self.config.chunk_timeout_s:
            return self._finish("navigation_chunk_timeout", failed=True)
        if distance <= self.config.goal_tolerance_m and yaw_error <= self.config.yaw_tolerance_rad:
            return self._finish("first_waypoint_reached", failed=False)
        if distance + self.config.stall_progress_m < active["best_distance_m"]:
            active["best_distance_m"] = distance
            active["last_progress_s"] = now
        elif active["mode"] == "pct_dwa" and now - active["last_progress_s"] > self.config.stall_timeout_s:
            return self._finish("navigation_stall", failed=True)
        if active["mode"] == "terminal_yaw":
            command = (
                0.0,
                0.0,
                _clamp(
                    self.config.terminal_yaw_kp * wrap_to_pi(goal[2] - current[2]),
                    self.config.terminal_yaw_max_radps,
                ),
            )
        else:
            try:
                raw = self.dwa_controller.command(
                    active["plan"].path_world,
                    current,
                    velocity,
                    local_map,
                )
                command = _bounded_velocity(raw, self.config)
            except Exception as error:
                return self._finish(
                    f"dwa_command_failed:{type(error).__name__}:{error}", failed=True
                )
        return ExecutionCommand(
            base_velocity=command,
            status="navigation_executing",
            trace={**self.status(), "distance_m": distance, "yaw_error_rad": yaw_error},
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
            "selected_waypoint_index": active["selected_index"],
            "selected_waypoint_body": list(active["selected_body"]),
            "predicted_goal_world": list(active["predicted_goal"]),
            "planner": active["mode"],
            "pct_path_world": None if plan is None else [list(row) for row in plan.path_world],
            "pct_snap_distance_m": None if plan is None else plan.snap_distance_m,
        }

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
        try:
            targets, mask = _fixed_arm_chunk(response)
            accepted = validate_arm_targets(
                targets,
                mask,
                current_tcp_base,
                safety=self.config.target_safety,
            )
            target = accepted[0]
            joints = _finite_vector(current_joints, None, "current_joints")
            plan = self.planner.plan(joints, target[:6], scene_collision)
            _validate_arm_plan(plan, len(joints), self.config)
            self.controller.reset(plan, target[6])
        except Exception as error:
            return _stop(f"arm_plan_failed:{type(error).__name__}:{error}", failed=True)
        self._active = {
            "request_id": response.request_id,
            "sequence_id": response.sequence_id,
            "route": response.route,
            "selected_target_index": 0,
            "target_tcp_base": target,
            "plan": plan,
            "started_s": now,
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
            target, done = self.controller.command(measured_joints)
            joints = _finite_vector(target, None, "arm_joint_target")
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
            trace=self.status(),
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
            "base_hold": [0.0, 0.0, 0.0],
            "selected_target_index": active["selected_target_index"],
            "target_tcp_base": list(active["target_tcp_base"]),
            "planner": plan.planner,
            "joint_path_length": len(plan.joint_path),
            "target_position_error_m": plan.target_position_error_m,
            "target_orientation_error_rad": plan.target_orientation_error_rad,
            "collision_free": plan.collision_free,
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


def _validate_pct_plan(plan: PCTPlan, goal: WorldPose, max_snap_m: float) -> None:
    if not isinstance(plan, PCTPlan) or len(plan.path_world) < 2:
        raise ValueError("PCT must return at least two path points")
    if not all(len(point) == 2 and all(math.isfinite(float(item)) for item in point) for point in plan.path_world):
        raise ValueError("PCT path contains invalid points")
    if not math.isfinite(plan.snap_distance_m) or plan.snap_distance_m > max_snap_m:
        raise ValueError("PCT endpoint snap exceeds the contract")
    endpoint_error = math.hypot(
        plan.snapped_goal_world[0] - goal[0],
        plan.snapped_goal_world[1] - goal[1],
    )
    if endpoint_error > max_snap_m + 1.0e-9:
        raise ValueError("PCT snapped goal differs from model goal")


def _validate_arm_plan(
    plan: ArmPlan,
    joint_count: int,
    config: ManipulationExecutionConfig,
) -> None:
    if not isinstance(plan, ArmPlan) or plan.planner.lower() not in {"curobo", "ik", "curobo+ik"}:
        raise ValueError("arm planner must report cuRobo/IK provenance")
    if not plan.reachable or not plan.collision_free or not plan.joint_path:
        raise ValueError("cuRobo/IK target is unreachable or colliding")
    if any(len(row) != joint_count or not all(math.isfinite(item) for item in row) for row in plan.joint_path):
        raise ValueError("cuRobo/IK joint path is invalid")
    if (
        not math.isfinite(plan.target_position_error_m)
        or plan.target_position_error_m > config.max_target_position_error_m
        or not math.isfinite(plan.target_orientation_error_rad)
        or plan.target_orientation_error_rad > config.max_target_orientation_error_rad
    ):
        raise ValueError("cuRobo/IK terminal target error exceeds the contract")


def _world_pose(value: Sequence[float]) -> WorldPose:
    values = _finite_vector(value, None, "current_base_world")
    if len(values) == 3:
        return values[0], values[1], wrap_to_pi(values[2])
    if len(values) == 7:
        from conveyor_bench.conveyorvla.waypoint import yaw_from_quaternion

        return values[0], values[1], yaw_from_quaternion(values[3:])
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
    "NavigationExecutionConfig",
    "PCTDWARecedingHorizonExecutor",
    "PCTPlan",
    "PCTPlanner",
]
