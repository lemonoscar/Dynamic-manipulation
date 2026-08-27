"""Isaac/PCT/DWA wiring for the joint-trajectory successor.

The module is deliberately independent of the legacy waypoint executors.  It
turns one already-committed runtime step into either a two-second navigation
window or ten direct joint/gripper targets.  It never selects a prefix, calls
IK/cuRobo, reads evaluator truth, or changes the model-selected route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    MANIPULATION_STRIDE_S,
    NAVIGATION_STRIDE_S,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (
    DirectJointCommand,
    JointTrajectoryRuntimeStep,
    NavigationReference,
    SuccessUpdate,
    TransferSuccessEvaluator,
)
from conveyor_bench.conveyorvla.waypoint import (
    nav_waypoint_world,
    wrap_to_pi,
    yaw_from_quaternion,
)
from conveyor_bench.conveyorvla.waypoint_execution import PCTPlan
from conveyor_bench.isaac.locomotion import guard_longitudinal_command


CONTROL_STRIDE_S = 0.02
NAVIGATION_WINDOW_S = ACTION_HORIZON * NAVIGATION_STRIDE_S
ARM_JOINT_NAMES = tuple(f"arm_joint{index}" for index in range(1, 7))
GRIPPER_JOINT_NAMES = ("arm_joint7", "arm_joint8")


@dataclass(frozen=True)
class JointNavigationConfig:
    control_stride_s: float = CONTROL_STRIDE_S
    execution_window_s: float = NAVIGATION_WINDOW_S
    goal_tolerance_m: float = 0.12
    yaw_tolerance_rad: float = 0.14
    pct_snap_max_m: float = 0.10

    def __post_init__(self) -> None:
        if not math.isclose(self.control_stride_s, CONTROL_STRIDE_S):
            raise ValueError("joint-trajectory control stride must remain 0.02 s")
        if not math.isclose(self.execution_window_s, NAVIGATION_WINDOW_S):
            raise ValueError("joint-trajectory NAV window must remain exactly 2.0 s")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (
                self.goal_tolerance_m,
                self.yaw_tolerance_rad,
                self.pct_snap_max_m,
            )
        ):
            raise ValueError("navigation tolerances must be finite and positive")

    @property
    def maximum_control_ticks(self) -> int:
        return round(self.execution_window_s / self.control_stride_s)


@dataclass(frozen=True)
class JointNavigationPlan:
    query_base_world: tuple[float, ...]
    reference_world: tuple[tuple[float, float, float], ...]
    pct_plan: PCTPlan
    started_timestamp_s: float
    trace: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JointNavigationControl:
    base_velocity: tuple[float, float, float]
    requires_requery: bool
    reached_local_goal: bool
    elapsed_s: float
    reason: str | None
    trace: Mapping[str, Any] = field(default_factory=dict)


class PCTDWAJointNavigationExecutor:
    """Plan to point ten, then run strict PCT/DWA for at most two seconds.

    The approved PCT API exposes one endpoint rather than via-points.  All ten
    model points are transformed, validated, and retained in the trace; point
    ten is passed to PCT as its local goal.  This limitation is explicit so the
    trace never over-claims that PCT consumed nine unsupported via-points.
    """

    def __init__(
        self,
        pct_planner: Any,
        dwa_controller: Any,
        config: JointNavigationConfig = JointNavigationConfig(),
    ) -> None:
        self.pct_planner = pct_planner
        self.dwa_controller = dwa_controller
        self.config = config
        self._active: JointNavigationPlan | None = None

    def begin(
        self,
        reference: NavigationReference,
        query_base_world: Sequence[float],
        *,
        timestamp_s: float,
    ) -> JointNavigationPlan:
        query = _finite_vector(query_base_world, 7, "query_base_world")
        timestamp = _finite_nonnegative(timestamp_s, "navigation timestamp")
        if len(reference.points_query_body) != ACTION_HORIZON:
            raise ValueError("NAV reference must contain all ten model points")
        if not math.isclose(reference.stride_s, NAVIGATION_STRIDE_S):
            raise ValueError("NAV reference stride must remain 0.20 s")
        world = tuple(
            nav_waypoint_world(query, point)
            for point in reference.points_query_body
        )
        if world[-1] != nav_waypoint_world(query, reference.local_goal_query_body):
            raise ValueError("NAV local goal must be reference point ten")
        query_yaw = yaw_from_quaternion(query[3:])
        current = (query[0], query[1], query[2], query_yaw)
        endpoint = (world[-1][0], world[-1][1], query[2], world[-1][2])
        pct_plan = self.pct_planner.plan(current, endpoint)
        if not isinstance(pct_plan, PCTPlan):
            raise TypeError("PCT planner must return the typed PCTPlan contract")
        if len(pct_plan.path_world) < 2:
            raise ValueError("PCT plan must contain at least two path points")
        if pct_plan.snap_distance_m > self.config.pct_snap_max_m:
            raise ValueError("PCT endpoint snap exceeds the joint-trajectory limit")
        plan = JointNavigationPlan(
            query_base_world=query,
            reference_world=world,
            pct_plan=pct_plan,
            started_timestamp_s=timestamp,
            trace={
                "reference_point_count": ACTION_HORIZON,
                "reference_world": [list(point) for point in world],
                "pct_input_mode": "endpoint_only_approved_api",
                "pct_endpoint_reference_index": ACTION_HORIZON - 1,
                "pct_endpoint_world": list(endpoint),
                "unsupported_via_points_claimed": False,
                "prefix_selected": False,
            },
        )
        self._active = plan
        return plan

    def command(
        self,
        current_base_world: Sequence[float],
        measured_body_velocity: Sequence[float],
        local_map: Any,
        *,
        timestamp_s: float,
    ) -> JointNavigationControl:
        plan = self._active
        if plan is None:
            raise RuntimeError("navigation command requested without an active plan")
        current = _finite_vector(current_base_world, 7, "current_base_world")
        velocity = _finite_vector(measured_body_velocity, 3, "measured_body_velocity")
        timestamp = _finite_nonnegative(timestamp_s, "navigation timestamp")
        elapsed = timestamp - plan.started_timestamp_s
        if elapsed < -1.0e-9:
            raise ValueError("navigation timestamp precedes its query")
        pose = (current[0], current[1], yaw_from_quaternion(current[3:]))
        goal = plan.pct_plan.snapped_goal_world
        distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        yaw_error = abs(wrap_to_pi(goal[3] - pose[2]))
        reached = bool(
            distance <= self.config.goal_tolerance_m
            and yaw_error <= self.config.yaw_tolerance_rad
        )
        timed_out = elapsed >= self.config.execution_window_s - 1.0e-9
        trace = {
            "elapsed_s": max(0.0, elapsed),
            "distance_to_snapped_goal_m": distance,
            "yaw_error_to_snapped_goal_rad": yaw_error,
        }
        if reached or timed_out:
            return JointNavigationControl(
                base_velocity=(0.0, 0.0, 0.0),
                requires_requery=True,
                reached_local_goal=reached,
                elapsed_s=max(0.0, elapsed),
                reason="local_goal_reached" if reached else "two_second_window_complete",
                trace=trace,
            )
        raw = self.dwa_controller.command(
            plan.pct_plan.path_world,
            pose,
            velocity,
            local_map,
        )
        guarded = guard_longitudinal_command(raw)
        return JointNavigationControl(
            base_velocity=guarded,
            requires_requery=False,
            reached_local_goal=False,
            elapsed_s=max(0.0, elapsed),
            reason=None,
            trace={**trace, "raw_dwa_command": list(raw), "guarded_command": list(guarded)},
        )

    @property
    def active_plan(self) -> JointNavigationPlan | None:
        return self._active

    def reset(self) -> None:
        self._active = None
        reset = getattr(self.dwa_controller, "reset", None)
        if callable(reset):
            reset()


class IsaacJointActionAdapter:
    """Build approved-reference RobotAction objects with continuous gripper targets."""

    def __init__(
        self,
        robot_action_factory: Callable[..., Any],
        *,
        arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
        gripper_joint_names: Sequence[str] = GRIPPER_JOINT_NAMES,
        gripper_closed_position: float = 0.0,
        gripper_open_position: float = 0.04,
    ) -> None:
        self.robot_action_factory = robot_action_factory
        self.arm_joint_names = tuple(str(name) for name in arm_joint_names)
        self.gripper_joint_names = tuple(str(name) for name in gripper_joint_names)
        self.gripper_closed_position = float(gripper_closed_position)
        self.gripper_open_position = float(gripper_open_position)
        if len(self.arm_joint_names) != 6 or not self.gripper_joint_names:
            raise ValueError("joint action adapter requires six arm joints and gripper joints")
        if not all(self.arm_joint_names) or not all(self.gripper_joint_names):
            raise ValueError("joint names must be non-empty")
        if not (
            math.isfinite(self.gripper_closed_position)
            and math.isfinite(self.gripper_open_position)
            and self.gripper_open_position > self.gripper_closed_position
        ):
            raise ValueError("gripper physical range must be finite and ordered")

    def hold(
        self,
        command: DirectJointCommand,
        *,
        route: JointTrajectoryRoute | None,
        sequence_id: int,
        source: str = "joint_trajectory_hold",
    ) -> Any:
        return self._build(
            command,
            base_velocity=(0.0, 0.0, 0.0),
            route=route,
            sequence_id=sequence_id,
            source=source,
            command_kind="hold",
        )

    def navigation(
        self,
        base_velocity: Sequence[float],
        arm_hold: DirectJointCommand,
        *,
        route: JointTrajectoryRoute,
        sequence_id: int,
    ) -> Any:
        if action_domain(route) is not JointTrajectoryDomain.NAVIGATION:
            raise ValueError("navigation action requires a NAV route")
        guarded = guard_longitudinal_command(base_velocity)
        return self._build(
            arm_hold,
            base_velocity=guarded,
            route=route,
            sequence_id=sequence_id,
            source=f"joint_trajectory_{route.value.lower()}",
            command_kind="navigation",
        )

    def manipulation(
        self,
        command: DirectJointCommand,
        *,
        route: JointTrajectoryRoute,
        sequence_id: int,
    ) -> Any:
        if action_domain(route) is not JointTrajectoryDomain.MANIPULATION:
            raise ValueError("direct joint action requires a Mani route")
        if command.base_velocity != (0.0, 0.0, 0.0):
            raise ValueError("Mani command must keep the base command exactly zero")
        return self._build(
            command,
            base_velocity=(0.0, 0.0, 0.0),
            route=route,
            sequence_id=sequence_id,
            source=f"joint_trajectory_{route.value.lower()}",
            command_kind="manipulation",
        )

    def _build(
        self,
        command: DirectJointCommand,
        *,
        base_velocity: tuple[float, float, float],
        route: JointTrajectoryRoute | None,
        sequence_id: int,
        source: str,
        command_kind: str,
    ) -> Any:
        target = _finite_vector(command.joint_position, 6, "arm joint target")
        fraction = _unit_fraction(command.gripper_open_fraction, "gripper target")
        physical = self.gripper_closed_position + fraction * (
            self.gripper_open_position - self.gripper_closed_position
        )
        metadata = {
            "joint_trajectory_policy": True,
            "model_route": None if route is None else route.value,
            "model_sequence_id": int(sequence_id),
            "direct_joint_command_index": int(command.index),
            "joint_trajectory_command_kind": command_kind,
            "segment_type": (
                "post_motion_hold"
                if command_kind in {"hold", "navigation"}
                else "direct_joint_motion"
            ),
            "arm_joint_names": self.arm_joint_names,
            "gripper_joint_names": self.gripper_joint_names,
            "gripper_joint_positions": tuple(physical for _ in self.gripper_joint_names),
            "gripper_open_fraction_requested": fraction,
            "base_command_requested": base_velocity,
            "manipulation_base_zero_contract": command_kind == "manipulation",
            "manipulation_base_lock": False,
            "manipulation_support_joint_lock": False,
            "uses_ik": False,
            "uses_curobo": False,
            "uses_prefix_selector": False,
        }
        return self.robot_action_factory(
            base_velocity=base_velocity,
            arm_joint_positions=target,
            # Explicit metadata makes the approved Isaac runtime stage the
            # continuous target; "hold" avoids binary open/close semantics.
            gripper_command="hold",
            source=source,
            metadata=metadata,
        )


@dataclass(frozen=True)
class JointControlTick:
    tick_index: int
    command_index: int | None
    route: JointTrajectoryRoute | None
    action: Any
    state_before: Any
    state_after: Any


@dataclass(frozen=True)
class JointSystemExecutionResult:
    control_ticks: int
    requires_requery: bool
    failed: bool
    reason: str | None
    final_state: Any
    trace: Mapping[str, Any] = field(default_factory=dict)


class IsaacJointTrajectorySystemExecutor:
    """Execute one policy step against an approved SimulationRuntime interface."""

    def __init__(
        self,
        simulation: Any,
        action_adapter: IsaacJointActionAdapter,
        navigation_executor: PCTDWAJointNavigationExecutor,
        *,
        render: bool = True,
        on_control_tick: Callable[[JointControlTick], None] | None = None,
    ) -> None:
        self.simulation = simulation
        self.action_adapter = action_adapter
        self.navigation_executor = navigation_executor
        self.render = bool(render)
        self.on_control_tick = on_control_tick

    def execute(
        self,
        step: JointTrajectoryRuntimeStep,
        *,
        local_map: Any = None,
    ) -> JointSystemExecutionResult:
        state = self.simulation.read()
        route = step.committed_route
        if step.action_domain is None:
            if step.hold is None:
                raise ValueError("pending/recover runtime step must provide an arm hold")
            action = self.action_adapter.hold(
                step.hold,
                route=route,
                sequence_id=step.sequence_id,
            )
            state = self._tick(action, route=route, command_index=None, tick_index=0)
            return JointSystemExecutionResult(
                control_ticks=1,
                requires_requery=True,
                failed=False,
                reason=step.recover_reason or "route_confirmation_pending",
                final_state=state,
                trace={"execution": "hold"},
            )
        if route is None or action_domain(route) is not step.action_domain:
            raise ValueError("runtime step domain does not match its committed route")
        if step.action_domain is JointTrajectoryDomain.MANIPULATION:
            return self._execute_manipulation(step, route, state)
        return self._execute_navigation(step, route, state, local_map)

    def _execute_manipulation(
        self,
        step: JointTrajectoryRuntimeStep,
        route: JointTrajectoryRoute,
        state: Any,
    ) -> JointSystemExecutionResult:
        chunk = step.manipulation
        if chunk is None or len(chunk.commands) != ACTION_HORIZON:
            raise ValueError("Mani runtime step must provide ten direct joint commands")
        ticks_per_command = round(MANIPULATION_STRIDE_S / CONTROL_STRIDE_S)
        if ticks_per_command != 2:
            raise AssertionError("Mani command must span exactly two 50 Hz ticks")
        tick_index = 0
        for command in chunk.commands:
            if not math.isclose(command.duration_s, MANIPULATION_STRIDE_S):
                raise ValueError("Mani command duration must remain 0.04 s")
            action = self.action_adapter.manipulation(
                command,
                route=route,
                sequence_id=step.sequence_id,
            )
            for _ in range(ticks_per_command):
                state = self._tick(
                    action,
                    route=route,
                    command_index=command.index,
                    tick_index=tick_index,
                )
                tick_index += 1
        return JointSystemExecutionResult(
            control_ticks=tick_index,
            requires_requery=True,
            failed=False,
            reason="ten_direct_joint_targets_complete",
            final_state=state,
            trace={
                "execution": "direct_joint_trajectory",
                "action_points": ACTION_HORIZON,
                "ticks_per_action_point": ticks_per_command,
                "base_command_exact_zero": True,
                "ik_or_curobo_used": False,
            },
        )

    def _execute_navigation(
        self,
        step: JointTrajectoryRuntimeStep,
        route: JointTrajectoryRoute,
        state: Any,
        local_map: Any,
    ) -> JointSystemExecutionResult:
        if step.navigation is None or step.hold is None:
            raise ValueError("NAV runtime step requires a reference and arm hold")
        try:
            plan = self.navigation_executor.begin(
                step.navigation,
                state.robot_root_pose,
                timestamp_s=float(state.timestamp),
            )
        except Exception as error:
            return self._failed_hold(step, route, state, "pct_planning_failed", error)
        ticks = 0
        last_control: JointNavigationControl | None = None
        try:
            while ticks < self.navigation_executor.config.maximum_control_ticks:
                last_control = self.navigation_executor.command(
                    state.robot_root_pose,
                    measured_body_velocity(state),
                    local_map,
                    timestamp_s=float(state.timestamp),
                )
                if last_control.requires_requery:
                    break
                action = self.action_adapter.navigation(
                    last_control.base_velocity,
                    step.hold,
                    route=route,
                    sequence_id=step.sequence_id,
                )
                state = self._tick(
                    action,
                    route=route,
                    command_index=None,
                    tick_index=ticks,
                )
                ticks += 1
        except Exception as error:
            self.navigation_executor.reset()
            return self._failed_hold(step, route, state, "dwa_control_failed", error, ticks)
        self.navigation_executor.reset()
        reason = (
            last_control.reason
            if last_control is not None and last_control.requires_requery
            else "two_second_window_complete"
        )
        return JointSystemExecutionResult(
            control_ticks=ticks,
            requires_requery=True,
            failed=False,
            reason=reason,
            final_state=state,
            trace={
                **dict(plan.trace),
                "execution": "pct_dwa_two_second_window",
                "control_ticks": ticks,
                "last_control": None if last_control is None else dict(last_control.trace),
            },
        )

    def _failed_hold(
        self,
        step: JointTrajectoryRuntimeStep,
        route: JointTrajectoryRoute,
        state: Any,
        category: str,
        error: Exception,
        prior_ticks: int = 0,
    ) -> JointSystemExecutionResult:
        if step.hold is None:
            raise RuntimeError(f"{category} and no fail-closed hold is available") from error
        action = self.action_adapter.hold(
            step.hold,
            route=route,
            sequence_id=step.sequence_id,
            source=f"joint_trajectory_{category}_hold",
        )
        state = self._tick(
            action,
            route=route,
            command_index=None,
            tick_index=prior_ticks,
        )
        return JointSystemExecutionResult(
            control_ticks=prior_ticks + 1,
            requires_requery=False,
            failed=True,
            reason=f"{category}:{type(error).__name__}:{error}",
            final_state=state,
            trace={"execution": "fail_closed_hold", "failure_category": category},
        )

    def _tick(
        self,
        action: Any,
        *,
        route: JointTrajectoryRoute | None,
        command_index: int | None,
        tick_index: int,
    ) -> Any:
        before = self.simulation.read()
        self.simulation.apply(action)
        self.simulation.step(render=self.render)
        after = self.simulation.read()
        if self.on_control_tick is not None:
            self.on_control_tick(
                JointControlTick(
                    tick_index=tick_index,
                    command_index=command_index,
                    route=route,
                    action=action,
                    state_before=before,
                    state_after=after,
                )
            )
        return after


@dataclass(frozen=True)
class NamedJointState:
    joint_position: tuple[float, float, float, float, float, float]
    joint_velocity: tuple[float, float, float, float, float, float]
    gripper_open_fraction: float


def measured_named_joint_state(
    state: Any,
    *,
    arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
    gripper_joint_names: Sequence[str] = GRIPPER_JOINT_NAMES,
    gripper_closed_position: float = 0.0,
    gripper_open_position: float = 0.04,
) -> NamedJointState:
    names = tuple(str(value) for value in getattr(state, "metadata", {}).get("joint_names", ()))
    positions = tuple(float(value) for value in getattr(state, "joint_positions", ()))
    velocities = tuple(float(value) for value in getattr(state, "joint_velocities", ()))
    if len(names) != len(positions) or len(names) != len(velocities):
        raise ValueError("simulation joint names, positions, and velocities do not align")
    index = {name: offset for offset, name in enumerate(names)}
    arm = tuple(str(name) for name in arm_joint_names)
    gripper = tuple(str(name) for name in gripper_joint_names)
    missing = [name for name in (*arm, *gripper) if name not in index]
    if missing:
        raise ValueError(f"simulation state is missing configured joints: {missing}")
    q = tuple(positions[index[name]] for name in arm)
    dq = tuple(velocities[index[name]] for name in arm)
    physical = sum(positions[index[name]] for name in gripper) / len(gripper)
    span = float(gripper_open_position) - float(gripper_closed_position)
    if not math.isfinite(span) or span <= 0.0:
        raise ValueError("gripper physical range must be finite and positive")
    fraction = min(1.0, max(0.0, (physical - gripper_closed_position) / span))
    if not all(math.isfinite(value) for value in (*q, *dq, fraction)):
        raise ValueError("measured joint state contains non-finite values")
    return NamedJointState(q, dq, fraction)  # type: ignore[arg-type]


def measured_body_velocity(state: Any) -> tuple[float, float, float]:
    raw = getattr(state, "metadata", {}).get("body_velocity")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        values = tuple(float(value) for value in raw[:3])
        if all(math.isfinite(value) for value in values):
            return values  # type: ignore[return-value]
    root = _finite_vector(state.robot_root_pose, 7, "robot_root_pose")
    velocity = _finite_vector(state.robot_root_velocity, 6, "robot_root_velocity")
    yaw = yaw_from_quaternion(root[3:])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return (
        cosine * velocity[0] + sine * velocity[1],
        -sine * velocity[0] + cosine * velocity[1],
        velocity[5],
    )


@dataclass(frozen=True)
class PlacementValidArea:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_surface: float
    maximum_center_height_above_surface_m: float = 0.20
    below_surface_tolerance_m: float = 0.01

    @classmethod
    def from_raw_task(cls, raw_task: Mapping[str, Any]) -> "PlacementValidArea":
        place = raw_task.get("place")
        region = place.get("placement_region") if isinstance(place, Mapping) else None
        if not isinstance(region, Mapping) or region.get("frame") != "world":
            raise ValueError("raw task needs a world-frame place.placement_region")
        return cls(
            x_min=float(region["x_min"]),
            x_max=float(region["x_max"]),
            y_min=float(region["y_min"]),
            y_max=float(region["y_max"]),
            z_surface=float(region["z_surface"]),
        )

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.z_surface,
            self.maximum_center_height_above_surface_m,
            self.below_surface_tolerance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement valid area must be finite")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("placement valid area bounds must be ordered")
        if self.maximum_center_height_above_surface_m <= 0.0 or self.below_surface_tolerance_m < 0.0:
            raise ValueError("placement height tolerances are invalid")

    def contains(self, object_pose_world: Sequence[float]) -> bool:
        pose = _finite_vector(object_pose_world, 7, "object_pose_world")
        return bool(
            self.x_min <= pose[0] <= self.x_max
            and self.y_min <= pose[1] <= self.y_max
            and self.z_surface - self.below_surface_tolerance_m <= pose[2]
            and pose[2] <= self.z_surface + self.maximum_center_height_above_surface_m
        )


@dataclass(frozen=True)
class IsaacTransferTruthUpdate:
    success: SuccessUpdate
    released: bool
    inside_target_valid_area: bool
    object_tcp_separation_m: float
    release_source: str


class IsaacTransferTruthAdapter:
    """Read evaluator-only Isaac truth without feeding it into policy or control."""

    def __init__(
        self,
        valid_area: PlacementValidArea,
        evaluator: TransferSuccessEvaluator | None = None,
        *,
        release_open_fraction: float = 0.80,
        minimum_object_tcp_separation_m: float = 0.06,
        arm_joint_names: Sequence[str] = ARM_JOINT_NAMES,
        gripper_joint_names: Sequence[str] = GRIPPER_JOINT_NAMES,
        gripper_closed_position: float = 0.0,
        gripper_open_position: float = 0.04,
    ) -> None:
        self.valid_area = valid_area
        self.evaluator = evaluator or TransferSuccessEvaluator()
        self.release_open_fraction = _unit_fraction(
            release_open_fraction, "release open threshold"
        )
        self.minimum_object_tcp_separation_m = float(minimum_object_tcp_separation_m)
        if not math.isfinite(self.minimum_object_tcp_separation_m) or self.minimum_object_tcp_separation_m <= 0.0:
            raise ValueError("release separation threshold must be finite and positive")
        self.arm_joint_names = tuple(arm_joint_names)
        self.gripper_joint_names = tuple(gripper_joint_names)
        self.gripper_closed_position = float(gripper_closed_position)
        self.gripper_open_position = float(gripper_open_position)

    def update(self, state: Any) -> IsaacTransferTruthUpdate:
        if state.object_pose is None or state.tcp_pose is None:
            raise ValueError("transfer evaluator requires object and TCP truth poses")
        object_pose = _finite_vector(state.object_pose, 7, "object_pose")
        tcp_pose = _finite_vector(state.tcp_pose, 7, "tcp_pose")
        separation = math.sqrt(
            sum((object_pose[index] - tcp_pose[index]) ** 2 for index in range(3))
        )
        joints = measured_named_joint_state(
            state,
            arm_joint_names=self.arm_joint_names,
            gripper_joint_names=self.gripper_joint_names,
            gripper_closed_position=self.gripper_closed_position,
            gripper_open_position=self.gripper_open_position,
        )
        fixed = getattr(state, "metadata", {}).get("grasp_fixed_joint_report")
        if isinstance(fixed, Mapping) and fixed.get("active") is True:
            released, release_source = False, "fixed_joint_active"
        elif isinstance(fixed, Mapping) and fixed.get("released") is True:
            released, release_source = True, "fixed_joint_released_report"
        else:
            released = bool(
                joints.gripper_open_fraction >= self.release_open_fraction
                and separation >= self.minimum_object_tcp_separation_m
            )
            release_source = "measured_gripper_and_object_tcp_separation"
        inside = self.valid_area.contains(object_pose)
        update = self.evaluator.update(
            float(state.timestamp),
            released=released,
            inside_target_valid_area=inside,
        )
        return IsaacTransferTruthUpdate(
            success=update,
            released=released,
            inside_target_valid_area=inside,
            object_tcp_separation_m=separation,
            release_source=release_source,
        )


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _unit_fraction(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return result


__all__ = [
    "ARM_JOINT_NAMES",
    "CONTROL_STRIDE_S",
    "GRIPPER_JOINT_NAMES",
    "IsaacJointActionAdapter",
    "IsaacJointTrajectorySystemExecutor",
    "IsaacTransferTruthAdapter",
    "IsaacTransferTruthUpdate",
    "JointControlTick",
    "JointNavigationConfig",
    "JointNavigationControl",
    "JointNavigationPlan",
    "JointSystemExecutionResult",
    "NamedJointState",
    "PCTDWAJointNavigationExecutor",
    "PlacementValidArea",
    "measured_body_velocity",
    "measured_named_joint_state",
]
