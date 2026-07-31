"""Deterministic privileged-state oracle for ConveyorBench V1 sorting.

The oracle has no simulator dependency.  It consumes ground-truth task state
and emits a world-frame TCP target plus a body-frame base command.  A runtime
adapter remains responsible for IK, locomotion, physics, and recording.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .protocol import Pose, RobotMode

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
_ZERO_BASE_COMMAND: Vec3 = (0.0, 0.0, 0.0)


class OraclePhase(str, Enum):
    SETTLE = "settle"
    SELECT = "select"
    PREGRASP = "pregrasp"
    TRACK = "track"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    CARRY = "carry"
    PREPLACE = "preplace"
    PLACE_DESCEND = "place_descend"
    OPEN = "open"
    RETREAT = "retreat"
    VERIFY_PLACE = "verify_place"
    COMPLETE = "complete"
    FAILED = "failed"


def _validate_vector(value: Sequence[float], length: int, name: str) -> None:
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(not math.isfinite(float(component)) for component in value):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class OracleConfig:
    """Resolved geometry, timing, and mode for one target-to-goal task.

    ``goal_center_world`` is the desired final object-center position.
    ``grasp_offset_world`` points from the object center to the commanded TCP.
    """

    target_object_id: str
    goal_center_world: Vec3
    robot_mode: RobotMode = RobotMode.FIXED_BASE
    object_height_m: float = 0.08
    grasp_offset_world: Vec3 = (0.0, 0.0, 0.0)
    tcp_orientation_wxyz: Vec4 = (-1.0, 0.0, 0.0, 0.0)
    intercept_horizon_s: float = 0.12
    intercept_staging_y_world: float | None = None
    intercept_entry_tolerance_m: float = 0.05
    close_on_target_contact: bool = False
    pregrasp_clearance_m: float = 0.12
    safe_carry_clearance_m: float = 0.16
    position_tolerance_m: float = 0.015
    grasp_tolerance_m: float = 0.010
    settle_duration_s: float = 0.25
    select_duration_s: float = 0.10
    grasp_contact_dwell_s: float = 0.06
    preplace_dwell_s: float = 0.06
    placement_dwell_s: float = 0.50
    episode_timeout_s: float = 20.0
    phase_timeout_s: float = 5.0
    close_timeout_s: float = 1.20
    release_timeout_s: float = 1.00
    verify_timeout_s: float = 3.00
    mobile_select_base_command_body: Vec3 = (0.05, 0.0, 0.0)
    mobile_carry_base_command_body: Vec3 = (0.04, 0.0, 0.0)
    mobile_lift_target_world: Vec3 | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_object_id, str) or not self.target_object_id:
            raise ValueError("target_object_id cannot be empty")
        if not isinstance(self.robot_mode, RobotMode):
            raise ValueError("robot_mode must be a RobotMode")
        if not isinstance(self.close_on_target_contact, bool):
            raise ValueError("close_on_target_contact must be a bool")
        _validate_vector(self.goal_center_world, 3, "goal_center_world")
        _validate_vector(self.grasp_offset_world, 3, "grasp_offset_world")
        _validate_vector(self.tcp_orientation_wxyz, 4, "tcp_orientation_wxyz")
        _validate_vector(
            self.mobile_select_base_command_body,
            3,
            "mobile_select_base_command_body",
        )
        _validate_vector(
            self.mobile_carry_base_command_body,
            3,
            "mobile_carry_base_command_body",
        )
        if self.mobile_lift_target_world is not None:
            _validate_vector(
                self.mobile_lift_target_world,
                3,
                "mobile_lift_target_world",
            )
        if (
            self.intercept_staging_y_world is not None
            and not math.isfinite(self.intercept_staging_y_world)
        ):
            raise ValueError(
                "intercept_staging_y_world must be finite when provided"
            )
        quaternion_norm = math.sqrt(
            sum(float(component) ** 2 for component in self.tcp_orientation_wxyz)
        )
        if not math.isclose(
            quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5
        ):
            raise ValueError("tcp_orientation_wxyz must be a unit quaternion")

        positive_names = (
            "object_height_m",
            "intercept_entry_tolerance_m",
            "pregrasp_clearance_m",
            "safe_carry_clearance_m",
            "position_tolerance_m",
            "grasp_tolerance_m",
            "episode_timeout_s",
            "phase_timeout_s",
            "close_timeout_s",
            "release_timeout_s",
            "verify_timeout_s",
            "placement_dwell_s",
        )
        for name in positive_names:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        nonnegative_names = (
            "intercept_horizon_s",
            "settle_duration_s",
            "select_duration_s",
            "grasp_contact_dwell_s",
            "preplace_dwell_s",
        )
        for name in nonnegative_names:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.placement_dwell_s > self.verify_timeout_s:
            raise ValueError("placement_dwell_s cannot exceed verify_timeout_s")


@dataclass(frozen=True)
class OracleObservation:
    sim_time_s: float
    target_position_world: Vec3
    target_velocity_world: Vec3
    tcp_position_world: Vec3
    left_contact_object_ids: tuple[str, ...] = ()
    right_contact_object_ids: tuple[str, ...] = ()
    target_held: bool = False
    target_lifted: bool = False
    target_in_goal: bool = False
    target_released: bool = False
    target_settled: bool = False
    wrong_object_grasped: bool = False
    robot_fallen: bool = False
    forbidden_collision: bool = False
    target_crossed_exit: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.sim_time_s) or self.sim_time_s < 0:
            raise ValueError("sim_time_s must be finite and non-negative")
        _validate_vector(
            self.target_position_world, 3, "target_position_world"
        )
        _validate_vector(
            self.target_velocity_world, 3, "target_velocity_world"
        )
        _validate_vector(self.tcp_position_world, 3, "tcp_position_world")
        for name, object_ids in (
            ("left_contact_object_ids", self.left_contact_object_ids),
            ("right_contact_object_ids", self.right_contact_object_ids),
        ):
            if len(set(object_ids)) != len(object_ids):
                raise ValueError(f"{name} must not contain duplicates")
            if any(not isinstance(object_id, str) or not object_id for object_id in object_ids):
                raise ValueError(f"{name} must contain non-empty object ids")


@dataclass(frozen=True)
class OracleCommand:
    target_tcp_pose_world: Pose
    gripper_command: float
    base_command_body: Vec3
    phase: OraclePhase
    selected_object_id: str | None
    terminal: bool = False
    success: bool = False
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.gripper_command not in (0.0, 1.0):
            raise ValueError("gripper_command must be 1=open or 0=close")
        _validate_vector(self.base_command_body, 3, "base_command_body")
        if self.success and not self.terminal:
            raise ValueError("success requires a terminal command")
        if self.success and self.failure_reason is not None:
            raise ValueError("successful command cannot have a failure_reason")
        if self.phase is OraclePhase.FAILED and not self.failure_reason:
            raise ValueError("failed command requires a failure_reason")


class DynamicSortOracle:
    """Finite-state privileged oracle for one moving target and assigned goal."""

    _PREGRASP_PHASES = {
        OraclePhase.SETTLE,
        OraclePhase.SELECT,
        OraclePhase.PREGRASP,
        OraclePhase.TRACK,
        OraclePhase.DESCEND,
        OraclePhase.CLOSE,
    }
    _HELD_PHASES = {
        OraclePhase.LIFT,
        OraclePhase.CARRY,
        OraclePhase.PREPLACE,
        OraclePhase.PLACE_DESCEND,
    }

    def __init__(self, config: OracleConfig):
        self.config = config
        self.reset(sim_time_s=0.0)

    def reset(self, *, sim_time_s: float = 0.0) -> None:
        if not math.isfinite(sim_time_s) or sim_time_s < 0:
            raise ValueError("sim_time_s must be finite and non-negative")
        self.phase = OraclePhase.SETTLE
        self._episode_started_at = sim_time_s
        self._phase_started_at = sim_time_s
        self._last_step_at = sim_time_s
        self._selected_object_id: str | None = None
        self._contact_started_at: float | None = None
        self._placement_started_at: float | None = None
        self._failure_reason: str | None = None
        self._terminal_gripper = 1.0
        self._carry_height_m = self._safe_carry_z(
            self.config.goal_center_world[2]
        )
        self._last_target_position = self._goal_high_position()
        self._lift_target_position = self._last_target_position

    def step(self, observation: OracleObservation) -> OracleCommand:
        if observation.sim_time_s < self._last_step_at:
            raise ValueError("observation sim_time_s cannot move backwards")
        self._last_step_at = observation.sim_time_s

        if self.phase is OraclePhase.COMPLETE:
            return self._command(
                self._last_target_position,
                gripper_command=1.0,
                terminal=True,
                success=True,
            )
        if self.phase is OraclePhase.FAILED:
            return self._command(
                self._last_target_position,
                gripper_command=self._terminal_gripper,
                terminal=True,
                failure_reason=self._failure_reason,
            )

        failure_reason = self._global_failure_reason(observation)
        if failure_reason is not None:
            return self._fail(observation, failure_reason)

        phase_elapsed = observation.sim_time_s - self._phase_started_at
        predicted_target = self._predicted_target_position(observation)
        pregrasp_target = self._pregrasp_position(predicted_target)
        staging_target = self._staging_pregrasp_position(predicted_target)
        grasp_target = self._grasp_position(predicted_target)

        if self.phase is OraclePhase.SETTLE:
            if phase_elapsed >= self.config.settle_duration_s:
                self._selected_object_id = self.config.target_object_id
                self._transition(OraclePhase.SELECT, observation.sim_time_s)
            return self._command(staging_target, gripper_command=1.0)

        if self.phase is OraclePhase.SELECT:
            if phase_elapsed >= self.config.select_duration_s:
                self._transition(OraclePhase.PREGRASP, observation.sim_time_s)
            return self._command(staging_target, gripper_command=1.0)

        if self.phase is OraclePhase.PREGRASP:
            if (
                self._near(observation.tcp_position_world, staging_target)
                and self._target_entered_intercept_window(predicted_target)
            ):
                self._transition(OraclePhase.TRACK, observation.sim_time_s)
                return self._command(staging_target, gripper_command=1.0)
            return self._command(staging_target, gripper_command=1.0)

        if self.phase is OraclePhase.TRACK:
            if self.config.intercept_staging_y_world is not None:
                # A staged interception does not chase the part laterally.
                # Prediction decides when to leave the wait pose; the belt
                # then carries the part through the fixed descend line.
                self._transition(OraclePhase.DESCEND, observation.sim_time_s)
                return self._command(
                    self._intercept_grasp_position(predicted_target),
                    gripper_command=1.0,
                )
            if self._near(observation.tcp_position_world, pregrasp_target):
                self._transition(OraclePhase.DESCEND, observation.sim_time_s)
                return self._command(grasp_target, gripper_command=1.0)
            return self._command(pregrasp_target, gripper_command=1.0)

        if self.phase is OraclePhase.DESCEND:
            target_contact = (
                self.config.target_object_id
                in (
                    observation.left_contact_object_ids
                    + observation.right_contact_object_ids
                )
            )
            if (
                self.config.close_on_target_contact
                and target_contact
            ) or self._near(
                observation.tcp_position_world,
                grasp_target,
                tolerance=self.config.grasp_tolerance_m,
            ):
                self._transition(OraclePhase.CLOSE, observation.sim_time_s)
                return self._command(grasp_target, gripper_command=0.0)
            return self._command(grasp_target, gripper_command=1.0)

        if self.phase is OraclePhase.CLOSE:
            target_contact = self._has_target_bilateral_contact(observation)
            if target_contact and observation.target_held:
                if self._contact_started_at is None:
                    self._contact_started_at = observation.sim_time_s
                if (
                    observation.sim_time_s - self._contact_started_at
                    >= self.config.grasp_contact_dwell_s
                ):
                    safe_z = self._safe_carry_z(
                        observation.target_position_world[2]
                    )
                    self._carry_height_m = safe_z
                    self._lift_target_position = (
                        tuple(self.config.mobile_lift_target_world)
                        if (
                            self.config.robot_mode
                            is RobotMode.WHOLE_BODY_POLICY
                            and self.config.mobile_lift_target_world
                            is not None
                        )
                        else (
                            observation.tcp_position_world[0],
                            observation.tcp_position_world[1],
                            safe_z,
                        )
                    )
                    self._transition(OraclePhase.LIFT, observation.sim_time_s)
                    return self._command(
                        self._lift_target_position, gripper_command=0.0
                    )
            else:
                self._contact_started_at = None
            if phase_elapsed >= self.config.close_timeout_s:
                return self._fail(observation, "grasp_timeout")
            return self._command(grasp_target, gripper_command=0.0)

        if self.phase is OraclePhase.LIFT:
            if (
                observation.target_lifted
                and self._near(
                    observation.tcp_position_world,
                    self._lift_target_position,
                )
            ):
                self._transition(OraclePhase.CARRY, observation.sim_time_s)
            return self._command(
                self._lift_target_position, gripper_command=0.0
            )

        high_goal = self._goal_high_position()
        if self.phase is OraclePhase.CARRY:
            if self._near(observation.tcp_position_world, high_goal):
                self._transition(OraclePhase.PREPLACE, observation.sim_time_s)
            return self._command(high_goal, gripper_command=0.0)

        if self.phase is OraclePhase.PREPLACE:
            if (
                self._near(observation.tcp_position_world, high_goal)
                and phase_elapsed >= self.config.preplace_dwell_s
            ):
                self._transition(
                    OraclePhase.PLACE_DESCEND, observation.sim_time_s
                )
                return self._command(
                    self._place_position(), gripper_command=0.0
                )
            return self._command(high_goal, gripper_command=0.0)

        if self.phase is OraclePhase.PLACE_DESCEND:
            place_target = self._place_position()
            if self._near(
                observation.tcp_position_world,
                place_target,
                tolerance=self.config.grasp_tolerance_m,
            ):
                self._transition(OraclePhase.OPEN, observation.sim_time_s)
                return self._command(place_target, gripper_command=1.0)
            return self._command(place_target, gripper_command=0.0)

        if self.phase is OraclePhase.OPEN:
            if observation.target_released:
                self._transition(OraclePhase.RETREAT, observation.sim_time_s)
                return self._command(high_goal, gripper_command=1.0)
            if phase_elapsed >= self.config.release_timeout_s:
                return self._fail(observation, "release_timeout")
            return self._command(
                self._place_position(), gripper_command=1.0
            )

        if self.phase is OraclePhase.RETREAT:
            if self._near(observation.tcp_position_world, high_goal):
                self._transition(
                    OraclePhase.VERIFY_PLACE, observation.sim_time_s
                )
            return self._command(high_goal, gripper_command=1.0)

        if self.phase is OraclePhase.VERIFY_PLACE:
            placement_valid = (
                observation.target_released
                and observation.target_in_goal
                and observation.target_settled
            )
            if placement_valid:
                if self._placement_started_at is None:
                    self._placement_started_at = observation.sim_time_s
                if (
                    observation.sim_time_s - self._placement_started_at
                    >= self.config.placement_dwell_s
                ):
                    self._transition(
                        OraclePhase.COMPLETE, observation.sim_time_s
                    )
                    return self._command(
                        high_goal,
                        gripper_command=1.0,
                        terminal=True,
                        success=True,
                    )
            else:
                self._placement_started_at = None
            if phase_elapsed >= self.config.verify_timeout_s:
                if not observation.target_released:
                    reason = "release_not_verified"
                elif not observation.target_in_goal:
                    reason = "wrong_goal"
                else:
                    reason = "placement_not_settled"
                return self._fail(observation, reason)
            return self._command(high_goal, gripper_command=1.0)

        raise RuntimeError(f"unhandled oracle phase: {self.phase}")

    def _global_failure_reason(
        self, observation: OracleObservation
    ) -> str | None:
        if observation.robot_fallen:
            return "robot_fallen"
        if observation.forbidden_collision:
            return "forbidden_collision"
        if observation.wrong_object_grasped or self._has_wrong_bilateral_contact(
            observation
        ):
            return "wrong_object"
        if (
            self.phase in self._PREGRASP_PHASES
            and observation.target_crossed_exit
        ):
            return "target_missed"
        if self.phase in self._HELD_PHASES and not observation.target_held:
            return "target_dropped"
        if (
            observation.sim_time_s - self._episode_started_at
            >= self.config.episode_timeout_s
        ):
            return "episode_timeout"
        if (
            observation.sim_time_s - self._phase_started_at
            >= self.config.phase_timeout_s
        ):
            return f"phase_timeout:{self.phase.value}"
        return None

    def _has_target_bilateral_contact(
        self, observation: OracleObservation
    ) -> bool:
        target_id = self.config.target_object_id
        return (
            target_id in observation.left_contact_object_ids
            and target_id in observation.right_contact_object_ids
        )

    def _has_wrong_bilateral_contact(
        self, observation: OracleObservation
    ) -> bool:
        bilateral_ids = set(observation.left_contact_object_ids).intersection(
            observation.right_contact_object_ids
        )
        bilateral_ids.discard(self.config.target_object_id)
        return bool(bilateral_ids)

    def _predicted_target_position(
        self, observation: OracleObservation
    ) -> Vec3:
        return tuple(
            float(position)
            + float(velocity) * self.config.intercept_horizon_s
            for position, velocity in zip(
                observation.target_position_world,
                observation.target_velocity_world,
                strict=True,
            )
        )

    def _grasp_position(self, target_position: Sequence[float]) -> Vec3:
        return tuple(
            float(position) + float(offset)
            for position, offset in zip(
                target_position,
                self.config.grasp_offset_world,
                strict=True,
            )
        )

    def _pregrasp_position(self, target_position: Sequence[float]) -> Vec3:
        grasp = self._grasp_position(target_position)
        return (
            grasp[0],
            grasp[1],
            grasp[2] + self.config.pregrasp_clearance_m,
        )

    def _staging_pregrasp_position(
        self, target_position: Sequence[float]
    ) -> Vec3:
        target = self._pregrasp_position(target_position)
        if self.config.intercept_staging_y_world is None:
            return target
        return (
            target[0],
            self.config.intercept_staging_y_world,
            target[2],
        )

    def _intercept_grasp_position(
        self, target_position: Sequence[float]
    ) -> Vec3:
        target = self._grasp_position(target_position)
        staging_y = self.config.intercept_staging_y_world
        if staging_y is None:
            return target
        return (target[0], staging_y, target[2])

    def _target_entered_intercept_window(
        self, target_position: Sequence[float]
    ) -> bool:
        staging_y = self.config.intercept_staging_y_world
        if staging_y is None:
            return True
        return (
            abs(float(target_position[1]) - staging_y)
            <= self.config.intercept_entry_tolerance_m
        )

    def _safe_carry_z(self, current_object_z: float) -> float:
        return (
            max(float(current_object_z), self.config.goal_center_world[2])
            + 0.5 * self.config.object_height_m
            + self.config.safe_carry_clearance_m
            + max(0.0, self.config.grasp_offset_world[2])
        )

    def _goal_high_position(self) -> Vec3:
        place = self._place_position()
        return (place[0], place[1], self._carry_height_m)

    def _place_position(self) -> Vec3:
        return tuple(
            float(center) + float(offset)
            for center, offset in zip(
                self.config.goal_center_world,
                self.config.grasp_offset_world,
                strict=True,
            )
        )

    def _near(
        self,
        actual: Sequence[float],
        target: Sequence[float],
        *,
        tolerance: float | None = None,
    ) -> bool:
        threshold = (
            self.config.position_tolerance_m
            if tolerance is None
            else tolerance
        )
        distance = math.sqrt(
            sum(
                (float(actual_value) - float(target_value)) ** 2
                for actual_value, target_value in zip(
                    actual, target, strict=True
                )
            )
        )
        return distance <= threshold

    def _base_command(self) -> Vec3:
        if self.config.robot_mode is RobotMode.FIXED_BASE:
            return _ZERO_BASE_COMMAND
        if self.phase is OraclePhase.SELECT:
            return self.config.mobile_select_base_command_body
        if self.phase is OraclePhase.CARRY:
            return self.config.mobile_carry_base_command_body
        return _ZERO_BASE_COMMAND

    def _command(
        self,
        target_position: Vec3,
        *,
        gripper_command: float,
        terminal: bool = False,
        success: bool = False,
        failure_reason: str | None = None,
    ) -> OracleCommand:
        self._last_target_position = tuple(
            float(component) for component in target_position
        )
        return OracleCommand(
            target_tcp_pose_world=Pose(
                self._last_target_position,
                self.config.tcp_orientation_wxyz,
            ),
            gripper_command=gripper_command,
            base_command_body=(
                _ZERO_BASE_COMMAND if terminal else self._base_command()
            ),
            phase=self.phase,
            selected_object_id=self._selected_object_id,
            terminal=terminal,
            success=success,
            failure_reason=failure_reason,
        )

    def _transition(self, phase: OraclePhase, sim_time_s: float) -> None:
        self.phase = phase
        self._phase_started_at = sim_time_s
        if phase is not OraclePhase.CLOSE:
            self._contact_started_at = None
        if phase is not OraclePhase.VERIFY_PLACE:
            self._placement_started_at = None

    def _fail(
        self, observation: OracleObservation, reason: str
    ) -> OracleCommand:
        self._failure_reason = reason
        self._terminal_gripper = 0.0 if observation.target_held else 1.0
        self._last_target_position = observation.tcp_position_world
        self._transition(OraclePhase.FAILED, observation.sim_time_s)
        return self._command(
            self._last_target_position,
            gripper_command=self._terminal_gripper,
            terminal=True,
            failure_reason=reason,
        )
