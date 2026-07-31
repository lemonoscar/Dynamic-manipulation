"""Deterministic privileged-state oracle for the V0 conveyor task.

The oracle is deliberately independent from Isaac Sim.  It consumes a compact
observation and returns a Cartesian gripper-center target.  The simulator owns
inverse kinematics, physics stepping, recording, and success verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, sqrt


class OraclePhase(str, Enum):
    SETTLE = "settle"
    PREGRASP = "pregrasp"
    TRACK = "track"
    DESCEND = "descend"
    CLOSE = "close"
    LIFT = "lift"
    HOLD = "hold"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class OracleConfig:
    belt_top_z: float = 0.50
    intercept_xy: tuple[float, float] = (0.70, 0.0)
    transport_direction_xy: tuple[float, float] = (1.0, 0.0)
    grasp_target_offset_xy: tuple[float, float] = (0.0, 0.0)
    grasp_orientation_wxyz: tuple[float, float, float, float] = (
        -1.0,
        0.0,
        0.0,
        0.0,
    )
    pregrasp_clearance: float = 0.15
    grasp_center_offset: float = 0.005
    lift_clearance: float = 0.14
    prediction_horizon: float = 0.12
    expected_transport_speed: float = 0.0
    settle_duration: float = 0.35
    tracking_trigger_distance: float = 0.24
    track_target_speed_mps: float = 0.30
    track_position_tolerance: float = 0.02
    approach_trigger_distance: float = 0.13
    pregrasp_position_tolerance: float = 0.010
    descend_position_tolerance: float = 0.008
    descend_settle_duration: float = 0.04
    descend_speed_mps: float = 0.10
    close_duration: float = 0.10
    contact_settle_duration: float = 0.06
    close_timeout: float = 1.20
    lift_timeout: float = 2.0
    lift_speed_mps: float = 0.10
    hold_duration: float = 1.05
    open_gripper: float = 0.044
    closed_gripper: float = 0.0

    def __post_init__(self) -> None:
        if len(self.intercept_xy) != 2:
            raise ValueError("intercept_xy must contain two values")
        if len(self.transport_direction_xy) != 2:
            raise ValueError("transport_direction_xy must contain two values")
        if len(self.grasp_target_offset_xy) != 2:
            raise ValueError("grasp_target_offset_xy must contain two values")
        norm = sqrt(sum(component**2 for component in self.transport_direction_xy))
        if not isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("transport_direction_xy must be a unit vector")
        if len(self.grasp_orientation_wxyz) != 4:
            raise ValueError("grasp_orientation_wxyz must contain four values")
        if self.tracking_trigger_distance < self.approach_trigger_distance:
            raise ValueError(
                "tracking_trigger_distance must cover approach_trigger_distance"
            )
        if self.track_target_speed_mps <= 0 or self.track_position_tolerance <= 0:
            raise ValueError("track speed and position tolerance must be positive")


@dataclass(frozen=True)
class OracleObservation:
    sim_time: float
    object_position: tuple[float, float, float]
    object_velocity: tuple[float, float, float]
    gripper_position: tuple[float, float, float]
    object_crossed_exit: bool = False
    robot_fallen: bool = False
    object_lifted: bool = False
    left_contact: bool = False
    right_contact: bool = False
    secure_grasp: bool = False
    forbidden_collision: bool = False


@dataclass(frozen=True)
class OracleCommand:
    target_position: tuple[float, float, float]
    # wxyz; chosen by the task so the fingers can straddle the transport axis.
    target_orientation: tuple[float, float, float, float]
    gripper_opening: float
    phase: OraclePhase
    terminal: bool = False
    success: bool = False
    failure_reason: str | None = None


class DynamicPickOracle:
    """Small receding-horizon state machine for one moving object."""

    def __init__(self, config: OracleConfig | None = None):
        self.config = config or OracleConfig()
        self.phase = OraclePhase.SETTLE
        self._phase_started_at = 0.0
        self._last_target = (
            self.config.intercept_xy[0],
            self.config.intercept_xy[1],
            self.config.belt_top_z + 0.20,
        )
        self._close_anchor = self._last_target
        self._lift_anchor = self._last_target
        self._descend_anchor = self._last_target
        self._descend_start_z = self._last_target[2]
        self._alignment_started_at: float | None = None
        self._contact_started_at: float | None = None
        self._last_step_at = 0.0

    def reset(
        self,
        *,
        sim_time: float,
        object_position: tuple[float, float, float],
    ) -> None:
        self.phase = OraclePhase.SETTLE
        self._phase_started_at = sim_time
        object_x, object_y, object_z = object_position
        intercept_x, intercept_y = _replace_transport_progress(
            (object_x, object_y),
            _transport_progress(
                self.config.intercept_xy,
                self.config.transport_direction_xy,
            ),
            self.config.transport_direction_xy,
        )
        self._last_target = (
            intercept_x,
            intercept_y,
            object_z + self.config.pregrasp_clearance,
        )
        self._close_anchor = self._last_target
        self._lift_anchor = self._last_target
        self._descend_anchor = self._last_target
        self._descend_start_z = self._last_target[2]
        self._alignment_started_at = None
        self._contact_started_at = None
        self._last_step_at = sim_time

    def step(self, observation: OracleObservation) -> OracleCommand:
        step_dt = max(0.0, observation.sim_time - self._last_step_at)
        self._last_step_at = observation.sim_time
        if self.phase in {OraclePhase.COMPLETE, OraclePhase.FAILED}:
            return self._command(
                self._last_target,
                self.config.closed_gripper,
                terminal=True,
                success=self.phase is OraclePhase.COMPLETE,
                failure_reason=None if self.phase is OraclePhase.COMPLETE else "oracle_failed",
            )

        if observation.robot_fallen:
            return self._fail(observation, "robot_fallen")
        if observation.forbidden_collision:
            return self._fail(observation, "forbidden_collision")
        if observation.object_crossed_exit:
            return self._fail(observation, "belt_exit")

        object_x, object_y, object_z = observation.object_position
        # Contact impulses are not transport velocity.  Using the measured
        # object velocity after the first touch creates positive feedback: the
        # fingers push the object, then chase the resulting rebound.  The V0
        # oracle knows the commanded conveyor speed and uses that stable prior.
        predicted_x, predicted_y = _advance_xy(
            (object_x, object_y),
            self.config.transport_direction_xy,
            self.config.expected_transport_speed
            * self.config.prediction_horizon,
        )
        intercept_progress = _transport_progress(
            self.config.intercept_xy,
            self.config.transport_direction_xy,
        )
        object_progress = _transport_progress(
            (object_x, object_y),
            self.config.transport_direction_xy,
        )
        pregrasp_x, pregrasp_y = _replace_transport_progress(
            (object_x, object_y),
            intercept_progress,
            self.config.transport_direction_xy,
        )
        elapsed = observation.sim_time - self._phase_started_at

        if self.phase is OraclePhase.SETTLE:
            target = (
                pregrasp_x,
                pregrasp_y,
                object_z + self.config.pregrasp_clearance,
            )
            if elapsed >= self.config.settle_duration:
                self._transition(OraclePhase.PREGRASP, observation.sim_time)
            return self._command(target, self.config.open_gripper)

        if self.phase is OraclePhase.PREGRASP:
            target = (
                pregrasp_x,
                pregrasp_y,
                object_z + self.config.pregrasp_clearance,
            )
            at_pregrasp = (
                _distance(observation.gripper_position, target)
                <= self.config.pregrasp_position_tolerance
            )
            object_in_tracking_reach = (
                object_progress
                >= intercept_progress - self.config.tracking_trigger_distance
            )
            if at_pregrasp and object_in_tracking_reach:
                self._transition(OraclePhase.TRACK, observation.sim_time)
            return self._command(target, self.config.open_gripper)

        if self.phase is OraclePhase.TRACK:
            target_x, target_y = _move_toward_xy(
                self._last_target[:2],
                (predicted_x, predicted_y),
                self.config.track_target_speed_mps * step_dt,
            )
            target = (
                target_x,
                target_y,
                object_z + self.config.pregrasp_clearance,
            )
            ready_to_descend = (
                object_progress
                >= intercept_progress - self.config.approach_trigger_distance
                and _horizontal_distance(observation.gripper_position, target)
                <= self.config.track_position_tolerance
            )
            if ready_to_descend:
                grasp_x, grasp_y = _offset_xy(
                    (predicted_x, predicted_y),
                    self.config.grasp_target_offset_xy,
                )
                self._descend_anchor = (
                    grasp_x,
                    grasp_y,
                    object_z + self.config.grasp_center_offset,
                )
                self._descend_start_z = target[2]
                self._transition(OraclePhase.DESCEND, observation.sim_time)
            return self._command(target, self.config.open_gripper)

        if self.phase is OraclePhase.DESCEND:
            target_x, target_y = _advance_xy(
                self._descend_anchor[:2],
                self.config.transport_direction_xy,
                self.config.expected_transport_speed * elapsed,
            )
            target = (
                target_x,
                target_y,
                max(
                    self._descend_anchor[2],
                    self._descend_start_z
                    - self.config.descend_speed_mps * elapsed,
                ),
            )
            horizontal_error = _horizontal_distance(
                observation.gripper_position,
                target,
            )
            vertical_error = abs(observation.gripper_position[2] - target[2])
            at_grasp_depth = (
                target[2] <= self._descend_anchor[2] + 1.0e-6
            )
            aligned = (
                at_grasp_depth
                and horizontal_error <= self.config.descend_position_tolerance
                and vertical_error <= self.config.descend_position_tolerance
            )
            if aligned and self._alignment_started_at is None:
                self._alignment_started_at = observation.sim_time
            elif not aligned:
                self._alignment_started_at = None
            if (
                self._alignment_started_at is not None
                and observation.sim_time - self._alignment_started_at
                >= self.config.descend_settle_duration
            ):
                self._close_anchor = target
                self._transition(OraclePhase.CLOSE, observation.sim_time)
                return self._command(target, self.config.closed_gripper)
            return self._command(target, self.config.open_gripper)

        if self.phase is OraclePhase.CLOSE:
            target_x, target_y = _advance_xy(
                self._close_anchor[:2],
                self.config.transport_direction_xy,
                self.config.expected_transport_speed * elapsed,
            )
            target = (
                target_x,
                target_y,
                self._close_anchor[2],
            )
            both_contacts = observation.left_contact and observation.right_contact
            if both_contacts and self._contact_started_at is None:
                self._contact_started_at = observation.sim_time
            elif not both_contacts:
                self._contact_started_at = None
            contact_is_stable = (
                self._contact_started_at is not None
                and observation.sim_time - self._contact_started_at
                >= self.config.contact_settle_duration
            )
            if elapsed >= self.config.close_duration and contact_is_stable:
                self._lift_anchor = target
                self._transition(OraclePhase.LIFT, observation.sim_time)
            elif elapsed >= self.config.close_timeout:
                return self._fail(observation, "grasp_not_secured")
            return self._command(target, self.config.closed_gripper)

        if self.phase is OraclePhase.LIFT:
            target_z = min(
                self.config.belt_top_z + self.config.lift_clearance,
                self._lift_anchor[2] + self.config.lift_speed_mps * elapsed,
            )
            target = (
                self._lift_anchor[0],
                self._lift_anchor[1],
                target_z,
            )
            if observation.secure_grasp:
                self._lift_anchor = target
                self._transition(OraclePhase.HOLD, observation.sim_time)
            elif elapsed >= self.config.lift_timeout:
                return self._fail(observation, "lift_timeout")
            return self._command(target, self.config.closed_gripper)

        if self.phase is OraclePhase.HOLD:
            target = self._lift_anchor
            if not observation.secure_grasp:
                return self._fail(observation, "dropped_during_hold")
            if elapsed >= self.config.hold_duration:
                self._transition(OraclePhase.COMPLETE, observation.sim_time)
                return self._command(
                    target,
                    self.config.closed_gripper,
                    terminal=True,
                    success=True,
                )
            return self._command(target, self.config.closed_gripper)

        raise RuntimeError(f"Unhandled oracle phase: {self.phase}")

    def _command(
        self,
        target: tuple[float, float, float],
        gripper_opening: float,
        *,
        terminal: bool = False,
        success: bool = False,
        failure_reason: str | None = None,
    ) -> OracleCommand:
        self._last_target = target
        return OracleCommand(
            target_position=target,
            target_orientation=self.config.grasp_orientation_wxyz,
            gripper_opening=gripper_opening,
            phase=self.phase,
            terminal=terminal,
            success=success,
            failure_reason=failure_reason,
        )

    def _transition(self, phase: OraclePhase, sim_time: float) -> None:
        self.phase = phase
        self._phase_started_at = sim_time

    def _fail(self, observation: OracleObservation, reason: str) -> OracleCommand:
        self._transition(OraclePhase.FAILED, observation.sim_time)
        return self._command(
            self._last_target,
            self.config.closed_gripper,
            terminal=True,
            success=False,
            failure_reason=reason,
        )


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _horizontal_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _transport_progress(
    xy: tuple[float, float],
    direction_xy: tuple[float, float],
) -> float:
    return xy[0] * direction_xy[0] + xy[1] * direction_xy[1]


def _advance_xy(
    xy: tuple[float, float],
    direction_xy: tuple[float, float],
    distance: float,
) -> tuple[float, float]:
    return (
        xy[0] + direction_xy[0] * distance,
        xy[1] + direction_xy[1] * distance,
    )


def _replace_transport_progress(
    xy: tuple[float, float],
    target_progress: float,
    direction_xy: tuple[float, float],
) -> tuple[float, float]:
    correction = target_progress - _transport_progress(xy, direction_xy)
    return _advance_xy(xy, direction_xy, correction)


def _offset_xy(
    xy: tuple[float, float],
    offset_xy: tuple[float, float],
) -> tuple[float, float]:
    return xy[0] + offset_xy[0], xy[1] + offset_xy[1]


def _move_toward_xy(
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    max_distance: float,
) -> tuple[float, float]:
    delta_x = target_xy[0] - current_xy[0]
    delta_y = target_xy[1] - current_xy[1]
    distance = sqrt(delta_x * delta_x + delta_y * delta_y)
    if distance <= max_distance or distance <= 1.0e-12:
        return target_xy
    scale = max_distance / distance
    return current_xy[0] + delta_x * scale, current_xy[1] + delta_y * scale
