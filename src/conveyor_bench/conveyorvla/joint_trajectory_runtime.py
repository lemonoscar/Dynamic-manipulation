"""Runtime primitives for full NAV references and direct Mani joint chunks."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    MANIPULATION_STRIDE_S,
    SUCCESS_DWELL_S,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
    direct_joint_targets,
    fixed_action,
    joint_trajectory_prompt,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import JointTrajectoryNormalizer


class RouteCommitStatus(str, Enum):
    INITIAL_PENDING = "INITIAL_PENDING"
    SWITCH_PENDING = "SWITCH_PENDING"
    COMMITTED = "COMMITTED"
    UNCHANGED = "UNCHANGED"


@dataclass(frozen=True)
class RouteCommitResult:
    status: RouteCommitStatus
    committed_route: JointTrajectoryRoute | None
    candidate_route: JointTrajectoryRoute
    execute_action: bool
    confirmation_count: int


class RouteCommitter:
    """Commit a model route only after two qualifying fresh observations."""

    def __init__(self, *, confirmation_observations: int = 2) -> None:
        if confirmation_observations != 2:
            raise ValueError("joint-trajectory v1 requires exactly two confirmations")
        self.confirmation_observations = confirmation_observations
        self.committed_route: JointTrajectoryRoute | None = None
        self.pending_route: JointTrajectoryRoute | None = None
        self.pending_count = 0
        self._last_sequence_id: int | None = None

    def observe(
        self, route_probs: Mapping[str, float], *, sequence_id: int
    ) -> RouteCommitResult:
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        if self._last_sequence_id is not None and sequence_id <= self._last_sequence_id:
            raise ValueError("route confirmation requires a fresh increasing observation")
        self._last_sequence_id = sequence_id
        probabilities = _route_probabilities(route_probs)
        maximum = max(probabilities.values())
        candidate = (
            self.committed_route
            if self.committed_route is not None
            and math.isclose(
                probabilities[self.committed_route], maximum, rel_tol=0.0, abs_tol=1.0e-12
            )
            else max(JointTrajectoryRoute, key=lambda route: probabilities[route])
        )
        if self.committed_route is None:
            confirmed = self._advance_pending(candidate)
            if not confirmed:
                return RouteCommitResult(
                    RouteCommitStatus.INITIAL_PENDING,
                    None,
                    candidate,
                    False,
                    self.pending_count,
                )
            self.committed_route = candidate
            self._clear_pending()
            return RouteCommitResult(
                RouteCommitStatus.COMMITTED,
                candidate,
                candidate,
                True,
                self.confirmation_observations,
            )

        committed = self.committed_route
        if candidate is committed or probabilities[candidate] <= probabilities[committed]:
            self._clear_pending()
            return RouteCommitResult(
                RouteCommitStatus.UNCHANGED,
                committed,
                candidate,
                True,
                0,
            )
        confirmed = self._advance_pending(candidate)
        if not confirmed:
            return RouteCommitResult(
                RouteCommitStatus.SWITCH_PENDING,
                committed,
                candidate,
                False,
                self.pending_count,
            )
        self.committed_route = candidate
        self._clear_pending()
        return RouteCommitResult(
            RouteCommitStatus.COMMITTED,
            candidate,
            candidate,
            True,
            self.confirmation_observations,
        )

    def reset(self) -> None:
        self.committed_route = None
        self._clear_pending()
        self._last_sequence_id = None

    def _advance_pending(self, candidate: JointTrajectoryRoute) -> bool:
        if self.pending_route is candidate:
            self.pending_count += 1
        else:
            self.pending_route = candidate
            self.pending_count = 1
        return self.pending_count >= self.confirmation_observations

    def _clear_pending(self) -> None:
        self.pending_route = None
        self.pending_count = 0


@dataclass(frozen=True)
class NavigationReference:
    points_query_body: tuple[tuple[float, float, float], ...]
    local_goal_query_body: tuple[float, float, float]
    stride_s: float = 0.20


def navigation_reference(
    predicted_action: Sequence[Sequence[float]],
) -> NavigationReference:
    """Pass all ten model points to PCT/DWA; the tenth is the local goal."""

    rows = fixed_action(predicted_action, JointTrajectoryDomain.NAVIGATION)
    return NavigationReference(
        points_query_body=tuple(
            (float(row[0]), float(row[1]), float(row[2])) for row in rows
        ),
        local_goal_query_body=(
            float(rows[-1][0]),
            float(rows[-1][1]),
            float(rows[-1][2]),
        ),
    )


@dataclass(frozen=True)
class JointSafetyLimits:
    lower: tuple[float, float, float, float, float, float]
    upper: tuple[float, float, float, float, float, float]
    max_rate_rad_s: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value)
            for values in (self.lower, self.upper, self.max_rate_rad_s)
            for value in values
        ):
            raise ValueError("joint safety limits must be finite")
        if any(lower >= upper for lower, upper in zip(self.lower, self.upper, strict=True)):
            raise ValueError("joint lower limits must be below upper limits")
        if any(value <= 0.0 for value in self.max_rate_rad_s):
            raise ValueError("joint rate limits must be positive")


@dataclass(frozen=True)
class DirectJointCommand:
    index: int
    joint_position: tuple[float, float, float, float, float, float]
    gripper_open_fraction: float
    base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration_s: float = MANIPULATION_STRIDE_S


@dataclass(frozen=True)
class DirectJointChunk:
    commands: tuple[DirectJointCommand, ...]
    position_saturation_count: int
    rate_saturation_count: int
    gripper_saturation_count: int

    @property
    def saturation_rate(self) -> float:
        saturated = (
            self.position_saturation_count
            + self.rate_saturation_count
            + self.gripper_saturation_count
        )
        return saturated / (ACTION_HORIZON * 7)


class DirectJointTrajectoryExecutor:
    """Prepare ten sequential low-level commands without IK or pose planning."""

    def __init__(self, limits: JointSafetyLimits) -> None:
        self.limits = limits

    def prepare(
        self,
        query_joint_position: Sequence[float],
        predicted_action: Sequence[Sequence[float]],
    ) -> DirectJointChunk:
        rows = direct_joint_targets(query_joint_position, predicted_action)
        previous = _finite_vector(query_joint_position, 6, "query_joint_position")
        commands = []
        position_saturation = 0
        rate_saturation = 0
        gripper_saturation = 0
        for index, row in enumerate(rows):
            bounded = []
            for axis in range(6):
                position = min(self.limits.upper[axis], max(self.limits.lower[axis], row[axis]))
                position_saturation += int(not math.isclose(position, row[axis], rel_tol=0.0, abs_tol=1.0e-12))
                maximum_delta = self.limits.max_rate_rad_s[axis] * MANIPULATION_STRIDE_S
                rate_bounded = min(previous[axis] + maximum_delta, max(previous[axis] - maximum_delta, position))
                rate_saturation += int(not math.isclose(rate_bounded, position, rel_tol=0.0, abs_tol=1.0e-12))
                bounded.append(rate_bounded)
            raw_gripper = float(row[6])
            gripper = min(1.0, max(0.0, raw_gripper))
            gripper_saturation += int(
                not math.isclose(gripper, raw_gripper, rel_tol=0.0, abs_tol=1.0e-12)
            )
            joint_position = tuple(bounded)
            commands.append(
                DirectJointCommand(
                    index=index,
                    joint_position=joint_position,  # type: ignore[arg-type]
                    gripper_open_fraction=gripper,
                )
            )
            previous = joint_position
        return DirectJointChunk(
            commands=tuple(commands),
            position_saturation_count=position_saturation,
            rate_saturation_count=rate_saturation,
            gripper_saturation_count=gripper_saturation,
        )

    def hold(
        self,
        joint_position: Sequence[float],
        gripper_open_fraction: float,
    ) -> DirectJointCommand:
        target = _finite_vector(joint_position, 6, "hold joint_position")
        gripper = float(gripper_open_fraction)
        if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
            raise ValueError("hold gripper must be within [0,1]")
        bounded = tuple(
            min(upper, max(lower, value))
            for value, lower, upper in zip(
                target, self.limits.lower, self.limits.upper, strict=True
            )
        )
        return DirectJointCommand(
            index=0,
            joint_position=bounded,  # type: ignore[arg-type]
            gripper_open_fraction=gripper,
        )


@dataclass(frozen=True)
class SuccessUpdate:
    success: bool
    dwell_s: float
    released: bool
    inside_target_valid_area: bool


class TransferSuccessEvaluator:
    """Evaluator-only release + target-area dwell; orientation is irrelevant."""

    def __init__(self, *, required_dwell_s: float = SUCCESS_DWELL_S) -> None:
        if not math.isfinite(required_dwell_s) or required_dwell_s <= 0.0:
            raise ValueError("success dwell must be positive and finite")
        self.required_dwell_s = float(required_dwell_s)
        self._dwell_started_s: float | None = None
        self._last_timestamp_s: float | None = None
        self.success = False

    def update(
        self,
        timestamp_s: float,
        *,
        released: bool,
        inside_target_valid_area: bool,
    ) -> SuccessUpdate:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("evaluator timestamp must be finite and non-negative")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("evaluator timestamps must be strictly increasing")
        self._last_timestamp_s = timestamp
        condition = bool(released) and bool(inside_target_valid_area)
        if not condition:
            self._dwell_started_s = None
            dwell = 0.0
        else:
            if self._dwell_started_s is None:
                self._dwell_started_s = timestamp
            dwell = timestamp - self._dwell_started_s
            if dwell >= self.required_dwell_s:
                self.success = True
        return SuccessUpdate(
            success=self.success,
            dwell_s=dwell,
            released=bool(released),
            inside_target_valid_area=bool(inside_target_valid_area),
        )

    def reset(self) -> None:
        self._dwell_started_s = None
        self._last_timestamp_s = None
        self.success = False


@dataclass(frozen=True)
class JointTrajectoryRuntimeRequest:
    request_id: str
    episode_id: str
    sequence_id: int
    instruction: str
    head_images: tuple[Any, Any]
    wrist_images: tuple[Any, Any]
    joint_position: tuple[float, float, float, float, float, float]
    joint_velocity: tuple[float, float, float, float, float, float]
    gripper_open_fraction: float

    def __post_init__(self) -> None:
        if not self.request_id.strip() or not self.episode_id.strip() or not self.instruction.strip():
            raise ValueError("runtime request IDs and instruction must be non-empty")
        if isinstance(self.sequence_id, bool) or not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValueError("runtime request sequence_id must be non-negative")
        if len(self.head_images) != 2 or len(self.wrist_images) != 2:
            raise ValueError("runtime request needs head/wrist [t-0.20,t]")
        _finite_vector(self.joint_position, 6, "joint_position")
        _finite_vector(self.joint_velocity, 6, "joint_velocity")
        if not math.isfinite(self.gripper_open_fraction) or not 0.0 <= self.gripper_open_fraction <= 1.0:
            raise ValueError("runtime request gripper must be within [0,1]")

    @property
    def mani_state(self) -> tuple[float, ...]:
        return (*self.joint_position, *self.joint_velocity, self.gripper_open_fraction)


@dataclass(frozen=True)
class JointTrajectoryRuntimeStep:
    request_id: str
    sequence_id: int
    predicted_route: JointTrajectoryRoute | None
    committed_route: JointTrajectoryRoute | None
    commit_status: RouteCommitStatus | None
    route_probs: Mapping[str, float]
    subtask: str
    action_domain: JointTrajectoryDomain | None
    navigation: NavigationReference | None
    manipulation: DirectJointChunk | None
    hold: DirectJointCommand | None
    pass2_executed: bool
    checkpoint_id: str
    normalization_sha256: str
    elapsed_ms: float
    recover_reason: str | None = None


class JointTrajectoryInferenceSession:
    """Run Pass 1, temporal commit, then Pass 2 only when action may execute."""

    def __init__(
        self,
        policy: Any,
        normalizer: JointTrajectoryNormalizer,
        joint_executor: DirectJointTrajectoryExecutor,
        *,
        checkpoint_id: str,
        normalization_sha256: str,
    ) -> None:
        if not str(checkpoint_id).strip() or not str(normalization_sha256).strip():
            raise ValueError("runtime checkpoint and normalization identities are required")
        self.policy = policy
        self.normalizer = normalizer
        self.joint_executor = joint_executor
        self.checkpoint_id = str(checkpoint_id)
        self.normalization_sha256 = str(normalization_sha256)
        self._committers: dict[str, RouteCommitter] = {}
        self._last_sequences: dict[str, int] = {}
        self._last_arm_targets: dict[str, DirectJointCommand] = {}

    def step(self, request: JointTrajectoryRuntimeRequest) -> JointTrajectoryRuntimeStep:
        started = time.perf_counter()
        previous = self._last_sequences.get(request.episode_id)
        if previous is not None and request.sequence_id <= previous:
            raise ValueError("runtime request is stale or replayed")
        self._last_sequences[request.episode_id] = request.sequence_id
        example = {
            "video": (request.head_images, request.wrist_images),
            "lang": joint_trajectory_prompt(request.instruction),
            # The Qwen interface selects only video/lang.  This normalized token
            # is read only if Pass 2 commits a Mani route.
            "mani_state": self.normalizer.normalize_mani_state(request.mani_state),
        }
        decision = self.policy.predict_routes([example])[0]
        if not decision.valid or decision.route is None:
            return self._hold_result(
                request,
                decision.route_probs,
                predicted_route=None,
                committed_route=self._committer(request.episode_id).committed_route,
                status=None,
                subtask="",
                reason=decision.recover_reason or "invalid_pass1_route",
                started=started,
            )
        committer = self._committer(request.episode_id)
        commit = committer.observe(decision.route_probs, sequence_id=request.sequence_id)
        if not commit.execute_action:
            return self._hold_result(
                request,
                decision.route_probs,
                predicted_route=decision.route,
                committed_route=commit.committed_route,
                status=commit.status,
                subtask=decision.subtask_text,
                reason=None,
                started=started,
            )
        if commit.committed_route is not decision.route:
            # A probability tie can preserve a prior committed route even when
            # tokenizer argmax chose another equal token.  Never condition the
            # action head on a prefix for the wrong committed route.
            return self._hold_result(
                request,
                decision.route_probs,
                predicted_route=decision.route,
                committed_route=commit.committed_route,
                status=commit.status,
                subtask=decision.subtask_text,
                reason="committed_route_prefix_mismatch",
                started=started,
            )
        normalized_action = self.policy.predict_actions([example], [decision])[0]
        raw_action = self.normalizer.denormalize_action(decision.route, normalized_action)
        domain = action_domain(decision.route)
        navigation = None
        manipulation = None
        if domain is JointTrajectoryDomain.NAVIGATION:
            navigation = navigation_reference(raw_action)
        else:
            manipulation = self.joint_executor.prepare(
                request.joint_position, raw_action
            )
            final = manipulation.commands[-1]
            self._last_arm_targets[request.episode_id] = self.joint_executor.hold(
                final.joint_position, final.gripper_open_fraction
            )
        arm_hold = (
            self._arm_hold(request)
            if domain is JointTrajectoryDomain.NAVIGATION
            else None
        )
        return JointTrajectoryRuntimeStep(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            predicted_route=decision.route,
            committed_route=commit.committed_route,
            commit_status=commit.status,
            route_probs=decision.route_probs,
            subtask=decision.subtask_text,
            action_domain=domain,
            navigation=navigation,
            manipulation=manipulation,
            # NAV owns only the base reference.  Keep the arm at its measured
            # query posture instead of leaving the inactive actuator undefined.
            hold=arm_hold,
            pass2_executed=True,
            checkpoint_id=self.checkpoint_id,
            normalization_sha256=self.normalization_sha256,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def reset_episode(self, episode_id: str) -> None:
        self._committers.pop(str(episode_id), None)
        self._last_sequences.pop(str(episode_id), None)
        self._last_arm_targets.pop(str(episode_id), None)

    def _committer(self, episode_id: str) -> RouteCommitter:
        return self._committers.setdefault(str(episode_id), RouteCommitter())

    def _hold_result(
        self,
        request: JointTrajectoryRuntimeRequest,
        route_probs: Mapping[str, float],
        *,
        predicted_route: JointTrajectoryRoute | None,
        committed_route: JointTrajectoryRoute | None,
        status: RouteCommitStatus | None,
        subtask: str,
        reason: str | None,
        started: float,
    ) -> JointTrajectoryRuntimeStep:
        return JointTrajectoryRuntimeStep(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            predicted_route=predicted_route,
            committed_route=committed_route,
            commit_status=status,
            route_probs=route_probs,
            subtask=subtask,
            action_domain=None,
            navigation=None,
            manipulation=None,
            hold=self._arm_hold(request),
            pass2_executed=False,
            checkpoint_id=self.checkpoint_id,
            normalization_sha256=self.normalization_sha256,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            recover_reason=reason,
        )

    def _arm_hold(
        self, request: JointTrajectoryRuntimeRequest
    ) -> DirectJointCommand:
        previous = self._last_arm_targets.get(request.episode_id)
        if previous is None:
            previous = self.joint_executor.hold(
                request.joint_position, request.gripper_open_fraction
            )
        hold = self.joint_executor.hold(
            previous.joint_position, previous.gripper_open_fraction
        )
        self._last_arm_targets[request.episode_id] = hold
        return hold

def _route_probabilities(
    route_probs: Mapping[str, float],
) -> Mapping[JointTrajectoryRoute, float]:
    if set(route_probs) != {route.value for route in JointTrajectoryRoute}:
        raise ValueError("route probabilities must contain exactly four active routes")
    result = {
        route: float(route_probs[route.value]) for route in JointTrajectoryRoute
    }
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("route probabilities must be finite and non-negative")
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
        raise ValueError("route probabilities must sum to one")
    return result


def _finite_vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


__all__ = [
    "DirectJointChunk",
    "DirectJointCommand",
    "DirectJointTrajectoryExecutor",
    "JointSafetyLimits",
    "JointTrajectoryInferenceSession",
    "JointTrajectoryRuntimeRequest",
    "JointTrajectoryRuntimeStep",
    "NavigationReference",
    "RouteCommitResult",
    "RouteCommitStatus",
    "RouteCommitter",
    "SuccessUpdate",
    "TransferSuccessEvaluator",
    "navigation_reference",
]
