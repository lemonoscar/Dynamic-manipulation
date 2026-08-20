"""State-free inference session for the waypoint-runtime/v1 protocol."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Mapping

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    ROUTE_TOKENS,
    WaypointActionDomain,
    WaypointRoute,
    action_domain,
    waypoint_prompt,
)
from conveyor_bench.conveyorvla.waypoint_data import WaypointNormalizer
from conveyor_bench.conveyorvla.waypoint_model import ConveyorVLAWaypointPolicy
from conveyor_bench.conveyorvla.waypoint_protocol import (
    RECOVER_ROUTE,
    WaypointRequest,
    WaypointResponse,
)


@dataclass(frozen=True)
class WaypointInferenceTrace:
    request_id: str
    episode_id: str
    sequence_id: int
    assistant_prefix: str
    normalized_action: tuple[tuple[float, ...], ...] | None
    denormalized_action: tuple[tuple[float, ...], ...] | None
    action_valid_mask: tuple[bool, ...]
    normalization_clip_rate: float
    recover_reason: str | None
    model_elapsed_ms: float


@dataclass(frozen=True)
class WaypointInferenceResult:
    response: WaypointResponse
    trace: WaypointInferenceTrace


class WaypointInferenceSession:
    """Run both Qwen passes without accepting state, phase, or semantic history."""

    def __init__(
        self,
        policy: ConveyorVLAWaypointPolicy,
        normalizer: WaypointNormalizer,
        *,
        checkpoint_id: str,
        normalization_sha256: str,
        camera_calibration_id: str,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.checkpoint_id = _required(checkpoint_id, "checkpoint_id")
        self.normalization_sha256 = _required(
            normalization_sha256, "normalization_sha256"
        )
        self.camera_calibration_id = _required(
            camera_calibration_id, "camera_calibration_id"
        )
        self._last_sequence_by_episode: dict[str, int] = {}

    def infer(self, value: WaypointRequest | Mapping[str, Any]) -> WaypointInferenceResult:
        request = value if isinstance(value, WaypointRequest) else WaypointRequest.from_mapping(value)
        stale_reason = self._request_rejection(request)
        if stale_reason is not None:
            return self._recover(request, stale_reason, 0.0)
        self._last_sequence_by_episode[request.episode_id] = request.sequence_id
        started = time.perf_counter()
        try:
            prediction = self.policy.predict(
                [
                    {
                        "video": (request.head_images, request.wrist_images),
                        "lang": waypoint_prompt(request.instruction),
                    }
                ]
            )[0]
        except Exception as error:
            elapsed = (time.perf_counter() - started) * 1000.0
            return self._recover(request, f"model_inference_failed:{type(error).__name__}", elapsed)
        elapsed = (time.perf_counter() - started) * 1000.0
        decision = prediction.decision
        if not decision.valid or decision.route is None:
            return self._recover(
                request,
                decision.recover_reason or "model_recover",
                elapsed,
                decision_probs=decision.decision_probs,
                route_probs=decision.route_probs,
                confidence=decision.route_confidence,
            )
        if decision.route is WaypointRoute.DONE:
            response = WaypointResponse(
                request_id=request.request_id,
                sequence_id=request.sequence_id,
                route=WaypointRoute.DONE.value,
                route_token=None,
                action_domain=WaypointActionDomain.NONE.value,
                subtask="",
                route_confidence=decision.route_confidence,
                decision_probs=decision.decision_probs,
                route_probs=decision.route_probs,
                nav_waypoints_body=None,
                arm_targets_base=None,
                action_valid_mask=(),
                checkpoint_id=self.checkpoint_id,
                normalization_sha256=self.normalization_sha256,
                timing={"model_elapsed_ms": elapsed},
            )
            return WaypointInferenceResult(
                response=response,
                trace=WaypointInferenceTrace(
                    request_id=request.request_id,
                    episode_id=request.episode_id,
                    sequence_id=request.sequence_id,
                    assistant_prefix=decision.assistant_prefix,
                    normalized_action=None,
                    denormalized_action=None,
                    action_valid_mask=(),
                    normalization_clip_rate=0.0,
                    recover_reason=None,
                    model_elapsed_ms=elapsed,
                ),
            )
        normalized = prediction.normalized_action
        if normalized is None:
            return self._recover(request, "active_route_has_no_action", elapsed)
        try:
            denormalized = self.normalizer.denormalize(decision.route, normalized)
        except (ValueError, TypeError) as error:
            return self._recover(
                request, f"action_denormalization_failed:{type(error).__name__}", elapsed
            )
        domain = action_domain(decision.route)
        mask = (True,) * ACTION_HORIZON
        clip_rate = _normalization_clip_rate(normalized, domain)
        response = WaypointResponse(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            route=decision.route.value,
            route_token=ROUTE_TOKENS[decision.route],
            action_domain=domain.value,
            subtask=decision.subtask_text,
            route_confidence=decision.route_confidence,
            decision_probs=decision.decision_probs,
            route_probs=decision.route_probs,
            nav_waypoints_body=(
                denormalized if domain is WaypointActionDomain.NAVIGATION else None
            ),
            arm_targets_base=(
                denormalized if domain is WaypointActionDomain.MANIPULATION else None
            ),
            action_valid_mask=mask,
            checkpoint_id=self.checkpoint_id,
            normalization_sha256=self.normalization_sha256,
            action_units=(
                ("m", "m", "rad")
                if domain is WaypointActionDomain.NAVIGATION
                else ("m", "m", "m", "rad", "rad", "rad", "fraction")
            ),
            timing={"model_elapsed_ms": elapsed},
        )
        return WaypointInferenceResult(
            response=response,
            trace=WaypointInferenceTrace(
                request_id=request.request_id,
                episode_id=request.episode_id,
                sequence_id=request.sequence_id,
                assistant_prefix=decision.assistant_prefix,
                normalized_action=normalized,
                denormalized_action=denormalized,
                action_valid_mask=mask,
                normalization_clip_rate=clip_rate,
                recover_reason=None,
                model_elapsed_ms=elapsed,
            ),
        )

    def _request_rejection(self, request: WaypointRequest) -> str | None:
        if request.camera_calibration_id != self.camera_calibration_id:
            return "camera_calibration_mismatch"
        previous = self._last_sequence_by_episode.get(request.episode_id)
        if previous is not None and request.sequence_id <= previous:
            return "stale_or_replayed_sequence"
        return None

    def _recover(
        self,
        request: WaypointRequest,
        reason: str,
        elapsed_ms: float,
        *,
        decision_probs: Mapping[str, float] | None = None,
        route_probs: Mapping[str, float] | None = None,
        confidence: float = 0.0,
    ) -> WaypointInferenceResult:
        response = WaypointResponse(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            route=RECOVER_ROUTE,
            route_token=None,
            action_domain=WaypointActionDomain.NONE.value,
            subtask="",
            route_confidence=max(0.0, min(1.0, float(confidence))),
            decision_probs=(
                {"ACTION": 0.0, "DONE": 1.0}
                if decision_probs is None
                else decision_probs
            ),
            route_probs=(
                {
                    WaypointRoute.NAV_TO_SOURCE.value: 0.0,
                    WaypointRoute.PICK.value: 0.0,
                    WaypointRoute.NAV_TO_TARGET.value: 0.0,
                    WaypointRoute.PLACE.value: 0.0,
                }
                if route_probs is None
                else route_probs
            ),
            nav_waypoints_body=None,
            arm_targets_base=None,
            action_valid_mask=(),
            checkpoint_id=self.checkpoint_id,
            normalization_sha256=self.normalization_sha256,
            timing={"model_elapsed_ms": max(0.0, elapsed_ms)},
            recover_reason=reason,
        )
        return WaypointInferenceResult(
            response=response,
            trace=WaypointInferenceTrace(
                request_id=request.request_id,
                episode_id=request.episode_id,
                sequence_id=request.sequence_id,
                assistant_prefix="",
                normalized_action=None,
                denormalized_action=None,
                action_valid_mask=(),
                normalization_clip_rate=0.0,
                recover_reason=reason,
                model_elapsed_ms=max(0.0, elapsed_ms),
            ),
        )


def _normalization_clip_rate(
    action: tuple[tuple[float, ...], ...], domain: WaypointActionDomain
) -> float:
    width = 3 if domain is WaypointActionDomain.NAVIGATION else 7
    values = [float(item) for row in action for item in row[:width]]
    if not values or not all(math.isfinite(item) for item in values):
        return 1.0
    return sum(abs(item) > 1.0 for item in values) / len(values)


def _required(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


__all__ = [
    "WaypointInferenceResult",
    "WaypointInferenceSession",
    "WaypointInferenceTrace",
]
