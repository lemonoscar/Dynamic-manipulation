"""Strict state-free wire protocol for ConveyorVLA Waypoint Policy v1."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    CAMERA_CALIBRATION_ID,
    LABEL_FRAME_ID,
    ROUTE_TOKENS,
    RUNTIME_PROTOCOL_VERSION,
    WaypointActionDomain,
    WaypointRoute,
    action_domain,
)


RECOVER_ROUTE = "RECOVER"
FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "phase",
        "operation",
        "locked_route",
        "locked_subtask",
        "state",
        "state28",
        "robot_state",
        "observation.state",
        "base_pose",
        "tcp_pose",
        "joint_positions",
        "joint_velocities",
        "object_state",
        "target_pose",
        "target_tcp_pose",
        "previous_subtask",
        "subtask_history",
        "history",
    }
)


class WaypointProtocolError(ValueError):
    """A request or response violates waypoint-runtime/v1."""


@dataclass(frozen=True)
class WaypointRequest:
    request_id: str
    episode_id: str
    sequence_id: int
    instruction: str
    head_images: tuple[Any, Any]
    wrist_images: tuple[Any, Any]
    camera_calibration_id: str = CAMERA_CALIBRATION_ID
    protocol_version: str = RUNTIME_PROTOCOL_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WaypointRequest":
        if not isinstance(value, Mapping):
            raise WaypointProtocolError("waypoint request must be an object")
        leaked = sorted(_forbidden_keys(value))
        if leaked:
            raise WaypointProtocolError(
                "waypoint request contains forbidden model context: " + ", ".join(leaked)
            )
        if value.get("protocol_version") != RUNTIME_PROTOCOL_VERSION:
            raise WaypointProtocolError("waypoint request protocol version is incompatible")
        images = value.get("images")
        if not isinstance(images, Mapping) or set(images) != {"head", "wrist"}:
            raise WaypointProtocolError("waypoint request images must contain only head and wrist")
        head = _image_pair(images["head"], "head")
        wrist = _image_pair(images["wrist"], "wrist")
        request_id = _nonempty(value.get("request_id"), "request_id")
        episode_id = _nonempty(value.get("episode_id"), "episode_id")
        instruction = _nonempty(value.get("instruction"), "instruction")
        calibration = _nonempty(
            value.get("camera_calibration_id"), "camera_calibration_id"
        )
        sequence_id = value.get("sequence_id")
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0:
            raise WaypointProtocolError("waypoint sequence_id must be a non-negative integer")
        return cls(
            request_id=request_id,
            episode_id=episode_id,
            sequence_id=sequence_id,
            instruction=instruction,
            head_images=head,
            wrist_images=wrist,
            camera_calibration_id=calibration,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "sequence_id": self.sequence_id,
            "instruction": self.instruction,
            "images": {
                "head": list(self.head_images),
                "wrist": list(self.wrist_images),
            },
            "camera_calibration_id": self.camera_calibration_id,
        }


@dataclass(frozen=True)
class WaypointResponse:
    request_id: str
    sequence_id: int
    route: str
    route_token: str | None
    action_domain: str
    subtask: str
    route_confidence: float
    decision_probs: Mapping[str, float]
    route_probs: Mapping[str, float]
    nav_waypoints_body: tuple[tuple[float, float, float], ...] | None
    arm_targets_base: tuple[tuple[float, ...], ...] | None
    action_valid_mask: tuple[bool, ...]
    checkpoint_id: str
    normalization_sha256: str
    label_frame_id: str = LABEL_FRAME_ID
    action_units: tuple[str, ...] = ()
    timing: Mapping[str, float] = field(default_factory=dict)
    recover_reason: str | None = None
    protocol_version: str = RUNTIME_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        _validate_response(self)

    @property
    def terminal(self) -> bool:
        return self.route in {WaypointRoute.DONE.value, RECOVER_ROUTE}

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "sequence_id": self.sequence_id,
            "route": self.route,
            "route_token": self.route_token,
            "action_domain": self.action_domain,
            "subtask": self.subtask,
            "route_confidence": self.route_confidence,
            "decision_probs": dict(self.decision_probs),
            "route_probs": dict(self.route_probs),
            "nav_waypoints_body": (
                None
                if self.nav_waypoints_body is None
                else [list(value) for value in self.nav_waypoints_body]
            ),
            "arm_targets_base": (
                None
                if self.arm_targets_base is None
                else [list(value) for value in self.arm_targets_base]
            ),
            "action_valid_mask": list(self.action_valid_mask),
            "checkpoint_id": self.checkpoint_id,
            "normalization_sha256": self.normalization_sha256,
            "label_frame_id": self.label_frame_id,
            "action_units": list(self.action_units),
            "timing": dict(self.timing),
            "recover_reason": self.recover_reason,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "WaypointResponse":
        if not isinstance(value, Mapping):
            raise WaypointProtocolError("waypoint response must be an object")
        nav = value.get("nav_waypoints_body")
        arm = value.get("arm_targets_base")
        return cls(
            protocol_version=str(value.get("protocol_version", "")),
            request_id=str(value.get("request_id", "")),
            sequence_id=value.get("sequence_id"),
            route=str(value.get("route", "")),
            route_token=value.get("route_token"),
            action_domain=str(value.get("action_domain", "")),
            subtask=str(value.get("subtask", "")),
            route_confidence=float(value.get("route_confidence", math.nan)),
            decision_probs=_probabilities(value.get("decision_probs"), "decision_probs"),
            route_probs=_probabilities(value.get("route_probs"), "route_probs"),
            nav_waypoints_body=(
                None if nav is None else _action_rows(nav, 3, "nav_waypoints_body")
            ),
            arm_targets_base=(
                None if arm is None else _action_rows(arm, 7, "arm_targets_base")
            ),
            action_valid_mask=tuple(bool(item) for item in value.get("action_valid_mask", ())),
            checkpoint_id=str(value.get("checkpoint_id", "")),
            normalization_sha256=str(value.get("normalization_sha256", "")),
            label_frame_id=str(value.get("label_frame_id", "")),
            action_units=tuple(str(item) for item in value.get("action_units", ())),
            timing=_timing(value.get("timing")),
            recover_reason=(
                None if value.get("recover_reason") is None else str(value["recover_reason"])
            ),
        )


def _validate_response(value: WaypointResponse) -> None:
    if value.protocol_version != RUNTIME_PROTOCOL_VERSION:
        raise WaypointProtocolError("waypoint response protocol version is incompatible")
    _nonempty(value.request_id, "request_id")
    if isinstance(value.sequence_id, bool) or not isinstance(value.sequence_id, int) or value.sequence_id < 0:
        raise WaypointProtocolError("waypoint response sequence_id is invalid")
    _nonempty(value.checkpoint_id, "checkpoint_id")
    _nonempty(value.normalization_sha256, "normalization_sha256")
    if value.label_frame_id != LABEL_FRAME_ID:
        raise WaypointProtocolError("waypoint response label frame is incompatible")
    if not math.isfinite(value.route_confidence) or not 0.0 <= value.route_confidence <= 1.0:
        raise WaypointProtocolError("waypoint route confidence must be within [0,1]")
    _validate_probability_keys(value.decision_probs, {"ACTION", "DONE"}, "decision_probs")
    _validate_probability_keys(
        value.route_probs,
        {route.value for route in WaypointRoute if route is not WaypointRoute.DONE},
        "route_probs",
        allow_zero_sum=value.route in {WaypointRoute.DONE.value, RECOVER_ROUTE},
    )
    if value.route in {WaypointRoute.DONE.value, RECOVER_ROUTE}:
        if (
            value.route_token is not None
            or value.action_domain != WaypointActionDomain.NONE.value
            or value.nav_waypoints_body is not None
            or value.arm_targets_base is not None
            or value.action_valid_mask
            or value.action_units
        ):
            raise WaypointProtocolError("terminal waypoint response must contain no action")
        if value.route == RECOVER_ROUTE and not value.recover_reason:
            raise WaypointProtocolError("RECOVER response requires a reason")
        return
    try:
        route = WaypointRoute(value.route)
    except ValueError as error:
        raise WaypointProtocolError(f"unsupported waypoint route: {value.route!r}") from error
    if route is WaypointRoute.DONE:
        raise AssertionError("DONE handled above")
    if value.route_token != ROUTE_TOKENS[route]:
        raise WaypointProtocolError("waypoint route token does not match route")
    domain = action_domain(route)
    if value.action_domain != domain.value:
        raise WaypointProtocolError("waypoint action domain does not match route")
    length = len(value.action_valid_mask)
    if not 1 <= length <= ACTION_HORIZON or not any(value.action_valid_mask):
        raise WaypointProtocolError("active waypoint mask must contain a valid prefix")
    if any(not left and right for left, right in zip(value.action_valid_mask, value.action_valid_mask[1:])):
        raise WaypointProtocolError("waypoint action mask must be a true prefix")
    if domain is WaypointActionDomain.NAVIGATION:
        if value.nav_waypoints_body is None or value.arm_targets_base is not None:
            raise WaypointProtocolError("NAV response must contain only body waypoints")
        _validate_rows(value.nav_waypoints_body, length, 3, "nav_waypoints_body")
        if value.action_units != ("m", "m", "rad"):
            raise WaypointProtocolError("NAV response units are incompatible")
    else:
        if value.arm_targets_base is None or value.nav_waypoints_body is not None:
            raise WaypointProtocolError("ARM response must contain only TCP targets")
        _validate_rows(value.arm_targets_base, length, 7, "arm_targets_base")
        if value.action_units != ("m", "m", "m", "rad", "rad", "rad", "fraction"):
            raise WaypointProtocolError("ARM response units are incompatible")
        if any(not 0.0 <= row[6] <= 1.0 for row in value.arm_targets_base):
            raise WaypointProtocolError("ARM gripper target must be within [0,1]")


def _forbidden_keys(value: Any, prefix: str = "") -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower()
            path = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_REQUEST_KEYS or path in FORBIDDEN_REQUEST_KEYS:
                result.add(path)
            result.update(_forbidden_keys(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            result.update(_forbidden_keys(item, prefix))
    return result


def _image_pair(value: Any, name: str) -> tuple[Any, Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence) or len(value) != 2:
        raise WaypointProtocolError(f"{name} images must contain exactly [t-0.20,t]")
    return value[0], value[1]


def _nonempty(value: Any, name: str) -> str:
    result = str(value).strip() if value is not None else ""
    if not result:
        raise WaypointProtocolError(f"waypoint {name} must be non-empty")
    return result


def _action_rows(value: Any, width: int, name: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise WaypointProtocolError(f"{name} must be a sequence")
    rows = []
    for index, row in enumerate(value):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != width:
            raise WaypointProtocolError(f"{name}[{index}] must contain {width} values")
        values = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in values):
            raise WaypointProtocolError(f"{name}[{index}] contains non-finite values")
        rows.append(values)
    return tuple(rows)


def _validate_rows(rows: Sequence[Sequence[float]], length: int, width: int, name: str) -> None:
    if len(rows) != length or any(len(row) != width for row in rows):
        raise WaypointProtocolError(f"{name} shape does not match action mask")
    if not all(math.isfinite(float(item)) for row in rows for item in row):
        raise WaypointProtocolError(f"{name} contains non-finite values")


def _probabilities(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise WaypointProtocolError(f"{name} must be an object")
    return {str(key): float(item) for key, item in value.items()}


def _validate_probability_keys(
    value: Mapping[str, float],
    keys: set[str],
    name: str,
    *,
    allow_zero_sum: bool = False,
) -> None:
    if set(value) != keys or any(
        not math.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0
        for item in value.values()
    ):
        raise WaypointProtocolError(f"{name} keys or values are invalid")
    total = sum(float(item) for item in value.values())
    if abs(total - 1.0) > 1.0e-4 and not (allow_zero_sum and abs(total) <= 1.0e-4):
        raise WaypointProtocolError(f"{name} must sum to one")


def _timing(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WaypointProtocolError("waypoint timing must be an object")
    result = {str(key): float(item) for key, item in value.items()}
    if not all(math.isfinite(item) and item >= 0.0 for item in result.values()):
        raise WaypointProtocolError("waypoint timing values must be finite and non-negative")
    return result


__all__ = [
    "FORBIDDEN_REQUEST_KEYS",
    "RECOVER_ROUTE",
    "WaypointProtocolError",
    "WaypointRequest",
    "WaypointResponse",
]
