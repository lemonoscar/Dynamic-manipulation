"""Minimal localhost transport contract for online M0-Mobile inference."""

from __future__ import annotations

import base64
import io
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

from conveyor_bench.m0_mobile import (
    DEFAULT_CONFIG_PATH,
    M0MobileError,
    M0MobileNormalizer,
    load_m0_mobile_config,
)
from conveyor_bench.v1.exporters import M0_MOBILE_STATE_LAYOUT


ONLINE_SCHEMA_VERSION = "conveyor-bench-m0-online-v1"
STATE_STATISTICS_SCHEMA_VERSION = "conveyor-bench-m0-mobile-state-stats-v1"
PROFILE = "m0_mobile_v1"
CAMERA_IDS = ("head_rgb", "wrist_rgb")
STATE_DIM = 28
ACTION_HORIZON = 16
ACTION_DIM = 10
PREGRASP_WORKSPACE_LIMITS_BASE = (
    (None, 0.622),
    (-0.060, None),
    (0.250, None),
)
MAX_JPEG_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 9 * 1024 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


class M0OnlineError(M0MobileError):
    """Raised when the online protocol fails closed."""


@dataclass(frozen=True)
class M0InferRequest:
    request_id: str
    sequence_id: int
    instruction: str
    state28: tuple[float, ...]
    jpeg_images: tuple[bytes, bytes]
    seed: int


@dataclass(frozen=True)
class M0InferenceResult:
    request_id: str
    sequence_id: int
    normalized_actions: tuple[tuple[float, ...], ...]
    physical_actions: tuple[tuple[float, ...], ...]
    server_inference_ms: float
    round_trip_ms: float
    seed: int


def build_state28(
    root_linear_velocity_body: Sequence[float],
    root_angular_velocity_body: Sequence[float],
    projected_gravity_body: Sequence[float],
    arm_joint_positions: Sequence[float],
    arm_joint_velocities: Sequence[float],
    tcp_position_base: Sequence[float],
    tcp_rotation_vector_base: Sequence[float],
    gripper_open_fraction: float,
) -> tuple[float, ...]:
    """Build the frozen M0 state in the same order as the dataset exporter."""

    gripper = _finite_number(gripper_open_fraction, "gripper_open_fraction")
    if not 0.0 <= gripper <= 1.0:
        raise M0OnlineError("gripper_open_fraction must be within [0, 1]")
    state = (
        _finite_vector(root_linear_velocity_body, 3, "root linear velocity")
        + _finite_vector(root_angular_velocity_body, 3, "root angular velocity")
        + _finite_vector(projected_gravity_body, 3, "projected gravity")
        + _finite_vector(arm_joint_positions, 6, "arm joint positions")
        + _finite_vector(arm_joint_velocities, 6, "arm joint velocities")
        + _finite_vector(tcp_position_base, 3, "TCP position")
        + _finite_vector(tcp_rotation_vector_base, 3, "TCP rotation vector")
        + (gripper,)
    )
    if len(state) != len(M0_MOBILE_STATE_LAYOUT):
        raise AssertionError("M0-Mobile state layout and builder disagree")
    return state


def build_live_state28(
    root_lin_b: Sequence[float],
    root_ang_b: Sequence[float],
    gravity_b: Sequence[float],
    arm_pos6: Sequence[float],
    arm_vel6: Sequence[float],
    tcp_xyz: Sequence[float],
    tcp_wxyz: Sequence[float],
    gripper_fraction: float,
) -> tuple[float, ...]:
    """Build state28 directly from the tensors exposed by the Isaac runtime."""

    quaternion = _finite_vector(tcp_wxyz, 4, "TCP quaternion")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
        raise M0OnlineError("TCP quaternion must be unit length")
    q = tuple(component / norm for component in quaternion)
    if q[0] < 0.0:
        q = tuple(-component for component in q)
    vector_norm = math.sqrt(sum(component * component for component in q[1:]))
    if vector_norm <= 1.0e-12:
        rotation_vector = (0.0, 0.0, 0.0)
    else:
        angle = 2.0 * math.atan2(vector_norm, max(0.0, q[0]))
        rotation_vector = tuple(component * angle / vector_norm for component in q[1:])
    return build_state28(
        root_lin_b,
        root_ang_b,
        gravity_b,
        arm_pos6,
        arm_vel6,
        tcp_xyz,
        rotation_vector,
        gripper_fraction,
    )


def guard_pregrasp_tcp_target(
    position_base: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[str, ...]]:
    """Clamp one diagnostic pregrasp target to the audited X5 workspace."""

    target = list(_finite_vector(position_base, 3, "pregrasp TCP target"))
    clipped: list[str] = []
    for index, (axis, limits) in enumerate(
        zip("xyz", PREGRASP_WORKSPACE_LIMITS_BASE, strict=True)
    ):
        low, high = limits
        guarded = max(low, target[index]) if low is not None else target[index]
        guarded = min(high, guarded) if high is not None else guarded
        if guarded != target[index]:
            target[index] = guarded
            clipped.append(axis)
    return (target[0], target[1], target[2]), tuple(clipped)


def encode_rgb_jpeg(image: Any, *, quality: int = 85) -> bytes:
    """Encode a PIL/numpy RGB image, or validate already encoded JPEG bytes."""

    if isinstance(image, bytes):
        return _jpeg_bytes(image)
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 95:
        raise M0OnlineError("JPEG quality must be an integer within [1, 95]")
    try:
        from PIL import Image
    except ImportError as error:
        raise M0OnlineError("Pillow is required for online JPEG encoding") from error
    try:
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            if all(hasattr(image, name) for name in ("detach", "cpu", "numpy")):
                image = image.detach().cpu().numpy()
            if getattr(image, "ndim", None) == 4 and image.shape[0] == 1:
                image = image[0]
            rgb = Image.fromarray(image).convert("RGB")
        stream = io.BytesIO()
        rgb.save(stream, format="JPEG", quality=quality)
    except (TypeError, ValueError, OSError) as error:
        raise M0OnlineError(f"cannot encode RGB image as JPEG: {error}") from error
    return _jpeg_bytes(stream.getvalue())


def decode_rgb_jpeg(payload: bytes) -> Any:
    """Decode a bounded JPEG into a loaded RGB PIL image."""

    data = _jpeg_bytes(payload)
    try:
        from PIL import Image
    except ImportError as error:
        raise M0OnlineError("Pillow is required for online JPEG decoding") from error
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or image.width * image.height > 16_000_000:
                raise M0OnlineError("camera payload is not a bounded JPEG image")
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise M0OnlineError(f"cannot decode camera JPEG: {error}") from error


def parse_infer_request(payload: Any) -> M0InferRequest:
    request = _exact_object(
        payload,
        {
            "schema_version",
            "request_id",
            "sequence_id",
            "instruction",
            "state28",
            "images",
            "seed",
        },
        "infer request",
    )
    if request["schema_version"] != ONLINE_SCHEMA_VERSION:
        raise M0OnlineError("infer request has an unsupported schema_version")
    request_id = _request_id(request["request_id"])
    sequence_id = _bounded_integer(request["sequence_id"], "sequence_id", 0, 2**63 - 1)
    seed = _bounded_integer(request["seed"], "seed", 0, 2**31 - 1)
    instruction = request["instruction"]
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4096:
        raise M0OnlineError("instruction must be a non-empty string of at most 4096 characters")
    images = request["images"]
    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence) or len(images) != 2:
        raise M0OnlineError("images must contain head_rgb then wrist_rgb")
    decoded: list[bytes] = []
    for expected_camera, raw_image in zip(CAMERA_IDS, images, strict=True):
        item = _exact_object(
            raw_image, {"camera_id", "encoding", "data_base64"}, "camera image"
        )
        if item["camera_id"] != expected_camera or item["encoding"] != "jpeg":
            raise M0OnlineError("images must be JPEGs ordered as head_rgb then wrist_rgb")
        encoded = item["data_base64"]
        if not isinstance(encoded, str) or len(encoded) > (MAX_JPEG_BYTES * 4 // 3 + 8):
            raise M0OnlineError("camera data_base64 is invalid or too large")
        try:
            decoded.append(_jpeg_bytes(base64.b64decode(encoded, validate=True)))
        except (ValueError, base64.binascii.Error) as error:
            raise M0OnlineError("camera data_base64 is not valid base64") from error
    return M0InferRequest(
        request_id=request_id,
        sequence_id=sequence_id,
        instruction=instruction.strip(),
        state28=_finite_vector(request["state28"], STATE_DIM, "state28"),
        jpeg_images=(decoded[0], decoded[1]),
        seed=seed,
    )


def make_infer_response(
    request: M0InferRequest,
    normalized_actions: Any,
    server_inference_ms: float,
) -> dict[str, Any]:
    actions = validate_normalized_action_chunk(normalized_actions)
    latency = _finite_number(server_inference_ms, "server_inference_ms")
    if latency < 0.0:
        raise M0OnlineError("server_inference_ms must be non-negative")
    return {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "sequence_id": request.sequence_id,
        "normalized_actions": actions,
        "server_inference_ms": latency,
        "seed": request.seed,
    }


def validate_normalized_action_chunk(value: Any) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M0OnlineError("normalized_actions must be an array")
    if len(value) != ACTION_HORIZON:
        raise M0OnlineError(f"normalized_actions must contain {ACTION_HORIZON} rows")
    return tuple(
        _finite_vector(row, ACTION_DIM, f"normalized_actions[{index}]")
        for index, row in enumerate(value)
    )


def project_action_chunk(
    value: Any,
    normalizer: M0MobileNormalizer,
    *,
    gripper_threshold: float = 0.5,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Clamp normalized actions and return safe normalized/physical chunks."""

    if len(normalizer.action_scale) != ACTION_DIM:
        raise M0OnlineError("normalizer action dimension must be 10")
    if 1 not in normalizer.hard_zero_indices or 9 not in normalizer.passthrough_indices:
        raise M0OnlineError("normalizer must hard-zero base_vy and pass through gripper")
    threshold = _finite_number(gripper_threshold, "gripper_threshold")
    if not normalizer.gripper_range[0] <= threshold <= normalizer.gripper_range[1]:
        raise M0OnlineError("gripper_threshold is outside the configured gripper range")
    low, high = normalizer.action_clip
    projected_rows: list[tuple[float, ...]] = []
    physical_rows: list[tuple[float, ...]] = []
    for row in validate_normalized_action_chunk(value):
        projected = []
        for index, component in enumerate(row):
            if index in normalizer.hard_zero_indices:
                projected.append(0.0)
            elif index in normalizer.passthrough_indices:
                projected.append(
                    min(normalizer.gripper_range[1], max(normalizer.gripper_range[0], component))
                )
            else:
                projected.append(min(high, max(low, component)))
        projected[9] = 1.0 if projected[9] >= threshold else 0.0
        physical = list(normalizer.denormalize_action(projected))
        physical[1] = 0.0
        physical[9] = projected[9]
        projected_rows.append(tuple(projected))
        physical_rows.append(tuple(physical))
    return tuple(projected_rows), tuple(physical_rows)


def quantize_go2_forward_intent(
    command: Sequence[float],
    *,
    activation_mps: float = 0.08,
    audited_minimum_mps: float = 0.16,
) -> tuple[float, float, float]:
    """Map a clear M0 forward intent onto the tested Go2 speed primitive."""

    vx, _vy, wz = _finite_vector(command, 3, "Go2 base command")
    activation = _finite_number(activation_mps, "activation_mps")
    minimum = _finite_number(audited_minimum_mps, "audited_minimum_mps")
    if not 0.0 < activation < minimum:
        raise M0OnlineError(
            "forward intent activation must be below the audited minimum"
        )
    if 0.0 < vx < activation:
        vx = 0.0
    elif activation <= vx < minimum:
        vx = minimum
    return vx, 0.0, wz


def load_state_statistics(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0OnlineError(f"cannot read state statistics {source}: {error}") from error
    statistics = _exact_object(
        value,
        {
            "schema_version",
            "accepted_source_schema_versions",
            "split",
            "state_key",
            "state_dimension",
            "state_layout",
            "state_layout_sha256",
            "count",
            "mean",
            "std",
            "std_definition",
            "source_files",
            "source_set_sha256",
        },
        "state statistics",
    )
    if (
        statistics["schema_version"] != STATE_STATISTICS_SCHEMA_VERSION
        or statistics["split"] != "train"
        or statistics["state_dimension"] != STATE_DIM
        or statistics["state_layout"] != list(M0_MOBILE_STATE_LAYOUT)
    ):
        raise M0OnlineError("state statistics do not match the M0-Mobile train contract")
    _finite_vector(statistics["mean"], STATE_DIM, "state statistics mean")
    std = _finite_vector(statistics["std"], STATE_DIM, "state statistics std")
    if any(component < 0.0 for component in std):
        raise M0OnlineError("state statistics std must be non-negative")
    return statistics


class M0OnlineClient:
    """Synchronous client intended for an SSH-forwarded localhost endpoint."""

    def __init__(
        self,
        endpoint: str,
        timeout_s: float = 30.0,
        *,
        normalizer: M0MobileNormalizer | None = None,
        jpeg_quality: int = 85,
    ) -> None:
        parts = urlsplit(endpoint)
        try:
            port = parts.port
        except ValueError as error:
            raise M0OnlineError(f"invalid inference endpoint: {error}") from error
        if (
            parts.scheme != "http"
            or parts.hostname != "127.0.0.1"
            or parts.username is not None
            or parts.password is not None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
            or (port is not None and not 1 <= port <= 65535)
        ):
            raise M0OnlineError("inference endpoint must be http://127.0.0.1[:port]")
        timeout = _finite_number(timeout_s, "timeout_s")
        if timeout <= 0.0:
            raise M0OnlineError("timeout_s must be positive")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 95:
            raise M0OnlineError("jpeg_quality must be an integer within [1, 95]")
        self.endpoint = endpoint.rstrip("/")
        if normalizer is None:
            config = load_m0_mobile_config()
            normalizer = M0MobileNormalizer.from_config(
                config, {"mean": [0.0] * STATE_DIM, "std": [1.0] * STATE_DIM}
            )
        self.normalizer = normalizer
        self.timeout_s = timeout
        self.jpeg_quality = jpeg_quality

    @classmethod
    def from_files(
        cls,
        endpoint: str,
        state_statistics: str | Path,
        *,
        config: str | Path = DEFAULT_CONFIG_PATH,
        timeout_s: float = 30.0,
        jpeg_quality: int = 85,
    ) -> M0OnlineClient:
        model_config = load_m0_mobile_config(config)
        normalizer = M0MobileNormalizer.from_config(
            model_config, load_state_statistics(state_statistics)
        )
        return cls(
            endpoint,
            timeout_s=timeout_s,
            normalizer=normalizer,
            jpeg_quality=jpeg_quality,
        )

    def health(self) -> Mapping[str, Any]:
        response = self._request("GET", "/health")
        health = _exact_object(
            response,
            {
                "schema_version",
                "status",
                "profile",
                "state_dim",
                "action_horizon",
                "action_dim",
                "model",
            },
            "health response",
        )
        expected = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "status": "ready",
            "profile": PROFILE,
            "state_dim": STATE_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
        }
        if any(health[key] != value for key, value in expected.items()):
            raise M0OnlineError("health response does not match the M0-Mobile contract")
        return {**expected, "model": _model_identity(health["model"])}

    def infer(
        self,
        head_rgb: Any,
        wrist_rgb: Any,
        instruction: str,
        state28: Sequence[float],
        *,
        sequence_id: int,
        request_id: str | None = None,
        seed: int = 20260803,
    ) -> M0InferenceResult:
        payload = {
            "schema_version": ONLINE_SCHEMA_VERSION,
            "request_id": request_id or uuid4().hex,
            "sequence_id": sequence_id,
            "instruction": instruction,
            "state28": state28,
            "images": [
                {
                    "camera_id": camera_id,
                    "encoding": "jpeg",
                    "data_base64": base64.b64encode(
                        encode_rgb_jpeg(image, quality=self.jpeg_quality)
                    ).decode("ascii"),
                }
                for camera_id, image in zip(CAMERA_IDS, (head_rgb, wrist_rgb), strict=True)
            ],
            "seed": seed,
        }
        validated_request = parse_infer_request(payload)
        payload["state28"] = validated_request.state28
        started = time.perf_counter()
        response = self._request("POST", "/infer", payload)
        round_trip_ms = (time.perf_counter() - started) * 1000.0
        parsed = _exact_object(
            response,
            {
                "schema_version",
                "request_id",
                "sequence_id",
                "normalized_actions",
                "server_inference_ms",
                "seed",
            },
            "infer response",
        )
        if parsed["schema_version"] != ONLINE_SCHEMA_VERSION:
            raise M0OnlineError("infer response has an unsupported schema_version")
        if (
            parsed["request_id"] != validated_request.request_id
            or parsed["sequence_id"] != validated_request.sequence_id
            or parsed["seed"] != validated_request.seed
        ):
            raise M0OnlineError("infer response does not echo request identity")
        server_ms = _finite_number(parsed["server_inference_ms"], "server_inference_ms")
        if server_ms < 0.0:
            raise M0OnlineError("server_inference_ms must be non-negative")
        normalized, physical = project_action_chunk(
            parsed["normalized_actions"], self.normalizer
        )
        return M0InferenceResult(
            request_id=validated_request.request_id,
            sequence_id=validated_request.sequence_id,
            normalized_actions=normalized,
            physical_actions=physical,
            server_inference_ms=server_ms,
            round_trip_ms=round_trip_ms,
            seed=validated_request.seed,
        )

    def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        try:
            body = (
                json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
                if payload is not None
                else None
            )
        except (TypeError, ValueError) as error:
            raise M0OnlineError(f"cannot encode inference request: {error}") from error
        if body is not None and len(body) > MAX_REQUEST_BYTES:
            raise M0OnlineError("inference request exceeds the protocol size limit")
        request = urllib.request.Request(
            self.endpoint + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                if response.headers.get_content_type() != "application/json":
                    raise M0OnlineError("inference server returned a non-JSON response")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raw_error = error.read(MAX_RESPONSE_BYTES + 1)
            try:
                message = json.loads(raw_error).get("error", error.reason)
            except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
                message = error.reason
            raise M0OnlineError(f"inference server HTTP {error.code}: {message}") from error
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
            raise M0OnlineError(f"inference server request failed: {error}") from error
        if len(raw) > MAX_RESPONSE_BYTES:
            raise M0OnlineError("inference response exceeds the protocol size limit")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise M0OnlineError(f"inference server returned invalid JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise M0OnlineError("inference server response must be a JSON object")
        return value


def health_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ONLINE_SCHEMA_VERSION,
        "status": "ready",
        "profile": PROFILE,
        "state_dim": STATE_DIM,
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "model": _model_identity(model),
    }


def _model_identity(value: Any) -> dict[str, Any]:
    identity = _exact_object(
        value,
        {
            "action_model_sha256",
            "state_statistics_sha256",
            "training_report_sha256",
            "training_steps",
            "dataset_records",
        },
        "model identity",
    )
    for key in (
        "action_model_sha256",
        "state_statistics_sha256",
        "training_report_sha256",
    ):
        digest = identity[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise M0OnlineError(f"model identity {key} must be a lowercase SHA-256")
    for key in ("training_steps", "dataset_records"):
        count = identity[key]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise M0OnlineError(f"model identity {key} must be a positive integer")
    return dict(identity)


def _exact_object(value: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0OnlineError(f"{name} must be a JSON object")
    if set(value) != keys:
        raise M0OnlineError(
            f"{name} fields must be exactly {sorted(keys)}; got {sorted(value)}"
        )
    return value


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != size:
        raise M0OnlineError(f"{name} must contain exactly {size} values")
    return tuple(_finite_number(component, name) for component in value)


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (bool, int, float)) and hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise M0OnlineError(f"{name} must contain only finite numbers")
    return float(value)


def _bounded_integer(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise M0OnlineError(f"{name} must be an integer within [{low}, {high}]")
    return value


def _request_id(value: Any) -> str:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise M0OnlineError("request_id contains invalid characters or length")
    return value


def _jpeg_bytes(value: Any) -> bytes:
    if not isinstance(value, bytes) or not 4 <= len(value) <= MAX_JPEG_BYTES:
        raise M0OnlineError("JPEG payload is invalid or too large")
    if not value.startswith(b"\xff\xd8") or not value.endswith(b"\xff\xd9"):
        raise M0OnlineError("camera payload is not a JPEG byte stream")
    return value


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "CAMERA_IDS",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "M0InferRequest",
    "M0InferenceResult",
    "M0OnlineClient",
    "M0OnlineError",
    "ONLINE_SCHEMA_VERSION",
    "PREGRASP_WORKSPACE_LIMITS_BASE",
    "STATE_DIM",
    "build_live_state28",
    "build_state28",
    "decode_rgb_jpeg",
    "encode_rgb_jpeg",
    "health_payload",
    "guard_pregrasp_tcp_target",
    "load_state_statistics",
    "make_infer_response",
    "parse_infer_request",
    "project_action_chunk",
    "quantize_go2_forward_intent",
    "validate_normalized_action_chunk",
]
