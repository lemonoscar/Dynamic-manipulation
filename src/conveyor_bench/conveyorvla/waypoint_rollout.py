"""State-free wire client and visual timing helpers for waypoint rollouts."""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from conveyor_bench.conveyorvla.waypoint import CAMERA_CALIBRATION_ID
from conveyor_bench.conveyorvla.waypoint_protocol import (
    RUNTIME_PROTOCOL_VERSION,
    WaypointRequest,
    WaypointResponse,
)


@dataclass(frozen=True)
class WaypointWireResult:
    response: WaypointResponse
    trace: Mapping[str, Any]
    diffusion_seed: int


class WaypointHTTPClient:
    """Call only the loopback state-free `/infer` endpoint."""

    def __init__(self, endpoint: str, *, timeout_s: float = 120.0) -> None:
        parsed = urllib.parse.urlparse(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("waypoint model endpoint must be a plain loopback HTTP URL")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("waypoint model timeout must be finite and positive")
        self.endpoint = endpoint.rstrip("/") + "/infer"
        self.timeout_s = float(timeout_s)

    def infer(self, request: WaypointRequest) -> WaypointWireResult:
        payload = json.dumps(
            request.to_mapping(), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        wire_request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                wire_request, timeout=self.timeout_s
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"waypoint model returned HTTP {response.status}"
                    )
                value = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"waypoint model returned HTTP {error.code}: {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"waypoint model request failed: {error}") from error
        if not isinstance(value, Mapping):
            raise RuntimeError("waypoint model response must be an object")
        raw_response = value.get("response")
        raw_trace = value.get("trace")
        seed = value.get("diffusion_seed")
        if not isinstance(raw_response, Mapping) or not isinstance(raw_trace, Mapping):
            raise RuntimeError("waypoint model response is missing response/trace")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise RuntimeError("waypoint model diffusion seed is invalid")
        response = WaypointResponse.from_mapping(raw_response)
        if (
            response.request_id != request.request_id
            or response.sequence_id != request.sequence_id
        ):
            raise RuntimeError("waypoint model response does not bind to its request")
        return WaypointWireResult(response, dict(raw_trace), seed)


@dataclass(frozen=True)
class EncodedVisualFrame:
    step_index: int
    head_jpeg_base64: str
    wrist_jpeg_base64: str


class TemporalJPEGBuffer:
    """Keep synchronized head/wrist frames and select exact t-0.20,t pairs."""

    def __init__(
        self,
        *,
        separation_steps: int,
        jpeg_quality: int = 90,
        capacity: int = 16,
    ) -> None:
        if separation_steps < 1 or not 1 <= jpeg_quality <= 100 or capacity < 2:
            raise ValueError("temporal JPEG buffer configuration is invalid")
        self.separation_steps = int(separation_steps)
        self.jpeg_quality = int(jpeg_quality)
        self._frames: deque[EncodedVisualFrame] = deque(maxlen=int(capacity))

    def add(self, step_index: int, camera_images: Mapping[str, Any]) -> bool:
        if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
            raise ValueError("camera step index must be a non-negative integer")
        head = camera_images.get("front")
        wrist = camera_images.get("wrist")
        if head is None or wrist is None:
            return False
        if self._frames and step_index <= self._frames[-1].step_index:
            if step_index == self._frames[-1].step_index:
                return False
            raise ValueError("camera frames must be added in increasing step order")
        self._frames.append(
            EncodedVisualFrame(
                step_index=step_index,
                head_jpeg_base64=_encode_jpeg(head, self.jpeg_quality),
                wrist_jpeg_base64=_encode_jpeg(wrist, self.jpeg_quality),
            )
        )
        return True

    def pair_after(
        self, previous_current_step: int | None
    ) -> tuple[EncodedVisualFrame, EncodedVisualFrame] | None:
        if not self._frames:
            return None
        current = self._frames[-1]
        if previous_current_step is not None and current.step_index <= previous_current_step:
            return None
        expected = current.step_index - self.separation_steps
        earlier = next(
            (frame for frame in reversed(self._frames) if frame.step_index == expected),
            None,
        )
        return None if earlier is None else (earlier, current)

    @property
    def step_indices(self) -> tuple[int, ...]:
        return tuple(frame.step_index for frame in self._frames)


def waypoint_request_from_frames(
    *,
    episode_id: str,
    sequence_id: int,
    instruction: str,
    frames: tuple[EncodedVisualFrame, EncodedVisualFrame],
    camera_calibration_id: str = CAMERA_CALIBRATION_ID,
) -> WaypointRequest:
    earlier, current = frames
    return WaypointRequest(
        protocol_version=RUNTIME_PROTOCOL_VERSION,
        request_id=f"{episode_id}-waypoint-{sequence_id:06d}",
        episode_id=str(episode_id),
        sequence_id=int(sequence_id),
        instruction=str(instruction),
        head_images=(earlier.head_jpeg_base64, current.head_jpeg_base64),
        wrist_images=(earlier.wrist_jpeg_base64, current.wrist_jpeg_base64),
        camera_calibration_id=str(camera_calibration_id),
    )


def tcp_pose_in_query_base(
    robot_root_pose: Sequence[float], tcp_pose_world: Sequence[float]
) -> tuple[float, float, float, float, float, float]:
    root_position, root_quaternion = _pose(robot_root_pose, "robot root pose")
    tcp_position, tcp_quaternion = _pose(tcp_pose_world, "TCP world pose")
    query_from_world = _quaternion_conjugate(root_quaternion)
    delta = tuple(tcp_position[index] - root_position[index] for index in range(3))
    position = _rotate_vector(query_from_world, delta)
    quaternion = _normalized_quaternion(
        _quaternion_multiply(query_from_world, tcp_quaternion),
        "TCP query-base quaternion",
    )
    return (*position, *_quaternion_to_rpy(quaternion))


def planner_base_from_query_base(
    world_from_planner: Sequence[Sequence[float]],
    robot_root_pose: Sequence[float],
) -> dict[str, list[float]]:
    import numpy as np

    planner = np.asarray(world_from_planner, dtype=np.float64)
    if planner.shape != (4, 4) or not np.isfinite(planner).all():
        raise ValueError("world/planner transform must be a finite 4x4 matrix")
    query = _pose_matrix(robot_root_pose)
    transform = np.linalg.inv(planner) @ query
    if not np.isfinite(transform).all():
        raise ValueError("planner/query transform is non-finite")
    return {
        "position_xyz": transform[:3, 3].tolist(),
        "quaternion_wxyz": list(_rotation_matrix_to_quaternion(transform[:3, :3])),
    }


def measured_arm_joints(
    joint_names: Sequence[Any],
    joint_positions: Sequence[Any],
    expected_names: Sequence[str],
) -> tuple[float, ...]:
    names = tuple(str(value) for value in joint_names)
    if len(names) != len(joint_positions):
        raise ValueError("joint names and positions do not align")
    indices = {name: index for index, name in enumerate(names)}
    if any(name not in indices for name in expected_names):
        missing = [name for name in expected_names if name not in indices]
        raise ValueError(f"simulation state is missing arm joints: {missing}")
    values = tuple(float(joint_positions[indices[name]]) for name in expected_names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("simulation arm joints are non-finite")
    return values


def measured_body_velocity(state: Any) -> tuple[float, float, float]:
    raw = getattr(state, "metadata", {}).get("body_velocity")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
        values = tuple(float(value) for value in raw[:3])
        if all(math.isfinite(value) for value in values):
            return values  # type: ignore[return-value]
    pose = tuple(float(value) for value in state.robot_root_pose)
    velocity = tuple(float(value) for value in state.robot_root_velocity)
    yaw = _quaternion_to_rpy(_normalized_quaternion(pose[3:7], "root quaternion"))[2]
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return (
        cos_yaw * velocity[0] + sin_yaw * velocity[1],
        -sin_yaw * velocity[0] + cos_yaw * velocity[1],
        velocity[5],
    )


def _encode_jpeg(image: Any, quality: int) -> str:
    import numpy as np
    from PIL import Image

    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"camera image must be [H,W,3|4], got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    mode = "RGBA" if array.shape[2] == 4 else "RGB"
    output = io.BytesIO()
    Image.fromarray(array, mode=mode).convert("RGB").save(
        output, format="JPEG", quality=quality
    )
    return base64.b64encode(output.getvalue()).decode("ascii")


def _pose(
    value: Sequence[float], name: str
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    values = tuple(float(item) for item in value)
    if len(values) != 7 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain seven finite values")
    return values[:3], _normalized_quaternion(values[3:], f"{name} quaternion")


def _pose_matrix(value: Sequence[float]) -> Any:
    import numpy as np

    position, quaternion = _pose(value, "robot root pose")
    w, x, y, z = quaternion
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    matrix[:3, 3] = position
    return matrix


def _normalized_quaternion(
    value: Sequence[float], name: str
) -> tuple[float, float, float, float]:
    quaternion = tuple(float(item) for item in value)
    if len(quaternion) != 4 or not all(math.isfinite(item) for item in quaternion):
        raise ValueError(f"{name} must contain four finite values")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1.0e-9:
        raise ValueError(f"{name} is degenerate")
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def _quaternion_conjugate(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    w, x, y, z = value
    return w, -x, -y, -z


def _quaternion_multiply(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    rotated = _quaternion_multiply(
        _quaternion_multiply(quaternion, (0.0, *vector)),
        _quaternion_conjugate(quaternion),
    )
    return rotated[1], rotated[2], rotated[3]


def _quaternion_to_rpy(
    quaternion: Sequence[float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def _rotation_matrix_to_quaternion(
    matrix: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    import numpy as np

    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be finite 3x3")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = (
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = (
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            )
    return _normalized_quaternion(quaternion, "rotation matrix quaternion")


__all__ = [
    "EncodedVisualFrame",
    "TemporalJPEGBuffer",
    "WaypointHTTPClient",
    "WaypointWireResult",
    "measured_arm_joints",
    "measured_body_velocity",
    "planner_base_from_query_base",
    "tcp_pose_in_query_base",
    "waypoint_request_from_frames",
]
