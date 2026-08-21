"""Strict adapters from waypoint v1 executors to the approved planner stack."""

from __future__ import annotations

import json
import math
import socket
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Mapping, Sequence

from conveyor_bench.conveyorvla.waypoint_execution import ArmPlan, PCTPlan


APPROVED_ARM_VLA_COMMIT = "388b6818f4c605a707d13c519fbb58b1d07acd92"
CUROBO_REQUEST_SCHEMA = "conveyorvla-waypoint-curobo-request-v1"
CUROBO_RESPONSE_SCHEMA = "conveyorvla-waypoint-curobo-response-v1"


class ArmVLAPCTPlannerAdapter:
    """Compose a model local goal with arm-vla's fail-closed PCT planner."""

    def __init__(
        self,
        planner: Any,
        *,
        simulation_state_factory: Callable[..., Any],
        nav_goal_factory: Callable[..., Any],
        reference_commit: str,
    ) -> None:
        _approved_reference(reference_commit)
        config = getattr(planner, "config", None)
        if config is None or getattr(config, "enabled", None) is not True:
            raise ValueError("arm-vla PCT planner must be explicitly enabled")
        if getattr(config, "fallback_to_astar", None) is not False:
            raise ValueError("waypoint PCT must explicitly disable A* fallback")
        if getattr(planner, "fallback_planner", None) is not None:
            raise ValueError("waypoint PCT adapter must not retain a fallback planner")
        self.planner = planner
        self.simulation_state_factory = simulation_state_factory
        self.nav_goal_factory = nav_goal_factory
        self.reference_commit = reference_commit

    def plan(
        self,
        current_world_pose: Sequence[float],
        predicted_world_goal: Sequence[float],
    ) -> PCTPlan:
        current = _finite_vector(current_world_pose, 4, "current PCT world pose")
        goal = _finite_vector(predicted_world_goal, 4, "predicted PCT world goal")
        current_quaternion = _yaw_quaternion(current[3])
        state = self.simulation_state_factory(
            step_index=0,
            timestamp=time.monotonic(),
            robot_root_pose=(current[0], current[1], current[2], *current_quaternion),
            robot_root_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            metadata={"consumer": "waypoint_executor_only", "model_input": False},
        )
        nav_goal = self.nav_goal_factory(
            x=goal[0],
            y=goal[1],
            yaw=goal[3],
            z=goal[2],
        )
        started = time.perf_counter()
        reference_plan = self.planner.plan(state, nav_goal)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metadata = getattr(reference_plan, "metadata", None)
        if not isinstance(metadata, Mapping) or metadata.get("planner") != "pct":
            raise ValueError("arm-vla planner did not return strict PCT provenance")
        if "fallback" in str(metadata.get("planner", "")).lower():
            raise ValueError("arm-vla PCT returned a forbidden fallback plan")
        path = tuple(
            (float(point[0]), float(point[1]))
            for point in getattr(reference_plan, "waypoints", ())
        )
        if len(path) < 2 or not all(
            all(math.isfinite(value) for value in point) for point in path
        ):
            raise ValueError("arm-vla PCT returned an invalid path")
        # arm-vla's path_3d stores planning-floor height after subtracting the
        # robot-root offset.  Endpoint snapping is therefore an XY contract;
        # retain the model query's root height in the typed snapped goal.
        snapped = (path[-1][0], path[-1][1], goal[2], goal[3])
        snap_distance = math.hypot(snapped[0] - goal[0], snapped[1] - goal[1])
        reported_snap = _optional_float(
            metadata.get("snap_end_distance_m", metadata.get("snap_end_dist"))
        )
        return PCTPlan(
            path_world=path,
            snapped_goal_world=snapped,
            snap_distance_m=snap_distance,
            metadata={
                **dict(metadata),
                "adapter": "ArmVLAPCTPlannerAdapter",
                "arm_vla_reference_commit": self.reference_commit,
                "current_world_pose": list(current),
                "model_predicted_goal_world": list(goal),
                "reported_snap_end_distance_m": reported_snap,
                "computed_snap_end_distance_m": snap_distance,
                "planner_elapsed_ms": elapsed_ms,
                "fallback_allowed": False,
            },
        )


class ArmVLADWAControllerAdapter:
    """Own arm-vla's path-stateful DWA controller behind a per-tick API."""

    def __init__(
        self,
        controller_type: Callable[..., Any],
        config: Any,
        *,
        reference_commit: str,
    ) -> None:
        _approved_reference(reference_commit)
        self.controller_type = controller_type
        self.config = config
        self.reference_commit = reference_commit
        self._key: tuple[Any, ...] | None = None
        self._controller: Any | None = None
        self.last_trace: dict[str, Any] = {}

    def reset(self) -> None:
        self._key = None
        self._controller = None
        self.last_trace = {}

    def command(
        self,
        path_world: Sequence[Sequence[float]],
        current_world_pose: Sequence[float],
        measured_body_velocity: Sequence[float],
        local_map: Any,
    ) -> tuple[float, float, float]:
        path = tuple(
            tuple(_finite_vector(point, 2, "DWA path point")) for point in path_world
        )
        pose = tuple(_finite_vector(current_world_pose, 3, "DWA world pose"))
        velocity = tuple(_finite_vector(measured_body_velocity, 3, "DWA velocity"))
        grid_map, raw_grid_map = _grid_maps(local_map)
        key = (path, id(grid_map), id(raw_grid_map))
        if self._controller is None or key != self._key:
            kwargs = {} if raw_grid_map is None else {"raw_grid_map": raw_grid_map}
            self._controller = self.controller_type(
                list(path), grid_map, self.config, **kwargs
            )
            self._key = key
        started = time.perf_counter()
        raw_command, debug = self._controller.compute_command(pose, velocity)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        command = tuple(_finite_vector(raw_command, 3, "arm-vla DWA command"))
        self.last_trace = {
            "adapter": "ArmVLADWAControllerAdapter",
            "arm_vla_reference_commit": self.reference_commit,
            "path_point_count": len(path),
            "current_world_pose": list(pose),
            "measured_body_velocity": list(velocity),
            "local_map_type": type(grid_map).__name__,
            "command": list(command),
            "elapsed_ms": elapsed_ms,
            "debug": _jsonable(debug),
        }
        return command


class JsonLineCuRoboTransport:
    """One-request JSON-line client for a persistent local cuRobo service."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, timeout_s: float = 15.0) -> None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("waypoint cuRobo transport is restricted to loopback")
        if not 1 <= int(port) <= 65535 or not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("cuRobo transport endpoint is invalid")
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = json.dumps(dict(request), separators=(",", ":")).encode("utf-8") + b"\n"
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            connection.settimeout(self.timeout_s)
            connection.sendall(payload)
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    raise RuntimeError("cuRobo service closed before returning JSON")
                response.extend(chunk)
                if len(response) > 64 * 1024 * 1024:
                    raise RuntimeError("cuRobo service response exceeds 64 MiB")
        value = json.loads(response)
        if not isinstance(value, Mapping):
            raise RuntimeError("cuRobo service response must be an object")
        return value


class WaypointCuRoboPlannerAdapter:
    """Send one absolute B_t TCP target to a direct-pose cuRobo service."""

    def __init__(
        self,
        transport: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        deployment: str,
        safety_gate: Callable[[Mapping[str, Any], Mapping[str, Any]], bool],
        reference_commit: str,
    ) -> None:
        _approved_reference(reference_commit)
        if deployment not in {"simulation", "real"}:
            raise ValueError("cuRobo deployment must be simulation or real")
        if not callable(safety_gate):
            raise ValueError("each cuRobo deployment requires its own safety gate")
        self.transport = transport
        self.deployment = deployment
        self.safety_gate = safety_gate
        self.reference_commit = reference_commit

    def plan(
        self,
        current_joints: Sequence[float],
        target_tcp_base: Sequence[float],
        scene_collision: Any,
    ) -> ArmPlan:
        joints = _finite_vector(current_joints, None, "current arm joints")
        if not joints:
            raise ValueError("current arm joints must be non-empty")
        target = _finite_vector(target_tcp_base, 6, "absolute TCP target")
        if not isinstance(scene_collision, Mapping):
            raise ValueError("cuRobo scene collision must be a typed mapping")
        transform = _planner_base_transform(scene_collision)
        request = {
            "schema_version": CUROBO_REQUEST_SCHEMA,
            "command": "plan_tcp_target",
            "deployment": self.deployment,
            "target_frame": "query-base-B_t",
            "target_units": ["m", "m", "m", "rad", "rad", "rad"],
            "current_joints": list(joints),
            "target_tcp_base": list(target),
            "scene_collision": dict(scene_collision),
        }
        started = time.perf_counter()
        response = self.transport(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.get("schema_version") != CUROBO_RESPONSE_SCHEMA:
            raise RuntimeError("cuRobo service response schema is incompatible")
        if response.get("arm_vla_reference_commit") != self.reference_commit:
            raise RuntimeError("cuRobo service reference commit is incompatible")
        if response.get("ok") is not True:
            raise RuntimeError(f"cuRobo direct-pose planning failed: {response.get('error')}")
        if not self.safety_gate(request, response):
            raise RuntimeError(f"{self.deployment} cuRobo safety gate rejected the plan")
        raw_path = response.get("joint_path")
        if not isinstance(raw_path, Sequence) or not raw_path:
            raise RuntimeError("cuRobo service returned no joint path")
        joint_path = tuple(
            tuple(_finite_vector(row, len(joints), "cuRobo joint path row"))
            for row in raw_path
        )
        cuboids = scene_collision.get("cuboids_base")
        cuboid_count = (
            len(cuboids)
            if isinstance(cuboids, Sequence) and not isinstance(cuboids, (str, bytes))
            else None
        )
        return ArmPlan(
            joint_path=joint_path,
            planner="curobo",
            reachable=bool(response.get("reachable")),
            collision_free=bool(response.get("collision_free")),
            target_position_error_m=_finite_scalar(
                response.get("target_position_error_m"), "cuRobo position error"
            ),
            target_orientation_error_rad=_finite_scalar(
                response.get("target_orientation_error_rad"),
                "cuRobo orientation error",
            ),
            metadata={
                **dict(response.get("metadata") or {}),
                "adapter": "WaypointCuRoboPlannerAdapter",
                "deployment": self.deployment,
                "target_frame": "query-base-B_t",
                "current_joints": list(joints),
                "target_tcp_base": list(target),
                "planner_base_from_query_base": transform,
                "scene_collision_cuboid_count": cuboid_count,
                "transport_elapsed_ms": elapsed_ms,
                "arm_vla_reference_commit": self.reference_commit,
            },
        )


def _planner_base_transform(scene_collision: Mapping[str, Any]) -> dict[str, list[float]]:
    if scene_collision.get("frame") != "curobo-planner-base":
        raise ValueError("cuRobo collision frame must be curobo-planner-base")
    raw = scene_collision.get("planner_base_from_query_base")
    if not isinstance(raw, Mapping):
        raise ValueError("cuRobo scene requires planner_base_from_query_base")
    position = _finite_vector(raw.get("position_xyz"), 3, "planner/query translation")
    quaternion = _finite_vector(
        raw.get("quaternion_wxyz"), 4, "planner/query quaternion"
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 1.0e-4:
        raise ValueError("planner/query quaternion must be normalized")
    cuboids = scene_collision.get("cuboids_base")
    if not isinstance(cuboids, Sequence) or isinstance(cuboids, (str, bytes)):
        raise ValueError("cuRobo scene requires cuboids_base")
    return {
        "position_xyz": list(position),
        "quaternion_wxyz": list(quaternion),
    }


@dataclass(frozen=True)
class JointPathControllerConfig:
    reached_tolerance_rad: float = 0.03

    def __post_init__(self) -> None:
        if not math.isfinite(self.reached_tolerance_rad) or self.reached_tolerance_rad <= 0.0:
            raise ValueError("joint-path tolerance must be finite and positive")


class JointPathController:
    """Track a pre-validated cuRobo path without changing its target semantics."""

    def __init__(self, config: JointPathControllerConfig = JointPathControllerConfig()) -> None:
        self.config = config
        self._path: tuple[tuple[float, ...], ...] = ()
        self._index = 0
        self._gripper_target = 0.0

    def reset(self, plan: ArmPlan, gripper_target: float) -> None:
        if not plan.joint_path:
            raise ValueError("joint controller requires a non-empty cuRobo path")
        gripper = _finite_scalar(gripper_target, "gripper target")
        if not 0.0 <= gripper <= 1.0:
            raise ValueError("gripper target must be within [0,1]")
        self._path = plan.joint_path
        self._index = 0
        self._gripper_target = gripper

    def command(self, measured_joints: Sequence[float]) -> tuple[tuple[float, ...], bool]:
        if not self._path:
            raise RuntimeError("joint controller has no active plan")
        measured = _finite_vector(
            measured_joints, len(self._path[0]), "measured arm joints"
        )
        while self._index < len(self._path) and _max_error(
            measured, self._path[self._index]
        ) <= self.config.reached_tolerance_rad:
            self._index += 1
        if self._index >= len(self._path):
            return self._path[-1], True
        return self._path[self._index], False

    def status(self) -> dict[str, Any]:
        return {
            "active": bool(self._path) and self._index < len(self._path),
            "target_index": self._index,
            "path_length": len(self._path),
            "gripper_target": self._gripper_target,
        }


def _grid_maps(value: Any) -> tuple[Any, Any | None]:
    if not isinstance(value, Mapping):
        return value, None
    if "grid_map" not in value:
        raise ValueError("typed DWA local map must contain grid_map")
    return value["grid_map"], value.get("raw_grid_map")


def _approved_reference(commit: str) -> None:
    if commit != APPROVED_ARM_VLA_COMMIT:
        raise ValueError("arm-vla reference commit differs from the approved contract")


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)


def _finite_vector(value: Any, length: int | None, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric sequence") from error
    if (length is not None and len(result) != length) or not all(
        math.isfinite(item) for item in result
    ):
        raise ValueError(f"{name} has an invalid shape or value")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _max_error(first: Sequence[float], second: Sequence[float]) -> float:
    return max(
        abs(float(left) - float(right))
        for left, right in zip(first, second, strict=True)
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "APPROVED_ARM_VLA_COMMIT",
    "ArmVLADWAControllerAdapter",
    "ArmVLAPCTPlannerAdapter",
    "CUROBO_REQUEST_SCHEMA",
    "CUROBO_RESPONSE_SCHEMA",
    "JointPathController",
    "JointPathControllerConfig",
    "JsonLineCuRoboTransport",
    "WaypointCuRoboPlannerAdapter",
]
