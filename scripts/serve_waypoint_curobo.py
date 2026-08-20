#!/usr/bin/env python3
"""Serve direct absolute TCP target planning through approved arm-vla cuRobo."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import socketserver
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint_planner_adapters import (  # noqa: E402
    APPROVED_ARM_VLA_COMMIT,
    CUROBO_REQUEST_SCHEMA,
    CUROBO_RESPONSE_SCHEMA,
)


MAX_START_JOINT_CLIP_RAD = 1.0e-3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--curobo-source-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "::1"))
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--ready-json", type=Path)
    return parser


class DirectPoseCuroboService:
    """Own one MotionPlanner and expose exact-or-fail direct pose planning."""

    def __init__(self, module: Any, planner: Any, *, reference_commit: str) -> None:
        if reference_commit != APPROVED_ARM_VLA_COMMIT:
            raise ValueError("cuRobo service reference commit is not approved")
        self.module = module
        self.planner = planner
        self.reference_commit = reference_commit
        self.joint_limits = module.load_joint_limits_from_urdf(module.ROBOT_URDF)
        self.started_at = time.time()
        self.request_count = 0

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "ping":
            return {
                "ok": True,
                "command": "ping",
                "arm_vla_reference_commit": self.reference_commit,
                "uptime_s": time.time() - self.started_at,
                "request_count": self.request_count,
            }
        if command == "capabilities":
            return {
                "ok": True,
                "command": "capabilities",
                "schema_version": CUROBO_RESPONSE_SCHEMA,
                "arm_vla_reference_commit": self.reference_commit,
                "features": {
                    "direct_absolute_tcp_target": True,
                    "input_target_frame": "query-base-B_t",
                    "planner_target_frame": "curobo-planner-base",
                    "orientation_fallback": False,
                    "world_collision": True,
                },
            }
        if command != "plan_tcp_target":
            raise ValueError(f"unsupported cuRobo service command: {command!r}")
        return self._plan_tcp_target(request)

    def _plan_tcp_target(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("schema_version") != CUROBO_REQUEST_SCHEMA:
            raise ValueError("cuRobo request schema is incompatible")
        if request.get("deployment") not in {"simulation", "real"}:
            raise ValueError("cuRobo request deployment is invalid")
        if request.get("target_frame") != "query-base-B_t":
            raise ValueError("cuRobo direct target must be expressed in query B_t")
        if request.get("target_units") != ["m", "m", "m", "rad", "rad", "rad"]:
            raise ValueError("cuRobo direct target units are incompatible")
        joints = _finite_vector(
            request.get("current_joints"),
            len(self.module.EXPECTED_JOINT_NAMES),
            "current_joints",
        )
        target = _finite_vector(request.get("target_tcp_base"), 6, "target_tcp_base")
        scene_collision = request.get("scene_collision")
        if not isinstance(scene_collision, Mapping) or "cuboids_base" not in scene_collision:
            raise ValueError("cuRobo request requires typed cuboids_base collision input")
        if scene_collision.get("frame") != "curobo-planner-base":
            raise ValueError("cuRobo collision frame must be curobo-planner-base")
        transform = _planner_base_transform(scene_collision)
        cuboids = scene_collision["cuboids_base"]
        if not isinstance(cuboids, Sequence) or isinstance(cuboids, (str, bytes)):
            raise ValueError("cuRobo cuboids_base must be a sequence")
        _validate_collision_cuboids(cuboids)

        self.request_count += 1
        self.module.PROFILER = self.module.Profiler()
        q_start = self.module.clip_q_to_joint_limits(
            np.asarray(joints, dtype=np.float32), self.joint_limits
        )
        if float(np.max(np.abs(q_start - np.asarray(joints)))) > MAX_START_JOINT_CLIP_RAD:
            raise ValueError("current joints exceed cuRobo limits by more than the clip tolerance")
        query_target_position = target[:3]
        query_target_quaternion = _rpy_to_quaternion(*target[3:])
        target_position, target_quaternion = _transform_pose(
            transform["position_xyz"],
            transform["quaternion_wxyz"],
            query_target_position,
            query_target_quaternion,
        )
        target_position_array = np.asarray(target_position, dtype=np.float32)
        target_quaternion_array = np.asarray(target_quaternion, dtype=np.float32)
        collision_scene = self.module.make_world_collision_scene(
            {"world_collision": {"cuboids_base": list(cuboids)}}
        )
        self.module.update_planner_world(self.planner, collision_scene)
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            joint_path, plan_info = self.module.plan_pose_path(
                planner=self.planner,
                q_start=q_start,
                target_position=target_position_array,
                target_quaternion=target_quaternion_array,
                segment_name="approach_to_grasp",
            )
            final_position, final_quaternion = self.module.run_fk(
                self.planner, np.asarray(joint_path[-1], dtype=np.float32)
            )
        if not isinstance(plan_info, Mapping) or plan_info.get("planner_success") is not True:
            raise RuntimeError("cuRobo did not report a successful direct-pose plan")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        position_error = float(
            np.linalg.norm(np.asarray(final_position) - target_position_array)
        )
        orientation_error = _quaternion_error_rad(
            final_quaternion, target_quaternion_array
        )
        path = np.asarray(joint_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != len(joints) or not np.isfinite(path).all():
            raise RuntimeError("cuRobo returned an invalid interpolated joint path")
        return {
            "schema_version": CUROBO_RESPONSE_SCHEMA,
            "arm_vla_reference_commit": self.reference_commit,
            "ok": True,
            "reachable": True,
            "collision_free": True,
            "joint_path": path.tolist(),
            "target_position_error_m": position_error,
            "target_orientation_error_rad": orientation_error,
            "metadata": {
                "planner": "curobo.MotionPlanner.plan_pose",
                "input_target_frame": "query-base-B_t",
                "planner_target_frame": "curobo-planner-base",
                "input_target_tcp_rpy": list(target),
                "planner_base_from_query_base": transform,
                "planner_target_position_xyz": target_position_array.tolist(),
                "planner_target_quaternion_wxyz": target_quaternion_array.tolist(),
                "plan_info": _jsonable(plan_info),
                "world_collision_cuboid_count": len(cuboids),
                "orientation_fallback_used": False,
                "planner_elapsed_ms": elapsed_ms,
                "profile": _jsonable(self.module.PROFILER.summary()),
            },
        }


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(64 * 1024 * 1024 + 1)
        try:
            if not raw or len(raw) > 64 * 1024 * 1024:
                raise ValueError("cuRobo request is empty or too large")
            request = json.loads(raw)
            if not isinstance(request, Mapping):
                raise ValueError("cuRobo request must be an object")
            response = self.server.service.handle(request)  # type: ignore[attr-defined]
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {
                "schema_version": CUROBO_RESPONSE_SCHEMA,
                "arm_vla_reference_commit": self.server.service.reference_commit,  # type: ignore[attr-defined]
                "ok": False,
                "reachable": False,
                "collision_free": False,
                "error": f"{type(error).__name__}: {error}",
            }
        self.wfile.write(
            json.dumps(response, separators=(",", ":")).encode("utf-8") + b"\n"
        )


class _Server(socketserver.TCPServer):
    allow_reuse_address = True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("cuRobo service port is invalid")
    reference_root = args.reference_root.expanduser().resolve()
    commit = _clean_reference_commit(reference_root)
    if commit != APPROVED_ARM_VLA_COMMIT:
        raise RuntimeError(
            f"arm-vla reference must be {APPROVED_ARM_VLA_COMMIT}, got {commit}"
        )
    os.environ["GO2_X5_WORKSPACE"] = str(reference_root)
    if args.curobo_source_root is not None:
        os.environ["GO2_X5_CUROBO_SOURCE_ROOT"] = str(
            args.curobo_source_root.expanduser().resolve()
        )
    module = _load_reference_module(reference_root)
    with contextlib.redirect_stdout(sys.stderr):
        planner = module.create_planner()
    service = DirectPoseCuroboService(module, planner, reference_commit=commit)
    with _Server((args.host, args.port), _RequestHandler) as server:
        server.service = service  # type: ignore[attr-defined]
        if args.ready_json is not None:
            ready = args.ready_json.expanduser().resolve()
            ready.parent.mkdir(parents=True, exist_ok=True)
            ready.write_text(
                json.dumps(
                    {
                        "ready": True,
                        "schema_version": CUROBO_RESPONSE_SCHEMA,
                        "host": args.host,
                        "port": args.port,
                        "arm_vla_reference_commit": commit,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "event": "waypoint_curobo_ready",
                    "host": args.host,
                    "port": args.port,
                    "arm_vla_reference_commit": commit,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        server.serve_forever(poll_interval=0.2)
    return 0


def _load_reference_module(reference_root: Path) -> Any:
    path = reference_root / "scripts" / "curobo" / "03_plan_grasp_trajectory.py"
    if not path.is_file():
        raise RuntimeError(f"approved arm-vla cuRobo module is missing: {path}")
    sys.path.insert(0, str(reference_root))
    spec = importlib.util.spec_from_file_location("waypoint_arm_vla_curobo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load approved arm-vla cuRobo module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clean_reference_commit(reference_root: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(reference_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("arm-vla cuRobo reference worktree must be clean")
    return commit


def _rpy_to_quaternion(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    quaternion = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _planner_base_transform(
    scene_collision: Mapping[str, Any],
) -> dict[str, tuple[float, ...]]:
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
    return {
        "position_xyz": position,
        "quaternion_wxyz": quaternion,
    }


def _transform_pose(
    parent_from_child_position: Sequence[float],
    parent_from_child_quaternion: Sequence[float],
    child_target_position: Sequence[float],
    child_target_quaternion: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    translation = _finite_vector(
        parent_from_child_position, 3, "planner/query translation"
    )
    rotation = _normalized_quaternion(
        parent_from_child_quaternion, "planner/query quaternion"
    )
    position = _finite_vector(child_target_position, 3, "query target position")
    orientation = _normalized_quaternion(
        child_target_quaternion, "query target quaternion"
    )
    rotated = _rotate_vector(rotation, position)
    target_position = tuple(
        translation[index] + rotated[index] for index in range(3)
    )
    target_quaternion = _normalized_quaternion(
        _quaternion_multiply(rotation, orientation), "planner target quaternion"
    )
    return target_position, target_quaternion  # type: ignore[return-value]


def _normalized_quaternion(
    value: Sequence[float], name: str
) -> tuple[float, float, float, float]:
    quaternion = _finite_vector(value, 4, name)
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1.0e-9:
        raise ValueError(f"{name} is degenerate")
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


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
    w, x, y, z = quaternion
    vx, vy, vz = vector
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def _quaternion_error_rad(first: Sequence[float], second: Sequence[float]) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if (
        left.shape != (4,)
        or right.shape != (4,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or left_norm < 1.0e-9
        or right_norm < 1.0e-9
    ):
        raise RuntimeError("cuRobo FK returned an invalid quaternion")
    left /= left_norm
    right /= right_norm
    dot = float(np.clip(abs(np.dot(left, right)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} has an invalid shape or value")
    return result


def _validate_collision_cuboids(cuboids: Sequence[Any]) -> None:
    names: set[str] = set()
    for index, raw in enumerate(cuboids):
        if not isinstance(raw, Mapping):
            raise ValueError(f"cuRobo cuboid {index} must be an object")
        name = str(raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"cuRobo cuboid {index} has a missing or duplicate name")
        names.add(name)
        pose = raw.get("pose_base")
        if not isinstance(pose, Mapping):
            raise ValueError(f"cuRobo cuboid {index} requires pose_base")
        _finite_vector(pose.get("position_xyz"), 3, f"cuboid {index} position")
        quaternion = _finite_vector(
            pose.get("quaternion_wxyz"), 4, f"cuboid {index} quaternion"
        )
        if math.sqrt(sum(value * value for value in quaternion)) < 1.0e-9:
            raise ValueError(f"cuRobo cuboid {index} quaternion is degenerate")
        dimensions = _finite_vector(raw.get("dims_xyz"), 3, f"cuboid {index} dimensions")
        if any(value <= 0.0 for value in dimensions):
            raise ValueError(f"cuRobo cuboid {index} dimensions must be positive")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


if __name__ == "__main__":
    raise SystemExit(main())
