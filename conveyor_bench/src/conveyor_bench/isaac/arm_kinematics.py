"""Deterministic kinematics for the project-local X5 arm.

The constants below are copied from ``assets/robots/go2_x5/go2_x5.urdf``.
Keeping this small solver in the benchmark avoids depending on a simulator
Jacobian body-index convention for the privileged demonstration oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Sequence

import numpy as np


# PCT's FinRay tip frame. Keep this identical to ``asset_config.TCP_OFFSET_X_M``
# and ``grasp_tcp_fixed_joint`` in the canonical local URDF.
_TCP_OFFSET_X_M = 0.15757
_BASE_TRANSFORM_XYZ = (0.12, 0.0, 0.43)
# Arm mount relative to the Go2 root (``base``) link.  Fixed-base diagnostics
# use the calibrated world transform above; the mobile runtime uses this
# local transform and converts world targets through the live robot root pose.
_ROOT_FRAME_ARM_MOUNT_XYZ = (0.12, 0.0, 0.05)
_JOINT_ORIGINS = (
    ((0.0, 0.0, 0.0605), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02, 0.0, 0.04), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((-0.264, 0.0, 0.0), (3.1416, 0.0, 0.0), (0, 1, 0)),
    ((0.245, 0.0, -0.056), (0.0, 0.0, 0.0), (0, 1, 0)),
    ((0.06775, 0.0005, -0.0865), (0.0, 0.0, 0.0), (0, 0, 1)),
    ((0.02895, 0.0, 0.0865), (-3.1416, 0.0, 0.0), (1, 0, 0)),
)
_LOWER_LIMITS = np.asarray((-2.618, 0.0, 0.0, -1.5708, -1.5708, -1.5708))
_UPPER_LIMITS = np.asarray((3.14, 3.14, 3.14, 1.5708, 1.5708, 1.5708))


class IKConvergenceError(RuntimeError):
    """Raised when a requested TCP pose is outside the calibrated workspace."""


@dataclass(frozen=True)
class ArmIKSolution:
    joint_positions: tuple[float, ...]
    solved_position: tuple[float, float, float]
    position_error_m: float
    orientation_error: float
    iterations: int


class CalibratedArmKinematics:
    """Finite-difference damped least-squares IK for the six arm joints."""

    def __init__(
        self,
        *,
        orientation_weight: float = 0.12,
        damping: float = 2.0e-4,
        max_iterations: int = 60,
        position_tolerance_m: float = 0.01,
        orientation_tolerance: float = 0.05,
        base_transform_xyz: Sequence[float] = _BASE_TRANSFORM_XYZ,
    ):
        if (
            orientation_weight <= 0
            or damping <= 0
            or max_iterations <= 0
            or position_tolerance_m <= 0
            or orientation_tolerance <= 0
        ):
            raise ValueError("solver parameters must be positive")
        base_transform_xyz = np.asarray(base_transform_xyz, dtype=np.float64)
        if base_transform_xyz.shape != (3,) or not np.isfinite(
            base_transform_xyz
        ).all():
            raise ValueError("base_transform_xyz must contain three finite values")
        self.orientation_weight = orientation_weight
        self.damping = damping
        self.max_iterations = max_iterations
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance = orientation_tolerance
        self._base_transform = _transform(base_transform_xyz)
        self._joint_transforms = tuple(
            (_transform(xyz, rpy), axis) for xyz, rpy, axis in _JOINT_ORIGINS
        )

    def forward(
        self, joint_positions: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return TCP position and rotation matrix in the world frame."""

        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("joint_positions must contain six finite values")
        pose = self._base_transform.copy()
        for (origin, axis), value in zip(
            self._joint_transforms, q, strict=True
        ):
            pose = pose @ origin @ _joint_rotation(axis, float(value))
        pose = pose @ _transform((_TCP_OFFSET_X_M, 0.0, 0.0))
        return pose[:3, 3].copy(), pose[:3, :3].copy()

    def solve(
        self,
        target_position: Sequence[float],
        target_orientation_wxyz: Sequence[float],
        *,
        seed: Sequence[float],
    ) -> ArmIKSolution:
        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_position must contain three finite values")
        target_rotation = _rotation_from_quaternion(target_orientation_wxyz)
        q = np.clip(np.asarray(seed, dtype=np.float64), _LOWER_LIMITS, _UPPER_LIMITS)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("seed must contain six finite values")

        iteration = 0
        for iteration in range(1, self.max_iterations + 1):
            position, rotation = self.forward(q)
            residual = self._residual(
                target,
                target_rotation,
                position,
                rotation,
            )
            position_error = float(np.linalg.norm(residual[:3]))
            orientation_error = float(np.linalg.norm(target_rotation - rotation))
            if position_error < 2.0e-4 and orientation_error < 2.0e-3:
                break

            jacobian = np.empty((12, 6), dtype=np.float64)
            epsilon = 1.0e-5
            for joint_index in range(6):
                perturbed = q.copy()
                perturbed[joint_index] += epsilon
                moved_position, moved_rotation = self.forward(perturbed)
                jacobian[:, joint_index] = np.concatenate(
                    (
                        (moved_position - position) / epsilon,
                        self.orientation_weight
                        * ((moved_rotation - rotation) / epsilon).reshape(-1),
                    )
                )
            normal_matrix = (
                jacobian.T @ jacobian + self.damping * np.eye(6)
            )
            delta = np.linalg.solve(normal_matrix, jacobian.T @ residual)
            q = np.clip(
                q + np.clip(delta, -0.15, 0.15),
                _LOWER_LIMITS,
                _UPPER_LIMITS,
            )

        solved_position, solved_rotation = self.forward(q)
        position_error = float(np.linalg.norm(target - solved_position))
        orientation_error = float(
            np.linalg.norm(target_rotation - solved_rotation)
        )
        if (
            position_error > self.position_tolerance_m
            or orientation_error > self.orientation_tolerance
        ):
            raise IKConvergenceError(
                "TCP target is outside the calibrated workspace: "
                f"position_error={position_error:.4f} m, "
                f"orientation_error={orientation_error:.4f}"
            )
        return ArmIKSolution(
            joint_positions=tuple(float(value) for value in q),
            solved_position=tuple(float(value) for value in solved_position),
            position_error_m=position_error,
            orientation_error=orientation_error,
            iterations=iteration,
        )

    def _residual(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate(
            (
                target_position - current_position,
                self.orientation_weight
                * (target_rotation - current_rotation).reshape(-1),
            )
        )

    @classmethod
    def in_robot_root_frame(cls, **kwargs) -> "CalibratedArmKinematics":
        """Construct a solver whose reference frame is the live Go2 root.

        The returned solver is independent of the root's world position and
        yaw.  Runtime code must transform the desired world TCP pose into the
        robot-root frame before calling :meth:`solve`.
        """

        if "base_transform_xyz" in kwargs:
            raise TypeError(
                "in_robot_root_frame fixes base_transform_xyz internally"
            )
        return cls(base_transform_xyz=_ROOT_FRAME_ARM_MOUNT_XYZ, **kwargs)

    @classmethod
    def in_policy_usd_root_frame(
        cls, **kwargs
    ) -> "CalibratedArmKinematics":
        """Compatibility alias for the canonical PCT URDF root frame."""

        if "base_transform_xyz" in kwargs:
            raise TypeError(
                "in_policy_usd_root_frame fixes base_transform_xyz internally"
            )
        return cls(base_transform_xyz=_ROOT_FRAME_ARM_MOUNT_XYZ, **kwargs)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _transform(
    xyz: Sequence[float],
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        _rotation_z(float(rpy[2]))
        @ _rotation_y(float(rpy[1]))
        @ _rotation_x(float(rpy[0]))
    )
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _joint_rotation(axis: tuple[int, int, int], angle: float) -> np.ndarray:
    if axis == (1, 0, 0):
        rotation = _rotation_x(angle)
    elif axis == (0, 1, 0):
        rotation = _rotation_y(angle)
    elif axis == (0, 0, 1):
        rotation = _rotation_z(angle)
    else:
        raise ValueError(f"unsupported joint axis: {axis}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def _rotation_from_quaternion(wxyz: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("target_orientation_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError("target_orientation_wxyz cannot be zero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )
    )
