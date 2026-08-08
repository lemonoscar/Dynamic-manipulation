"""Versioned, simulator-independent configuration for ConveyorBench V2.

V2 is a benchmark-suite revision over the frozen V1 canonical data protocol.
Scene and curriculum additions therefore receive V2 identities while recorded
states, actions, events, and episode manifests keep ``conveyor-bench-v1`` as
their canonical protocol version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import Any, Sequence

from conveyor_bench.v1.config import PROTOCOL_VERSION


BENCHMARK_SUITE_VERSION = "conveyor-bench-v2"
CANONICAL_PROTOCOL_VERSION = PROTOCOL_VERSION
TASK_CONTEXT_SCHEMA_VERSION = "conveyor-bench-v2-task-context-1"
CONFIG_SCHEMA_VERSION = 2


class SceneId(str, Enum):
    TRANSVERSE_NEAR_SORT_V2 = "transverse_near_sort_v2"
    MOBILE_REMOTE_DELIVERY_V2 = "mobile_remote_delivery_v2"


def _finite_vector(
    value: Sequence[float],
    length: int,
    name: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != length
    ):
        raise ValueError(f"{name} must contain exactly {length} values")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    if positive and any(component <= 0.0 for component in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _positive_number(value: float, name: str, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0.0
        or (not allow_zero and value == 0.0)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


@dataclass(frozen=True)
class GoalZoneContract:
    zone_id: str
    display_name: str
    center_xyz_m: tuple[float, float, float]
    half_extents_xyz_m: tuple[float, float, float]
    delivery_root_goal_xy_m: tuple[float, float] | None = None
    delivery_goal_yaw_rad: float | None = None
    delivery_standoff_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.zone_id, str) or not self.zone_id:
            raise ValueError("zone_id cannot be empty")
        if not isinstance(self.display_name, str) or not self.display_name:
            raise ValueError("display_name cannot be empty")
        object.__setattr__(
            self,
            "center_xyz_m",
            _finite_vector(self.center_xyz_m, 3, "center_xyz_m"),
        )
        object.__setattr__(
            self,
            "half_extents_xyz_m",
            _finite_vector(
                self.half_extents_xyz_m,
                3,
                "half_extents_xyz_m",
                positive=True,
            ),
        )
        navigation_values = (
            self.delivery_root_goal_xy_m,
            self.delivery_goal_yaw_rad,
            self.delivery_standoff_m,
        )
        if any(value is not None for value in navigation_values) and not all(
            value is not None for value in navigation_values
        ):
            raise ValueError(
                "delivery navigation fields must be provided together"
            )
        if self.delivery_root_goal_xy_m is not None:
            object.__setattr__(
                self,
                "delivery_root_goal_xy_m",
                _finite_vector(
                    self.delivery_root_goal_xy_m,
                    2,
                    "delivery_root_goal_xy_m",
                ),
            )
            assert self.delivery_goal_yaw_rad is not None
            if (
                isinstance(self.delivery_goal_yaw_rad, bool)
                or not isinstance(self.delivery_goal_yaw_rad, Real)
                or not math.isfinite(self.delivery_goal_yaw_rad)
            ):
                raise ValueError("delivery_goal_yaw_rad must be finite")
            assert self.delivery_standoff_m is not None
            object.__setattr__(
                self,
                "delivery_standoff_m",
                _positive_number(
                    self.delivery_standoff_m,
                    "delivery_standoff_m",
                ),
            )

    @property
    def min_xyz(self) -> tuple[float, float, float]:
        return tuple(
            center - half
            for center, half in zip(
                self.center_xyz_m,
                self.half_extents_xyz_m,
                strict=True,
            )
        )

    @property
    def max_xyz(self) -> tuple[float, float, float]:
        return tuple(
            center + half
            for center, half in zip(
                self.center_xyz_m,
                self.half_extents_xyz_m,
                strict=True,
            )
        )

    def to_snapshot(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "zone_id": self.zone_id,
            "display_name": self.display_name,
            "center_xyz_m": list(self.center_xyz_m),
            "goal_half_extents_xyz_m": list(self.half_extents_xyz_m),
        }
        if self.delivery_root_goal_xy_m is not None:
            value.update(
                {
                    "delivery_root_goal_xy_m": list(
                        self.delivery_root_goal_xy_m
                    ),
                    "delivery_goal_yaw_rad": self.delivery_goal_yaw_rad,
                    "delivery_standoff_m": self.delivery_standoff_m,
                }
            )
        return value


@dataclass(frozen=True)
class SceneContract:
    scene_id: SceneId
    layout_id: str
    default_max_duration_s: float
    goal_zones: tuple[GoalZoneContract, ...]
    minimum_loaded_base_displacement_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, SceneId):
            raise ValueError("scene_id must be a SceneId")
        if not isinstance(self.layout_id, str) or not self.layout_id:
            raise ValueError("layout_id cannot be empty")
        object.__setattr__(
            self,
            "default_max_duration_s",
            _positive_number(
                self.default_max_duration_s,
                "default_max_duration_s",
            ),
        )
        object.__setattr__(
            self,
            "minimum_loaded_base_displacement_m",
            _positive_number(
                self.minimum_loaded_base_displacement_m,
                "minimum_loaded_base_displacement_m",
                allow_zero=True,
            ),
        )
        if not self.goal_zones:
            raise ValueError("scene goal_zones cannot be empty")
        if any(
            not isinstance(zone, GoalZoneContract) for zone in self.goal_zones
        ):
            raise ValueError("goal_zones must contain GoalZoneContract values")
        zone_ids = tuple(zone.zone_id for zone in self.goal_zones)
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("scene goal-zone IDs must be unique")
        has_navigation = tuple(
            zone.delivery_root_goal_xy_m is not None for zone in self.goal_zones
        )
        if self.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2 and not all(
            has_navigation
        ):
            raise ValueError("remote delivery zones require navigation contracts")
        if self.scene_id is SceneId.TRANSVERSE_NEAR_SORT_V2 and any(
            has_navigation
        ):
            raise ValueError("near-sort zones cannot carry remote navigation goals")

    @property
    def goal_zone_by_id(self) -> dict[str, GoalZoneContract]:
        return {zone.zone_id: zone for zone in self.goal_zones}

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "layout_id": self.layout_id,
            "default_max_duration_s": self.default_max_duration_s,
            "minimum_loaded_base_displacement_m": (
                self.minimum_loaded_base_displacement_m
            ),
            "goal_zones": [zone.to_snapshot() for zone in self.goal_zones],
        }


@dataclass(frozen=True)
class TaskCombination:
    scene_id: SceneId
    task_family: str
    robot_mode: str

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, SceneId):
            raise ValueError("scene_id must be a SceneId")
        if self.task_family not in {
            "single_target",
            "language_conditioned",
            "continuous_multi_target",
        }:
            raise ValueError("unsupported task_family in V2 matrix")
        if self.robot_mode not in {"fixed_base", "whole_body_policy"}:
            raise ValueError("unsupported robot_mode in V2 matrix")

    def to_snapshot(self) -> dict[str, str]:
        return {
            "scene_id": self.scene_id.value,
            "task_family": self.task_family,
            "robot_mode": self.robot_mode,
        }


def _default_scenes() -> tuple[SceneContract, ...]:
    near_half_extents = (0.105, 0.125, 0.075)
    remote_half_extents = (0.105, 0.125, 0.075)
    return (
        SceneContract(
            scene_id=SceneId.TRANSVERSE_NEAR_SORT_V2,
            layout_id="transverse_dynamic_sort_station_v1",
            default_max_duration_s=45.0,
            goal_zones=(
                GoalZoneContract(
                    "sort_bin_blue",
                    "blue sorting tray",
                    (0.34, 0.40, 0.40),
                    near_half_extents,
                ),
                GoalZoneContract(
                    "sort_bin_yellow",
                    "yellow sorting tray",
                    (0.34, -0.40, 0.40),
                    near_half_extents,
                ),
            ),
        ),
        SceneContract(
            scene_id=SceneId.MOBILE_REMOTE_DELIVERY_V2,
            layout_id="mobile_remote_delivery_station_v2",
            default_max_duration_s=60.0,
            minimum_loaded_base_displacement_m=0.65,
            goal_zones=(
                GoalZoneContract(
                    "delivery_bin_blue",
                    "blue remote delivery bin",
                    (-0.16, 1.20, 0.46),
                    remote_half_extents,
                    delivery_root_goal_xy_m=(-0.16, 0.78),
                    delivery_goal_yaw_rad=math.pi / 2.0,
                    delivery_standoff_m=0.42,
                ),
                GoalZoneContract(
                    "delivery_bin_yellow",
                    "yellow remote delivery bin",
                    (-0.16, -1.20, 0.46),
                    remote_half_extents,
                    delivery_root_goal_xy_m=(-0.16, -0.78),
                    delivery_goal_yaw_rad=-math.pi / 2.0,
                    delivery_standoff_m=0.42,
                ),
            ),
        ),
    )


def _default_combinations() -> tuple[TaskCombination, ...]:
    near = SceneId.TRANSVERSE_NEAR_SORT_V2
    remote = SceneId.MOBILE_REMOTE_DELIVERY_V2
    return (
        TaskCombination(near, "single_target", "fixed_base"),
        TaskCombination(near, "single_target", "whole_body_policy"),
        TaskCombination(near, "language_conditioned", "fixed_base"),
        TaskCombination(near, "language_conditioned", "whole_body_policy"),
        TaskCombination(near, "continuous_multi_target", "fixed_base"),
        TaskCombination(remote, "single_target", "whole_body_policy"),
        TaskCombination(remote, "language_conditioned", "whole_body_policy"),
    )


@dataclass(frozen=True)
class V2SuiteConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    benchmark_suite_version: str = BENCHMARK_SUITE_VERSION
    canonical_protocol_version: str = CANONICAL_PROTOCOL_VERSION
    task_context_schema_version: str = TASK_CONTEXT_SCHEMA_VERSION
    continuous_target_count: int = 2
    belt_speeds_mps: tuple[float, ...] = (0.06, 0.08, 0.10)
    scenes: tuple[SceneContract, ...] = field(default_factory=_default_scenes)
    allowed_combinations: tuple[TaskCombination, ...] = field(
        default_factory=_default_combinations
    )

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION}")
        if self.benchmark_suite_version != BENCHMARK_SUITE_VERSION:
            raise ValueError(
                f"benchmark_suite_version must be {BENCHMARK_SUITE_VERSION!r}"
            )
        if self.canonical_protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                "V2 canonical_protocol_version must remain conveyor-bench-v1"
            )
        if self.task_context_schema_version != TASK_CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"task_context_schema_version must be "
                f"{TASK_CONTEXT_SCHEMA_VERSION!r}"
            )
        if self.continuous_target_count != 2:
            raise ValueError("V2 initially requires exactly two continuous targets")
        speeds = tuple(
            _positive_number(speed, "belt_speeds_mps")
            for speed in self.belt_speeds_mps
        )
        if len(set(speeds)) != len(speeds) or not speeds:
            raise ValueError("belt_speeds_mps must be non-empty and unique")
        object.__setattr__(self, "belt_speeds_mps", speeds)
        if any(not isinstance(scene, SceneContract) for scene in self.scenes):
            raise ValueError("scenes must contain SceneContract values")
        scene_ids = tuple(scene.scene_id for scene in self.scenes)
        if set(scene_ids) != set(SceneId) or len(scene_ids) != len(set(scene_ids)):
            raise ValueError("V2 config must define every scene exactly once")
        if any(
            not isinstance(combination, TaskCombination)
            for combination in self.allowed_combinations
        ):
            raise ValueError(
                "allowed_combinations must contain TaskCombination values"
            )
        keys = tuple(
            (
                combination.scene_id,
                combination.task_family,
                combination.robot_mode,
            )
            for combination in self.allowed_combinations
        )
        if len(keys) != len(set(keys)):
            raise ValueError("allowed task combinations must be unique")

    def scene(self, scene_id: SceneId | str) -> SceneContract:
        try:
            resolved = (
                scene_id
                if isinstance(scene_id, SceneId)
                else SceneId(scene_id)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported V2 scene_id: {scene_id!r}") from error
        for scene in self.scenes:
            if scene.scene_id is resolved:
                return scene
        raise ValueError(f"unsupported V2 scene_id: {scene_id!r}")

    def supports(self, scene_id: SceneId, task_family: str, robot_mode: str) -> bool:
        return any(
            combination.scene_id is scene_id
            and combination.task_family == task_family
            and combination.robot_mode == robot_mode
            for combination in self.allowed_combinations
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_suite_version": self.benchmark_suite_version,
            "canonical_protocol_version": self.canonical_protocol_version,
            "task_context_schema_version": self.task_context_schema_version,
            "continuous_target_count": self.continuous_target_count,
            "belt_speeds_mps": list(self.belt_speeds_mps),
            "scenes": {
                scene.scene_id.value: scene.to_snapshot() for scene in self.scenes
            },
            "allowed_combinations": [
                combination.to_snapshot()
                for combination in self.allowed_combinations
            ],
        }


DEFAULT_SUITE_CONFIG = V2SuiteConfig()


__all__ = [
    "BENCHMARK_SUITE_VERSION",
    "CANONICAL_PROTOCOL_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_SUITE_CONFIG",
    "TASK_CONTEXT_SCHEMA_VERSION",
    "GoalZoneContract",
    "SceneContract",
    "SceneId",
    "TaskCombination",
    "V2SuiteConfig",
]
