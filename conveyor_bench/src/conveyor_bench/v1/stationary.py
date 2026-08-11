"""Frozen, fail-closed contract for the V1 stationary diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


STATIONARY_TARGET_ASSET_ID = "part_red_block"
STATIONARY_DESTINATION_ZONE_ID = "sort_bin_blue"
STATIONARY_SPAWN_ORIGIN_XY_M = (0.65, 0.10)
V3_NUREC_BACKEND = "isaac_rtx_native_nurec"


@dataclass(frozen=True)
class StationaryScenario:
    split: str
    object_xy_offset_m: tuple[float, float]
    root_xy_offset_m: tuple[float, float]
    root_yaw_rad: float


STATIONARY_SCENARIOS = {
    1101: StationaryScenario("train", (0.0, 0.0), (0.0, 0.0), 0.0),
    1102: StationaryScenario("train", (0.020, 0.020), (0.0, 0.0), 0.0),
    1103: StationaryScenario("train", (-0.020, -0.020), (0.0, 0.0), 0.0),
    2101: StationaryScenario("val", (0.010, -0.025), (0.0, 0.0), 0.0),
    3101: StationaryScenario("test", (-0.010, 0.025), (0.0, 0.0), 0.0),
}


def require_stationary_scenario(seed: Any) -> StationaryScenario:
    """Resolve a registered scenario without accepting bools or coercion."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("task.seed must be an integer")
    try:
        return STATIONARY_SCENARIOS[seed]
    except KeyError as error:
        raise ValueError(f"task.seed {seed} is not a registered scenario") from error


def stationary_spawn_xy(scenario: StationaryScenario) -> tuple[float, float]:
    return tuple(
        origin + offset
        for origin, offset in zip(
            STATIONARY_SPAWN_ORIGIN_XY_M,
            scenario.object_xy_offset_m,
            strict=True,
        )
    )


def validate_stationary_episode_contract(
    episode: Mapping[str, Any],
) -> StationaryScenario:
    """Bind a stationary manifest to its registered seed and fixed task."""

    task = _mapping(episode.get("task"), "episode.task")
    if task.get("task_type") != "stationary_sort":
        raise ValueError("task.task_type must be stationary_sort")
    belt_speed = _number(task.get("belt_speed_mps"), "task.belt_speed_mps")
    if not math.isclose(belt_speed, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("task.belt_speed_mps must be zero")

    seed = task.get("seed")
    scenario = require_stationary_scenario(seed)
    seeds = _mapping(episode.get("seeds"), "episode.seeds")
    if seeds.get("episode") != seed or seeds.get("layout") != seed:
        raise ValueError("episode and layout seeds must equal task.seed")

    metadata = _mapping(task.get("metadata"), "task.metadata")
    target_asset_id = _stationary_target_asset_id(episode)
    expected_metadata = {
        "task_family": "single_target",
        "target_asset_id": target_asset_id,
        "destination_zone_id": STATIONARY_DESTINATION_ZONE_ID,
        "benchmark_role": "stationary_belt_diagnostic",
        "belt_motion": "stationary",
        "active_object_count": 1,
    }
    mismatched = [
        key for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    ]
    if mismatched:
        raise ValueError(f"task metadata fields do not match: {mismatched}")

    raw_scenario = _mapping(
        metadata.get("stationary_scenario"),
        "task.metadata.stationary_scenario",
    )
    if raw_scenario.get("scenario_id") != seed:
        raise ValueError("stationary scenario_id must equal task.seed")
    if raw_scenario.get("scenario_split") != scenario.split:
        raise ValueError("stationary scenario_split does not match the registry")
    _require_vector_match(
        raw_scenario.get("object_xy_offset_m"),
        scenario.object_xy_offset_m,
        "stationary object_xy_offset_m",
    )
    _require_vector_match(
        raw_scenario.get("root_xy_offset_m"),
        scenario.root_xy_offset_m,
        "stationary root_xy_offset_m",
    )
    root_yaw = _number(
        raw_scenario.get("root_yaw_rad"),
        "stationary root_yaw_rad",
    )
    if not math.isclose(
        root_yaw, scenario.root_yaw_rad, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("stationary root_yaw_rad does not match the registry")

    objects = _sequence(task.get("objects"), "task.objects")
    if len(objects) != 1:
        raise ValueError("stationary_sort requires exactly one task object")
    target = _mapping(objects[0], "task.objects[0]")
    if target.get("asset_id") != target_asset_id:
        raise ValueError(
            f"stationary task object must use {target_asset_id}"
        )
    if target.get("goal_zone_id") != STATIONARY_DESTINATION_ZONE_ID:
        raise ValueError("stationary task object must target sort_bin_blue")
    instance_id = target.get("instance_id")
    scored = _sequence(task.get("scored_object_ids"), "task.scored_object_ids")
    if not isinstance(instance_id, str) or list(scored) != [instance_id]:
        raise ValueError("stationary scored_object_ids must contain the target")
    expected_spawn_x, expected_spawn_y = stationary_spawn_xy(scenario)
    for key, expected in (
        ("spawn_x_by_id", expected_spawn_x),
        ("spawn_y_by_id", expected_spawn_y),
    ):
        spawn_by_id = _mapping(metadata.get(key), f"task.metadata.{key}")
        if set(spawn_by_id) != {instance_id} or not math.isclose(
            _number(spawn_by_id.get(instance_id), key),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"stationary {key} does not match the registry")
    goal_zones = _sequence(task.get("goal_zones"), "task.goal_zones")
    if not any(
        isinstance(zone, Mapping)
        and zone.get("zone_id") == STATIONARY_DESTINATION_ZONE_ID
        for zone in goal_zones
    ):
        raise ValueError("stationary task must register sort_bin_blue")
    return scenario


def _stationary_target_asset_id(episode: Mapping[str, Any]) -> str:
    """Resolve the V1 default or one verified profile-owned rigid fixture."""

    metadata = episode.get("metadata")
    if not isinstance(metadata, Mapping):
        return STATIONARY_TARGET_ASSET_ID
    scene = metadata.get("scene_profile")
    if not isinstance(scene, Mapping) or scene.get("backend") != V3_NUREC_BACKEND:
        return STATIONARY_TARGET_ASSET_ID
    fixture = _mapping(
        scene.get("object_fixture_contract"),
        "episode.metadata.scene_profile.object_fixture_contract",
    )
    if (
        fixture.get("all_rigid_bodies_valid") is not True
        or fixture.get("all_visuals_composed") is not True
    ):
        raise ValueError("V3 stationary object fixture did not pass")
    objects = _sequence(fixture.get("objects"), "V3 object fixtures")
    object_ids = [
        item.get("object_id")
        for item in objects
        if isinstance(item, Mapping)
    ]
    if (
        len(objects) != 1
        or len(object_ids) != 1
        or not isinstance(object_ids[0], str)
        or not object_ids[0]
    ):
        raise ValueError("V3 stationary profile requires one object fixture")
    return object_ids[0]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_vector_match(
    value: Any,
    expected: Sequence[float],
    name: str,
) -> None:
    vector = _sequence(value, name)
    if len(vector) != len(expected) or any(
        not math.isclose(
            _number(component, name), reference, rel_tol=0.0, abs_tol=1.0e-12
        )
        for component, reference in zip(vector, expected, strict=True)
    ):
        raise ValueError(f"{name} does not match the registry")
