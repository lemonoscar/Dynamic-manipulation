"""Immutable identities and local physical goals for Waypoint Policy v2."""

from __future__ import annotations

from typing import Mapping

from conveyor_bench.conveyorvla.waypoint import WaypointRoute


MODEL_CONTRACT_ID_V2 = "qwen3vl-layerwise-dual-fm-waypoint-v2"
# Frozen legacy identity.  Its ARM gripper channel came from measured finger
# opening and must remain loadable for historical checkpoint evaluation only.
DATASET_SCHEMA_VERSION_V2 = "conveyorvla-waypoint-dense-transition-v2"
DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER = (
    "conveyorvla-waypoint-dense-transition-v2-command-gripper-v1"
)
DATASET_SCHEMA_VERSIONS_V2 = frozenset(
    {DATASET_SCHEMA_VERSION_V2, DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER}
)
RUNTIME_PROTOCOL_VERSION_V2 = "conveyorvla-waypoint-runtime/v2"
POLICY_CONFIG_SCHEMA_VERSION_V2 = "conveyorvla-waypoint-policy-config-v2"
DATASET_TRANSFORM_VERSION_V2 = "conveyorvla-waypoint-v1-to-v2-terminal-hold-v1"
DATASET_TRANSFORM_VERSION_V2_COMMAND_GRIPPER = (
    "conveyorvla-waypoint-v1-to-v2-terminal-hold-command-gripper-v2"
)

EXPECTED_NEXT_ROUTE: Mapping[WaypointRoute, WaypointRoute] = {
    WaypointRoute.NAV_TO_SOURCE: WaypointRoute.PICK,
    WaypointRoute.PICK: WaypointRoute.NAV_TO_TARGET,
    WaypointRoute.NAV_TO_TARGET: WaypointRoute.PLACE,
    WaypointRoute.PLACE: WaypointRoute.DONE,
}

BOUNDARY_EVENTS: Mapping[str, str] = {
    "NAV_TO_SOURCE->PICK": "base_stopped_source_in_reach",
    "PICK->NAV_TO_TARGET": "grasp_lifted_carry_ready",
    "NAV_TO_TARGET->PLACE": "base_stopped_target_in_reach",
    "PLACE->DONE": "released_in_target",
}

LOCAL_CRL_GOALS: Mapping[WaypointRoute, str] = {
    WaypointRoute.NAV_TO_SOURCE: "source is reachable and the base is ready",
    WaypointRoute.PICK: "object is firmly grasped, lifted and carry-ready",
    WaypointRoute.NAV_TO_TARGET: "target is reachable while carrying the object",
    WaypointRoute.PLACE: "object is released inside the target",
}


__all__ = [
    "BOUNDARY_EVENTS",
    "DATASET_SCHEMA_VERSION_V2",
    "DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER",
    "DATASET_SCHEMA_VERSIONS_V2",
    "DATASET_TRANSFORM_VERSION_V2",
    "DATASET_TRANSFORM_VERSION_V2_COMMAND_GRIPPER",
    "EXPECTED_NEXT_ROUTE",
    "LOCAL_CRL_GOALS",
    "MODEL_CONTRACT_ID_V2",
    "POLICY_CONFIG_SCHEMA_VERSION_V2",
    "RUNTIME_PROTOCOL_VERSION_V2",
]
