"""Physical descriptors for the first V3 real-object collection gate."""

from __future__ import annotations

from conveyor_bench.v1.assets import ObjectAsset
from conveyor_bench.v1.tasking import CurriculumSplit


# The first V3 policy remains a single-object grasp policy.  The transferred
# apple/orange/bottle assets stay available for later gripper-feasibility
# fixtures; only the 65 mm can fits the audited 88 mm FinRay opening today.
COLA_OBJECT = ObjectAsset.from_dict(
    {
        "object_id": "cola",
        "display_name": "Coca-Cola can",
        "category": "can",
        "attributes": {"color": "red", "material": "aluminum"},
        "language_aliases": {
            "en": ["Coca-Cola can", "red can"],
            "zh": ["可乐罐", "红色易拉罐"],
        },
        "geometry": {
            "kind": "cylinder",
            "radius_m": 0.0325,
            "height_m": 0.12,
            "axis": "z",
            "sides": 32,
        },
        "physics": {
            "mass_kg": 0.12,
            "static_friction": 1.10,
            "dynamic_friction": 0.90,
            "restitution": 0.0,
            "angular_damping": 5.0,
        },
        "stable_poses_wxyz": [[1.0, 0.0, 0.0, 0.0]],
        "grasp_affordances": [
            {
                "id": "top_parallel_y_can_body",
                "approach_axis": "-z",
                "finger_closing_axis": "y",
                "tcp_offset_xyz": [0.0, 0.0, 0.006],
                "required_opening_m": 0.072,
            }
        ],
        "split": "seen",
        "real_twin_id": "mesa-can-0364ab96f338493c972248102b462aa4",
        "license": "ssh-sidecar-asset-metadata",
        "provenance": "conveyorvla-v3-assets-20260811/objects/cola",
    }
)

V3_OBJECT_ASSETS = (COLA_OBJECT,)
V3_OBJECT_SPLITS = {
    CurriculumSplit.TRAIN: (COLA_OBJECT.object_id,),
    CurriculumSplit.VAL: (),
    CurriculumSplit.UNSEEN: (),
}
V3_STATIONARY_TARGET_ASSET_ID = COLA_OBJECT.object_id


__all__ = [
    "COLA_OBJECT",
    "V3_OBJECT_ASSETS",
    "V3_OBJECT_SPLITS",
    "V3_STATIONARY_TARGET_ASSET_ID",
]
