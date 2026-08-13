"""Physical descriptors for the current real-object collection gate."""

from __future__ import annotations

from conveyor_bench.schema.assets import ObjectAsset
from conveyor_bench.schema.tasking import CurriculumSplit


# The current policy remains a single-object grasp policy.  The transferred
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
        "provenance": "assets/conveyorvla-v3/objects/cola",
    }
)

OBJECT_ASSETS = (COLA_OBJECT,)
OBJECT_SPLITS = {
    CurriculumSplit.TRAIN: (COLA_OBJECT.object_id,),
    CurriculumSplit.VAL: (),
    CurriculumSplit.UNSEEN: (),
}
STATIONARY_TARGET_ASSET_ID = COLA_OBJECT.object_id

# The source USD is authored in normalized mesh units with its can axis on Y.
# These values were measured from the hash-locked USD on Isaac Sim 5.1.  The
# transform maps that visual to its annotation size (65 x 65 x 120 mm) while
# the analytic collider remains authored directly in SI units.
COLA_SOURCE_VISUAL_AABB_SIZE = (
    1.9999998807907104,
    3.5763320922851562,
    2.0000009536743164,
)
COLA_VISUAL_SCALE_XYZ = (
    0.03250000193715107,
    0.033553930927964805,
    0.03249998450279975,
)
COLA_VISUAL_ORIENTATION_WXYZ = (
    0.7071067811865476,
    0.7071067811865475,
    0.0,
    0.0,
)
VISUAL_FIXTURES = {
    COLA_OBJECT.object_id: {
        "source_aabb_size": COLA_SOURCE_VISUAL_AABB_SIZE,
        "scale_xyz": COLA_VISUAL_SCALE_XYZ,
        "orientation_wxyz": COLA_VISUAL_ORIENTATION_WXYZ,
    }
}


__all__ = [
    "COLA_OBJECT",
    "OBJECT_ASSETS",
    "OBJECT_SPLITS",
    "STATIONARY_TARGET_ASSET_ID",
    "VISUAL_FIXTURES",
]
