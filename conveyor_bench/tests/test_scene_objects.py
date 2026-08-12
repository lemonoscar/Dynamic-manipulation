from conveyor_bench.schema.tasking import CurriculumSplit
from conveyor_bench.sidecar.objects import (
    COLA_SOURCE_VISUAL_AABB_SIZE,
    COLA_VISUAL_SCALE_XYZ,
    OBJECT_ASSETS,
    OBJECT_SPLITS,
    STATIONARY_TARGET_ASSET_ID,
    VISUAL_FIXTURES,
)


def test_real_object_is_a_gripper_feasible_cola_can() -> None:
    assert len(OBJECT_ASSETS) == 1
    cola = OBJECT_ASSETS[0]

    assert cola.object_id == "cola"
    assert cola.category == "can"
    assert cola.geometry == {
        "kind": "cylinder",
        "radius_m": 0.0325,
        "height_m": 0.12,
        "axis": "z",
        "sides": 32,
    }
    assert cola.mass_kg == 0.12
    assert cola.grasp_affordances[0].approach_axis == "-z"
    assert cola.grasp_affordances[0].required_opening_m <= 0.088
    assert STATIONARY_TARGET_ASSET_ID == cola.object_id
    fixture = VISUAL_FIXTURES[cola.object_id]
    assert fixture["source_aabb_size"] == COLA_SOURCE_VISUAL_AABB_SIZE
    assert fixture["scale_xyz"] == COLA_VISUAL_SCALE_XYZ
    scaled_local_size = tuple(
        source * scale
        for source, scale in zip(
            COLA_SOURCE_VISUAL_AABB_SIZE,
            COLA_VISUAL_SCALE_XYZ,
            strict=True,
        )
    )
    assert all(
        abs(actual - expected) < 1.0e-12
        for actual, expected in zip(
            scaled_local_size,
            (0.065, 0.12, 0.065),
            strict=True,
        )
    )


def test_object_split_is_single_target_and_non_leaking() -> None:
    assert OBJECT_SPLITS == {
        CurriculumSplit.TRAIN: ("cola",),
        CurriculumSplit.VAL: (),
        CurriculumSplit.UNSEEN: (),
    }
