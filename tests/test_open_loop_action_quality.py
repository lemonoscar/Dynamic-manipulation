from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_open_loop_action_quality.py"
SPEC = importlib.util.spec_from_file_location("open_loop_action_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALITY)


def test_action_group_ignores_invalid_cross_expert_suffix() -> None:
    stats = QUALITY.ActionGroupStats(
        ("vx", "wz"),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        sign_deadbands=(0.05, 0.05),
    )
    predicted = np.array(
        [[0.5, -0.5], [99.0, 99.0], [99.0, 99.0]], dtype=np.float64
    )
    target = np.array(
        [[1.0, -1.0], [-99.0, -99.0], [-99.0, -99.0]], dtype=np.float64
    )
    stats.update(predicted, target, (True, False, False))

    report = stats.finalize()
    full = report["full_valid_chunk"]
    assert report["rows"] == 1
    assert full["count"] == 2
    assert full["rmse"] == pytest.approx(0.5)
    assert full["mae"] == pytest.approx(0.5)
    assert full["sign_agreement"] == pytest.approx(1.0)
    assert full["out_of_contract_rate"] == pytest.approx(0.0)


def test_action_group_checks_nonfinite_values_in_the_whole_sampled_chunk() -> None:
    stats = QUALITY.ActionGroupStats(
        ("vx", "wz"),
        bounds=((-1.0, 1.0), (-1.0, 1.0)),
        sign_deadbands=(0.05, 0.05),
    )
    stats.update(
        ((0.0, 0.0), (float("nan"), 0.0)),
        ((0.0, 0.0), (0.0, 0.0)),
        (True, False),
    )
    assert not stats.finalize()["all_output_values_finite"]


def test_domain_denormalization_matches_navigation_and_gripper_contracts() -> None:
    navigation = QUALITY.denormalize_domain_actions(
        ((2.0, -2.0),),
        action_indices=(0, 2),
        action_scale=(0.3, 1.0, 0.35, 0.3, 0.3, 0.2, 0.5, 0.5, 0.5, 1.0),
        action_clip=(-1.0, 1.0),
        passthrough_indices=frozenset({9}),
        gripper_range=(0.0, 1.0),
    )
    np.testing.assert_allclose(navigation, ((0.3, -0.35),))

    manipulation = QUALITY.denormalize_domain_actions(
        ((1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.5),),
        action_indices=(3, 4, 5, 6, 7, 8, 9),
        action_scale=(0.3, 1.0, 0.35, 0.3, 0.3, 0.2, 0.5, 0.5, 0.5, 1.0),
        action_clip=(-1.0, 1.0),
        passthrough_indices=frozenset({9}),
        gripper_range=(0.0, 1.0),
    )
    np.testing.assert_allclose(
        manipulation, ((0.3, 0.0, -0.2, 0.5, 0.0, -0.5, 1.0),)
    )


def test_navigation_trajectory_exposes_wrong_way_prefix() -> None:
    stats = QUALITY.NavigationTrajectoryStats(action_rate_hz=25.0)
    stats.update(
        predicted_physical=((-0.2, 0.0),) * 5,
        target_physical=((0.2, 0.0),) * 5,
        valid_mask=(True,) * 5,
    )
    report = stats.finalize()
    assert report["predicted_reverse_rate"] == pytest.approx(1.0)
    assert report["target_reverse_rate"] == pytest.approx(0.0)
    linear = report["executed_prefix"]["by_dimension"]["linear_displacement_m"]
    assert linear["predicted_mean"] == pytest.approx(-0.04)
    assert linear["target_mean"] == pytest.approx(0.04)
    assert linear["sign_agreement"] == pytest.approx(0.0)


def test_stability_summary_detects_navigation_sign_flip_and_gripper_disagreement() -> None:
    rows = [
        {
            "sample_id": "nav",
            "phase_name": "NAV_TO_SOURCE",
            "action_domain": "NAVIGATION",
            "action_valid_mask": [True, True],
            "predictions": [
                [[0.2, 0.0], [0.2, 0.0]],
                [[-0.2, 0.0], [-0.2, 0.0]],
            ],
        },
        {
            "sample_id": "pick",
            "phase_name": "PICK",
            "action_domain": "MANIPULATION",
            "action_valid_mask": [True],
            "predictions": [
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]],
            ],
        },
    ]
    report = QUALITY.summarize_stability(
        rows, phase_names=("NAV_TO_SOURCE", "PICK")
    )
    assert report["NAV_TO_SOURCE"]["navigation_prefix_sign_flip_rate"] == 1.0
    assert report["PICK"]["gripper_seed_disagreement_rate"] == 1.0


def test_stability_selection_includes_boundary_and_interior_rows() -> None:
    annotations = [
        {
            "phase_id": phase,
            "base_index": phase * 100 + index,
            "is_boundary_window": index < 3,
            "seconds_to_boundary": float(index - 1),
        }
        for phase in range(4)
        for index in range(10)
    ]
    selected = QUALITY.select_stability_indices(
        annotations, phase_ids=(0, 1, 2, 3), samples_per_phase=4
    )
    assert len(selected) == 16
    for phase in range(4):
        rows = [annotations[index] for index in selected if annotations[index]["phase_id"] == phase]
        assert sum(row["is_boundary_window"] for row in rows) == 2
        assert sum(not row["is_boundary_window"] for row in rows) == 2
