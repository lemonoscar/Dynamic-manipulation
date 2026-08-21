from __future__ import annotations

import copy

import pytest

from scripts import compare_waypoint_v2_pilots as comparison


def _resolved(repeats: int) -> dict:
    config = {
        "action_model": {"num_inference_timesteps": 4},
        "loss": {"repeated_diffusion_steps": repeats},
        "optimization": {"learning_rate": 1.0e-5},
    }
    return {
        "resolved_policy_config": config,
        "arguments": {
            "seed": 7,
            "attention_implementation": "sdpa",
            "limit_train_episodes": 0,
        },
        "dataset_manifest_sha256": "dataset",
        "normalization_sha256": "normalizer",
        "qwen_base": {"sha256": "qwen"},
        "source_git": {"commit": "abc"},
        "world_size": 4,
        "deepspeed_zero_stage": 2,
        "batch_size_per_process": 3,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 24,
        "train_rows": 100,
        "training_subset_indices": None,
        "max_steps": 4,
        "warmup_steps": 1,
        "initialization": {
            "qwen": "clean",
            "navigation_head": "new",
            "manipulation_head": "new",
            "legacy_checkpoint_loaded": False,
        },
    }


def _events(step_seconds: float) -> list[dict]:
    return [
        {
            "event": "train_step",
            "step": step,
            "valid_optimizer_step": True,
            "navigation_loss": 10.0 / step,
            "manipulation_loss": 20.0 / step,
            "vlm_gradient_norm": 3.0 + step,
            "navigation_gradient_norm": 2.0 + step,
            "manipulation_gradient_norm": 1.0 + step,
            "optimizer_step_time_s": step_seconds,
            "samples_per_second": 24.0 / step_seconds,
            "gpu_hours_per_step": 4.0 * step_seconds / 3600.0,
            "peak_reserved_memory_mib": 40_000.0,
        }
        for step in range(1, 5)
    ]


def _evaluation(step: int, *, nav_ade: float, arm_position: float) -> dict:
    indices = list(range(32))
    return {
        "schema_version": comparison.OPEN_LOOP_SCHEMA,
        "identity": {
            "checkpoint_step": step,
            "dataset_manifest_sha256": "dataset",
            "selected_indices": indices,
        },
        "selection": {"indices": indices},
        "gate": {
            "legacy_action_route_gate": {
                "quality_metrics": {"arm_orientation_mean_rad": 0.2}
            }
        },
        "oracle_prefix_action": {
            "cross_seed": {
                "navigation_ade_mean_m": nav_ade,
                "arm_position_mean_m": arm_position,
            }
        },
        "action_diagnostics": {
            "navigation": {
                "direction_accuracy": 0.9,
                "terminal_hold_suffix_mae": 0.1,
            },
            "manipulation": {"terminal_hold_suffix_mae": 0.2},
        },
        "fixed_validation_bank": {
            "manifest_sha256": "bank",
            "navigation": {"mean_loss": 1.0},
            "manipulation": {"mean_loss": 2.0},
        },
    }


def test_comparison_selects_equal_step_and_equal_compute_pairs() -> None:
    report = comparison.compare_pilots(
        _resolved(1),
        _resolved(4),
        _events(10.0),
        _events(20.0),
        [
            _evaluation(2, nav_ade=1.0, arm_position=1.0),
            _evaluation(4, nav_ade=0.8, arm_position=0.8),
        ],
        [_evaluation(2, nav_ade=0.7, arm_position=0.7)],
        cv_window=4,
    )
    assert report["step_matched"]["checkpoint_steps"] == {"s1": 2, "s4": 2}
    assert report["gpu_hour_matched"]["checkpoint_steps"] == {"s1": 4, "s4": 2}
    assert report["promotion_evidence"]["s4_pilot_candidate"] is True


def test_comparison_rejects_lr_or_scheduler_confounds() -> None:
    s4 = _resolved(4)
    s4["resolved_policy_config"]["optimization"]["learning_rate"] = 2.0e-5
    with pytest.raises(ValueError, match="differ beyond"):
        comparison.compare_pilots(
            _resolved(1),
            s4,
            _events(10.0),
            _events(20.0),
            [_evaluation(2, nav_ade=1.0, arm_position=1.0)],
            [_evaluation(2, nav_ade=1.0, arm_position=1.0)],
            cv_window=2,
        )


def test_comparison_rejects_different_validation_noise_bank() -> None:
    s4_eval = copy.deepcopy(_evaluation(2, nav_ade=1.0, arm_position=1.0))
    s4_eval["fixed_validation_bank"]["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="fixed noise/time banks"):
        comparison.compare_pilots(
            _resolved(1),
            _resolved(4),
            _events(10.0),
            _events(20.0),
            [_evaluation(2, nav_ade=1.0, arm_position=1.0)],
            [s4_eval],
            cv_window=2,
        )
