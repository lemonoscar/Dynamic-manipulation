#!/usr/bin/env python3
"""Evaluate sampled routed actions against expert actions on an offline split.

The primary pass fixes the annotated expert route so that the report measures the
two action experts rather than mixing action error with VLM routing error.  Only
the valid same-expert prefix contributes to quality metrics.  A small repeated-
seed probe measures diffusion sampling stability on the same observations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


SCHEMA_VERSION = "conveyor-vla-al0-open-loop-action-quality-eval-1"
STANDARD_PROFILE = "basic-v1"
EXECUTED_PREFIX_STEPS = 5
SIGN_DEADBAND_NORMALIZED = 0.05
STANDARD_THRESHOLDS = {
    "max_phase_normalized_rmse": 0.75,
    "min_phase_skill_vs_zero": 0.0,
    "max_normalized_out_of_contract_rate": 0.05,
    "min_navigation_prefix_direction_accuracy": 0.80,
    "max_nav_to_source_prefix_reverse_rate": 0.05,
    "min_manipulation_gripper_accuracy": 0.90,
    "max_phase_mean_sampling_std": 0.20,
}


class VectorPairStats:
    """Streaming paired-vector statistics with per-dimension accounting."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        bounds: Sequence[tuple[float, float] | None],
        sign_deadbands: Sequence[float | None],
    ) -> None:
        if not names or len(names) != len(bounds) or len(names) != len(sign_deadbands):
            raise ValueError("names, bounds, and sign deadbands must have equal nonzero length")
        self.names = tuple(str(value) for value in names)
        self.bounds = tuple(bounds)
        self.sign_deadbands = tuple(sign_deadbands)
        size = len(self.names)
        self.count = np.zeros(size, dtype=np.int64)
        self.finite = np.zeros(size, dtype=np.int64)
        self.sum_pred = np.zeros(size, dtype=np.float64)
        self.sum_target = np.zeros(size, dtype=np.float64)
        self.sum_pred_sq = np.zeros(size, dtype=np.float64)
        self.sum_target_sq = np.zeros(size, dtype=np.float64)
        self.sum_cross = np.zeros(size, dtype=np.float64)
        self.sum_abs_error = np.zeros(size, dtype=np.float64)
        self.sum_sq_error = np.zeros(size, dtype=np.float64)
        self.sum_error = np.zeros(size, dtype=np.float64)
        self.minimum_pred = np.full(size, np.inf, dtype=np.float64)
        self.maximum_pred = np.full(size, -np.inf, dtype=np.float64)
        self.minimum_target = np.full(size, np.inf, dtype=np.float64)
        self.maximum_target = np.full(size, -np.inf, dtype=np.float64)
        self.out_of_contract = np.zeros(size, dtype=np.int64)
        self.saturated = np.zeros(size, dtype=np.int64)
        self.sign_eligible = np.zeros(size, dtype=np.int64)
        self.sign_correct = np.zeros(size, dtype=np.int64)

    def update(self, predicted: Any, target: Any) -> None:
        predicted_array = np.asarray(predicted, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if predicted_array.ndim == 1:
            predicted_array = predicted_array.reshape(1, -1)
        if target_array.ndim == 1:
            target_array = target_array.reshape(1, -1)
        expected = (predicted_array.shape[0], len(self.names))
        if predicted_array.shape != expected or target_array.shape != expected:
            raise ValueError(
                f"paired vectors must have shape (N, {len(self.names)}), got "
                f"{predicted_array.shape}/{target_array.shape}"
            )
        if not np.isfinite(target_array).all():
            raise ValueError("expert target contains a non-finite value")
        self.count += predicted_array.shape[0]
        for index in range(len(self.names)):
            values = predicted_array[:, index]
            targets = target_array[:, index]
            finite = np.isfinite(values)
            self.finite[index] += int(finite.sum())
            if not finite.any():
                continue
            values = values[finite]
            targets = targets[finite]
            errors = values - targets
            self.sum_pred[index] += float(values.sum())
            self.sum_target[index] += float(targets.sum())
            self.sum_pred_sq[index] += float(np.square(values).sum())
            self.sum_target_sq[index] += float(np.square(targets).sum())
            self.sum_cross[index] += float((values * targets).sum())
            self.sum_abs_error[index] += float(np.abs(errors).sum())
            self.sum_sq_error[index] += float(np.square(errors).sum())
            self.sum_error[index] += float(errors.sum())
            self.minimum_pred[index] = min(self.minimum_pred[index], float(values.min()))
            self.maximum_pred[index] = max(self.maximum_pred[index], float(values.max()))
            self.minimum_target[index] = min(
                self.minimum_target[index], float(targets.min())
            )
            self.maximum_target[index] = max(
                self.maximum_target[index], float(targets.max())
            )
            bounds = self.bounds[index]
            if bounds is not None:
                low, high = bounds
                self.out_of_contract[index] += int(
                    np.logical_or(values < low, values > high).sum()
                )
                tolerance = max(1.0e-6, (high - low) * 5.0e-4)
                self.saturated[index] += int(
                    np.logical_or(values <= low + tolerance, values >= high - tolerance).sum()
                )
            deadband = self.sign_deadbands[index]
            if deadband is not None:
                eligible = np.abs(targets) >= deadband
                self.sign_eligible[index] += int(eligible.sum())
                self.sign_correct[index] += int(
                    (np.signbit(values[eligible]) == np.signbit(targets[eligible])).sum()
                )

    def finalize(self) -> dict[str, Any]:
        dimensions = {
            name: self._dimension(index) for index, name in enumerate(self.names)
        }
        finite = int(self.finite.sum())
        count = int(self.count.sum())
        sq_error = float(self.sum_sq_error.sum())
        target_sq = float(self.sum_target_sq.sum())
        sign_eligible = int(self.sign_eligible.sum())
        return {
            "count": count,
            "finite_count": finite,
            "finite_rate": _ratio(finite, count),
            "mae": _ratio(float(self.sum_abs_error.sum()), finite),
            "rmse": math.sqrt(_ratio(sq_error, finite)) if finite else None,
            "bias": _ratio(float(self.sum_error.sum()), finite),
            "zero_baseline_rmse": math.sqrt(_ratio(target_sq, finite)) if finite else None,
            "skill_vs_zero": (1.0 - sq_error / target_sq) if target_sq > 0.0 else None,
            "out_of_contract_rate": _ratio(
                int(self.out_of_contract.sum()), finite
            ),
            "saturation_rate": _ratio(int(self.saturated.sum()), finite),
            "sign_eligible_count": sign_eligible,
            "sign_agreement": _ratio(int(self.sign_correct.sum()), sign_eligible),
            "by_dimension": dimensions,
        }

    def _dimension(self, index: int) -> dict[str, Any]:
        count = int(self.count[index])
        finite = int(self.finite[index])
        if not finite:
            return {
                "count": count,
                "finite_count": 0,
                "finite_rate": _ratio(0, count),
            }
        sum_pred = float(self.sum_pred[index])
        sum_target = float(self.sum_target[index])
        sum_pred_sq = float(self.sum_pred_sq[index])
        sum_target_sq = float(self.sum_target_sq[index])
        sum_cross = float(self.sum_cross[index])
        sq_error = float(self.sum_sq_error[index])
        pred_variance_sum = max(0.0, sum_pred_sq - sum_pred * sum_pred / finite)
        target_variance_sum = max(0.0, sum_target_sq - sum_target * sum_target / finite)
        covariance_sum = sum_cross - sum_pred * sum_target / finite
        correlation = None
        if pred_variance_sum > 0.0 and target_variance_sum > 0.0:
            correlation = covariance_sum / math.sqrt(
                pred_variance_sum * target_variance_sum
            )
        sign_eligible = int(self.sign_eligible[index])
        return {
            "count": count,
            "finite_count": finite,
            "finite_rate": finite / count,
            "mae": float(self.sum_abs_error[index]) / finite,
            "rmse": math.sqrt(sq_error / finite),
            "bias": float(self.sum_error[index]) / finite,
            "predicted_mean": sum_pred / finite,
            "predicted_std": math.sqrt(pred_variance_sum / finite),
            "predicted_min": float(self.minimum_pred[index]),
            "predicted_max": float(self.maximum_pred[index]),
            "target_mean": sum_target / finite,
            "target_std": math.sqrt(target_variance_sum / finite),
            "target_min": float(self.minimum_target[index]),
            "target_max": float(self.maximum_target[index]),
            "pearson_correlation": correlation,
            "r2_vs_mean": (
                1.0 - sq_error / target_variance_sum
                if target_variance_sum > 0.0
                else None
            ),
            "skill_vs_zero": 1.0 - sq_error / sum_target_sq if sum_target_sq > 0.0 else None,
            "out_of_contract_rate": int(self.out_of_contract[index]) / finite,
            "saturation_rate": int(self.saturated[index]) / finite,
            "sign_eligible_count": sign_eligible,
            "sign_agreement": _ratio(int(self.sign_correct[index]), sign_eligible),
        }


class ActionGroupStats:
    """First action, executed prefix, full valid chunk, and adjacent deltas."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        bounds: Sequence[tuple[float, float] | None],
        sign_deadbands: Sequence[float | None],
    ) -> None:
        factory = lambda: VectorPairStats(  # noqa: E731 - keeps the contract together
            names, bounds=bounds, sign_deadbands=sign_deadbands
        )
        self.first_action = factory()
        self.executed_prefix = factory()
        self.full_valid_chunk = factory()
        self.adjacent_delta = factory()
        self.rows = 0
        self.zero_valid_rows = 0
        self.output_values = 0
        self.finite_output_values = 0

    def update(
        self,
        predicted: Any,
        target: Any,
        valid_mask: Sequence[bool],
    ) -> None:
        predicted_array = np.asarray(predicted, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=np.bool_)
        if predicted_array.ndim != 2 or target_array.shape != predicted_array.shape:
            raise ValueError("predicted and target actions must be equal 2-D arrays")
        if mask.shape != (predicted_array.shape[0],):
            raise ValueError("action valid mask must match the action horizon")
        self.rows += 1
        self.output_values += int(predicted_array.size)
        self.finite_output_values += int(np.isfinite(predicted_array).sum())
        valid_indices = np.flatnonzero(mask)
        if not len(valid_indices):
            self.zero_valid_rows += 1
            return
        if not np.array_equal(valid_indices, np.arange(len(valid_indices))):
            raise ValueError("action_valid_mask must be a valid prefix")
        valid_predicted = predicted_array[valid_indices]
        valid_target = target_array[valid_indices]
        self.full_valid_chunk.update(valid_predicted, valid_target)
        self.first_action.update(valid_predicted[:1], valid_target[:1])
        prefix = min(EXECUTED_PREFIX_STEPS, len(valid_indices))
        self.executed_prefix.update(valid_predicted[:prefix], valid_target[:prefix])
        if len(valid_indices) > 1:
            self.adjacent_delta.update(
                np.diff(valid_predicted, axis=0), np.diff(valid_target, axis=0)
            )

    def finalize(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "zero_valid_rows": self.zero_valid_rows,
            "all_output_values_finite": self.output_values == self.finite_output_values,
            "output_values": self.output_values,
            "finite_output_values": self.finite_output_values,
            "first_action": self.first_action.finalize(),
            "executed_prefix": self.executed_prefix.finalize(),
            "full_valid_chunk": self.full_valid_chunk.finalize(),
            "adjacent_delta": self.adjacent_delta.finalize(),
        }


class NavigationTrajectoryStats:
    def __init__(self, action_rate_hz: float) -> None:
        if action_rate_hz <= 0.0:
            raise ValueError("action rate must be positive")
        self.dt = 1.0 / action_rate_hz
        bounds = (None, None)
        deadbands = (0.005, 0.01)
        names = ("linear_displacement_m", "yaw_change_rad")
        self.executed_prefix = VectorPairStats(
            names, bounds=bounds, sign_deadbands=deadbands
        )
        self.full_valid_chunk = VectorPairStats(
            names, bounds=bounds, sign_deadbands=deadbands
        )
        self.prefix_rows = 0
        self.prefix_predicted_reverse = 0
        self.prefix_target_reverse = 0
        self.prefix_vx_steps = 0
        self.prefix_predicted_negative_vx_steps = 0
        self.prefix_target_negative_vx_steps = 0

    def update(
        self,
        predicted_physical: Any,
        target_physical: Any,
        valid_mask: Sequence[bool],
    ) -> None:
        predicted = np.asarray(predicted_physical, dtype=np.float64)
        target = np.asarray(target_physical, dtype=np.float64)
        valid_count = int(np.asarray(valid_mask, dtype=np.bool_).sum())
        if not valid_count:
            return
        predicted = predicted[:valid_count]
        target = target[:valid_count]
        prefix = min(EXECUTED_PREFIX_STEPS, valid_count)
        predicted_prefix = predicted[:prefix]
        target_prefix = target[:prefix]
        predicted_integral = predicted_prefix.sum(axis=0) * self.dt
        target_integral = target_prefix.sum(axis=0) * self.dt
        self.executed_prefix.update(predicted_integral, target_integral)
        self.full_valid_chunk.update(
            predicted.sum(axis=0) * self.dt, target.sum(axis=0) * self.dt
        )
        self.prefix_rows += 1
        self.prefix_predicted_reverse += int(predicted_integral[0] < -0.005)
        self.prefix_target_reverse += int(target_integral[0] < -0.005)
        self.prefix_vx_steps += prefix
        self.prefix_predicted_negative_vx_steps += int(
            (predicted_prefix[:, 0] < -0.02).sum()
        )
        self.prefix_target_negative_vx_steps += int(
            (target_prefix[:, 0] < -0.02).sum()
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "action_period_s": self.dt,
            "executed_prefix_steps": EXECUTED_PREFIX_STEPS,
            "executed_prefix": self.executed_prefix.finalize(),
            "full_valid_chunk": self.full_valid_chunk.finalize(),
            "prefix_rows": self.prefix_rows,
            "predicted_reverse_rate": _ratio(
                self.prefix_predicted_reverse, self.prefix_rows
            ),
            "target_reverse_rate": _ratio(self.prefix_target_reverse, self.prefix_rows),
            "predicted_negative_vx_step_rate": _ratio(
                self.prefix_predicted_negative_vx_steps, self.prefix_vx_steps
            ),
            "target_negative_vx_step_rate": _ratio(
                self.prefix_target_negative_vx_steps, self.prefix_vx_steps
            ),
        }


class GripperStats:
    def __init__(self) -> None:
        self.full_count = 0
        self.full_correct = 0
        self.full_predicted_open = 0
        self.full_target_open = 0
        self.prefix_count = 0
        self.prefix_correct = 0

    def update(
        self,
        predicted_physical: Any,
        target_physical: Any,
        valid_mask: Sequence[bool],
    ) -> None:
        predicted = np.asarray(predicted_physical, dtype=np.float64)
        target = np.asarray(target_physical, dtype=np.float64)
        valid_count = int(np.asarray(valid_mask, dtype=np.bool_).sum())
        if not valid_count:
            return
        predicted_open = predicted[:valid_count, -1] >= 0.5
        target_open = target[:valid_count, -1] >= 0.5
        self.full_count += valid_count
        self.full_correct += int((predicted_open == target_open).sum())
        self.full_predicted_open += int(predicted_open.sum())
        self.full_target_open += int(target_open.sum())
        prefix = min(EXECUTED_PREFIX_STEPS, valid_count)
        self.prefix_count += prefix
        self.prefix_correct += int(
            (predicted_open[:prefix] == target_open[:prefix]).sum()
        )

    def finalize(self) -> dict[str, Any]:
        return {
            "threshold_open_fraction": 0.5,
            "full_valid_chunk": {
                "count": self.full_count,
                "binary_accuracy": _ratio(self.full_correct, self.full_count),
                "predicted_open_rate": _ratio(
                    self.full_predicted_open, self.full_count
                ),
                "target_open_rate": _ratio(self.full_target_open, self.full_count),
            },
            "executed_prefix": {
                "count": self.prefix_count,
                "binary_accuracy": _ratio(self.prefix_correct, self.prefix_count),
            },
        }


def denormalize_domain_actions(
    actions: Any,
    *,
    action_indices: Sequence[int],
    action_scale: Sequence[float],
    action_clip: tuple[float, float],
    passthrough_indices: set[int] | frozenset[int],
    gripper_range: tuple[float, float],
) -> np.ndarray:
    """Vectorized equivalent of M0MobileNormalizer for one routed domain."""

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(action_indices):
        raise ValueError("domain action width does not match its global indices")
    result = np.empty_like(values)
    for local_index, global_index in enumerate(action_indices):
        if global_index in passthrough_indices:
            result[:, local_index] = np.clip(
                values[:, local_index], gripper_range[0], gripper_range[1]
            )
        else:
            result[:, local_index] = (
                np.clip(values[:, local_index], action_clip[0], action_clip[1])
                * float(action_scale[global_index])
            )
    return result


def select_stability_indices(
    annotations: Sequence[Mapping[str, Any]],
    phase_ids: Sequence[int],
    samples_per_phase: int,
) -> list[int]:
    """Choose a deterministic boundary/interior probe for every phase."""

    if samples_per_phase <= 0:
        return []
    selected: list[int] = []
    for phase_id in phase_ids:
        phase_indices = [
            index
            for index, row in enumerate(annotations)
            if int(row["phase_id"]) == int(phase_id)
        ]
        if len(phase_indices) < samples_per_phase:
            raise ValueError(
                f"phase {phase_id} has {len(phase_indices)} rows, fewer than "
                f"the requested stability sample count {samples_per_phase}"
            )
        boundary = sorted(
            (
                index
                for index in phase_indices
                if bool(annotations[index]["is_boundary_window"])
            ),
            key=lambda index: (
                abs(float(annotations[index]["seconds_to_boundary"])),
                int(annotations[index]["base_index"]),
            ),
        )
        boundary_count = min(len(boundary), samples_per_phase // 2)
        chosen = boundary[:boundary_count]
        interior = [
            index
            for index in phase_indices
            if not bool(annotations[index]["is_boundary_window"])
        ]
        remaining = samples_per_phase - len(chosen)
        if len(interior) < remaining:
            interior.extend(index for index in boundary[boundary_count:])
        positions = np.linspace(0, len(interior) - 1, num=remaining, dtype=int)
        chosen.extend(interior[int(position)] for position in positions)
        selected.extend(chosen)
    return selected


def summarize_stability(
    samples: Sequence[Mapping[str, Any]],
    *,
    phase_names: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase_name in phase_names:
        phase_samples = [row for row in samples if row["phase_name"] == phase_name]
        standard_deviations: list[float] = []
        pairwise_errors: list[float] = []
        navigation_flips = 0
        gripper_disagreements = 0
        for row in phase_samples:
            actions = np.asarray(row["predictions"], dtype=np.float64)
            valid_count = int(sum(bool(value) for value in row["action_valid_mask"]))
            valid = actions[:, :valid_count, :] if valid_count else actions[:, :0, :]
            if valid.size:
                standard_deviations.extend(valid.std(axis=0).reshape(-1).tolist())
                reference = valid[0]
                for seed_actions in valid[1:]:
                    pairwise_errors.append(
                        float(np.sqrt(np.square(seed_actions - reference).mean()))
                    )
            prefix = min(EXECUTED_PREFIX_STEPS, valid_count)
            if row["action_domain"] == "NAVIGATION" and prefix:
                integrals = actions[:, :prefix, 0].sum(axis=1)
                signs = np.where(integrals > 0.025, 1, np.where(integrals < -0.025, -1, 0))
                navigation_flips += int(len(set(int(value) for value in signs)) > 1)
            if row["action_domain"] == "MANIPULATION" and valid_count:
                gripper = actions[:, :valid_count, -1] >= 0.5
                gripper_disagreements += int(
                    np.any(gripper != gripper[0:1], axis=0).any()
                )
        array = np.asarray(standard_deviations, dtype=np.float64)
        result[phase_name] = {
            "samples": len(phase_samples),
            "valid_component_count": int(array.size),
            "mean_sampling_std": float(array.mean()) if array.size else None,
            "p95_sampling_std": float(np.quantile(array, 0.95)) if array.size else None,
            "max_sampling_std": float(array.max()) if array.size else None,
            "mean_rmse_to_first_seed": (
                float(np.mean(pairwise_errors)) if pairwise_errors else None
            ),
            "navigation_prefix_sign_flip_rate": (
                navigation_flips / len(phase_samples)
                if phase_samples and phase_samples[0]["action_domain"] == "NAVIGATION"
                else None
            ),
            "gripper_seed_disagreement_rate": (
                gripper_disagreements / len(phase_samples)
                if phase_samples and phase_samples[0]["action_domain"] == "MANIPULATION"
                else None
            ),
        }
    return result


def evaluate_standard_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = report["standard"]["thresholds"]
    gates: list[dict[str, Any]] = []

    def add(
        name: str,
        scope: str,
        observed: Any,
        relation: str,
        threshold: Any,
        passed: bool,
    ) -> None:
        gates.append(
            {
                "name": name,
                "scope": scope,
                "observed": observed,
                "relation": relation,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    add(
        "complete_split_coverage",
        "contract",
        report["evaluation"]["evaluated_rows"],
        "==",
        report["evaluation"]["dataset_rows"],
        report["evaluation"]["evaluated_rows"] == report["evaluation"]["dataset_rows"],
    )
    all_finite = all(
        value["normalized"]["all_output_values_finite"]
        for value in report["phase_metrics"].values()
    )
    add("all_sampled_actions_finite", "contract", all_finite, "is", True, all_finite)
    for phase, values in report["phase_metrics"].items():
        normalized = values["normalized"]["full_valid_chunk"]
        rmse = normalized["rmse"]
        skill = normalized["skill_vs_zero"]
        out_rate = normalized["out_of_contract_rate"]
        add(
            "phase_normalized_rmse",
            phase,
            rmse,
            "<=",
            thresholds["max_phase_normalized_rmse"],
            rmse is not None and rmse <= thresholds["max_phase_normalized_rmse"],
        )
        add(
            "phase_skill_vs_zero",
            phase,
            skill,
            ">",
            thresholds["min_phase_skill_vs_zero"],
            skill is not None and skill > thresholds["min_phase_skill_vs_zero"],
        )
        add(
            "normalized_out_of_contract_rate",
            phase,
            out_rate,
            "<=",
            thresholds["max_normalized_out_of_contract_rate"],
            out_rate is not None
            and out_rate <= thresholds["max_normalized_out_of_contract_rate"],
        )
        if values["action_domain"] == "NAVIGATION":
            direction = values["navigation_trajectory"]["executed_prefix"][
                "sign_agreement"
            ]
            add(
                "navigation_prefix_direction_accuracy",
                phase,
                direction,
                ">=",
                thresholds["min_navigation_prefix_direction_accuracy"],
                direction is not None
                and direction
                >= thresholds["min_navigation_prefix_direction_accuracy"],
            )
        else:
            gripper = values["gripper"]["full_valid_chunk"]["binary_accuracy"]
            add(
                "manipulation_gripper_accuracy",
                phase,
                gripper,
                ">=",
                thresholds["min_manipulation_gripper_accuracy"],
                gripper is not None
                and gripper >= thresholds["min_manipulation_gripper_accuracy"],
            )
    reverse = report["phase_metrics"]["NAV_TO_SOURCE"]["navigation_trajectory"][
        "predicted_reverse_rate"
    ]
    add(
        "nav_to_source_prefix_reverse_rate",
        "NAV_TO_SOURCE",
        reverse,
        "<=",
        thresholds["max_nav_to_source_prefix_reverse_rate"],
        reverse is not None
        and reverse <= thresholds["max_nav_to_source_prefix_reverse_rate"],
    )
    for phase, values in report["stability_metrics"].items():
        mean_std = values["mean_sampling_std"]
        add(
            "mean_sampling_std",
            phase,
            mean_std,
            "<=",
            thresholds["max_phase_mean_sampling_std"],
            mean_std is not None
            and mean_std <= thresholds["max_phase_mean_sampling_std"],
        )
    return gates


def render_markdown(report: Mapping[str, Any]) -> str:
    passed = report["verdict"]["passed"]
    lines = [
        "# Open-loop action quality report",
        "",
        f"Verdict: **{'PASS' if passed else 'FAIL'}** under `{report['standard']['profile']}`.",
        "",
        "This is sampled action inference on recorded expert observations. It is not a "
        "closed-loop simulator rollout and does not replay expert actions as predictions.",
        "",
        "## Identity",
        "",
        f"- Checkpoint: `{report['identity']['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report['identity'].get('checkpoint_sha256')}`",
        f"- Code commit: `{report['identity']['code_commit']}`",
        f"- Dataset manifest SHA-256: `{report['identity']['dataset_manifest_sha256']}`",
        f"- Split/rows: `{report['evaluation']['split']}` / "
        f"`{report['evaluation']['evaluated_rows']}`",
        "",
        "## Phase quality",
        "",
        "| Phase | Rows | Valid values | Norm RMSE | Norm MAE | Skill vs zero | "
        "Prefix RMSE | Out-of-contract |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase, values in report["phase_metrics"].items():
        full = values["normalized"]["full_valid_chunk"]
        prefix = values["normalized"]["executed_prefix"]
        lines.append(
            f"| {phase} | {values['normalized']['rows']} | {full['count']} | "
            f"{_format_metric(full['rmse'])} | {_format_metric(full['mae'])} | "
            f"{_format_metric(full['skill_vs_zero'])} | "
            f"{_format_metric(prefix['rmse'])} | "
            f"{_format_percent(full['out_of_contract_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Task-specific diagnostics",
            "",
            "| Phase | Diagnostic | Value |",
            "|---|---|---:|",
        ]
    )
    for phase, values in report["phase_metrics"].items():
        if values["action_domain"] == "NAVIGATION":
            nav = values["navigation_trajectory"]
            lines.append(
                f"| {phase} | 5-step direction accuracy | "
                f"{_format_percent(nav['executed_prefix']['sign_agreement'])} |"
            )
            lines.append(
                f"| {phase} | 5-step predicted reverse rows | "
                f"{_format_percent(nav['predicted_reverse_rate'])} |"
            )
            lines.append(
                f"| {phase} | negative-vx predicted steps | "
                f"{_format_percent(nav['predicted_negative_vx_step_rate'])} |"
            )
        else:
            grip = values["gripper"]
            lines.append(
                f"| {phase} | gripper binary accuracy (full valid chunk) | "
                f"{_format_percent(grip['full_valid_chunk']['binary_accuracy'])} |"
            )
            lines.append(
                f"| {phase} | gripper binary accuracy (first 5) | "
                f"{_format_percent(grip['executed_prefix']['binary_accuracy'])} |"
            )
    lines.extend(
        [
            "",
            "## Sampling stability",
            "",
            "| Phase | Samples | Mean std | P95 std | Max std | RMSE to seed 0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for phase, values in report["stability_metrics"].items():
        lines.append(
            f"| {phase} | {values['samples']} | "
            f"{_format_metric(values['mean_sampling_std'])} | "
            f"{_format_metric(values['p95_sampling_std'])} | "
            f"{_format_metric(values['max_sampling_std'])} | "
            f"{_format_metric(values['mean_rmse_to_first_seed'])} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Scope | Gate | Observed | Requirement | Result |",
            "|---|---|---:|---:|---|",
        ]
    )
    for gate in report["verdict"]["gates"]:
        lines.append(
            f"| {gate['scope']} | {gate['name']} | "
            f"{_format_metric(gate['observed'])} | {gate['relation']} "
            f"{_format_metric(gate['threshold'])} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `report.json`: full aggregate and per-dimension metrics.",
            "- `predictions.jsonl`: every sampled chunk, expert target, and valid mask.",
            "- `stability_predictions.jsonl`: repeated-seed actions for the stability subset.",
            "- `report.md`: this human-readable summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def _format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("stability seeds must contain at least two unique integers")
    return seeds


def _oracle_hidden(model: Any, examples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    inputs = dict(
        model._temporal_inputs(  # noqa: SLF001 - intentional oracle-route evaluation
            examples,
            include_solutions=True,
            solutions_override=[str(example["solution"]) for example in examples],
            supervise_solutions=False,
        )
    )
    outputs = model.qwen_vl_interface(
        **inputs,
        output_attentions=False,
        output_hidden_states=True,
        use_cache=False,
        logits_to_keep=1,
        return_dict=True,
    )
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        raise RuntimeError("Qwen processor did not return an attention mask")
    return outputs.hidden_states[-1], attention_mask


def _sample_oracle_heads(
    model: Any,
    examples: Sequence[Mapping[str, Any]],
    hidden: Any,
    attention_mask: Any,
    *,
    seed: int,
    torch_module: Any,
    action_domain_type: Any,
) -> list[np.ndarray]:
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    actions: list[np.ndarray | None] = [None] * len(examples)
    for domain, action_model in (
        (action_domain_type.NAVIGATION, model.navigation_model),
        (action_domain_type.MANIPULATION, model.manipulation_model),
    ):
        indices = [
            index
            for index, example in enumerate(examples)
            if int(example["action_domain_id"]) == int(domain)
        ]
        if not indices:
            continue
        index_tensor = torch_module.as_tensor(indices, device=hidden.device)
        device = next(action_model.parameters()).device
        dtype = next(action_model.parameters()).dtype
        state = torch_module.as_tensor(
            [examples[index]["state"] for index in indices],
            device=device,
            dtype=dtype,
        )
        autocast = (
            torch_module.autocast(device_type=device.type, dtype=dtype)
            if device.type == "cuda" and dtype in {torch_module.float16, torch_module.bfloat16}
            else contextlib.nullcontext()
        )
        with autocast:
            sampled = action_model.sample(
                hidden.index_select(0, index_tensor).to(device=device, dtype=dtype),
                state,
                encoder_attention_mask=attention_mask.index_select(0, index_tensor).to(device),
                action_dimension_mask=torch_module.ones(
                    (len(indices), action_model.config.action_dim),
                    device=device,
                    dtype=torch_module.bool,
                ),
            )
        for index, value in zip(indices, sampled.float().cpu().numpy(), strict=True):
            actions[index] = np.asarray(value, dtype=np.float64)
    if any(value is None for value in actions):
        raise RuntimeError("oracle dispatcher failed to assign one expert per example")
    return [value for value in actions if value is not None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--hierarchy-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--initial-action-checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-rows-per-phase",
        type=int,
        help="smoke-only cap; omit for the standard complete-split evaluation",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--stability-samples-per-phase", type=int, default=16)
    parser.add_argument(
        "--stability-seeds",
        type=_parse_seeds,
        default=_parse_seeds("20260819,20260820,20260821,20260822"),
    )
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.max_rows_per_phase is not None and args.max_rows_per_phase <= 0:
        raise ValueError("max rows per phase must be positive when supplied")
    if args.stability_samples_per_phase <= 0:
        raise ValueError("stability samples per phase must be positive")
    if args.checkpoint_sha256 is not None and (
        len(args.checkpoint_sha256) != 64
        or any(value not in "0123456789abcdef" for value in args.checkpoint_sha256.lower())
    ):
        raise ValueError("checkpoint SHA-256 must contain 64 hexadecimal characters")

    repo = args.repo.expanduser().resolve()
    hierarchy_root = args.hierarchy_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {output_dir}")
    output_dir.mkdir(parents=True)
    sys.path[:0] = [str(repo / "src"), str(repo / "scripts")]

    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader, Subset

    from conveyor_bench.conveyorvla.config import load_m0_mobile_config
    from conveyor_bench.conveyorvla.hierarchical_data import (
        ConveyorVLAAL0HierarchicalDataset,
    )
    from conveyor_bench.conveyorvla.subtasks import (
        MANIPULATION_ACTION_INDICES,
        NAVIGATION_ACTION_INDICES,
        PHASE_ORDER,
        ActionDomain,
        action_domain,
    )
    from conveyor_bench.conveyorvla.temporal import (
        build_temporal_policy_config,
        load_temporal_config,
    )
    from train_hierarchical import _build_model

    started = time.monotonic()
    accelerator = Accelerator(mixed_precision="bf16")
    config = build_temporal_policy_config(
        load_m0_mobile_config(repo / "configs" / "model.json"),
        load_temporal_config(repo / "configs" / "temporal.json"),
    )
    model, _transfer = _build_model(
        config,
        args.model_root.expanduser().resolve(),
        SimpleNamespace(
            attention_implementation=args.attention_implementation,
            initial_action_checkpoint=args.initial_action_checkpoint.expanduser().resolve(),
            repeated_diffusion_steps=1,
        ),
    )
    dataset = ConveyorVLAAL0HierarchicalDataset(
        hierarchy_root, config, split=args.split, component="joint"
    )
    model = accelerator.prepare(model)
    accelerator.load_state(checkpoint)
    model.eval()
    unwrapped = accelerator.unwrap_model(model)

    normalizer = dataset.normalizer
    action_rate_hz = float(config["data"]["action_rate_hz"])
    phase_names = [phase.name for phase in PHASE_ORDER]
    phase_domains = {phase.name: action_domain(phase) for phase in PHASE_ORDER}
    domain_indices = {
        ActionDomain.NAVIGATION: tuple(NAVIGATION_ACTION_INDICES),
        ActionDomain.MANIPULATION: tuple(MANIPULATION_ACTION_INDICES),
    }
    domain_names = {
        ActionDomain.NAVIGATION: ("vx", "wz"),
        ActionDomain.MANIPULATION: (
            "tcp_dx",
            "tcp_dy",
            "tcp_dz",
            "tcp_drx",
            "tcp_dry",
            "tcp_drz",
            "gripper_open_fraction",
        ),
    }

    def make_group(domain: Any, *, physical: bool) -> ActionGroupStats:
        indices = domain_indices[domain]
        if physical:
            bounds = tuple(
                normalizer.gripper_range
                if index in normalizer.passthrough_indices
                else (
                    normalizer.action_clip[0] * normalizer.action_scale[index],
                    normalizer.action_clip[1] * normalizer.action_scale[index],
                )
                for index in indices
            )
            deadbands = tuple(
                None
                if index in normalizer.passthrough_indices
                else SIGN_DEADBAND_NORMALIZED * normalizer.action_scale[index]
                for index in indices
            )
        else:
            bounds = tuple(
                normalizer.gripper_range
                if index in normalizer.passthrough_indices
                else normalizer.action_clip
                for index in indices
            )
            deadbands = tuple(
                None
                if index in normalizer.passthrough_indices
                else SIGN_DEADBAND_NORMALIZED
                for index in indices
            )
        return ActionGroupStats(
            domain_names[domain], bounds=bounds, sign_deadbands=deadbands
        )

    normalized_groups = {
        phase: make_group(domain, physical=False)
        for phase, domain in phase_domains.items()
    }
    physical_groups = {
        phase: make_group(domain, physical=True)
        for phase, domain in phase_domains.items()
    }
    boundary_groups = {
        phase: {
            "boundary_window": make_group(domain, physical=False),
            "interior": make_group(domain, physical=False),
        }
        for phase, domain in phase_domains.items()
    }
    navigation_stats = {
        phase: NavigationTrajectoryStats(action_rate_hz)
        for phase, domain in phase_domains.items()
        if domain is ActionDomain.NAVIGATION
    }
    gripper_stats = {
        phase: GripperStats()
        for phase, domain in phase_domains.items()
        if domain is ActionDomain.MANIPULATION
    }

    prediction_path = output_dir / "predictions.jsonl"
    prediction_temporary = prediction_path.with_name(f".{prediction_path.name}.tmp")
    evaluated_rows = 0
    batch_index = 0
    with prediction_temporary.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for phase in PHASE_ORDER:
            indices = [
                index
                for index, row in enumerate(dataset.annotations)
                if int(row["phase_id"]) == int(phase)
            ]
            if args.max_rows_per_phase is not None:
                indices = indices[: args.max_rows_per_phase]
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=list,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
            )
            loader = accelerator.prepare(loader)
            phase_rows = 0
            for examples in loader:
                hidden, attention_mask = _oracle_hidden(unwrapped, examples)
                prediction_seed = args.seed + batch_index
                predictions = _sample_oracle_heads(
                    unwrapped,
                    examples,
                    hidden,
                    attention_mask,
                    seed=prediction_seed,
                    torch_module=torch,
                    action_domain_type=ActionDomain,
                )
                for example, predicted in zip(examples, predictions, strict=True):
                    phase_name = str(example["phase_name"])
                    domain = phase_domains[phase_name]
                    target = np.asarray(example["action"], dtype=np.float64)
                    valid_mask = tuple(bool(value) for value in example["action_valid_mask"])
                    if predicted.shape != target.shape:
                        raise RuntimeError(
                            f"sampled/target shape mismatch for {example['sample_id']}: "
                            f"{predicted.shape}/{target.shape}"
                        )
                    indices_for_domain = domain_indices[domain]
                    predicted_physical = denormalize_domain_actions(
                        predicted,
                        action_indices=indices_for_domain,
                        action_scale=normalizer.action_scale,
                        action_clip=normalizer.action_clip,
                        passthrough_indices=normalizer.passthrough_indices,
                        gripper_range=normalizer.gripper_range,
                    )
                    target_physical = denormalize_domain_actions(
                        target,
                        action_indices=indices_for_domain,
                        action_scale=normalizer.action_scale,
                        action_clip=normalizer.action_clip,
                        passthrough_indices=normalizer.passthrough_indices,
                        gripper_range=normalizer.gripper_range,
                    )
                    normalized_groups[phase_name].update(predicted, target, valid_mask)
                    physical_groups[phase_name].update(
                        predicted_physical, target_physical, valid_mask
                    )
                    window = "boundary_window" if example["is_boundary_window"] else "interior"
                    boundary_groups[phase_name][window].update(
                        predicted, target, valid_mask
                    )
                    if domain is ActionDomain.NAVIGATION:
                        navigation_stats[phase_name].update(
                            predicted_physical, target_physical, valid_mask
                        )
                    else:
                        gripper_stats[phase_name].update(
                            predicted_physical, target_physical, valid_mask
                        )
                    trace = {
                        "sample_id": str(example["sample_id"]),
                        "base_index": int(example["base_index"]),
                        "phase_name": phase_name,
                        "action_domain": domain.name,
                        "is_boundary_window": bool(example["is_boundary_window"]),
                        "boundary_transition": example["boundary_transition"],
                        "seconds_to_boundary": float(example["seconds_to_boundary"]),
                        "prediction_seed": prediction_seed,
                        "action_valid_mask": list(valid_mask),
                        "predicted_normalized": predicted.tolist(),
                        "target_normalized": target.tolist(),
                        "predicted_physical": predicted_physical.tolist(),
                        "target_physical": target_physical.tolist(),
                    }
                    stream.write(json.dumps(trace, separators=(",", ":")) + "\n")
                evaluated_rows += len(examples)
                phase_rows += len(examples)
                batch_index += 1
                if batch_index % 20 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "full_split",
                                "phase": phase.name,
                                "phase_rows": phase_rows,
                                "total_rows": evaluated_rows,
                                "dataset_rows": len(dataset),
                            }
                        ),
                        flush=True,
                    )
    os.replace(prediction_temporary, prediction_path)

    stability_indices = select_stability_indices(
        dataset.annotations,
        [int(phase) for phase in PHASE_ORDER],
        args.stability_samples_per_phase,
    )
    stability_loader = DataLoader(
        Subset(dataset, stability_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    stability_loader = accelerator.prepare(stability_loader)
    stability_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for stability_batch, examples in enumerate(stability_loader):
            hidden, attention_mask = _oracle_hidden(unwrapped, examples)
            sampled_by_seed = [
                _sample_oracle_heads(
                    unwrapped,
                    examples,
                    hidden,
                    attention_mask,
                    seed=seed,
                    torch_module=torch,
                    action_domain_type=ActionDomain,
                )
                for seed in args.stability_seeds
            ]
            for example_index, example in enumerate(examples):
                stability_rows.append(
                    {
                        "sample_id": str(example["sample_id"]),
                        "phase_name": str(example["phase_name"]),
                        "action_domain": str(example["action_domain_name"]),
                        "is_boundary_window": bool(example["is_boundary_window"]),
                        "action_valid_mask": [
                            bool(value) for value in example["action_valid_mask"]
                        ],
                        "seeds": list(args.stability_seeds),
                        "predictions": [
                            sampled[example_index].tolist()
                            for sampled in sampled_by_seed
                        ],
                    }
                )
            print(
                json.dumps(
                    {
                        "stage": "stability",
                        "batch": stability_batch + 1,
                        "rows": len(stability_rows),
                        "total": len(stability_indices),
                    }
                ),
                flush=True,
            )
    stability_path = output_dir / "stability_predictions.jsonl"
    stability_temporary = stability_path.with_name(f".{stability_path.name}.tmp")
    with stability_temporary.open("w", encoding="utf-8") as stream:
        for row in stability_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(stability_temporary, stability_path)

    phase_metrics: dict[str, Any] = {}
    for phase_name in phase_names:
        domain = phase_domains[phase_name]
        phase_metrics[phase_name] = {
            "action_domain": domain.name,
            "normalized": normalized_groups[phase_name].finalize(),
            "physical": physical_groups[phase_name].finalize(),
            "boundary_comparison": {
                name: group.finalize()
                for name, group in boundary_groups[phase_name].items()
            },
        }
        if domain is ActionDomain.NAVIGATION:
            phase_metrics[phase_name]["navigation_trajectory"] = navigation_stats[
                phase_name
            ].finalize()
        else:
            phase_metrics[phase_name]["gripper"] = gripper_stats[phase_name].finalize()

    manifest_path = hierarchy_root / "manifest.json"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "standard": {
            "profile": STANDARD_PROFILE,
            "thresholds": dict(STANDARD_THRESHOLDS),
            "route_mode": "oracle annotated phase for action-head isolation",
            "executed_prefix_steps": EXECUTED_PREFIX_STEPS,
            "sign_deadband_normalized": SIGN_DEADBAND_NORMALIZED,
        },
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "code_repo": str(repo),
            "code_commit": _git_commit(repo),
            "hierarchy_root": str(hierarchy_root),
            "dataset_manifest_sha256": _sha256(manifest_path),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_device_count_visible": torch.cuda.device_count(),
            "cuda_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "attention_implementation": args.attention_implementation,
            "elapsed_seconds": time.monotonic() - started,
        },
        "evaluation": {
            "split": args.split,
            "selection": (
                "all rows"
                if args.max_rows_per_phase is None
                else f"smoke cap: first {args.max_rows_per_phase} rows per phase"
            ),
            "dataset_rows": len(dataset),
            "evaluated_rows": evaluated_rows,
            "full_split_seed_base": args.seed,
            "stability_samples_per_phase": args.stability_samples_per_phase,
            "stability_seeds": list(args.stability_seeds),
            "action_rate_hz": action_rate_hz,
        },
        "phase_metrics": phase_metrics,
        "stability_metrics": summarize_stability(
            stability_rows, phase_names=phase_names
        ),
    }
    gates = evaluate_standard_gates(report)
    report["verdict"] = {
        "passed": all(gate["passed"] for gate in gates),
        "passed_gates": sum(gate["passed"] for gate in gates),
        "total_gates": len(gates),
        "failed_gates": [gate for gate in gates if not gate["passed"]],
        "gates": gates,
    }
    _write_json(output_dir / "report.json", report)
    _write_text(output_dir / "report.md", render_markdown(report))
    print(json.dumps(report["verdict"], indent=2, sort_keys=True), flush=True)
    if args.fail_on_gate and not report["verdict"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
