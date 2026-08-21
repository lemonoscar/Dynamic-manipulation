#!/usr/bin/env python3
"""Compare paired Waypoint-v2 S1/S4 pilots at equal steps and GPU-hours."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "conveyorvla-waypoint-v2-paired-pilot-report-v1"
OPEN_LOOP_SCHEMA = "conveyorvla-waypoint-v2-open-loop-report-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-resolved", required=True, type=Path)
    parser.add_argument("--s4-resolved", required=True, type=Path)
    parser.add_argument("--s1-events", required=True, type=Path)
    parser.add_argument("--s4-events", required=True, type=Path)
    parser.add_argument("--s1-eval", required=True, action="append", type=Path)
    parser.add_argument("--s4-eval", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-window", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_pilots(
        _read_json(args.s1_resolved),
        _read_json(args.s4_resolved),
        _read_events(args.s1_events),
        _read_events(args.s4_events),
        [_read_json(path) for path in args.s1_eval],
        [_read_json(path) for path in args.s4_eval],
        cv_window=args.cv_window,
    )
    _write_new_json(args.output.expanduser().resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "step_matched": report["step_matched"]["checkpoint_steps"],
                "gpu_hour_matched": report["gpu_hour_matched"]["checkpoint_steps"],
                "s4_pilot_candidate": report["promotion_evidence"][
                    "s4_pilot_candidate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def compare_pilots(
    s1_resolved: Mapping[str, Any],
    s4_resolved: Mapping[str, Any],
    s1_events: Sequence[Mapping[str, Any]],
    s4_events: Sequence[Mapping[str, Any]],
    s1_evaluations: Sequence[Mapping[str, Any]],
    s4_evaluations: Sequence[Mapping[str, Any]],
    *,
    cv_window: int,
) -> dict[str, Any]:
    if cv_window <= 1:
        raise ValueError("pilot CV window must exceed one step")
    identity = _paired_identity(s1_resolved, s4_resolved)
    s1_steps = _train_steps(s1_events)
    s4_steps = _train_steps(s4_events)
    s1_reports = _evaluation_index(s1_evaluations)
    s4_reports = _evaluation_index(s4_evaluations)
    _validate_evaluation_pairing(s1_reports, s4_reports, identity)

    common_steps = sorted(set(s1_reports).intersection(s4_reports))
    if not common_steps:
        raise ValueError("S1/S4 pilots have no step-matched evaluation checkpoint")
    step = common_steps[-1]
    step_matched = _comparison(
        s1_reports[step],
        s4_reports[step],
        s1_steps,
        s4_steps,
        cv_window=cv_window,
    )

    s1_compute, s4_compute = _compute_matched_reports(
        s1_reports, s4_reports, s1_steps, s4_steps
    )
    gpu_hour_matched = _comparison(
        s1_compute,
        s4_compute,
        s1_steps,
        s4_steps,
        cv_window=cv_window,
    )
    promotion = _promotion_evidence(step_matched, gpu_hour_matched)
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "pass",
        "paired_identity": identity,
        "step_matched": step_matched,
        "gpu_hour_matched": gpu_hour_matched,
        "promotion_evidence": promotion,
        "interpretation": (
            "S4 may advance only when action quality improves without an obvious "
            "regression; lower gradient CV alone is insufficient. A second paired "
            "training seed is still required for a final S4 conclusion."
        ),
    }


def _paired_identity(
    s1: Mapping[str, Any], s4: Mapping[str, Any]
) -> dict[str, Any]:
    config1 = copy.deepcopy(_mapping(s1.get("resolved_policy_config"), "S1 config"))
    config4 = copy.deepcopy(_mapping(s4.get("resolved_policy_config"), "S4 config"))
    repeat1 = int(_mapping(config1.get("loss"), "S1 loss")["repeated_diffusion_steps"])
    repeat4 = int(_mapping(config4.get("loss"), "S4 loss")["repeated_diffusion_steps"])
    if (repeat1, repeat4) != (1, 4):
        raise ValueError(f"paired pilots must be S1/S4, got S{repeat1}/S{repeat4}")
    config1["loss"]["repeated_diffusion_steps"] = "PAIRED"
    config4["loss"]["repeated_diffusion_steps"] = "PAIRED"
    if config1 != config4:
        raise ValueError("S1/S4 resolved configs differ beyond repeated FM draws")

    arguments1 = _mapping(s1.get("arguments"), "S1 arguments")
    arguments4 = _mapping(s4.get("arguments"), "S4 arguments")
    fields = (
        "dataset_manifest_sha256",
        "normalization_sha256",
        "qwen_base",
        "source_git",
        "world_size",
        "batch_size_per_process",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "train_rows",
        "training_subset_indices",
        "max_steps",
        "warmup_steps",
    )
    mismatches = [field for field in fields if s1.get(field) != s4.get(field)]
    argument_fields = ("seed", "attention_implementation", "limit_train_episodes")
    mismatches.extend(
        f"arguments.{field}"
        for field in argument_fields
        if arguments1.get(field) != arguments4.get(field)
    )
    initialization1 = _mapping(s1.get("initialization"), "S1 initialization")
    initialization4 = _mapping(s4.get("initialization"), "S4 initialization")
    for field in ("qwen", "navigation_head", "manipulation_head", "legacy_checkpoint_loaded"):
        if initialization1.get(field) != initialization4.get(field):
            mismatches.append(f"initialization.{field}")
    if mismatches:
        raise ValueError("paired pilot identity mismatch: " + ", ".join(mismatches))
    return {
        "dataset_manifest_sha256": s1["dataset_manifest_sha256"],
        "normalization_sha256": s1["normalization_sha256"],
        "source_git": s1["source_git"],
        "qwen_base": s1["qwen_base"],
        "seed": arguments1["seed"],
        "world_size": s1["world_size"],
        "batch_size_per_process": s1["batch_size_per_process"],
        "gradient_accumulation_steps": s1["gradient_accumulation_steps"],
        "effective_batch_size": s1["effective_batch_size"],
        "max_steps": s1["max_steps"],
        "warmup_steps": s1["warmup_steps"],
        "only_config_difference": "loss.repeated_diffusion_steps: 1 -> 4",
        "action_model_num_inference_timesteps": config1["action_model"][
            "num_inference_timesteps"
        ],
    }


def _train_steps(events: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    if any(event.get("event") == "failed" for event in events):
        raise ValueError("paired pilot contains a failed event")
    train = [event for event in events if event.get("event") == "train_step"]
    steps = [int(event["step"]) for event in train]
    if not steps or steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("paired pilot train_step events are not consecutive")
    cumulative = 0.0
    result = {}
    required = (
        "navigation_loss",
        "manipulation_loss",
        "vlm_gradient_norm",
        "navigation_gradient_norm",
        "manipulation_gradient_norm",
        "optimizer_step_time_s",
        "samples_per_second",
        "gpu_hours_per_step",
        "peak_reserved_memory_mib",
    )
    for event in train:
        for field in required:
            value = event.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"step {event.get('step')} has invalid {field}")
        if event.get("valid_optimizer_step") is not True:
            raise ValueError(f"step {event.get('step')} is not a valid optimizer step")
        cumulative += float(event["gpu_hours_per_step"])
        result[int(event["step"])] = {**event, "cumulative_gpu_hours": cumulative}
    return result


def _evaluation_index(
    reports: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    if not reports:
        raise ValueError("paired pilot needs open-loop evaluation reports")
    result = {}
    for report in reports:
        if report.get("schema_version") != OPEN_LOOP_SCHEMA:
            raise ValueError("paired pilot received a non-v2 open-loop report")
        step = int(_mapping(report.get("identity"), "evaluation identity")["checkpoint_step"])
        if step in result:
            raise ValueError(f"duplicate evaluation report for checkpoint step {step}")
        result[step] = report
    return result


def _validate_evaluation_pairing(
    s1: Mapping[int, Mapping[str, Any]],
    s4: Mapping[int, Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> None:
    reports = list(s1.values()) + list(s4.values())
    first = reports[0]
    first_identity = _mapping(first.get("identity"), "evaluation identity")
    first_selection = _mapping(first.get("selection"), "evaluation selection")
    first_bank = _mapping(first.get("fixed_validation_bank"), "fixed bank")
    for report in reports:
        current_identity = _mapping(report.get("identity"), "evaluation identity")
        current_selection = _mapping(report.get("selection"), "evaluation selection")
        current_bank = _mapping(report.get("fixed_validation_bank"), "fixed bank")
        if current_identity.get("dataset_manifest_sha256") != identity[
            "dataset_manifest_sha256"
        ]:
            raise ValueError("evaluation dataset differs from paired training data")
        if current_identity.get("selected_indices") != first_identity.get(
            "selected_indices"
        ) or current_selection.get("indices") != first_selection.get("indices"):
            raise ValueError("paired evaluations use different validation rows")
        if current_bank.get("manifest_sha256") != first_bank.get("manifest_sha256"):
            raise ValueError("paired evaluations use different fixed noise/time banks")


def _compute_matched_reports(
    s1_reports: Mapping[int, Mapping[str, Any]],
    s4_reports: Mapping[int, Mapping[str, Any]],
    s1_steps: Mapping[int, Mapping[str, Any]],
    s4_steps: Mapping[int, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    candidates = []
    for left, right in itertools.product(s1_reports.values(), s4_reports.values()):
        left_step = int(left["identity"]["checkpoint_step"])
        right_step = int(right["identity"]["checkpoint_step"])
        if left_step not in s1_steps or right_step not in s4_steps:
            raise ValueError("evaluation checkpoint exceeds recorded train events")
        left_hours = float(s1_steps[left_step]["cumulative_gpu_hours"])
        right_hours = float(s4_steps[right_step]["cumulative_gpu_hours"])
        relative_gap = abs(left_hours - right_hours) / max(left_hours, right_hours)
        candidates.append(
            (relative_gap, -min(left_hours, right_hours), left_step, right_step, left, right)
        )
    _gap, _budget, _left_step, _right_step, left, right = min(candidates)
    return left, right


def _comparison(
    s1_report: Mapping[str, Any],
    s4_report: Mapping[str, Any],
    s1_steps: Mapping[int, Mapping[str, Any]],
    s4_steps: Mapping[int, Mapping[str, Any]],
    *,
    cv_window: int,
) -> dict[str, Any]:
    step1 = int(s1_report["identity"]["checkpoint_step"])
    step4 = int(s4_report["identity"]["checkpoint_step"])
    summary1 = {
        "training": _training_summary(s1_steps, step1, cv_window),
        "action_quality": _action_quality(s1_report),
    }
    summary4 = {
        "training": _training_summary(s4_steps, step4, cv_window),
        "action_quality": _action_quality(s4_report),
    }
    hours1 = float(s1_steps[step1]["cumulative_gpu_hours"])
    hours4 = float(s4_steps[step4]["cumulative_gpu_hours"])
    return {
        "checkpoint_steps": {"s1": step1, "s4": step4},
        "gpu_hours": {"s1": hours1, "s4": hours4},
        "gpu_hour_relative_gap": abs(hours1 - hours4) / max(hours1, hours4),
        "s1": summary1,
        "s4": summary4,
        "s4_minus_s1": {
            key: _delta(summary1["action_quality"].get(key), value)
            for key, value in summary4["action_quality"].items()
        },
    }


def _training_summary(
    steps: Mapping[int, Mapping[str, Any]], checkpoint_step: int, window: int
) -> dict[str, Any]:
    if checkpoint_step not in steps:
        raise ValueError(f"checkpoint step {checkpoint_step} has no training event")
    selected = [
        event
        for step, event in sorted(steps.items())
        if step <= checkpoint_step
    ][-window:]
    cv_fields = (
        "navigation_loss",
        "manipulation_loss",
        "vlm_gradient_norm",
        "navigation_gradient_norm",
        "manipulation_gradient_norm",
        "auxiliary_gradient_norm",
    )
    return {
        "window_steps": [int(selected[0]["step"]), int(selected[-1]["step"])],
        "cumulative_gpu_hours": steps[checkpoint_step]["cumulative_gpu_hours"],
        "mean_optimizer_step_time_s": _mean_field(selected, "optimizer_step_time_s"),
        "mean_samples_per_second": _mean_field(selected, "samples_per_second"),
        "max_peak_reserved_memory_mib": max(
            float(event["peak_reserved_memory_mib"]) for event in selected
        ),
        "rolling_cv": {
            field: _cv(
                [float(event[field]) for event in selected if field in event]
            )
            for field in cv_fields
        },
    }


def _action_quality(report: Mapping[str, Any]) -> dict[str, float | None]:
    action = _mapping(report.get("oracle_prefix_action"), "action report")
    cross = _mapping(action.get("cross_seed"), "cross-seed action report")
    gate = _mapping(report.get("gate"), "evaluation gate")
    legacy = _mapping(gate.get("legacy_action_route_gate"), "legacy gate")
    quality = _mapping(legacy.get("quality_metrics"), "quality metrics")
    diagnostics = _mapping(report.get("action_diagnostics"), "action diagnostics")
    nav = _mapping(diagnostics.get("navigation"), "navigation diagnostics")
    arm = _mapping(diagnostics.get("manipulation"), "manipulation diagnostics")
    bank = _mapping(report.get("fixed_validation_bank"), "fixed bank")
    return {
        "navigation_ade_mean_m": _optional_float(cross.get("navigation_ade_mean_m")),
        "arm_position_mean_m": _optional_float(cross.get("arm_position_mean_m")),
        "navigation_direction_accuracy": _optional_float(nav.get("direction_accuracy")),
        "arm_orientation_mean_rad": _optional_float(
            quality.get("arm_orientation_mean_rad")
        ),
        "navigation_terminal_hold_suffix_mae": _optional_float(
            nav.get("terminal_hold_suffix_mae")
        ),
        "manipulation_terminal_hold_suffix_mae": _optional_float(
            arm.get("terminal_hold_suffix_mae")
        ),
        "fixed_bank_navigation_loss": _optional_float(
            _mapping(bank.get("navigation"), "navigation bank").get("mean_loss")
        ),
        "fixed_bank_manipulation_loss": _optional_float(
            _mapping(bank.get("manipulation"), "manipulation bank").get("mean_loss")
        ),
    }


def _promotion_evidence(
    step_matched: Mapping[str, Any], gpu_matched: Mapping[str, Any]
) -> dict[str, Any]:
    def improved(comparison: Mapping[str, Any], key: str, *, higher: bool = False) -> bool:
        left = comparison["s1"]["action_quality"].get(key)
        right = comparison["s4"]["action_quality"].get(key)
        if left is None or right is None:
            return False
        return right >= left * 1.02 if higher else right <= left * 0.98

    def no_regression(comparison: Mapping[str, Any]) -> bool:
        for key in ("navigation_ade_mean_m", "arm_position_mean_m"):
            left = comparison["s1"]["action_quality"].get(key)
            right = comparison["s4"]["action_quality"].get(key)
            if left is None or right is None or right > left * 1.05:
                return False
        left = comparison["s1"]["action_quality"].get(
            "navigation_direction_accuracy"
        )
        right = comparison["s4"]["action_quality"].get(
            "navigation_direction_accuracy"
        )
        return left is not None and right is not None and right >= left - 0.02

    evidence = {}
    for name, comparison in (
        ("step_matched", step_matched),
        ("gpu_hour_matched", gpu_matched),
    ):
        evidence[name] = {
            "action_improvement": any(
                (
                    improved(comparison, "navigation_ade_mean_m"),
                    improved(comparison, "arm_position_mean_m"),
                    improved(
                        comparison,
                        "navigation_direction_accuracy",
                        higher=True,
                    ),
                )
            ),
            "no_obvious_regression": no_regression(comparison),
        }
    candidate = all(
        item["action_improvement"] and item["no_obvious_regression"]
        for item in evidence.values()
    )
    return {
        **evidence,
        "s4_pilot_candidate": candidate,
        "second_seed_required": True,
        "gradient_smoothing_alone_can_promote": False,
    }


def _delta(left: Any, right: Any) -> dict[str, float | None]:
    if left is None or right is None:
        return {"absolute": None, "relative_to_s1": None}
    absolute = float(right) - float(left)
    relative = None if float(left) == 0.0 else absolute / abs(float(left))
    return {"absolute": absolute, "relative_to_s1": relative}


def _cv(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return None if mean == 0.0 else statistics.pstdev(values) / abs(mean)


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("pilot evaluation contains a non-finite metric")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def _read_events(path: Path) -> list[Mapping[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        return [
            _mapping(json.loads(line), f"{path} event")
            for line in stream
            if line.strip()
        ]


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"pilot comparison output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
