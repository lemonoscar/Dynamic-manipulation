#!/usr/bin/env python3
"""Evaluate route and oracle-prefix waypoint quality from a ZeRO checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import check_waypoint_checkpoint as checkpoint_gate  # noqa: E402
from scripts import train_waypoint as training  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint import (  # noqa: E402
    ACTION_HORIZON,
    ArmTargetSafety,
    NavWaypointSafety,
    WaypointRoute,
    wrap_to_pi,
)
from conveyor_bench.conveyorvla.waypoint_data import (  # noqa: E402
    ConveyorVLAWaypointDataset,
)


FORMAT_RECOVER_REASONS = {
    "missing_end_subtask",
    "subtask_too_long",
    "invalid_subtask_tokens",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--diffusion-seeds", default="17,29,43,71")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest, resolved, dataset_root = checkpoint_gate._validate_binding(checkpoint)
    run_args = _mapping(resolved["arguments"], "resolved arguments")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(resolved["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    if args.rows <= 0 or args.batch_size <= 0:
        raise M0MobileError("open-loop rows and batch size must be positive")
    group_size = accelerator.num_processes * args.batch_size
    if args.rows % group_size:
        raise M0MobileError(
            "open-loop rows must be divisible by world_size times batch_size"
        )
    seeds = _parse_seeds(args.diffusion_seeds)
    set_seed(int(run_args["seed"]), device_specific=True)
    train_dataset = ConveyorVLAWaypointDataset(dataset_root, split="train")
    train_indices = resolved.get("training_subset_indices")
    loader_dataset = (
        train_dataset
        if train_indices is None
        else Subset(train_dataset, [int(value) for value in train_indices])
    )
    loader_routes = (
        train_dataset.routes
        if train_indices is None
        else [train_dataset.routes[int(index)] for index in train_indices]
    )
    loader_boundaries = (
        train_dataset.boundaries
        if train_indices is None
        else [train_dataset.boundaries[int(index)] for index in train_indices]
    )
    sampler = training.DomainBalancedSampler(
        loader_routes,
        training._row_sample_weights(loader_routes, loader_boundaries),
        batch_size=int(resolved["batch_size_per_process"]),
        seed=int(run_args["seed"]),
    )
    loader = DataLoader(
        loader_dataset,
        batch_size=int(resolved["batch_size_per_process"]),
        sampler=sampler,
        num_workers=0,
        collate_fn=list,
        pin_memory=True,
        drop_last=True,
    )
    config = training._load_config(Path(str(resolved["config"])))
    model, token_ids = training._build_model(
        config,
        Path(str(resolved["model_root"])),
        train_dataset,
        str(run_args["attention_implementation"]),
    )
    if token_ids != manifest["special_token_ids"]:
        raise M0MobileError("open-loop processor token IDs differ from the checkpoint")
    optimizer, _groups = training._optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        training.common._schedule(
            int(resolved["max_steps"]),
            int(resolved["warmup_steps"]),
        ),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        loader,
        scheduler,
    )
    del loader
    accelerator.load_state(checkpoint)
    model.eval()
    policy = accelerator.unwrap_model(model)
    evaluation = ConveyorVLAWaypointDataset(dataset_root, split=args.split)
    selected = training._balanced_subset_indices(evaluation, args.rows)
    if selected is None:
        raise M0MobileError("open-loop balanced selection unexpectedly returned all rows")
    local_indices = selected[accelerator.process_index :: accelerator.num_processes]
    if len(local_indices) % args.batch_size:
        raise M0MobileError("open-loop rank shard is not batch aligned")
    local_route_rows: list[dict[str, Any]] = []
    local_seed_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for offset in range(0, len(local_indices), args.batch_size):
        batch_indices = local_indices[offset : offset + args.batch_size]
        examples = [evaluation[index] for index in batch_indices]
        predictions = policy.predict(examples)
        for index, example, prediction in zip(
            batch_indices,
            examples,
            predictions,
            strict=True,
        ):
            decision = prediction.decision
            local_route_rows.append(
                {
                    "index": index,
                    "sample_id": example["sample_id"],
                    "target": example["route"],
                    "predicted": (
                        decision.route.value
                        if decision.valid and decision.route is not None
                        else "RECOVER"
                    ),
                    "valid": decision.valid,
                    "recover_reason": decision.recover_reason,
                    "format_invalid": decision.recover_reason in FORMAT_RECOVER_REASONS,
                    "confidence": decision.route_confidence,
                    "subtask": decision.subtask_text,
                    "assistant_prefix": decision.assistant_prefix,
                }
            )
        for seed in seeds:
            batch_seed = seed + offset * 1009
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            actions = policy.predict_oracle_actions(examples)
            for index, example, action in zip(
                batch_indices,
                examples,
                actions,
                strict=True,
            ):
                local_seed_rows[seed].append(
                    _action_row(evaluation, index, example, action, seed)
                )
    route_rows = _gather_rows(local_route_rows)
    seed_rows = {
        seed: _gather_rows(local_seed_rows[seed])
        for seed in seeds
    }
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        report = _report(
            checkpoint,
            manifest,
            resolved,
            args.split,
            selected,
            route_rows,
            seed_rows,
        )
        training.common._write_json_atomic(args.report.expanduser().resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    return 0


def _action_row(
    dataset: ConveyorVLAWaypointDataset,
    index: int,
    example: Mapping[str, Any],
    normalized_action: Sequence[Sequence[float]] | None,
    seed: int,
) -> dict[str, Any]:
    route = WaypointRoute(str(example["route"]))
    base = {
        "index": index,
        "sample_id": example["sample_id"],
        "route": route.value,
        "seed": seed,
        "action_valid_mask": [bool(value) for value in example["action_valid_mask"]],
    }
    if route is WaypointRoute.DONE:
        return {**base, "domain": "NONE"}
    if normalized_action is None:
        return {**base, "domain": example["action_domain"], "missing_action": True}
    target_normalized = example.get("action")
    if target_normalized is None:
        raise M0MobileError("active open-loop example has no target action")
    predicted = dataset.normalizer.denormalize(route, normalized_action)
    target = dataset.normalizer.denormalize(route, target_normalized)
    mask = tuple(bool(value) for value in example["action_valid_mask"])
    valid_indices = [position for position, value in enumerate(mask) if value]
    if not valid_indices:
        raise M0MobileError("active open-loop example has no valid target prefix")
    if example["action_domain"] == "NAVIGATION":
        position_errors = [
            math.hypot(
                predicted[position][0] - target[position][0],
                predicted[position][1] - target[position][1],
            )
            for position in valid_indices
        ]
        yaw_errors = [
            abs(wrap_to_pi(predicted[position][2] - target[position][2]))
            for position in valid_indices
        ]
        first = valid_indices[0]
        direction_dot = (
            predicted[first][0] * target[first][0]
            + predicted[first][1] * target[first][1]
        )
        return {
            **base,
            "domain": "NAVIGATION",
            "predicted_normalized": [list(row) for row in normalized_action],
            "target_normalized": [list(row) for row in target_normalized],
            "predicted_physical": [list(row) for row in predicted],
            "target_physical": [list(row) for row in target],
            "ade_m": statistics.fmean(position_errors),
            "fde_m": position_errors[-1],
            "yaw_error_rad": statistics.fmean(yaw_errors),
            "first_waypoint_direction_correct": direction_dot > 0.0,
            "segment_violation": _nav_segment_violation(predicted, valid_indices),
            "normalization_out_of_bounds": _normalized_oob(normalized_action, valid_indices),
        }
    position_errors = [
        math.sqrt(
            sum(
                (predicted[position][axis] - target[position][axis]) ** 2
                for axis in range(3)
            )
        )
        for position in valid_indices
    ]
    orientation_errors = [
        math.sqrt(
            sum(
                wrap_to_pi(predicted[position][axis] - target[position][axis]) ** 2
                for axis in range(3, 6)
            )
        )
        for position in valid_indices
    ]
    gripper_correct = [
        (predicted[position][6] >= 0.5) == (target[position][6] >= 0.5)
        for position in valid_indices
    ]
    return {
        **base,
        "domain": "MANIPULATION",
        "predicted_normalized": [list(row) for row in normalized_action],
        "target_normalized": [list(row) for row in target_normalized],
        "predicted_physical": [list(row) for row in predicted],
        "target_physical": [list(row) for row in target],
        "tcp_position_error_m": statistics.fmean(position_errors),
        "tcp_orientation_error_rad": statistics.fmean(orientation_errors),
        "gripper_accuracy": sum(gripper_correct) / len(gripper_correct),
        "workspace_violation": _arm_workspace_violation(predicted, valid_indices),
        "inter_target_step_violation": _arm_step_violation(predicted, valid_indices),
        "normalization_out_of_bounds": _normalized_oob(normalized_action, valid_indices),
    }


def _nav_segment_violation(
    rows: Sequence[Sequence[float]], indices: Sequence[int]
) -> bool:
    safety = NavWaypointSafety()
    previous = (0.0, 0.0, 0.0)
    for index in indices:
        row = rows[index]
        if (
            math.hypot(row[0] - previous[0], row[1] - previous[1])
            > safety.max_segment_translation_m
            or abs(wrap_to_pi(row[2] - previous[2])) > safety.max_segment_yaw_rad
        ):
            return True
        previous = row[:3]
    return False


def _arm_workspace_violation(
    rows: Sequence[Sequence[float]], indices: Sequence[int]
) -> bool:
    safety = ArmTargetSafety()
    return any(
        any(
            not safety.workspace_min_xyz[axis]
            <= rows[index][axis]
            <= safety.workspace_max_xyz[axis]
            for axis in range(3)
        )
        or not 0.0 <= rows[index][6] <= 1.0
        for index in indices
    )


def _arm_step_violation(
    rows: Sequence[Sequence[float]], indices: Sequence[int]
) -> bool:
    safety = ArmTargetSafety()
    return any(
        math.sqrt(sum((rows[end][axis] - rows[start][axis]) ** 2 for axis in range(3)))
        > safety.max_translation_step_m
        or max(
            abs(wrap_to_pi(rows[end][axis] - rows[start][axis]))
            for axis in range(3, 6)
        )
        > safety.max_axis_rotation_step_rad
        for start, end in zip(indices, indices[1:])
    )


def _normalized_oob(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> bool:
    return any(
        not math.isfinite(float(value)) or abs(float(value)) > 1.0
        for index in indices
        for value in rows[index]
    )


def _gather_rows(local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return local_rows
    gathered: list[list[dict[str, Any]] | None] = [
        None for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather_object(gathered, local_rows)
    return [row for rank_rows in gathered if rank_rows for row in rank_rows]


def _report(
    checkpoint: Path,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    split: str,
    selected: Sequence[int],
    route_rows: Sequence[Mapping[str, Any]],
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    labels = [route.value for route in WaypointRoute]
    columns = [*labels, "RECOVER"]
    confusion = {
        target: {
            predicted: sum(
                row["target"] == target and row["predicted"] == predicted
                for row in route_rows
            )
            for predicted in columns
        }
        for target in labels
    }
    per_seed = {
        str(seed): _seed_metrics(rows)
        for seed, rows in seed_rows.items()
    }
    nav_ades = [
        float(row["ade_m"])
        for rows in seed_rows.values()
        for row in rows
        if row.get("domain") == "NAVIGATION" and "ade_m" in row
    ]
    arm_positions = [
        float(row["tcp_position_error_m"])
        for rows in seed_rows.values()
        for row in rows
        if row.get("domain") == "MANIPULATION" and "tcp_position_error_m" in row
    ]
    nav_seed_means = [
        float(metrics["navigation"]["ade_m"])
        for metrics in per_seed.values()
        if metrics["navigation"]["ade_m"] is not None
    ]
    arm_seed_means = [
        float(metrics["manipulation"]["tcp_position_error_m"])
        for metrics in per_seed.values()
        if metrics["manipulation"]["tcp_position_error_m"] is not None
    ]
    format_invalid = any(row["format_invalid"] for row in route_rows)
    missing_actions = [
        row
        for rows in seed_rows.values()
        for row in rows
        if row.get("domain") in {"NAVIGATION", "MANIPULATION"}
        and row.get("missing_action")
    ]
    nonfinite_metrics = [
        {"sample_id": row["sample_id"], "seed": row["seed"], "field": field}
        for rows in seed_rows.values()
        for row in rows
        for field in (
            "ade_m",
            "fde_m",
            "yaw_error_rad",
            "tcp_position_error_m",
            "tcp_orientation_error_rad",
            "gripper_accuracy",
        )
        if field in row and not math.isfinite(float(row[field]))
    ]
    return {
        "schema_version": "conveyorvla-waypoint-open-loop-report-v1",
        "status": (
            "pass"
            if not format_invalid and not missing_actions and not nonfinite_metrics
            else "fail"
        ),
        "identity": {
            "checkpoint": str(checkpoint),
            "checkpoint_step": manifest["global_step"],
            "checkpoint_source_git": manifest["source_git"],
            "evaluator_source_git": training._source_git_identity(PROJECT_ROOT),
            "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "split": split,
            "selected_indices": list(selected),
            "training_subset": resolved["training_subset"],
        },
        "route": {
            "rows": len(route_rows),
            "format_invalid_count": sum(row["format_invalid"] for row in route_rows),
            "recover_count": sum(row["predicted"] == "RECOVER" for row in route_rows),
            "accuracy": _mean(row["target"] == row["predicted"] for row in route_rows),
            "confusion_matrix": confusion,
            "recover_reasons": dict(
                Counter(
                    str(row["recover_reason"])
                    for row in route_rows
                    if row["recover_reason"] is not None
                )
            ),
            "samples": list(route_rows),
        },
        "oracle_prefix_action": {
            "online_use_forbidden": True,
            "online_route_used_for_action_sampling": False,
            "diffusion_seeds": list(seed_rows),
            "per_seed": per_seed,
            "cross_seed": {
                "navigation_ade_mean_m": _mean(nav_ades),
                "navigation_seed_mean_variance": _variance(nav_seed_means),
                "navigation_worst_sample_ade_m": max(nav_ades, default=None),
                "navigation_worst_sample": _worst_sample(
                    seed_rows, "ade_m", "NAVIGATION"
                ),
                "arm_position_mean_m": _mean(arm_positions),
                "arm_seed_mean_variance": _variance(arm_seed_means),
                "arm_worst_sample_position_error_m": max(arm_positions, default=None),
                "arm_worst_sample": _worst_sample(
                    seed_rows, "tcp_position_error_m", "MANIPULATION"
                ),
            },
            "missing_action_count": len(missing_actions),
            "nonfinite_metrics": nonfinite_metrics,
            "samples_by_seed": {
                str(seed): list(rows) for seed, rows in seed_rows.items()
            },
        },
    }


def _worst_sample(
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    metric: str,
    domain: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for rows in seed_rows.values()
        for row in rows
        if row.get("domain") == domain and metric in row
    ]
    if not candidates:
        return None
    row = max(candidates, key=lambda value: float(value[metric]))
    return {
        "sample_id": row["sample_id"],
        "index": row["index"],
        "seed": row["seed"],
        metric: row[metric],
    }


def _seed_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nav = [
        row
        for row in rows
        if row.get("domain") == "NAVIGATION" and not row.get("missing_action")
    ]
    arm = [
        row
        for row in rows
        if row.get("domain") == "MANIPULATION" and not row.get("missing_action")
    ]
    return {
        "navigation": {
            "samples": len(nav),
            "ade_m": _mean(row["ade_m"] for row in nav),
            "fde_m": _mean(row["fde_m"] for row in nav),
            "yaw_error_rad": _mean(row["yaw_error_rad"] for row in nav),
            "first_waypoint_direction_accuracy": _mean(
                row["first_waypoint_direction_correct"] for row in nav
            ),
            "segment_violation_rate": _mean(row["segment_violation"] for row in nav),
            "normalization_oob_rate": _mean(
                row["normalization_out_of_bounds"] for row in nav
            ),
        },
        "manipulation": {
            "samples": len(arm),
            "tcp_position_error_m": _mean(row["tcp_position_error_m"] for row in arm),
            "tcp_orientation_error_rad": _mean(
                row["tcp_orientation_error_rad"] for row in arm
            ),
            "gripper_accuracy": _mean(row["gripper_accuracy"] for row in arm),
            "workspace_violation_rate": _mean(row["workspace_violation"] for row in arm),
            "inter_target_step_violation_rate": _mean(
                row["inter_target_step_violation"] for row in arm
            ),
            "normalization_oob_rate": _mean(
                row["normalization_out_of_bounds"] for row in arm
            ),
        },
    }


def _mean(values: Iterable[float | bool]) -> float | None:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else None


def _variance(values: Sequence[float]) -> float | None:
    return statistics.pvariance(values) if values else None


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise M0MobileError("diffusion seeds must be comma-separated integers") from error
    if len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise M0MobileError("open-loop evaluation needs at least two unique seeds")
    return seeds


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
