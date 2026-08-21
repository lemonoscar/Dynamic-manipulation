#!/usr/bin/env python3
"""Transition-centric open-loop gate for an immutable Waypoint-v2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import check_waypoint_checkpoint as checkpoint_gate  # noqa: E402
from scripts import evaluate_waypoint_open_loop as base  # noqa: E402
from scripts import train_waypoint as training  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint import (  # noqa: E402
    ACTION_HORIZON,
    WaypointRoute,
    wrap_to_pi,
)
from conveyor_bench.conveyorvla.waypoint_v2 import (  # noqa: E402
    BOUNDARY_EVENTS,
    MODEL_CONTRACT_ID_V2,
)
from conveyor_bench.conveyorvla.waypoint_v2_data import (  # noqa: E402
    ConveyorVLAWaypointV2Dataset,
)
from conveyor_bench.conveyorvla.waypoint_v2_model import (  # noqa: E402
    ConveyorVLAWaypointV2Policy,
)


PLOT_SCHEMA = "conveyorvla-waypoint-v2-action-plots-v1"
REPORT_SCHEMA = "conveyorvla-waypoint-v2-open-loop-report-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--profile", choices=("diagnostic", "overfit"), default="diagnostic"
    )
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--diffusion-seeds", default="17,29,43,71")
    parser.add_argument("--fixed-bank-seed", type=int, default=20260822)
    parser.add_argument("--fixed-bank-draws", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest, resolved, dataset_root = checkpoint_gate._validate_binding(checkpoint)
    if manifest.get("model_contract_id") != MODEL_CONTRACT_ID_V2:
        raise M0MobileError("the waypoint-v2 evaluator rejects non-v2 checkpoints")
    run_args = _mapping(resolved.get("arguments"), "resolved arguments")
    accumulation = int(resolved["gradient_accumulation_steps"])
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    training._validate_accumulation_config(accelerator, accumulation)
    if args.rows < 32 or args.batch_size <= 0 or args.fixed_bank_draws <= 0:
        raise M0MobileError("waypoint-v2 open-loop sizes are invalid")
    group_size = accelerator.num_processes * args.batch_size
    if args.rows % group_size:
        raise M0MobileError(
            "waypoint-v2 rows must divide world_size times evaluation batch"
        )
    seeds = base._parse_seeds(args.diffusion_seeds)
    set_seed(int(run_args["seed"]), device_specific=True)
    config = training._load_config(Path(str(resolved["config"])))
    if not training._is_v2_config(config):
        raise M0MobileError("waypoint-v2 checkpoint resolved to a non-v2 config")
    train_dataset = ConveyorVLAWaypointV2Dataset(dataset_root, split="train")
    training._validate_v2_dataset_config(config, train_dataset.manifest)
    raw_train_indices = resolved.get("training_subset_indices")
    train_indices = (
        None
        if raw_train_indices is None
        else [int(value) for value in raw_train_indices]
    )
    loader_dataset = (
        train_dataset
        if train_indices is None
        else Subset(train_dataset, train_indices)
    )
    loader_routes = (
        train_dataset.routes
        if train_indices is None
        else [train_dataset.routes[index] for index in train_indices]
    )
    loader_weights = training._v2_row_sample_weights(train_dataset, train_indices)
    sampler = training.DomainBalancedSampler(
        loader_routes,
        loader_weights,
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
    model, token_ids = training._build_model(
        config,
        Path(str(resolved["model_root"])),
        str(run_args["attention_implementation"]),
    )
    if token_ids != manifest["special_token_ids"]:
        raise M0MobileError("waypoint-v2 evaluator token IDs differ from checkpoint")
    optimizer, _groups = training._optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        training.common._schedule(
            int(resolved["max_steps"]), int(resolved["warmup_steps"])
        ),
    )
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    training._validate_accumulation_runtime(
        accelerator,
        training.common._deepspeed_engine(accelerator),
        accumulation,
    )
    del loader
    accelerator.load_state(checkpoint)
    model.eval()
    policy = accelerator.unwrap_model(model)
    if not isinstance(policy, ConveyorVLAWaypointV2Policy):
        raise M0MobileError("loaded policy is not Waypoint-v2")
    evaluation = ConveyorVLAWaypointV2Dataset(dataset_root, split=args.split)
    if args.profile == "overfit":
        if args.split != "train" or train_indices is None:
            raise M0MobileError("v2 overfit evaluation needs the recorded train subset")
        candidates = train_indices
    else:
        candidates = list(range(len(evaluation)))
    selected = _transition_centric_selection(evaluation, candidates, args.rows)
    batches = _synchronized_batches(
        selected,
        per_rank_batch_size=args.batch_size,
        world_size=accelerator.num_processes,
    )

    local_route_rows: list[dict[str, Any]] = []
    local_seed_rows: dict[int, list[dict[str, Any]]] = {
        seed: [] for seed in seeds
    }
    local_fixed_rows: list[dict[str, Any]] = []
    for batch_number, batch_indices in enumerate(batches, start=1):
        if accelerator.is_main_process:
            print(
                json.dumps(
                    {
                        "event": "waypoint_v2_evaluation_batch",
                        "batch": batch_number,
                        "batches": len(batches),
                        "rows": len(batch_indices),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        examples = [evaluation[index] for index in batch_indices]
        online = policy.predict_v2(examples)
        crl = policy.oracle_crl_diagnostics(examples)
        fixed = policy.fixed_bank_fm_losses(
            examples,
            bank_seed=args.fixed_bank_seed,
            draws=args.fixed_bank_draws,
        )
        if accelerator.is_main_process:
            local_fixed_rows.extend(_fixed_rows(fixed, args.fixed_bank_draws))
            for index, example, prediction, crl_row in zip(
                batch_indices,
                examples,
                online,
                crl,
                strict=True,
            ):
                decision = prediction.decision
                local_route_rows.append(
                    {
                        "index": index,
                        "sample_id": example["sample_id"],
                        "source_episode_id": example["source_episode_id"],
                        "target": example["route"],
                        "next_route": example["next_route"],
                        "predicted": (
                            decision.route.value
                            if decision.valid and decision.route is not None
                            else "RECOVER"
                        ),
                        "valid": decision.valid,
                        "recover_reason": decision.recover_reason,
                        "format_invalid": (
                            decision.recover_reason in base.FORMAT_RECOVER_REASONS
                        ),
                        "confidence": decision.route_confidence,
                        "decision_probs": dict(decision.decision_probs),
                        "route_probs": dict(decision.route_probs),
                        "subtask": decision.subtask_text,
                        "assistant_prefix": decision.assistant_prefix,
                        "transition_id": example["transition_id"],
                        "boundary_transition": evaluation.boundaries[index],
                        "boundary_class": example["boundary_class"],
                        "boundary_signed_time_s": example[
                            "boundary_signed_time_s"
                        ],
                        "transition_window": example["transition_window"],
                        "phase_progress": example["phase_progress"],
                        "original_valid_prefix_k": example[
                            "original_valid_prefix_k"
                        ],
                        "prefix_target_k": example["prefix_target_k"],
                        "predicted_prefix_k": prediction.trusted_prefix_k,
                        "prefix_scores": prediction.prefix_scores,
                        "boundary_probs": prediction.boundary_probs,
                        "predicted_phase_progress": prediction.phase_progress,
                        "predicted_time_to_boundary_s": (
                            prediction.time_to_boundary_s
                        ),
                        "suffix_reason": example["suffix_reason"],
                        "crl": crl_row,
                    }
                )
        for seed in seeds:
            batch_seed = seed + (batch_number - 1) * 1009
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            actions = policy.predict_oracle_actions(examples)
            if accelerator.is_main_process:
                for index, example, action in zip(
                    batch_indices, examples, actions, strict=True
                ):
                    row = base._action_row(
                        evaluation, index, example, action, seed
                    )
                    row.update(
                        {
                            "boundary_transition": evaluation.boundaries[index],
                            "boundary_class": example["boundary_class"],
                            "transition_window": example["transition_window"],
                            "phase_progress": example["phase_progress"],
                            "original_valid_prefix_k": example[
                                "original_valid_prefix_k"
                            ],
                            "prefix_target_k": example["prefix_target_k"],
                            "suffix_reason": example["suffix_reason"],
                        }
                    )
                    local_seed_rows[seed].append(row)

    route_rows = base._gather_rows(local_route_rows)
    seed_rows = {
        seed: base._gather_rows(local_seed_rows[seed]) for seed in seeds
    }
    fixed_rows = base._gather_rows(local_fixed_rows)
    accelerator.wait_for_everyone()
    report = _report(
        checkpoint,
        manifest,
        resolved,
        config,
        args,
        selected,
        route_rows,
        seed_rows,
        fixed_rows,
    )
    if accelerator.is_main_process:
        report["action_plots"] = _write_action_plots(
            args.plot_dir.expanduser().resolve(),
            seed_rows[seeds[0]],
            route_rows,
            seed=seeds[0],
        )
        training.common._write_json_atomic(args.report.expanduser().resolve(), report)
        print(json.dumps(_console_summary(report), indent=2, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    return 0 if report["status"] == "pass" else 1


def _transition_centric_selection(
    dataset: ConveyorVLAWaypointV2Dataset,
    candidates: Sequence[int],
    rows: int,
) -> list[int]:
    candidate_set = set(int(value) for value in candidates)
    if len(candidate_set) < rows:
        raise M0MobileError("waypoint-v2 evaluation candidate set is too small")
    groups: dict[str, list[int]] = defaultdict(list)
    transition_name: dict[str, str] = {}
    for index in sorted(candidate_set):
        transition_id = dataset.transition_ids[index]
        boundary = dataset.boundaries[index]
        if transition_id is not None and boundary is not None:
            groups[transition_id].append(index)
            transition_name[transition_id] = boundary
    selected: list[int] = []
    selected_set: set[int] = set()

    def add(index: int) -> None:
        if index in candidate_set and index not in selected_set and len(selected) < rows:
            selected.append(index)
            selected_set.add(index)

    for boundary in BOUNDARY_EVENTS:
        event_ids = sorted(
            transition_id
            for transition_id, value in transition_name.items()
            if value == boundary
        )
        if not event_ids:
            raise M0MobileError(f"evaluation candidates omit transition {boundary}")
        for index in groups[event_ids[0]]:
            add(index)
    if len(selected) >= rows:
        raise M0MobileError("evaluation rows cannot fit full four-transition windows")

    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in sorted(candidate_set):
        if dataset.transition_ids[index] is None:
            progress_bin = min(2, int(dataset.phase_progress[index] * 3.0))
            buckets[(dataset.routes[index], progress_bin)].append(index)
    bucket_keys = sorted(buckets)
    while len(selected) < rows:
        progressed = False
        for key in bucket_keys:
            while buckets[key] and buckets[key][0] in selected_set:
                buckets[key].pop(0)
            if buckets[key]:
                add(buckets[key].pop(0))
                progressed = True
            if len(selected) == rows:
                break
        if not progressed:
            break
    for index in sorted(candidate_set):
        add(index)
        if len(selected) == rows:
            break
    if len(selected) != rows:
        raise M0MobileError("could not construct waypoint-v2 evaluation selection")
    routes = {dataset.routes[index] for index in selected}
    boundaries = {
        dataset.boundaries[index]
        for index in selected
        if dataset.boundaries[index] is not None
    }
    if routes != {route.value for route in WaypointRoute}:
        raise M0MobileError("waypoint-v2 evaluation does not cover every route")
    if boundaries != set(BOUNDARY_EVENTS):
        raise M0MobileError("waypoint-v2 evaluation does not cover every boundary")
    return selected


def _synchronized_batches(
    selected: Sequence[int], per_rank_batch_size: int, world_size: int
) -> list[list[int]]:
    """Keep ZeRO-3 ranks on identical examples and identical module branches."""
    synchronized_batch_size = per_rank_batch_size * world_size
    if synchronized_batch_size <= 0 or len(selected) % synchronized_batch_size:
        raise M0MobileError(
            "waypoint-v2 selection is not aligned to the synchronized batch size"
        )
    return [
        list(selected[offset : offset + synchronized_batch_size])
        for offset in range(0, len(selected), synchronized_batch_size)
    ]


def _fixed_rows(
    fixed: Mapping[str, torch.Tensor | int], draws: int
) -> list[dict[str, Any]]:
    rows = []
    for domain in ("navigation", "manipulation"):
        count = int(fixed[f"{domain}_fixed_bank_samples"])
        if not count:
            continue
        rows.append(
            {
                "domain": domain,
                "samples": count,
                "draw_losses": [
                    float(
                        fixed[f"{domain}_fixed_bank_draw_{draw}_loss"]
                        .detach()
                        .float()
                        .item()
                    )
                    for draw in range(draws)
                ],
            }
        )
    return rows


def _report(
    checkpoint: Path,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    selected: Sequence[int],
    route_rows: Sequence[Mapping[str, Any]],
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    fixed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_report = base._report(
        checkpoint,
        manifest,
        resolved,
        args.split,
        args.profile,
        selected,
        route_rows,
        seed_rows,
    )
    transition = _transition_metrics(route_rows)
    prefix = _prefix_metrics(route_rows)
    action = _action_diagnostics(seed_rows)
    fixed_bank = _fixed_bank_metrics(
        fixed_rows,
        args.fixed_bank_seed,
        args.fixed_bank_draws,
        selected,
        route_rows,
        config,
    )
    crl = _crl_metrics(route_rows)
    boundary = _boundary_metrics(route_rows)
    coverage_pass = (
        set(transition["per_transition"]) == set(BOUNDARY_EVENTS)
        and {row["target"] for row in route_rows}
        == {route.value for route in WaypointRoute}
    )
    structural_pass = bool(base_report["gate"]["structural_pass"]) and coverage_pass
    quality_pass = bool(base_report["gate"]["quality_pass"])
    status = "pass" if structural_pass and quality_pass else "fail"
    return {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "profile": args.profile,
        "identity": {
            **base_report["identity"],
            "fixed_bank_sha256": fixed_bank["manifest_sha256"],
        },
        "selection": {
            "rows": len(selected),
            "indices": list(selected),
            "route_counts": _counts(row["target"] for row in route_rows),
            "boundary_counts": _counts(
                row["boundary_transition"]
                for row in route_rows
                if row["boundary_transition"] is not None
            ),
        },
        "gate": {
            "structural_pass": structural_pass,
            "coverage_pass": coverage_pass,
            "quality_pass": quality_pass,
            "legacy_action_route_gate": base_report["gate"],
        },
        "route": base_report["route"],
        "transition": transition,
        "boundary_progress": boundary,
        "prefix": prefix,
        "oracle_prefix_action": base_report["oracle_prefix_action"],
        "action_diagnostics": action,
        "fixed_validation_bank": fixed_bank,
        "crl": crl,
        "truth_is_evaluation_only": True,
        "truth_written_to_model_request": False,
        "truth_written_to_control_chain": False,
    }


def _transition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["transition_id"] is not None:
            events[str(row["transition_id"])].append(row)
    event_reports = []
    for transition_id, event_rows in sorted(events.items()):
        ordered = sorted(
            (
                row
                for row in event_rows
                if row["boundary_signed_time_s"] is not None
            ),
            key=lambda row: float(row["boundary_signed_time_s"]),
        )
        if not ordered:
            continue
        transition = str(ordered[0]["boundary_transition"])
        old_route, new_route = transition.split("->", maxsplit=1)
        before = [row for row in ordered if float(row["boundary_signed_time_s"]) < 0]
        after = [row for row in ordered if float(row["boundary_signed_time_s"]) >= 0]
        first_new = next((row for row in ordered if row["predicted"] == new_route), None)
        boundary_query = min(
            range(len(ordered)),
            key=lambda index: abs(float(ordered[index]["boundary_signed_time_s"])),
        )
        first_new_query = next(
            (index for index, row in enumerate(ordered) if row["predicted"] == new_route),
            None,
        )
        differences = [
            float(row["route_probs"].get(new_route, 0.0))
            - float(row["route_probs"].get(old_route, 0.0))
            for row in ordered
        ]
        crossover = _logit_crossover(ordered, differences)
        predictions = [str(row["predicted"]) for row in ordered]
        flicker = sum(
            left == right and left != middle
            for left, middle, right in zip(
                predictions, predictions[1:], predictions[2:], strict=False
            )
        )
        event_reports.append(
            {
                "transition_id": transition_id,
                "transition": transition,
                "episode": ordered[0]["source_episode_id"],
                "queries": len(ordered),
                "early_switch_rate": _mean(
                    row["predicted"] == new_route for row in before
                ),
                "late_switch_rate": _mean(
                    row["predicted"] == old_route for row in after
                ),
                "switch_lag_s": (
                    None
                    if first_new is None
                    else float(first_new["boundary_signed_time_s"])
                ),
                "switch_lag_queries": (
                    None
                    if first_new_query is None
                    else first_new_query - boundary_query
                ),
                "logit_crossover_s": crossover,
                "flicker_count": flicker,
                "rows": [
                    {
                        "sample_id": row["sample_id"],
                        "signed_time_s": row["boundary_signed_time_s"],
                        "target": row["target"],
                        "predicted": row["predicted"],
                        "old_logit_probability": row["route_probs"].get(old_route),
                        "new_logit_probability": row["route_probs"].get(new_route),
                    }
                    for row in ordered
                ],
            }
        )
    per_transition = {}
    for transition in BOUNDARY_EVENTS:
        values = [row for row in event_reports if row["transition"] == transition]
        lags_s = [
            abs(float(row["switch_lag_s"]))
            for row in values
            if row["switch_lag_s"] is not None
        ]
        lags_q = [
            abs(float(row["switch_lag_queries"]))
            for row in values
            if row["switch_lag_queries"] is not None
        ]
        per_transition[transition] = {
            "events": len(values),
            "early_switch_rate": _mean(
                row["early_switch_rate"]
                for row in values
                if row["early_switch_rate"] is not None
            ),
            "late_switch_rate": _mean(
                row["late_switch_rate"]
                for row in values
                if row["late_switch_rate"] is not None
            ),
            "absolute_lag_median_s": _median(lags_s),
            "absolute_lag_p95_s": _percentile(lags_s, 0.95),
            "absolute_lag_median_queries": _median(lags_q),
            "absolute_lag_p95_queries": _percentile(lags_q, 0.95),
            "flicker_count": sum(int(row["flicker_count"]) for row in values),
        }
    interior = [row for row in rows if not row["transition_window"]]
    return {
        "per_transition": per_transition,
        "events": event_reports,
        "phase_interior_macro_accuracy": _macro_accuracy(interior),
        "phase_interior_rows": len(interior),
        "flicker_count": sum(int(row["flicker_count"]) for row in event_reports),
    }


def _logit_crossover(
    rows: Sequence[Mapping[str, Any]], differences: Sequence[float]
) -> float | None:
    for index in range(1, len(rows)):
        left, right = differences[index - 1], differences[index]
        if left == 0.0:
            return float(rows[index - 1]["boundary_signed_time_s"])
        if left * right <= 0.0:
            left_time = float(rows[index - 1]["boundary_signed_time_s"])
            right_time = float(rows[index]["boundary_signed_time_s"])
            denominator = abs(left) + abs(right)
            return (
                left_time
                if denominator == 0.0
                else left_time + (right_time - left_time) * abs(left) / denominator
            )
    if not differences:
        return None
    index = min(range(len(differences)), key=lambda value: abs(differences[value]))
    return float(rows[index]["boundary_signed_time_s"])


def _boundary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["boundary_probs"] is not None]
    if not scored:
        return {
            "enabled": False,
            "boundary_f1": None,
            "boundary_auroc": None,
            "progress_mae": None,
            "time_to_boundary_mae_s": None,
        }
    labels = [bool(row["transition_window"]) for row in scored]
    scores = [
        1.0 - float(row["boundary_probs"].get("INTERIOR", 0.0))
        for row in scored
    ]
    predictions = [score >= 0.5 for score in scores]
    true_positive = sum(prediction and label for prediction, label in zip(predictions, labels))
    false_positive = sum(
        prediction and not label for prediction, label in zip(predictions, labels)
    )
    false_negative = sum(
        not prediction and label for prediction, label in zip(predictions, labels)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    progress_errors = [
        abs(float(row["predicted_phase_progress"]) - float(row["phase_progress"]))
        for row in scored
        if row["predicted_phase_progress"] is not None
    ]
    time_errors = [
        abs(
            float(row["predicted_time_to_boundary_s"])
            - max(0.0, -float(row["boundary_signed_time_s"]))
        )
        for row in scored
        if row["predicted_time_to_boundary_s"] is not None
        and row["boundary_signed_time_s"] is not None
        and float(row["boundary_signed_time_s"]) <= 0.0
    ]
    return {
        "enabled": True,
        "boundary_f1": None if not denominator else 2 * true_positive / denominator,
        "boundary_auroc": _binary_auc(labels, scores),
        "progress_mae": _mean(progress_errors),
        "time_to_boundary_mae_s": _mean(time_errors),
        "rows": len(scored),
    }


def _prefix_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = [
        row
        for row in rows
        if row["target"] != WaypointRoute.DONE.value
        and row["predicted_prefix_k"] is not None
    ]
    if not active:
        return {"rows": 0, "mae_k": None, "overrun_rate": None}
    errors = [
        int(row["predicted_prefix_k"]) - int(row["prefix_target_k"])
        for row in active
    ]
    by_route = {}
    for route in (value.value for value in WaypointRoute if value is not WaypointRoute.DONE):
        route_rows = [row for row in active if row["target"] == route]
        by_route[route] = {
            "rows": len(route_rows),
            "mean_predicted_k": _mean(
                int(row["predicted_prefix_k"]) for row in route_rows
            ),
            "overrun_rate": _mean(
                int(row["predicted_prefix_k"]) > int(row["prefix_target_k"])
                for row in route_rows
            ),
        }
    return {
        "rows": len(active),
        "mae_k": statistics.fmean(abs(value) for value in errors),
        "overrun_rate": _mean(value > 0 for value in errors),
        "mean_under_run_points": _mean(max(0, -value) for value in errors),
        "mean_over_run_points": _mean(max(0, value) for value in errors),
        "by_route": by_route,
    }


def _fixed_bank_metrics(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    draws: int,
    selected: Sequence[int],
    route_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for domain in ("navigation", "manipulation"):
        domain_rows = [row for row in rows if row["domain"] == domain]
        draw_losses = []
        for draw in range(draws):
            numerator = sum(
                int(row["samples"]) * float(row["draw_losses"][draw])
                for row in domain_rows
            )
            denominator = sum(int(row["samples"]) for row in domain_rows)
            draw_losses.append(None if not denominator else numerator / denominator)
        finite = [float(value) for value in draw_losses if value is not None]
        result[domain] = {
            "samples": sum(int(row["samples"]) for row in domain_rows),
            "draw_losses": draw_losses,
            "mean_loss": _mean(finite),
            "draw_std": (
                None if not finite else statistics.pstdev(finite)
            ),
        }
    bank_manifest = {
        "schema_version": "conveyorvla-waypoint-v2-fixed-fm-bank-v1",
        "algorithm": "sample-id-sha256-cpu-philox-beta-alpha-inverse",
        "seed": seed,
        "draws": draws,
        "selected_indices": list(selected),
        "sample_ids": sorted(str(row["sample_id"]) for row in route_rows),
        "noise_beta_alpha": config["action_model"]["noise_beta_alpha"],
        "noise_beta_beta": config["action_model"]["noise_beta_beta"],
        "noise_s": config["action_model"]["noise_s"],
    }
    encoded = json.dumps(bank_manifest, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest": bank_manifest,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        **result,
    }


def _action_diagnostics(
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    rows = [row for values in seed_rows.values() for row in values]
    result = {}
    for domain, channels in (
        ("NAVIGATION", ("x_m", "y_m", "yaw_rad")),
        (
            "MANIPULATION",
            ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper"),
        ),
    ):
        domain_rows = [
            row
            for row in rows
            if row.get("domain") == domain and "predicted_physical" in row
        ]
        channel_metrics = {}
        for channel, name in enumerate(channels):
            pairs = [
                (
                    float(row["target_physical"][step][channel]),
                    float(row["predicted_physical"][step][channel]),
                )
                for row in domain_rows
                for step, valid in enumerate(row["action_valid_mask"])
                if valid
            ]
            angular = (
                domain == "NAVIGATION" and channel == 2
            ) or (
                domain == "MANIPULATION" and 3 <= channel <= 5
            )
            errors = [
                wrap_to_pi(prediction - target)
                if angular
                else prediction - target
                for target, prediction in pairs
            ]
            channel_metrics[name] = {
                "mae": _mean(abs(value) for value in errors),
                "mse": _mean(value * value for value in errors),
                "correlation": _pearson(pairs),
            }
        horizon = []
        for step in range(ACTION_HORIZON):
            errors = [
                statistics.fmean(
                    abs(
                        wrap_to_pi(
                            float(row["predicted_physical"][step][channel])
                            - float(row["target_physical"][step][channel])
                        )
                        if (
                            (domain == "NAVIGATION" and channel == 2)
                            or (
                                domain == "MANIPULATION"
                                and 3 <= channel <= 5
                            )
                        )
                        else float(row["predicted_physical"][step][channel])
                        - float(row["target_physical"][step][channel])
                    )
                    for channel in range(len(channels))
                )
                for row in domain_rows
                if row["action_valid_mask"][step]
            ]
            horizon.append(_mean(errors))
        terminal_hold_errors = [
            abs(
                wrap_to_pi(
                    float(row["predicted_physical"][step][channel])
                    - float(row["target_physical"][step][channel])
                )
                if (
                    (domain == "NAVIGATION" and channel == 2)
                    or (domain == "MANIPULATION" and 3 <= channel <= 5)
                )
                else float(row["predicted_physical"][step][channel])
                - float(row["target_physical"][step][channel])
            )
            for row in domain_rows
            if row.get("suffix_reason") == "boundary"
            for step in range(int(row["original_valid_prefix_k"]), ACTION_HORIZON)
            for channel in range(len(channels))
        ]
        result[domain.lower()] = {
            "rows": len(domain_rows),
            "channels": channel_metrics,
            "horizon_mae": horizon,
            "terminal_hold_suffix_mae": _mean(terminal_hold_errors),
        }
    nav_rows = [row for row in rows if row.get("domain") == "NAVIGATION"]
    arm_rows = [row for row in rows if row.get("domain") == "MANIPULATION"]
    result["navigation"]["direction_accuracy"] = _mean(
        bool(row["first_waypoint_direction_correct"]) for row in nav_rows
    )
    result["navigation"]["mean_predicted_yaw_step_rad"] = _mean(
        abs(
            wrap_to_pi(
                float(row["predicted_physical"][step][2])
                - float(row["predicted_physical"][step - 1][2])
            )
        )
        for row in nav_rows
        for step in range(1, ACTION_HORIZON)
    )
    result["manipulation"]["mean_predicted_translation_step_m"] = _mean(
        math.sqrt(
            sum(
                (
                    float(row["predicted_physical"][step][axis])
                    - float(row["predicted_physical"][step - 1][axis])
                )
                ** 2
                for axis in range(3)
            )
        )
        for row in arm_rows
        for step in range(1, ACTION_HORIZON)
    )
    result["manipulation"]["mean_predicted_rpy_step_rad"] = _mean(
        statistics.fmean(
            abs(
                wrap_to_pi(
                    float(row["predicted_physical"][step][axis])
                    - float(row["predicted_physical"][step - 1][axis])
                )
            )
            for axis in range(3, 6)
        )
        for row in arm_rows
        for step in range(1, ACTION_HORIZON)
    )
    return result


def _crl_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = [row for row in rows if row.get("crl") is not None]
    if not active:
        return {"enabled": False, "rows": 0}
    by_route = {}
    for route in (value.value for value in WaypointRoute if value is not WaypointRoute.DONE):
        route_rows = [row for row in active if row["target"] == route]
        progress_value = [
            (float(row["phase_progress"]), float(row["crl"]["correct_goal_similarity"]))
            for row in route_rows
        ]
        by_route[route] = {
            "rows": len(route_rows),
            "goal_margin": _mean(float(row["crl"]["goal_margin"]) for row in route_rows),
            "action_shuffle_drop": _mean(
                float(row["crl"]["action_shuffle_drop"]) for row in route_rows
            ),
            "progress_spearman": _spearman(progress_value),
            "progress_kendall": _kendall(progress_value),
        }
    return {
        "enabled": True,
        "runtime_route_override": False,
        "rows": len(active),
        "goal_margin": _mean(float(row["crl"]["goal_margin"]) for row in active),
        "action_shuffle_drop": _mean(
            float(row["crl"]["action_shuffle_drop"]) for row in active
        ),
        "by_route": by_route,
    }


def _write_action_plots(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    route_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if output.exists():
        raise M0MobileError(f"waypoint-v2 plot output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise M0MobileError("matplotlib is required for waypoint-v2 plots") from error
    route_by_sample = {str(row["sample_id"]): row for row in route_rows}
    files = []
    for row in rows:
        if row.get("domain") not in {"NAVIGATION", "MANIPULATION"}:
            continue
        channels = (
            ("x [m]", "y [m]", "yaw [rad]")
            if row["domain"] == "NAVIGATION"
            else (
                "x [m]",
                "y [m]",
                "z [m]",
                "roll [rad]",
                "pitch [rad]",
                "yaw [rad]",
                "gripper",
            )
        )
        figure, axes = plt.subplots(
            len(channels),
            1,
            figsize=(10, 2.2 * len(channels)),
            sharex=True,
            constrained_layout=True,
        )
        axes_list = [axes] if len(channels) == 1 else list(axes)
        steps = list(range(1, ACTION_HORIZON + 1))
        for channel, (axis, label) in enumerate(zip(axes_list, channels, strict=True)):
            axis.plot(
                steps,
                [value[channel] for value in row["target_physical"]],
                label="target",
                linewidth=2,
            )
            axis.plot(
                steps,
                [value[channel] for value in row["predicted_physical"]],
                label="predicted",
                linewidth=1.5,
            )
            original_k = int(row.get("original_valid_prefix_k", ACTION_HORIZON))
            if row.get("suffix_reason") == "boundary" and original_k < ACTION_HORIZON:
                axis.axvspan(original_k + 0.5, ACTION_HORIZON + 0.5, alpha=0.12)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes_list[0].legend(loc="best")
        axes_list[-1].set_xlabel("20-step horizon")
        online = route_by_sample.get(str(row["sample_id"]), {})
        figure.suptitle(
            f"{row['sample_id']} | target={row['route']} | "
            f"predicted_route={online.get('predicted')} | seed={seed}"
        )
        safe = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(row["sample_id"])
        )
        path = output / f"{int(row['index']):06d}_{safe}_{row['domain'].lower()}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        files.append(
            {
                "relative_path": path.name,
                "sha256": _sha256(path),
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "channels": len(channels),
                "horizon": ACTION_HORIZON,
            }
        )
    manifest = {
        "schema_version": PLOT_SCHEMA,
        "seed": seed,
        "files": files,
    }
    training.common._write_json_atomic(output / "manifest.json", manifest)
    return {
        "directory": str(output),
        "manifest_sha256": _sha256(output / "manifest.json"),
        "file_count": len(files),
        "seed": seed,
    }


def _console_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "checkpoint": report["identity"]["checkpoint"],
        "rows": report["selection"]["rows"],
        "route_accuracy": report["route"]["accuracy"],
        "transition": report["transition"]["per_transition"],
        "prefix": report["prefix"],
        "fixed_validation_bank": report["fixed_validation_bank"],
        "crl": report["crl"],
        "action_plots": report.get("action_plots"),
    }


def _macro_accuracy(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = []
    for route in WaypointRoute:
        route_rows = [row for row in rows if row["target"] == route.value]
        if route_rows:
            values.append(_mean(row["predicted"] == route.value for row in route_rows))
    return _mean(value for value in values if value is not None)


def _binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positive = [score for label, score in zip(labels, scores, strict=True) if label]
    negative = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positive or not negative:
        return None
    wins = sum(
        1.0 if left > right else 0.5 if left == right else 0.0
        for left in positive
        for right in negative
    )
    return wins / (len(positive) * len(negative))


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [value[0] for value in pairs]
    right = [value[1] for value in pairs]
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return None if denominator == 0.0 else numerator / denominator


def _spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = _ranks([value[0] for value in pairs])
    right = _ranks([value[1] for value in pairs])
    return _pearson(list(zip(left, right, strict=True)))


def _kendall(pairs: Sequence[tuple[float, float]]) -> float | None:
    concordant = discordant = 0
    for first in range(len(pairs)):
        for second in range(first + 1, len(pairs)):
            product = (pairs[first][0] - pairs[second][0]) * (
                pairs[first][1] - pairs[second][1]
            )
            concordant += product > 0
            discordant += product < 0
    total = concordant + discordant
    return None if not total else (concordant - discordant) / total


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def _mean(values: Iterable[float | bool | int]) -> float | None:
    rows = [float(value) for value in values]
    return None if not rows else statistics.fmean(rows)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else statistics.median(values)


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _counts(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
