#!/usr/bin/env python3
"""Probe frozen Qwen visual hidden states for waypoint transition observability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint import (  # noqa: E402
    ACTION_HORIZON,
    WaypointRoute,
)
from conveyor_bench.conveyorvla.waypoint_data import (  # noqa: E402
    ConveyorVLAWaypointDataset,
)


BOUNDARY_WINDOW_S = 1.0
ROUTE_INDEX = {route.value: index for index, route in enumerate(WaypointRoute)}
BOUNDARY_INDEX = {"INTERIOR": 0, "BEFORE": 1, "AFTER": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--val-rows", type=int, default=640)
    parser.add_argument("--test-rows", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = _run(args, output)
        _write_json(output / "observability_report.json", report)
        _plot_transition_curves(output / "transition_observability.png", report)
        print(json.dumps(_summary(report), indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        _write_json(
            output / "failure.json",
            {
                "schema_version": "conveyorvla-waypoint-observability-failure-v1",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _run(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    from scripts import serve_waypoint

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    service, export_report = serve_waypoint.load_service(
        argparse.Namespace(
            export_dir=args.export_dir,
            device=args.device,
            attention_implementation=args.attention_implementation,
            seed=args.seed,
        )
    )
    policy = service.session.policy
    policy.requires_grad_(False)
    policy.eval()

    split_limits = {
        "train": args.train_rows,
        "val": args.val_rows,
        "test": args.test_rows,
    }
    features: dict[str, torch.Tensor] = {}
    labels: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, list[int]] = {}
    for split, limit in split_limits.items():
        records = _read_records(args.dataset_root, split)
        split_labels = transition_labels(records)
        split_indices = balanced_indices(split_labels, limit, args.seed + len(selected))
        dataset = ConveyorVLAWaypointDataset(args.dataset_root, split=split)
        features[split] = extract_visual_features(
            policy,
            dataset,
            split_indices,
            batch_size=args.batch_size,
        )
        labels[split] = [split_labels[index] for index in split_indices]
        selected[split] = split_indices
        np.save(output / f"{split}_features.npy", features[split].numpy())
        _write_json(
            output / f"{split}_selection.json",
            {
                "split": split,
                "indices": split_indices,
                "labels": labels[split],
            },
        )

    del policy, service
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    probe_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    probes: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}
    tasks = {
        "route": ("classification", len(ROUTE_INDEX), lambda row: ROUTE_INDEX[row["route"]], lambda row: True),
        "boundary": ("classification", len(BOUNDARY_INDEX), lambda row: BOUNDARY_INDEX[row["boundary_class"]], lambda row: True),
        "prefix_k": ("classification", ACTION_HORIZON + 1, lambda row: int(row["original_valid_prefix_k"]), lambda row: row["route"] != WaypointRoute.DONE.value),
        "time_to_boundary_s": ("regression", 1, lambda row: float(row["time_to_boundary_s"]), lambda row: bool(row["has_future_boundary"])),
    }
    for task, (kind, output_dim, target, eligible) in tasks.items():
        task_features, task_targets, task_rows = _task_data(
            features, labels, target=target, eligible=eligible, kind=kind
        )
        probes[task] = {}
        predictions[task] = {}
        for architecture in ("linear", "mlp"):
            result, test_prediction = fit_probe(
                task_features,
                task_targets,
                kind=kind,
                output_dim=output_dim,
                architecture=architecture,
                epochs=args.epochs,
                seed=args.seed,
                device=probe_device,
            )
            if task == "prefix_k":
                truth = task_targets["test"].tolist()
                result["test"].update(_prefix_metrics(test_prediction, truth))
            probes[task][architecture] = result
            predictions[task][architecture] = {
                "values": test_prediction,
                "rows": task_rows["test"],
            }

    transition_curves = _transition_curves(labels["test"], predictions)
    dataset_root = args.dataset_root.expanduser().resolve()
    export_root = args.export_dir.expanduser().resolve()
    return {
        "schema_version": "conveyorvla-waypoint-observability-probe-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "purpose": "diagnostic_only_not_runtime_phase_detector",
        "feature_contract": {
            "source": "frozen_qwen_final_hidden_attention_mean",
            "assistant_solution_supplied": False,
            "inputs": ["instruction", "head[t-0.20,t]", "wrist[t-0.20,t]"],
            "robot_state_fields": 0,
            "boundary_window_s": BOUNDARY_WINDOW_S,
        },
        "export": {
            "root": str(export_root),
            "manifest_sha256": _sha256(export_root / "inference_manifest.json"),
            **export_report,
        },
        "dataset": {
            "root": str(dataset_root),
            "manifest_sha256": _sha256(dataset_root / "manifest.json"),
            "split_unit": "source_episode_id",
            "selected_rows": {key: len(value) for key, value in selected.items()},
            "selection_files": {key: f"{key}_selection.json" for key in selected},
            "feature_sha256": {
                key: _sha256(output / f"{key}_features.npy") for key in selected
            },
            "selection_sha256": {
                key: _sha256(output / f"{key}_selection.json") for key in selected
            },
        },
        "source_git": _git_identity(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "hostname": platform.node(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu_uuids": os.environ.get("CONVEYORVLA_GPU_UUIDS"),
            "device": str(probe_device),
            "device_name": (
                torch.cuda.get_device_name(probe_device)
                if probe_device.type == "cuda"
                else None
            ),
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "probes": probes,
        "transition_curves": transition_curves,
        "interpretation": _interpret(probes),
    }


def transition_labels(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive visual-only probe targets without changing the frozen v1 data."""

    result: list[dict[str, Any] | None] = [None] * len(records)
    episodes: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        episodes[str(record["source_episode_id"])].append(index)
    for episode_indices in episodes.values():
        ordered = sorted(episode_indices, key=lambda index: float(records[index]["timestamp"]))
        events: list[tuple[int, int, str]] = []
        for position in range(1, len(ordered)):
            previous = str(records[ordered[position - 1]]["route"])
            current = str(records[ordered[position]]["route"])
            if current != previous:
                events.append((position, ordered[position], f"{previous}->{current}"))
        for position, index in enumerate(ordered):
            record = records[index]
            timestamp = float(record["timestamp"])
            previous_event = next((event for event in reversed(events) if event[0] <= position), None)
            next_event = next((event for event in events if event[0] > position), None)
            time_since = (
                math.inf
                if previous_event is None
                else timestamp - float(records[previous_event[1]]["timestamp"])
            )
            time_until = (
                math.inf
                if next_event is None
                else float(records[next_event[1]]["timestamp"]) - timestamp
            )
            if time_until <= BOUNDARY_WINDOW_S + 1.0e-6:
                boundary_class = "BEFORE"
                transition = next_event[2] if next_event is not None else None
                signed_time = -time_until
            elif time_since <= BOUNDARY_WINDOW_S + 1.0e-6:
                boundary_class = "AFTER"
                transition = previous_event[2] if previous_event is not None else None
                signed_time = time_since
            else:
                boundary_class = "INTERIOR"
                transition = None
                signed_time = None
            mask = tuple(bool(value) for value in record["action_valid_mask"])
            result[index] = {
                "source_episode_id": str(record["source_episode_id"]),
                "source_row_id": int(record["source_row_id"]),
                "route": str(record["route"]),
                "boundary_class": boundary_class,
                "transition": transition,
                "signed_boundary_time_s": signed_time,
                "has_future_boundary": next_event is not None,
                "time_to_boundary_s": None if next_event is None else time_until,
                "original_valid_prefix_k": sum(mask),
            }
    if any(value is None for value in result):
        raise M0MobileError("observability labels did not cover every record")
    return [value for value in result if value is not None]


def balanced_indices(labels: Sequence[Mapping[str, Any]], limit: int, seed: int) -> list[int]:
    if limit <= 0 or limit > len(labels):
        raise M0MobileError("observability row limit must fit the split")
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(labels):
        buckets[(str(row["route"]), str(row["boundary_class"]), str(row["transition"]))].append(index)
    generator = random.Random(seed)
    for values in buckets.values():
        generator.shuffle(values)
    selected: list[int] = []
    active = sorted(buckets)
    while active and len(selected) < limit:
        next_active = []
        for key in active:
            values = buckets[key]
            if values:
                selected.append(values.pop())
            if values:
                next_active.append(key)
            if len(selected) == limit:
                break
        active = next_active
    if len(selected) != limit:
        raise M0MobileError("cannot build the balanced observability selection")
    return selected


@torch.inference_mode()
def extract_visual_features(
    policy: nn.Module,
    dataset: ConveyorVLAWaypointDataset,
    indices: Sequence[int],
    *,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    for offset in range(0, len(indices), batch_size):
        examples = [dataset[index] for index in indices[offset : offset + batch_size]]
        inputs = dict(
            policy.qwen.build_waypoint_inputs(
                examples,
                solutions=None,
                supervise_solutions=False,
            )
        )
        if "labels" in inputs:
            raise M0MobileError("observability probe unexpectedly received assistant labels")
        outputs = policy.qwen(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise M0MobileError("Qwen did not return observability hidden states")
        hidden = outputs.hidden_states[-1].float()
        attention = inputs["attention_mask"].to(hidden).unsqueeze(-1)
        pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1.0)
        rows.append(pooled.cpu())
    return torch.cat(rows, dim=0)


def _task_data(
    features: Mapping[str, torch.Tensor],
    labels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target: Any,
    eligible: Any,
    kind: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[Mapping[str, Any]]]]:
    task_features: dict[str, torch.Tensor] = {}
    task_targets: dict[str, torch.Tensor] = {}
    task_rows: dict[str, list[Mapping[str, Any]]] = {}
    for split in ("train", "val", "test"):
        indices = [index for index, row in enumerate(labels[split]) if eligible(row)]
        if not indices:
            raise M0MobileError(f"observability {kind} task has no {split} rows")
        task_features[split] = features[split].index_select(0, torch.tensor(indices))
        values = [target(labels[split][index]) for index in indices]
        dtype = torch.long if kind == "classification" else torch.float32
        task_targets[split] = torch.tensor(values, dtype=dtype)
        task_rows[split] = [labels[split][index] for index in indices]
    return task_features, task_targets, task_rows


def fit_probe(
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    kind: str,
    output_dim: int,
    architecture: str,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[Any]]:
    torch.manual_seed(seed + (0 if architecture == "linear" else 1))
    mean = features["train"].mean(dim=0)
    std = features["train"].std(dim=0).clamp_min(1.0e-6)
    normalized = {key: ((value - mean) / std).to(device) for key, value in features.items()}
    labels = {key: value.to(device) for key, value in targets.items()}
    input_dim = int(features["train"].shape[1])
    model: nn.Module = (
        nn.Linear(input_dim, output_dim)
        if architecture == "linear"
        else nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, output_dim))
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(normalized["train"]), generator=generator)
        for start in range(0, len(order), 256):
            indices = order[start : start + 256].to(device)
            prediction = model(normalized["train"].index_select(0, indices))
            truth = labels["train"].index_select(0, indices)
            loss = (
                nn.functional.cross_entropy(prediction, truth)
                if kind == "classification"
                else nn.functional.smooth_l1_loss(prediction[:, 0], truth)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = model(normalized["val"])
            validation_loss = float(
                (
                    nn.functional.cross_entropy(validation, labels["val"])
                    if kind == "classification"
                    else nn.functional.smooth_l1_loss(validation[:, 0], labels["val"])
                ).cpu()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise M0MobileError("observability probe did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    metrics: dict[str, Any] = {"best_validation_loss": best_loss}
    predictions: dict[str, list[Any]] = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            raw = model(normalized[split])
            if kind == "classification":
                predicted = raw.argmax(dim=1)
                metrics[split] = _classification_metrics(predicted.cpu(), targets[split], output_dim)
                predictions[split] = predicted.cpu().tolist()
            else:
                predicted = raw[:, 0].clamp_min(0.0)
                metrics[split] = _regression_metrics(predicted.cpu(), targets[split])
                predictions[split] = predicted.cpu().tolist()
    return metrics, predictions["test"]


def _classification_metrics(predicted: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, float]:
    accuracy = float((predicted == target).float().mean())
    f1 = []
    for value in range(classes):
        tp = int(((predicted == value) & (target == value)).sum())
        fp = int(((predicted == value) & (target != value)).sum())
        fn = int(((predicted != value) & (target == value)).sum())
        f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    counts = torch.bincount(target, minlength=classes)
    majority = float(counts.max() / counts.sum())
    return {"accuracy": accuracy, "macro_f1": sum(f1) / classes, "majority_accuracy": majority}


def _regression_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = predicted - target
    mae = float(error.abs().mean())
    denominator = float(((target - target.mean()) ** 2).sum())
    r2 = 0.0 if denominator <= 0.0 else 1.0 - float((error**2).sum()) / denominator
    baseline = float((target - target.median()).abs().mean())
    return {"mae": mae, "r2": r2, "median_baseline_mae": baseline}


def _transition_curves(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    boundary_values = predictions["boundary"]["mlp"]["values"]
    for transition in (
        "NAV_TO_SOURCE->PICK",
        "PICK->NAV_TO_TARGET",
        "NAV_TO_TARGET->PLACE",
        "PLACE->DONE",
    ):
        values = []
        for row, prediction in zip(rows, boundary_values, strict=True):
            if row["transition"] == transition and row["signed_boundary_time_s"] is not None:
                values.append(
                    {
                        "signed_time_s": float(row["signed_boundary_time_s"]),
                        "target": BOUNDARY_INDEX[str(row["boundary_class"])],
                        "predicted": int(prediction),
                    }
                )
        ordered = sorted(values, key=lambda value: value["signed_time_s"])
        curves[transition] = {
            "rows": ordered,
            "count": len(ordered),
            "accuracy": (
                None
                if not ordered
                else sum(row["target"] == row["predicted"] for row in ordered)
                / len(ordered)
            ),
        }
    return curves


def _prefix_metrics(predicted: Sequence[int], target: Sequence[int]) -> dict[str, float]:
    if len(predicted) != len(target) or not target:
        raise M0MobileError("prefix probe predictions and targets do not align")
    errors = [int(value) - int(truth) for value, truth in zip(predicted, target, strict=True)]
    return {
        "mae_k": sum(abs(value) for value in errors) / len(errors),
        "overrun_rate": sum(value > 0 for value in errors) / len(errors),
        "underrun_rate": sum(value < 0 for value in errors) / len(errors),
    }


def _interpret(probes: Mapping[str, Any]) -> dict[str, Any]:
    linear = probes["boundary"]["linear"]["test"]
    mlp = probes["boundary"]["mlp"]["test"]
    if linear["macro_f1"] >= 0.65:
        conclusion = "boundary_information_linearly_observable"
    elif mlp["macro_f1"] >= 0.65:
        conclusion = "boundary_information_nonlinearly_observable"
    else:
        conclusion = "boundary_observability_not_demonstrated"
    return {
        "conclusion": conclusion,
        "threshold_is_predeclared_diagnostic_not_production_gate": 0.65,
        "linear_boundary_macro_f1": linear["macro_f1"],
        "mlp_boundary_macro_f1": mlp["macro_f1"],
    }


def _read_records(root: Path, split: str) -> list[Mapping[str, Any]]:
    path = root.expanduser().resolve() / f"{split}.jsonl"
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise M0MobileError(f"{path} contains a non-object row")
                records.append(value)
    if not records:
        raise M0MobileError(f"{split} waypoint split is empty")
    return records


def _plot_transition_curves(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, (transition, curve) in zip(axes.flat, report["transition_curves"].items(), strict=True):
        rows = curve["rows"]
        axis.scatter(
            [row["signed_time_s"] for row in rows],
            [row["predicted"] for row in rows],
            s=9,
            alpha=0.55,
            label="MLP predicted",
        )
        axis.scatter(
            [row["signed_time_s"] for row in rows],
            [row["target"] for row in rows],
            s=7,
            alpha=0.3,
            label="target",
        )
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_title(transition)
        axis.set_xlabel("signed time to boundary (s)")
        axis.set_ylabel("0 interior / 1 before / 2 after")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "checkpoint_id": report["export"]["checkpoint_id"],
        "selected_rows": report["dataset"]["selected_rows"],
        "interpretation": report["interpretation"],
        "test_metrics": {
            task: {architecture: values["test"] for architecture, values in probes.items()}
            for task, probes in report["probes"].items()
        },
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.epochs <= 0:
        raise M0MobileError("observability batch size and epochs must be positive")
    if any(value <= 0 for value in (args.train_rows, args.val_rows, args.test_rows)):
        raise M0MobileError("observability split row counts must be positive")
    for path, name in ((args.export_dir, "export"), (args.dataset_root, "dataset")):
        if not path.expanduser().resolve().is_dir():
            raise M0MobileError(f"observability {name} root does not exist")
    if args.output_dir.expanduser().resolve().exists():
        raise M0MobileError("observability output directory already exists")


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status": status}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
