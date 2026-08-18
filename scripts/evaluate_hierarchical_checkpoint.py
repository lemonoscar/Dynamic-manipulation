#!/usr/bin/env python3
"""Evaluate one hierarchical ConveyorVLA checkpoint on balanced held-out rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedType, gather_object, set_seed  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from conveyor_bench.conveyorvla.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_m0_mobile_config,
    resolve_model_root,
)
from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    ConveyorVLAAL0HierarchicalDataset,
)
from conveyor_bench.conveyorvla.subtasks import (  # noqa: E402
    PHASE_ORDER,
    Phase,
    parse_subtask_solution,
)
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    DEFAULT_TEMPORAL_CONFIG_PATH,
    build_temporal_policy_config,
    load_temporal_config,
)
from train_hierarchical import _build_model, _checkpoint_step, _optimizer, _schedule  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--initial-action-checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--samples-per-phase", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_phase <= 0:
        raise ValueError("samples per phase must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    set_seed(args.seed, device_specific=True)
    config = build_temporal_policy_config(
        load_m0_mobile_config(args.config),
        load_temporal_config(args.temporal_config),
    )
    model_root = resolve_model_root(config, args.model_root)
    training_args = SimpleNamespace(
        attention_implementation=args.attention_implementation,
        initial_action_checkpoint=args.initial_action_checkpoint,
        repeated_diffusion_steps=1,
        vlm_learning_rate=2e-6,
        lm_learning_rate=1e-5,
        dit_learning_rate=2e-5,
        boundary_learning_rate=1e-4,
        weight_decay=1e-8,
    )
    model, _transfer = _build_model(config, model_root, training_args)
    dataset = ConveyorVLAAL0HierarchicalDataset(
        args.hierarchy_root,
        config,
        split=args.split,
        component="joint",
    )
    indices = _balanced_indices(dataset, args.samples_per_phase)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list,
        pin_memory=True,
    )
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        optimizer, _groups = _optimizer(model, training_args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _schedule(args.max_steps, args.warmup_steps),
        )
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        accelerator.load_state(
            checkpoint,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
    else:
        # A consolidated checkpoint contains model weights but deliberately no
        # optimizer or scheduler state. Register only objects that can be
        # restored so Accelerate does not look for absent optimizer files.
        model, loader = accelerator.prepare(model, loader)
        accelerator.load_state(checkpoint)
    model.eval()

    phase_count = len(PHASE_ORDER)
    counts = torch.zeros(phase_count, device=accelerator.device, dtype=torch.float64)
    subtask_sums = torch.zeros_like(counts)
    action_sums = torch.zeros_like(counts)
    confusion = torch.zeros(
        (phase_count, phase_count + 1),
        device=accelerator.device,
        dtype=torch.int64,
    )
    generation_examples: list[dict[str, str]] = []
    started = time.monotonic()
    unwrapped = accelerator.unwrap_model(model)
    with torch.inference_mode():
        for examples in loader:
            prompt = str(examples[0]["lang"])
            if "Completed subtasks" in prompt or "Previous model prediction" in prompt:
                raise ValueError(
                    "empty-history evaluation received an externally supplied history prompt"
                )
            expected = Phase(int(examples[0]["phase_id"]))
            phase_index = PHASE_ORDER.index(expected)
            subtask = model(examples, objective="subtask")["subtask_loss"]
            action = model(examples, objective="action")["action_loss"]
            counts[phase_index] += 1
            subtask_sums[phase_index] += subtask.detach().double()
            action_sums[phase_index] += action.detach().double()
            generated = unwrapped.qwen_vl_interface.generate_temporal_subtask_texts(
                [examples[0]["video"]],
                [examples[0]["lang"]],
                history_span_s=unwrapped.temporal_history_span_s,
            )[0]
            try:
                predicted = parse_subtask_solution(generated).phase
                predicted_index = PHASE_ORDER.index(predicted)
                error_text = ""
            except ValueError as error:
                predicted_index = phase_count
                error_text = str(error)
            if len(generation_examples) < phase_count:
                generation_examples.append(
                    {
                        "expected": expected.name,
                        "generated": generated,
                        "parse_error": error_text,
                    }
                )
            confusion[phase_index, predicted_index] += 1

    counts = accelerator.reduce(counts, reduction="sum")
    subtask_sums = accelerator.reduce(subtask_sums, reduction="sum")
    action_sums = accelerator.reduce(action_sums, reduction="sum")
    confusion = accelerator.reduce(confusion, reduction="sum")
    generation_examples = gather_object(generation_examples)
    if accelerator.is_main_process:
        phase_metrics = {}
        for index, phase in enumerate(PHASE_ORDER):
            count = int(counts[index].item())
            phase_metrics[phase.name] = {
                "samples": count,
                "subtask_loss": float(subtask_sums[index].item() / count),
                "action_loss": float(action_sums[index].item() / count),
                "generation_correct": int(confusion[index, index].item()),
                "generation_invalid": int(confusion[index, phase_count].item()),
            }
        total = int(counts.sum().item())
        correct = int(sum(confusion[index, index].item() for index in range(phase_count)))
        invalid = int(confusion[:, phase_count].sum().item())
        generation_distribution = {
            phase.name: int(confusion[:, index].sum().item())
            for index, phase in enumerate(PHASE_ORDER)
        }
        generation_distribution["INVALID"] = invalid
        all_nav_to_source = (
            generation_distribution[Phase.NAV_TO_SOURCE.name] == total
        )
        generation_problems = []
        if invalid:
            generation_problems.append(
                f"{invalid}/{total} generated answers failed the strict parser"
            )
        if all_nav_to_source:
            generation_problems.append(
                "every empty-history answer was NAV_TO_SOURCE"
            )
        phases_without_correct_prediction = [
            phase.name
            for index, phase in enumerate(PHASE_ORDER)
            if int(confusion[index, index].item()) == 0
        ]
        if phases_without_correct_prediction:
            generation_problems.append(
                "no correct empty-history prediction for: "
                + ", ".join(phases_without_correct_prediction)
            )
        report = {
            "schema_version": "conveyor-vla-al0-hierarchical-eval-4",
            "checkpoint": str(checkpoint),
            "checkpoint_step": _checkpoint_step(checkpoint),
            "split": args.split,
            "samples": total,
            "samples_per_phase": args.samples_per_phase,
            "subtask_loss": float(subtask_sums.sum().item() / total),
            "action_loss": float(action_sums.sum().item() / total),
            "generation_accuracy": correct / total,
            "generation_invalid_rate": invalid / total,
            "generation_distribution": generation_distribution,
            "empty_history_all_nav_to_source": all_nav_to_source,
            "empty_history_generation_gate": {
                "ok": not generation_problems,
                "requirements": [
                    "strict_parser_invalid_rate_equals_zero",
                    "not_all_predictions_are_NAV_TO_SOURCE",
                    "at_least_one_correct_prediction_per_canonical_phase",
                ],
                "problems": generation_problems,
            },
            "prompt_history_contract": "empty; no oracle or previous prediction supplied",
            "confusion_columns": [phase.name for phase in PHASE_ORDER] + ["INVALID"],
            "confusion": confusion.cpu().tolist(),
            "generation_examples": generation_examples,
            "phase_metrics": phase_metrics,
            "world_size": accelerator.num_processes,
            "elapsed_seconds": time.monotonic() - started,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    return 0


def _balanced_indices(
    dataset: ConveyorVLAAL0HierarchicalDataset,
    samples_per_phase: int,
) -> list[int]:
    by_phase = {
        phase: [
            index
            for index, annotation in enumerate(dataset.annotations)
            if int(annotation["phase_id"]) == int(phase)
        ][:samples_per_phase]
        for phase in PHASE_ORDER
    }
    if any(len(indices) != samples_per_phase for indices in by_phase.values()):
        raise ValueError("held-out split does not contain enough rows for every phase")
    return [
        index
        for offset in range(samples_per_phase)
        for phase in PHASE_ORDER
        for index in (by_phase[phase][offset],)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
