#!/usr/bin/env python3
"""Jointly fine-tune Qwen Pass 1/2 and the two ConveyorVLA AL0 DiTs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from conveyor_bench.conveyorvla.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    M0MobileError,
    load_m0_mobile_config,
    resolve_model_root,
)
from conveyor_bench.conveyorvla.dit import (  # noqa: E402
    DOMAIN_ACTION_REINITIALIZED_KEYS,
    M0DiTActionHead,
    transfer_conveyorvla_action_trunk,
)
from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    ConveyorVLAAL0HierarchicalDataset,
)
from conveyor_bench.conveyorvla.policy import (  # noqa: E402
    ConveyorVLAAL0TwoPassPolicy,
    Qwen3VLInterface,
    m0_dit_config,
    transfer_qwen_checkpoint_weights,
)
from conveyor_bench.conveyorvla.subtasks import (  # noqa: E402
    MANIPULATION_ACTION_DIM,
    NAVIGATION_ACTION_DIM,
)
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    DEFAULT_TEMPORAL_CONFIG_PATH,
    build_temporal_policy_config,
    load_temporal_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--initial-action-checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG_PATH
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--save-interval-steps", type=int, default=500)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subtask-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--vlm-learning-rate", type=float, default=2e-6)
    parser.add_argument("--lm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--dit-learning-rate", type=float, default=2e-5)
    parser.add_argument("--boundary-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-8)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--repeated-diffusion-steps", type=int, default=1)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    output_reserved = False
    try:
        _validate_args(args)
        output = args.output_dir.expanduser().resolve()
        _reserve_output(accelerator, output, args.resume_from)
        output_reserved = True
        set_seed(args.seed, device_specific=True)
        config = build_temporal_policy_config(
            load_m0_mobile_config(args.config),
            load_temporal_config(args.temporal_config),
        )
        model_root = resolve_model_root(config, args.model_root)
        model, transfer_report = _build_model(config, model_root, args)
        train_dataset = ConveyorVLAAL0HierarchicalDataset(
            args.hierarchy_root,
            config,
            split="train",
            component="joint",
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(train_dataset.sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=list,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
        optimizer, parameter_report = _optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _schedule(args.max_steps, args.warmup_steps),
        )
        if accelerator.is_main_process:
            model.qwen_vl_interface.processor.save_pretrained(output / "processor")
            _write_json_atomic(
                output / "resolved_run.json",
                {
                    "schema_version": "conveyor-vla-al0-seen-two-pass-run-1",
                    "status": "initializing",
                    "hierarchy_root": str(args.hierarchy_root.expanduser().resolve()),
                    "hierarchy_manifest_sha256": _sha256(
                        args.hierarchy_root.expanduser().resolve() / "manifest.json"
                    ),
                    "initial_action_checkpoint": str(
                        args.initial_action_checkpoint.expanduser().resolve()
                    ),
                    "initial_action_checkpoint_sha256": _sha256(
                        args.initial_action_checkpoint.expanduser().resolve()
                    ),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "world_size": accelerator.num_processes,
                    "mixed_precision": accelerator.mixed_precision,
                    "max_steps": args.max_steps,
                    "batch_size_per_process": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": (
                        args.batch_size
                        * args.gradient_accumulation_steps
                        * accelerator.num_processes
                    ),
                    "train_rows": len(train_dataset),
                    "train_phase_counts": train_dataset.phase_counts,
                    "parameter_groups": parameter_report,
                    "transfer": transfer_report,
                    "arguments": vars(args),
                },
            )
        accelerator.wait_for_everyone()
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        global_step = 0
        if args.resume_from is not None:
            resume = args.resume_from.expanduser().resolve()
            accelerator.load_state(resume)
            global_step = _checkpoint_step(resume)
        _set_run_status(accelerator, output, "running", global_step)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_metrics: dict[str, float] = {}
        while global_step < args.max_steps:
            for examples in train_loader:
                with accelerator.accumulate(model):
                    subtask = model(examples, objective="subtask")
                    subtask_loss = subtask["subtask_loss"]
                    if not isinstance(subtask_loss, torch.Tensor):
                        raise M0MobileError("subtask loss is not a tensor")
                    _finite(subtask_loss, "subtask loss")
                    accelerator.backward(args.subtask_loss_weight * subtask_loss)

                    action = model(examples, objective="action")
                    action_loss = action["action_loss"]
                    if not isinstance(action_loss, torch.Tensor):
                        raise M0MobileError("action loss is not a tensor")
                    _finite(action_loss, "action loss")
                    accelerator.backward(args.action_loss_weight * action_loss)

                    gradient_norm = torch.tensor(float("nan"), device=action_loss.device)
                    if accelerator.sync_gradients:
                        gradient_norm = accelerator.clip_grad_norm_(
                            model.parameters(), args.max_gradient_norm
                        )
                        _finite(gradient_norm, "gradient norm")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue
                global_step += 1
                last_metrics = {
                    "subtask_loss": _distributed_mean(accelerator, subtask_loss),
                    "action_loss": _distributed_mean(accelerator, action_loss),
                    "gradient_norm": _distributed_mean(accelerator, gradient_norm),
                }
                if global_step == 1 or global_step % args.log_interval_steps == 0:
                    _event(
                        accelerator,
                        output,
                        "train_step",
                        step=global_step,
                        **last_metrics,
                        learning_rates=[group["lr"] for group in optimizer.param_groups],
                    )
                if global_step % args.save_interval_steps == 0:
                    _save_checkpoint(accelerator, output, global_step)
                if global_step >= args.max_steps:
                    break
        _save_checkpoint(accelerator, output, global_step)
        _set_run_status(accelerator, output, "complete", global_step, last_metrics)
        return 0
    except Exception as error:
        if accelerator.is_main_process and output_reserved:
            _event(accelerator, args.output_dir.expanduser().resolve(), "failed", error=str(error))
        raise


def _build_model(
    config: dict[str, Any],
    model_root: Path,
    args: argparse.Namespace,
) -> tuple[ConveyorVLAAL0TwoPassPolicy, dict[str, Any]]:
    qwen = Qwen3VLInterface.from_local(
        model_root / config["vlm"]["relative_path"],
        checkpoint_vocab_size=config["vlm"]["checkpoint_vocab_size"],
        dtype=torch.bfloat16,
        attention_implementation=args.attention_implementation,
    )
    upstream = (model_root / config["checkpoint_transfer"]["relative_path"]).resolve()
    qwen_report = transfer_qwen_checkpoint_weights(qwen, upstream)
    initial = load_file(
        args.initial_action_checkpoint.expanduser().resolve(), device="cpu"
    )
    heads = []
    head_reports = []
    for action_dim in (NAVIGATION_ACTION_DIM, MANIPULATION_ACTION_DIM):
        head_config = copy.deepcopy(config)
        head_config["action_model"]["action_dim"] = action_dim
        head = M0DiTActionHead(m0_dit_config(head_config))
        report = transfer_conveyorvla_action_trunk(head, initial)
        heads.append(head)
        head_reports.append(
            {
                "action_dim": action_dim,
                "loaded_tensors": len(report.loaded_keys),
                "reinitialized_keys": list(report.reinitialized_keys),
            }
        )
    model = ConveyorVLAAL0TwoPassPolicy(
        qwen,
        heads[0],
        heads[1],
        temporal_history_span_s=float(config["data"]["history_span_s"]),
        repeated_diffusion_steps=args.repeated_diffusion_steps,
    )
    model.enable_full_finetuning()
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise M0MobileError("full fine-tuning contract left frozen parameters")
    return model, {
        "qwen_loaded_tensors": qwen_report.loaded_tensors,
        "action_heads": head_reports,
        "upstream_checkpoint_sha256": _sha256(upstream),
    }


def _optimizer(
    model: ConveyorVLAAL0TwoPassPolicy,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    lm_suffixes = ("embed_tokens.weight", "lm_head.weight")
    qwen_lm = []
    qwen_core = []
    for name, parameter in model.qwen_vl_interface.named_parameters():
        (qwen_lm if name.endswith(lm_suffixes) else qwen_core).append(parameter)
    groups: list[tuple[str, list[torch.nn.Parameter], float]] = [
        ("vlm_core", qwen_core, args.vlm_learning_rate),
        ("vlm_embeddings_lm_head", qwen_lm, args.lm_learning_rate),
    ]
    for name, head in (
        ("navigation", model.navigation_model),
        ("manipulation", model.manipulation_model),
    ):
        boundary = []
        core = []
        for parameter_name, parameter in head.named_parameters():
            (boundary if parameter_name in DOMAIN_ACTION_REINITIALIZED_KEYS else core).append(
                parameter
            )
        groups.extend(
            [
                (f"{name}_dit_core", core, args.dit_learning_rate),
                (f"{name}_dit_boundary", boundary, args.boundary_learning_rate),
            ]
        )
    flat = [parameter for _name, parameters, _lr in groups for parameter in parameters]
    if not flat or len({id(parameter) for parameter in flat}) != len(flat):
        raise M0MobileError("optimizer parameter groups are empty or overlap")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if {id(parameter) for parameter in flat} != expected:
        raise M0MobileError("optimizer does not cover every trainable parameter exactly once")
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "lr": learning_rate, "name": name}
            for name, parameters, learning_rate in groups
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    report = [
        {
            "name": name,
            "learning_rate": learning_rate,
            "parameter_tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
        }
        for name, parameters, learning_rate in groups
    ]
    return optimizer, report


def _schedule(max_steps: int, warmup_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return scale


def _reserve_output(
    accelerator: Accelerator,
    output: Path,
    resume_from: Path | None,
) -> None:
    if accelerator.is_main_process:
        if resume_from is None:
            output.mkdir(parents=True, exist_ok=False)
        elif not output.is_dir() or not resume_from.expanduser().resolve().is_dir():
            raise M0MobileError("resume output and checkpoint directories must exist")
    accelerator.wait_for_everyone()


def _save_checkpoint(accelerator: Accelerator, output: Path, step: int) -> None:
    checkpoint = output / "checkpoints" / f"step_{step:06d}"
    accelerator.save_state(checkpoint, safe_serialization=True)
    if accelerator.is_main_process:
        _write_json_atomic(checkpoint / "trainer_state.json", {"global_step": step})
        _event(accelerator, output, "checkpoint", step=step, path=str(checkpoint))
    accelerator.wait_for_everyone()


def _checkpoint_step(path: Path) -> int:
    try:
        value = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        step = int(value["global_step"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise M0MobileError(f"invalid resume checkpoint {path}: {error}") from error
    if step < 0:
        raise M0MobileError("resume global step cannot be negative")
    return step


def _set_run_status(
    accelerator: Accelerator,
    output: Path,
    status: str,
    step: int,
    metrics: Mapping[str, float] | None = None,
) -> None:
    if accelerator.is_main_process:
        _write_json_atomic(
            output / "run_state.json",
            {
                "schema_version": "conveyor-vla-al0-seen-two-pass-state-1",
                "status": status,
                "global_step": step,
                "metrics": dict(metrics or {}),
            },
        )
        _event(accelerator, output, "status", status=status, step=step)
    accelerator.wait_for_everyone()


def _event(
    accelerator: Accelerator,
    output: Path,
    event: str,
    **values: Any,
) -> None:
    if not accelerator.is_main_process:
        return
    output.mkdir(parents=True, exist_ok=True)
    payload = {"event": event, **values}
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()


def _distributed_mean(accelerator: Accelerator, value: torch.Tensor) -> float:
    gathered = accelerator.gather(value.detach().float().reshape(1))
    return float(gathered.mean().cpu())


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all()):
        raise M0MobileError(f"{name} is not finite")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    positive_ints: Iterable[tuple[str, int]] = (
        ("max steps", args.max_steps),
        ("batch size", args.batch_size),
        ("gradient accumulation steps", args.gradient_accumulation_steps),
        ("save interval", args.save_interval_steps),
        ("log interval", args.log_interval_steps),
        ("repeated diffusion steps", args.repeated_diffusion_steps),
    )
    if any(value <= 0 for _name, value in positive_ints):
        raise M0MobileError("training step, batch, interval, and repeat values must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise M0MobileError("warmup steps must be within [0, max steps)")
    if args.num_workers < 0:
        raise M0MobileError("num workers cannot be negative")
    floats = (
        args.subtask_loss_weight,
        args.action_loss_weight,
        args.vlm_learning_rate,
        args.lm_learning_rate,
        args.dit_learning_rate,
        args.boundary_learning_rate,
        args.max_gradient_norm,
    )
    if any(not math.isfinite(value) or value <= 0 for value in floats):
        raise M0MobileError("loss weights, learning rates, and gradient norm must be positive")
    if not args.hierarchy_root.expanduser().resolve().is_dir():
        raise M0MobileError("hierarchy root does not exist")
    if not args.initial_action_checkpoint.expanduser().resolve().is_file():
        raise M0MobileError("initial action checkpoint does not exist")


if __name__ == "__main__":
    raise SystemExit(main())
