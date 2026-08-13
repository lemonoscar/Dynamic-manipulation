#!/usr/bin/env python3
"""Train ConveyorVLA AL0 with frozen local Qwen3-VL features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from torch.nn.parallel import DistributedDataParallel  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler  # noqa: E402

from conveyor_bench.conveyorvla.dataset import M0MobileDataset  # noqa: E402
from conveyor_bench.conveyorvla.lerobot_v3 import (  # noqa: E402
    ConveyorVLAAL0LeRobotDataset,
)
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_TEMPORAL_CONFIG_PATH,
    build_temporal_policy_config,
    load_temporal_config,
)
from conveyor_bench.conveyorvla.dit import (  # noqa: E402
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
)
from conveyor_bench.conveyorvla.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MODEL_FAMILY,
    MODEL_NAME,
    MODEL_VARIANT,
    M0MobileError,
    load_m0_mobile_config,
    resolve_model_root,
)
from conveyor_bench.conveyorvla.policy import (  # noqa: E402
    ConveyorVLAAL0Policy,
    ConveyorVLAAL0TemporalPolicy,
    Qwen3VLInterface,
    m0_dit_config,
    transfer_robocasa_policy_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--episode-root",
        action="append",
        type=Path,
        help="episode directory containing exports/m0_mobile.jsonl (repeatable)",
    )
    source.add_argument(
        "--lerobot-root",
        type=Path,
        help="official LeRobot v3 dataset produced by the AL0 converter",
    )
    parser.add_argument("--state-statistics", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=DEFAULT_TEMPORAL_CONFIG_PATH,
    )
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--initial-action-checkpoint",
        type=Path,
        help="Continue action-head adaptation from a strict safetensors checkpoint.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size-per-device", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-interval-steps", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--action-scale",
        nargs=ACTION_DIM,
        type=float,
        metavar="SCALE",
        help="Override the 10 positive physical action-normalization scales.",
    )
    parser.add_argument("--allow-fixed-base", action="store_true")
    speed_filter = parser.add_mutually_exclusive_group()
    speed_filter.add_argument("--all-belt-speeds", action="store_true")
    speed_filter.add_argument(
        "--belt-speed",
        type=float,
        help="Require this exact non-negative belt speed in every record.",
    )
    parser.add_argument(
        "--task-type",
        action="append",
        choices=("stationary_sort", "dynamic_sort"),
        help="Accepted source task type; repeat to allow both.",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    return parser


def _positive(value: int | None, fallback: int, name: str) -> int:
    resolved = fallback if value is None else value
    if isinstance(resolved, bool) or resolved <= 0:
        raise M0MobileError(f"{name} must be positive")
    return resolved


def _nonnegative(value: int | None, fallback: int, name: str) -> int:
    resolved = fallback if value is None else value
    if isinstance(resolved, bool) or resolved < 0:
        raise M0MobileError(f"{name} must be non-negative")
    return resolved


def _distributed_device() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise M0MobileError(f"{MODEL_NAME} training requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise M0MobileError("LOCAL_RANK exceeds visible CUDA devices")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device("cuda", local_rank), rank, local_rank, world_size


def _checkpoint_path(config: dict, root: Path) -> Path:
    path = (root / config["checkpoint_transfer"]["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise M0MobileError("checkpoint path escapes model root") from error
    if not path.is_file():
        raise M0MobileError(f"checkpoint does not exist: {path}")
    return path


def _reserve_output(path: Path, rank: int, world_size: int) -> Path:
    output = path.expanduser().resolve()
    exists = torch.tensor(
        [int(output.exists())],
        device=torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))),
    )
    if world_size > 1:
        dist.broadcast(exists, src=0)
    if exists.item():
        raise M0MobileError(f"output directory already exists: {output}")
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
        dist.barrier()
    return output


def _datasets(args: argparse.Namespace, config: dict) -> tuple[Any, list[dict]]:
    if args.lerobot_root is not None:
        if any(
            (
                args.all_belt_speeds,
                args.belt_speed is not None,
                args.task_type is not None,
                args.allow_fixed_base,
            )
        ):
            raise M0MobileError("legacy source filters cannot be used with --lerobot-root")
        dataset = ConveyorVLAAL0LeRobotDataset(args.lerobot_root, config)
        return dataset, [
            {
                "lerobot_root": str(dataset.root),
                "repo_id": dataset.manifest["repo_id"],
                "episodes": dataset.manifest["episode_count"],
                "records": len(dataset),
                "query_fps": dataset.manifest["query_fps"],
                "action_rate_hz": dataset.manifest["action_rate_hz"],
            }
        ]
    if args.state_statistics is None:
        raise M0MobileError("--state-statistics is required with --episode-root")
    initial_filter = config["data"]["initial_training_filter"]
    if args.all_belt_speeds and not args.task_type:
        raise M0MobileError(
            "--all-belt-speeds requires at least one explicit --task-type"
        )
    expected_speed = (
        None
        if args.all_belt_speeds
        else (
            args.belt_speed
            if args.belt_speed is not None
            else initial_filter["belt_speed_mps"]
        )
    )
    expected_task_types = (
        None if args.task_type is None else frozenset(args.task_type)
    )
    datasets = []
    sources = []
    for raw_root in args.episode_root:
        episode_root = raw_root.expanduser().resolve()
        jsonl = episode_root / "exports" / "m0_mobile.jsonl"
        dataset = M0MobileDataset(
            jsonl,
            episode_root,
            args.state_statistics,
            config=config,
            allow_fixed_base=args.allow_fixed_base,
            expected_belt_speed_mps=expected_speed,
            expected_task_types=expected_task_types,
        )
        datasets.append(dataset)
        sources.append(
            {
                "episode_root": str(episode_root),
                "jsonl": str(jsonl),
                "records": len(dataset),
                "expected_belt_speed_mps": expected_speed,
                "expected_task_types": (
                    None
                    if expected_task_types is None
                    else sorted(expected_task_types)
                ),
            }
        )
    return ConcatDataset(datasets), sources


def _optimizer(
    action_model: M0DiTActionHead,
    config: dict,
) -> torch.optim.AdamW:
    optimizer_config = config["training"]["optimizer"]
    boundary = []
    core = []
    for name, parameter in action_model.named_parameters():
        (boundary if name in GO2_X5_REINITIALIZED_ACTION_KEYS else core).append(
            parameter
        )
    if len(boundary) != len(GO2_X5_REINITIALIZED_ACTION_KEYS) or not core:
        raise M0MobileError("action optimizer parameter groups are incomplete")
    return torch.optim.AdamW(
        [
            {
                "params": core,
                "lr": optimizer_config["action_core_learning_rate"],
            },
            {
                "params": boundary,
                "lr": optimizer_config["boundary_learning_rate"],
            },
        ],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["epsilon"],
        weight_decay=optimizer_config["weight_decay"],
    )


def _scheduler(
    optimizer: torch.optim.Optimizer,
    max_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        denominator = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _event(rank: int, event: str, **values) -> None:
    print(json.dumps({"rank": rank, "event": event, **values}, sort_keys=True), flush=True)


def _save_action_model(
    action_model: M0DiTActionHead,
    output: Path,
    filename: str = "action_model_final.safetensors",
) -> str:
    destination = output / filename
    temporary = output / f".{filename}.tmp"
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in action_model.state_dict().items()
    }
    save_file(
        tensors,
        temporary,
        metadata={
            "schema_version": "conveyor-bench-m0-mobile-action-checkpoint-1",
            "model_family": MODEL_FAMILY,
            "model_variant": MODEL_VARIANT,
            "model_name": MODEL_NAME,
        },
    )
    os.replace(temporary, destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_state_statistics(source: Path | Mapping[str, Any], output: Path) -> str:
    """Copy the exact normalizer input beside the deployable action head."""

    payload = (
        json.dumps(source, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if isinstance(source, Mapping)
        else source.expanduser().resolve().read_bytes()
    )
    digest = hashlib.sha256(payload).hexdigest()
    destination = output / "state_statistics.json"
    temporary = output / ".state_statistics.json.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return digest


def _temporal_training_config(base: dict, temporal: dict) -> dict:
    """Reuse released artifacts/training settings with the temporal AL0 contract."""

    return build_temporal_policy_config(base, temporal)


def _apply_dataset_temporal_history(
    config: dict,
    source: dict,
    manifest: Mapping[str, Any],
) -> None:
    """Preserve the image interval actually used by a LeRobot derivative."""

    history_offsets = manifest.get("history_offsets_model_ticks")
    history_span_s = manifest.get("history_span_s")
    if history_offsets is None and history_span_s is None:
        return
    if (
        not isinstance(history_offsets, list)
        or len(history_offsets) != 2
        or history_offsets[1] != 0
        or not isinstance(history_offsets[0], int)
        or history_offsets[0] >= 0
        or isinstance(history_span_s, bool)
        or not isinstance(history_span_s, (int, float))
        or not math.isfinite(float(history_span_s))
        or float(history_span_s) <= 0.0
    ):
        raise M0MobileError("LeRobot temporal history metadata is invalid")
    config["data"]["history_offsets_model_ticks"] = history_offsets
    config["data"]["history_span_s"] = float(history_span_s)
    source["history_offsets_model_ticks"] = history_offsets
    source["history_span_s"] = float(history_span_s)


def _apply_action_scale(config: dict, values: list[float] | None) -> None:
    """Record an explicit dataset-specific physical action scale."""

    if values is None:
        return
    if len(values) != ACTION_DIM or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise M0MobileError(f"--action-scale requires {ACTION_DIM} positive finite values")
    config["normalization"]["action"]["scale"] = [float(value) for value in values]


def _load_initial_action_checkpoint(
    action_model: M0DiTActionHead,
    source: Path,
) -> str:
    path = source.expanduser().resolve()
    if not path.is_file():
        raise M0MobileError(f"initial action checkpoint does not exist: {path}")
    tensors = load_file(path, device="cpu")
    expected = action_model.state_dict()
    if set(tensors) != set(expected):
        raise M0MobileError("initial action checkpoint keys do not match the action model")
    bad_shapes = [
        key
        for key, value in tensors.items()
        if value.shape != expected[key].shape
    ]
    if bad_shapes:
        raise M0MobileError(
            "initial action checkpoint shapes do not match: "
            + ", ".join(sorted(bad_shapes))
        )
    action_model.load_state_dict(tensors, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device, rank, local_rank, world_size = _distributed_device()
    try:
        base_config = load_m0_mobile_config(args.config)
        temporal_training = args.lerobot_root is not None
        config = (
            _temporal_training_config(
                base_config,
                load_temporal_config(args.temporal_config),
            )
            if temporal_training
            else base_config
        )
        if args.action_scale is not None and not temporal_training:
            raise M0MobileError("--action-scale is only supported with --lerobot-root")
        _apply_action_scale(config, args.action_scale)
        training = config["training"]
        max_steps = _positive(args.max_steps, training["max_train_steps"], "max_steps")
        batch_size = _positive(
            args.batch_size_per_device,
            training["smoke_batch_size_per_device"],
            "batch_size_per_device",
        )
        accumulation = _positive(
            args.gradient_accumulation_steps,
            training["gradient_accumulation_steps"],
            "gradient_accumulation_steps",
        )
        warmup = _nonnegative(args.warmup_steps, training["warmup_steps"], "warmup_steps")
        save_interval = _positive(
            args.save_interval_steps,
            training["save_interval_steps"],
            "save_interval_steps",
        )
        workers = _nonnegative(args.num_workers, training["dataloader_workers"], "num_workers")
        output = _reserve_output(args.output_dir, rank, world_size)
        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)

        dataset, sources = _datasets(args, config)
        if temporal_training:
            _apply_dataset_temporal_history(config, sources[0], dataset.manifest)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=workers,
            collate_fn=list,
            persistent_workers=workers > 0,
        )
        _event(rank, "data_ready", records=len(dataset), batches=len(loader))

        root = resolve_model_root(config, args.model_root)
        qwen = Qwen3VLInterface.from_local(
            root / config["vlm"]["relative_path"],
            checkpoint_vocab_size=config["vlm"]["checkpoint_vocab_size"],
            dtype=torch.bfloat16,
            attention_implementation=args.attention_implementation,
        )
        policy_class = (
            ConveyorVLAAL0TemporalPolicy
            if temporal_training
            else ConveyorVLAAL0Policy
        )
        policy_kwargs = {
            "repeated_diffusion_steps": training["repeated_diffusion_steps"],
        }
        if temporal_training:
            policy_kwargs["temporal_history_span_s"] = config["data"][
                "history_span_s"
            ]
        policy = policy_class(
            qwen,
            M0DiTActionHead(m0_dit_config(config)),
            **policy_kwargs,
        )
        transfer = transfer_robocasa_policy_weights(
            policy, _checkpoint_path(config, root)
        )
        initial_action_sha256 = (
            _load_initial_action_checkpoint(
                policy.action_model, args.initial_action_checkpoint
            )
            if args.initial_action_checkpoint is not None
            else None
        )
        policy.freeze_qwen()
        policy.qwen_vl_interface.to(device)
        policy.action_model.to(device)
        optimizer = _optimizer(policy.action_model, config)
        scheduler = _scheduler(optimizer, max_steps, warmup)
        raw_action_model = policy.action_model
        if world_size > 1:
            policy.action_model = DistributedDataParallel(
                raw_action_model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        _event(rank, "policy_ready")

        train_step = 0
        micro_step = 0
        epoch = 0
        accumulated_loss = 0.0
        final_loss = float("nan")
        final_gradient_norm = float("nan")
        periodic_checkpoints = []
        while train_step < max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for examples in loader:
                synchronize = (micro_step + 1) % accumulation == 0
                context = (
                    policy.action_model.no_sync()
                    if world_size > 1 and not synchronize
                    else nullcontext()
                )
                with context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = policy(examples)["action_loss"]
                    if not torch.isfinite(loss):
                        raise M0MobileError("training loss is not finite")
                    (loss / accumulation).backward()
                accumulated_loss += float(loss.detach().item())
                micro_step += 1
                if not synchronize:
                    continue
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    raw_action_model.parameters(),
                    training["max_gradient_norm"],
                )
                if not torch.isfinite(gradient_norm):
                    raise M0MobileError("gradient norm is not finite")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                train_step += 1
                mean_loss = torch.tensor(
                    [accumulated_loss / accumulation],
                    device=device,
                    dtype=torch.float32,
                )
                if world_size > 1:
                    dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
                    mean_loss /= world_size
                accumulated_loss = 0.0
                final_loss = float(mean_loss.item())
                final_gradient_norm = float(gradient_norm.item())
                if train_step == 1 or train_step % training["log_interval_steps"] == 0:
                    _event(
                        rank,
                        "train_step",
                        step=train_step,
                        loss=float(mean_loss.item()),
                        gradient_norm=float(gradient_norm.item()),
                        learning_rates=[group["lr"] for group in optimizer.param_groups],
                    )
                if train_step % save_interval == 0 and train_step < max_steps:
                    if world_size > 1:
                        dist.barrier()
                    if rank == 0:
                        filename = f"action_model_step_{train_step:06d}.safetensors"
                        digest = _save_action_model(
                            raw_action_model,
                            output,
                            filename,
                        )
                        periodic_checkpoints.append(
                            {
                                "step": train_step,
                                "relative_path": filename,
                                "sha256": digest,
                            }
                        )
                        _event(
                            rank,
                            "checkpoint",
                            step=train_step,
                            relative_path=filename,
                            sha256=digest,
                        )
                    if world_size > 1:
                        dist.barrier()
                if train_step >= max_steps:
                    break
            epoch += 1

        if world_size > 1:
            dist.barrier()
        if rank == 0:
            action_sha256 = _save_action_model(raw_action_model, output)
            statistics_source = (
                dataset.state_statistics
                if temporal_training
                else args.state_statistics
            )
            assert statistics_source is not None
            statistics_sha256 = _publish_state_statistics(statistics_source, output)
            config_payload = (
                json.dumps(config, indent=2, sort_keys=True).encode("utf-8")
                + b"\n"
            )
            config_sha256 = hashlib.sha256(config_payload).hexdigest()
            report = {
                "schema_version": "conveyor-bench-m0-mobile-training-report-1",
                "model_identity": {
                    "family": MODEL_FAMILY,
                    "variant": MODEL_VARIANT,
                    "name": MODEL_NAME,
                },
                "ok": True,
                "world_size": world_size,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "dataset_records": len(dataset),
                "data_format": "lerobot_v3" if temporal_training else "raw_jsonl",
                "sources": sources,
                "max_steps": max_steps,
                "batch_size_per_device": batch_size,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": batch_size * world_size * accumulation,
                "final_loss": final_loss,
                "final_gradient_norm": final_gradient_norm,
                "peak_allocated_mib_rank0": round(
                    torch.cuda.max_memory_allocated(device) / 2**20, 2
                ),
                "peak_reserved_mib_rank0": round(
                    torch.cuda.max_memory_reserved(device) / 2**20, 2
                ),
                "attention_implementation": args.attention_implementation,
                "checkpoint_transfer": {
                    "loaded_qwen_tensors": transfer.loaded_qwen_tensors,
                    "loaded_action_tensors": transfer.loaded_action_tensors,
                    "reinitialized_tensors": len(transfer.reinitialized_keys),
                },
                "state_statistics_sha256": statistics_sha256,
                "state_statistics_relative_path": "state_statistics.json",
                "action_model_sha256": action_sha256,
                "initial_action_model_sha256": initial_action_sha256,
                "periodic_checkpoints": periodic_checkpoints,
                "config_relative_path": "conveyorvla_al0_config.json",
                "config_sha256": config_sha256,
            }
            (output / "training_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output / "conveyorvla_al0_config.json").write_bytes(config_payload)
            _event(rank, "complete", report=report)
        if world_size > 1:
            dist.barrier()
        return 0
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        _event(rank, "failed", error=str(error))
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
