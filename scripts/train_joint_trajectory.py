#!/usr/bin/env python3
"""Train ConveyorVLA Joint-Trajectory Policy v1 on fresh applied-command data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import train_hierarchical as common  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator, DataLoaderConfiguration  # noqa: E402
from accelerate.utils import DistributedDataParallelKwargs, set_seed  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.joint_trajectory import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    MODEL_CONTRACT_ID,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import (  # noqa: E402
    ConveyorVLAJointTrajectoryDataset,
    audit_joint_trajectory_dataset,
)
from conveyor_bench.conveyorvla.joint_trajectory_model import (  # noqa: E402
    ConveyorVLAJointTrajectoryPolicy,
    JointTrajectoryAuxiliaryHeads,
    JointTrajectoryExpertConfig,
    JointTrajectoryFlowMatchingExpert,
    JointTrajectoryLossConfig,
    JointTrajectoryQwenInterface,
    selective_warmstart,
)
from conveyor_bench.conveyorvla.joint_trajectory_training import (  # noqa: E402
    AccumulationMicroBatchSampler,
    StratifiedJointTrajectoryBatchSampler,
    TrainingStages,
    build_optimizer,
    build_scheduler,
    consolidated_checkpoint_identity,
    load_consolidated_checkpoint,
    load_joint_trajectory_config,
    select_disposable_overfit_episodes,
    set_training_stage,
    validate_global_batch,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "manipulation_navi_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--warmstart-checkpoint", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--micro-batch-per-rank", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-interval-steps", type=int, default=250)
    parser.add_argument("--log-interval-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="allow an explicit short max-steps value for the disposable 12-episode gate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_joint_trajectory_config(args.config)
    _validate_cli(args, config)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
        # All Stage-B parameters are registered before wrapping, then Qwen is
        # frozen during Stage A.  DDP must mark those gradients unused until
        # the scheduled unfreeze instead of waiting for reducer hooks forever.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=DataLoaderConfiguration(
            split_batches=True,
            even_batches=True,
        ),
    )
    validate_global_batch(
        accelerator.num_processes,
        args.micro_batch_per_rank,
        args.gradient_accumulation_steps,
    )
    audit = audit_joint_trajectory_dataset(args.dataset_root)
    if not audit["ok"]:
        raise M0MobileError(
            "joint-trajectory dataset failed its gate: " + "; ".join(audit["problems"])
        )
    dataset = ConveyorVLAJointTrajectoryDataset(args.dataset_root, split="train")
    if args.overfit:
        stages = TrainingStages.for_disposable_overfit(
            len(dataset), max_steps=int(args.max_steps)
        )
    else:
        stages = TrainingStages.from_rows(len(dataset))
    max_steps = stages.total_steps if args.max_steps is None else args.max_steps
    if not args.overfit and max_steps != stages.total_steps:
        raise M0MobileError(
            f"formal run max_steps must equal two data-equivalent epochs ({stages.total_steps})"
        )
    output = args.output_dir.expanduser().resolve()
    resume = _resume_binding(args, config, audit, stages, max_steps)
    common._reserve_output(accelerator, output, args.resume_from)
    set_seed(args.seed, device_specific=True)

    overfit_episode_ids = (
        select_disposable_overfit_episodes(
            dataset.routes,
            dataset.episode_ids,
            dataset.transition_ids,
            dataset.boundary_signed_times,
            dataset.gripper_transitions,
            seed=args.seed,
        )
        if args.overfit
        else ()
    )

    global_sampler = StratifiedJointTrajectoryBatchSampler(
        dataset.routes,
        dataset.episode_ids,
        dataset.transition_ids,
        dataset.boundary_signed_times,
        dataset.progress_buckets,
        dataset.gripper_transitions,
        seed=args.seed,
        batches_per_epoch=stages.equivalent_epoch_steps,
        eligible_episode_ids=overfit_episode_ids or None,
        allow_episode_reuse=args.overfit,
        minimum_distinct_episodes=4 if args.overfit else 56,
    )
    micro_sampler = AccumulationMicroBatchSampler(
        global_sampler,
        world_size=accelerator.num_processes,
        micro_batch_per_rank=args.micro_batch_per_rank,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=micro_sampler,
        num_workers=args.num_workers,
        collate_fn=list,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    model, token_ids = _build_model(config, args.model_root, args.attention_implementation)
    warmstart_report: Mapping[str, Any] | None = None
    warmstart_identity: Mapping[str, Any] | None = None
    if resume is None:
        warmstart_identity = consolidated_checkpoint_identity(args.warmstart_checkpoint)
        source_state = load_consolidated_checkpoint(args.warmstart_checkpoint)
        warmstart_report = selective_warmstart(model, source_state).as_dict()
        del source_state
    # Register every future Stage-B parameter with DDP/ZeRO.  Stage A is
    # applied only after distributed wrapping; otherwise later unfreezing may
    # leave Qwen parameters outside reducer hooks.
    model.enable_full_finetuning()
    optimizer, parameter_groups = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, stages, config)

    if accelerator.is_main_process and resume is None:
        model.qwen.processor.save_pretrained(output / "processor")
        common._write_json_atomic(output / "dataset_audit.json", audit)
        common._write_json_atomic(output / "resolved_policy_config.json", config)
        common._write_json_atomic(output / "warmstart_report.json", warmstart_report or {})
        common._write_json_atomic(
            output / "resolved_run.json",
            _resolved_run(
                args,
                config,
                audit,
                dataset,
                stages,
                max_steps,
                token_ids,
                parameter_groups,
                warmstart_report,
                warmstart_identity,
                overfit_episode_ids,
                accelerator,
            ),
        )
    accelerator.wait_for_everyone()
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )
    global_step = 0
    first_loader = loader
    if resume is not None:
        accelerator.load_state(Path(str(resume["checkpoint"])))
        global_step = int(resume["global_step"])
        global_sampler._epoch = global_step // len(global_sampler)
        completed_in_epoch = global_step % len(global_sampler)
        if completed_in_epoch:
            first_loader = accelerator.skip_first_batches(
                loader,
                completed_in_epoch * args.gradient_accumulation_steps,
            )
    set_training_stage(accelerator.unwrap_model(model), global_step, stages)
    _set_status(accelerator, output, "running", global_step, {})
    last_checkpoint = global_step
    last_metrics: dict[str, float | int | str] = {}
    metric_window: dict[str, tuple[torch.Tensor, int]] = {}
    current_stage = stages.stage(global_step)
    active_loader = first_loader
    optimizer.zero_grad(set_to_none=True)
    optimizer_step_started = time.perf_counter()
    try:
        while global_step < max_steps:
            for examples in active_loader:
                expected_stage = stages.stage(global_step)
                if expected_stage != current_stage:
                    current_stage = set_training_stage(
                        accelerator.unwrap_model(model), global_step, stages
                    )
                    _event(accelerator, output, "training_stage", step=global_step, stage=current_stage)
                with accelerator.accumulate(model):
                    result = model(examples)
                    loss = _tensor(result["loss"], "loss")
                    common._finite(loss, "joint-trajectory loss")
                    _accumulate_metric_window(metric_window, result, len(examples))
                    accelerator.backward(loss)
                    gradient_norm = torch.full(
                        (), float("nan"), device=accelerator.device
                    )
                    if accelerator.sync_gradients:
                        gradient_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            float(config["optimization"]["max_gradient_norm"]),
                        )
                        common._finite(gradient_norm, "joint-trajectory gradient norm")
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if not accelerator.sync_gradients:
                    continue
                global_step += 1
                next_stage = stages.stage(global_step)
                if next_stage != current_stage:
                    current_stage = set_training_stage(
                        accelerator.unwrap_model(model), global_step, stages
                    )
                    _event(
                        accelerator,
                        output,
                        "training_stage",
                        step=global_step,
                        stage=current_stage,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize(accelerator.device)
                elapsed = time.perf_counter() - optimizer_step_started
                peak_memory = (
                    torch.cuda.max_memory_allocated(accelerator.device) / 2**20
                    if torch.cuda.is_available()
                    else 0.0
                )
                last_metrics = _metrics(
                    accelerator,
                    metric_window,
                    gradient_norm,
                    optimizer,
                    elapsed,
                    peak_memory,
                    current_stage,
                )
                metric_window = {}
                if global_step == 1 or global_step % args.log_interval_steps == 0:
                    _event(
                        accelerator,
                        output,
                        "train_step",
                        step=global_step,
                        valid_optimizer_step=True,
                        **last_metrics,
                    )
                    _set_status(accelerator, output, "running", global_step, last_metrics)
                if global_step % args.save_interval_steps == 0:
                    _save_checkpoint(accelerator, output, global_step)
                    last_checkpoint = global_step
                if global_step >= max_steps:
                    break
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                optimizer_step_started = time.perf_counter()
            active_loader = loader
        if last_checkpoint != global_step:
            _save_checkpoint(accelerator, output, global_step)
        _set_status(accelerator, output, "complete", global_step, last_metrics)
        return 0
    except Exception as error:
        _set_status(
            accelerator,
            output,
            "failed",
            global_step,
            {"error": f"{type(error).__name__}: {error}"},
        )
        raise


def _build_model(
    config: Mapping[str, Any],
    model_root: Path,
    attention_implementation: str,
) -> tuple[ConveyorVLAJointTrajectoryPolicy, Mapping[str, Any]]:
    root = model_root.expanduser().resolve()
    qwen = JointTrajectoryQwenInterface.from_local(
        root / str(config["vlm"]["relative_path"]),
        dtype=torch.bfloat16,
        attention_implementation=attention_implementation,
    )
    action = config["action_model"]
    shared = {
        key: action[key]
        for key in (
            "action_horizon",
            "cross_attention_dim",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "dropout",
            "max_seq_len",
            "noise_beta_alpha",
            "noise_beta_beta",
            "noise_s",
            "num_timestep_buckets",
            "num_inference_timesteps",
        )
    }
    navigation = JointTrajectoryFlowMatchingExpert(
        JointTrajectoryExpertConfig(
            action_dim=int(action["navigation_action_dim"]),
            state_dim=0,
            **shared,
        )
    )
    manipulation = JointTrajectoryFlowMatchingExpert(
        JointTrajectoryExpertConfig(
            action_dim=int(action["manipulation_action_dim"]),
            state_dim=int(action["manipulation_state_dim"]),
            **shared,
        )
    )
    loss = config["loss"]
    policy = ConveyorVLAJointTrajectoryPolicy(
        qwen,
        navigation,
        manipulation,
        JointTrajectoryAuxiliaryHeads(
            int(action["cross_attention_dim"]),
            int(config["auxiliary"]["progress_hidden_size"]),
        ),
        max_subtask_tokens=int(config["router"]["max_subtask_tokens"]),
        loss_config=JointTrajectoryLossConfig(
            lambda_answer=float(loss["lambda_answer"]),
            lambda_route=float(loss["lambda_route"]),
            lambda_navigation=float(loss["lambda_navigation"]),
            lambda_manipulation=float(loss["lambda_manipulation"]),
            lambda_boundary=float(loss["lambda_boundary"]),
            lambda_progress=float(loss["lambda_progress"]),
            manipulation_joint_weight=float(loss["manipulation_joint_weight"]),
            manipulation_gripper_weight=float(loss["manipulation_gripper_weight"]),
            repeated_diffusion_steps=int(loss["repeated_diffusion_steps"]),
            boundary_rank_margin=float(loss["boundary_rank_margin"]),
        ),
    )
    ids = policy.router.token_ids
    return policy, {
        "pred_action": ids.pred_action,
        "routes": list(ids.route_ids),
        "subtask_start": ids.subtask_start,
        "subtask_end": ids.subtask_end,
    }


def _validate_cli(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    for name in ("dataset_root", "output_dir"):
        path = getattr(args, name).expanduser().resolve()
        try:
            path.relative_to(PROJECT_ROOT)
        except ValueError:
            pass
        else:
            raise M0MobileError(
                f"{name} must stay outside the Git worktree: {path}"
            )
    if args.resume_from is None and args.warmstart_checkpoint is None:
        raise M0MobileError("a fresh run requires --warmstart-checkpoint")
    if args.resume_from is not None and args.warmstart_checkpoint is not None:
        raise M0MobileError("resume and selective warm-start are mutually exclusive")
    if args.overfit and args.resume_from is not None:
        raise M0MobileError("disposable overfit runs cannot resume")
    if args.overfit and (args.max_steps is None or args.max_steps <= 0):
        raise M0MobileError("disposable overfit requires positive explicit --max-steps")
    if args.num_workers < 0:
        raise M0MobileError("num-workers cannot be negative")
    for name in (
        "micro_batch_per_rank",
        "gradient_accumulation_steps",
        "save_interval_steps",
        "log_interval_steps",
    ):
        if getattr(args, name) <= 0:
            raise M0MobileError(f"{name} must be positive")
    if args.save_interval_steps != int(config["training"]["save_interval_steps"]):
        raise M0MobileError("joint-trajectory checkpoints must be saved every 250 steps")
    if args.resume_from is not None and not args.resume_from.expanduser().resolve().is_dir():
        raise M0MobileError("resume checkpoint directory does not exist")


def _resume_binding(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    stages: TrainingStages,
    max_steps: int,
) -> Mapping[str, Any] | None:
    if args.resume_from is None:
        return None
    checkpoint = args.resume_from.expanduser().resolve()
    manifest_path = checkpoint / "joint_trajectory_checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise M0MobileError("resume checkpoint lacks joint-trajectory manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "run_kind": "formal",
        "model_contract_id": MODEL_CONTRACT_ID,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_manifest_sha256": audit["manifest_sha256"],
        "policy_config_sha256": _sha256(args.config.expanduser().resolve()),
        "stage_a_steps": stages.stage_a_steps,
        "max_steps": max_steps,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise M0MobileError(f"resume checkpoint binding changed: {key}")
    step = common._checkpoint_step(checkpoint)
    if step != int(manifest.get("global_step", -1)) or step >= max_steps:
        raise M0MobileError("resume checkpoint step is inconsistent or already complete")
    if checkpoint.parents[1] != args.output_dir.expanduser().resolve():
        raise M0MobileError("joint-trajectory resume must use its original output directory")
    return {"checkpoint": str(checkpoint), "global_step": step}


def _resolved_run(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    audit: Mapping[str, Any],
    dataset: ConveyorVLAJointTrajectoryDataset,
    stages: TrainingStages,
    max_steps: int,
    token_ids: Mapping[str, Any],
    parameter_groups: list[dict[str, Any]],
    warmstart_report: Mapping[str, Any] | None,
    warmstart_identity: Mapping[str, Any] | None,
    overfit_episode_ids: Sequence[str],
    accelerator: Accelerator,
) -> Mapping[str, Any]:
    return {
        "schema_version": "conveyorvla-joint-trajectory-resolved-run-v1",
        "run_kind": "disposable_12_episode_overfit" if overfit_episode_ids else "formal",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_identity(),
        "model_contract_id": MODEL_CONTRACT_ID,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_root": str(dataset.root),
        "dataset_manifest_sha256": audit["manifest_sha256"],
        "normalization_sha256": dataset.manifest["normalization_sha256"],
        "normalizer_id": dataset.normalizer.payload["normalizer_id"],
        "policy_config_sha256": _sha256(args.config.expanduser().resolve()),
        "resolved_policy_config": config,
        "warmstart_checkpoint": str(args.warmstart_checkpoint.expanduser().resolve()),
        "warmstart_checkpoint_files": warmstart_identity,
        "warmstart_report": warmstart_report,
        "overfit_episode_ids": list(overfit_episode_ids),
        "special_token_ids": token_ids,
        "parameter_groups": parameter_groups,
        "eligible_train_rows": len(dataset),
        "equivalent_epoch_steps": stages.equivalent_epoch_steps,
        "stage_a_steps": stages.stage_a_steps,
        "max_steps": max_steps,
        "save_interval_steps": args.save_interval_steps,
        "seed": args.seed,
        "distributed": {
            "world_size": accelerator.num_processes,
            "micro_batch_per_rank": args.micro_batch_per_rank,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": 64,
            "split_batches": True,
        },
        "disabled_runtime_paths": ["DONE", "prefix", "IK", "cuRobo"],
    }


def _metrics(
    accelerator: Accelerator,
    metric_window: Mapping[str, tuple[torch.Tensor, int]],
    gradient_norm: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    elapsed_s: float,
    peak_memory_mib: float,
    stage: str,
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "stage": stage,
        "gradient_norm": common._distributed_mean(accelerator, gradient_norm),
        "optimizer_step_time_s": float(elapsed_s),
        "peak_allocated_memory_mib": float(peak_memory_mib),
        "samples_per_second": 64.0 / float(elapsed_s),
    }
    keys = sorted(metric_window)
    packed = []
    for key in keys:
        numerator, denominator = metric_window[key]
        packed.extend(
            (
                numerator.to(device=accelerator.device, dtype=torch.float32),
                torch.tensor(float(denominator), device=accelerator.device),
            )
        )
    reduced = accelerator.reduce(torch.stack(packed), reduction="sum")
    global_weights: dict[str, int] = {}
    for index, key in enumerate(keys):
        numerator = float(reduced[2 * index].item())
        denominator = float(reduced[2 * index + 1].item())
        metrics[key] = 0.0 if denominator == 0.0 else numerator / denominator
        global_weights[key] = int(round(denominator))
    metrics["navigation_samples"] = global_weights["navigation_loss"]
    metrics["manipulation_samples"] = global_weights["manipulation_loss"]
    metrics["boundary_pairs"] = global_weights["boundary_loss"]
    metrics["progress_samples"] = global_weights["progress_loss"]
    for group in optimizer.param_groups:
        metrics[f"lr_{group['name']}"] = float(group["lr"])
    if stage == "A" and any(
        metrics[f"lr_{name}"] != 0.0
        for name in (
            "qwen_core",
            "qwen_vision",
            "route_embeddings_lm_head",
            "auxiliary_heads",
        )
    ):
        raise M0MobileError("Stage A Qwen/vision/auxiliary learning rate is non-zero")
    return metrics


def _accumulate_metric_window(
    window: dict[str, tuple[torch.Tensor, int]],
    result: Mapping[str, Any],
    batch_size: int,
) -> None:
    weights = {
        "loss": None,
        "answer_loss": None,
        "route_loss": None,
        "route_accuracy": None,
        "navigation_loss": "navigation_samples",
        "navigation_objective": None,
        "manipulation_loss": "manipulation_samples",
        "manipulation_objective": None,
        "manipulation_joint_loss": "manipulation_samples",
        "manipulation_gripper_loss": "manipulation_samples",
        "boundary_loss": "boundary_pairs",
        "boundary_objective": None,
        "progress_loss": "progress_samples",
        "progress_mae": "progress_samples",
    }
    for key, weight_key in weights.items():
        value = _tensor(result.get(key), key).detach().float()
        if value.numel() != 1:
            raise M0MobileError(f"{key} metric is not scalar")
        weight = batch_size if weight_key is None else int(result.get(weight_key, -1))
        if weight < 0:
            raise M0MobileError(f"{key} metric has an invalid sample count")
        numerator = value.reshape(()) * weight
        previous = window.get(key)
        window[key] = (
            numerator if previous is None else previous[0] + numerator,
            weight if previous is None else previous[1] + weight,
        )


def _save_checkpoint(accelerator: Accelerator, output: Path, step: int) -> None:
    common._save_checkpoint(accelerator, output, step)
    if accelerator.is_main_process:
        resolved = json.loads((output / "resolved_run.json").read_text(encoding="utf-8"))
        checkpoint = output / "checkpoints" / f"step_{step:06d}"
        common._write_json_atomic(
            checkpoint / "joint_trajectory_checkpoint_manifest.json",
            {
                "schema_version": "conveyorvla-joint-trajectory-checkpoint-v1",
                "run_kind": resolved["run_kind"],
                "global_step": step,
                "model_contract_id": MODEL_CONTRACT_ID,
                "dataset_schema_version": DATASET_SCHEMA_VERSION,
                "dataset_manifest_sha256": resolved["dataset_manifest_sha256"],
                "normalization_sha256": resolved["normalization_sha256"],
                "normalizer_id": resolved["normalizer_id"],
                "policy_config_sha256": resolved["policy_config_sha256"],
                "stage_a_steps": resolved["stage_a_steps"],
                "max_steps": resolved["max_steps"],
            },
        )
    accelerator.wait_for_everyone()


def _set_status(
    accelerator: Accelerator,
    output: Path,
    status: str,
    step: int,
    metrics: Mapping[str, Any],
) -> None:
    if accelerator.is_main_process:
        common._write_json_atomic(
            output / "run_state.json",
            {
                "schema_version": "conveyorvla-joint-trajectory-run-state-v1",
                "status": status,
                "global_step": step,
                "metrics": dict(metrics),
            },
        )
        _event(accelerator, output, "status", status=status, step=step)
    accelerator.wait_for_everyone()


def _event(
    accelerator: Accelerator, output: Path, event: str, **values: Any
) -> None:
    if not accelerator.is_main_process:
        return
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    **values,
                },
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


def _git_identity() -> Mapping[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"head": head, "dirty": dirty}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise M0MobileError(f"{name} is not a tensor")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
