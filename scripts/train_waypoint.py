#!/usr/bin/env python3
"""Train the state-free ConveyorVLA Waypoint Policy v1."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Reuse the battle-tested Accelerate/ZeRO checkpoint and gradient plumbing used
# by the existing trainer. Importing it also configures rank-local TMPDIR before
# torch/deepspeed initialization.
from scripts import train_hierarchical as common  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    MODEL_CONTRACT_ID,
    WaypointRoute,
)
from conveyor_bench.conveyorvla.waypoint_data import (  # noqa: E402
    ConveyorVLAWaypointDataset,
    audit_waypoint_dataset,
)
from conveyor_bench.conveyorvla.waypoint_model import (  # noqa: E402
    ConveyorVLAWaypointPolicy,
    LayerwiseFlowMatchingActionHead,
    LayerwiseFlowMatchingConfig,
    WaypointLossConfig,
    WaypointQwenInterface,
    lambda_self_schedule,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "waypoint_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--save-interval-steps", type=int, default=1_000)
    parser.add_argument("--save-first-checkpoint-step", type=int, default=20)
    parser.add_argument("--log-interval-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    _validate_args(args, config)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    output_reserved = False
    try:
        audit = audit_waypoint_dataset(args.dataset_root)
        if not audit["ok"]:
            raise M0MobileError(
                "waypoint dataset failed its gate: " + "; ".join(audit["problems"])
            )
        output = args.output_dir.expanduser().resolve()
        common._reserve_output(accelerator, output, None)
        output_reserved = True
        set_seed(args.seed, device_specific=True)
        dataset = ConveyorVLAWaypointDataset(args.dataset_root, split="train")
        sampler = WeightedRandomSampler(
            torch.as_tensor(dataset.sample_weights(), dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=list,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            drop_last=True,
        )
        model, token_ids = _build_model(
            config,
            args.model_root,
            dataset,
            args.attention_implementation,
        )
        optimizer, parameter_groups = _optimizer(model, config)
        warmup_steps = (
            int(config["optimization"]["warmup_steps"])
            if args.warmup_steps is None
            else args.warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            common._schedule(args.max_steps, warmup_steps),
        )
        if accelerator.is_main_process:
            model.qwen.processor.save_pretrained(output / "processor")
            common._write_json_atomic(output / "dataset_audit.json", audit)
            common._write_json_atomic(output / "resolved_policy_config.json", config)
            common._write_json_atomic(
                output / "resolved_run.json",
                _resolved_run(
                    args,
                    config,
                    dataset,
                    audit,
                    token_ids,
                    parameter_groups,
                    accelerator,
                    warmup_steps,
                ),
            )
        accelerator.wait_for_everyone()
        model, optimizer, loader, scheduler = accelerator.prepare(
            model, optimizer, loader, scheduler
        )
        global_step = 0
        last_checkpoint_step = 0
        last_metrics: dict[str, Any] = {}
        _set_status(accelerator, output, "running", global_step)
        model.train()
        deepspeed_engine = common._deepspeed_engine(accelerator)
        optimizer.zero_grad(set_to_none=True)
        while global_step < args.max_steps:
            for examples in loader:
                progress = global_step / args.max_steps
                self_weight = lambda_self_schedule(progress)
                with accelerator.accumulate(model):
                    oracle = model(examples, objective="oracle")
                    _finite_losses(oracle, ("loss", "answer_loss", "route_loss", "navigation_loss", "manipulation_loss"))
                    oracle_loss = _tensor(oracle["loss"], "oracle loss")
                    common._backward_loss(
                        accelerator,
                        deepspeed_engine,
                        oracle_loss,
                        gradient_boundary=accelerator.sync_gradients and self_weight == 0.0,
                    )
                    if self_weight > 0.0:
                        self_conditioned = model(examples, objective="self_conditioned")
                        _finite_losses(
                            self_conditioned,
                            ("loss", "navigation_loss", "manipulation_loss"),
                        )
                        self_loss = _tensor(self_conditioned["loss"], "self-conditioned loss")
                        common._backward_loss(
                            accelerator,
                            deepspeed_engine,
                            self_weight * self_loss,
                            gradient_boundary=accelerator.sync_gradients,
                        )
                    else:
                        self_conditioned = _zero_self_metrics(oracle_loss)
                        self_loss = oracle_loss.detach() * 0.0

                    gradient_norm = torch.full((), float("nan"), device=oracle_loss.device)
                    component_norms = _nan_component_norms(oracle_loss.device)
                    if accelerator.sync_gradients:
                        component_norms = common._component_gradient_norms(accelerator, optimizer)
                        for name, value in component_norms.items():
                            common._finite(value, name)
                            if float(value.detach()) <= 0.0:
                                raise M0MobileError(f"{name} is zero")
                        if deepspeed_engine is None:
                            gradient_norm = accelerator.clip_grad_norm_(
                                model.parameters(),
                                float(config["optimization"]["max_gradient_norm"]),
                            )
                        else:
                            gradient_norm = torch.stack(tuple(component_norms.values())).square().sum().sqrt()
                        common._finite(gradient_norm, "gradient norm")
                    if deepspeed_engine is None:
                        optimizer.step()
                    elif accelerator.sync_gradients:
                        deepspeed_engine.step()
                        common._clear_deepspeed_partitioned_gradients(deepspeed_engine)
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not accelerator.sync_gradients:
                    continue
                global_step += 1
                learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
                if not learning_rates or any(
                    not math.isfinite(value) or value <= 0.0 for value in learning_rates
                ):
                    raise M0MobileError("optimizer learning rate is not finite and positive")
                last_metrics = _step_metrics(
                    accelerator,
                    oracle,
                    self_conditioned,
                    self_weight,
                    gradient_norm,
                    component_norms,
                    learning_rates,
                )
                if global_step == 1 or global_step % args.log_interval_steps == 0:
                    _event(
                        accelerator,
                        output,
                        "train_step",
                        step=global_step,
                        valid_optimizer_step=True,
                        **last_metrics,
                    )
                if (
                    global_step == args.save_first_checkpoint_step
                    or global_step % args.save_interval_steps == 0
                ):
                    _save_waypoint_checkpoint(accelerator, output, global_step)
                    last_checkpoint_step = global_step
                if global_step >= args.max_steps:
                    break
        if last_checkpoint_step != global_step:
            _save_waypoint_checkpoint(accelerator, output, global_step)
        _set_status(accelerator, output, "complete", global_step, last_metrics)
        return 0
    except Exception as error:
        if accelerator.is_main_process and output_reserved:
            _write_failure(args.output_dir.expanduser().resolve(), error)
        raise


def _build_model(
    config: Mapping[str, Any],
    model_root: Path,
    dataset: ConveyorVLAWaypointDataset,
    attention_implementation: str,
) -> tuple[ConveyorVLAWaypointPolicy, Mapping[str, int | list[int]]]:
    root = model_root.expanduser().resolve()
    qwen = WaypointQwenInterface.from_local(
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
    navigation = LayerwiseFlowMatchingActionHead(
        LayerwiseFlowMatchingConfig(
            action_dim=int(action["navigation_action_dim"]), **shared
        )
    )
    manipulation = LayerwiseFlowMatchingActionHead(
        LayerwiseFlowMatchingConfig(
            action_dim=int(action["manipulation_action_dim"]), **shared
        )
    )
    loss = config["loss"]
    policy = ConveyorVLAWaypointPolicy(
        qwen,
        navigation,
        manipulation,
        route_confidence_min=float(config["router"]["route_confidence_min"]),
        max_subtask_tokens=int(config["router"]["max_subtask_tokens"]),
        loss_config=WaypointLossConfig(
            lambda_answer=float(loss["lambda_answer"]),
            lambda_route=float(loss["lambda_route"]),
            lambda_nav=float(loss["lambda_navigation"]),
            lambda_arm=float(loss["lambda_manipulation"]),
            repeated_diffusion_steps=int(loss["repeated_diffusion_steps"]),
        ),
        route_class_weights=_route_class_weights(dataset.routes),
    )
    policy.enable_full_finetuning()
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise M0MobileError("waypoint full-finetuning contract left frozen parameters")
    ids = policy.router.token_ids
    return policy, {
        "pred_action": ids.pred_action,
        "pred_done": ids.pred_done,
        "routes": list(ids.route_ids),
        "subtask_start": ids.subtask_start,
        "subtask_end": ids.subtask_end,
    }


def _optimizer(
    model: ConveyorVLAWaypointPolicy,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    optimization = config["optimization"]
    lm_suffixes = ("embed_tokens.weight", "lm_head.weight")
    qwen_lm: list[torch.nn.Parameter] = []
    qwen_core: list[torch.nn.Parameter] = []
    for name, parameter in model.qwen.named_parameters():
        (qwen_lm if name.endswith(lm_suffixes) else qwen_core).append(parameter)
    groups = (
        ("vlm_core", qwen_core, float(optimization["vlm_core_learning_rate"])),
        ("vlm_embeddings_lm_head", qwen_lm, float(optimization["vlm_embeddings_lm_head_learning_rate"])),
        ("navigation_head", list(model.navigation_head.parameters()), float(optimization["navigation_head_learning_rate"])),
        ("manipulation_head", list(model.manipulation_head.parameters()), float(optimization["manipulation_head_learning_rate"])),
    )
    if any(not parameters for _name, parameters, _lr in groups):
        raise M0MobileError("waypoint optimizer parameter group is empty")
    flat = [parameter for _name, parameters, _lr in groups for parameter in parameters]
    if len({id(parameter) for parameter in flat}) != len(flat):
        raise M0MobileError("waypoint optimizer parameter groups overlap")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if {id(parameter) for parameter in flat} != expected:
        raise M0MobileError("waypoint optimizer does not cover every trainable parameter")
    optimizer = torch.optim.AdamW(
        [
            {"name": name, "params": parameters, "lr": learning_rate}
            for name, parameters, learning_rate in groups
        ],
        betas=tuple(float(value) for value in optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    report = [
        {
            "name": name,
            "learning_rate": learning_rate,
            "parameter_tensors": len(parameters),
            "parameters": sum(int(parameter.numel()) for parameter in parameters),
        }
        for name, parameters, learning_rate in groups
    ]
    return optimizer, report


def _route_class_weights(routes: Iterable[str]) -> dict[str, float]:
    counts = {route.value: 0 for route in WaypointRoute}
    for value in routes:
        counts[WaypointRoute(value).value] += 1
    if any(count <= 0 for count in counts.values()):
        raise M0MobileError("train split is missing a waypoint route class")
    total = sum(counts.values())
    return {
        route: total / (len(counts) * count)
        for route, count in counts.items()
    }


def _step_metrics(
    accelerator: Accelerator,
    oracle: Mapping[str, Any],
    self_conditioned: Mapping[str, Any],
    self_weight: float,
    gradient_norm: torch.Tensor,
    component_norms: Mapping[str, torch.Tensor],
    learning_rates: list[float],
) -> dict[str, Any]:
    oracle_loss = _tensor(oracle["loss"], "oracle loss")
    self_loss = _tensor(self_conditioned["loss"], "self-conditioned loss")
    result: dict[str, Any] = {
        "loss": common._distributed_mean(
            accelerator, oracle_loss.detach() + self_weight * self_loss.detach()
        ),
        "answer_loss": common._distributed_mean(accelerator, oracle["answer_loss"]),
        "route_loss": common._distributed_mean(accelerator, oracle["route_loss"]),
        "navigation_loss": common._distributed_mean(
            accelerator, oracle["navigation_loss"]
        ),
        "manipulation_loss": common._distributed_mean(
            accelerator, oracle["manipulation_loss"]
        ),
        "self_conditioned_loss": common._distributed_mean(accelerator, self_loss),
        "lambda_self": self_weight,
        "gradient_norm": common._distributed_mean(accelerator, gradient_norm),
        "learning_rates": learning_rates,
    }
    result.update(
        {
            name: float(value.detach().cpu())
            for name, value in component_norms.items()
        }
    )
    for key in (
        "navigation_samples",
        "manipulation_samples",
        "done_samples",
    ):
        result[f"oracle_{key}"] = common._distributed_sum_int(
            accelerator, int(oracle[key])
        )
    for key in (
        "navigation_samples",
        "manipulation_samples",
        "route_matches",
        "route_mismatches",
        "route_recovers",
    ):
        result[f"self_{key}"] = common._distributed_sum_int(
            accelerator, int(self_conditioned[key])
        )
    return result


def _zero_self_metrics(reference: torch.Tensor) -> dict[str, torch.Tensor | int]:
    zero = reference.detach() * 0.0
    return {
        "loss": zero,
        "navigation_loss": zero,
        "manipulation_loss": zero,
        "navigation_samples": 0,
        "manipulation_samples": 0,
        "route_matches": 0,
        "route_mismatches": 0,
        "route_recovers": 0,
    }


def _nan_component_norms(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.full((), float("nan"), device=device)
        for name in (
            "vlm_gradient_norm",
            "navigation_gradient_norm",
            "manipulation_gradient_norm",
        )
    }


def _finite_losses(values: Mapping[str, Any], keys: Iterable[str]) -> None:
    for key in keys:
        common._finite(_tensor(values[key], key), key.replace("_", " "))


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise M0MobileError(f"{name} is not a tensor")
    return value


def _resolved_run(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    dataset: ConveyorVLAWaypointDataset,
    audit: Mapping[str, Any],
    token_ids: Mapping[str, Any],
    parameter_groups: list[dict[str, Any]],
    accelerator: Accelerator,
    warmup_steps: int,
) -> dict[str, Any]:
    effective_batch = (
        args.batch_size
        * args.gradient_accumulation_steps
        * accelerator.num_processes
    )
    dataset_root = args.dataset_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    source_git = _source_git_identity(PROJECT_ROOT)
    code_snapshot = os.environ.get("CONVEYORVLA_CODE_SNAPSHOT")
    if code_snapshot and code_snapshot != source_git["commit"]:
        raise M0MobileError(
            "CONVEYORVLA_CODE_SNAPSHOT does not match the checked-out Git commit"
        )
    camera_contract = {
        "camera_calibration_id": dataset.manifest["camera_calibration_id"],
        "visual_history": dataset.manifest["visual_history"],
    }
    normalization = {
        "relative_path": dataset.manifest["normalization_relative_path"],
        "sha256": dataset.manifest["normalization_sha256"],
    }
    return {
        "schema_version": "conveyorvla-waypoint-training-run-v1",
        "status": "initializing",
        "model_contract_id": MODEL_CONTRACT_ID,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": common._sha256(dataset_root / "manifest.json"),
        "dataset_audit_manifest_sha256": audit["manifest_sha256"],
        "normalization_sha256": dataset.manifest["normalization_sha256"],
        "normalization": normalization,
        "camera_contract": camera_contract,
        "dataset_action_contract": dataset.manifest["action_contract"],
        "config": str(config_path),
        "config_sha256": common._sha256(config_path),
        "model_root": str(args.model_root.expanduser().resolve()),
        "qwen_base": _qwen_base_identity(
            args.model_root.expanduser().resolve()
            / str(config["vlm"]["relative_path"])
        ),
        "initialization": {
            "qwen": "clean_local_qwen3_vl_4b",
            "navigation_head": "new_random_3d_layerwise_fm",
            "manipulation_head": "new_random_7d_layerwise_fm",
            "legacy_checkpoint_loaded": False,
            "optimizer_resume": False,
        },
        "model_input_robot_state_fields": 0,
        "special_token_ids": token_ids,
        "route_confidence_min": config["router"]["route_confidence_min"],
        "parameter_groups": parameter_groups,
        "world_size": accelerator.num_processes,
        "batch_size_per_process": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch,
        "train_rows": len(dataset),
        "optimizer_steps_per_equivalent_sampling_epoch": len(dataset)
        / effective_batch,
        "equivalent_sampling_epochs_at_max_steps": args.max_steps
        * effective_batch
        / len(dataset),
        "max_steps": args.max_steps,
        "warmup_steps": warmup_steps,
        "mixed_precision": accelerator.mixed_precision,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_uuids": os.environ.get("CONVEYORVLA_GPU_UUIDS"),
        "conda_environment": os.environ.get("CONVEYORVLA_CONDA_ENV"),
        "code_snapshot": code_snapshot,
        "source_git": source_git,
        "rank_tmp_root": os.environ.get("CONVEYORVLA_RANK_TMP_ROOT"),
        "rank_tmpdir": (
            None if common.RANK_TMPDIR is None else str(common.RANK_TMPDIR)
        ),
        "deepspeed_preloaded": common.DEEPSPEED_PRELOADED,
        "hostname": os.uname().nodename,
        "argv": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "resolved_policy_config": config,
    }


def _load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"invalid waypoint policy config {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise M0MobileError("waypoint policy config must be a JSON object")
    if value.get("schema_version") != "conveyorvla-waypoint-policy-config-v1":
        raise M0MobileError("waypoint policy config schema is incompatible")
    if value.get("model_contract_id") != MODEL_CONTRACT_ID:
        raise M0MobileError("waypoint model contract ID is incompatible")
    if value.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise M0MobileError("waypoint dataset contract ID is incompatible")
    return value


def _qwen_base_identity(model_dir: Path) -> dict[str, Any]:
    names = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    paths = [model_dir / name for name in names if (model_dir / name).is_file()]
    paths.extend(sorted(model_dir.glob("model-*.safetensors")))
    if not paths:
        single = model_dir / "model.safetensors"
        if single.is_file():
            paths.append(single)
    if not paths or not any(path.suffix == ".safetensors" for path in paths):
        raise M0MobileError("clean Qwen base checkpoint files are missing")
    return {
        "model_dir": str(model_dir),
        "files": {
            path.name: {
                "size": path.stat().st_size,
                "sha256": common._sha256(path),
            }
            for path in paths
        },
    }


def _source_git_identity(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise M0MobileError(f"cannot record source Git identity: {error}") from error

    commit = run("rev-parse", "HEAD").strip()
    porcelain = tuple(
        line
        for line in run(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines()
        if line
    )
    return {
        "commit": commit,
        "dirty_state_artifact": {
            "format": "git-status-porcelain-v1",
            "is_dirty": bool(porcelain),
            "entries": list(porcelain),
        },
    }


def _validate_args(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    positive: Iterable[tuple[str, int]] = (
        ("max steps", args.max_steps),
        ("batch size", args.batch_size),
        ("gradient accumulation", args.gradient_accumulation_steps),
        ("save interval", args.save_interval_steps),
        ("log interval", args.log_interval_steps),
    )
    if any(value <= 0 for _name, value in positive):
        raise M0MobileError("waypoint step, batch, accumulation, and intervals must be positive")
    if args.num_workers < 0:
        raise M0MobileError("waypoint num workers cannot be negative")
    if not 0 <= args.save_first_checkpoint_step <= args.max_steps:
        raise M0MobileError("first waypoint checkpoint step is outside the run")
    warmup = (
        int(config["optimization"]["warmup_steps"])
        if args.warmup_steps is None
        else args.warmup_steps
    )
    if warmup < 0 or warmup >= args.max_steps:
        raise M0MobileError("waypoint warmup must be within [0, max steps)")
    if not args.dataset_root.expanduser().resolve().is_dir():
        raise M0MobileError("waypoint dataset root does not exist")
    model_dir = (
        args.model_root.expanduser().resolve()
        / str(config["vlm"]["relative_path"])
    )
    if not model_dir.is_dir():
        raise M0MobileError(f"clean Qwen model directory does not exist: {model_dir}")
    action = config["action_model"]
    required = {
        "action_horizon": 20,
        "cross_attention_dim": 2560,
        "hidden_size": 1024,
        "num_layers": 16,
        "navigation_action_dim": 3,
        "manipulation_action_dim": 7,
        "state_encoder": False,
        "shared_parameters": False,
    }
    if any(action.get(key) != value for key, value in required.items()):
        raise M0MobileError("waypoint production action-model contract was modified")
    schedule = config["loss"]["lambda_self_schedule"]
    if schedule != {
        "zero_until_progress": 0.05,
        "linear_to_progress": 0.4,
        "maximum": 0.5,
    }:
        raise M0MobileError("waypoint self-conditioned schedule was modified")


def _set_status(
    accelerator: Accelerator,
    output: Path,
    status: str,
    step: int,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    if accelerator.is_main_process:
        common._write_json_atomic(
            output / "run_state.json",
            {
                "schema_version": "conveyorvla-waypoint-training-state-v1",
                "status": status,
                "global_step": step,
                "metrics": dict(metrics or {}),
            },
        )
        _event(accelerator, output, "status", status=status, step=step)
    accelerator.wait_for_everyone()


def _save_waypoint_checkpoint(
    accelerator: Accelerator,
    output: Path,
    step: int,
) -> None:
    common._save_checkpoint(accelerator, output, step)
    if accelerator.is_main_process:
        resolved_path = output / "resolved_run.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        checkpoint = output / "checkpoints" / f"step_{step:06d}"
        common._write_json_atomic(
            checkpoint / "waypoint_checkpoint_manifest.json",
            {
                "schema_version": "conveyorvla-waypoint-checkpoint-v1",
                "global_step": step,
                "model_contract_id": resolved["model_contract_id"],
                "dataset_schema_version": resolved["dataset_schema_version"],
                "dataset_manifest_sha256": resolved["dataset_manifest_sha256"],
                "normalization_sha256": resolved["normalization_sha256"],
                "normalization": resolved["normalization"],
                "camera_contract": resolved["camera_contract"],
                "dataset_action_contract": resolved["dataset_action_contract"],
                "special_token_ids": resolved["special_token_ids"],
                "qwen_base": resolved["qwen_base"],
                "action_contract": resolved["resolved_policy_config"]["action_model"],
                "route_confidence_min": resolved["route_confidence_min"],
                "processor_relative_path": "../../processor",
                "resolved_policy_config_sha256": resolved["config_sha256"],
                "resolved_run_sha256": common._sha256(resolved_path),
                "source_git": resolved["source_git"],
                "legacy_state_projection_present": False,
            },
        )
    accelerator.wait_for_everyone()


def _event(
    accelerator: Accelerator,
    output: Path,
    event: str,
    **values: Any,
) -> None:
    if not accelerator.is_main_process:
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **values,
    }
    with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        stream.flush()


def _write_failure(output: Path, error: Exception) -> None:
    state = output / "run_state.json"
    try:
        previous = json.loads(state.read_text(encoding="utf-8"))
        step = int(previous.get("global_step", 0))
    except (OSError, ValueError, json.JSONDecodeError):
        step = 0
    common._write_json_atomic(
        state,
        {
            "schema_version": "conveyorvla-waypoint-training-state-v1",
            "status": "failed",
            "global_step": step,
            "error": str(error),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
