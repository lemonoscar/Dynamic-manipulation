#!/usr/bin/env python3
"""Load and validate a ConveyorVLA Waypoint v1 ZeRO checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# This import configures the guarded rank-local TMPDIR before torch/deepspeed.
from scripts import train_waypoint as training  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import set_seed  # noqa: E402
from torch.utils.data import DataLoader, WeightedRandomSampler  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_data import (  # noqa: E402
    ConveyorVLAWaypointDataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest, resolved, dataset_root = _validate_binding(checkpoint)
    run_args = _mapping(resolved.get("arguments"), "resolved arguments")
    accumulation = int(resolved["gradient_accumulation_steps"])
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision="bf16",
        step_scheduler_with_optimizer=False,
    )
    set_seed(int(run_args["seed"]), device_specific=True)
    dataset = ConveyorVLAWaypointDataset(dataset_root, split="train")
    sampler = WeightedRandomSampler(
        torch.as_tensor(dataset.sample_weights(), dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(int(run_args["seed"])),
    )
    loader = DataLoader(
        dataset,
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
        dataset,
        str(run_args["attention_implementation"]),
    )
    if token_ids != manifest["special_token_ids"]:
        raise M0MobileError("checkpoint special token IDs do not match the processor")
    optimizer, _parameter_groups = training._optimizer(model, config)
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
    loaded_step = training.common._checkpoint_step(checkpoint)
    if loaded_step != int(manifest["global_step"]):
        raise M0MobileError("checkpoint trainer step and manifest step differ")
    local_bad, local_values = _nonfinite_parameter_partitions(model)
    bad = int(
        accelerator.reduce(
            torch.tensor(local_bad, device=accelerator.device, dtype=torch.int64),
            reduction="sum",
        ).item()
    )
    values = int(
        accelerator.reduce(
            torch.tensor(local_values, device=accelerator.device, dtype=torch.int64),
            reduction="sum",
        ).item()
    )
    if bad or values <= 0:
        raise M0MobileError("loaded checkpoint has non-finite or empty parameter shards")
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not learning_rates or any(
        not math.isfinite(value) or value <= 0.0 for value in learning_rates
    ):
        raise M0MobileError("loaded checkpoint learning rates are invalid")
    report = {
        "schema_version": "conveyorvla-waypoint-checkpoint-load-report-v1",
        "status": "pass",
        "checkpoint": str(checkpoint),
        "global_step": loaded_step,
        "world_size": accelerator.num_processes,
        "parameter_partition_values": values,
        "nonfinite_parameter_partitions": bad,
        "learning_rates": learning_rates,
        "scheduler_last_epoch": int(scheduler.state_dict()["last_epoch"]),
        "model_contract_id": manifest["model_contract_id"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "normalization_sha256": manifest["normalization_sha256"],
        "source_git": manifest["source_git"],
    }
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.report is not None:
            training.common._write_json_atomic(args.report.expanduser().resolve(), report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    return 0


def _validate_binding(
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not checkpoint.is_dir():
        raise M0MobileError(f"checkpoint directory does not exist: {checkpoint}")
    manifest = _read_json(checkpoint / "waypoint_checkpoint_manifest.json")
    if manifest.get("schema_version") != "conveyorvla-waypoint-checkpoint-v1":
        raise M0MobileError("waypoint checkpoint schema is incompatible")
    required = {
        "camera_contract",
        "dataset_action_contract",
        "normalization",
        "resolved_policy_config_sha256",
        "source_git",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise M0MobileError(f"checkpoint binding is incomplete: {missing}")
    output = checkpoint.parents[1]
    resolved_path = output / "resolved_run.json"
    resolved = _read_json(resolved_path)
    if training.common._sha256(resolved_path) != manifest["resolved_run_sha256"]:
        raise M0MobileError("checkpoint resolved-run binding is corrupt")
    config_path = Path(str(resolved["config"]))
    if training.common._sha256(config_path) != manifest["resolved_policy_config_sha256"]:
        raise M0MobileError("checkpoint policy-config binding is corrupt")
    dataset_root = Path(str(resolved["dataset_root"]))
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = _read_json(dataset_manifest_path)
    if training.common._sha256(dataset_manifest_path) != manifest["dataset_manifest_sha256"]:
        raise M0MobileError("checkpoint dataset-manifest binding is corrupt")
    normalization = _mapping(manifest["normalization"], "normalization binding")
    normalization_path = dataset_root / str(normalization["relative_path"])
    if training.common._sha256(normalization_path) != normalization["sha256"]:
        raise M0MobileError("checkpoint normalization binding is corrupt")
    if normalization["sha256"] != manifest["normalization_sha256"]:
        raise M0MobileError("checkpoint normalization hashes disagree")
    if manifest["camera_contract"] != {
        "camera_calibration_id": dataset_manifest["camera_calibration_id"],
        "visual_history": dataset_manifest["visual_history"],
    }:
        raise M0MobileError("checkpoint camera/calibration binding is corrupt")
    if manifest["dataset_action_contract"] != dataset_manifest["action_contract"]:
        raise M0MobileError("checkpoint waypoint stride/frame binding is corrupt")
    source_git = _mapping(manifest["source_git"], "source Git binding")
    dirty = _mapping(source_git.get("dirty_state_artifact"), "dirty-state artifact")
    if dirty.get("is_dirty") is not False or dirty.get("entries") != []:
        raise M0MobileError("checkpoint source Git state was dirty")
    current_git = training._source_git_identity(PROJECT_ROOT)
    if source_git["commit"] != current_git["commit"]:
        raise M0MobileError("checkpoint source commit differs from the current checkout")
    if current_git["dirty_state_artifact"]["is_dirty"]:
        raise M0MobileError("current checkpoint-validation checkout is dirty")
    processor = (checkpoint / str(manifest["processor_relative_path"])).resolve()
    if not processor.is_dir():
        raise M0MobileError("checkpoint processor binding does not resolve")
    qwen = _mapping(manifest["qwen_base"], "Qwen base binding")
    qwen_dir = Path(str(qwen["model_dir"]))
    for name, identity in _mapping(qwen["files"], "Qwen files").items():
        file_identity = _mapping(identity, f"Qwen file {name}")
        path = qwen_dir / name
        if path.stat().st_size != int(file_identity["size"]):
            raise M0MobileError(f"Qwen base size changed: {name}")
        if training.common._sha256(path) != file_identity["sha256"]:
            raise M0MobileError(f"Qwen base hash changed: {name}")
    world_size = int(resolved["world_size"])
    model_dir = checkpoint / "pytorch_model"
    if len(tuple(model_dir.glob("zero_pp_rank_*_model_states.pt"))) != world_size:
        raise M0MobileError("checkpoint model shard count is incomplete")
    if len(tuple(model_dir.glob("bf16_zero_pp_rank_*_optim_states.pt"))) != world_size:
        raise M0MobileError("checkpoint optimizer shard count is incomplete")
    return manifest, resolved, dataset_root


def _nonfinite_parameter_partitions(model: torch.nn.Module) -> tuple[int, int]:
    bad = 0
    values = 0
    for parameter in model.parameters():
        shard = getattr(parameter, "ds_tensor", parameter.data)
        if not isinstance(shard, torch.Tensor) or not shard.is_floating_point():
            continue
        values += shard.numel()
        if shard.numel() and not bool(torch.isfinite(shard).all()):
            bad += 1
    return bad, values


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read checkpoint binding {path}: {error}") from error
    if not isinstance(value, dict):
        raise M0MobileError(f"checkpoint binding must be an object: {path}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
