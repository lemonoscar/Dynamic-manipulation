#!/usr/bin/env python3
"""Run a two-camera, full-checkpoint AL0 GPU smoke without collecting data."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from conveyor_bench.m0_dit import (  # noqa: E402
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
)
from conveyor_bench.m0_mobile import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    M0MobileError,
    audit_model_artifacts,
    load_m0_mobile_config,
    resolve_model_root,
)
from conveyor_bench.m0_policy import (  # noqa: E402
    M0MobilePolicy,
    Qwen3VLInterface,
    m0_dit_config,
    transfer_robocasa_policy_weights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=20260803)
    return parser


def _distributed_device() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise M0MobileError("the full policy smoke requires CUDA")
    if local_rank >= torch.cuda.device_count():
        raise M0MobileError("LOCAL_RANK exceeds the visible CUDA device count")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    return torch.device("cuda", local_rank), rank, local_rank, world_size


def _checkpoint_path(config: dict, root: Path) -> Path:
    relative = Path(config["checkpoint_transfer"]["relative_path"])
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise M0MobileError("checkpoint path escapes the model root") from error
    if not path.is_file():
        raise M0MobileError(f"checkpoint does not exist: {path}")
    return path


def _example(config: dict, rank: int) -> dict:
    try:
        from PIL import Image
    except ImportError as error:
        raise M0MobileError("Pillow is required for the full policy smoke") from error
    data = config["data"]
    mask = data["action_dimension_mask"]
    action = [0.0] * data["action_dim"]
    action[0] = 0.1
    action[3] = 0.2
    action[5] = -0.2
    action[9] = 1.0
    colors = ((160 + rank, 20, 20), (20, 20, 160 + rank))
    return {
        "image": [Image.new("RGB", tuple(data["image_size"]), color) for color in colors],
        "lang": "Pick the moving red part and place it in the blue sorting tray.",
        "state": [[0.0] * data["state_dim"]],
        "action": [action[:] for _ in range(data["action_horizon"])],
        "action_mask": mask,
    }


def _event(rank: int, event: str, **values) -> None:
    print(json.dumps({"rank": rank, "event": event, **values}, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device, rank, local_rank, world_size = _distributed_device()
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        config = load_m0_mobile_config(args.config)
        root = resolve_model_root(config, args.model_root)
        artifacts = audit_model_artifacts(config, root, verify_hashes=False)
        _event(rank, "artifacts_ready", files=len(artifacts))

        qwen = Qwen3VLInterface.from_local(
            root / config["vlm"]["relative_path"],
            checkpoint_vocab_size=config["vlm"]["checkpoint_vocab_size"],
            dtype=torch.bfloat16,
            attention_implementation=args.attention_implementation,
        )
        policy = M0MobilePolicy(
            qwen,
            M0DiTActionHead(m0_dit_config(config)),
            repeated_diffusion_steps=config["training"]["repeated_diffusion_steps"],
        )
        transfer = transfer_robocasa_policy_weights(
            policy,
            _checkpoint_path(config, root),
        )
        if transfer.loaded_qwen_tensors != 714 or transfer.loaded_action_tensors != 242:
            raise M0MobileError("checkpoint transfer tensor counts are unexpected")
        policy.freeze_qwen()
        policy.qwen_vl_interface.to(device)
        policy.action_model.to(device)
        policy.train()
        _event(
            rank,
            "policy_ready",
            loaded_qwen_tensors=transfer.loaded_qwen_tensors,
            loaded_action_tensors=transfer.loaded_action_tensors,
        )

        boundary = [
            parameter
            for name, parameter in policy.action_model.named_parameters()
            if name in GO2_X5_REINITIALIZED_ACTION_KEYS
        ]
        if len(boundary) != len(GO2_X5_REINITIALIZED_ACTION_KEYS):
            raise M0MobileError("the six Go2-X5 boundary tensors were not found")
        optimizer = torch.optim.AdamW(boundary, lr=1.0e-4)
        example = _example(config, rank)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = policy([example])["action_loss"]
        if not torch.isfinite(loss):
            raise M0MobileError("full policy loss is not finite")
        loss.backward()
        gradients = [parameter.grad for parameter in boundary if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise M0MobileError("boundary gradients are missing or non-finite")
        gradient_norm = math.sqrt(
            sum(value.float().square().sum().item() for value in gradients)
        )
        optimizer.step()
        _event(rank, "train_step", loss=float(loss.item()), gradient_norm=gradient_norm)

        policy.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = policy.predict_normalized_actions([example])
        if actions.shape != (1, 16, 10) or not torch.isfinite(actions).all():
            raise M0MobileError("sampled action chunk is invalid")
        if torch.count_nonzero(actions[..., 1]).item() != 0:
            raise M0MobileError("masked base_vy is not exactly zero")
        torch.cuda.synchronize(device)
        loss_sum = loss.detach().float()
        if world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        report = {
            "ok": True,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(device),
            "mean_loss": float(loss_sum.item() / world_size),
            "gradient_norm": gradient_norm,
            "action_shape": list(actions.shape),
            "base_vy_nonzero": int(torch.count_nonzero(actions[..., 1]).item()),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 2),
            "checkpoint": {
                "loaded_qwen_tensors": transfer.loaded_qwen_tensors,
                "loaded_action_tensors": transfer.loaded_action_tensors,
                "reinitialized_tensors": len(transfer.reinitialized_keys),
            },
        }
        _event(rank, "complete", report=report)
        if rank == 0:
            print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return 0
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        _event(rank, "failed", error=str(error))
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
