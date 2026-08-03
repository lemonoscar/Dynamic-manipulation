#!/usr/bin/env python3
"""Overfit a synthetic M0-Mobile batch and verify AML checkpoint inference."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from conveyor_bench.m0_aml import AMLActionHead, AMLConfig  # noqa: E402
from conveyor_bench.v1.exporters import (  # noqa: E402
    M0_MOBILE_ACTION_DIMENSION_MASK,
)


def _synthetic_batch(config: AMLConfig, batch_size: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    context = torch.randn(
        batch_size, 8, config.context_dim, generator=generator
    )
    state = torch.randn(batch_size, config.state_dim, generator=generator)
    state_weights = torch.randn(
        config.state_dim, config.action_dim, generator=generator
    ) / math.sqrt(config.state_dim)
    context_weights = torch.randn(
        config.context_dim, config.action_dim, generator=generator
    ) / math.sqrt(config.context_dim)
    base = torch.tanh(
        state @ state_weights + context.mean(dim=1) @ context_weights
    )
    phase = torch.linspace(-0.25, 0.25, config.action_horizon)[None, :, None]
    target = torch.tanh(base[:, None, :] + phase)
    target[:, :, 1] = 0.0
    target[:, :, 9] = torch.sigmoid(target[:, :, 9])
    noise = torch.randn(
        target.shape, generator=generator, dtype=target.dtype
    )
    time = torch.full((batch_size,), 0.25)
    return tuple(value.to(device) for value in (context, state, target, noise, time))


def _autocast(device: torch.device, dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _forward(
    model: AMLActionHead,
    context: torch.Tensor,
    state: torch.Tensor,
    target: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    dtype_name: str,
) -> torch.Tensor:
    with _autocast(device, dtype_name):
        return model.aml_loss(
            context,
            state,
            target,
            action_dimension_mask=mask,
            noise=noise,
            time=time,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--max-final-loss-ratio", type=float, default=0.20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.cpu_threads <= 0:
        raise ValueError("steps, batch-size, and cpu-threads must be positive")
    if not 0.0 < args.max_final_loss_ratio < 1.0:
        raise ValueError("max-final-loss-ratio must be within (0, 1)")
    if args.dtype == "bfloat16" and args.device != "cuda":
        raise ValueError("bfloat16 smoke is supported only on CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20260803)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260803)
    device = torch.device(args.device)
    config = AMLConfig()
    model = AMLActionHead(config).to(device)
    context, state, target, noise, time = _synthetic_batch(
        config, args.batch_size, device
    )
    mask = torch.tensor(
        M0_MOBILE_ACTION_DIMENSION_MASK, device=device, dtype=torch.bool
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.eval()
    with torch.no_grad():
        initial_loss = float(
            _forward(
                model,
                context,
                state,
                target,
                noise,
                time,
                mask,
                device,
                args.dtype,
            ).item()
        )
    model.train()
    gradients_finite = True
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _forward(
            model,
            context,
            state,
            target,
            noise,
            time,
            mask,
            device,
            args.dtype,
        )
        loss.backward()
        gradients_finite = gradients_finite and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_output = model(context, state, noise, time)
        final_loss = float(
            _forward(
                model,
                context,
                state,
                target,
                noise,
                time,
                mask,
                device,
                args.dtype,
            ).item()
        )
        with _autocast(device, args.dtype):
            sample = model.sample(context, state, noise=noise)

    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {"config": asdict(config), "model_state_dict": model.state_dict()},
        checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded = AMLActionHead(AMLConfig(**checkpoint["config"])).to(device).eval()
    reloaded.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        reloaded_output = reloaded(context, state, noise, time)

    reload_max_error = float((reloaded_output - final_output).abs().max().item())
    loss_ratio = final_loss / initial_loss
    passed = (
        math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and gradients_finite
        and torch.isfinite(sample).all().item()
        and sample.shape == target.shape
        and reload_max_error == 0.0
        and loss_ratio <= args.max_final_loss_ratio
    )
    report = {
        "passed": passed,
        "scope": "AML objective/sampler smoke; not a full M0 DiT-B reproduction",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "dtype": args.dtype,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_loss_ratio": loss_ratio,
        "max_final_loss_ratio": args.max_final_loss_ratio,
        "gradients_finite": gradients_finite,
        "sample_shape": list(sample.shape),
        "sample_finite": torch.isfinite(sample).all().item(),
        "reload_max_abs_error": reload_max_error,
        "config": asdict(config),
        "checkpoint": checkpoint_path.name,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
