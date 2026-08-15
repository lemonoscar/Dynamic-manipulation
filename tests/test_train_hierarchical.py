from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("safetensors")


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_hierarchical.py"
SPEC = importlib.util.spec_from_file_location("train_hierarchical", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def test_resume_repairs_world_size_accelerated_scheduler_step() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-6)
    learning_rate_lambda = TRAIN._schedule(max_steps=10_000, warmup_steps=200)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_lambda)
    scheduler.last_epoch = 12_000
    scheduler._step_count = 12_001
    scheduler._last_lr = [2e-7]
    optimizer.param_groups[0]["lr"] = 2e-7

    report = TRAIN._align_scheduler_after_resume(
        SimpleNamespace(scheduler=scheduler), optimizer, global_step=3_000
    )

    expected = 2e-6 * learning_rate_lambda(3_000)
    assert report["repaired"] is True
    assert report["loaded_scheduler_step"] == 12_000
    assert report["global_step"] == 3_000
    assert report["learning_rates"] == pytest.approx([expected])
    assert scheduler.last_epoch == 3_000
    assert scheduler._step_count == 3_001
    assert scheduler.get_last_lr() == [pytest.approx(expected)]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)


def test_resume_keeps_already_aligned_scheduler_unchanged() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, TRAIN._schedule(max_steps=10_000, warmup_steps=200)
    )
    scheduler.last_epoch = 3_000
    scheduler._last_lr = [optimizer.param_groups[0]["lr"]]

    report = TRAIN._align_scheduler_after_resume(
        SimpleNamespace(scheduler=scheduler), optimizer, global_step=3_000
    )

    assert report["repaired"] is False
    assert scheduler.last_epoch == 3_000
