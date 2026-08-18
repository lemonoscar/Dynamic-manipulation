from __future__ import annotations

import errno
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


def test_teacher_forcing_schedule_reaches_zero_before_training_end() -> None:
    assert TRAIN._teacher_forcing_probability(0, 2, 6) == 1.0
    assert TRAIN._teacher_forcing_probability(2, 2, 6) == 1.0
    assert TRAIN._teacher_forcing_probability(4, 2, 6) == pytest.approx(0.5)
    assert TRAIN._teacher_forcing_probability(6, 2, 6) == 0.0


def test_routing_counts_are_summed_across_processes() -> None:
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        reduce=lambda value, reduction: value * 4,
    )

    assert TRAIN._distributed_sum_int(accelerator, 3) == 12


def test_training_cli_has_no_semantic_history_injection_controls() -> None:
    options = TRAIN.build_parser()._option_string_actions

    assert "--history-dropout-probability" not in options
    assert "--history-corruption-probability" not in options
    assert not hasattr(TRAIN, "_prepare_training_examples")


def test_distributed_ranks_use_isolated_tmpdirs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONVEYORVLA_RANK_TMP_ROOT", str(tmp_path))
    monkeypatch.setenv("LOCAL_RANK", "3")

    isolated = TRAIN._configure_rank_tmpdir()

    assert isolated == tmp_path / "rank-3"
    assert isolated.is_dir()
    assert TRAIN.os.environ["TMPDIR"] == str(isolated)


def test_shared_storage_cleanup_retries_transient_nfs_error(
    tmp_path, monkeypatch
) -> None:
    calls = []

    def remove(path):
        calls.append(path)
        if len(calls) < 3:
            raise OSError(errno.ENOTEMPTY, "directory not empty")
        return "removed"

    monkeypatch.setattr(TRAIN.time, "sleep", lambda _seconds: None)

    assert (
        TRAIN._rmtree_with_shared_storage_retries(remove, tmp_path)
        == "removed"
    )
    assert calls == [tmp_path, tmp_path, tmp_path]


def test_zero3_component_norms_use_partitioned_gradient_buffers() -> None:
    zero_optimizer = SimpleNamespace(
        averaged_gradients={
            0: [torch.tensor([3.0, 4.0])],
            1: [torch.tensor([12.0])],
            2: [torch.tensor([5.0])],
        },
        sub_group_to_group_id={0: 0, 1: 1, 2: 2},
        param_groups=[
            {"name": "vlm_core"},
            {"name": "navigation_dit_core"},
            {"name": "manipulation_dit_core"},
        ],
        loss_scale=2.0,
    )

    norms = TRAIN._component_gradient_norms(
        SimpleNamespace(
            device=torch.device("cpu"),
            reduce=lambda value, reduction: value,
        ),
        SimpleNamespace(optimizer=zero_optimizer),
    )

    assert float(norms["vlm_gradient_norm"]) == pytest.approx(2.5)
    assert float(norms["navigation_gradient_norm"]) == pytest.approx(6.0)
    assert float(norms["manipulation_gradient_norm"]) == pytest.approx(2.5)


def test_deepspeed_backward_defers_the_gradient_boundary() -> None:
    class Engine:
        def __init__(self):
            self.boundaries = []
            self.losses = []

        def set_gradient_accumulation_boundary(self, *, is_boundary):
            self.boundaries.append(is_boundary)

        def backward(self, loss):
            self.losses.append(loss)

    engine = Engine()
    accelerator = SimpleNamespace(backward=lambda _loss: pytest.fail("wrong path"))
    first = torch.tensor(1.0)
    second = torch.tensor(2.0)

    TRAIN._backward_loss(
        accelerator,
        engine,
        first,
        gradient_boundary=False,
    )
    TRAIN._backward_loss(
        accelerator,
        engine,
        second,
        gradient_boundary=True,
    )

    assert engine.boundaries == [False, True]
    assert engine.losses == [first, second]


def test_zero3_partition_buffer_is_cleared_after_step() -> None:
    buffer = torch.ones(5)
    zero_optimizer = SimpleNamespace(
        grad_partitions_flat_buffer=buffer,
        averaged_gradients={0: [buffer]},
    )

    TRAIN._clear_deepspeed_partitioned_gradients(
        SimpleNamespace(optimizer=zero_optimizer)
    )

    assert torch.count_nonzero(buffer) == 0
    assert zero_optimizer.averaged_gradients == {}
