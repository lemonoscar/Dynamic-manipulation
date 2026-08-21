from pathlib import Path

import pytest


pytest.importorskip("accelerate")

from conveyor_bench.conveyorvla.config import M0MobileError
from scripts.check_waypoint_checkpoint import _validate_checkpoint_shards


def _touch(root: Path, *names: str) -> None:
    root.mkdir()
    for name in names:
        (root / name).touch()


def _optimizer_names(world_size: int = 4) -> tuple[str, ...]:
    return tuple(
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(world_size)
    )


@pytest.mark.parametrize("recorded_stage", [2, None])
def test_checkpoint_shards_accept_zero2_replicated_model(
    tmp_path: Path, recorded_stage: int | None
) -> None:
    model_dir = tmp_path / "pytorch_model"
    _touch(model_dir, "mp_rank_00_model_states.pt", *_optimizer_names())

    _validate_checkpoint_shards(model_dir, world_size=4, zero_stage=recorded_stage)


@pytest.mark.parametrize("recorded_stage", [3, None])
def test_checkpoint_shards_accept_zero3_partitioned_model(
    tmp_path: Path, recorded_stage: int | None
) -> None:
    model_dir = tmp_path / "pytorch_model"
    model_names = tuple(
        f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt" for rank in range(4)
    )
    _touch(model_dir, *model_names, *_optimizer_names())

    _validate_checkpoint_shards(model_dir, world_size=4, zero_stage=recorded_stage)


def test_checkpoint_shards_reject_incomplete_zero2_optimizer_state(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "pytorch_model"
    _touch(model_dir, "mp_rank_00_model_states.pt", *_optimizer_names()[:3])

    with pytest.raises(M0MobileError, match="optimizer shard count"):
        _validate_checkpoint_shards(model_dir, world_size=4, zero_stage=2)
