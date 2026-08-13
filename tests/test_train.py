from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from conveyor_bench.conveyorvla.dit import M0DiTActionHead, M0DiTConfig
from conveyor_bench.conveyorvla.config import load_m0_mobile_config
from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.temporal import load_temporal_config


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
SPEC = importlib.util.spec_from_file_location("train", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def _tiny_action_model() -> M0DiTActionHead:
    return M0DiTActionHead(
        M0DiTConfig(
            action_dim=3,
            state_dim=4,
            action_horizon=4,
            vlm_hidden_dim=24,
            input_embedding_dim=16,
            hidden_size=32,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=2,
            dropout=0.0,
            max_seq_len=16,
            num_target_vision_tokens=2,
        )
    )


def test_optimizer_has_core_and_six_boundary_parameters() -> None:
    model = _tiny_action_model()
    optimizer = TRAIN._optimizer(model, load_m0_mobile_config())

    assert len(optimizer.param_groups) == 2
    assert len(optimizer.param_groups[1]["params"]) == 6
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.0e-4)


def test_cosine_schedule_warms_up_and_keeps_nonzero_floor() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = TRAIN._scheduler(optimizer, max_steps=10, warmup_steps=2)

    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])

    assert observed[0] == pytest.approx(0.5)
    assert observed[1] == pytest.approx(1.0)
    assert observed[-1] == pytest.approx(0.05)


def test_publish_state_statistics_copies_exact_deployment_input(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "run"
    output.mkdir()
    source.write_bytes(b'{"count": 2}\n')

    digest = TRAIN._publish_state_statistics(source, output)

    assert (output / "state_statistics.json").read_bytes() == source.read_bytes()
    assert digest == "386a7dbdc927cbf76d42f11a7f376fddefbeeb3fe2cb473cf972ce54ae108642"


def test_initial_action_checkpoint_restores_exact_tensor_mapping(tmp_path) -> None:
    source_model = _tiny_action_model()
    checkpoint = tmp_path / "action.safetensors"
    TRAIN.save_file(source_model.state_dict(), checkpoint)
    target_model = _tiny_action_model()
    for parameter in target_model.parameters():
        parameter.data.zero_()

    digest = TRAIN._load_initial_action_checkpoint(target_model, checkpoint)

    assert len(digest) == 64
    for key, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[key], value)


def test_all_belt_speeds_requires_explicit_task_type(tmp_path) -> None:
    args = TRAIN.build_parser().parse_args(
        [
            "--episode-root",
            str(tmp_path / "episode"),
            "--state-statistics",
            str(tmp_path / "statistics.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--all-belt-speeds",
        ]
    )

    with pytest.raises(M0MobileError, match="explicit --task-type"):
        TRAIN._datasets(args, load_m0_mobile_config())


def test_temporal_training_reuses_artifacts_with_20_step_contract() -> None:
    config = TRAIN._temporal_training_config(
        load_m0_mobile_config(),
        load_temporal_config(),
    )

    assert config["action_model"]["action_horizon"] == 20
    assert config["data"]["history_offsets_model_ticks"] == [-2, 0]
    assert config["data"]["image_size"] == [224, 224]
    assert config["data"]["action_dimension_mask"] == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert config["vlm"]["relative_path"] == "Qwen3-VL-4B-Instruct"
    assert config["training"]["repeated_diffusion_steps"] == 4


def test_saved_temporal_config_is_accepted_by_online_normalizer(tmp_path) -> None:
    config = TRAIN._temporal_training_config(
        load_m0_mobile_config(),
        load_temporal_config(),
    )
    path = tmp_path / "conveyorvla_al0_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_m0_mobile_config(path)

    assert loaded["action_model"]["action_horizon"] == 20
    assert loaded["normalization"]["action"]["scale"][3] == pytest.approx(0.3)


def test_dataset_temporal_history_overrides_saved_runtime_interval() -> None:
    config = TRAIN._temporal_training_config(
        load_m0_mobile_config(),
        load_temporal_config(),
    )
    source = {}

    TRAIN._apply_dataset_temporal_history(
        config,
        source,
        {"history_offsets_model_ticks": [-5, 0], "history_span_s": 0.2},
    )

    assert config["data"]["history_offsets_model_ticks"] == [-5, 0]
    assert config["data"]["history_span_s"] == pytest.approx(0.2)
    assert source["history_offsets_model_ticks"] == [-5, 0]
