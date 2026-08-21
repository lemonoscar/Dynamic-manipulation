from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("safetensors")
from torch import nn

from scripts.train_waypoint import (
    DomainBalancedSampler,
    _balanced_subset_indices,
    _episode_subset_indices,
    _load_config,
    _optimizer,
    _resume_data_position,
    _validate_accumulation_config,
    _validate_accumulation_runtime,
    _validate_args,
    _v2_row_sample_weights,
    build_parser,
)


class _FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen = nn.Module()
        self.qwen.model = nn.Module()
        self.qwen.model.core = nn.Linear(2, 2)
        self.qwen.model.embed_tokens = nn.Embedding(4, 2)
        self.qwen.model.lm_head = nn.Linear(2, 4, bias=False)
        self.navigation_head = nn.Linear(2, 3)
        self.manipulation_head = nn.Linear(2, 7)


def _args(tmp_path: Path) -> Namespace:
    dataset = tmp_path / "dataset"
    model_root = tmp_path / "models"
    (model_root / "Qwen3-VL-4B-Instruct").mkdir(parents=True, exist_ok=True)
    dataset.mkdir(exist_ok=True)
    return Namespace(
        dataset_root=dataset,
        output_dir=tmp_path / "output",
        model_root=model_root,
        config=Path("configs/waypoint_v1.json"),
        resume_from=None,
        max_steps=1000,
        batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        save_interval_steps=100,
        save_first_checkpoint_step=20,
        log_interval_steps=1,
        num_workers=0,
        limit_train_rows=0,
        limit_train_episodes=0,
        attention_implementation="sdpa",
        seed=1,
    )


def test_waypoint_training_cli_has_no_legacy_checkpoint_or_state_argument():
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}
    assert "initial_action_checkpoint" not in destinations
    assert "state" not in destinations
    assert {
        "dataset_root",
        "output_dir",
        "model_root",
        "config",
        "resume_from",
    } <= destinations
    assert parser.get_default("save_interval_steps") == 500


def test_production_config_is_fixed_and_validated(tmp_path: Path):
    config = _load_config(Path("configs/waypoint_v1.json"))
    _validate_args(_args(tmp_path), config)
    assert config["action_model"]["num_layers"] == 16
    assert config["action_model"]["state_encoder"] is False
    assert config["action_model"]["shared_parameters"] is False
    changed = {**config, "action_model": {**config["action_model"], "hidden_size": 768}}
    with pytest.raises(Exception, match="contract was modified"):
        _validate_args(_args(tmp_path), changed)


def test_v2_s1_s4_configs_change_only_independent_fm_draw_count(tmp_path: Path):
    s1 = _load_config(Path("configs/waypoint_v2_b1_s1.json"))
    s4 = _load_config(Path("configs/waypoint_v2_b1_s4.json"))
    assert s1["action_model"]["num_inference_timesteps"] == 4
    assert s4["action_model"]["num_inference_timesteps"] == 4
    left = {**s1, "loss": {**s1["loss"], "repeated_diffusion_steps": 4}}
    assert left == s4
    _validate_args(_args(tmp_path), s1)
    _validate_args(_args(tmp_path), s4)


def test_training_subset_is_deterministic_and_covers_routes_and_boundaries():
    routes = [
        "NAV_TO_SOURCE",
        "PICK",
        "NAV_TO_TARGET",
        "PLACE",
        "DONE",
    ] * 8
    boundaries = [None] * len(routes)
    for index, name in enumerate(("NAV_PICK", "PICK_NAV", "NAV_PLACE", "PLACE_DONE")):
        boundaries[index + 5] = name
    dataset = type(
        "Dataset",
        (),
        {
            "routes": routes,
            "boundaries": boundaries,
            "__len__": lambda self: len(routes),
        },
    )()
    selected = _balanced_subset_indices(dataset, 32)
    assert selected == _balanced_subset_indices(dataset, 32)
    assert len(selected) == len(set(selected)) == 32
    assert {routes[index] for index in selected} == set(routes)
    assert {boundaries[index] for index in selected if boundaries[index]} == {
        "NAV_PICK",
        "PICK_NAV",
        "NAV_PLACE",
        "PLACE_DONE",
    }


def test_domain_balanced_sampler_keeps_both_experts_in_every_batch():
    routes = [
        "NAV_TO_SOURCE",
        "PICK",
        "NAV_TO_TARGET",
        "PLACE",
        "DONE",
    ] * 6
    sampler = DomainBalancedSampler(
        routes,
        [1.0] * len(routes),
        batch_size=3,
        seed=9,
    )
    indices = list(sampler)
    next_indices = list(sampler)
    assert indices != next_indices
    for epoch_indices in (indices, next_indices):
        for start in range(0, len(epoch_indices), 3):
            batch_routes = {routes[index] for index in epoch_indices[start : start + 3]}
            assert batch_routes.intersection({"NAV_TO_SOURCE", "NAV_TO_TARGET"})
            assert batch_routes.intersection({"PICK", "PLACE"})
            assert "DONE" in batch_routes


def test_gradient_accumulation_has_one_runtime_source_of_truth():
    accelerator = SimpleNamespace(
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                deepspeed_config={"gradient_accumulation_steps": 2}
            )
        ),
        gradient_accumulation_steps=2,
    )
    _validate_accumulation_config(accelerator, 2)
    _validate_accumulation_runtime(
        accelerator,
        SimpleNamespace(gradient_accumulation_steps=lambda: 2),
        2,
    )
    accelerator.state.deepspeed_plugin.deepspeed_config[
        "gradient_accumulation_steps"
    ] = 8
    with pytest.raises(Exception, match="conflicts"):
        _validate_accumulation_config(accelerator, 2)
    with pytest.raises(Exception, match="engine resolved"):
        _validate_accumulation_runtime(
            accelerator,
            SimpleNamespace(gradient_accumulation_steps=lambda: 8),
            2,
        )


def test_resume_data_position_skips_exact_completed_micro_batches():
    assert _resume_data_position(9051, 2, 1000) == {
        "global_step": 1000,
        "loader_micro_batches_per_pass": 9051,
        "optimizer_steps_per_loader_pass": 4526,
        "completed_loader_passes": 0,
        "optimizer_step_in_pass": 1000,
        "skipped_micro_batches": 2000,
    }
    assert _resume_data_position(9051, 2, 4526)["completed_loader_passes"] == 1
    assert _resume_data_position(9051, 2, 4526)["skipped_micro_batches"] == 0


def test_optimizer_has_exact_qwen_nav_arm_groups_without_overlap():
    model = _FakePolicy()
    config = _load_config(Path("configs/waypoint_v1.json"))
    optimizer, report = _optimizer(model, config)
    assert [group["name"] for group in optimizer.param_groups] == [
        "vlm_core",
        "vlm_embeddings_lm_head",
        "navigation_head",
        "manipulation_head",
    ]
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters()
    }
    assert all(group["parameters"] > 0 for group in report)


def test_v2_optimizer_adds_only_enabled_auxiliary_parameters():
    model = _FakePolicy()
    model.auxiliary_heads = nn.Linear(2, 2)
    config = _load_config(Path("configs/waypoint_v2_overfit_all_s1_full.json"))
    optimizer, _report = _optimizer(model, config)
    assert [group["name"] for group in optimizer.param_groups] == [
        "vlm_core",
        "vlm_embeddings_lm_head",
        "navigation_head",
        "manipulation_head",
        "auxiliary_heads",
    ]
    model.auxiliary_heads.requires_grad_(False)
    optimizer, _report = _optimizer(model, config)
    assert [group["name"] for group in optimizer.param_groups][-1] == "manipulation_head"


def test_v2_sampler_weights_events_and_episode_subset_without_state():
    routes = []
    boundaries = []
    transition_ids = []
    episodes = []
    progress = []
    for episode_index in range(12):
        episode = f"episode-{episode_index:02d}"
        for route_index, route in enumerate(
            ("NAV_TO_SOURCE", "PICK", "NAV_TO_TARGET", "PLACE", "DONE")
        ):
            routes.append(route)
            episodes.append(episode)
            progress.append(route_index / 4.0)
            if route_index < 4:
                boundaries.append(f"transition-{route_index}")
                transition_ids.append(f"{episode}:transition-{route_index}")
            else:
                boundaries.append(None)
                transition_ids.append(None)
    dataset = type(
        "V2Dataset",
        (),
        {
            "routes": routes,
            "boundaries": boundaries,
            "transition_ids": transition_ids,
            "source_episode_ids": episodes,
            "phase_progress": progress,
            "__len__": lambda self: len(routes),
        },
    )()
    selected = _episode_subset_indices(dataset, 8)
    assert len({episodes[index] for index in selected}) == 8
    weights = _v2_row_sample_weights(dataset, selected)
    assert len(weights) == len(selected)
    assert all(value > 0.0 for value in weights)
