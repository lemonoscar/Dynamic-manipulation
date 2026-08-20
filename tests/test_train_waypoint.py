from argparse import Namespace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("accelerate")
pytest.importorskip("safetensors")
from torch import nn

from scripts.train_waypoint import (
    _balanced_subset_indices,
    _load_config,
    _optimizer,
    _route_class_weights,
    _validate_args,
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
    (model_root / "Qwen3-VL-4B-Instruct").mkdir(parents=True)
    dataset.mkdir()
    return Namespace(
        dataset_root=dataset,
        output_dir=tmp_path / "output",
        model_root=model_root,
        config=Path("configs/waypoint_v1.json"),
        max_steps=1000,
        batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        save_interval_steps=100,
        save_first_checkpoint_step=20,
        log_interval_steps=1,
        num_workers=0,
        limit_train_rows=0,
        attention_implementation="sdpa",
        seed=1,
    )


def test_waypoint_training_cli_has_no_legacy_checkpoint_or_state_argument():
    destinations = {action.dest for action in build_parser()._actions}
    assert "initial_action_checkpoint" not in destinations
    assert "state" not in destinations
    assert {"dataset_root", "output_dir", "model_root", "config"} <= destinations


def test_production_config_is_fixed_and_validated(tmp_path: Path):
    config = _load_config(Path("configs/waypoint_v1.json"))
    _validate_args(_args(tmp_path), config)
    assert config["action_model"]["num_layers"] == 16
    assert config["action_model"]["state_encoder"] is False
    assert config["action_model"]["shared_parameters"] is False
    changed = {**config, "action_model": {**config["action_model"], "hidden_size": 768}}
    with pytest.raises(Exception, match="contract was modified"):
        _validate_args(_args(tmp_path), changed)


def test_route_weights_are_inverse_frequency_and_require_all_routes():
    routes = (
        ["NAV_TO_SOURCE"] * 2
        + ["PICK"] * 4
        + ["NAV_TO_TARGET"] * 5
        + ["PLACE"] * 10
        + ["DONE"] * 20
    )
    weights = _route_class_weights(routes)
    assert weights["NAV_TO_SOURCE"] / weights["DONE"] == pytest.approx(10.0)
    with pytest.raises(Exception, match="missing"):
        _route_class_weights(routes[:-20])


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
