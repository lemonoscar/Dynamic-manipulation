import copy
import hashlib
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
    _deepspeed_zero_stage,
    _episode_subset_indices,
    _load_config,
    _optimizer,
    _resume_binding,
    _resume_data_position,
    _self_conditioned_weight,
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
        resume_extension=False,
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
        "resume_extension",
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


def test_v2_b2_b3_b4_configs_add_exactly_one_rollbackable_mechanism(
    tmp_path: Path,
):
    b1 = _load_config(Path("configs/waypoint_v2_b1_s1.json"))
    b2 = _load_config(Path("configs/waypoint_v2_b2_s1.json"))
    b3 = _load_config(Path("configs/waypoint_v2_b3_s1.json"))
    b4 = _load_config(Path("configs/waypoint_v2_b4_s1.json"))

    expected = copy.deepcopy(b1)
    expected["auxiliary"]["enable_boundary_progress"] = True
    expected["loss"]["lambda_boundary"] = 0.2
    expected["loss"]["lambda_progress"] = 0.1
    assert expected == b2

    expected = copy.deepcopy(b2)
    expected["auxiliary"]["enable_prefix"] = True
    expected["loss"]["lambda_prefix"] = 0.2
    assert expected == b3

    expected = copy.deepcopy(b3)
    expected["auxiliary"]["enable_crl"] = True
    expected["auxiliary"]["tau_route_s"] = {
        "NAV_TO_SOURCE": 6.199999999999999,
        "PICK": 7.199999999999999,
        "NAV_TO_TARGET": 20.2,
        "PLACE": 6.400000000000006,
    }
    expected["loss"]["lambda_crl"] = 0.1
    assert expected == b4
    for config in (b2, b3, b4):
        _validate_args(_args(tmp_path), config)


def test_v2_command_gripper_config_changes_only_dataset_identity(tmp_path: Path):
    legacy = _load_config(Path("configs/waypoint_v2_b2_s1.json"))
    command = _load_config(
        Path("configs/waypoint_v2_b2_s1_command_gripper.json")
    )
    expected = copy.deepcopy(legacy)
    expected["dataset_schema_version"] = (
        "conveyorvla-waypoint-dense-transition-v2-command-gripper-v1"
    )
    assert command == expected
    _validate_args(_args(tmp_path), command)


def test_v2_command_gripper_s4_changes_only_fm_draw_count(tmp_path: Path):
    s1 = _load_config(
        Path("configs/waypoint_v2_b2_s1_command_gripper.json")
    )
    s4 = _load_config(
        Path("configs/waypoint_v2_b2_s4_command_gripper.json")
    )
    expected = copy.deepcopy(s1)
    expected["loss"]["repeated_diffusion_steps"] = 4
    assert s4 == expected
    assert s4["action_model"]["num_inference_timesteps"] == 4
    _validate_args(_args(tmp_path), s4)


def test_v2_command_gripper_s4_delays_self_conditioning_to_step_1500(
    tmp_path: Path,
):
    base = _load_config(
        Path("configs/waypoint_v2_b2_s4_command_gripper.json")
    )
    delayed = _load_config(
        Path("configs/waypoint_v2_b2_s4_command_gripper_self1500.json")
    )
    expected = copy.deepcopy(base)
    expected["loss"]["lambda_self_schedule"] = {
        "zero_until_step": 1500,
        "linear_to_step": 2550,
        "maximum": 0.5,
    }
    assert delayed == expected
    args = _args(tmp_path)
    args.max_steps = 3000
    _validate_args(args, delayed)
    schedule = delayed["loss"]["lambda_self_schedule"]
    assert _self_conditioned_weight(0, 3000, schedule) == 0.0
    assert _self_conditioned_weight(1499, 3000, schedule) == 0.0
    assert _self_conditioned_weight(1500, 3000, schedule) == pytest.approx(
        0.5 / 1050
    )
    assert _self_conditioned_weight(2024, 3000, schedule) == pytest.approx(0.25)
    assert _self_conditioned_weight(2549, 3000, schedule) == pytest.approx(0.5)
    legacy_schedule = base["loss"]["lambda_self_schedule"]
    assert _self_conditioned_weight(150, 3000, legacy_schedule) == 0.0
    assert _self_conditioned_weight(151, 3000, legacy_schedule) == pytest.approx(
        0.5 / 1050
    )

    args.max_steps = 2000
    with pytest.raises(Exception, match="step schedule is invalid"):
        _validate_args(args, delayed)


def test_v2_b5_adds_only_manifest_bound_on_policy_sampling(tmp_path: Path):
    b4 = _load_config(Path("configs/waypoint_v2_b4_s1.json"))
    b5 = _load_config(Path("configs/waypoint_v2_b5_s1.json"))
    expected = copy.deepcopy(b4)
    expected["sampling"] = {
        "original_success": 0.60,
        "transition_window": 0.25,
        "on_policy_correction": 0.15,
    }
    assert expected == b5
    _validate_args(_args(tmp_path), b5)


def test_correction_mixture_preserves_event_pairs_and_target_ratios():
    routes = []
    transition_ids = []
    signed_times = []
    categories = []
    for event in range(40):
        left, right = (
            ("NAV_TO_SOURCE", "PICK")
            if event % 2 == 0
            else ("PICK", "NAV_TO_TARGET")
        )
        routes.extend((left, right))
        transition_ids.extend((f"event-{event}", f"event-{event}"))
        signed_times.extend((-0.2, 0.0))
        categories.extend(("transition_window", "transition_window"))
    for category in ("original_success", "on_policy_correction"):
        for _repeat in range(40):
            routes.extend(("NAV_TO_SOURCE", "PICK", "DONE"))
            transition_ids.extend((None, None, None))
            signed_times.extend((None, None, None))
            categories.extend((category, category, category))
    sampler = DomainBalancedSampler(
        routes,
        [1.0] * len(routes),
        batch_size=3,
        seed=31,
        transition_ids=transition_ids,
        boundary_signed_times=signed_times,
        mixture_categories=categories,
        mixture_fractions={
            "original_success": 0.60,
            "transition_window": 0.25,
            "on_policy_correction": 0.15,
        },
    )
    indices = list(sampler)
    counts = {category: 0 for category in set(categories)}
    paired_batches = 0
    for start in range(0, len(indices), 3):
        batch = indices[start : start + 3]
        for index in batch:
            counts[categories[index]] += 1
        ids = [transition_ids[index] for index in batch if transition_ids[index]]
        paired_batches += int(len(ids) == 2 and ids[0] == ids[1])
        batch_routes = {routes[index] for index in batch}
        assert batch_routes.intersection({"NAV_TO_SOURCE", "NAV_TO_TARGET"})
        assert batch_routes.intersection({"PICK", "PLACE"})
    total = len(indices)
    assert counts["original_success"] / total == pytest.approx(0.60, abs=0.08)
    assert counts["transition_window"] / total == pytest.approx(0.25, abs=0.08)
    assert counts["on_policy_correction"] / total == pytest.approx(0.15, abs=0.06)
    assert paired_batches > 0


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


def test_transition_sampler_emits_before_after_pairs_as_event_units():
    routes = [
        "NAV_TO_SOURCE",
        "PICK",
        "PICK",
        "NAV_TO_TARGET",
        "NAV_TO_TARGET",
        "PLACE",
        "PLACE",
        "DONE",
        "NAV_TO_SOURCE",
        "PICK",
        "DONE",
        "NAV_TO_TARGET",
    ]
    transition_ids = [
        "event-0",
        "event-0",
        "event-1",
        "event-1",
        "event-2",
        "event-2",
        "event-3",
        "event-3",
        None,
        None,
        None,
        None,
    ]
    signed_times = [-0.2, 0.0, -0.2, 0.0, -0.2, 0.0, -0.2, 0.0] + [None] * 4
    sampler = DomainBalancedSampler(
        routes,
        [1.0] * len(routes),
        batch_size=3,
        seed=19,
        transition_ids=transition_ids,
        boundary_signed_times=signed_times,
    )
    indices = list(sampler)
    for batch_number, start in enumerate(range(0, len(indices), 3)):
        batch = indices[start : start + 3]
        batch_routes = {routes[index] for index in batch}
        assert batch_routes.intersection({"NAV_TO_SOURCE", "NAV_TO_TARGET"})
        assert batch_routes.intersection({"PICK", "PLACE"})
        assert "DONE" in batch_routes
        if batch_number % 2 == 0:
            event_counts = {}
            for index in batch:
                event = transition_ids[index]
                if event is not None:
                    event_counts[event] = event_counts.get(event, 0) + 1
            assert 2 in event_counts.values()


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


def test_deepspeed_zero_stage_is_resolved_from_runtime_config():
    accelerator = SimpleNamespace(
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(
                deepspeed_config={"zero_optimization": {"stage": 2}}
            )
        )
    )

    assert _deepspeed_zero_stage(accelerator) == 2
    accelerator.state.deepspeed_plugin = None
    assert _deepspeed_zero_stage(accelerator) is None


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


def test_resume_extension_allows_only_longer_schedule_and_batch_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from scripts import check_waypoint_checkpoint

    args = _args(tmp_path)
    args.config = Path("configs/waypoint_v2_overfit_all_s1_fast_aux.json").resolve()
    args.resume_extension = True
    args.max_steps = 2000
    args.batch_size = 4
    args.gradient_accumulation_steps = 4
    args.limit_train_episodes = 8
    checkpoint = tmp_path / "parent" / "checkpoints" / "step_000500"
    checkpoint.mkdir(parents=True)
    args.resume_from = checkpoint
    (checkpoint / "trainer_state.json").write_text('{"global_step": 500}\n')
    (checkpoint / "waypoint_checkpoint_manifest.json").write_text("{}\n")
    (checkpoint.parents[1] / "resolved_run.json").write_text("{}\n")
    config = _load_config(args.config)
    manifest = {
        "model_contract_id": config["model_contract_id"],
        "dataset_schema_version": config["dataset_schema_version"],
        "dataset_manifest_sha256": "dataset-sha",
        "resolved_policy_config_sha256": hashlib.sha256(
            args.config.read_bytes()
        ).hexdigest(),
        "global_step": 500,
        "source_git": {"commit": "parent"},
        "special_token_ids": {"pred_action": 1},
    }
    resolved = {
        "model_root": str(args.model_root.resolve()),
        "world_size": 4,
        "warmup_steps": 20,
        "arguments": {
            "max_steps": 500,
            "batch_size": 3,
            "gradient_accumulation_steps": 2,
            "limit_train_rows": 0,
            "limit_train_episodes": 8,
            "attention_implementation": "sdpa",
            "seed": 1,
        },
    }
    monkeypatch.setattr(
        check_waypoint_checkpoint,
        "_validate_binding",
        lambda _checkpoint: (manifest, resolved, args.dataset_root),
    )

    result = _resume_binding(
        args,
        config,
        {"manifest_sha256": "dataset-sha"},
        SimpleNamespace(num_processes=4),
        warmup_steps=20,
    )

    assert result is not None
    assert result["global_step"] == 500
    assert result["extension"] is True
    assert result["scheduler_rebased_to_max_steps"] == 2000
    assert result["data_iteration_restart"] is True
    assert result["self_schedule_progress_floor"] == 1.0
    assert set(result["contract_changes"]) == {
        "batch_size",
        "gradient_accumulation_steps",
        "max_steps",
    }

    args.resume_extension = False
    with pytest.raises(Exception, match="training contract changed"):
        _resume_binding(
            args,
            config,
            {"manifest_sha256": "dataset-sha"},
            SimpleNamespace(num_processes=4),
            warmup_steps=20,
        )


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


def test_overfit_fast_aux_config_changes_only_auxiliary_head_lr():
    regular = _load_config(
        Path("configs/waypoint_v2_overfit_all_s1_full.json")
    )
    fast = _load_config(
        Path("configs/waypoint_v2_overfit_all_s1_fast_aux.json")
    )
    expected = copy.deepcopy(regular)
    expected["optimization"]["auxiliary_head_learning_rate"] = 2.0e-4
    assert expected == fast


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
