import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    ROUTE_TOKENS,
    SPECIAL_TOKENS,
    WaypointActionDomain,
    WaypointRoute,
    canonical_solution,
)
from conveyor_bench.conveyorvla.waypoint_model import (
    LayerwiseFlowMatchingActionHead,
    LayerwiseFlowMatchingConfig,
)
from conveyor_bench.conveyorvla.waypoint_v2 import LOCAL_CRL_GOALS
from conveyor_bench.conveyorvla.waypoint_v2_model import (
    ConveyorVLAWaypointV2Policy,
    WaypointV2AuxiliaryConfig,
    WaypointV2AuxiliaryHeads,
    WaypointV2LossConfig,
    _boundary_transition,
    _has_representable_prefix_target,
    _jittered_boundary_signed_times,
    _prefix_target_distribution,
    _soft_boundary_targets,
)


TOKEN_IDS = {token: index + 10 for index, token in enumerate(SPECIAL_TOKENS)}


def test_boundary_transition_is_recovered_from_immutable_transition_id():
    assert _boundary_transition(
        {
            "transition_id": (
                "n200:episode_000001:NAV_TO_SOURCE->PICK:source-row-42"
            )
        }
    ) == "NAV_TO_SOURCE->PICK"
    assert _boundary_transition({"transition_id": None}) is None


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[TOKEN_IDS[text]])

    def convert_tokens_to_ids(self, token):
        return TOKEN_IDS.get(token, -1)


class _Backbone(nn.Module):
    def __init__(self, cross_dim: int, layer_count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(6, 64))
        self.hidden = nn.ParameterList(
            [nn.Parameter(torch.randn(6, cross_dim) * 0.1) for _ in range(layer_count)]
        )
        self.calls = 0

    def forward(self, input_ids, labels=None, **_kwargs):
        self.calls += 1
        batch_size = input_ids.shape[0]
        logits = self.logits[None].expand(batch_size, -1, -1)
        loss = None
        if labels is not None:
            supervised = labels != -100
            loss = F.cross_entropy(logits[supervised], labels[supervised])
        hidden_states = tuple(
            value[None].expand(batch_size, -1, -1) for value in self.hidden
        )
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


class _Qwen(nn.Module):
    def __init__(self, cross_dim: int, layer_count: int) -> None:
        super().__init__()
        self.model = _Backbone(cross_dim, layer_count)
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())

    def build_waypoint_inputs(self, examples, *, supervise_solutions=True, **_kwargs):
        labels = torch.full((len(examples), 6), -100, dtype=torch.long)
        for index, example in enumerate(examples):
            route = WaypointRoute(example["route"])
            labels[index, 1] = TOKEN_IDS[
                "<|pred_done|>" if route is WaypointRoute.DONE else "<|pred_action|>"
            ]
            if route is not WaypointRoute.DONE:
                labels[index, 2] = TOKEN_IDS[ROUTE_TOKENS[route]]
                labels[index, 3] = TOKEN_IDS["<|subtask|>"]
                labels[index, 4] = TOKEN_IDS["<|end_subtask|>"]
        result = {
            "input_ids": torch.zeros_like(labels),
            "attention_mask": torch.ones_like(labels),
        }
        if supervise_solutions:
            result["labels"] = labels
        return result

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def enable_full_finetuning(self):
        self.requires_grad_(True)


def _head(action_dim: int) -> LayerwiseFlowMatchingActionHead:
    return LayerwiseFlowMatchingActionHead(
        LayerwiseFlowMatchingConfig(
            action_dim=action_dim,
            cross_attention_dim=8,
            hidden_size=8,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            max_seq_len=32,
            num_inference_timesteps=4,
        )
    )


def _example(route: WaypointRoute, index: int) -> dict:
    active = route is not WaypointRoute.DONE
    width = 3 if route in {WaypointRoute.NAV_TO_SOURCE, WaypointRoute.NAV_TO_TARGET} else 7
    domain = "NAVIGATION" if width == 3 else "MANIPULATION"
    boundary_class = ("BEFORE", "AFTER", "INTERIOR")[index % 3]
    transition = boundary_class != "INTERIOR"
    transition_name = (
        "NAV_TO_SOURCE->PICK"
        if index < 2
        else "PLACE->DONE"
    )
    return {
        "video": ((object(), object()), (object(), object())),
        "lang": "test task",
        "sample_id": f"episode:test:{index}",
        "solution": canonical_solution(route),
        "route": route.value,
        "action_domain": domain if active else "NONE",
        "action": (
            [[0.01 * (step + 1 + index)] * width for step in range(ACTION_HORIZON)]
            if active
            else None
        ),
        "action_valid_mask": [active] * ACTION_HORIZON,
        "boundary_class": boundary_class,
        "boundary_transition": transition_name if transition else None,
        "boundary_signed_time_s": (
            None if not transition else -0.5 if boundary_class == "BEFORE" else 0.5
        ),
        "phase_progress": index / 4.0,
        "prefix_target_k": 4 + index if active else 0,
        "original_valid_prefix_k": 4 + index if active else 0,
        "transition_id": f"episode:{transition_name}" if transition else None,
        "transition_window": transition,
        "time_to_boundary_s": float(index + 1),
        "time_to_boundary_valid": active,
        "on_policy_correction": False,
        "crl_goal_index": -1 if not active else tuple(LOCAL_CRL_GOALS).index(route),
    }


def _policy(*, repeats: int, auxiliary: bool) -> ConveyorVLAWaypointV2Policy:
    tau = {route.value: 4.0 + index for index, route in enumerate(LOCAL_CRL_GOALS)}
    aux_config = WaypointV2AuxiliaryConfig(
        cross_attention_dim=8,
        action_hidden_size=8,
        hidden_size=16,
        crl_dim=8,
        enable_boundary_progress=auxiliary,
        enable_prefix=auxiliary,
        enable_crl=auxiliary,
        tau_route_s=tau if auxiliary else None,
    )
    return ConveyorVLAWaypointV2Policy(
        _Qwen(8, 2),
        _head(3),
        _head(7),
        WaypointV2AuxiliaryHeads(aux_config),
        route_confidence_min=0.55,
        loss_config=WaypointV2LossConfig(
            repeated_diffusion_steps=repeats,
            lambda_boundary=0.2 if auxiliary else 0.0,
            lambda_progress=0.1 if auxiliary else 0.0,
            lambda_prefix=0.2 if auxiliary else 0.0,
            lambda_crl=0.1 if auxiliary else 0.0,
        ),
    )


def test_s4_uses_one_qwen_forward_and_means_four_independent_draws() -> None:
    torch.manual_seed(7)
    policy = _policy(repeats=4, auxiliary=True)
    policy.enable_v2_finetuning()
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
        _example(WaypointRoute.NAV_TO_TARGET, 2),
        _example(WaypointRoute.PLACE, 3),
        _example(WaypointRoute.DONE, 4),
    ]
    result = policy.oracle_loss(examples)
    assert policy.qwen.model.calls == 1
    assert torch.isfinite(result["loss"])
    for domain in ("navigation", "manipulation"):
        draws = torch.stack([result[f"{domain}_draw_{index}_loss"] for index in range(4)])
        assert result[f"{domain}_loss"].item() == pytest.approx(draws.mean().item())
        assert result[f"{domain}_draw_std"].item() > 0.0
    for name in ("boundary_loss", "progress_loss", "prefix_loss", "crl_loss"):
        assert torch.isfinite(result[name]) and result[name].item() >= 0.0
    result["loss"].backward()
    for module in (
        policy.qwen,
        policy.navigation_head,
        policy.manipulation_head,
        policy.auxiliary_heads.boundary_head,
        policy.auxiliary_heads.boundary_rank_head,
        policy.auxiliary_heads.progress_head,
        policy.auxiliary_heads.time_to_boundary_head,
        policy.auxiliary_heads.prefix_head,
        policy.auxiliary_heads.crl_state,
        policy.auxiliary_heads.crl_goal,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_boundary_soft_labels_use_stable_small_jitter_without_hard_flip() -> None:
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
        _example(WaypointRoute.NAV_TO_TARGET, 2),
    ]
    first = _jittered_boundary_signed_times(examples)
    second = _jittered_boundary_signed_times(examples)
    assert first == second
    assert first[0] is not None and abs(first[0] + 0.5) <= 0.05
    assert first[1] is not None and abs(first[1] - 0.5) <= 0.05
    assert first[2] is None
    targets = _soft_boundary_targets(examples, first, torch.zeros(()))
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(3))
    assert targets[0, 0] > targets[0, 2]
    assert targets[1, 2] > targets[1, 0]
    torch.testing.assert_close(targets[2], torch.tensor((0.0, 1.0, 0.0)))


def test_transition_tokens_are_removed_from_hard_qwen_ce() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
        _example(WaypointRoute.NAV_TO_TARGET, 2),
        _example(WaypointRoute.PLACE, 3),
        _example(WaypointRoute.DONE, 4),
    ]
    labels = policy.qwen.build_waypoint_inputs(examples)["labels"]
    masked = policy._mask_hard_transition_tokens(examples, labels)
    token_ids = policy.router.token_ids

    for row, route in ((0, WaypointRoute.NAV_TO_SOURCE), (1, WaypointRoute.PICK)):
        route_index = tuple(LOCAL_CRL_GOALS).index(route)
        position = (labels[row] == token_ids.route_ids[route_index]).nonzero().item()
        assert masked[row, position].item() == -100
        assert masked[row, 1].item() == token_ids.pred_action
    torch.testing.assert_close(masked[2], labels[2])
    for row in (3, 4):
        assert masked[row, 1].item() == -100
    place_index = tuple(LOCAL_CRL_GOALS).index(WaypointRoute.PLACE)
    assert masked[3, 2].item() == token_ids.route_ids[place_index]


def test_transition_route_loss_uses_continuous_old_new_targets() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    example = _example(WaypointRoute.NAV_TO_SOURCE, 0)
    labels = policy.qwen.build_waypoint_inputs([example])["labels"]
    logits = torch.zeros((1, 6, 64), requires_grad=True)
    route_position = 2
    old_index = tuple(LOCAL_CRL_GOALS).index(WaypointRoute.NAV_TO_SOURCE)
    new_index = tuple(LOCAL_CRL_GOALS).index(WaypointRoute.PICK)
    route_ids = torch.tensor(policy.router.token_ids.route_ids)
    logits.data[0, route_position - 1, route_ids[old_index]] = 2.0
    _, _, active_route_loss = policy._route_token_loss([example], labels, logits)

    signed = _jittered_boundary_signed_times([example])[0]
    new_probability = torch.sigmoid(torch.tensor(float(signed) / 0.2))
    candidate_logits = logits[0, route_position - 1].index_select(0, route_ids)
    log_probs = F.log_softmax(candidate_logits, dim=-1)
    expected = -(
        (1.0 - new_probability) * log_probs[old_index]
        + new_probability * log_probs[new_index]
    )
    torch.testing.assert_close(active_route_loss, expected)
    assert 0.0 < new_probability.item() < 1.0


def test_zero_length_original_prefix_is_not_mislabeled_as_k_one() -> None:
    example = _example(WaypointRoute.NAV_TO_SOURCE, 0)
    example["original_valid_prefix_k"] = 0
    example["prefix_target_k"] = 1
    assert not _has_representable_prefix_target(example)

    heads = WaypointV2AuxiliaryHeads(
        WaypointV2AuxiliaryConfig(
            cross_attention_dim=8,
            action_hidden_size=8,
            hidden_size=16,
            crl_dim=8,
            enable_prefix=True,
        )
    )
    result = heads.losses(
        torch.randn(1, 8),
        [example],
        predicted_route_probabilities=torch.full((1, 4), 0.25),
        predicted_actions=torch.zeros(1, ACTION_HORIZON, 7),
        fm_action_features=torch.zeros(1, ACTION_HORIZON, 8),
    )
    assert result["prefix_loss"].item() == 0.0
    assert result["prefix_mae_k"].item() == 0.0
    assert result["prefix_overrun_rate"].item() == 0.0


def test_b1_freezes_all_auxiliary_parameters() -> None:
    policy = _policy(repeats=1, auxiliary=False)
    policy.enable_v2_finetuning()
    assert not any(
        parameter.requires_grad for parameter in policy.auxiliary_heads.parameters()
    )
    result = policy.oracle_loss(
        [
            _example(WaypointRoute.NAV_TO_SOURCE, 0),
            _example(WaypointRoute.PICK, 1),
        ]
    )
    assert result["boundary_loss"].item() == 0.0
    assert result["prefix_loss"].item() == 0.0
    result["loss"].backward()
    assert all(
        parameter.grad is None for parameter in policy.auxiliary_heads.parameters()
    )


def test_prefix_targets_reward_long_safe_prefix_and_penalize_overrun() -> None:
    targets = torch.tensor([5, 5])
    transition = torch.tensor([False, True])
    distribution = _prefix_target_distribution(targets, transition, torch.float32)
    assert distribution.shape == (2, ACTION_HORIZON)
    assert distribution[0].argmax().item() + 1 == 5
    assert distribution[0, 4] > distribution[0, 3] > distribution[0, 2]
    assert distribution[0, 5] < distribution[0, 0]
    assert distribution[1, 3] == pytest.approx(distribution[1, 4])
    assert distribution[1, 4] == pytest.approx(distribution[1, 5])


def test_prefix_training_uses_model_actions_without_consuming_fm_rng() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
    ]
    layers = tuple(torch.randn(2, 6, 8) for _ in range(2))
    attention_mask = torch.ones(2, 6, dtype=torch.long)
    reference = layers[-1].mean(dim=1)
    rng_before = torch.random.get_rng_state()
    actions, features = policy._training_prefix_inputs(
        examples, layers, attention_mask, reference
    )
    rng_after = torch.random.get_rng_state()

    changed = [dict(example) for example in examples]
    for example in changed:
        width = len(example["action"][0])
        example["action"] = [[99.0] * width for _ in range(ACTION_HORIZON)]
    changed_actions, changed_features = policy._training_prefix_inputs(
        changed, layers, attention_mask, reference
    )

    torch.testing.assert_close(actions, changed_actions)
    torch.testing.assert_close(features, changed_features)
    torch.testing.assert_close(rng_before, rng_after)
    assert actions.shape == (2, ACTION_HORIZON, 7)
    assert features.shape == (2, ACTION_HORIZON, 8)


def test_fixed_validation_bank_is_order_seed_and_training_draw_independent() -> None:
    torch.manual_seed(19)
    s1 = _policy(repeats=1, auxiliary=False)
    s4 = _policy(repeats=4, auxiliary=False)
    s4.load_state_dict(s1.state_dict())
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
        _example(WaypointRoute.NAV_TO_TARGET, 2),
        _example(WaypointRoute.PLACE, 3),
    ]
    first = s1.fixed_bank_fm_losses(examples, bank_seed=20260822)
    torch.manual_seed(999)
    reordered = s1.fixed_bank_fm_losses(
        list(reversed(examples)), bank_seed=20260822
    )
    four_draw_training = s4.fixed_bank_fm_losses(examples, bank_seed=20260822)
    for domain in ("navigation", "manipulation"):
        key = f"{domain}_fixed_bank_loss"
        assert first[key].item() == pytest.approx(reordered[key].item())
        assert first[key].item() == pytest.approx(four_draw_training[key].item())
        assert first[f"{domain}_fixed_bank_draw_std"].item() > 0.0
    assert first["fixed_bank_draws"] == 4


def test_training_prefix_policy_actions_are_stop_gradient() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
    ]
    layers = tuple(
        torch.randn(2, 3, 8, requires_grad=True) for _ in range(2)
    )
    actions, features = policy._training_prefix_inputs(
        examples,
        layers,
        torch.ones(2, 3, dtype=torch.long),
        torch.zeros(2, 8),
    )
    assert actions.shape == (2, ACTION_HORIZON, 7)
    assert features.shape == (2, ACTION_HORIZON, 8)
    assert not actions.requires_grad
    assert not features.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device alignment test")
def test_crl_goal_buffers_follow_sharded_embedding_device() -> None:
    config = WaypointV2AuxiliaryConfig(
        cross_attention_dim=8,
        action_hidden_size=8,
        hidden_size=16,
        crl_dim=8,
        enable_crl=True,
        tau_route_s={route.value: 4.0 for route in LOCAL_CRL_GOALS},
    )
    heads = WaypointV2AuxiliaryHeads(config).cuda()
    heads.goal_bytes = heads.goal_bytes.cpu()
    heads.goal_mask = heads.goal_mask.cpu()
    goals = heads._goal_embeddings()
    assert goals.device.type == "cuda"
    assert goals.shape == (len(LOCAL_CRL_GOALS), config.crl_dim)


def test_oracle_crl_diagnostics_are_aligned_and_finite() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    examples = [
        _example(WaypointRoute.NAV_TO_SOURCE, 0),
        _example(WaypointRoute.PICK, 1),
        _example(WaypointRoute.NAV_TO_TARGET, 2),
        _example(WaypointRoute.PLACE, 3),
        _example(WaypointRoute.DONE, 4),
    ]
    diagnostics = policy.oracle_crl_diagnostics(examples)
    assert diagnostics[-1] is None
    for row in diagnostics[:-1]:
        assert row is not None
        assert set(row) == {
            "correct_goal_similarity",
            "wrong_goal_max_similarity",
            "goal_margin",
            "shuffled_action_goal_similarity",
            "action_shuffle_drop",
        }
        assert all(math.isfinite(value) for value in row.values())


def test_oracle_crl_diagnostics_keep_zero3_module_order_for_done_batches() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    calls = {"state": 0, "action": 0, "goal": 0}
    hooks = [
        policy.auxiliary_heads.crl_state.register_forward_hook(
            lambda *_args: calls.__setitem__("state", calls["state"] + 1)
        ),
        policy.auxiliary_heads.crl_action.register_forward_hook(
            lambda *_args: calls.__setitem__("action", calls["action"] + 1)
        ),
        policy.auxiliary_heads.crl_goal.register_forward_hook(
            lambda *_args: calls.__setitem__("goal", calls["goal"] + 1)
        ),
    ]
    try:
        diagnostics = policy.oracle_crl_diagnostics(
            [_example(WaypointRoute.DONE, 0), _example(WaypointRoute.DONE, 1)]
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert diagnostics == (None, None)
    assert calls == {"state": 2, "action": 2, "goal": 1}


def test_prefix_prediction_keeps_both_zero3_action_encoders_aligned() -> None:
    policy = _policy(repeats=1, auxiliary=True)
    calls = {"navigation": 0, "manipulation": 0}
    hooks = [
        policy.navigation_head.action_encoder.register_forward_hook(
            lambda *_args: calls.__setitem__(
                "navigation", calls["navigation"] + 1
            )
        ),
        policy.manipulation_head.action_encoder.register_forward_hook(
            lambda *_args: calls.__setitem__(
                "manipulation", calls["manipulation"] + 1
            )
        ),
    ]
    try:
        features = policy._prediction_fm_action_features(
            [SimpleNamespace(action_domain=WaypointActionDomain.NAVIGATION)],
            [0],
            [tuple((0.0, 0.0, 0.0) for _ in range(ACTION_HORIZON))],
            torch.zeros((1, 8)),
        )
    finally:
        for hook in hooks:
            hook.remove()
    assert features.shape == (1, ACTION_HORIZON, 8)
    assert calls == {"navigation": 1, "manipulation": 1}
