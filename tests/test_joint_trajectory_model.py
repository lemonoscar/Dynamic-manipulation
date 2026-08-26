from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    ACTIVE_SPECIAL_TOKENS,
    ROUTE_TOKENS,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    canonical_solution,
)
from conveyor_bench.conveyorvla.joint_trajectory_model import (
    ConveyorVLAJointTrajectoryPolicy,
    JointTrajectoryAuxiliaryHeads,
    JointTrajectoryExpertConfig,
    JointTrajectoryFlowMatchingExpert,
    JointTrajectoryLossConfig,
    JointTrajectoryRouter,
    _warmstart_source_key,
    selective_warmstart,
)


TOKENS = (*ACTIVE_SPECIAL_TOKENS, "<|pred_done|>")
TOKEN_IDS = {token: index + 10 for index, token in enumerate(TOKENS)}
TEXT_ID = 60


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[TOKEN_IDS[text]])

    def convert_tokens_to_ids(self, token):
        return TOKEN_IDS.get(token, -1)

    def decode(self, token_ids, **_kwargs):
        return "move carefully" if list(token_ids) == [TEXT_ID] else "invalid"


class _Backbone(nn.Module):
    def __init__(self, cross_dim=8, layers=2):
        super().__init__()
        self.logits_table = nn.Parameter(torch.zeros(16, 96))
        self.hidden = nn.ParameterList(
            [nn.Parameter(torch.randn(16, cross_dim) * 0.05) for _ in range(layers)]
        )
        self.visual = nn.Linear(1, 1)
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids, **_kwargs):
        batch, tokens = input_ids.shape
        logits = self.logits_table[:tokens][None].repeat(batch, 1, 1)
        # The router reads the last position after forced ACTION.
        route_id = TOKEN_IDS[ROUTE_TOKENS[JointTrajectoryRoute.NAV_TO_SOURCE]]
        marker = torch.zeros_like(logits)
        marker[:, -1, route_id] = 8.0
        logits = logits + marker
        hidden_states = tuple(
            value[:tokens][None].expand(batch, -1, -1) for value in self.hidden
        )
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    def generate(self, input_ids, **_kwargs):
        suffix = torch.tensor(
            [[TEXT_ID, TOKEN_IDS["<|end_subtask|>"]]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)
        return torch.cat((input_ids, suffix), dim=1)

    def gradient_checkpointing_enable(self, **_kwargs):
        return None

    def enable_input_require_grads(self):
        return None


class _Qwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())
        self.seen_mani_state = False

    def build_joint_trajectory_inputs(
        self, examples, *, solutions=None, supervise_solutions=True
    ):
        self.seen_mani_state = self.seen_mani_state or any(
            example.get("mani_state") is not None for example in examples
        )
        length = 3 if solutions is None else 6
        result = {
            "input_ids": torch.zeros(len(examples), length, dtype=torch.long),
            "attention_mask": torch.ones(len(examples), length, dtype=torch.long),
        }
        if solutions is not None and supervise_solutions:
            labels = torch.full((len(examples), length), -100, dtype=torch.long)
            for index, example in enumerate(examples):
                route = JointTrajectoryRoute(example["route"])
                labels[index, 1] = TOKEN_IDS["<|pred_action|>"]
                labels[index, 2] = TOKEN_IDS[ROUTE_TOKENS[route]]
                labels[index, 3] = TOKEN_IDS["<|subtask|>"]
                labels[index, 4] = TEXT_ID
                labels[index, 5] = TOKEN_IDS["<|end_subtask|>"]
            result["labels"] = labels
        return result

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def enable_full_finetuning(self):
        self.requires_grad_(True)
        self.model.config.use_cache = False


def _expert(action_dim, state_dim):
    return JointTrajectoryFlowMatchingExpert(
        JointTrajectoryExpertConfig(
            action_dim=action_dim,
            state_dim=state_dim,
            cross_attention_dim=8,
            hidden_size=8,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            max_seq_len=16,
            num_inference_timesteps=10,
        )
    )


def _policy():
    return ConveyorVLAJointTrajectoryPolicy(
        _Qwen(),
        _expert(3, 0),
        _expert(7, 13),
        JointTrajectoryAuxiliaryHeads(8, 8),
        loss_config=JointTrajectoryLossConfig(),
    )


def _example(route, index, *, transition=None, signed=None, episode=None):
    domain = (
        JointTrajectoryDomain.NAVIGATION
        if route in {JointTrajectoryRoute.NAV_TO_SOURCE, JointTrajectoryRoute.NAV_TO_TARGET}
        else JointTrajectoryDomain.MANIPULATION
    )
    return {
        "video": ((object(), object()), (object(), object())),
        "lang": "prompt",
        "solution": canonical_solution(route),
        "route": route.value,
        "action_domain": domain.value,
        "action": [[0.01 * (step + 1)] * (3 if domain is JointTrajectoryDomain.NAVIGATION else 7) for step in range(ACTION_HORIZON)],
        "action_valid_mask": [True] * ACTION_HORIZON,
        "mani_state": None if domain is JointTrajectoryDomain.NAVIGATION else [0.0] * 13,
        "sample_id": f"sample-{index}",
        "episode_id": episode or f"episode-{index}",
        "transition_id": None if transition is None else f"event-{transition}",
        "boundary_transition": transition,
        "boundary_signed_time_s": signed,
        "transition_window": transition is not None,
        "physical_progress": 0.2 * (index + 1),
        "physical_progress_valid": True,
        "route_importance_weight": 1.0,
    }


def test_experts_have_self_cross_ffn_no_future_tokens_and_strict_state_boundary():
    nav = _expert(3, 0)
    mani = _expert(7, 13)
    assert not hasattr(nav, "future_tokens") and not hasattr(mani, "future_tokens")
    for block in (*nav.blocks, *mani.blocks):
        assert hasattr(block, "self_attention")
        assert hasattr(block, "cross_attention")
        assert hasattr(block, "ff")
    layers = (torch.randn(2, 6, 8), torch.randn(2, 6, 8))
    attention = torch.ones(2, 6, dtype=torch.long)
    nav_action = torch.randn(2, 10, 3)
    with pytest.raises(ValueError, match="must not receive state"):
        nav(layers, nav_action, encoder_attention_mask=attention, state=torch.zeros(2, 13))
    with pytest.raises(ValueError, match="requires state"):
        mani(layers, torch.randn(2, 10, 7), encoder_attention_mask=attention)
    with pytest.raises(ValueError, match="all ten"):
        nav(
            layers,
            nav_action,
            encoder_attention_mask=attention,
            action_valid_mask=torch.tensor([[True] * 9 + [False]] * 2),
        )


def test_stage_a_freezes_everything_except_action_experts_then_stage_b_unfreezes_all():
    policy = _policy()
    policy.enable_action_warmup()
    assert not any(parameter.requires_grad for parameter in policy.qwen.parameters())
    assert not any(parameter.requires_grad for parameter in policy.auxiliary_heads.parameters())
    assert all(parameter.requires_grad for parameter in policy.navigation_expert.parameters())
    assert all(parameter.requires_grad for parameter in policy.manipulation_expert.parameters())
    policy.enable_full_finetuning()
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_answer_route_boundary_progress_and_grouped_fm_are_one_coherent_loss():
    torch.manual_seed(4)
    policy = _policy()
    transition = "NAV_TO_SOURCE->PICK"
    examples = [
        _example(JointTrajectoryRoute.NAV_TO_SOURCE, 0, transition=transition, signed=-0.1, episode="pair"),
        _example(JointTrajectoryRoute.PICK, 1, transition=transition, signed=0.1, episode="pair"),
        _example(JointTrajectoryRoute.NAV_TO_TARGET, 2),
        _example(JointTrajectoryRoute.PLACE, 3),
    ]
    for example, weight in zip(examples, (2.0, 0.5, 1.5, 0.25), strict=True):
        example["route_importance_weight"] = weight
    inputs = policy.qwen.build_joint_trajectory_inputs(
        examples, solutions=[example["solution"] for example in examples]
    )
    masked = policy.answer_labels(examples, inputs["labels"])
    assert masked[0, 2] == -100 and masked[0, 4] == -100
    assert masked[2, 2] == -100 and masked[2, 4] == TEXT_ID
    qwen_output = policy.qwen(
        **{key: value for key, value in inputs.items() if key != "labels"}
    )
    route_logits, route_targets = policy.route_logits_and_targets(
        examples, inputs["labels"], qwen_output.logits
    )
    route_rows = -(route_targets * route_logits.float().log_softmax(dim=-1)).sum(dim=-1)
    expected_route_loss = (
        route_rows * torch.tensor([2.0, 0.5, 1.5, 0.25])
    ).mean()
    output = policy(examples)
    assert torch.isfinite(output["loss"])
    assert output["navigation_samples"] == 2
    assert output["manipulation_samples"] == 2
    assert output["boundary_pairs"] == 1
    torch.testing.assert_close(output["route_loss"], expected_route_loss)
    torch.testing.assert_close(output["navigation_objective"], output["navigation_loss"])
    torch.testing.assert_close(output["manipulation_objective"], output["manipulation_loss"])
    torch.testing.assert_close(
        output["boundary_objective"], 4.0 * output["boundary_loss"]
    )
    assert output["manipulation_loss"].item() == pytest.approx(
        0.75 * output["manipulation_joint_loss"].item()
        + 0.25 * output["manipulation_gripper_loss"].item()
    )
    output["loss"].backward()
    for module in (
        policy.qwen,
        policy.navigation_expert,
        policy.manipulation_expert,
        policy.auxiliary_heads,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_soft_route_target_only_uses_old_and_new_routes():
    policy = _policy()
    transition = "PICK->NAV_TO_TARGET"
    examples = [
        _example(JointTrajectoryRoute.PICK, 0, transition=transition, signed=-0.15),
        _example(JointTrajectoryRoute.NAV_TO_TARGET, 1, transition=transition, signed=0.15),
    ]
    inputs = policy.qwen.build_joint_trajectory_inputs(
        examples, solutions=[example["solution"] for example in examples]
    )
    outputs = policy.qwen(**{key: value for key, value in inputs.items() if key != "labels"})
    _logits, targets = policy.route_logits_and_targets(examples, inputs["labels"], outputs.logits)
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(2))
    assert targets[:, 0].eq(0).all() and targets[:, 3].eq(0).all()
    assert targets[0, 1] > targets[0, 2]
    assert targets[1, 2] > targets[1, 1]


def test_router_has_no_done_candidate_or_fixed_confidence_rejection():
    router = JointTrajectoryRouter(_Qwen())
    decision = router.decode([{"video": ((1, 2), (3, 4)), "lang": "prompt"}])[0]
    assert decision.valid
    assert decision.route is JointTrajectoryRoute.NAV_TO_SOURCE
    assert set(decision.route_probs) == {route.value for route in JointTrajectoryRoute}
    assert "done" not in decision.assistant_prefix.lower()
    assert not hasattr(router, "route_confidence_min")


def test_selective_warmstart_loads_qwen_cross_ffn_time_and_reinitializes_new_semantics():
    policy = _policy()
    source = {}
    for target_key, value in policy.state_dict().items():
        source_key = _warmstart_source_key(target_key)
        if source_key is not None:
            source[source_key] = torch.full_like(value, 0.125)
    source["navigation_head.future_tokens.weight"] = torch.ones(10, 8)
    source["auxiliary_heads.progress_head.0.weight"] = torch.ones(8, 8)
    report = selective_warmstart(policy, source)
    assert any(key.startswith("qwen.") for key in report.loaded)
    assert any("cross_attention" in key for key in report.loaded)
    assert any("self_attention" in key for key in report.reinitialized)
    assert any("progress_head" in key for key in report.reinitialized)
    assert any("future_tokens" in key for key in report.rejected)
    assert any("progress_head" in key for key in report.rejected)
    qwen_key = next(key for key in source if key.startswith("qwen."))
    incompatible = dict(source)
    incompatible[qwen_key] = torch.zeros(source[qwen_key].numel() + 1)
    with pytest.raises(Exception, match="incompatible Qwen"):
        selective_warmstart(_policy(), incompatible)
