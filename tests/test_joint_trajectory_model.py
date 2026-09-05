from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conveyor_bench.conveyorvla.dit import M0DiTActionHead, M0DiTConfig
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
    JointTrajectoryLossConfig,
    JointTrajectoryRouter,
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
    return M0DiTActionHead(
        M0DiTConfig(
            action_dim=action_dim,
            state_dim=state_dim,
            action_horizon=ACTION_HORIZON,
            vlm_hidden_dim=8,
            input_embedding_dim=8,
            hidden_size=12,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            max_seq_len=16,
            num_target_vision_tokens=2,
            num_inference_timesteps=4,
            dropout=0.0,
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


def test_experts_are_abot_dit_heads_with_strict_state_boundary():
    nav = _expert(3, 0)
    mani = _expert(7, 13)
    assert nav.future_tokens.num_embeddings == 2
    assert mani.future_tokens.num_embeddings == 2
    for block in (*nav.model.transformer_blocks, *mani.model.transformer_blocks):
        assert hasattr(block, "attn1")
        assert hasattr(block, "ff")
    hidden = torch.randn(2, 6, 8)
    attention = torch.ones(2, 6, dtype=torch.long)
    nav_action = torch.randn(2, 10, 3)
    with pytest.raises(ValueError, match="must not receive state"):
        nav(hidden, nav_action, torch.zeros(2, 13), encoder_attention_mask=attention)
    with pytest.raises(ValueError, match="state must have shape"):
        mani(hidden, torch.randn(2, 10, 7), None, encoder_attention_mask=attention)
    with pytest.raises(ValueError, match="true prefix"):
        nav(
            hidden,
            nav_action,
            None,
            encoder_attention_mask=attention,
            action_valid_mask=torch.tensor([[True, False, True] + [False] * 7] * 2),
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


def test_policy_conditions_action_heads_on_only_qwen_final_hidden_state():
    policy = _policy()
    seen: list[torch.Tensor] = []
    hook = policy.navigation_expert.model.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[1].detach().clone())
    )
    example = _example(JointTrajectoryRoute.NAV_TO_SOURCE, 0)

    output = policy([example])
    hook.remove()

    assert torch.isfinite(output["loss"])
    assert seen
    expected = policy.qwen.model.hidden[-1][:6].unsqueeze(0)
    torch.testing.assert_close(seen[0], expected)
