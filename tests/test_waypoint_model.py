from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    ROUTE_TOKENS,
    SPECIAL_TOKENS,
    WaypointRoute,
    canonical_solution,
)
from conveyor_bench.conveyorvla.waypoint_model import (
    ConstrainedWaypointRouter,
    ConveyorVLAWaypointPolicy,
    LayerwiseFlowMatchingActionHead,
    LayerwiseFlowMatchingConfig,
    lambda_self_schedule,
    waypoint_token_ids,
)


TOKEN_IDS = {token: index + 10 for index, token in enumerate(SPECIAL_TOKENS)}


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        if text not in TOKEN_IDS:
            raise AssertionError(f"unexpected tokenizer input: {text!r}")
        return SimpleNamespace(input_ids=[TOKEN_IDS[text]])

    def convert_tokens_to_ids(self, token):
        return TOKEN_IDS.get(token, -1)

    def decode(self, token_ids, **_kwargs):
        return "".join(
            "move now" if token_id == 30 else _token_for_id(token_id)
            for token_id in token_ids
        )


def _token_for_id(token_id):
    for token, candidate in TOKEN_IDS.items():
        if candidate == token_id:
            return token
    return f"text-{token_id}"


class _RouterBackbone(nn.Module):
    def __init__(self, *, done=False, confident=True, invalid_subtask=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.done = done
        self.confident = confident
        self.invalid_subtask = invalid_subtask

    def forward(self, input_ids, **_kwargs):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 64)
        margin = 8.0 if self.confident else 0.0
        if input_ids.shape[1] == 3:
            key = "<|pred_done|>" if self.done else "<|pred_action|>"
            logits[:, -1, TOKEN_IDS[key]] = margin
        else:
            logits[:, -1, TOKEN_IDS[ROUTE_TOKENS[WaypointRoute.NAV_TO_SOURCE]]] = margin
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, **_kwargs):
        text_id = TOKEN_IDS["<|pred_done|>"] if self.invalid_subtask else 30
        suffix = torch.tensor(
            [[text_id, TOKEN_IDS["<|end_subtask|>"]]],
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).expand(input_ids.shape[0], -1)
        return torch.cat((input_ids, suffix), dim=1)


class _RouterQwen(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())

    def build_waypoint_inputs(self, examples, **_kwargs):
        return {
            "input_ids": torch.ones(len(examples), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(examples), 3, dtype=torch.long),
        }


class _OracleBackbone(nn.Module):
    def __init__(self, cross_dim, layer_count):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(6, 64))
        self.hidden = nn.ParameterList(
            [nn.Parameter(torch.randn(6, cross_dim) * 0.1) for _ in range(layer_count)]
        )

    def forward(self, input_ids, labels, **_kwargs):
        batch_size = input_ids.shape[0]
        logits = self.logits[None].expand(batch_size, -1, -1)
        supervised = labels != -100
        loss = F.cross_entropy(logits[supervised], labels[supervised])
        hidden_states = tuple(value[None].expand(batch_size, -1, -1) for value in self.hidden)
        return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)


class _OracleQwen(nn.Module):
    def __init__(self, cross_dim, layer_count):
        super().__init__()
        self.model = _OracleBackbone(cross_dim, layer_count)
        self.processor = SimpleNamespace(tokenizer=_Tokenizer())

    def build_waypoint_inputs(self, examples, **_kwargs):
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
        return {
            "input_ids": torch.zeros_like(labels),
            "attention_mask": torch.ones_like(labels),
            "labels": labels,
        }

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def enable_full_finetuning(self):
        self.requires_grad_(True)


def _head(action_dim):
    return LayerwiseFlowMatchingActionHead(
        LayerwiseFlowMatchingConfig(
            action_dim=action_dim,
            cross_attention_dim=8,
            hidden_size=8,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=4,
            max_seq_len=32,
            num_inference_timesteps=2,
        )
    )


def _example(route):
    active = route is not WaypointRoute.DONE
    width = 3 if route in {WaypointRoute.NAV_TO_SOURCE, WaypointRoute.NAV_TO_TARGET} else 7
    domain = "NAVIGATION" if width == 3 else "MANIPULATION"
    return {
        "video": ((object(), object()), (object(), object())),
        "lang": "test task",
        "solution": canonical_solution(route),
        "route": route.value,
        "action_domain": domain if active else "NONE",
        "action": [[0.01 * (index + 1)] * width for index in range(ACTION_HORIZON)] if active else None,
        "action_valid_mask": [active and index < 5 for index in range(ACTION_HORIZON)],
    }


def test_special_tokens_are_single_and_unique_and_schedule_is_exact():
    interface = SimpleNamespace(processor=SimpleNamespace(tokenizer=_Tokenizer()))
    ids = waypoint_token_ids(interface)
    assert len({ids.pred_action, ids.pred_done, *ids.route_ids, ids.subtask_start, ids.subtask_end}) == 8
    assert lambda_self_schedule(0.0) == 0.0
    assert lambda_self_schedule(0.05) == 0.0
    assert lambda_self_schedule(0.225) == pytest.approx(0.25)
    assert lambda_self_schedule(0.40) == 0.5
    assert lambda_self_schedule(1.0) == 0.5


def test_router_constrains_action_route_and_fails_closed():
    decision = ConstrainedWaypointRouter(
        _RouterQwen(_RouterBackbone()), route_confidence_min=0.55
    ).decode([{}])[0]
    assert decision.valid
    assert decision.route is WaypointRoute.NAV_TO_SOURCE
    assert decision.subtask_text == "move now"
    assert decision.assistant_prefix == canonical_solution(WaypointRoute.NAV_TO_SOURCE).replace(
        "Walk toward the box holding the Coke can.", "move now"
    )

    low = ConstrainedWaypointRouter(
        _RouterQwen(_RouterBackbone(confident=False)), route_confidence_min=0.55
    ).decode([{}])[0]
    assert not low.valid
    assert low.recover_reason == "route_confidence_below_threshold"

    invalid = ConstrainedWaypointRouter(
        _RouterQwen(_RouterBackbone(invalid_subtask=True)), route_confidence_min=0.55
    ).decode([{}])[0]
    assert not invalid.valid
    assert invalid.recover_reason == "invalid_subtask_tokens"

    done = ConstrainedWaypointRouter(
        _RouterQwen(_RouterBackbone(done=True)), route_confidence_min=0.55
    ).decode([{}])[0]
    assert done.valid and done.done and done.assistant_prefix == "<|pred_done|>"


def test_layerwise_head_is_state_free_masks_suffix_and_samples():
    torch.manual_seed(1)
    head = _head(3)
    assert not hasattr(head, "state_encoder")
    layers = tuple(torch.randn(2, 6, 8, requires_grad=True) for _ in range(2))
    actions = torch.randn(2, ACTION_HORIZON, 3)
    valid = torch.tensor([[True] * 4 + [False] * 16] * 2)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.25, 0.75])
    attention = torch.ones(2, 6, dtype=torch.long)
    loss = head(
        layers,
        actions,
        encoder_attention_mask=attention,
        action_valid_mask=valid,
        noise=noise,
        time=time,
    )
    changed = actions.clone()
    changed[:, 4:] = 1000.0
    changed_loss = head(
        layers,
        changed,
        encoder_attention_mask=attention,
        action_valid_mask=valid,
        noise=noise,
        time=time,
    )
    assert torch.isfinite(loss)
    assert changed_loss.detach().item() == pytest.approx(loss.detach().item())
    loss.backward()
    assert all(layer.grad is not None and torch.isfinite(layer.grad).all() for layer in layers)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.parameters())
    sampled = head.sample(tuple(layer.detach() for layer in layers), encoder_attention_mask=attention)
    assert sampled.shape == (2, ACTION_HORIZON, 3)
    assert torch.isfinite(sampled).all()


def test_oracle_loss_reaches_qwen_and_both_independent_experts():
    torch.manual_seed(2)
    qwen = _OracleQwen(cross_dim=8, layer_count=2)
    nav, arm = _head(3), _head(7)
    policy = ConveyorVLAWaypointPolicy(
        qwen,
        nav,
        arm,
        route_confidence_min=0.55,
    )
    assert {id(value) for value in nav.parameters()}.isdisjoint(
        {id(value) for value in arm.parameters()}
    )
    result = policy.oracle_loss(
        [_example(WaypointRoute.NAV_TO_SOURCE), _example(WaypointRoute.PICK)]
    )
    assert torch.isfinite(result["loss"])
    assert result["navigation_samples"] == 1
    assert result["manipulation_samples"] == 1
    result["loss"].backward()
    for module in (qwen, nav, arm):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )


def test_missing_expert_touches_every_parameter_without_fake_samples():
    qwen = _OracleQwen(cross_dim=8, layer_count=2)
    nav, arm = _head(3), _head(7)
    policy = ConveyorVLAWaypointPolicy(qwen, nav, arm, route_confidence_min=0.55)
    result = policy.oracle_loss([_example(WaypointRoute.NAV_TO_SOURCE)])
    assert result["manipulation_samples"] == 0
    result["loss"].backward()
    assert all(parameter.grad is not None for parameter in arm.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in arm.parameters())
