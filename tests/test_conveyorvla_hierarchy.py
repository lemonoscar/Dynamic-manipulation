from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conveyor_bench.conveyorvla.dit import (
    DOMAIN_ACTION_REINITIALIZED_KEYS,
    M0DiTActionHead,
    M0DiTConfig,
    transfer_conveyorvla_action_trunk,
)
from conveyor_bench.conveyorvla.policy import (
    ConveyorVLAAL0TwoPassPolicy,
)
from conveyor_bench.conveyorvla.subtasks import (
    ActionDomain,
    Phase,
    action_domain,
    compose_gated_action10,
    parse_subtask_solution,
    phase_from_pct,
    phase_instruction,
    project_action10,
    subtask_history,
    subtask_prompt,
    subtask_solution,
)


def test_pct_phases_map_to_two_disjoint_action_domains() -> None:
    assert phase_from_pct("exec_nav_to_pick") is Phase.NAV_TO_SOURCE
    assert phase_from_pct("exec_nav_to_place") is Phase.NAV_TO_TARGET
    assert action_domain(Phase.NAV_TO_SOURCE) is ActionDomain.NAVIGATION
    assert action_domain(Phase.PICK) is ActionDomain.MANIPULATION
    with pytest.raises(ValueError, match="unsupported PCT"):
        phase_from_pct("plan_pick")


def test_canonical_subtask_answer_is_visible_parseable_and_determines_dit() -> None:
    solution = subtask_solution(Phase.NAV_TO_TARGET)
    decision = parse_subtask_solution(solution)

    assert solution == (
        "<|pred_action|><|subtask|>Turn around and walk to the empty box."
        "<|end_subtask|>"
    )
    assert decision.phase is Phase.NAV_TO_TARGET
    assert decision.domain is ActionDomain.NAVIGATION
    assert subtask_history(Phase.PLACE) == tuple(
        phase_instruction(phase)
        for phase in (Phase.NAV_TO_SOURCE, Phase.PICK, Phase.NAV_TO_TARGET)
    )
    assert "Completed subtasks:" in subtask_prompt(
        "Move the Coke can.", subtask_history(Phase.PICK)
    )
    with pytest.raises(ValueError, match="unsupported canonical"):
        parse_subtask_solution(
            "<|pred_action|><|subtask|>Maybe move somewhere.<|end_subtask|>"
        )


def test_domain_projection_and_runtime_gate_cannot_move_inactive_actuators() -> None:
    action10 = tuple(float(index) for index in range(10))

    assert project_action10(action10, ActionDomain.NAVIGATION) == (0.0, 2.0)
    assert project_action10(action10, ActionDomain.MANIPULATION) == tuple(
        float(index) for index in range(3, 10)
    )
    assert compose_gated_action10(
        (0.25, -0.4),
        ActionDomain.NAVIGATION,
        gripper_latch=0.7,
    ) == (0.25, 0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7)
    assert compose_gated_action10(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0),
        ActionDomain.MANIPULATION,
        gripper_latch=0.2,
    ) == (0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)


def test_10d_checkpoint_transfers_all_but_domain_action_io_shapes() -> None:
    common = {
        "state_dim": 5,
        "action_horizon": 4,
        "vlm_hidden_dim": 8,
        "input_embedding_dim": 8,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "num_layers": 2,
        "max_seq_len": 16,
        "num_target_vision_tokens": 2,
    }
    source = M0DiTActionHead(M0DiTConfig(action_dim=10, **common))
    target = M0DiTActionHead(M0DiTConfig(action_dim=2, **common))

    report = transfer_conveyorvla_action_trunk(target, source.state_dict())

    assert set(report.reinitialized_keys) == DOMAIN_ACTION_REINITIALIZED_KEYS
    assert set(report.loaded_keys) == set(target.state_dict()) - set(
        DOMAIN_ACTION_REINITIALIZED_KEYS
    )
    assert torch.equal(
        target.state_dict()["model.proj_out_2.weight"],
        source.state_dict()["model.proj_out_2.weight"],
    )


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Parameter(torch.randn(8))

    def build_temporal_inputs(
        self,
        videos: object,
        instructions: object,
        *,
        history_span_s: float,
        solutions: object = None,
    ) -> dict[str, torch.Tensor]:
        batch = len(videos)  # type: ignore[arg-type]
        result = {
            "input_ids": torch.ones(batch, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch, 5, dtype=torch.long),
        }
        if solutions is not None:
            result["labels"] = torch.ones(batch, 5, dtype=torch.long)
        return result

    def forward(self, **inputs: object) -> SimpleNamespace:
        batch, tokens = inputs["input_ids"].shape  # type: ignore[union-attr]
        hidden = self.feature.view(1, 1, -1).expand(batch, tokens, -1)
        labels = inputs.get("labels")
        return SimpleNamespace(
            loss=self.feature.square().mean() if labels is not None else None,
            hidden_states=(hidden,),
        )

    def enable_full_finetuning(self) -> None:
        self.requires_grad_(True)


def test_two_pass_joint_losses_reach_qwen_and_both_dits() -> None:
    common = {
        "state_dim": 4,
        "action_horizon": 4,
        "vlm_hidden_dim": 8,
        "input_embedding_dim": 8,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "attention_head_dim": 4,
        "num_layers": 2,
        "max_seq_len": 16,
        "num_target_vision_tokens": 2,
    }
    qwen = _FakeQwen()
    navigation = M0DiTActionHead(M0DiTConfig(action_dim=2, **common))
    manipulation = M0DiTActionHead(M0DiTConfig(action_dim=7, **common))
    policy = ConveyorVLAAL0TwoPassPolicy(
        qwen,  # type: ignore[arg-type]
        navigation,
        manipulation,
        temporal_history_span_s=0.2,
    )
    examples = []
    for domain, action_dim in (
        (ActionDomain.NAVIGATION, 2),
        (ActionDomain.MANIPULATION, 7),
    ):
        examples.append(
            {
                "video": ((object(), object()), (object(), object())),
                "lang": "Do the task.",
                "solution": subtask_solution(
                    Phase.NAV_TO_SOURCE
                    if domain is ActionDomain.NAVIGATION
                    else Phase.PICK
                ),
                "action_domain_id": int(domain),
                "state": ((0.0,) * 4,),
                "action": ((0.0,) * action_dim,) * 4,
                "action_mask": (True,) * action_dim,
            }
        )

    subtask = policy(examples, objective="subtask")["subtask_loss"]
    assert isinstance(subtask, torch.Tensor)
    subtask.backward()
    assert qwen.feature.grad is not None
    policy.zero_grad(set_to_none=True)

    action = policy(examples, objective="action")["action_loss"]
    assert isinstance(action, torch.Tensor)
    action.backward()
    assert qwen.feature.grad is not None
    assert any(parameter.grad is not None for parameter in navigation.parameters())
    assert any(parameter.grad is not None for parameter in manipulation.parameters())
