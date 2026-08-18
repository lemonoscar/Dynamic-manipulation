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
from conveyor_bench.conveyorvla.hierarchical_data import (
    _train_navigation_action_statistics,
)
from conveyor_bench.conveyorvla.policy import (
    ConveyorVLAAL0TwoPassPolicy,
)
from conveyor_bench.conveyorvla.subtasks import (
    ActionDomain,
    NAVIGATION_ARM_JOINT_REFERENCES,
    Phase,
    action_domain,
    compose_manipulation_action10,
    compose_navigation_action,
    parse_subtask_solution,
    phase_from_pct,
    phase_instruction,
    project_action10,
    subtask_prompt,
    subtask_solution,
)


def test_pct_phases_map_to_two_disjoint_action_domains() -> None:
    assert phase_from_pct("plan_nav_to_pick") is Phase.NAV_TO_SOURCE
    assert phase_from_pct("exec_nav_to_pick") is Phase.NAV_TO_SOURCE
    assert phase_from_pct("verify_pick_reachable") is Phase.NAV_TO_SOURCE
    assert phase_from_pct("plan_pick") is Phase.PICK
    assert phase_from_pct("verify_pick_success") is Phase.PICK
    assert phase_from_pct("exec_nav_to_place") is Phase.NAV_TO_TARGET
    assert phase_from_pct("plan_nav_to_place") is Phase.NAV_TO_TARGET
    assert phase_from_pct("verify_place_reachable") is Phase.NAV_TO_TARGET
    assert phase_from_pct("plan_place") is Phase.PLACE
    assert action_domain(Phase.NAV_TO_SOURCE) is ActionDomain.NAVIGATION
    assert action_domain(Phase.PICK) is ActionDomain.MANIPULATION
    with pytest.raises(ValueError, match="unsupported PCT"):
        phase_from_pct("plan_unknown")


def test_navigation_scale_is_derived_only_from_valid_train_actions() -> None:
    first = [0.0] * 200
    second = [0.0] * 200
    for step in range(20):
        first[step * 10] = 0.5
        first[step * 10 + 2] = -0.7
        second[step * 10] = 100.0
        second[step * 10 + 2] = 100.0

    class _Dataset:
        class _HF:
            def select(self, indices):
                values = [first, second]
                return {"action": [values[index] for index in indices]}

        hf_dataset = _HF()

    annotations = [
        {
            "split": "train",
            "action_domain_id": int(ActionDomain.NAVIGATION),
            "base_index": 0,
            "action_valid_mask": [True] * 20,
        },
        {
            "split": "val",
            "action_domain_id": int(ActionDomain.NAVIGATION),
            "base_index": 1,
            "action_valid_mask": [True] * 20,
        },
    ]

    report = _train_navigation_action_statistics(_Dataset(), annotations)

    assert report["row_count"] == 1
    assert report["valid_future_action_count"] == 20
    assert report["recommended_physical_scale"] == pytest.approx([0.525, 0.735])


def test_canonical_subtask_answer_is_visible_parseable_and_determines_dit() -> None:
    solution = subtask_solution(Phase.NAV_TO_TARGET)
    decision = parse_subtask_solution(solution)

    assert solution == (
        "<|pred_action|><|subtask|>Turn around and walk to the empty box."
        "<|end_subtask|>"
    )
    assert decision.phase is Phase.NAV_TO_TARGET
    assert decision.domain is ActionDomain.NAVIGATION
    empty_prompt = subtask_prompt("Move the Coke can.")
    assert "Completed subtasks" not in empty_prompt
    assert "Previous model prediction" not in empty_prompt
    with pytest.raises(TypeError):
        subtask_prompt("Move the Coke can.", phase_instruction(Phase.PICK))
    with pytest.raises(ValueError, match="unsupported canonical"):
        parse_subtask_solution(
            "<|pred_action|><|subtask|>Maybe move somewhere.<|end_subtask|>"
        )


def test_domain_projection_and_runtime_composer_cannot_move_inactive_actuators() -> None:
    action10 = tuple(float(index) for index in range(10))

    assert project_action10(action10, ActionDomain.NAVIGATION) == (0.0, 2.0)
    assert project_action10(action10, ActionDomain.MANIPULATION) == tuple(
        float(index) for index in range(3, 10)
    )
    source = compose_navigation_action(
        Phase.NAV_TO_SOURCE,
        (0.25, -0.4),
    )
    target = compose_navigation_action(
        Phase.NAV_TO_TARGET,
        (0.1, 0.2),
        measured_arm_joint_positions=(0.5,) * 6,
    )
    assert source.base_velocity == (0.25, 0.0, -0.4)
    assert source.reference_mode == "stow_open"
    assert source.arm_joint_positions == NAVIGATION_ARM_JOINT_REFERENCES[
        Phase.NAV_TO_SOURCE
    ]
    assert source.gripper_open_fraction == 1.0
    assert source.tcp_delta_used is False
    assert target.reference_mode == "carry_closed"
    assert target.gripper_open_fraction == 0.0
    assert all(value < 0.5 for value in target.arm_joint_positions)
    assert compose_manipulation_action10(
        (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)
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
        self.generated = ()
        self.build_calls = []
        self.build_batch_sizes = []
        self.supervision_calls = []

    def build_temporal_inputs(
        self,
        videos: object,
        instructions: object,
        *,
        history_span_s: float,
        solutions: object = None,
        supervise_solutions: bool = True,
    ) -> dict[str, torch.Tensor]:
        self.build_calls.append(solutions is not None)
        self.supervision_calls.append(supervise_solutions)
        batch = len(videos)  # type: ignore[arg-type]
        self.build_batch_sizes.append(batch)
        result = {
            "input_ids": torch.ones(batch, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch, 5, dtype=torch.long),
        }
        if solutions is not None and supervise_solutions:
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

    def generate_temporal_subtask_texts(self, videos, instructions, **_kwargs):
        if self.generated:
            return self.generated
        return tuple(subtask_solution(Phase.NAV_TO_SOURCE) for _ in videos)

    def generate_temporal_subtasks(self, videos, instructions, **kwargs):
        return tuple(
            parse_subtask_solution(value)
            for value in self.generate_temporal_subtask_texts(
                videos, instructions, **kwargs
            )
        )


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
                "phase_id": int(
                    Phase.NAV_TO_SOURCE
                    if domain is ActionDomain.NAVIGATION
                    else Phase.PICK
                ),
                "state": ((0.0,) * 4,),
                "action": ((0.0,) * action_dim,) * 4,
                "action_mask": (True,) * action_dim,
                "action_valid_mask": (True, True, False, False),
                "sample_id": f"sample-{int(domain)}",
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

    policy.zero_grad(set_to_none=True)
    navigation_only = policy(examples[:1], objective="action")["action_loss"]
    assert isinstance(navigation_only, torch.Tensor)
    navigation_only.backward()
    assert not any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in manipulation.parameters()
    )


def test_zero_teacher_forcing_uses_generated_route_and_skips_wrong_expert() -> None:
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
    qwen.generated = (
        subtask_solution(Phase.NAV_TO_SOURCE),
        subtask_solution(Phase.NAV_TO_SOURCE),
    )
    policy = ConveyorVLAAL0TwoPassPolicy(
        qwen,  # type: ignore[arg-type]
        M0DiTActionHead(M0DiTConfig(action_dim=2, **common)),
        M0DiTActionHead(M0DiTConfig(action_dim=7, **common)),
        temporal_history_span_s=0.2,
    )
    examples = [
        {
            "video": ((object(), object()), (object(), object())),
            "lang": "Do the task.",
            "solution": subtask_solution(phase),
            "phase_id": int(phase),
            "action_domain_id": int(action_domain(phase)),
            "state": ((0.0,) * 4,),
            "action": ((0.0,) * action_dim,) * 4,
            "action_mask": (True,) * action_dim,
            "action_valid_mask": (True, True, True, False),
            "sample_id": phase.name,
        }
        for phase, action_dim in ((Phase.NAV_TO_SOURCE, 2), (Phase.PICK, 7))
    ]

    result = policy(
        examples,
        objective="action",
        teacher_forcing_probability=0.0,
        routing_seed=7,
    )

    assert result["teacher_forced_samples"] == 0
    assert result["predicted_route_correct"] == 1
    assert result["predicted_route_wrong"] == 1
    assert result["navigation_samples"] == 1
    assert result["manipulation_samples"] == 0
    assert qwen.build_batch_sizes == [2]
    assert qwen.supervision_calls == [False]


def test_online_two_pass_generation_runs_second_full_forward_and_dispatches() -> None:
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
    qwen.generated = (
        subtask_solution(Phase.NAV_TO_SOURCE),
        subtask_solution(Phase.PICK),
    )
    policy = ConveyorVLAAL0TwoPassPolicy(
        qwen,  # type: ignore[arg-type]
        M0DiTActionHead(M0DiTConfig(action_dim=2, **common)),
        M0DiTActionHead(M0DiTConfig(action_dim=7, **common)),
        temporal_history_span_s=0.2,
    ).eval()
    examples = [
        {
            "video": ((object(), object()), (object(), object())),
            "lang": "Do the task.",
            "solution": subtask_solution(phase),
            "state": ((0.0,) * 4,),
        }
        for phase in (Phase.NAV_TO_SOURCE, Phase.PICK)
    ]

    predictions = policy.predict_routed_actions(examples)

    assert [item.decision.phase for item in predictions] == [
        Phase.NAV_TO_SOURCE,
        Phase.PICK,
    ]
    assert len(predictions[0].normalized_actions[0]) == 2
    assert len(predictions[1].normalized_actions[0]) == 7
    assert qwen.build_calls == [True]
