from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from conveyor_bench.conveyorvla.dit import (
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
    M0DiTConfig,
)
from conveyor_bench.conveyorvla.config import load_m0_mobile_config
from conveyor_bench.conveyorvla.policy import (
    ConveyorVLAAL0TemporalPolicy,
    M0MobilePolicy,
    Qwen3VLInterface,
    m0_dit_config,
    transfer_robocasa_policy_weights,
)


class _Processor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        self.messages = messages
        batch_size = len(messages)
        attention_mask = torch.ones(batch_size, 5, dtype=torch.long)
        for index in range(batch_size):
            attention_mask[index, : index + 1] = 0
        return {
            "input_ids": torch.ones(batch_size, 5, dtype=torch.long),
            "attention_mask": attention_mask,
        }


class _Qwen(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.randn(hidden_size))
        self.last_attention_mask = None
        self.last_kwargs = {}

    def forward(self, input_ids, attention_mask, **kwargs):
        self.last_attention_mask = attention_mask.detach().clone()
        self.last_kwargs = kwargs
        hidden = self.anchor.view(1, 1, -1).expand(input_ids.shape[0], 5, -1)
        return SimpleNamespace(hidden_states=(hidden,))


def _tiny_config() -> M0DiTConfig:
    return M0DiTConfig(
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
        num_timestep_buckets=100,
        num_inference_timesteps=2,
    )


def _examples(batch_size: int = 2):
    return [
        {
            "image": [object(), object()],
            "lang": "pick the moving part",
            "state": [[0.0, 0.1, 0.2, 0.3]],
            "action": [[0.1, 9.0, -0.1]] * 4,
            "action_mask": [True, False, True],
        }
        for _ in range(batch_size)
    ]


def _temporal_examples(batch_size: int = 2):
    examples = _examples(batch_size)
    for example in examples:
        example["video"] = ((object(), object()), (object(), object()))
        del example["image"]
    return examples


def test_policy_batch_forward_and_masked_sampling() -> None:
    torch.manual_seed(5)
    config = _tiny_config()
    interface = Qwen3VLInterface(_Qwen(config.vlm_hidden_dim), _Processor())
    policy = M0MobilePolicy(
        interface,
        M0DiTActionHead(config),
        repeated_diffusion_steps=2,
    )

    seen_action_masks = []
    hook = policy.action_model.model.transformer_blocks[0].attn1.register_forward_pre_hook(
        lambda _module, inputs: seen_action_masks.append(inputs[2].detach().clone())
    )
    loss = policy(_examples())["action_loss"]
    hook.remove()
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.action_model.action_decoder.layer2.weight.grad is not None
    expected_mask = torch.tensor(
        [[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.long
    )
    assert torch.equal(interface.model.last_attention_mask.cpu(), expected_mask)
    assert torch.equal(seen_action_masks[0].cpu(), expected_mask.repeat(2, 1))
    assert interface.model.last_kwargs["logits_to_keep"] == 1

    actions = policy.predict_normalized_actions(_examples(1))
    assert actions.shape == (1, 4, 3)
    assert torch.count_nonzero(actions[..., 1]) == 0


def test_temporal_policy_builds_two_ordered_camera_clips() -> None:
    torch.manual_seed(6)
    config = _tiny_config()
    processor = _Processor()
    interface = Qwen3VLInterface(_Qwen(config.vlm_hidden_dim), processor)
    policy = ConveyorVLAAL0TemporalPolicy(
        interface,
        M0DiTActionHead(config),
    )

    loss = policy(_temporal_examples(1))["action_loss"]
    content = processor.messages[0][0]["content"]

    assert torch.isfinite(loss)
    assert [item["type"] for item in content] == [
        "text",
        "video",
        "text",
        "video",
        "text",
    ]
    assert len(content[1]["video"]) == len(content[3]["video"]) == 2


def test_release_config_builds_go2_x5_dit_contract() -> None:
    config = m0_dit_config(load_m0_mobile_config())
    assert (config.state_dim, config.action_dim, config.action_horizon) == (28, 10, 16)
    assert config.vlm_hidden_dim == 2560
    assert config.input_embedding_dim == 768
    assert config.num_layers == 16
    assert config.interleave_self_attention is True


def test_frozen_qwen_stays_in_eval_mode_when_policy_trains() -> None:
    config = _tiny_config()
    interface = Qwen3VLInterface(_Qwen(config.vlm_hidden_dim), _Processor())
    policy = M0MobilePolicy(interface, M0DiTActionHead(config))

    policy.freeze_qwen()
    policy.train()

    assert policy.training
    assert policy.action_model.training
    assert not policy.qwen_vl_interface.training
    assert not any(parameter.requires_grad for parameter in interface.parameters())


def test_bfloat16_action_inference_has_a_local_autocast_boundary() -> None:
    config = _tiny_config()
    interface = Qwen3VLInterface(_Qwen(config.vlm_hidden_dim), _Processor())
    policy = M0MobilePolicy(
        interface,
        M0DiTActionHead(config).to(torch.bfloat16),
    )

    loss = policy(_examples(1))["action_loss"]
    loss.backward()
    policy.eval()

    actions = policy.predict_normalized_actions(_examples(1))

    assert torch.isfinite(loss)
    assert actions.dtype == torch.bfloat16
    assert torch.isfinite(actions).all()


def _robocasa_checkpoint(policy: M0MobilePolicy) -> dict[str, torch.Tensor]:
    source = {
        key: torch.full_like(value, 0.125)
        for key, value in policy.state_dict().items()
    }
    config = policy.action_model.config
    source.update(
        {
            "action_model.state_encoder.layer1.weight": torch.randn(
                config.hidden_size, 58
            ),
            "action_model.state_encoder.layer1.bias": torch.randn(config.hidden_size),
            "action_model.action_encoder.layer1.weight": torch.randn(
                config.input_embedding_dim, 29
            ),
            "action_model.action_encoder.layer1.bias": torch.randn(
                config.input_embedding_dim
            ),
            "action_model.action_decoder.layer2.weight": torch.randn(
                29, config.hidden_size
            ),
            "action_model.action_decoder.layer2.bias": torch.randn(29),
        }
    )
    return source


def test_policy_transfer_requires_exact_keys_and_resets_only_six_boundaries() -> None:
    config = _tiny_config()
    interface = Qwen3VLInterface(_Qwen(config.vlm_hidden_dim), _Processor())
    policy = M0MobilePolicy(interface, M0DiTActionHead(config))
    checkpoint = _robocasa_checkpoint(policy)
    boundary_keys = {
        f"action_model.{key}" for key in GO2_X5_REINITIALIZED_ACTION_KEYS
    }
    initial_boundary = {
        key: policy.state_dict()[key].clone() for key in boundary_keys
    }

    report = transfer_robocasa_policy_weights(policy, checkpoint)

    assert set(report.reinitialized_keys) == boundary_keys
    assert report.loaded_qwen_tensors == 1
    assert report.loaded_action_tensors == len(policy.action_model.state_dict()) - 6
    assert all(
        torch.equal(policy.state_dict()[key], initial_boundary[key])
        for key in boundary_keys
    )

    missing_boundary = dict(checkpoint)
    del missing_boundary["action_model.state_encoder.layer1.bias"]
    with pytest.raises(RuntimeError, match="checkpoint structure mismatch"):
        transfer_robocasa_policy_weights(policy, missing_boundary)
