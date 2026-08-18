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
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
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


class _GeneratingQwen(_Qwen):
    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.generate_training_modes = []

    def generate(self, input_ids, **_kwargs):
        self.generate_training_modes.append(self.training)
        suffix = torch.full(
            (input_ids.shape[0], 1),
            99,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return torch.cat((input_ids, suffix), dim=1)


class _GenerationTokenizer:
    def __init__(self):
        self.decoded_values = []

    def convert_tokens_to_ids(self, _token):
        return 99

    def batch_decode(self, values, **_kwargs):
        self.decoded_values = [list(map(int, row)) for row in values]
        return [f"decoded-{int(row[0])}" for row in values]


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
        temporal_history_span_s=0.2,
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
    assert processor.kwargs["video_metadata"] == [
        [
            {
                "total_num_frames": 2,
                "fps": 5.0,
                "duration": 0.4,
                "frames_indices": [0, 1],
            },
            {
                "total_num_frames": 2,
                "fps": 5.0,
                "duration": 0.4,
                "frames_indices": [0, 1],
            },
        ]
    ]


def test_temporal_generation_temporarily_disables_training_mode(monkeypatch) -> None:
    model = _GeneratingQwen(24)
    interface = Qwen3VLInterface(
        model,
        SimpleNamespace(tokenizer=_GenerationTokenizer()),
    )
    monkeypatch.setattr(
        interface,
        "build_temporal_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
        },
    )
    model.train()

    generated = interface.generate_temporal_subtask_texts(
        [((object(), object()), (object(), object()))],
        ["route the current subtask"],
        history_span_s=0.2,
    )

    assert generated == ("decoded-99",)
    assert model.generate_training_modes == [False]
    assert model.training is True


def test_temporal_generation_trims_each_batch_row_at_its_first_end_token(
    monkeypatch,
) -> None:
    class _PaddedGeneratingQwen(_GeneratingQwen):
        def generate(self, input_ids, **_kwargs):
            suffix = torch.tensor(
                [[41, 99, 0, 0], [42, 43, 44, 99]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            return torch.cat((input_ids, suffix), dim=1)

    model = _PaddedGeneratingQwen(24)
    tokenizer = _GenerationTokenizer()
    interface = Qwen3VLInterface(model, SimpleNamespace(tokenizer=tokenizer))
    monkeypatch.setattr(
        interface,
        "build_temporal_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.ones((2, 5), dtype=torch.long),
            "attention_mask": torch.ones((2, 5), dtype=torch.long),
        },
    )

    generated = interface.generate_temporal_subtask_texts(
        [
            ((object(), object()), (object(), object())),
            ((object(), object()), (object(), object())),
        ],
        ["route the current subtask", "route the current subtask"],
        history_span_s=0.2,
    )

    assert generated == ("decoded-41", "decoded-42")
    assert tokenizer.decoded_values == [[41, 99], [42, 43, 44, 99]]


def test_second_pass_accepts_control_only_prediction_without_supervision() -> None:
    class _Tokenizer:
        def __call__(self, _text, **_kwargs):
            return SimpleNamespace(input_ids=[10, 11])

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 12

    class _ControlOnlyProcessor:
        tokenizer = _Tokenizer()

        def apply_chat_template(self, _messages, **_kwargs):
            tokens = torch.tensor([[10, 11, 12]], dtype=torch.long)
            return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    interface = Qwen3VLInterface(_Qwen(24), _ControlOnlyProcessor())
    arguments = {
        "videos": [((object(), object()), (object(), object()))],
        "instructions": ["route the current subtask"],
        "history_span_s": 0.2,
        "solutions": ["<|im_end|>"],
    }

    with pytest.raises(RuntimeError, match="assistant answer span is empty"):
        interface.build_temporal_inputs(**arguments)
    inputs = interface.build_temporal_inputs(
        **arguments,
        supervise_solutions=False,
    )

    assert "labels" not in inputs


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
