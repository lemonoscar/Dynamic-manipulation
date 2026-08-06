# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ConveyorVLA AL0 policy assembly for the Go2-X5 benchmark."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from conveyor_bench.m0_dit import (
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
    M0DiTConfig,
)
from conveyor_bench.m0_mobile import M0MobileError


@dataclass(frozen=True)
class PolicyCheckpointReport:
    loaded_qwen_tensors: int
    loaded_action_tensors: int
    reinitialized_keys: tuple[str, ...]


class Qwen3VLInterface(nn.Module):
    """Small local-only wrapper matching the released ABot checkpoint prefix."""

    def __init__(self, model: nn.Module, processor: Any) -> None:
        super().__init__()
        self.model = model
        self.processor = processor

    @classmethod
    def from_local(
        cls,
        model_dir: str | Path,
        *,
        checkpoint_vocab_size: int,
        dtype: torch.dtype = torch.bfloat16,
        attention_implementation: str | None = None,
    ) -> Qwen3VLInterface:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise M0MobileError(
                "Transformers with Qwen3-VL support is required for ConveyorVLA AL0"
            ) from error

        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise M0MobileError(f"Qwen model directory does not exist: {path}")
        implementation = attention_implementation or (
            "flash_attention_2"
            if torch.cuda.is_available() and find_spec("flash_attn") is not None
            else "sdpa"
        )
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        processor.tokenizer.padding_side = "left"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=implementation,
        )
        model.config.hidden_size = model.config.text_config.hidden_size
        current_vocab_size = model.get_input_embeddings().num_embeddings
        if current_vocab_size != checkpoint_vocab_size:
            model.resize_token_embeddings(
                checkpoint_vocab_size,
                mean_resizing=False,
            )
        if model.get_input_embeddings().num_embeddings != checkpoint_vocab_size:
            raise M0MobileError("Qwen embedding resize did not reach checkpoint size")
        return cls(model, processor)

    def build_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
    ) -> Mapping[str, torch.Tensor]:
        if len(images) != len(instructions) or not images:
            raise ValueError("images and instructions must be non-empty equal batches")
        messages = []
        for sample_images, instruction in zip(images, instructions, strict=True):
            if len(sample_images) != 2:
                raise ValueError(
                    "ConveyorVLA AL0 expects head and wrist images in that order"
                )
            content = [
                {"type": "image", "image": _rgb_image(image)}
                for image in sample_images
            ]
            content.append({"type": "text", "text": _instruction(instruction)})
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def build_temporal_inputs(
        self,
        videos: Sequence[Sequence[Sequence[Any]]],
        instructions: Sequence[str],
    ) -> Mapping[str, torch.Tensor]:
        """Build two ordered-frame clips: head first, then wrist."""

        if len(videos) != len(instructions) or not videos:
            raise ValueError("videos and instructions must be non-empty equal batches")
        messages = []
        for sample_clips, instruction in zip(videos, instructions, strict=True):
            if len(sample_clips) != 2 or any(len(clip) != 2 for clip in sample_clips):
                raise ValueError(
                    "AL0 temporal input expects two frames for head and wrist"
                )
            content = [
                {"type": "text", "text": "Head camera, oldest to newest:"},
                {
                    "type": "video",
                    "video": [_rgb_image(frame) for frame in sample_clips[0]],
                },
                {"type": "text", "text": "Wrist camera, oldest to newest:"},
                {
                    "type": "video",
                    "video": [_rgb_image(frame) for frame in sample_clips[1]],
                },
                {"type": "text", "text": _instruction(instruction)},
            ]
            messages.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def forward(self, **inputs: torch.Tensor) -> Any:
        return self.model(**inputs)


class ConveyorVLAAL0Policy(nn.Module):
    """Qwen visual-language encoder followed by the checkpoint-compatible DiT."""

    def __init__(
        self,
        qwen_vl_interface: Qwen3VLInterface,
        action_model: M0DiTActionHead,
        *,
        repeated_diffusion_steps: int = 4,
    ) -> None:
        super().__init__()
        if repeated_diffusion_steps <= 0:
            raise ValueError("repeated_diffusion_steps must be positive")
        self.qwen_vl_interface = qwen_vl_interface
        self.action_model = action_model
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self._qwen_frozen = False

    def freeze_qwen(self) -> None:
        self._qwen_frozen = True
        self.qwen_vl_interface.requires_grad_(False)
        self.qwen_vl_interface.eval()

    def train(self, mode: bool = True) -> ConveyorVLAAL0Policy:
        super().train(mode)
        if self._qwen_frozen:
            self.qwen_vl_interface.eval()
        return self

    def forward(self, examples: Sequence[Mapping[str, Any]]) -> Mapping[str, torch.Tensor]:
        hidden, attention_mask = self._encode(examples)
        device = next(self.action_model.parameters()).device
        dtype = next(self.action_model.parameters()).dtype
        actions = torch.as_tensor(
            [example["action"] for example in examples],
            device=device,
            dtype=dtype,
        )
        state = torch.as_tensor(
            [example["state"] for example in examples],
            device=device,
            dtype=dtype,
        )
        action_mask = torch.as_tensor(
            [example["action_mask"] for example in examples],
            device=device,
            dtype=torch.bool,
        )
        repeats = self.repeated_diffusion_steps
        with _action_autocast(device, dtype):
            loss = self.action_model(
                hidden.repeat(repeats, 1, 1),
                actions.repeat(repeats, 1, 1),
                state.repeat(repeats, 1, 1),
                encoder_attention_mask=attention_mask.repeat(repeats, 1),
                action_dimension_mask=action_mask.repeat(repeats, 1),
            )
        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_normalized_actions(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        hidden, attention_mask = self._encode(examples)
        device = next(self.action_model.parameters()).device
        dtype = next(self.action_model.parameters()).dtype
        state = torch.as_tensor(
            [example["state"] for example in examples],
            device=device,
            dtype=dtype,
        )
        action_mask = torch.as_tensor(
            [example["action_mask"] for example in examples],
            device=device,
            dtype=torch.bool,
        )
        with _action_autocast(device, dtype):
            return self.action_model.sample(
                hidden,
                state,
                encoder_attention_mask=attention_mask,
                action_dimension_mask=action_mask,
            )

    def _encode(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("examples must be non-empty")
        inputs = self.qwen_vl_interface.build_inputs(
            [example["image"] for example in examples],
            [_instruction(example["lang"]) for example in examples],
        )
        return self._encode_inputs(inputs)

    def _encode_inputs(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        device = next(self.action_model.parameters()).device
        dtype = next(self.action_model.parameters()).dtype
        return hidden.to(device=device, dtype=dtype), attention_mask.to(device)


class ConveyorVLAAL0TemporalPolicy(ConveyorVLAAL0Policy):
    """AL0 DiT with Qwen ordered-frame clips and a 20-step action head."""

    def _encode(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("examples must be non-empty")
        inputs = self.qwen_vl_interface.build_temporal_inputs(
            [example["video"] for example in examples],
            [_instruction(example["lang"]) for example in examples],
        )
        return self._encode_inputs(inputs)


def m0_dit_config(config: Mapping[str, Any]) -> M0DiTConfig:
    action = _mapping(config.get("action_model"), "action_model")
    return M0DiTConfig(
        action_dim=_integer(action, "action_dim"),
        state_dim=_integer(action, "state_dim"),
        action_horizon=_integer(action, "action_horizon"),
        vlm_hidden_dim=_integer(action, "cross_attention_dim"),
        input_embedding_dim=_integer(action, "input_embedding_dim"),
        hidden_size=_integer(action, "hidden_size"),
        num_attention_heads=_integer(action, "num_attention_heads"),
        attention_head_dim=_integer(action, "attention_head_dim"),
        num_layers=_integer(action, "num_layers"),
        dropout=float(action.get("dropout", 0.2)),
        max_seq_len=_integer(action, "max_seq_len"),
        num_target_vision_tokens=_integer(action, "num_target_vision_tokens"),
        noise_beta_alpha=float(action.get("noise_beta_alpha", 1.5)),
        noise_beta_beta=float(action.get("noise_beta_beta", 1.0)),
        noise_s=float(action.get("noise_s", 0.999)),
        time_epsilon=float(action.get("time_epsilon", 0.05)),
        num_timestep_buckets=_integer(action, "num_timestep_buckets"),
        num_inference_timesteps=_integer(action, "num_inference_timesteps"),
        interleave_self_attention=bool(
            action.get("interleave_self_attention", True)
        ),
    )


def transfer_robocasa_policy_weights(
    policy: ConveyorVLAAL0Policy,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
) -> PolicyCheckpointReport:
    """Transfer the released Qwen/DiT trunk and reject every unknown difference."""

    source = (
        torch.load(
            Path(checkpoint),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    if not isinstance(source, Mapping):
        raise RuntimeError("upstream ABot-M0 checkpoint must contain a tensor mapping")
    target = policy.state_dict()
    reinitialized = {
        f"action_model.{key}" for key in GO2_X5_REINITIALIZED_ACTION_KEYS
    }
    unexpected = set(source) - set(target)
    missing = set(target) - set(source)
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint structure mismatch: "
            f"unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )
    compatible: dict[str, torch.Tensor] = {}
    bad_shapes: list[str] = []
    for key, target_value in target.items():
        source_value = source[key]
        if not isinstance(source_value, torch.Tensor):
            raise RuntimeError(f"checkpoint value is not a tensor: {key}")
        if key in reinitialized:
            continue
        if source_value.shape != target_value.shape:
            bad_shapes.append(
                f"{key}: source={tuple(source_value.shape)} "
                f"target={tuple(target_value.shape)}"
            )
        else:
            compatible[key] = source_value
    if bad_shapes:
        raise RuntimeError(
            "unapproved checkpoint shape mismatch: " + "; ".join(bad_shapes)
        )
    result = policy.load_state_dict(compatible, strict=False)
    if set(result.missing_keys) != reinitialized or result.unexpected_keys:
        raise RuntimeError("checkpoint load result violates the migration contract")
    return PolicyCheckpointReport(
        loaded_qwen_tensors=sum(
            key.startswith("qwen_vl_interface.") for key in compatible
        ),
        loaded_action_tensors=sum(key.startswith("action_model.") for key in compatible),
        reinitialized_keys=tuple(sorted(reinitialized)),
    )


def _rgb_image(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return value.convert("RGB") if hasattr(value, "convert") else value
    try:
        from PIL import Image
    except ImportError as error:
        raise M0MobileError("Pillow is required to load exported camera images") from error
    path = Path(value)
    if not path.is_file():
        raise M0MobileError(f"camera image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _instruction(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("language instruction must be a non-empty string")
    return value.strip()


def _action_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    mixed = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)) or (
        device.type == "cpu" and dtype == torch.bfloat16
    )
    return torch.autocast(device_type=device.type, dtype=dtype) if mixed else nullcontext()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M0MobileError(f"action_model.{key} must be a positive integer")
    return value


# Frozen alias for older imports and checkpoints. New code should use the
# ConveyorVLA name; the implementation remains tensor-compatible.
M0MobilePolicy = ConveyorVLAAL0Policy


__all__ = [
    "ConveyorVLAAL0Policy",
    "ConveyorVLAAL0TemporalPolicy",
    "M0MobilePolicy",
    "PolicyCheckpointReport",
    "Qwen3VLInterface",
    "m0_dit_config",
    "transfer_robocasa_policy_weights",
]
