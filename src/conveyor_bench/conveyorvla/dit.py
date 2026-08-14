# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible ABot-M0 DiT action head adapted for Go2-X5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta


GO2_X5_REINITIALIZED_ACTION_KEYS = frozenset(
    {
        "state_encoder.layer1.weight",
        "state_encoder.layer1.bias",
        "action_encoder.layer1.weight",
        "action_encoder.layer1.bias",
        "action_decoder.layer2.weight",
        "action_decoder.layer2.bias",
    }
)
DOMAIN_ACTION_REINITIALIZED_KEYS = frozenset(
    {
        "action_encoder.layer1.weight",
        "action_decoder.layer2.weight",
        "action_decoder.layer2.bias",
    }
)


@dataclass(frozen=True)
class M0DiTConfig:
    action_dim: int = 10
    state_dim: int = 28
    action_horizon: int = 16
    vlm_hidden_dim: int = 2560
    input_embedding_dim: int = 768
    hidden_size: int = 2560
    num_attention_heads: int = 12
    attention_head_dim: int = 64
    num_layers: int = 16
    dropout: float = 0.2
    max_seq_len: int = 1024
    num_target_vision_tokens: int = 32
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    time_epsilon: float = 0.05
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4
    interleave_self_attention: bool = True

    def __post_init__(self) -> None:
        positive = (
            "action_dim",
            "state_dim",
            "action_horizon",
            "vlm_hidden_dim",
            "input_embedding_dim",
            "hidden_size",
            "num_attention_heads",
            "attention_head_dim",
            "num_layers",
            "max_seq_len",
            "num_target_vision_tokens",
            "num_timestep_buckets",
            "num_inference_timesteps",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("all AL0 DiT dimensions and counts must be positive")
        if self.num_attention_heads * self.attention_head_dim != self.input_embedding_dim:
            raise ValueError("attention dimensions must equal input_embedding_dim")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if not 0.0 < self.noise_s <= 1.0:
            raise ValueError("noise_s must be within (0, 1]")
        if not 0.0 < self.time_epsilon < 1.0:
            raise ValueError("time_epsilon must be within (0, 1)")


@dataclass(frozen=True)
class ActionTransferReport:
    loaded_keys: tuple[str, ...]
    reinitialized_keys: tuple[str, ...]


class _SinusoidalActionTime(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim % 2:
            raise ValueError("action time embedding dimension must be even")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        ) * (math.log(10000.0) / half_dim)
        frequencies = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat((frequencies.sin(), frequencies.cos()), dim=-1)


class _ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = _SinusoidalActionTime(hidden_size)

    def forward(self, actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape [batch]")
        expanded_time = timesteps.unsqueeze(1).expand(-1, horizon)
        action_embedding = self.layer1(actions)
        time_embedding = self.pos_encoding(expanded_time).to(action_embedding.dtype)
        hidden = F.silu(self.layer2(torch.cat((action_embedding, time_embedding), dim=-1)))
        return self.layer3(hidden)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(value)))


class _TimestepProjection(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.channels // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        ) / (half_dim - 1)
        embedding = timesteps.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
        return torch.cat((embedding.cos(), embedding.sin()), dim=-1)


class _TimestepEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class _TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = _TimestepProjection(256)
        self.timestep_embedder = _TimestepEmbedding(256, embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timesteps))


class _AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1.0e-5)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(time_embedding)).chunk(2, dim=1)
        return self.norm(hidden) * (1.0 + scale[:, None]) + shift[:, None]


class _Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None,
        heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        key_value_dim = query_dim if cross_attention_dim is None else cross_attention_dim
        inner_dim = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_k = nn.Linear(key_value_dim, inner_dim)
        self.to_v = nn.Linear(key_value_dim, inner_dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key_value = hidden if encoder_hidden is None else encoder_hidden
        query = self._heads(self.to_q(hidden))
        key = self._heads(self.to_k(key_value))
        value = self._heads(self.to_v(key_value))
        mask = _attention_mask(attention_mask, query.shape[0], key.shape[2])
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
        )
        merged = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        return self.to_out(merged)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)


class _GELU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(hidden), approximate="tanh")


class _FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float, final_dropout: bool) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            _GELU(dim, 4 * dim),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
        ]
        if final_dropout:
            modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        cross_attention_dim: int | None,
        heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = _AdaLayerNorm(dim)
        self.attn1 = _Attention(dim, cross_attention_dim, heads, head_dim, dropout)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1.0e-5)
        self.ff = _FeedForward(dim, dropout, final_dropout=True)
        self.final_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(hidden, time_embedding)
        attended = self.attn1(normalized, encoder_hidden, encoder_attention_mask)
        hidden = hidden + self.final_dropout(attended)
        return hidden + self.ff(self.norm3(hidden))


class _DiT(nn.Module):
    def __init__(self, config: M0DiTConfig) -> None:
        super().__init__()
        dim = config.input_embedding_dim
        self.interleave_self_attention = config.interleave_self_attention
        self.timestep_encoder = _TimestepEncoder(dim)
        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    dim,
                    None
                    if self.interleave_self_attention and index % 2 == 1
                    else config.vlm_hidden_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.dropout,
                )
                for index in range(config.num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1.0e-6)
        self.proj_out_1 = nn.Linear(dim, 2 * dim)
        self.proj_out_2 = nn.Linear(dim, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_embedding = self.timestep_encoder(timestep).to(hidden_states.dtype)
        for index, block in enumerate(self.transformer_blocks):
            self_attention = self.interleave_self_attention and index % 2 == 1
            hidden_states = block(
                hidden_states,
                None if self_attention else encoder_hidden_states,
                None if self_attention else encoder_attention_mask,
                time_embedding,
            )
        shift, scale = self.proj_out_1(F.silu(time_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1.0 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


class M0DiTActionHead(nn.Module):
    """AL0 clean-action head initialized from the upstream ABot-M0 layout."""

    def __init__(self, config: M0DiTConfig = M0DiTConfig()) -> None:
        super().__init__()
        self.config = config
        self.model = _DiT(config)
        self.state_encoder = _MLP(
            config.state_dim, config.hidden_size, config.input_embedding_dim
        )
        self.action_encoder = _ActionEncoder(config.action_dim, config.input_embedding_dim)
        self.action_decoder = _MLP(config.hidden_size, config.hidden_size, config.action_dim)
        self.future_tokens = nn.Embedding(
            config.num_target_vision_tokens, config.input_embedding_dim
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, device="cpu"),
            torch.tensor(config.noise_beta_beta, device="cpu"),
        )

    def forward(
        self,
        vl_embeddings: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        action_dimension_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = self._validate_inputs(vl_embeddings, actions, state)
        state = _state_token(state)
        mask = _action_dimension_mask(
            action_dimension_mask, batch_size, self.config.action_dim, actions.device
        )
        noise = torch.randn_like(actions) if noise is None else noise.to(actions)
        if noise.shape != actions.shape:
            raise ValueError("noise must have the same shape as actions")
        if mask is not None:
            expanded_mask = mask[:, None, :]
            actions = actions * expanded_mask
            noise = noise * expanded_mask
        if time is None:
            sample = self.beta_dist.sample((batch_size,)).to(actions)
            time = (self.config.noise_s - sample) / self.config.noise_s
        else:
            time = time.to(actions)
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        time_3d = time[:, None, None]
        noisy_actions = time_3d * actions + (1.0 - time_3d) * noise
        predicted_actions = self._predict_clean(
            vl_embeddings,
            state,
            noisy_actions,
            time,
            encoder_attention_mask,
        )
        denominator = (1.0 - time_3d).clamp_min(self.config.time_epsilon)
        target_velocity = (actions - noisy_actions) / denominator
        predicted_velocity = (predicted_actions - noisy_actions) / denominator
        squared_error = (predicted_velocity - target_velocity).square()
        if mask is None:
            return squared_error.mean()
        expanded_mask = mask[:, None, :].expand_as(squared_error)
        return (squared_error * expanded_mask).sum() / expanded_mask.sum()

    @torch.no_grad()
    def sample(
        self,
        vl_embeddings: torch.Tensor,
        state: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
        action_dimension_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        batch_size = vl_embeddings.shape[0]
        state = _state_token(state)
        if state.shape != (batch_size, 1, self.config.state_dim):
            raise ValueError("state must have shape [batch, 1, state_dim]")
        sample_steps = self.config.num_inference_timesteps if steps is None else steps
        if sample_steps <= 0:
            raise ValueError("steps must be positive")
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        actions = (
            torch.randn(shape, device=state.device, dtype=state.dtype)
            if noise is None
            else noise.to(state).clone()
        )
        if actions.shape != shape:
            raise ValueError("noise has the wrong shape")
        mask = _action_dimension_mask(
            action_dimension_mask, batch_size, self.config.action_dim, state.device
        )
        if mask is not None:
            actions *= mask[:, None, :]
        step_size = 1.0 / sample_steps
        for index in range(sample_steps):
            time = torch.full(
                (batch_size,),
                index / sample_steps,
                device=state.device,
                dtype=state.dtype,
            )
            predicted = self._predict_clean(
                vl_embeddings, state, actions, time, encoder_attention_mask
            )
            actions = actions + step_size * (predicted - actions) / (
                1.0 - time[:, None, None]
            )
            if mask is not None:
                actions *= mask[:, None, :]
        return actions

    def _validate_inputs(
        self,
        vl_embeddings: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
    ) -> int:
        if vl_embeddings.ndim != 3 or vl_embeddings.shape[-1] != self.config.vlm_hidden_dim:
            raise ValueError("vl_embeddings has the wrong shape")
        batch_size = vl_embeddings.shape[0]
        if actions.shape != (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError("actions has the wrong shape")
        if _state_token(state).shape != (batch_size, 1, self.config.state_dim):
            raise ValueError("state must have shape [batch, 1, state_dim]")
        return batch_size

    def _predict_clean(
        self,
        vl_embeddings: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        discrete_time = (time * self.config.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_actions, discrete_time)
        positions = torch.arange(
            action_features.shape[1], device=action_features.device
        )
        action_features = action_features + self.position_embedding(positions)[None]
        state_features = self.state_encoder(state)
        future = self.future_tokens.weight[None].expand(vl_embeddings.shape[0], -1, -1)
        hidden = torch.cat((state_features, future, action_features), dim=1)
        output = self.model(
            hidden,
            vl_embeddings,
            discrete_time,
            encoder_attention_mask,
        )
        return self.action_decoder(output)[:, -self.config.action_horizon :]


def transfer_robocasa_action_weights(
    model: M0DiTActionHead,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
    *,
    reinitialize_keys: Sequence[str] = tuple(
        sorted(GO2_X5_REINITIALIZED_ACTION_KEYS)
    ),
) -> ActionTransferReport:
    """Load all compatible action tensors and fail on unapproved differences."""

    source_state = (
        torch.load(
            Path(checkpoint),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if isinstance(checkpoint, (str, Path))
        else checkpoint
    )
    source = {
        key.removeprefix("action_model."): value
        for key, value in source_state.items()
        if key.startswith("action_model.")
    }
    target = model.state_dict()
    reinitialized = set(reinitialize_keys)
    if reinitialized != GO2_X5_REINITIALIZED_ACTION_KEYS:
        raise ValueError("Go2-X5 action boundary reinitialization set changed")
    if not reinitialized <= target.keys():
        raise RuntimeError("target model lacks a required reinitialized tensor")
    unexpected = set(source) - set(target)
    missing = set(target) - set(source)
    if unexpected or missing:
        raise RuntimeError(
            f"checkpoint action structure mismatch: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    compatible: dict[str, torch.Tensor] = {}
    bad_shapes: list[str] = []
    for key, target_value in target.items():
        if key in reinitialized:
            continue
        source_value = source[key]
        if source_value.shape != target_value.shape:
            bad_shapes.append(
                f"{key}: source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
            )
        else:
            compatible[key] = source_value
    if bad_shapes:
        raise RuntimeError("unapproved checkpoint shape mismatch: " + "; ".join(bad_shapes))
    result = model.load_state_dict(compatible, strict=False)
    if set(result.missing_keys) != reinitialized or result.unexpected_keys:
        raise RuntimeError("checkpoint transfer result disagrees with the migration contract")
    return ActionTransferReport(
        loaded_keys=tuple(sorted(compatible)),
        reinitialized_keys=tuple(sorted(reinitialized)),
    )


def transfer_conveyorvla_action_trunk(
    model: M0DiTActionHead,
    checkpoint: Mapping[str, torch.Tensor],
) -> ActionTransferReport:
    """Reuse a trained 10-D AL0 trunk while resetting only domain I/O shapes."""

    source = {
        key.removeprefix("action_model."): value
        for key, value in checkpoint.items()
    }
    target_shapes = parameter_state_shapes(model)
    unexpected = set(source) - set(target_shapes)
    missing = set(target_shapes) - set(source)
    if unexpected or missing:
        raise RuntimeError(
            f"ConveyorVLA action structure mismatch: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    compatible: dict[str, torch.Tensor] = {}
    reinitialized: set[str] = set()
    for key, target_shape in target_shapes.items():
        source_value = source[key]
        if not isinstance(source_value, torch.Tensor):
            raise RuntimeError(f"checkpoint value is not a tensor: {key}")
        if source_value.shape == target_shape:
            compatible[key] = source_value
        elif key in DOMAIN_ACTION_REINITIALIZED_KEYS:
            reinitialized.add(key)
        else:
            raise RuntimeError(
                f"unapproved domain-head shape mismatch for {key}: "
                f"source={tuple(source_value.shape)} target={tuple(target_shape)}"
            )
    if reinitialized != DOMAIN_ACTION_REINITIALIZED_KEYS:
        raise RuntimeError(
            "domain action transfer must reinitialize exactly the action I/O tensors"
        )
    copy_parameter_tensors(model, compatible)
    return ActionTransferReport(
        loaded_keys=tuple(sorted(compatible)),
        reinitialized_keys=tuple(sorted(reinitialized)),
    )


def parameter_state_shapes(module: nn.Module) -> dict[str, torch.Size]:
    """Return logical parameter shapes before or after ZeRO-3 partitioning."""

    parameters = dict(module.named_parameters(remove_duplicate=False))
    state_keys = set(module.state_dict())
    if state_keys != set(parameters):
        missing = state_keys - set(parameters)
        extra = set(parameters) - state_keys
        raise RuntimeError(
            f"checkpoint contract requires parameter-only state: buffers={sorted(missing)}, "
            f"extra_parameters={sorted(extra)}"
        )
    return {
        key: torch.Size(getattr(parameter, "ds_shape", parameter.shape))
        for key, parameter in parameters.items()
    }


def copy_parameter_tensors(
    module: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    """Copy exact tensors into ordinary or ZeRO-3 partitioned parameters."""

    parameters = dict(module.named_parameters(remove_duplicate=False))
    if not set(tensors) <= set(parameters):
        raise RuntimeError("checkpoint contains a tensor that is not a model parameter")
    partitioned = any(hasattr(parameters[key], "ds_id") for key in tensors)
    if partitioned:
        try:
            import deepspeed
            import torch.distributed as distributed
        except ImportError as error:
            raise RuntimeError("ZeRO-3 checkpoint loading requires DeepSpeed") from error
        if not distributed.is_initialized():
            raise RuntimeError("ZeRO-3 checkpoint loading requires distributed init")
        rank = distributed.get_rank()
        for key, source in tensors.items():
            parameter = parameters[key]
            with deepspeed.zero.GatheredParameters([parameter], modifier_rank=0):
                if rank == 0:
                    parameter.data.copy_(
                        source.to(device=parameter.device, dtype=parameter.dtype)
                    )
        return
    with torch.no_grad():
        for key, source in tensors.items():
            parameter = parameters[key]
            parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def _state_token(state: torch.Tensor) -> torch.Tensor:
    return state.unsqueeze(1) if state.ndim == 2 else state


def _action_dimension_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    action_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    if mask.shape != (batch_size, action_dim):
        raise ValueError("action_dimension_mask has the wrong shape")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("each action mask must enable at least one dimension")
    return mask


def _attention_mask(
    mask: torch.Tensor | None,
    batch_size: int,
    key_length: int,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.shape != (batch_size, key_length):
        raise ValueError("encoder_attention_mask must have shape [batch, tokens]")
    return mask.to(dtype=torch.bool)[:, None, None, :]


__all__ = [
    "DOMAIN_ACTION_REINITIALIZED_KEYS",
    "GO2_X5_REINITIALIZED_ACTION_KEYS",
    "ActionTransferReport",
    "M0DiTActionHead",
    "M0DiTConfig",
    "copy_parameter_tensors",
    "parameter_state_shapes",
    "transfer_robocasa_action_weights",
    "transfer_conveyorvla_action_trunk",
]
