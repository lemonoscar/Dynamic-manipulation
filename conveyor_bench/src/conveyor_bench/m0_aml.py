"""Minimal, dependency-light AML action head for M0-Mobile smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AMLConfig:
    action_dim: int = 10
    state_dim: int = 28
    context_dim: int = 64
    action_horizon: int = 16
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 512
    inference_steps: int = 4
    time_epsilon: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "action_dim",
            "state_dim",
            "context_dim",
            "action_horizon",
            "hidden_dim",
            "num_heads",
            "num_layers",
            "feedforward_dim",
            "inference_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 < self.time_epsilon < 1.0:
            raise ValueError("time_epsilon must be within (0, 1)")


class AMLActionHead(nn.Module):
    """Predict clean action chunks and train with M0-style velocity matching."""

    def __init__(self, config: AMLConfig = AMLConfig()) -> None:
        super().__init__()
        self.config = config
        self.context_projection = nn.Linear(config.context_dim, config.hidden_dim)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, config.action_horizon, config.hidden_dim)
        )
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, config.num_layers)
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        batch_size = noisy_actions.shape[0]
        if context.ndim != 3 or context.shape[0] != batch_size:
            raise ValueError("context must have shape [batch, tokens, context_dim]")
        if context.shape[-1] != config.context_dim:
            raise ValueError("context has the wrong feature dimension")
        if state.shape != (batch_size, config.state_dim):
            raise ValueError("state must have shape [batch, state_dim]")
        if noisy_actions.shape != (
            batch_size,
            config.action_horizon,
            config.action_dim,
        ):
            raise ValueError("noisy_actions has the wrong shape")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        memory = torch.cat(
            (
                self.state_projection(state).unsqueeze(1),
                self.context_projection(context),
            ),
            dim=1,
        )
        action_tokens = (
            self.action_projection(noisy_actions)
            + self.time_projection(time[:, None]).unsqueeze(1)
            + self.position_embedding
        )
        return self.output(self.decoder(action_tokens, memory))

    def aml_loss(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        action_dimension_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = target_actions.shape[0]
        if noise is None:
            noise = torch.randn_like(target_actions)
        if noise.shape != target_actions.shape:
            raise ValueError("noise and target_actions must have the same shape")
        noise = noise.to(
            device=target_actions.device,
            dtype=target_actions.dtype,
        )
        if time is None:
            sample = torch.distributions.Beta(1.5, 1.0).sample((batch_size,))
            time = (0.999 - sample).to(
                device=target_actions.device,
                dtype=target_actions.dtype,
            ) / 0.999
        elif time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        else:
            time = time.to(
                device=target_actions.device,
                dtype=target_actions.dtype,
            )
        time_3d = time[:, None, None]
        noisy_actions = time_3d * target_actions + (1.0 - time_3d) * noise
        predicted_actions = self(context, state, noisy_actions, time)
        denominator = (1.0 - time_3d).clamp_min(self.config.time_epsilon)
        target_velocity = (target_actions - noisy_actions) / denominator
        predicted_velocity = (predicted_actions - noisy_actions) / denominator
        squared_error = (predicted_velocity - target_velocity).square()
        if action_dimension_mask is None:
            return squared_error.mean()
        mask = action_dimension_mask.to(
            device=target_actions.device,
            dtype=target_actions.dtype,
        )
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if mask.shape != (batch_size, self.config.action_dim):
            raise ValueError("action_dimension_mask has the wrong shape")
        if not torch.any(mask):
            raise ValueError("action_dimension_mask must enable at least one dimension")
        expanded_mask = mask[:, None, :].expand_as(squared_error)
        return (squared_error * expanded_mask).sum() / expanded_mask.sum()

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        state: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        sample_steps = self.config.inference_steps if steps is None else steps
        if sample_steps <= 0:
            raise ValueError("steps must be positive")
        shape = (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        actions = (
            torch.randn(shape, device=state.device, dtype=state.dtype)
            if noise is None
            else noise.to(device=state.device, dtype=state.dtype).clone()
        )
        if actions.shape != shape:
            raise ValueError("noise has the wrong shape")
        step_size = 1.0 / sample_steps
        for index in range(sample_steps):
            time = torch.full(
                (batch_size,),
                index / sample_steps,
                device=state.device,
                dtype=state.dtype,
            )
            predicted_actions = self(context, state, actions, time)
            denominator = 1.0 - time[:, None, None]
            actions = actions + step_size * (
                predicted_actions - actions
            ) / denominator
        return actions


__all__ = ["AMLActionHead", "AMLConfig"]
