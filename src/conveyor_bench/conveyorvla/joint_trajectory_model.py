"""Two-pass Qwen policy with final-layer ABot-M0 NAV and Mani DiT experts."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.dit import (
    M0DiTActionHead,
    _ActionEncoder,
    _AdaLayerNorm,
    _Attention,
    _FeedForward,
    _MLP,
    _TimestepEncoder,
)
from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    ACTIVE_SPECIAL_TOKENS,
    HISTORY_SPAN_S,
    MANIPULATION_ACTION_DIM,
    MANIPULATION_STATE_DIM,
    NAVIGATION_ACTION_DIM,
    PRED_ACTION_TOKEN,
    ROUTE_TOKENS,
    SUBTASK_END_TOKEN,
    SUBTASK_START_TOKEN,
    TRANSITION_TAU_S,
    TRAIN_BOUNDARY_PAIRS_PER_BATCH,
    TRAIN_DOMAIN_ROWS_PER_BATCH,
    TRAIN_GLOBAL_BATCH_SIZE,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
    transition_routes,
)
from conveyor_bench.conveyorvla.waypoint_model import WaypointQwenInterface
from conveyor_bench.conveyorvla.waypoint import SPECIAL_TOKENS as WAYPOINT_SPECIAL_TOKENS


ACTIVE_ROUTES = tuple(JointTrajectoryRoute)


@dataclass(frozen=True)
class JointTrajectoryTokenIds:
    pred_action: int
    route_ids: tuple[int, int, int, int]
    subtask_start: int
    subtask_end: int


@dataclass(frozen=True)
class JointTrajectoryRouteDecision:
    route: JointTrajectoryRoute | None
    assistant_prefix: str
    subtask_text: str
    route_confidence: float
    route_probs: Mapping[str, float]
    valid: bool
    recover_reason: str | None = None


@dataclass(frozen=True)
class JointTrajectoryPrediction:
    decision: JointTrajectoryRouteDecision
    normalized_action: tuple[tuple[float, ...], ...] | None


@dataclass(frozen=True)
class JointTrajectoryExpertConfig:
    action_dim: int
    state_dim: int
    action_horizon: int = ACTION_HORIZON
    cross_attention_dim: int = 2560
    hidden_size: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 64
    dropout: float = 0.0
    max_seq_len: int = 32
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 10

    def __post_init__(self) -> None:
        positive = (
            "action_dim",
            "action_horizon",
            "cross_attention_dim",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "max_seq_len",
            "num_timestep_buckets",
            "num_inference_timesteps",
        )
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError("joint-trajectory expert dimensions must be positive")
        if self.state_dim not in {0, MANIPULATION_STATE_DIM}:
            raise ValueError("expert state_dim must be zero or the 13D Mani state")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError(f"joint-trajectory horizon must be {ACTION_HORIZON}")
        if self.num_attention_heads * self.attention_head_dim != self.hidden_size:
            raise ValueError("expert attention dimensions must equal hidden_size")
        if self.max_seq_len < ACTION_HORIZON + int(self.state_dim > 0):
            raise ValueError("expert max_seq_len is shorter than its action/state tokens")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("expert dropout must be within [0,1)")
        if not 0.0 < self.noise_s <= 1.0:
            raise ValueError("expert noise_s must be within (0,1]")
        if self.num_inference_timesteps != 10:
            raise ValueError("joint-trajectory v1 inference must use ten FM integration steps")


@dataclass(frozen=True)
class JointTrajectoryLossConfig:
    lambda_answer: float = 1.0
    lambda_route: float = 1.0
    lambda_navigation: float = 1.0
    lambda_manipulation: float = 1.0
    lambda_boundary: float = 0.2
    lambda_progress: float = 0.1
    manipulation_joint_weight: float = 0.75
    manipulation_gripper_weight: float = 0.25
    repeated_diffusion_steps: int = 1
    boundary_rank_margin: float = 0.2

    def __post_init__(self) -> None:
        weights = (
            self.lambda_answer,
            self.lambda_route,
            self.lambda_navigation,
            self.lambda_manipulation,
            self.lambda_boundary,
            self.lambda_progress,
            self.manipulation_joint_weight,
            self.manipulation_gripper_weight,
            self.boundary_rank_margin,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("joint-trajectory loss weights must be finite and non-negative")
        if not math.isclose(
            self.manipulation_joint_weight + self.manipulation_gripper_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Mani joint/gripper weights must sum to one")
        if self.repeated_diffusion_steps != 1:
            raise ValueError("joint-trajectory v1 uses exactly one FM noise/time draw")


@dataclass(frozen=True)
class SelectiveWarmstartReport:
    loaded: tuple[str, ...]
    reinitialized: tuple[str, ...]
    rejected: tuple[str, ...]
    source_to_target: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "conveyorvla-joint-trajectory-warmstart-report-v1",
            "loaded": list(self.loaded),
            "reinitialized": list(self.reinitialized),
            "rejected": list(self.rejected),
            "source_to_target": dict(sorted(self.source_to_target.items())),
        }


class JointTrajectoryQwenInterface(WaypointQwenInterface):
    """Qwen interface that exposes only video/task/prefix to both passes."""

    def build_joint_trajectory_inputs(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        solutions: Sequence[str] | None = None,
        supervise_solutions: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        if not examples:
            raise ValueError("joint-trajectory examples must be non-empty")
        # Deliberately select only these three fields.  mani_state and all
        # training truth remain outside Qwen even when present in the row.
        return self.build_temporal_inputs(
            [example["video"] for example in examples],
            [str(example["lang"]) for example in examples],
            history_span_s=HISTORY_SPAN_S,
            solutions=solutions,
            supervise_solutions=supervise_solutions,
        )


class JointTrajectoryRouter:
    """Constrained ACTION + four-way route + bounded subtask decoder.

    There is no DONE candidate and no confidence threshold.  Runtime temporal
    confirmation is handled by RouteCommitter, outside the model request.
    """

    def __init__(
        self,
        qwen: JointTrajectoryQwenInterface,
        *,
        max_subtask_tokens: int = 24,
    ) -> None:
        if max_subtask_tokens <= 0:
            raise ValueError("max_subtask_tokens must be positive")
        self.qwen = qwen
        self.token_ids = joint_trajectory_token_ids(qwen)
        self.max_subtask_tokens = int(max_subtask_tokens)

    @torch.inference_mode()
    def decode(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> tuple[JointTrajectoryRouteDecision, ...]:
        if not examples:
            raise ValueError("joint-trajectory router examples must be non-empty")
        inputs = dict(self.qwen.build_joint_trajectory_inputs(examples))
        inputs.pop("labels", None)
        model = self.qwen.model
        was_training = model.training
        model.eval()
        try:
            action_ids = torch.full(
                (len(examples), 1),
                self.token_ids.pred_action,
                device=inputs["input_ids"].device,
                dtype=inputs["input_ids"].dtype,
            )
            action_inputs = _append_tokens(inputs, action_ids)
            route_output = model(
                **action_inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            route_candidates = torch.tensor(
                self.token_ids.route_ids, device=route_output.logits.device
            )
            route_probs = torch.softmax(
                route_output.logits[:, -1]
                .index_select(-1, route_candidates)
                .float(),
                dim=-1,
            )
            route_choice = route_probs.argmax(dim=-1)
            chosen_route_ids = route_candidates.index_select(0, route_choice).to(
                device=inputs["input_ids"].device,
                dtype=inputs["input_ids"].dtype,
            )[:, None]
            subtask_ids = torch.full_like(
                chosen_route_ids, self.token_ids.subtask_start
            )
            generation_inputs = _append_tokens(
                action_inputs,
                torch.cat((chosen_route_ids, subtask_ids), dim=1),
            )
            generated = model.generate(
                **generation_inputs,
                max_new_tokens=self.max_subtask_tokens + 1,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.token_ids.subtask_end,
                synced_gpus=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )
            generated_ids = generated.sequences if hasattr(generated, "sequences") else generated
            suffix = generated_ids[:, generation_inputs["input_ids"].shape[1] :]
        finally:
            model.train(was_training)

        tokenizer = self.qwen.processor.tokenizer
        legacy_done_id = tokenizer.convert_tokens_to_ids("<|pred_done|>")
        forbidden = {
            self.token_ids.pred_action,
            *self.token_ids.route_ids,
            self.token_ids.subtask_start,
        }
        if isinstance(legacy_done_id, int) and legacy_done_id >= 0:
            forbidden.add(legacy_done_id)
        results = []
        for index in range(len(examples)):
            probability_map = {
                route.value: float(route_probs[index, route_index].item())
                for route_index, route in enumerate(ACTIVE_ROUTES)
            }
            route = ACTIVE_ROUTES[int(route_choice[index].item())]
            confidence = probability_map[route.value]
            ends = torch.nonzero(
                suffix[index] == self.token_ids.subtask_end, as_tuple=False
            )
            if not ends.numel():
                results.append(
                    _invalid_decision("missing_end_subtask", confidence, probability_map)
                )
                continue
            end = int(ends[0].item())
            text_ids = suffix[index, :end]
            if (
                end > self.max_subtask_tokens
                or text_ids.numel() == 0
                or any(int(token_id) in forbidden for token_id in text_ids)
            ):
                results.append(
                    _invalid_decision("invalid_subtask_tokens", confidence, probability_map)
                )
                continue
            text = tokenizer.decode(
                text_ids.detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                results.append(_invalid_decision("empty_subtask", confidence, probability_map))
                continue
            prefix = (
                PRED_ACTION_TOKEN
                + ROUTE_TOKENS[route]
                + SUBTASK_START_TOKEN
                + text
                + SUBTASK_END_TOKEN
            )
            results.append(
                JointTrajectoryRouteDecision(
                    route=route,
                    assistant_prefix=prefix,
                    subtask_text=text,
                    route_confidence=confidence,
                    route_probs=probability_map,
                    valid=True,
                )
            )
        return tuple(results)


class _JointTrajectoryBlock(nn.Module):
    """One action self-attention, Qwen cross-attention, and FFN block."""

    def __init__(self, config: JointTrajectoryExpertConfig) -> None:
        super().__init__()
        dim = config.hidden_size
        self.self_norm = _AdaLayerNorm(dim)
        self.self_attention = _Attention(
            dim,
            None,
            config.num_attention_heads,
            config.attention_head_dim,
            config.dropout,
        )
        self.cross_norm = _AdaLayerNorm(dim)
        self.cross_attention = _Attention(
            dim,
            config.cross_attention_dim,
            config.num_attention_heads,
            config.attention_head_dim,
            config.dropout,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1.0e-5)
        self.ff = _FeedForward(dim, config.dropout, final_dropout=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(hidden, time_embedding)
        hidden = hidden + self.dropout(self.self_attention(normalized))
        normalized = self.cross_norm(hidden, time_embedding)
        hidden = hidden + self.dropout(
            self.cross_attention(
                normalized, encoder_hidden, encoder_attention_mask
            )
        )
        return hidden + self.ff(self.ff_norm(hidden))


class JointTrajectoryFlowMatchingExpert(nn.Module):
    """Layerwise FM expert with no unused future-token branch."""

    def __init__(self, config: JointTrajectoryExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = _ActionEncoder(config.action_dim, config.hidden_size)
        self.action_decoder = _MLP(config.hidden_size, config.hidden_size, config.action_dim)
        self.state_encoder = (
            None
            if config.state_dim == 0
            else _MLP(config.state_dim, config.hidden_size, config.hidden_size)
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.timestep_encoder = _TimestepEncoder(config.hidden_size)
        self.blocks = nn.ModuleList(
            [_JointTrajectoryBlock(config) for _ in range(config.num_layers)]
        )
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.beta_dist = Beta(
            torch.tensor(config.noise_beta_alpha, device="cpu"),
            torch.tensor(config.noise_beta_beta, device="cpu"),
        )

    def forward(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        actions: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor,
        state: torch.Tensor | None = None,
        action_valid_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        reduction: str = "element_mean",
    ) -> torch.Tensor:
        error = self.flow_squared_error(
            layerwise_vl_embeddings,
            actions,
            encoder_attention_mask=encoder_attention_mask,
            state=state,
            action_valid_mask=action_valid_mask,
            noise=noise,
            time=time,
        )
        if reduction == "element_mean":
            return error.mean()
        if reduction == "sample_mean":
            return error.mean(dim=(1, 2)).mean()
        if reduction == "dimension_mean":
            return error.mean(dim=(0, 1))
        if reduction == "none":
            return error
        raise ValueError("unsupported joint-trajectory FM reduction")

    def flow_squared_error(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        actions: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor,
        state: torch.Tensor | None = None,
        action_valid_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layers, batch_size = self._validate_inputs(
            layerwise_vl_embeddings, actions, encoder_attention_mask, state
        )
        if action_valid_mask is not None:
            valid = action_valid_mask.to(device=actions.device, dtype=torch.bool)
            if valid.shape != (batch_size, ACTION_HORIZON) or not bool(valid.all()):
                raise ValueError("joint-trajectory FM requires all ten targets valid")
        noise = torch.randn_like(actions) if noise is None else noise.to(actions)
        if noise.shape != actions.shape:
            raise ValueError("FM noise has the wrong shape")
        if time is None:
            beta = self.beta_dist.sample((batch_size,)).to(actions)
            time = (self.config.noise_s - beta) / self.config.noise_s
        else:
            time = time.to(actions)
        if time.shape != (batch_size,) or not bool(torch.isfinite(time).all()):
            raise ValueError("FM time must be finite with shape [batch]")
        noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * actions
        target_velocity = actions - noise
        predicted = self._velocity(
            layers, noisy, time, encoder_attention_mask, state
        )
        return (predicted - target_velocity).square()

    @torch.no_grad()
    def sample(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        *,
        encoder_attention_mask: torch.Tensor,
        state: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        layers = tuple(layerwise_vl_embeddings)
        if len(layers) != self.config.num_layers:
            raise ValueError("wrong number of layerwise Qwen hidden states")
        batch_size = layers[0].shape[0]
        sample_steps = self.config.num_inference_timesteps if steps is None else int(steps)
        if sample_steps <= 0:
            raise ValueError("FM integration steps must be positive")
        shape = (batch_size, ACTION_HORIZON, self.config.action_dim)
        actions = (
            torch.randn(shape, device=layers[0].device, dtype=layers[0].dtype)
            if noise is None
            else noise.to(layers[0]).clone()
        )
        if actions.shape != shape:
            raise ValueError("FM sample noise has the wrong shape")
        step_size = 1.0 / sample_steps
        for index in range(sample_steps):
            time = torch.full(
                (batch_size,),
                index / sample_steps,
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = actions + step_size * self._velocity(
                layers, actions, time, encoder_attention_mask, state
            )
        return actions

    def _validate_inputs(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        actions: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], int]:
        layers = tuple(layerwise_vl_embeddings)
        if len(layers) != self.config.num_layers or any(
            value.ndim != 3 or value.shape[-1] != self.config.cross_attention_dim
            for value in layers
        ):
            raise ValueError("layerwise Qwen hidden states have the wrong shape")
        batch_size, token_count = layers[0].shape[:2]
        if any(value.shape[:2] != (batch_size, token_count) for value in layers):
            raise ValueError("layerwise Qwen hidden states are not aligned")
        if actions.shape != (batch_size, ACTION_HORIZON, self.config.action_dim):
            raise ValueError("joint-trajectory actions have the wrong shape")
        if encoder_attention_mask.shape != (batch_size, token_count):
            raise ValueError("Qwen attention mask has the wrong shape")
        if self.config.state_dim == 0:
            if state is not None:
                raise ValueError("NAV expert must not receive state")
        elif state is None or state.shape != (batch_size, self.config.state_dim):
            raise ValueError("Mani expert requires state shape [batch,13]")
        return layers, batch_size

    def _velocity(
        self,
        layers: Sequence[torch.Tensor],
        actions: torch.Tensor,
        time: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        discrete_time = (time * self.config.num_timestep_buckets).long()
        action_hidden = self.action_encoder(actions, discrete_time)
        hidden_parts = []
        if self.state_encoder is not None:
            if state is None:
                raise ValueError("Mani expert state is missing")
            hidden_parts.append(self.state_encoder(state).unsqueeze(1))
        hidden_parts.append(action_hidden)
        hidden = torch.cat(hidden_parts, dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.position_embedding(positions)[None]
        time_embedding = self.timestep_encoder(discrete_time).to(hidden.dtype)
        for block, encoder_hidden in zip(self.blocks, layers, strict=True):
            hidden = block(
                hidden, encoder_hidden, encoder_attention_mask, time_embedding
            )
        return self.action_decoder(hidden[:, -ACTION_HORIZON:])


class JointTrajectoryAuxiliaryHeads(nn.Module):
    def __init__(self, cross_attention_dim: int, hidden_size: int = 256) -> None:
        super().__init__()
        if cross_attention_dim <= 0 or hidden_size <= 0:
            raise ValueError("auxiliary dimensions must be positive")
        self.progress_head = nn.Sequential(
            nn.Linear(cross_attention_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def progress(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.progress_head(hidden).squeeze(-1))


class ConveyorVLAJointTrajectoryPolicy(nn.Module):
    """Teacher-prefix training and online-identical two-pass inference."""

    def __init__(
        self,
        qwen: JointTrajectoryQwenInterface,
        navigation_expert: M0DiTActionHead,
        manipulation_expert: M0DiTActionHead,
        auxiliary_heads: JointTrajectoryAuxiliaryHeads,
        *,
        max_subtask_tokens: int = 24,
        loss_config: JointTrajectoryLossConfig = JointTrajectoryLossConfig(),
    ) -> None:
        super().__init__()
        if (
            navigation_expert.config.action_dim != NAVIGATION_ACTION_DIM
            or navigation_expert.config.state_dim != 0
            or navigation_expert.config.action_horizon != ACTION_HORIZON
        ):
            raise ValueError("NAV expert contract is incompatible")
        if (
            manipulation_expert.config.action_dim != MANIPULATION_ACTION_DIM
            or manipulation_expert.config.state_dim != MANIPULATION_STATE_DIM
            or manipulation_expert.config.action_horizon != ACTION_HORIZON
        ):
            raise ValueError("Mani expert contract is incompatible")
        if navigation_expert.config.vlm_hidden_dim != manipulation_expert.config.vlm_hidden_dim:
            raise ValueError("NAV and Mani experts need the same Qwen feature width")
        self.qwen = qwen
        self.navigation_expert = navigation_expert
        self.manipulation_expert = manipulation_expert
        self.auxiliary_heads = auxiliary_heads
        self.router = JointTrajectoryRouter(
            qwen, max_subtask_tokens=max_subtask_tokens
        )
        self.loss_config = loss_config
        self._qwen_frozen = False

    def enable_action_warmup(self) -> None:
        self.requires_grad_(False)
        self.navigation_expert.requires_grad_(True)
        self.manipulation_expert.requires_grad_(True)
        self._qwen_frozen = True
        self.qwen.eval()

    def enable_full_finetuning(self) -> None:
        self.requires_grad_(True)
        self.qwen.enable_full_finetuning()
        self._qwen_frozen = False

    def train(self, mode: bool = True) -> "ConveyorVLAJointTrajectoryPolicy":
        super().train(mode)
        if self._qwen_frozen:
            self.qwen.eval()
        return self

    def forward(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, torch.Tensor | int]:
        return self.oracle_loss(examples)

    def oracle_loss(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, torch.Tensor | int]:
        if not examples:
            raise ValueError("joint-trajectory training batch must be non-empty")
        inputs = dict(
            self.qwen.build_joint_trajectory_inputs(
                examples,
                solutions=[str(example["solution"]) for example in examples],
                supervise_solutions=True,
            )
        )
        labels = inputs.pop("labels", None)
        if labels is None:
            raise RuntimeError("Qwen processor did not build assistant labels")
        context = torch.no_grad() if self._qwen_frozen else nullcontext()
        with context:
            outputs = self.qwen(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        answer_labels = self.answer_labels(examples, labels)
        answer_loss = _causal_lm_loss(outputs.logits, answer_labels)
        route_logits, route_targets = self.route_logits_and_targets(
            examples, labels, outputs.logits
        )
        route_row_loss = -(
            route_targets * F.log_softmax(route_logits.float(), dim=-1)
        ).sum(dim=-1)
        route_weights = torch.tensor(
            [float(example.get("route_importance_weight", 1.0)) for example in examples],
            device=route_row_loss.device,
            dtype=route_row_loss.dtype,
        )
        if not bool(torch.isfinite(route_weights).all()) or bool((route_weights <= 0.0).any()):
            raise ValueError("route importance weights must be finite and positive")
        # Balanced route sampling has E[w]=1 for w=4*p_data(route).  Do not
        # renormalize weights inside a micro-batch: a pure-route micro-batch
        # would otherwise cancel the correction entirely.
        route_loss = (route_row_loss * route_weights).mean()
        route_accuracy = (
            route_logits.argmax(dim=-1)
            == torch.tensor(
                [ACTIVE_ROUTES.index(JointTrajectoryRoute(str(example["route"]))) for example in examples],
                device=route_logits.device,
            )
        ).float().mean()
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        nav_loss, nav_samples = self._domain_loss(
            examples,
            hidden,
            attention_mask,
            JointTrajectoryDomain.NAVIGATION,
        )
        mani_loss, mani_joint_loss, mani_gripper_loss, mani_samples = self._domain_loss(
            examples,
            hidden,
            attention_mask,
            JointTrajectoryDomain.MANIPULATION,
        )
        boundary_loss, boundary_pairs = self.boundary_rank_loss(
            examples, route_logits
        )
        pooled = self._assistant_hidden(labels, outputs.hidden_states[-1])
        predicted_progress = self.auxiliary_heads.progress(pooled)
        valid_progress = torch.tensor(
            [bool(example["physical_progress_valid"]) for example in examples],
            device=predicted_progress.device,
            dtype=torch.bool,
        )
        if bool(valid_progress.any()):
            progress_targets = torch.tensor(
                [
                    0.0
                    if not bool(example["physical_progress_valid"])
                    else float(example["physical_progress"])
                    for example in examples
                ],
                device=predicted_progress.device,
                dtype=predicted_progress.dtype,
            )
            progress_loss = F.mse_loss(
                predicted_progress[valid_progress], progress_targets[valid_progress]
            )
            progress_mae = (
                predicted_progress[valid_progress] - progress_targets[valid_progress]
            ).abs().mean()
        else:
            progress_loss = predicted_progress.sum() * 0.0
            progress_mae = progress_loss.detach()
        config = self.loss_config
        micro_batch_size = len(examples)
        domain_scale = (
            TRAIN_GLOBAL_BATCH_SIZE
            / micro_batch_size
            / TRAIN_DOMAIN_ROWS_PER_BATCH
        )
        boundary_scale = (
            TRAIN_GLOBAL_BATCH_SIZE
            / micro_batch_size
            / TRAIN_BOUNDARY_PAIRS_PER_BATCH
        )
        navigation_objective = nav_loss * nav_samples * domain_scale
        manipulation_objective = mani_loss * mani_samples * domain_scale
        boundary_objective = boundary_loss * boundary_pairs * boundary_scale
        total = (
            config.lambda_answer * answer_loss
            + config.lambda_route * route_loss
            + config.lambda_navigation * navigation_objective
            + config.lambda_manipulation * manipulation_objective
            + config.lambda_boundary * boundary_objective
            + config.lambda_progress * progress_loss
        )
        return {
            "loss": total,
            "answer_loss": answer_loss,
            "route_loss": route_loss,
            "route_accuracy": route_accuracy,
            "navigation_loss": nav_loss,
            "navigation_objective": navigation_objective,
            "manipulation_loss": mani_loss,
            "manipulation_objective": manipulation_objective,
            "manipulation_joint_loss": mani_joint_loss,
            "manipulation_gripper_loss": mani_gripper_loss,
            "boundary_loss": boundary_loss,
            "boundary_objective": boundary_objective,
            "boundary_pairs": boundary_pairs,
            "progress_loss": progress_loss,
            "progress_mae": progress_mae,
            "progress_samples": int(valid_progress.sum().item()),
            "navigation_samples": nav_samples,
            "manipulation_samples": mani_samples,
        }

    def answer_labels(
        self,
        examples: Sequence[Mapping[str, Any]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask route always and route-specific text in transition windows."""

        result = labels.clone()
        token_ids = self.router.token_ids
        for row_index, example in enumerate(examples):
            route = JointTrajectoryRoute(str(example["route"]))
            route_id = token_ids.route_ids[ACTIVE_ROUTES.index(route)]
            route_position = _label_position(labels[row_index], route_id)
            result[row_index, route_position] = -100
            if bool(example["transition_window"]):
                start = _label_position(labels[row_index], token_ids.subtask_start)
                end = _label_position(labels[row_index], token_ids.subtask_end)
                if end <= start:
                    raise RuntimeError("subtask token span is invalid")
                result[row_index, start + 1 : end] = -100
        return result

    def route_logits_and_targets(
        self,
        examples: Sequence[Mapping[str, Any]],
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.tensor(
            self.router.token_ids.route_ids, device=logits.device
        )
        rows = []
        targets = []
        for row_index, example in enumerate(examples):
            route = JointTrajectoryRoute(str(example["route"]))
            position = _label_position(
                labels[row_index],
                self.router.token_ids.route_ids[ACTIVE_ROUTES.index(route)],
            )
            rows.append(logits[row_index, position - 1].index_select(0, candidates))
            target = torch.zeros(len(ACTIVE_ROUTES), device=logits.device, dtype=torch.float32)
            if bool(example["transition_window"]):
                transition = str(example["boundary_transition"])
                old, new = transition_routes(transition)
                signed = float(example["boundary_signed_time_s"])
                tau = TRANSITION_TAU_S[transition]
                if abs(signed) <= 3.0 * tau:
                    probability_new = torch.sigmoid(
                        torch.tensor(signed / tau, device=logits.device, dtype=torch.float32)
                    )
                    target[ACTIVE_ROUTES.index(old)] = 1.0 - probability_new
                    target[ACTIVE_ROUTES.index(new)] = probability_new
                else:
                    target[ACTIVE_ROUTES.index(route)] = 1.0
            else:
                target[ACTIVE_ROUTES.index(route)] = 1.0
            targets.append(target)
        return torch.stack(rows), torch.stack(targets)

    def boundary_rank_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        route_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        events: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        for index, example in enumerate(examples):
            if not bool(example["transition_window"]):
                continue
            key = (str(example["episode_id"]), str(example["transition_id"]))
            before, after = events.setdefault(key, ([], []))
            signed = float(example["boundary_signed_time_s"])
            (before if signed < 0.0 else after).append(index)
        losses = []
        for before, after in events.values():
            if not before or not after:
                continue
            before_index = min(
                before, key=lambda index: abs(float(examples[index]["boundary_signed_time_s"]))
            )
            after_index = min(
                after, key=lambda index: abs(float(examples[index]["boundary_signed_time_s"]))
            )
            old, new = transition_routes(str(examples[before_index]["boundary_transition"]))
            old_index = ACTIVE_ROUTES.index(old)
            new_index = ACTIVE_ROUTES.index(new)
            before_score = route_logits[before_index, new_index] - route_logits[before_index, old_index]
            after_score = route_logits[after_index, new_index] - route_logits[after_index, old_index]
            losses.append(
                F.softplus(
                    self.loss_config.boundary_rank_margin
                    - (after_score.float() - before_score.float())
                )
            )
        if not losses:
            return route_logits.sum() * 0.0, 0
        return torch.stack(losses).mean(), len(losses)

    @torch.inference_mode()
    def predict_routes(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> tuple[JointTrajectoryRouteDecision, ...]:
        """Run Pass 1 only; runtime may hold before deciding to run Pass 2."""

        return self.router.decode(examples)

    @torch.inference_mode()
    def predict_actions(
        self,
        examples: Sequence[Mapping[str, Any]],
        decisions: Sequence[JointTrajectoryRouteDecision],
    ) -> tuple[tuple[tuple[float, ...], ...] | None, ...]:
        """Run Pass 2 for already committed model-produced prefixes."""

        if not examples or len(examples) != len(decisions):
            raise ValueError("Pass 2 examples and decisions must be non-empty and aligned")
        if any(not decision.valid or decision.route is None for decision in decisions):
            raise ValueError("Pass 2 requires valid model-produced route decisions")
        actions: list[tuple[tuple[float, ...], ...] | None] = [None] * len(examples)
        inputs = dict(
            self.qwen.build_joint_trajectory_inputs(
                examples,
                solutions=[decision.assistant_prefix for decision in decisions],
                supervise_solutions=False,
            )
        )
        inputs.pop("labels", None)
        outputs = self.qwen(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        hidden = outputs.hidden_states[-1]
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        for domain, expert in (
            (JointTrajectoryDomain.NAVIGATION, self.navigation_expert),
            (JointTrajectoryDomain.MANIPULATION, self.manipulation_expert),
        ):
            indices = [
                index
                for index, decision in enumerate(decisions)
                if action_domain(decision.route) is domain
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(indices, device=hidden.device)
            device = next(expert.parameters()).device
            dtype = next(expert.parameters()).dtype
            selected_hidden = hidden.index_select(0, index_tensor).to(
                device=device,
                dtype=dtype,
            )
            selected_attention = attention_mask.index_select(0, index_tensor).to(device)
            state = None
            if domain is JointTrajectoryDomain.MANIPULATION:
                state = torch.as_tensor(
                    [examples[index]["mani_state"] for index in indices],
                    device=device,
                    dtype=dtype,
                )
            with _action_autocast(device, dtype):
                sampled = expert.sample(
                    selected_hidden,
                    state=state,
                    encoder_attention_mask=selected_attention,
                )
            for index, value in zip(indices, sampled.float().cpu().tolist(), strict=True):
                actions[index] = tuple(
                    tuple(float(component) for component in row) for row in value
                )
        if any(action is None for action in actions):
            raise RuntimeError("Pass 2 did not produce every committed-domain action")
        return tuple(actions)

    @torch.inference_mode()
    def predict(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> tuple[JointTrajectoryPrediction, ...]:
        decisions = self.predict_routes(examples)
        valid_indices = [index for index, decision in enumerate(decisions) if decision.valid]
        actions: list[tuple[tuple[float, ...], ...] | None] = [None] * len(examples)
        if valid_indices:
            selected = [examples[index] for index in valid_indices]
            selected_decisions = [decisions[index] for index in valid_indices]
            sampled = self.predict_actions(selected, selected_decisions)
            for global_index, value in zip(valid_indices, sampled, strict=True):
                actions[global_index] = value
        return tuple(
            JointTrajectoryPrediction(decision, actions[index])
            for index, decision in enumerate(decisions)
        )

    def _domain_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        domain: JointTrajectoryDomain,
    ) -> Any:
        indices = [
            index
            for index, example in enumerate(examples)
            if str(example["action_domain"]) == domain.value
        ]
        expert = (
            self.navigation_expert
            if domain is JointTrajectoryDomain.NAVIGATION
            else self.manipulation_expert
        )
        if not indices:
            zero = self._zero_expert_loss(hidden, attention_mask, expert)
            return (zero, 0) if domain is JointTrajectoryDomain.NAVIGATION else (zero, zero, zero, 0)
        index_tensor = torch.tensor(indices, device=hidden.device)
        device = next(expert.parameters()).device
        dtype = next(expert.parameters()).dtype
        selected_hidden = hidden.index_select(0, index_tensor).to(
            device=device,
            dtype=dtype,
        )
        selected_attention = attention_mask.index_select(0, index_tensor).to(device)
        actions = torch.as_tensor(
            [examples[index]["action"] for index in indices],
            device=device,
            dtype=dtype,
        )
        valid = torch.as_tensor(
            [examples[index]["action_valid_mask"] for index in indices],
            device=device,
            dtype=torch.bool,
        )
        state = None
        if domain is JointTrajectoryDomain.MANIPULATION:
            state = torch.as_tensor(
                [examples[index]["mani_state"] for index in indices],
                device=device,
                dtype=dtype,
            )
        with _action_autocast(device, dtype):
            dimensions = expert(
                selected_hidden,
                actions,
                state=state,
                encoder_attention_mask=selected_attention,
                action_valid_mask=valid,
                reduction="dimension_mean",
            )
        if domain is JointTrajectoryDomain.NAVIGATION:
            return dimensions.mean(), len(indices)
        joint = dimensions[:6].mean()
        gripper = dimensions[6]
        combined = (
            self.loss_config.manipulation_joint_weight * joint
            + self.loss_config.manipulation_gripper_weight * gripper
        )
        return combined, joint, gripper, len(indices)

    def _zero_expert_loss(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        expert: M0DiTActionHead,
    ) -> torch.Tensor:
        if not hidden.shape[0]:
            raise RuntimeError("dummy expert pass needs one Qwen row")
        device = next(expert.parameters()).device
        dtype = next(expert.parameters()).dtype
        selected_hidden = hidden[:1].to(device=device, dtype=dtype)
        actions = torch.zeros(
            (1, ACTION_HORIZON, expert.config.action_dim), device=device, dtype=dtype
        )
        state = (
            None
            if expert.config.state_dim == 0
            else torch.zeros((1, expert.config.state_dim), device=device, dtype=dtype)
        )
        with _action_autocast(device, dtype):
            loss = expert(
                selected_hidden,
                actions,
                state=state,
                encoder_attention_mask=attention_mask[:1].to(device),
                action_valid_mask=torch.ones((1, ACTION_HORIZON), device=device, dtype=torch.bool),
                noise=torch.zeros_like(actions),
                time=torch.zeros(1, device=device, dtype=dtype),
            )
        return loss * 0.0

    def _assistant_hidden(
        self, labels: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack(
            [
                hidden[index, _label_position(labels[index], self.router.token_ids.subtask_end)]
                for index in range(labels.shape[0])
            ]
        )


def joint_trajectory_token_ids(interface: JointTrajectoryQwenInterface) -> JointTrajectoryTokenIds:
    tokenizer = interface.processor.tokenizer

    def single(token: str) -> int:
        encoded = tokenizer(token, add_special_tokens=False)
        values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if len(values) != 1:
            raise M0MobileError(f"joint-trajectory token is not atomic: {token!r}")
        token_id = int(values[0])
        if tokenizer.convert_tokens_to_ids(token) != token_id:
            raise M0MobileError(f"joint-trajectory token ID is invalid: {token!r}")
        return token_id

    ids = JointTrajectoryTokenIds(
        pred_action=single(PRED_ACTION_TOKEN),
        route_ids=tuple(single(ROUTE_TOKENS[route]) for route in ACTIVE_ROUTES),  # type: ignore[arg-type]
        subtask_start=single(SUBTASK_START_TOKEN),
        subtask_end=single(SUBTASK_END_TOKEN),
    )
    all_ids = (ids.pred_action, *ids.route_ids, ids.subtask_start, ids.subtask_end)
    if len(set(all_ids)) != len(ACTIVE_SPECIAL_TOKENS):
        raise M0MobileError("joint-trajectory active tokens must have unique IDs")
    return ids


def reinitialize_joint_trajectory_token_embeddings(
    interface: JointTrajectoryQwenInterface,
) -> tuple[int, ...]:
    """Reset waypoint-token rows that overlap ABot's released action-token IDs."""

    tokenizer = interface.processor.tokenizer
    token_ids = tuple(
        int(tokenizer.convert_tokens_to_ids(token)) for token in WAYPOINT_SPECIAL_TOKENS
    )
    if len(set(token_ids)) != len(WAYPOINT_SPECIAL_TOKENS) or any(
        token_id < 0 for token_id in token_ids
    ):
        raise M0MobileError("waypoint special-token IDs are invalid")
    model = interface.model
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    weights = [input_weight]
    if output_module is not None and output_module.weight is not input_weight:
        weights.append(output_module.weight)
    text_config = getattr(model.config, "text_config", model.config)
    initializer_range = float(getattr(text_config, "initializer_range", 0.02))
    with torch.no_grad():
        for weight in weights:
            if max(token_ids) >= weight.shape[0]:
                raise M0MobileError("waypoint token exceeds the ABot checkpoint vocabulary")
            indices = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
            values = torch.empty(
                (len(token_ids), weight.shape[1]),
                device=weight.device,
                dtype=weight.dtype,
            )
            nn.init.normal_(values, mean=0.0, std=initializer_range)
            weight.index_copy_(0, indices, values)
    return token_ids


def selective_warmstart(
    model: ConveyorVLAJointTrajectoryPolicy,
    source_state: Mapping[str, torch.Tensor],
    *,
    strict_qwen: bool = True,
) -> SelectiveWarmstartReport:
    """Load compatible v2 Qwen/cross-attn/FFN/time weights by explicit map."""

    source = {
        (key[7:] if key.startswith("module.") else key): value.detach().cpu()
        for key, value in source_state.items()
        if isinstance(value, torch.Tensor)
    }
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    source_to_target: dict[str, str] = {}
    rejected: list[str] = []
    reinitialized: list[str] = []
    for target_key, target_value in target.items():
        source_key = _warmstart_source_key(target_key)
        if source_key is None:
            reinitialized.append(target_key)
            continue
        value = source.get(source_key)
        if value is None:
            reinitialized.append(target_key)
            if strict_qwen and target_key.startswith("qwen."):
                rejected.append(f"missing-required:{source_key}")
            continue
        if value.shape != target_value.shape:
            reinitialized.append(target_key)
            rejected.append(
                f"shape:{source_key}:{tuple(value.shape)}->{target_key}:{tuple(target_value.shape)}"
            )
            continue
        compatible[target_key] = value.to(dtype=target_value.dtype)
        source_to_target[source_key] = target_key
    fatal_qwen = [
        value
        for value in rejected
        if value.startswith("missing-required:qwen.")
        or value.startswith("shape:qwen.")
    ]
    if strict_qwen and fatal_qwen:
        raise M0MobileError(
            "selective warm-start has missing or incompatible Qwen tensors: "
            + ", ".join(fatal_qwen[:3])
        )
    result = model.load_state_dict(compatible, strict=False)
    unexpected = sorted(result.unexpected_keys)
    if unexpected:
        raise M0MobileError(f"selective warm-start produced unexpected keys: {unexpected[:3]}")
    selected_sources = set(source_to_target)
    for key in source:
        if key in selected_sources:
            continue
        if any(
            marker in key
            for marker in (
                "future_tokens",
                "action_encoder",
                "action_decoder",
                "position_embedding",
                "auxiliary_heads.progress",
                "prefix",
                "crl",
            )
        ):
            rejected.append(f"semantic-reinit:{key}")
        else:
            rejected.append(f"unused-source:{key}")
    return SelectiveWarmstartReport(
        loaded=tuple(sorted(compatible)),
        reinitialized=tuple(sorted(reinitialized)),
        rejected=tuple(sorted(set(rejected))),
        source_to_target=dict(sorted(source_to_target.items())),
    )


def _warmstart_source_key(target_key: str) -> str | None:
    if target_key.startswith("qwen."):
        return target_key
    for new_domain, old_domain in (
        ("navigation_expert", "navigation_head"),
        ("manipulation_expert", "manipulation_head"),
    ):
        prefix = f"{new_domain}."
        if not target_key.startswith(prefix):
            continue
        suffix = target_key[len(prefix) :]
        if suffix.startswith("timestep_encoder."):
            return f"{old_domain}.{suffix}"
        if suffix.startswith("blocks."):
            mapped = suffix.replace(".cross_norm.", ".norm1.")
            mapped = mapped.replace(".cross_attention.", ".attn1.")
            mapped = mapped.replace(".ff_norm.", ".norm3.")
            mapped = mapped.replace("blocks.", "transformer_blocks.", 1)
            if any(
                marker in suffix
                for marker in (".cross_norm.", ".cross_attention.", ".ff_norm.", ".ff.")
            ):
                return f"{old_domain}.{mapped}"
        return None
    return None


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous().float()
    shifted_labels = labels[:, 1:].contiguous()
    if not bool((shifted_labels != -100).any()):
        raise RuntimeError("answer loss has no supervised tokens")
    return F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _append_tokens(
    inputs: Mapping[str, torch.Tensor], token_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in inputs.items() if key != "labels"}
    if token_ids.ndim != 2 or token_ids.shape[0] != inputs["input_ids"].shape[0]:
        raise ValueError("forced token IDs must have shape [batch,tokens]")
    result["input_ids"] = torch.cat((inputs["input_ids"], token_ids), dim=1)
    if "attention_mask" in inputs:
        result["attention_mask"] = torch.cat(
            (
                inputs["attention_mask"],
                torch.ones_like(token_ids, dtype=inputs["attention_mask"].dtype),
            ),
            dim=1,
        )
    result.pop("position_ids", None)
    result.pop("cache_position", None)
    return result


def _label_position(labels: torch.Tensor, token_id: int) -> int:
    positions = torch.nonzero(labels == token_id, as_tuple=False)
    if positions.numel() != 1:
        raise RuntimeError(f"assistant must contain token ID {token_id} exactly once")
    position = int(positions[0].item())
    if position <= 0:
        raise RuntimeError("assistant token has no preceding prediction position")
    return position


def _invalid_decision(
    reason: str, confidence: float, route_probs: Mapping[str, float]
) -> JointTrajectoryRouteDecision:
    return JointTrajectoryRouteDecision(
        route=None,
        assistant_prefix="",
        subtask_text="",
        route_confidence=max(0.0, min(1.0, float(confidence))),
        route_probs=route_probs,
        valid=False,
        recover_reason=reason,
    )


def _action_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    mixed = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)) or (
        device.type == "cpu" and dtype == torch.bfloat16
    )
    return torch.autocast(device_type=device.type, dtype=dtype) if mixed else nullcontext()


__all__ = [
    "ACTIVE_ROUTES",
    "ConveyorVLAJointTrajectoryPolicy",
    "JointTrajectoryAuxiliaryHeads",
    "JointTrajectoryExpertConfig",
    "JointTrajectoryFlowMatchingExpert",
    "JointTrajectoryLossConfig",
    "JointTrajectoryPrediction",
    "JointTrajectoryQwenInterface",
    "JointTrajectoryRouteDecision",
    "JointTrajectoryRouter",
    "JointTrajectoryTokenIds",
    "SelectiveWarmstartReport",
    "joint_trajectory_token_ids",
    "reinitialize_joint_trajectory_token_embeddings",
    "selective_warmstart",
]
