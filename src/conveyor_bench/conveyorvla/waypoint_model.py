"""Qwen3-VL router and dual layerwise flow-matching waypoint policy."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.dit import (
    _ActionEncoder,
    _MLP,
    _TimestepEncoder,
    _TransformerBlock,
    _action_valid_mask,
)
from conveyor_bench.conveyorvla.policy import Qwen3VLInterface
from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    HISTORY_SPAN_S,
    PRED_ACTION_TOKEN,
    PRED_DONE_TOKEN,
    ROUTE_TOKENS,
    SPECIAL_TOKENS,
    SUBTASK_END_TOKEN,
    SUBTASK_START_TOKEN,
    WaypointActionDomain,
    WaypointRoute,
    action_domain,
)


@dataclass(frozen=True)
class WaypointTokenIds:
    pred_action: int
    pred_done: int
    route_ids: tuple[int, ...]
    subtask_start: int
    subtask_end: int

    @property
    def route_by_id(self) -> Mapping[int, WaypointRoute]:
        return {
            token_id: route
            for token_id, route in zip(
                self.route_ids,
                _ACTIVE_ROUTES,
                strict=True,
            )
        }


@dataclass(frozen=True)
class ConstrainedRouteDecision:
    route: WaypointRoute | None
    assistant_prefix: str
    subtask_text: str
    route_confidence: float
    decision_probs: Mapping[str, float]
    route_probs: Mapping[str, float]
    valid: bool
    recover_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.route is WaypointRoute.DONE

    @property
    def action_domain(self) -> WaypointActionDomain:
        return (
            WaypointActionDomain.NONE
            if self.route is None
            else action_domain(self.route)
        )


@dataclass(frozen=True)
class WaypointPrediction:
    decision: ConstrainedRouteDecision
    normalized_action: tuple[tuple[float, ...], ...] | None


@dataclass(frozen=True)
class LayerwiseFlowMatchingConfig:
    action_dim: int
    action_horizon: int = ACTION_HORIZON
    cross_attention_dim: int = 2560
    hidden_size: int = 1024
    num_layers: int = 16
    num_attention_heads: int = 16
    attention_head_dim: int = 64
    dropout: float = 0.0
    max_seq_len: int = 1024
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4

    def __post_init__(self) -> None:
        integers = (
            self.action_dim,
            self.action_horizon,
            self.cross_attention_dim,
            self.hidden_size,
            self.num_layers,
            self.num_attention_heads,
            self.attention_head_dim,
            self.max_seq_len,
            self.num_timestep_buckets,
            self.num_inference_timesteps,
        )
        if any(value <= 0 for value in integers):
            raise ValueError("all layerwise FM dimensions and counts must be positive")
        if self.action_horizon != ACTION_HORIZON:
            raise ValueError(f"waypoint action horizon must be {ACTION_HORIZON}")
        if self.num_attention_heads * self.attention_head_dim != self.hidden_size:
            raise ValueError("attention heads must exactly cover hidden_size")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if not 0.0 < self.noise_s <= 1.0:
            raise ValueError("noise_s must be within (0, 1]")


@dataclass(frozen=True)
class WaypointLossConfig:
    lambda_answer: float = 1.0
    lambda_route: float = 1.0
    lambda_nav: float = 1.0
    lambda_arm: float = 1.0
    repeated_diffusion_steps: int = 1

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.lambda_answer,
                self.lambda_route,
                self.lambda_nav,
                self.lambda_arm,
            )
        ):
            raise ValueError("waypoint loss weights must be finite and non-negative")
        if self.repeated_diffusion_steps <= 0:
            raise ValueError("repeated_diffusion_steps must be positive")


_ACTIVE_ROUTES = (
    WaypointRoute.NAV_TO_SOURCE,
    WaypointRoute.PICK,
    WaypointRoute.NAV_TO_TARGET,
    WaypointRoute.PLACE,
)


class WaypointQwenInterface(Qwen3VLInterface):
    """Local-only Qwen interface with the exact waypoint special-token set."""

    @classmethod
    def from_local(
        cls,
        model_dir: str | Path,
        *,
        dtype: torch.dtype = torch.bfloat16,
        attention_implementation: str | None = None,
    ) -> "WaypointQwenInterface":
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise M0MobileError(
                "Transformers with Qwen3-VL support is required for waypoint v1"
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
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        if len(processor.tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        interface = cls(model, processor)
        interface.waypoint_token_ids = waypoint_token_ids(interface)
        return interface

    def build_waypoint_inputs(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        solutions: Sequence[str] | None = None,
        supervise_solutions: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        if not examples:
            raise ValueError("waypoint examples must be non-empty")
        return self.build_temporal_inputs(
            [example["video"] for example in examples],
            [str(example["lang"]) for example in examples],
            history_span_s=HISTORY_SPAN_S,
            solutions=solutions,
            supervise_solutions=supervise_solutions,
        )


class ConstrainedWaypointRouter:
    """Fail-closed ACTION/DONE + route-token + bounded-subtask decoder."""

    def __init__(
        self,
        qwen: WaypointQwenInterface,
        *,
        route_confidence_min: float,
        max_subtask_tokens: int = 24,
    ) -> None:
        if not 0.0 <= route_confidence_min <= 1.0:
            raise ValueError("route_confidence_min must be within [0, 1]")
        if max_subtask_tokens <= 0:
            raise ValueError("max_subtask_tokens must be positive")
        self.qwen = qwen
        self.token_ids = waypoint_token_ids(qwen)
        self.route_confidence_min = float(route_confidence_min)
        self.max_subtask_tokens = int(max_subtask_tokens)

    @torch.inference_mode()
    def decode(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[ConstrainedRouteDecision, ...]:
        if not examples:
            raise ValueError("waypoint router examples must be non-empty")
        inputs = dict(self.qwen.build_waypoint_inputs(examples))
        inputs.pop("labels", None)
        model = self.qwen.model
        was_training = model.training
        model.eval()
        try:
            first_output = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            decision_candidates = torch.tensor(
                [self.token_ids.pred_action, self.token_ids.pred_done],
                device=first_output.logits.device,
            )
            decision_probs = torch.softmax(
                first_output.logits[:, -1].index_select(-1, decision_candidates).float(),
                dim=-1,
            )
            decision_choice = decision_probs.argmax(dim=-1)
            if bool((decision_choice == 0).any()):
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
                    self.token_ids.route_ids,
                    device=route_output.logits.device,
                )
                route_probs = torch.softmax(
                    route_output.logits[:, -1].index_select(-1, route_candidates).float(),
                    dim=-1,
                )
                route_choice = route_probs.argmax(dim=-1)
                chosen_route_ids = route_candidates.index_select(0, route_choice).to(
                    device=inputs["input_ids"].device,
                    dtype=inputs["input_ids"].dtype,
                )[:, None]
                subtask_ids = torch.full_like(chosen_route_ids, self.token_ids.subtask_start)
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
                generated_ids = (
                    generated.sequences if hasattr(generated, "sequences") else generated
                )
                suffix = generated_ids[:, generation_inputs["input_ids"].shape[1] :]
            else:
                route_probs = torch.zeros(
                    (len(examples), len(_ACTIVE_ROUTES)),
                    device=decision_probs.device,
                    dtype=decision_probs.dtype,
                )
                route_choice = torch.zeros(
                    len(examples), device=decision_probs.device, dtype=torch.long
                )
                suffix = torch.empty(
                    (len(examples), 0),
                    device=inputs["input_ids"].device,
                    dtype=inputs["input_ids"].dtype,
                )
        finally:
            model.train(was_training)

        results = []
        tokenizer = self.qwen.processor.tokenizer
        for index in range(len(examples)):
            action_probability = float(decision_probs[index, 0].item())
            done_probability = float(decision_probs[index, 1].item())
            probability_map = {
                "ACTION": action_probability,
                "DONE": done_probability,
            }
            route_probability_map = {
                route.value: float(route_probs[index, route_index].item())
                for route_index, route in enumerate(_ACTIVE_ROUTES)
            }
            if int(decision_choice[index].item()) == 1:
                results.append(
                    ConstrainedRouteDecision(
                        route=WaypointRoute.DONE,
                        assistant_prefix=PRED_DONE_TOKEN,
                        subtask_text="",
                        route_confidence=done_probability,
                        decision_probs=probability_map,
                        route_probs=route_probability_map,
                        valid=True,
                    )
                )
                continue
            route = _ACTIVE_ROUTES[int(route_choice[index].item())]
            confidence = action_probability * route_probability_map[route.value]
            end_positions = torch.nonzero(
                suffix[index] == self.token_ids.subtask_end,
                as_tuple=False,
            )
            if not end_positions.numel():
                results.append(
                    _recover_decision(
                        "missing_end_subtask",
                        confidence,
                        probability_map,
                        route_probability_map,
                    )
                )
                continue
            end = int(end_positions[0].item())
            if end > self.max_subtask_tokens:
                results.append(
                    _recover_decision(
                        "subtask_too_long",
                        confidence,
                        probability_map,
                        route_probability_map,
                    )
                )
                continue
            text_ids = suffix[index, :end]
            if text_ids.numel() == 0 or any(
                int(token_id) in {
                    self.token_ids.pred_action,
                    self.token_ids.pred_done,
                    *self.token_ids.route_ids,
                    self.token_ids.subtask_start,
                }
                for token_id in text_ids
            ):
                results.append(
                    _recover_decision(
                        "invalid_subtask_tokens",
                        confidence,
                        probability_map,
                        route_probability_map,
                    )
                )
                continue
            subtask_text = tokenizer.decode(
                text_ids.detach().cpu().tolist(),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            assistant_prefix = (
                PRED_ACTION_TOKEN
                + ROUTE_TOKENS[route]
                + SUBTASK_START_TOKEN
                + subtask_text
                + SUBTASK_END_TOKEN
            )
            if confidence < self.route_confidence_min:
                results.append(
                    _recover_decision(
                        "route_confidence_below_threshold",
                        confidence,
                        probability_map,
                        route_probability_map,
                    )
                )
                continue
            results.append(
                ConstrainedRouteDecision(
                    route=route,
                    assistant_prefix=assistant_prefix,
                    subtask_text=subtask_text,
                    route_confidence=confidence,
                    decision_probs=probability_map,
                    route_probs=route_probability_map,
                    valid=True,
                )
            )
        return tuple(results)


class LayerwiseFlowMatchingActionHead(nn.Module):
    """State-free DiT whose blocks consume the matching last Qwen layers."""

    def __init__(self, config: LayerwiseFlowMatchingConfig) -> None:
        super().__init__()
        self.config = config
        self.action_encoder = _ActionEncoder(config.action_dim, config.hidden_size)
        self.action_decoder = _MLP(config.hidden_size, config.hidden_size, config.action_dim)
        self.future_tokens = nn.Embedding(config.action_horizon, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.timestep_encoder = _TimestepEncoder(config.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                _TransformerBlock(
                    config.hidden_size,
                    config.cross_attention_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
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
        action_valid_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layers, batch_size = self._validate_inputs(
            layerwise_vl_embeddings,
            actions,
            encoder_attention_mask,
        )
        valid = _action_valid_mask(
            action_valid_mask,
            batch_size,
            self.config.action_horizon,
            actions.device,
        )
        if valid is None or not bool(valid.any()) or not torch.all(valid.any(dim=1)):
            raise ValueError("each layerwise FM sample needs a valid action prefix")
        noise = torch.randn_like(actions) if noise is None else noise.to(actions)
        if noise.shape != actions.shape:
            raise ValueError("noise must have the same shape as actions")
        if time is None:
            beta = self.beta_dist.sample((batch_size,)).to(actions)
            time = (self.config.noise_s - beta) / self.config.noise_s
        else:
            time = time.to(actions)
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [batch]")
        time_3d = time[:, None, None]
        noisy = (1.0 - time_3d) * noise + time_3d * actions
        target_velocity = actions - noise
        predicted_velocity = self._velocity(
            layers,
            noisy,
            time,
            encoder_attention_mask,
        )
        element_mask = valid[:, :, None].expand_as(actions)
        squared_error = (predicted_velocity - target_velocity).square()
        return (squared_error * element_mask).sum() / element_mask.sum()

    @torch.no_grad()
    def sample(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        *,
        encoder_attention_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        layers = tuple(layerwise_vl_embeddings)
        if len(layers) != self.config.num_layers:
            raise ValueError("wrong number of layerwise Qwen hidden states")
        batch_size = layers[0].shape[0]
        sample_steps = self.config.num_inference_timesteps if steps is None else steps
        if sample_steps <= 0:
            raise ValueError("flow-matching sample steps must be positive")
        shape = (batch_size, self.config.action_horizon, self.config.action_dim)
        actions = (
            torch.randn(shape, device=layers[0].device, dtype=layers[0].dtype)
            if noise is None
            else noise.to(layers[0]).clone()
        )
        if actions.shape != shape:
            raise ValueError("flow-matching noise has the wrong shape")
        step_size = 1.0 / sample_steps
        for index in range(sample_steps):
            time = torch.full(
                (batch_size,),
                index / sample_steps,
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = actions + step_size * self._velocity(
                layers,
                actions,
                time,
                encoder_attention_mask,
            )
        return actions

    def _validate_inputs(
        self,
        layerwise_vl_embeddings: Sequence[torch.Tensor],
        actions: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], int]:
        layers = tuple(layerwise_vl_embeddings)
        if len(layers) != self.config.num_layers:
            raise ValueError("wrong number of layerwise Qwen hidden states")
        if not layers or any(
            value.ndim != 3 or value.shape[-1] != self.config.cross_attention_dim
            for value in layers
        ):
            raise ValueError("layerwise Qwen hidden states have the wrong shape")
        batch_size, token_count = layers[0].shape[:2]
        if any(value.shape[:2] != (batch_size, token_count) for value in layers):
            raise ValueError("layerwise Qwen hidden states are not aligned")
        if actions.shape != (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError("waypoint actions have the wrong shape")
        if encoder_attention_mask.shape != (batch_size, token_count):
            raise ValueError("Qwen attention mask has the wrong shape")
        return layers, batch_size

    def _velocity(
        self,
        layers: Sequence[torch.Tensor],
        actions: torch.Tensor,
        time: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        discrete_time = (time * self.config.num_timestep_buckets).long()
        action_features = self.action_encoder(actions, discrete_time)
        positions = torch.arange(actions.shape[1], device=actions.device)
        action_features = action_features + self.position_embedding(positions)[None]
        future = self.future_tokens.weight[None].expand(actions.shape[0], -1, -1)
        hidden = torch.cat((future, action_features), dim=1)
        time_embedding = self.timestep_encoder(discrete_time).to(hidden.dtype)
        for block, encoder_hidden in zip(
            self.transformer_blocks,
            layers,
            strict=True,
        ):
            hidden = block(
                hidden,
                encoder_hidden,
                encoder_attention_mask,
                time_embedding,
            )
        return self.action_decoder(hidden[:, -self.config.action_horizon :])


class ConveyorVLAWaypointPolicy(nn.Module):
    """Oracle-prefix main training and online-identical self conditioning."""

    def __init__(
        self,
        qwen: WaypointQwenInterface,
        navigation_head: LayerwiseFlowMatchingActionHead,
        manipulation_head: LayerwiseFlowMatchingActionHead,
        *,
        route_confidence_min: float,
        max_subtask_tokens: int = 24,
        loss_config: WaypointLossConfig = WaypointLossConfig(),
        route_class_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if navigation_head.config.action_dim != 3:
            raise ValueError("navigation waypoint head must output 3 dimensions")
        if manipulation_head.config.action_dim != 7:
            raise ValueError("manipulation waypoint head must output 7 dimensions")
        if navigation_head.config.num_layers != manipulation_head.config.num_layers:
            raise ValueError("NAV and ARM heads must have equal layer counts")
        self.qwen = qwen
        self.navigation_head = navigation_head
        self.manipulation_head = manipulation_head
        self.router = ConstrainedWaypointRouter(
            qwen,
            route_confidence_min=route_confidence_min,
            max_subtask_tokens=max_subtask_tokens,
        )
        self.loss_config = loss_config
        self.route_class_weights = {
            route.value: float((route_class_weights or {}).get(route.value, 1.0))
            for route in WaypointRoute
        }
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.route_class_weights.values()
        ):
            raise ValueError("route class weights must be finite and positive")

    def enable_full_finetuning(self) -> None:
        self.qwen.enable_full_finetuning()
        self.navigation_head.requires_grad_(True)
        self.manipulation_head.requires_grad_(True)

    def forward(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        objective: str = "oracle",
    ) -> Mapping[str, torch.Tensor | int | float]:
        if objective == "oracle":
            return self.oracle_loss(examples)
        if objective == "self_conditioned":
            return self.self_conditioned_loss(examples)
        raise ValueError("waypoint objective must be oracle or self_conditioned")

    def oracle_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, torch.Tensor | int | float]:
        if not examples:
            raise ValueError("oracle waypoint examples must be non-empty")
        inputs = dict(
            self.qwen.build_waypoint_inputs(
                examples,
                solutions=[str(example["solution"]) for example in examples],
            )
        )
        outputs = self.qwen(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if outputs.loss is None or outputs.hidden_states is None:
            raise RuntimeError("Qwen did not return oracle CE and hidden states")
        route_loss = self._route_token_loss(examples, inputs["labels"], outputs.logits)
        layers = self._last_action_layers(outputs.hidden_states)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        nav_loss, nav_samples = self._domain_loss(
            examples,
            layers,
            attention_mask,
            domain=WaypointActionDomain.NAVIGATION,
            head=self.navigation_head,
        )
        arm_loss, arm_samples = self._domain_loss(
            examples,
            layers,
            attention_mask,
            domain=WaypointActionDomain.MANIPULATION,
            head=self.manipulation_head,
        )
        config = self.loss_config
        total = (
            config.lambda_answer * outputs.loss
            + config.lambda_route * route_loss
            + config.lambda_nav * nav_loss
            + config.lambda_arm * arm_loss
        )
        return {
            "loss": total,
            "answer_loss": outputs.loss,
            "route_loss": route_loss,
            "navigation_loss": nav_loss,
            "manipulation_loss": arm_loss,
            "navigation_samples": nav_samples,
            "manipulation_samples": arm_samples,
            "done_samples": sum(
                str(example["route"]) == WaypointRoute.DONE.value
                for example in examples
            ),
        }

    def self_conditioned_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, torch.Tensor | int]:
        decisions = self.router.decode(examples)
        matched_indices = [
            index
            for index, decision in enumerate(decisions)
            if decision.valid
            and decision.route is not None
            and decision.route is not WaypointRoute.DONE
            and decision.route.value == str(examples[index]["route"])
            and any(bool(value) for value in examples[index]["action_valid_mask"])
        ]
        route_matches = len(matched_indices)
        route_mismatches = sum(
            decision.valid
            and decision.route is not None
            and decision.route.value != str(examples[index]["route"])
            for index, decision in enumerate(decisions)
        )
        route_recovers = sum(not decision.valid for decision in decisions)
        if not matched_indices:
            zero = _parameter_touch(self.navigation_head) + _parameter_touch(
                self.manipulation_head
            )
            return {
                "loss": zero,
                "self_conditioned_loss": zero,
                "navigation_loss": _parameter_touch(self.navigation_head),
                "manipulation_loss": _parameter_touch(self.manipulation_head),
                "navigation_samples": 0,
                "manipulation_samples": 0,
                "route_matches": route_matches,
                "route_mismatches": route_mismatches,
                "route_recovers": route_recovers,
            }
        selected = [examples[index] for index in matched_indices]
        inputs = dict(
            self.qwen.build_waypoint_inputs(
                selected,
                solutions=[decisions[index].assistant_prefix for index in matched_indices],
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
        layers = self._last_action_layers(outputs.hidden_states)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        nav_loss, nav_samples = self._domain_loss(
            selected,
            layers,
            attention_mask,
            domain=WaypointActionDomain.NAVIGATION,
            head=self.navigation_head,
        )
        arm_loss, arm_samples = self._domain_loss(
            selected,
            layers,
            attention_mask,
            domain=WaypointActionDomain.MANIPULATION,
            head=self.manipulation_head,
        )
        count = nav_samples + arm_samples
        combined = (
            nav_samples * nav_loss + arm_samples * arm_loss
        ) / max(1, count)
        return {
            "loss": combined,
            "self_conditioned_loss": combined,
            "navigation_loss": nav_loss,
            "manipulation_loss": arm_loss,
            "navigation_samples": nav_samples,
            "manipulation_samples": arm_samples,
            "route_matches": route_matches,
            "route_mismatches": route_mismatches,
            "route_recovers": route_recovers,
        }

    @torch.inference_mode()
    def predict(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[WaypointPrediction, ...]:
        decisions = self.router.decode(examples)
        active_indices = [
            index
            for index, decision in enumerate(decisions)
            if decision.valid
            and decision.route is not None
            and decision.route is not WaypointRoute.DONE
        ]
        actions: list[tuple[tuple[float, ...], ...] | None] = [None] * len(examples)
        if active_indices:
            selected = [examples[index] for index in active_indices]
            inputs = dict(
                self.qwen.build_waypoint_inputs(
                    selected,
                    solutions=[decisions[index].assistant_prefix for index in active_indices],
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
            layers = self._last_action_layers(outputs.hidden_states)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                raise RuntimeError("Qwen processor did not return an attention mask")
            for domain, head in (
                (WaypointActionDomain.NAVIGATION, self.navigation_head),
                (WaypointActionDomain.MANIPULATION, self.manipulation_head),
            ):
                local_indices = [
                    local_index
                    for local_index, global_index in enumerate(active_indices)
                    if decisions[global_index].action_domain is domain
                ]
                if not local_indices:
                    continue
                index_tensor = torch.as_tensor(local_indices, device=layers[0].device)
                head_device = next(head.parameters()).device
                head_dtype = next(head.parameters()).dtype
                selected_layers = tuple(
                    layer.index_select(0, index_tensor).to(
                        device=head_device,
                        dtype=head_dtype,
                    )
                    for layer in layers
                )
                selected_attention = attention_mask.index_select(0, index_tensor).to(
                    head_device
                )
                with _action_autocast(head_device, head_dtype):
                    sampled = head.sample(
                        selected_layers,
                        encoder_attention_mask=selected_attention,
                    )
                for local_index, value in zip(
                    local_indices,
                    sampled.float().cpu().tolist(),
                    strict=True,
                ):
                    actions[active_indices[local_index]] = tuple(
                        tuple(float(component) for component in row) for row in value
                    )
        return tuple(
            WaypointPrediction(decision=decision, normalized_action=actions[index])
            for index, decision in enumerate(decisions)
        )

    def _last_action_layers(
        self, hidden_states: Sequence[torch.Tensor] | None
    ) -> tuple[torch.Tensor, ...]:
        if hidden_states is None:
            raise RuntimeError("Qwen did not return hidden states")
        count = self.navigation_head.config.num_layers
        if len(hidden_states) < count:
            raise RuntimeError("Qwen has fewer hidden layers than the action heads")
        return tuple(hidden_states[-count:])

    def _domain_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        layers: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
        *,
        domain: WaypointActionDomain,
        head: LayerwiseFlowMatchingActionHead,
    ) -> tuple[torch.Tensor, int]:
        indices = [
            index
            for index, example in enumerate(examples)
            if str(example["action_domain"]) == domain.value
            and example.get("action") is not None
            and any(bool(value) for value in example["action_valid_mask"])
        ]
        if not indices:
            return _parameter_touch(head), 0
        index_tensor = torch.as_tensor(indices, device=layers[0].device)
        device = next(head.parameters()).device
        dtype = next(head.parameters()).dtype
        selected_layers = tuple(
            layer.index_select(0, index_tensor).to(device=device, dtype=dtype)
            for layer in layers
        )
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
        selected_attention = attention_mask.index_select(0, index_tensor).to(device)
        repeats = self.loss_config.repeated_diffusion_steps
        with _action_autocast(device, dtype):
            loss = head(
                tuple(layer.repeat(repeats, 1, 1) for layer in selected_layers),
                actions.repeat(repeats, 1, 1),
                encoder_attention_mask=selected_attention.repeat(repeats, 1),
                action_valid_mask=valid.repeat(repeats, 1),
            )
        return loss, len(indices)

    def _route_token_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        token_ids = self.router.token_ids
        decision_candidates = torch.tensor(
            [token_ids.pred_action, token_ids.pred_done],
            device=logits.device,
        )
        route_candidates = torch.tensor(token_ids.route_ids, device=logits.device)
        losses = []
        weights = []
        for row_index, example in enumerate(examples):
            route = WaypointRoute(str(example["route"]))
            decision_target = (
                token_ids.pred_done
                if route is WaypointRoute.DONE
                else token_ids.pred_action
            )
            decision_position = _label_position(labels[row_index], decision_target)
            losses.append(
                F.cross_entropy(
                    logits[row_index, decision_position - 1]
                    .index_select(0, decision_candidates)
                    .float()[None],
                    torch.tensor(
                        [1 if route is WaypointRoute.DONE else 0],
                        device=logits.device,
                    ),
                    reduction="none",
                )[0]
            )
            weights.append(self.route_class_weights[route.value])
            if route is not WaypointRoute.DONE:
                route_index = _ACTIVE_ROUTES.index(route)
                route_position = _label_position(
                    labels[row_index], token_ids.route_ids[route_index]
                )
                losses.append(
                    F.cross_entropy(
                        logits[row_index, route_position - 1]
                        .index_select(0, route_candidates)
                        .float()[None],
                        torch.tensor([route_index], device=logits.device),
                        reduction="none",
                    )[0]
                )
                weights.append(self.route_class_weights[route.value])
        weight_tensor = torch.as_tensor(weights, device=logits.device, dtype=losses[0].dtype)
        return (torch.stack(losses) * weight_tensor).sum() / weight_tensor.sum()


def waypoint_token_ids(interface: Qwen3VLInterface) -> WaypointTokenIds:
    tokenizer = interface.processor.tokenizer

    def single(token: str) -> int:
        encoded = tokenizer(token, add_special_tokens=False)
        values = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if len(values) != 1:
            raise M0MobileError(f"waypoint token must encode as one token: {token!r} -> {values}")
        token_id = int(values[0])
        if token_id < 0 or tokenizer.convert_tokens_to_ids(token) != token_id:
            raise M0MobileError(f"waypoint tokenizer ID is invalid for {token!r}")
        return token_id

    ids = WaypointTokenIds(
        pred_action=single(PRED_ACTION_TOKEN),
        pred_done=single(PRED_DONE_TOKEN),
        route_ids=tuple(single(ROUTE_TOKENS[route]) for route in _ACTIVE_ROUTES),
        subtask_start=single(SUBTASK_START_TOKEN),
        subtask_end=single(SUBTASK_END_TOKEN),
    )
    all_ids = (
        ids.pred_action,
        ids.pred_done,
        *ids.route_ids,
        ids.subtask_start,
        ids.subtask_end,
    )
    if len(set(all_ids)) != len(SPECIAL_TOKENS):
        raise M0MobileError("waypoint special tokens must have unique token IDs")
    return ids


def lambda_self_schedule(progress: float) -> float:
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("optimizer progress must be within [0, 1]")
    if progress <= 0.05:
        return 0.0
    if progress < 0.40:
        return 0.5 * (progress - 0.05) / 0.35
    return 0.5


def _append_tokens(
    inputs: Mapping[str, torch.Tensor], token_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in inputs.items() if key != "labels"}
    if token_ids.ndim != 2 or token_ids.shape[0] != inputs["input_ids"].shape[0]:
        raise ValueError("forced token IDs must have shape [batch, tokens]")
    result["input_ids"] = torch.cat((inputs["input_ids"], token_ids), dim=1)
    if "attention_mask" in inputs:
        result["attention_mask"] = torch.cat(
            (
                inputs["attention_mask"],
                torch.ones_like(token_ids, dtype=inputs["attention_mask"].dtype),
            ),
            dim=1,
        )
    # Qwen3-VL owns multimodal RoPE position construction.  Extending a cached
    # 1-D position tensor here would be incorrect for its 3-axis mRoPE layout.
    result.pop("position_ids", None)
    result.pop("cache_position", None)
    return result


def _recover_decision(
    reason: str,
    confidence: float,
    decision_probs: Mapping[str, float],
    route_probs: Mapping[str, float],
) -> ConstrainedRouteDecision:
    return ConstrainedRouteDecision(
        route=None,
        assistant_prefix="",
        subtask_text="",
        route_confidence=confidence,
        decision_probs=decision_probs,
        route_probs=route_probs,
        valid=False,
        recover_reason=reason,
    )


def _label_position(labels: torch.Tensor, token_id: int) -> int:
    positions = torch.nonzero(labels == token_id, as_tuple=False)
    if positions.numel() != 1:
        raise RuntimeError(
            f"supervised assistant must contain token ID {token_id} exactly once"
        )
    position = int(positions[0].item())
    if position <= 0:
        raise RuntimeError("assistant token has no preceding prediction position")
    return position


def _parameter_touch(module: nn.Module) -> torch.Tensor:
    terms = [parameter.sum() * 0.0 for parameter in module.parameters()]
    if not terms:
        raise RuntimeError("waypoint action head unexpectedly has no parameters")
    return torch.stack(terms).sum()


def _action_autocast(device: torch.device, dtype: torch.dtype) -> Any:
    mixed = (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)) or (
        device.type == "cpu" and dtype == torch.bfloat16
    )
    return torch.autocast(device_type=device.type, dtype=dtype) if mixed else nullcontext()


__all__ = [
    "ConstrainedRouteDecision",
    "ConstrainedWaypointRouter",
    "ConveyorVLAWaypointPolicy",
    "LayerwiseFlowMatchingActionHead",
    "LayerwiseFlowMatchingConfig",
    "WaypointLossConfig",
    "WaypointPrediction",
    "WaypointQwenInterface",
    "WaypointTokenIds",
    "lambda_self_schedule",
    "waypoint_token_ids",
]
