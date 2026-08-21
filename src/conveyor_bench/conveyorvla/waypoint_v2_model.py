"""Waypoint-v2 transition, prefix, and local-goal objectives."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from conveyor_bench.conveyorvla.waypoint import (
    ACTION_HORIZON,
    PRED_DONE_TOKEN,
    WaypointActionDomain,
    WaypointRoute,
)
from conveyor_bench.conveyorvla.waypoint_model import (
    ConveyorVLAWaypointPolicy,
    ConstrainedRouteDecision,
    LayerwiseFlowMatchingActionHead,
    WaypointLossConfig,
    WaypointQwenInterface,
    _action_autocast,
    _distributed_any,
    _label_position,
    _sample_domain_actions,
    _zero_domain_loss,
)
from conveyor_bench.conveyorvla.waypoint_v2 import LOCAL_CRL_GOALS


_ACTIVE_ROUTES = tuple(LOCAL_CRL_GOALS)
_ROUTE_INDEX = {route.value: index for index, route in enumerate(_ACTIVE_ROUTES)}
_BOUNDARY_INDEX = {"BEFORE": 0, "INTERIOR": 1, "AFTER": 2}


@dataclass(frozen=True)
class WaypointV2AuxiliaryConfig:
    cross_attention_dim: int
    action_hidden_size: int = 1024
    hidden_size: int = 256
    crl_dim: int = 128
    crl_temperature: float = 0.07
    enable_boundary_progress: bool = False
    enable_prefix: bool = False
    enable_crl: bool = False
    tau_route_s: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.cross_attention_dim,
                self.action_hidden_size,
                self.hidden_size,
                self.crl_dim,
            )
        ):
            raise ValueError("waypoint-v2 auxiliary dimensions must be positive")
        if not math.isfinite(self.crl_temperature) or self.crl_temperature <= 0.0:
            raise ValueError("waypoint-v2 CRL temperature must be finite and positive")
        if self.enable_crl:
            tau = self.tau_route_s or {}
            if set(tau) != {route.value for route in _ACTIVE_ROUTES} or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in tau.values()
            ):
                raise ValueError("waypoint-v2 CRL needs one positive tau per active route")


@dataclass(frozen=True)
class WaypointV2LossConfig:
    lambda_answer: float = 1.0
    lambda_route: float = 1.0
    lambda_navigation: float = 1.0
    lambda_manipulation: float = 1.0
    lambda_boundary: float = 0.0
    lambda_progress: float = 0.0
    lambda_prefix: float = 0.0
    lambda_crl: float = 0.0
    repeated_diffusion_steps: int = 1
    on_policy_correction_weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.lambda_answer,
            self.lambda_route,
            self.lambda_navigation,
            self.lambda_manipulation,
            self.lambda_boundary,
            self.lambda_progress,
            self.lambda_prefix,
            self.lambda_crl,
            self.on_policy_correction_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("waypoint-v2 loss weights must be finite and non-negative")
        if self.repeated_diffusion_steps <= 0:
            raise ValueError("waypoint-v2 repeated diffusion steps must be positive")

    def base(self) -> WaypointLossConfig:
        return WaypointLossConfig(
            lambda_answer=self.lambda_answer,
            lambda_route=self.lambda_route,
            lambda_nav=self.lambda_navigation,
            lambda_arm=self.lambda_manipulation,
            repeated_diffusion_steps=self.repeated_diffusion_steps,
        )


@dataclass(frozen=True)
class WaypointV2Prediction:
    decision: ConstrainedRouteDecision
    normalized_action: tuple[tuple[float, ...], ...] | None
    trusted_prefix_k: int | None
    prefix_scores: tuple[float, ...] | None
    boundary_probs: Mapping[str, float] | None
    phase_progress: float | None
    time_to_boundary_s: float | None


class WaypointV2AuxiliaryHeads(nn.Module):
    """Training-only transition/CRL heads plus a runtime prefix scorer."""

    def __init__(self, config: WaypointV2AuxiliaryConfig) -> None:
        super().__init__()
        self.config = config
        cross = config.cross_attention_dim
        hidden = config.hidden_size
        self.boundary_head = nn.Sequential(
            nn.Linear(cross, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        self.boundary_rank_head = nn.Sequential(
            nn.Linear(cross, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.progress_head = nn.Sequential(
            nn.Linear(cross, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.time_to_boundary_head = nn.Sequential(
            nn.Linear(cross, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.route_embedding = nn.Embedding(len(_ACTIVE_ROUTES), 32)
        self.prefix_embedding = nn.Embedding(ACTION_HORIZON, 32)
        self.prefix_action = nn.Linear(7, 32)
        self.prefix_fm_action = nn.Linear(config.action_hidden_size, 32)
        self.prefix_head = nn.Sequential(
            nn.Linear(cross + 128, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.crl_state = nn.Linear(cross, config.crl_dim)
        self.crl_action = nn.Linear(7, config.crl_dim)
        self.goal_byte_embedding = nn.Embedding(256, hidden)
        self.crl_goal = nn.Linear(hidden, config.crl_dim)
        goal_rows = [
            list(LOCAL_CRL_GOALS[route].encode("utf-8")) for route in _ACTIVE_ROUTES
        ]
        width = max(len(row) for row in goal_rows)
        goal_bytes = torch.zeros((len(goal_rows), width), dtype=torch.long)
        goal_mask = torch.zeros((len(goal_rows), width), dtype=torch.bool)
        for index, row in enumerate(goal_rows):
            goal_bytes[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            goal_mask[index, : len(row)] = True
        self.register_buffer("goal_bytes", goal_bytes, persistent=True)
        self.register_buffer("goal_mask", goal_mask, persistent=True)
        self._set_trainable_groups()

    def _set_trainable_groups(self) -> None:
        boundary_modules = (
            self.boundary_head,
            self.boundary_rank_head,
            self.progress_head,
            self.time_to_boundary_head,
        )
        prefix_modules = (
            self.route_embedding,
            self.prefix_embedding,
            self.prefix_action,
            self.prefix_fm_action,
            self.prefix_head,
        )
        crl_modules = (
            self.crl_state,
            self.crl_action,
            self.goal_byte_embedding,
            self.crl_goal,
        )
        for module in boundary_modules:
            module.requires_grad_(self.config.enable_boundary_progress)
        for module in prefix_modules:
            module.requires_grad_(self.config.enable_prefix)
        for module in crl_modules:
            module.requires_grad_(self.config.enable_crl)

    def pool(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
            raise ValueError("waypoint-v2 pooled Qwen inputs are not aligned")
        mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype)[:, :, None]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        parameter = next(self.parameters())
        return pooled.to(device=parameter.device, dtype=parameter.dtype)

    def losses(
        self,
        pooled: torch.Tensor,
        examples: Sequence[Mapping[str, Any]],
        *,
        predicted_decision_probabilities: torch.Tensor | None = None,
        predicted_route_probabilities: torch.Tensor | None = None,
        fm_action_features: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if pooled.shape[0] != len(examples):
            raise ValueError("waypoint-v2 auxiliary batch is not aligned")
        zero = pooled.sum() * 0.0
        result = {
            "boundary_loss": zero,
            "boundary_classification_loss": zero,
            "boundary_rank_loss": zero,
            "boundary_pairwise_loss": zero,
            "route_crossover_loss": zero,
            "time_to_boundary_loss": zero,
            "progress_loss": zero,
            "prefix_loss": zero,
            "crl_loss": zero,
            "boundary_accuracy": zero.detach(),
            "boundary_pairwise_accuracy": zero.detach(),
            "route_crossover_accuracy": zero.detach(),
            "time_to_boundary_mae_s": zero.detach(),
            "progress_mae": zero.detach(),
            "prefix_mae_k": zero.detach(),
            "prefix_overrun_rate": zero.detach(),
            "crl_goal_margin": zero.detach(),
        }
        if self.config.enable_boundary_progress:
            boundary_targets = torch.tensor(
                [_BOUNDARY_INDEX[str(example["boundary_class"])] for example in examples],
                device=pooled.device,
            )
            boundary_logits = self.boundary_head(pooled)
            signed_targets = _jittered_boundary_signed_times(examples)
            soft_boundary_targets = _soft_boundary_targets(
                examples, signed_targets, pooled
            )
            classification_loss = -(
                soft_boundary_targets
                * F.log_softmax(boundary_logits.float(), dim=-1)
            ).sum(dim=-1).mean()
            transition_indices = [
                index
                for index, example in enumerate(examples)
                if bool(example["transition_window"])
                and example.get("boundary_signed_time_s") is not None
            ]
            boundary_rank = torch.tanh(
                self.boundary_rank_head(pooled).squeeze(-1)
            )
            if transition_indices:
                transition_tensor = torch.tensor(
                    transition_indices, device=pooled.device
                )
                rank_targets = torch.tensor(
                    [
                        max(
                            -1.0,
                            min(
                                1.0,
                                float(signed_targets[index]),
                            ),
                        )
                        for index in transition_indices
                    ],
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
                rank_loss = F.smooth_l1_loss(
                    boundary_rank.index_select(0, transition_tensor).float(),
                    rank_targets.float(),
                )
            else:
                rank_loss = zero
            pairwise_loss, pairwise_accuracy = _boundary_pairwise_loss(
                boundary_rank, examples, pooled
            )
            crossover_loss, crossover_accuracy = self._route_crossover_loss(
                examples,
                pooled,
                predicted_decision_probabilities,
                predicted_route_probabilities,
                signed_targets,
            )
            predicted_time = F.softplus(
                self.time_to_boundary_head(pooled).squeeze(-1)
            )
            valid_time_indices = [
                index
                for index, example in enumerate(examples)
                if bool(example["time_to_boundary_valid"])
            ]
            if valid_time_indices:
                time_tensor = torch.tensor(valid_time_indices, device=pooled.device)
                time_targets = torch.tensor(
                    [
                        float(examples[index]["time_to_boundary_s"])
                        for index in valid_time_indices
                    ],
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
                selected_time = predicted_time.index_select(0, time_tensor)
                time_loss = F.smooth_l1_loss(
                    torch.log1p(selected_time.float()),
                    torch.log1p(time_targets.float()),
                )
                time_mae = (selected_time - time_targets).abs().mean()
            else:
                time_loss = zero
                time_mae = zero.detach()
            progress_targets = torch.tensor(
                [float(example["phase_progress"]) for example in examples],
                device=pooled.device,
                dtype=pooled.dtype,
            )
            progress = torch.sigmoid(self.progress_head(pooled).squeeze(-1))
            result["boundary_classification_loss"] = classification_loss
            result["boundary_rank_loss"] = rank_loss
            result["boundary_pairwise_loss"] = pairwise_loss
            result["route_crossover_loss"] = crossover_loss
            result["time_to_boundary_loss"] = time_loss
            result["boundary_loss"] = (
                classification_loss
                + 0.25 * rank_loss
                + 0.25 * pairwise_loss
                + 0.5 * crossover_loss
                + 0.25 * time_loss
            )
            result["progress_loss"] = F.smooth_l1_loss(
                progress.float(), progress_targets.float()
            )
            result["boundary_accuracy"] = (
                boundary_logits.argmax(dim=-1) == boundary_targets
            ).float().mean()
            result["boundary_pairwise_accuracy"] = pairwise_accuracy
            result["route_crossover_accuracy"] = crossover_accuracy
            result["time_to_boundary_mae_s"] = time_mae
            result["progress_mae"] = (progress - progress_targets).abs().mean()

        active = [
            index
            for index, example in enumerate(examples)
            if str(example["route"]) in _ROUTE_INDEX and example.get("action") is not None
        ]
        if not active:
            return result
        index_tensor = torch.tensor(active, device=pooled.device)
        active_pooled = pooled.index_select(0, index_tensor)
        route_indices = torch.tensor(
            [_ROUTE_INDEX[str(examples[index]["route"])] for index in active],
            device=pooled.device,
        )
        actions, valid = _padded_actions(examples, active, pooled)
        if self.config.enable_prefix:
            if (
                predicted_route_probabilities is None
                or predicted_route_probabilities.shape
                != (len(examples), len(_ACTIVE_ROUTES))
                or fm_action_features is None
                or fm_action_features.shape
                != (
                    len(examples),
                    ACTION_HORIZON,
                    self.config.action_hidden_size,
                )
            ):
                raise ValueError(
                    "waypoint-v2 prefix needs predicted route and FM action features"
                )
            scores = self.prefix_scores(
                active_pooled,
                route_indices,
                actions,
                route_probabilities=predicted_route_probabilities.index_select(
                    0, index_tensor
                ),
                fm_action_features=fm_action_features.index_select(0, index_tensor),
            )
            targets = torch.tensor(
                [int(examples[index]["prefix_target_k"]) for index in active],
                device=pooled.device,
            )
            transition = torch.tensor(
                [bool(examples[index]["transition_window"]) for index in active],
                device=pooled.device,
            )
            distribution = _prefix_target_distribution(targets, transition, pooled.dtype)
            result["prefix_loss"] = -(
                distribution * F.log_softmax(scores.float(), dim=-1)
            ).sum(dim=-1).mean()
            prediction = scores.argmax(dim=-1) + 1
            result["prefix_mae_k"] = (prediction - targets).abs().float().mean()
            result["prefix_overrun_rate"] = (prediction > targets).float().mean()
        if self.config.enable_crl:
            action_summary = (actions * valid[:, :, None]).sum(dim=1) / valid.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            state_action = F.normalize(
                self.crl_state(active_pooled) + self.crl_action(action_summary), dim=-1
            )
            goals = self._goal_embeddings().index_select(0, route_indices)
            logits = state_action.float() @ goals.float().T / self.config.crl_temperature
            positive = route_indices[:, None] == route_indices[None, :]
            row_loss = _multi_positive_nce(logits, positive)
            column_loss = _multi_positive_nce(logits.T, positive.T)
            weights = self._crl_weights(examples, active, route_indices, pooled)
            result["crl_loss"] = (
                (row_loss * weights).sum() / weights.sum().clamp_min(1.0)
                + (column_loss * weights).sum() / weights.sum().clamp_min(1.0)
            ) * 0.5
            similarity = state_action.float() @ goals.float().T
            positive_mean = similarity.masked_fill(~positive, 0.0).sum(dim=-1) / positive.sum(
                dim=-1
            ).clamp_min(1)
            negative = similarity.masked_fill(positive, -torch.inf).amax(dim=-1)
            finite_negative = torch.where(
                torch.isfinite(negative), negative, torch.zeros_like(negative)
            )
            result["crl_goal_margin"] = (positive_mean - finite_negative).mean()
        return result

    def _route_crossover_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        reference: torch.Tensor,
        decision_probabilities: torch.Tensor | None,
        route_probabilities: torch.Tensor | None,
        signed_targets: Sequence[float | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = reference.sum() * 0.0
        expected_shape = (len(examples), len(_ACTIVE_ROUTES))
        if (
            decision_probabilities is None
            or decision_probabilities.shape != (len(examples), 2)
            or route_probabilities is None
            or route_probabilities.shape != expected_shape
        ):
            raise ValueError(
                "waypoint-v2 boundary needs predicted decision and route probabilities"
            )
        predicted = []
        targets = []
        for index, example in enumerate(examples):
            transition = example.get("boundary_transition")
            signed = signed_targets[index]
            if transition is None or signed is None:
                continue
            old_route, new_route = str(transition).split("->", maxsplit=1)
            if new_route == WaypointRoute.DONE.value:
                new_probability = decision_probabilities[index, 1]
            else:
                old_index = _ROUTE_INDEX[old_route]
                new_index = _ROUTE_INDEX[new_route]
                old_probability = route_probabilities[index, old_index]
                new_probability = route_probabilities[index, new_index]
                new_probability = new_probability / (
                    old_probability + new_probability
                ).clamp_min(1.0e-6)
            predicted.append(new_probability)
            targets.append(torch.sigmoid(reference.new_tensor(float(signed) / 0.2)))
        if not predicted:
            return zero, zero.detach()
        prediction = torch.stack(predicted).float().clamp(1.0e-6, 1.0 - 1.0e-6)
        target = torch.stack(targets).float()
        loss = F.binary_cross_entropy(prediction, target)
        accuracy = ((prediction >= 0.5) == (target >= 0.5)).float().mean()
        return loss, accuracy

    def prefix_scores(
        self,
        pooled: torch.Tensor,
        route_indices: torch.Tensor,
        actions: torch.Tensor,
        *,
        route_probabilities: torch.Tensor | None = None,
        fm_action_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.shape != (pooled.shape[0], ACTION_HORIZON, 7):
            raise ValueError("waypoint-v2 prefix actions must be [batch,20,7]")
        candidates = torch.arange(ACTION_HORIZON, device=pooled.device)
        pooled_features = pooled[:, None].expand(-1, ACTION_HORIZON, -1)
        route_features = (
            self.route_embedding(route_indices)
            if route_probabilities is None
            else route_probabilities.to(self.route_embedding.weight)
            @ self.route_embedding.weight
        )[:, None].expand(-1, ACTION_HORIZON, -1)
        prefix_features = self.prefix_embedding(candidates)[None].expand(
            pooled.shape[0], -1, -1
        )
        action_features = self.prefix_action(actions)
        if fm_action_features is None or fm_action_features.shape != (
            pooled.shape[0],
            ACTION_HORIZON,
            self.config.action_hidden_size,
        ):
            raise ValueError("waypoint-v2 prefix needs aligned FM action features")
        fm_features = self.prefix_fm_action(fm_action_features)
        return self.prefix_head(
            torch.cat(
                (
                    pooled_features,
                    route_features,
                    prefix_features,
                    action_features,
                    fm_features,
                ),
                dim=-1,
            )
        ).squeeze(-1)

    def _goal_embeddings(self) -> torch.Tensor:
        embedding_device = self.goal_byte_embedding.weight.device
        embedded = self.goal_byte_embedding(self.goal_bytes.to(embedding_device))
        mask = self.goal_mask.to(device=embedding_device, dtype=embedded.dtype)[
            :, :, None
        ]
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.crl_goal(pooled), dim=-1)

    def _crl_weights(
        self,
        examples: Sequence[Mapping[str, Any]],
        active: Sequence[int],
        route_indices: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        del route_indices
        tau = self.config.tau_route_s or {}
        values = []
        for index in active:
            example = examples[index]
            route = str(example["route"])
            time_value = (
                float(example["time_to_boundary_s"])
                if bool(example["time_to_boundary_valid"])
                else float(tau[route])
            )
            weight = math.exp(-max(0.0, time_value) / float(tau[route]))
            if bool(example.get("on_policy_correction", False)):
                weight *= float(example.get("on_policy_correction_weight", 1.0))
            values.append(weight)
        return torch.tensor(values, device=reference.device, dtype=reference.dtype)


class ConveyorVLAWaypointV2Policy(ConveyorVLAWaypointPolicy):
    """One-Qwen-forward v2 training with independently rollbackable heads."""

    def __init__(
        self,
        qwen: WaypointQwenInterface,
        navigation_head: LayerwiseFlowMatchingActionHead,
        manipulation_head: LayerwiseFlowMatchingActionHead,
        auxiliary_heads: WaypointV2AuxiliaryHeads,
        *,
        route_confidence_min: float,
        max_subtask_tokens: int = 24,
        loss_config: WaypointV2LossConfig = WaypointV2LossConfig(),
    ) -> None:
        super().__init__(
            qwen,
            navigation_head,
            manipulation_head,
            route_confidence_min=route_confidence_min,
            max_subtask_tokens=max_subtask_tokens,
            loss_config=loss_config.base(),
        )
        if (
            auxiliary_heads.config.cross_attention_dim
            != navigation_head.config.cross_attention_dim
        ):
            raise ValueError("waypoint-v2 auxiliary and Qwen dimensions differ")
        self.auxiliary_heads = auxiliary_heads
        self.v2_loss_config = loss_config
        enabled = auxiliary_heads.config
        boundary_enabled = (
            loss_config.lambda_boundary > 0.0 or loss_config.lambda_progress > 0.0
        )
        if boundary_enabled != enabled.enable_boundary_progress:
            raise ValueError("waypoint-v2 boundary/progress loss and module flags disagree")
        if (loss_config.lambda_prefix > 0.0) != enabled.enable_prefix:
            raise ValueError("waypoint-v2 prefix loss and module flag disagree")
        if (loss_config.lambda_crl > 0.0) != enabled.enable_crl:
            raise ValueError("waypoint-v2 CRL loss and module flag disagree")

    def enable_v2_finetuning(self) -> None:
        self.qwen.enable_full_finetuning()
        self.navigation_head.requires_grad_(True)
        self.manipulation_head.requires_grad_(True)
        self.auxiliary_heads._set_trainable_groups()

    def oracle_loss(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, torch.Tensor | int | float]:
        if not examples:
            raise ValueError("oracle waypoint-v2 examples must be non-empty")
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
            raise RuntimeError("Qwen did not return waypoint-v2 CE and hidden states")
        route_loss, decision_loss, active_route_loss = self._route_token_loss(
            examples, inputs["labels"], outputs.logits
        )
        layers = self._last_action_layers(outputs.hidden_states)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        nav_loss, nav_samples, nav_draws = self._domain_loss_v2(
            examples,
            layers,
            attention_mask,
            domain=WaypointActionDomain.NAVIGATION,
            head=self.navigation_head,
        )
        arm_loss, arm_samples, arm_draws = self._domain_loss_v2(
            examples,
            layers,
            attention_mask,
            domain=WaypointActionDomain.MANIPULATION,
            head=self.manipulation_head,
        )
        pooled = self.auxiliary_heads.pool(layers[-1], attention_mask)
        decision_probabilities = self._predicted_decision_probabilities(
            examples, inputs["labels"], outputs.logits
        ).to(pooled)
        route_probabilities = self._predicted_route_probabilities(
            examples, inputs["labels"], outputs.logits
        ).to(pooled)
        fm_action_features = (
            self._training_fm_action_features(examples, pooled)
            if self.auxiliary_heads.config.enable_prefix
            else None
        )
        auxiliary = self.auxiliary_heads.losses(
            pooled,
            examples,
            predicted_decision_probabilities=decision_probabilities,
            predicted_route_probabilities=route_probabilities,
            fm_action_features=fm_action_features,
        )
        config = self.v2_loss_config
        total = (
            config.lambda_answer * outputs.loss
            + config.lambda_route * route_loss
            + config.lambda_navigation * nav_loss
            + config.lambda_manipulation * arm_loss
            + config.lambda_boundary * auxiliary["boundary_loss"]
            + config.lambda_progress * auxiliary["progress_loss"]
            + config.lambda_prefix * auxiliary["prefix_loss"]
            + config.lambda_crl * auxiliary["crl_loss"]
        )
        result: dict[str, torch.Tensor | int | float] = {
            "loss": total,
            "answer_loss": outputs.loss,
            "route_loss": route_loss,
            "decision_loss": decision_loss,
            "active_route_loss": active_route_loss,
            "navigation_loss": nav_loss,
            "manipulation_loss": arm_loss,
            "navigation_draw_std": nav_draws.float().std(unbiased=False),
            "manipulation_draw_std": arm_draws.float().std(unbiased=False),
            "navigation_samples": nav_samples,
            "manipulation_samples": arm_samples,
            "done_samples": sum(
                str(example["route"]) == WaypointRoute.DONE.value
                for example in examples
            ),
            **auxiliary,
        }
        for index, value in enumerate(nav_draws):
            result[f"navigation_draw_{index}_loss"] = value
        for index, value in enumerate(arm_draws):
            result[f"manipulation_draw_{index}_loss"] = value
        return result

    def _domain_loss_v2(
        self,
        examples: Sequence[Mapping[str, Any]],
        layers: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
        *,
        domain: WaypointActionDomain,
        head: LayerwiseFlowMatchingActionHead,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        indices = [
            index
            for index, example in enumerate(examples)
            if str(example["action_domain"]) == domain.value
            and example.get("action") is not None
            and any(bool(value) for value in example["action_valid_mask"])
        ]
        repeats = self.v2_loss_config.repeated_diffusion_steps
        if not indices:
            zero = _zero_domain_loss(layers, attention_mask, head)
            return zero, 0, zero.expand(repeats)
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
        with _action_autocast(device, dtype):
            per_sample = head(
                tuple(layer.repeat(repeats, 1, 1) for layer in selected_layers),
                actions.repeat(repeats, 1, 1),
                encoder_attention_mask=selected_attention.repeat(repeats, 1),
                action_valid_mask=valid.repeat(repeats, 1),
                reduction="none",
            )
        per_draw = per_sample.reshape(repeats, len(indices)).mean(dim=1)
        return per_draw.mean(), len(indices), per_draw

    def _predicted_route_probabilities(
        self,
        examples: Sequence[Mapping[str, Any]],
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.tensor(
            self.router.token_ids.route_ids, device=logits.device
        )
        rows = []
        for row_index, example in enumerate(examples):
            route = WaypointRoute(str(example["route"]))
            if route is WaypointRoute.DONE:
                rows.append(torch.zeros(len(_ACTIVE_ROUTES), device=logits.device))
                continue
            route_index = _ACTIVE_ROUTES.index(route)
            position = _label_position(
                labels[row_index], self.router.token_ids.route_ids[route_index]
            )
            rows.append(
                torch.softmax(
                    logits[row_index, position - 1]
                    .index_select(0, candidates)
                    .float(),
                    dim=-1,
                )
            )
        return torch.stack(rows)

    def _predicted_decision_probabilities(
        self,
        examples: Sequence[Mapping[str, Any]],
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.tensor(
            (
                self.router.token_ids.pred_action,
                self.router.token_ids.pred_done,
            ),
            device=logits.device,
        )
        rows = []
        for row_index, example in enumerate(examples):
            route = WaypointRoute(str(example["route"]))
            target = (
                self.router.token_ids.pred_done
                if route is WaypointRoute.DONE
                else self.router.token_ids.pred_action
            )
            position = _label_position(labels[row_index], target)
            rows.append(
                torch.softmax(
                    logits[row_index, position - 1]
                    .index_select(0, candidates)
                    .float(),
                    dim=-1,
                )
            )
        return torch.stack(rows)

    def _training_fm_action_features(
        self,
        examples: Sequence[Mapping[str, Any]],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        values: list[torch.Tensor | None] = [None] * len(examples)
        for domain, head in (
            (WaypointActionDomain.NAVIGATION, self.navigation_head),
            (WaypointActionDomain.MANIPULATION, self.manipulation_head),
        ):
            indices = [
                index
                for index, example in enumerate(examples)
                if str(example["action_domain"]) == domain.value
                and example.get("action") is not None
            ]
            if not indices:
                continue
            parameter = next(head.parameters())
            actions = torch.tensor(
                [examples[index]["action"] for index in indices],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            time = torch.zeros(len(indices), device=parameter.device, dtype=torch.long)
            with _action_autocast(parameter.device, parameter.dtype):
                encoded = head.action_encoder(actions, time)
            for index, feature in zip(indices, encoded, strict=True):
                values[index] = feature.to(reference)
        zero = torch.zeros(
            (
                ACTION_HORIZON,
                self.auxiliary_heads.config.action_hidden_size,
            ),
            device=reference.device,
            dtype=reference.dtype,
        )
        return torch.stack([zero if value is None else value for value in values])

    @torch.inference_mode()
    def fixed_bank_fm_losses(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        bank_seed: int,
        draws: int = 4,
    ) -> Mapping[str, torch.Tensor | int]:
        """Evaluate both FM heads against an order-independent noise/time bank."""

        if not examples or draws <= 0 or bank_seed < 0:
            raise ValueError("fixed FM bank needs examples, positive draws, and a seed")
        inputs = dict(
            self.qwen.build_waypoint_inputs(
                examples,
                solutions=[str(example["solution"]) for example in examples],
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
        result: dict[str, torch.Tensor | int] = {"fixed_bank_draws": draws}
        for name, domain, head in (
            (
                "navigation",
                WaypointActionDomain.NAVIGATION,
                self.navigation_head,
            ),
            (
                "manipulation",
                WaypointActionDomain.MANIPULATION,
                self.manipulation_head,
            ),
        ):
            per_draw, sample_count = self._fixed_bank_domain_loss(
                examples,
                layers,
                attention_mask,
                domain=domain,
                head=head,
                bank_seed=bank_seed,
                draws=draws,
            )
            result[f"{name}_fixed_bank_loss"] = per_draw.mean()
            result[f"{name}_fixed_bank_draw_std"] = per_draw.float().std(
                unbiased=False
            )
            result[f"{name}_fixed_bank_samples"] = sample_count
            for index, value in enumerate(per_draw):
                result[f"{name}_fixed_bank_draw_{index}_loss"] = value
        return result

    def _fixed_bank_domain_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        layers: Sequence[torch.Tensor],
        attention_mask: torch.Tensor,
        *,
        domain: WaypointActionDomain,
        head: LayerwiseFlowMatchingActionHead,
        bank_seed: int,
        draws: int,
    ) -> tuple[torch.Tensor, int]:
        if not math.isclose(head.config.noise_beta_beta, 1.0):
            raise ValueError("fixed FM bank currently requires beta_beta=1")
        indices = [
            index
            for index, example in enumerate(examples)
            if str(example["action_domain"]) == domain.value
            and example.get("action") is not None
            and any(bool(value) for value in example["action_valid_mask"])
        ]
        if not indices:
            zero = _zero_domain_loss(layers, attention_mask, head).detach()
            return zero.expand(draws), 0
        index_tensor = torch.as_tensor(indices, device=layers[0].device)
        parameter = next(head.parameters())
        device, dtype = parameter.device, parameter.dtype
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
        noise_rows = []
        time_rows = []
        for draw in range(draws):
            draw_noise = []
            draw_time = []
            for index in indices:
                sample_id = str(examples[index]["sample_id"])
                generator = torch.Generator(device="cpu").manual_seed(
                    _fixed_bank_seed(bank_seed, domain.value, sample_id, draw)
                )
                draw_noise.append(
                    torch.randn(
                        (ACTION_HORIZON, head.config.action_dim),
                        generator=generator,
                        dtype=torch.float32,
                    )
                )
                beta_uniform = torch.rand((), generator=generator).clamp_min(1.0e-12)
                beta = beta_uniform.pow(1.0 / head.config.noise_beta_alpha)
                draw_time.append((head.config.noise_s - beta) / head.config.noise_s)
            noise_rows.append(torch.stack(draw_noise))
            time_rows.append(torch.stack(draw_time))
        noise = torch.stack(noise_rows).to(device=device, dtype=dtype)
        time = torch.stack(time_rows).to(device=device, dtype=dtype)
        selected_attention = attention_mask.index_select(0, index_tensor).to(device)
        with _action_autocast(device, dtype):
            per_sample = head(
                tuple(layer.repeat(draws, 1, 1) for layer in selected_layers),
                actions.repeat(draws, 1, 1),
                encoder_attention_mask=selected_attention.repeat(draws, 1),
                action_valid_mask=valid.repeat(draws, 1),
                noise=noise.flatten(0, 1),
                time=time.flatten(),
                reduction="none",
            )
        return per_sample.reshape(draws, len(indices)).mean(dim=1), len(indices)

    @torch.inference_mode()
    def oracle_crl_diagnostics(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, float] | None, ...]:
        """Measure local-goal separation without exposing CRL to runtime routing."""

        if not examples:
            raise ValueError("CRL diagnostics need a non-empty batch")
        if not self.auxiliary_heads.config.enable_crl:
            return tuple(None for _ in examples)
        inputs = dict(
            self.qwen.build_waypoint_inputs(
                examples,
                solutions=[str(example["solution"]) for example in examples],
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
        pooled = self.auxiliary_heads.pool(layers[-1], attention_mask)
        active = [
            index
            for index, example in enumerate(examples)
            if str(example["route"]) in _ROUTE_INDEX
            and example.get("action") is not None
        ]
        result: list[Mapping[str, float] | None] = [None] * len(examples)
        if not active:
            return tuple(result)
        index_tensor = torch.tensor(active, device=pooled.device)
        active_pooled = pooled.index_select(0, index_tensor)
        actions, valid = _padded_actions(examples, active, pooled)
        action_summary = (actions * valid[:, :, None]).sum(dim=1) / valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        state_action = F.normalize(
            self.auxiliary_heads.crl_state(active_pooled)
            + self.auxiliary_heads.crl_action(action_summary),
            dim=-1,
        )
        goal_embeddings = self.auxiliary_heads._goal_embeddings()
        similarities = state_action.float() @ goal_embeddings.float().T
        route_indices = torch.tensor(
            [_ROUTE_INDEX[str(examples[index]["route"])] for index in active],
            device=pooled.device,
        )
        correct = similarities.gather(1, route_indices[:, None]).squeeze(1)
        wrong = similarities.scatter(1, route_indices[:, None], -torch.inf).amax(dim=1)
        if len(active) > 1:
            shuffled_summary = action_summary.roll(1, dims=0)
            shuffled_state_action = F.normalize(
                self.auxiliary_heads.crl_state(active_pooled)
                + self.auxiliary_heads.crl_action(shuffled_summary),
                dim=-1,
            )
            shuffled_correct = (
                shuffled_state_action.float() @ goal_embeddings.float().T
            ).gather(1, route_indices[:, None]).squeeze(1)
        else:
            shuffled_correct = correct
        for local, global_index in enumerate(active):
            result[global_index] = {
                "correct_goal_similarity": float(correct[local].item()),
                "wrong_goal_max_similarity": float(wrong[local].item()),
                "goal_margin": float((correct[local] - wrong[local]).item()),
                "shuffled_action_goal_similarity": float(
                    shuffled_correct[local].item()
                ),
                "action_shuffle_drop": float(
                    (correct[local] - shuffled_correct[local]).item()
                ),
            }
        return tuple(result)

    @torch.inference_mode()
    def predict_v2(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> tuple[WaypointV2Prediction, ...]:
        decisions = self.router.decode(examples)
        active_indices = [
            index
            for index, decision in enumerate(decisions)
            if decision.valid
            and decision.route is not None
            and decision.route is not WaypointRoute.DONE
        ]
        actions: list[tuple[tuple[float, ...], ...] | None] = [None] * len(examples)
        trusted: list[int | None] = [None] * len(examples)
        prefix_values: list[tuple[float, ...] | None] = [None] * len(examples)
        boundary_values: list[Mapping[str, float] | None] = [None] * len(examples)
        progress_values: list[float | None] = [None] * len(examples)
        time_values: list[float | None] = [None] * len(examples)
        device = next(self.qwen.parameters()).device
        if _distributed_any(bool(active_indices), device):
            selected_indices = active_indices or [0]
            selected = [examples[index] for index in selected_indices]
            inputs = dict(
                self.qwen.build_waypoint_inputs(
                    selected,
                    solutions=(
                        [decisions[index].assistant_prefix for index in active_indices]
                        if active_indices
                        else [PRED_DONE_TOKEN]
                    ),
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
                    for local_index, global_index in enumerate(selected_indices)
                    if active_indices and decisions[global_index].action_domain is domain
                ]
                sampled = _sample_domain_actions(
                    head, layers, attention_mask, local_indices
                )
                if sampled is None:
                    continue
                for local_index, value in zip(
                    local_indices, sampled.float().cpu().tolist(), strict=True
                ):
                    actions[selected_indices[local_index]] = tuple(
                        tuple(float(component) for component in row) for row in value
                    )
            pooled = self.auxiliary_heads.pool(layers[-1], attention_mask)
            if self.auxiliary_heads.config.enable_boundary_progress:
                logits = self.auxiliary_heads.boundary_head(pooled).float()
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                progress = torch.sigmoid(
                    self.auxiliary_heads.progress_head(pooled).squeeze(-1)
                ).float().cpu().tolist()
                predicted_time = F.softplus(
                    self.auxiliary_heads.time_to_boundary_head(pooled).squeeze(-1)
                ).float().cpu().tolist()
                for local, global_index in enumerate(selected_indices):
                    if not active_indices:
                        continue
                    boundary_values[global_index] = {
                        name: float(probabilities[local][index])
                        for name, index in _BOUNDARY_INDEX.items()
                    }
                    progress_values[global_index] = float(progress[local])
                    time_values[global_index] = float(predicted_time[local])
            if self.auxiliary_heads.config.enable_prefix:
                padded = []
                route_indices = []
                for local, global_index in enumerate(selected_indices):
                    action = actions[global_index]
                    if action is None:
                        padded.append([[0.0] * 7 for _ in range(ACTION_HORIZON)])
                        route_indices.append(0)
                        continue
                    rows = [list(row) + [0.0] * (7 - len(row)) for row in action]
                    padded.append(rows)
                    route = decisions[global_index].route
                    route_indices.append(_ROUTE_INDEX[route.value])  # type: ignore[union-attr]
                route_index_tensor = torch.tensor(
                    route_indices, device=pooled.device
                )
                route_probabilities = torch.tensor(
                    [
                        [
                            decisions[global_index].route_probs[route.value]
                            for route in _ACTIVE_ROUTES
                        ]
                        for global_index in selected_indices
                    ],
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
                fm_features = self._prediction_fm_action_features(
                    decisions,
                    selected_indices,
                    actions,
                    pooled,
                )
                scores = self.auxiliary_heads.prefix_scores(
                    pooled,
                    route_index_tensor,
                    torch.tensor(padded, device=pooled.device, dtype=pooled.dtype),
                    route_probabilities=route_probabilities,
                    fm_action_features=fm_features,
                )
                for local, global_index in enumerate(selected_indices):
                    if not active_indices:
                        continue
                    values = tuple(float(value) for value in scores[local].float().cpu())
                    prefix_values[global_index] = values
                    trusted[global_index] = max(
                        1, min(ACTION_HORIZON, int(scores[local].argmax().item()) + 1)
                    )
            else:
                for global_index in active_indices:
                    trusted[global_index] = ACTION_HORIZON
        return tuple(
            WaypointV2Prediction(
                decision=decision,
                normalized_action=actions[index],
                trusted_prefix_k=trusted[index],
                prefix_scores=prefix_values[index],
                boundary_probs=boundary_values[index],
                phase_progress=progress_values[index],
                time_to_boundary_s=time_values[index],
            )
            for index, decision in enumerate(decisions)
        )

    def _prediction_fm_action_features(
        self,
        decisions: Sequence[ConstrainedRouteDecision],
        selected_indices: Sequence[int],
        actions: Sequence[tuple[tuple[float, ...], ...] | None],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        values: list[torch.Tensor | None] = [None] * len(selected_indices)
        for domain, head in (
            (WaypointActionDomain.NAVIGATION, self.navigation_head),
            (WaypointActionDomain.MANIPULATION, self.manipulation_head),
        ):
            local_indices = [
                local
                for local, global_index in enumerate(selected_indices)
                if actions[global_index] is not None
                and decisions[global_index].action_domain is domain
            ]
            if not local_indices:
                continue
            parameter = next(head.parameters())
            action_tensor = torch.tensor(
                [actions[selected_indices[local]] for local in local_indices],
                device=parameter.device,
                dtype=parameter.dtype,
            )
            time = torch.zeros(
                len(local_indices), device=parameter.device, dtype=torch.long
            )
            with _action_autocast(parameter.device, parameter.dtype):
                encoded = head.action_encoder(action_tensor, time)
            for local, feature in zip(local_indices, encoded, strict=True):
                values[local] = feature.to(reference)
        zero = torch.zeros(
            (
                ACTION_HORIZON,
                self.auxiliary_heads.config.action_hidden_size,
            ),
            device=reference.device,
            dtype=reference.dtype,
        )
        return torch.stack([zero if value is None else value for value in values])


def _padded_actions(
    examples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    masks = []
    for index in indices:
        action = examples[index]["action"]
        rows.append([list(value) + [0.0] * (7 - len(value)) for value in action])
        masks.append([bool(value) for value in examples[index]["action_valid_mask"]])
    return (
        torch.tensor(rows, device=reference.device, dtype=reference.dtype),
        torch.tensor(masks, device=reference.device, dtype=reference.dtype),
    )


def _prefix_target_distribution(
    targets: torch.Tensor, transition: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    candidates = torch.arange(1, ACTION_HORIZON + 1, device=targets.device)[None]
    target = targets[:, None]
    utility = torch.where(
        candidates <= target,
        candidates.to(dtype) / target.clamp_min(1).to(dtype),
        -2.0 - (candidates - target).to(dtype),
    )
    uncertain = transition[:, None] & ((candidates - target).abs() <= 1)
    utility = torch.where(uncertain, torch.ones_like(utility), utility)
    return torch.softmax(utility.float() / 0.2, dim=-1).to(dtype)


def _jittered_boundary_signed_times(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[float | None, ...]:
    """Apply a stable 50 ms label jitter without perturbing FM RNG streams."""

    values = []
    for example in examples:
        signed = example.get("boundary_signed_time_s")
        if not bool(example["transition_window"]) or signed is None:
            values.append(None)
            continue
        identity = str(example.get("sample_id", example.get("transition_id", "")))
        digest = hashlib.sha256(
            f"waypoint-v2-boundary-jitter-v1:{identity}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        values.append(float(signed) + (unit - 0.5) * 0.1)
    return tuple(values)


def _soft_boundary_targets(
    examples: Sequence[Mapping[str, Any]],
    signed_targets: Sequence[float | None],
    reference: torch.Tensor,
) -> torch.Tensor:
    if len(examples) != len(signed_targets):
        raise ValueError("waypoint-v2 boundary targets are not aligned")
    rows = []
    for example, signed in zip(examples, signed_targets, strict=True):
        if not bool(example["transition_window"]) or signed is None:
            rows.append(reference.new_tensor((0.0, 1.0, 0.0)))
            continue
        after = torch.sigmoid(reference.new_tensor(float(signed) / 0.2))
        rows.append(torch.stack((1.0 - after, after * 0.0, after)))
    return torch.stack(rows)


def _boundary_pairwise_loss(
    scores: torch.Tensor,
    examples: Sequence[Mapping[str, Any]],
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    correct = []
    for left in range(len(examples)):
        left_id = examples[left].get("transition_id")
        left_time = examples[left].get("boundary_signed_time_s")
        if left_id is None or left_time is None:
            continue
        for right in range(left + 1, len(examples)):
            if examples[right].get("transition_id") != left_id:
                continue
            right_time = examples[right].get("boundary_signed_time_s")
            if right_time is None or float(right_time) == float(left_time):
                continue
            direction = 1.0 if float(right_time) > float(left_time) else -1.0
            difference = direction * (scores[right] - scores[left])
            losses.append(F.relu(reference.new_tensor(0.05) - difference))
            correct.append((difference > 0.0).to(reference.dtype))
    if not losses:
        zero = reference.sum() * 0.0
        return zero, zero.detach()
    return torch.stack(losses).mean(), torch.stack(correct).mean()


def _multi_positive_nce(logits: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    if logits.shape != positive.shape or logits.ndim != 2:
        raise ValueError("waypoint-v2 CRL logits and positives are not aligned")
    if not bool(positive.any(dim=-1).all()):
        raise ValueError("waypoint-v2 CRL row has no positive")
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    return torch.logsumexp(logits, dim=-1) - torch.logsumexp(
        positive_logits, dim=-1
    )


def _fixed_bank_seed(
    bank_seed: int,
    domain: str,
    sample_id: str,
    draw: int,
) -> int:
    payload = f"waypoint-v2-fixed-fm-v1\0{bank_seed}\0{domain}\0{sample_id}\0{draw}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (
        2**63 - 1
    )


__all__ = [
    "ConveyorVLAWaypointV2Policy",
    "WaypointV2AuxiliaryConfig",
    "WaypointV2AuxiliaryHeads",
    "WaypointV2LossConfig",
    "WaypointV2Prediction",
]
