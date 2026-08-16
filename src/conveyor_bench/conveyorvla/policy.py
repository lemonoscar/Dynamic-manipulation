# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""ConveyorVLA AL0 policy assembly for the Go2-X5 benchmark."""

from __future__ import annotations

import hashlib
import math
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from conveyor_bench.conveyorvla.dit import (
    GO2_X5_REINITIALIZED_ACTION_KEYS,
    M0DiTActionHead,
    M0DiTConfig,
    copy_parameter_tensors,
    parameter_state_shapes,
)
from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.subtasks import (
    ActionDomain,
    Phase,
    SUBTASK_END_TOKEN,
    SUBTASK_SPECIAL_TOKENS,
    SubtaskDecision,
    parse_subtask_solution,
)


@dataclass(frozen=True)
class PolicyCheckpointReport:
    loaded_qwen_tensors: int
    loaded_action_tensors: int
    reinitialized_keys: tuple[str, ...]


@dataclass(frozen=True)
class QwenCheckpointReport:
    loaded_tensors: int


@dataclass(frozen=True)
class RoutedActionPrediction:
    decision: SubtaskDecision
    normalized_actions: tuple[tuple[float, ...], ...]


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
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(SUBTASK_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )
        if len(processor.tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(processor.tokenizer), mean_resizing=False)
        token_ids = [
            processor.tokenizer.convert_tokens_to_ids(token)
            for token in SUBTASK_SPECIAL_TOKENS
        ]
        if len(set(token_ids)) != len(token_ids) or any(
            token_id is None or token_id < 0 for token_id in token_ids
        ):
            raise M0MobileError("Qwen tokenizer did not register unique subtask tokens")
        interface = cls(model, processor)
        interface.registered_subtask_tokens = added
        return interface

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
        *,
        history_span_s: float,
        solutions: Sequence[str] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """Build two ordered clips and optionally supervise only the assistant span."""

        if len(videos) != len(instructions) or not videos:
            raise ValueError("videos and instructions must be non-empty equal batches")
        if not math.isfinite(history_span_s) or history_span_s <= 0.0:
            raise ValueError("temporal history span must be positive and finite")
        if solutions is not None and len(solutions) != len(videos):
            raise ValueError("solutions and videos must have equal batch sizes")
        messages = []
        for index, (sample_clips, instruction) in enumerate(
            zip(videos, instructions, strict=True)
        ):
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
            sample_messages = [{"role": "user", "content": content}]
            if solutions is not None:
                sample_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": _instruction(solutions[index])}
                        ],
                    }
                )
            messages.append(sample_messages)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=solutions is None,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            video_metadata=[
                [
                    {
                        "total_num_frames": 2,
                        "fps": 1.0 / history_span_s,
                        "duration": 2.0 * history_span_s,
                        "frames_indices": [0, 1],
                    }
                    for _clip in sample_clips
                ]
                for sample_clips in videos
            ],
        )
        if solutions is not None:
            inputs["labels"] = self._assistant_labels(inputs["input_ids"])
        device = next(self.model.parameters()).device
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _assistant_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask every token except the final assistant answer in each sequence."""

        tokenizer = self.processor.tokenizer
        marker = torch.as_tensor(
            tokenizer(
                "<|im_start|>assistant\n",
                add_special_tokens=False,
            ).input_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        if marker.numel() == 0:
            raise RuntimeError("Qwen assistant marker tokenization is empty")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        labels = torch.full_like(input_ids, -100)
        for row_index, row in enumerate(input_ids):
            starts = [
                index + marker.numel()
                for index in range(row.numel() - marker.numel() + 1)
                if torch.equal(row[index : index + marker.numel()], marker)
            ]
            if not starts:
                raise RuntimeError("Qwen input is missing the assistant answer marker")
            start = starts[-1]
            end = start
            while end < row.numel() and int(row[end]) != im_end_id:
                end += 1
            if end <= start:
                raise RuntimeError("Qwen assistant answer span is empty")
            labels[row_index, start:end] = row[start:end]
        return labels

    def enable_full_finetuning(self) -> None:
        """Make the entire visual-language backbone trainable with checkpointing."""

        self.requires_grad_(True)
        self.model.config.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    @torch.inference_mode()
    def generate_temporal_subtask_texts(
        self,
        videos: Sequence[Sequence[Sequence[Any]]],
        instructions: Sequence[str],
        *,
        history_span_s: float,
        max_new_tokens: int = 48,
    ) -> tuple[str, ...]:
        """Run Pass 1 with greedy decoding and return the unparsed answers."""

        inputs = dict(
            self.build_temporal_inputs(
                videos,
                instructions,
                history_span_s=history_span_s,
            )
        )
        prompt_width = int(inputs["input_ids"].shape[1])
        end_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            SUBTASK_END_TOKEN
        )
        if end_token_id is None or end_token_id < 0:
            raise RuntimeError("Qwen tokenizer is missing the subtask end token")
        was_training = self.model.training
        self.model.eval()
        try:
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=end_token_id,
                synced_gpus=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )
        finally:
            self.model.train(was_training)
        return tuple(
            self.processor.tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=False,
            )
        )

    @torch.inference_mode()
    def generate_temporal_subtasks(
        self,
        videos: Sequence[Sequence[Sequence[Any]]],
        instructions: Sequence[str],
        *,
        history_span_s: float,
        max_new_tokens: int = 48,
    ) -> tuple[SubtaskDecision, ...]:
        """Run Pass 1 with greedy decoding and fail closed on non-canonical text."""

        answers = self.generate_temporal_subtask_texts(
            videos,
            instructions,
            history_span_s=history_span_s,
            max_new_tokens=max_new_tokens,
        )
        return tuple(parse_subtask_solution(answer) for answer in answers)

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
        action_valid_mask = torch.as_tensor(
            [
                example.get("action_valid_mask", [True] * actions.shape[1])
                for example in examples
            ],
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
                action_valid_mask=action_valid_mask.repeat(repeats, 1),
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

    def __init__(
        self,
        *args: Any,
        temporal_history_span_s: float = 0.20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            not math.isfinite(temporal_history_span_s)
            or temporal_history_span_s <= 0.0
        ):
            raise ValueError("temporal history span must be positive and finite")
        self.temporal_history_span_s = float(temporal_history_span_s)

    def _encode(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not examples:
            raise ValueError("examples must be non-empty")
        inputs = self.qwen_vl_interface.build_temporal_inputs(
            [example["video"] for example in examples],
            [_instruction(example["lang"]) for example in examples],
            history_span_s=self.temporal_history_span_s,
        )
        return self._encode_inputs(inputs)


class ConveyorVLAAL0TwoPassPolicy(nn.Module):
    """One fully trainable Qwen with language routing into two unchanged DiTs."""

    def __init__(
        self,
        qwen_vl_interface: Qwen3VLInterface,
        navigation_model: M0DiTActionHead,
        manipulation_model: M0DiTActionHead,
        *,
        temporal_history_span_s: float,
        repeated_diffusion_steps: int = 1,
    ) -> None:
        super().__init__()
        if not math.isfinite(temporal_history_span_s) or temporal_history_span_s <= 0:
            raise ValueError("temporal history span must be positive and finite")
        if repeated_diffusion_steps <= 0:
            raise ValueError("repeated diffusion steps must be positive")
        self.qwen_vl_interface = qwen_vl_interface
        self.navigation_model = navigation_model
        self.manipulation_model = manipulation_model
        self.temporal_history_span_s = float(temporal_history_span_s)
        self.repeated_diffusion_steps = int(repeated_diffusion_steps)

    def enable_full_finetuning(self) -> None:
        """Enable gradients for the complete VLM and both action experts."""

        self.qwen_vl_interface.enable_full_finetuning()
        self.navigation_model.requires_grad_(True)
        self.manipulation_model.requires_grad_(True)

    def forward(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        objective: str,
        teacher_forcing_probability: float = 1.0,
        routing_seed: int = 0,
    ) -> Mapping[str, torch.Tensor | int]:
        """Expose separate passes so their activation graphs need not coexist."""

        if objective == "subtask":
            return self._subtask_loss(examples)
        if objective == "action":
            return self._action_loss(
                examples,
                teacher_forcing_probability=teacher_forcing_probability,
                routing_seed=routing_seed,
            )
        raise ValueError("objective must be subtask or action")

    def _subtask_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, torch.Tensor | int]:
        inputs = self._temporal_inputs(examples, include_solutions=True)
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Qwen did not return supervised subtask loss")
        supervised_tokens = int((inputs["labels"] != -100).sum().item())
        return {"subtask_loss": loss, "supervised_tokens": supervised_tokens}

    def _action_loss(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        teacher_forcing_probability: float,
        routing_seed: int,
    ) -> Mapping[str, torch.Tensor | int]:
        if not examples:
            raise ValueError("examples must be non-empty")
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher forcing probability must be within [0, 1]")

        generated = (
            self.qwen_vl_interface.generate_temporal_subtask_texts(
                [example["video"] for example in examples],
                [_instruction(example["lang"]) for example in examples],
                history_span_s=self.temporal_history_span_s,
            )
            if teacher_forcing_probability < 1.0
            else ("",) * len(examples)
        )
        routed_indices: list[int] = []
        second_pass_solutions: list[str] = []
        teacher_forced = 0
        predicted_correct = 0
        predicted_wrong = 0
        predicted_invalid = 0
        for index, example in enumerate(examples):
            use_teacher = _teacher_forcing_choice(
                example,
                teacher_forcing_probability,
                routing_seed,
            )
            expected = Phase(int(example["phase_id"]))
            if use_teacher:
                decision = parse_subtask_solution(_instruction(example["solution"]))
                teacher_forced += 1
                second_pass_solutions.append(decision.assistant_solution)
            else:
                try:
                    decision = parse_subtask_solution(generated[index])
                except ValueError:
                    predicted_invalid += 1
                    second_pass_solutions.append(
                        generated[index].strip()
                        or "<|pred_action|><|subtask|>INVALID<|end_subtask|>"
                    )
                    continue
                second_pass_solutions.append(decision.assistant_solution)
                if decision.phase is not expected:
                    predicted_wrong += 1
                    continue
                predicted_correct += 1
            if decision.phase is not expected:
                raise RuntimeError("teacher-forced route disagrees with the label")
            routed_indices.append(index)

        result: dict[str, torch.Tensor | int] = {
            "teacher_forced_samples": teacher_forced,
            "predicted_route_correct": predicted_correct,
            "predicted_route_wrong": predicted_wrong,
            "predicted_route_invalid": predicted_invalid,
        }
        inputs = dict(
            self._temporal_inputs(
                examples,
                include_solutions=True,
                solutions_override=second_pass_solutions,
            )
        )
        inputs.pop("labels")
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        hidden = outputs.hidden_states[-1]
        losses: list[tuple[int, torch.Tensor]] = []
        for domain, name, model in (
            (ActionDomain.NAVIGATION, "navigation", self.navigation_model),
            (ActionDomain.MANIPULATION, "manipulation", self.manipulation_model),
        ):
            indices = [
                index
                for index in routed_indices
                for example in (examples[index],)
                if int(example["action_domain_id"]) == int(domain)
                and any(bool(value) for value in example["action_valid_mask"])
            ]
            result[f"{name}_samples"] = len(indices)
            selected_indices = indices or [0]
            index_tensor = torch.as_tensor(selected_indices, device=hidden.device)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            selected_hidden = hidden.index_select(0, index_tensor).to(
                device=device, dtype=dtype
            )
            selected_attention = attention_mask.index_select(0, index_tensor).to(device)
            actions = torch.as_tensor(
                (
                    [examples[index]["action"] for index in selected_indices]
                    if indices
                    else [
                        [
                            [0.0] * model.config.action_dim
                            for _step in range(model.config.action_horizon)
                        ]
                    ]
                ),
                device=device,
                dtype=dtype,
            )
            state = torch.as_tensor(
                [examples[index]["state"] for index in selected_indices],
                device=device,
                dtype=dtype,
            )
            action_mask = torch.as_tensor(
                (
                    [examples[index]["action_mask"] for index in selected_indices]
                    if indices
                    else [[True] * model.config.action_dim]
                ),
                device=device,
                dtype=torch.bool,
            )
            action_valid_mask = torch.as_tensor(
                (
                    [
                        examples[index]["action_valid_mask"]
                        for index in selected_indices
                    ]
                    if indices
                    else [[True] * model.config.action_horizon]
                ),
                device=device,
                dtype=torch.bool,
            )
            repeats = self.repeated_diffusion_steps
            with _action_autocast(device, dtype):
                loss = model(
                    selected_hidden.repeat(repeats, 1, 1),
                    actions.repeat(repeats, 1, 1),
                    state.repeat(repeats, 1, 1),
                    encoder_attention_mask=selected_attention.repeat(repeats, 1),
                    action_dimension_mask=action_mask.repeat(repeats, 1),
                    action_valid_mask=action_valid_mask.repeat(repeats, 1),
                )
            if not indices:
                loss = loss * 0.0
            losses.append((len(indices), loss))
            result[f"{name}_loss"] = loss
        sample_count = sum(count for count, _loss in losses)
        if sample_count == 0:
            result["action_loss"] = sum(loss for _count, loss in losses)
        else:
            result["action_loss"] = (
                sum(count * loss for count, loss in losses) / sample_count
            )
        return result

    def _temporal_inputs(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        include_solutions: bool,
        solutions_override: Sequence[str] | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if not examples:
            raise ValueError("examples must be non-empty")
        if solutions_override is not None and len(solutions_override) != len(examples):
            raise ValueError("solution override and examples must have equal lengths")
        solutions = None
        if include_solutions:
            solutions = [
                _instruction(value)
                for value in (
                    solutions_override
                    if solutions_override is not None
                    else [example["solution"] for example in examples]
                )
            ]
        return self.qwen_vl_interface.build_temporal_inputs(
            [example["video"] for example in examples],
            [_instruction(example["lang"]) for example in examples],
            history_span_s=self.temporal_history_span_s,
            solutions=solutions,
        )

    @torch.inference_mode()
    def predict_subtasks(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[SubtaskDecision, ...]:
        """Run the actual first pass used by the online dispatcher."""

        return self.qwen_vl_interface.generate_temporal_subtasks(
            [example["video"] for example in examples],
            [_instruction(example["lang"]) for example in examples],
            history_span_s=self.temporal_history_span_s,
        )

    @torch.inference_mode()
    def predict_routed_actions(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[RoutedActionPrediction, ...]:
        """Generate the subtask, run a second Qwen forward, then map to one DiT."""

        decisions = self.predict_subtasks(examples)
        inputs = dict(
            self._temporal_inputs(
                examples,
                include_solutions=True,
                solutions_override=[item.assistant_solution for item in decisions],
            )
        )
        inputs.pop("labels")
        outputs = self.qwen_vl_interface(
            **inputs,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise RuntimeError("Qwen processor did not return an attention mask")
        hidden = outputs.hidden_states[-1]
        actions: list[tuple[tuple[float, ...], ...] | None] = [None] * len(examples)
        for domain, model in (
            (ActionDomain.NAVIGATION, self.navigation_model),
            (ActionDomain.MANIPULATION, self.manipulation_model),
        ):
            indices = [
                index
                for index, decision in enumerate(decisions)
                if decision.domain is domain
            ]
            if not indices:
                continue
            index_tensor = torch.as_tensor(indices, device=hidden.device)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype
            state = torch.as_tensor(
                [examples[index]["state"] for index in indices],
                device=device,
                dtype=dtype,
            )
            with _action_autocast(device, dtype):
                sampled = model.sample(
                    hidden.index_select(0, index_tensor).to(device=device, dtype=dtype),
                    state,
                    encoder_attention_mask=attention_mask.index_select(
                        0, index_tensor
                    ).to(device),
                    action_dimension_mask=torch.ones(
                        (len(indices), model.config.action_dim),
                        device=device,
                        dtype=torch.bool,
                    ),
                )
            for index, value in zip(indices, sampled.float().cpu().tolist(), strict=True):
                actions[index] = tuple(tuple(float(item) for item in row) for row in value)
        if any(value is None for value in actions):
            raise RuntimeError("dispatcher failed to assign one expert per prediction")
        return tuple(
            RoutedActionPrediction(decision, action)  # type: ignore[arg-type]
            for decision, action in zip(decisions, actions, strict=True)
        )


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


def transfer_qwen_checkpoint_weights(
    interface: Qwen3VLInterface,
    checkpoint: str | Path | Mapping[str, torch.Tensor],
) -> QwenCheckpointReport:
    """Load only the released Qwen branch and reject partial or reshaped tensors."""

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
        raise RuntimeError("upstream checkpoint must contain a tensor mapping")
    prefix = "qwen_vl_interface."
    qwen_source = {
        key.removeprefix(prefix): value
        for key, value in source.items()
        if key.startswith(prefix)
    }
    target_shapes = parameter_state_shapes(interface)
    unexpected = set(qwen_source) - set(target_shapes)
    missing = set(target_shapes) - set(qwen_source)
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen checkpoint structure mismatch: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    bad_shapes = [
        key
        for key, value in qwen_source.items()
        if not isinstance(value, torch.Tensor) or value.shape != target_shapes[key]
    ]
    if bad_shapes:
        raise RuntimeError(
            "Qwen checkpoint tensor shapes do not match: "
            + ", ".join(sorted(bad_shapes))
        )
    copy_parameter_tensors(interface, qwen_source)
    return QwenCheckpointReport(loaded_tensors=len(qwen_source))


def _rgb_image(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
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


def _teacher_forcing_choice(
    example: Mapping[str, Any],
    probability: float,
    seed: int,
) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    identity = example.get("sample_id", example.get("base_index", example.get("phase_id")))
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return score < probability


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
    "ConveyorVLAAL0TwoPassPolicy",
    "M0MobilePolicy",
    "PolicyCheckpointReport",
    "QwenCheckpointReport",
    "Qwen3VLInterface",
    "RoutedActionPrediction",
    "m0_dit_config",
    "transfer_robocasa_policy_weights",
    "transfer_qwen_checkpoint_weights",
]
