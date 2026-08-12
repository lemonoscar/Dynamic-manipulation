"""Frozen configuration for the pure-Python ConveyorBench V1 protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

PROTOCOL_VERSION = "conveyor-bench-v1"


@dataclass(frozen=True)
class EvaluationConfig:
    """Thresholds for release, placement, and settled-dwell evaluation."""

    settled_linear_speed_mps: float = 0.02
    settled_angular_speed_radps: float = 0.10
    placement_dwell_s: float = 0.50
    require_settled_placement: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.require_settled_placement, bool):
            raise ValueError("require_settled_placement must be a bool")
        for name in (
            "settled_linear_speed_mps",
            "settled_angular_speed_radps",
            "placement_dwell_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.settled_linear_speed_mps < 0:
            raise ValueError("settled_linear_speed_mps cannot be negative")
        if self.settled_angular_speed_radps < 0:
            raise ValueError("settled_angular_speed_radps cannot be negative")
        if self.placement_dwell_s <= 0:
            raise ValueError("placement_dwell_s must be positive")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Timing and label contract shared by all V1 task instances."""

    protocol_version: str = PROTOCOL_VERSION
    physics_hz: int = 400
    control_hz: int = 50
    camera_hz: int = 25
    model_hz: int = 25
    history_offsets_steps: tuple[int, ...] = (-2, 0)
    m0_chunk_size: int = 16
    dynamicvla_chunk_size: int = 20
    label_offset_steps: int = 5
    future_horizons_steps: tuple[int, ...] = (0, 2, 5, 10, 20)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION!r}")
        if not isinstance(self.evaluation, EvaluationConfig):
            raise ValueError("evaluation must be an EvaluationConfig")
        for name in ("physics_hz", "control_hz", "camera_hz", "model_hz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if self.control_hz % self.model_hz:
            raise ValueError("control_hz must be divisible by model_hz")
        if self.control_hz % self.camera_hz:
            raise ValueError("control_hz must be divisible by camera_hz")

        if not self.history_offsets_steps:
            raise ValueError("history_offsets_steps cannot be empty")
        if any(
            isinstance(step, bool) or not isinstance(step, int)
            for step in self.history_offsets_steps
        ):
            raise ValueError("history_offsets_steps must contain integers")
        if tuple(sorted(set(self.history_offsets_steps))) != self.history_offsets_steps:
            raise ValueError("history_offsets_steps must be strictly increasing")
        if self.history_offsets_steps[-1] != 0:
            raise ValueError("history_offsets_steps must end at the current step (0)")
        if any(step > 0 for step in self.history_offsets_steps):
            raise ValueError("history_offsets_steps cannot contain future steps")

        for name in ("m0_chunk_size", "dynamicvla_chunk_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.label_offset_steps, bool)
            or not isinstance(self.label_offset_steps, int)
            or self.label_offset_steps < 0
        ):
            raise ValueError("label_offset_steps must be a non-negative integer")

        if not self.future_horizons_steps:
            raise ValueError("future_horizons_steps cannot be empty")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0
            for step in self.future_horizons_steps
        ):
            raise ValueError(
                "future_horizons_steps must contain non-negative integers"
            )
        if tuple(sorted(set(self.future_horizons_steps))) != self.future_horizons_steps:
            raise ValueError("future_horizons_steps must be strictly increasing")
        if self.future_horizons_steps[0] != 0:
            raise ValueError("future_horizons_steps must start at 0")

    @classmethod
    def v1(cls) -> "BenchmarkConfig":
        return cls()

    def chunk_size_for(self, profile: str) -> int:
        """Return the frozen action chunk size for a named V1 baseline."""

        if profile == "m0":
            return self.m0_chunk_size
        if profile == "dynamicvla":
            return self.dynamicvla_chunk_size
        raise ValueError(f"unsupported action chunk profile: {profile!r}")
