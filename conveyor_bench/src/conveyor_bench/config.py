"""Versioned benchmark constants."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationConfig:
    """C0/C1 success thresholds frozen for protocol V0."""

    lift_height_m: float = 0.05
    hold_time_s: float = 1.0
    static_belt_tolerance_mps: float = 0.01
    dynamic_belt_min_speed_mps: float = 0.02

    def __post_init__(self) -> None:
        if self.lift_height_m <= 0:
            raise ValueError("lift_height_m must be positive")
        if self.hold_time_s <= 0:
            raise ValueError("hold_time_s must be positive")
        if self.static_belt_tolerance_mps < 0:
            raise ValueError("static_belt_tolerance_mps cannot be negative")
        if self.dynamic_belt_min_speed_mps <= self.static_belt_tolerance_mps:
            raise ValueError(
                "dynamic_belt_min_speed_mps must exceed static_belt_tolerance_mps"
            )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level V0 protocol configuration."""

    protocol_version: str = "conveyor-bench-v0"
    physics_hz: int = 200
    control_hz: int = 50
    camera_hz: int = 25
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        for name in ("physics_hz", "control_hz", "camera_hz"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if self.control_hz % self.camera_hz:
            raise ValueError("control_hz must be divisible by camera_hz")

    @classmethod
    def v0(cls) -> "BenchmarkConfig":
        return cls()
