"""Pure-Python protocol core for ConveyorBench."""

from .config import BenchmarkConfig, EvaluationConfig
from .metrics import EpisodeEvaluation, evaluate_episode
from .protocol import (
    EpisodeManifest,
    EpisodeStatus,
    Event,
    EventKind,
    FailureReason,
    StepSample,
    TaskManifest,
    TaskType,
    TimingTrace,
)
from .recorder import EpisodeRecorder

__all__ = [
    "BenchmarkConfig",
    "EpisodeEvaluation",
    "EpisodeManifest",
    "EpisodeRecorder",
    "EpisodeStatus",
    "EvaluationConfig",
    "Event",
    "EventKind",
    "FailureReason",
    "StepSample",
    "TaskManifest",
    "TaskType",
    "TimingTrace",
    "evaluate_episode",
]
