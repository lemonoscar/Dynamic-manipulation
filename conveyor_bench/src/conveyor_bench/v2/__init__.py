"""Pure-Python ConveyorBench V2 task coordination core."""

from .coordinator import (
    CoordinatorStatus,
    CoordinatorTerminatedError,
    SequentialTargetCoordinator,
    TargetTransition,
)
from .config import (
    BENCHMARK_SUITE_VERSION,
    CANONICAL_PROTOCOL_VERSION,
    DEFAULT_SUITE_CONFIG,
    TASK_CONTEXT_SCHEMA_VERSION,
    SceneId,
    V2SuiteConfig,
)
from .tasking import (
    ServiceGate,
    ServiceGateKind,
    V2TaskContext,
    build_task_context,
    build_task_manifest,
    validate_task_combination,
)

__all__ = [
    "BENCHMARK_SUITE_VERSION",
    "CANONICAL_PROTOCOL_VERSION",
    "CoordinatorStatus",
    "CoordinatorTerminatedError",
    "DEFAULT_SUITE_CONFIG",
    "TASK_CONTEXT_SCHEMA_VERSION",
    "SceneId",
    "SequentialTargetCoordinator",
    "ServiceGate",
    "ServiceGateKind",
    "TargetTransition",
    "V2SuiteConfig",
    "V2TaskContext",
    "build_task_context",
    "build_task_manifest",
    "validate_task_combination",
]
