"""Atomic episode writer for protocol manifests, streams, and summaries."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4

from .config import BenchmarkConfig
from .metrics import EpisodeEvaluation, evaluate_episode
from .protocol import (
    EpisodeManifest,
    EpisodeStatus,
    Event,
    EventKind,
    FailureReason,
    StepSample,
    to_jsonable,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(to_jsonable(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _close_and_publish(stream: TextIO, temporary: Path, final: Path) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    os.replace(temporary, final)


class EpisodeRecorder:
    """Record one episode and publish its directory with one atomic rename."""

    def __init__(
        self,
        output_root: str | Path,
        manifest: EpisodeManifest,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig.v0()
        self.manifest = manifest
        if manifest.protocol_version != self.config.protocol_version:
            raise ValueError(
                "manifest protocol_version does not match benchmark configuration"
            )

        episodes_root = Path(output_root) / "episodes"
        episodes_root.mkdir(parents=True, exist_ok=True)
        self.final_path = episodes_root / manifest.episode_id
        if self.final_path.exists():
            raise FileExistsError(f"episode already exists: {self.final_path}")

        self._staging_path = episodes_root / (
            f".{manifest.episode_id}.{uuid4().hex}.inprogress"
        )
        self._staging_path.mkdir()
        _atomic_write_json(
            self._staging_path / "manifest.json",
            {
                "episode": manifest,
                "benchmark_config": self.config,
            },
        )

        self._steps_temporary = self._staging_path / ".steps.jsonl.tmp"
        self._events_temporary = self._staging_path / ".events.jsonl.tmp"
        self._steps_stream = self._steps_temporary.open("x", encoding="utf-8")
        self._events_stream = self._events_temporary.open("x", encoding="utf-8")
        self._samples: list[StepSample] = []
        self._event_count = 0
        self._finalized = False
        self._last_sim_step: int | None = None
        self._last_sim_time_s: float | None = None

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._finalized:
            return False
        if exception_type is None:
            self.finalize()
        else:
            self.abort(
                FailureReason.ABORTED,
                {"exception_type": exception_type.__name__, "message": str(exception)},
            )
        return False

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def artifact_directory(self) -> Path:
        """Directory for episode-local artifacts before atomic publication."""

        self._ensure_open()
        return self._staging_path

    def record_step(self, sample: StepSample) -> None:
        self._ensure_open()
        if sample.env_id != self.manifest.env_id:
            raise ValueError("sample env_id does not match episode manifest")
        if self._last_sim_step is not None and sample.sim_step <= self._last_sim_step:
            raise ValueError("sim_step must increase strictly within an episode")
        if (
            self._last_sim_time_s is not None
            and sample.sim_time_s <= self._last_sim_time_s
        ):
            raise ValueError("sim_time_s must increase strictly within an episode")
        self._write_line(self._steps_stream, sample)
        self._samples.append(sample)
        self._last_sim_step = sample.sim_step
        self._last_sim_time_s = sample.sim_time_s

    def record_event(self, event: Event) -> None:
        self._ensure_open()
        self._write_line(self._events_stream, event)
        self._event_count += 1

    def finalize(
        self, evaluation: EpisodeEvaluation | None = None
    ) -> tuple[Path, EpisodeEvaluation]:
        self._ensure_open()
        result = evaluation or evaluate_episode(
            self.config, self.manifest.task, self._samples
        )
        self.record_event(
            Event(
                kind=EventKind.EPISODE_END,
                time_s=self._last_sim_time_s or 0.0,
                payload={
                    "success": result.success,
                    "failure_reason": result.failure_reason.value,
                },
            )
        )

        _close_and_publish(
            self._steps_stream,
            self._steps_temporary,
            self._staging_path / "steps.jsonl",
        )
        _close_and_publish(
            self._events_stream,
            self._events_temporary,
            self._staging_path / "events.jsonl",
        )
        summary = {
            "episode_id": self.manifest.episode_id,
            "task_id": self.manifest.task.task_id,
            "task_type": self.manifest.task.task_type,
            "status": (
                EpisodeStatus.SUCCESS if result.success else EpisodeStatus.FAILURE
            ),
            "success": result.success,
            "failure_reason": result.failure_reason,
            "sample_count": len(self._samples),
            "event_count": self._event_count,
            "completed_at_utc": _utc_now(),
            "metrics": result.metrics,
        }
        _atomic_write_json(self._staging_path / "summary.json", summary)
        os.replace(self._staging_path, self.final_path)
        self._finalized = True
        return self.final_path, result

    def abort(
        self,
        reason: FailureReason = FailureReason.ABORTED,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, EpisodeEvaluation]:
        if reason is FailureReason.NONE:
            raise ValueError("abort requires a failure reason")
        metrics = {
            "sample_count": len(self._samples),
            "abort_metadata": metadata or {},
        }
        return self.finalize(EpisodeEvaluation(False, reason, metrics))

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("episode recorder has already been finalized")

    @staticmethod
    def _write_line(stream: TextIO, value: Any) -> None:
        json.dump(to_jsonable(value), stream, separators=(",", ":"), sort_keys=True)
        stream.write("\n")
