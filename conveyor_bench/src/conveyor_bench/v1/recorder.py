"""Streaming, atomic episode recorder for ConveyorBench V1."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO
from uuid import uuid4

from .config import BenchmarkConfig
from .metrics import EpisodeEvaluation, OnlineEpisodeMetrics
from .protocol import (
    ActionChunkTrace,
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
        json.dump(
            to_jsonable(value),
            stream,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
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
    """Write normalized JSONL streams and atomically publish one episode."""

    _STREAM_NAMES = ("steps", "objects", "action_chunks", "events")

    def __init__(
        self,
        output_root: str | Path,
        manifest: EpisodeManifest,
        config: BenchmarkConfig | None = None,
    ) -> None:
        self.config = config or BenchmarkConfig.v1()
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
            {"episode": manifest, "benchmark_config": self.config},
        )

        self._temporary_paths: dict[str, Path] = {}
        self._streams: dict[str, TextIO] = {}
        for name in self._STREAM_NAMES:
            temporary = self._staging_path / f".{name}.jsonl.tmp"
            self._temporary_paths[name] = temporary
            self._streams[name] = temporary.open("x", encoding="utf-8")

        self._metrics = OnlineEpisodeMetrics(self.config, self.manifest.task)
        self._seen_action_chunk_ids: set[str] = set()
        self._active_action_chunks: dict[str, ActionChunkTrace] = {}
        self._step_count = 0
        self._object_record_count = 0
        self._action_chunk_count = 0
        self._event_count = 0
        self._last_sim_step: int | None = None
        self._last_sim_time_s: float | None = None
        self._last_model_tick: int | None = None
        self._last_event_time_s: float | None = None
        self._finalized = False

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
        self._ensure_open()
        return self._staging_path

    @property
    def online_metrics(self) -> OnlineEpisodeMetrics:
        return self._metrics

    def record_action_chunk(self, trace: ActionChunkTrace) -> None:
        self._ensure_open()
        trace.validate_against(self.config)
        for action in trace.actions:
            action.validate_for_robot_mode(self.manifest.task.robot_mode)
        if trace.chunk_id in self._seen_action_chunk_ids:
            raise ValueError(f"duplicate action chunk id: {trace.chunk_id!r}")
        if (
            self._last_model_tick is not None
            and trace.source_observation_tick > self._last_model_tick
        ):
            raise ValueError("action chunk source observation has not been recorded")
        self._write_line(self._streams["action_chunks"], trace)
        self._seen_action_chunk_ids.add(trace.chunk_id)
        if trace.execute_from_tick is not None:
            self._active_action_chunks[trace.chunk_id] = trace
        self._action_chunk_count += 1

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
        if (
            self._last_model_tick is not None
            and sample.model_tick < self._last_model_tick
        ):
            raise ValueError("model_tick cannot decrease within an episode")
        sample.validate_against(self.manifest.task, self.config)
        self._evict_finished_action_chunks(sample.model_tick)
        self._validate_action_reference(sample)

        step_record = {
            field.name: getattr(sample, field.name)
            for field in fields(sample)
            if field.name not in {"objects", "future_object_states"}
        }
        self._write_line(self._streams["steps"], step_record)

        labels_by_object = {
            obj.instance_id: tuple(
                label
                for label in sample.future_object_states
                if label.instance_id == obj.instance_id
            )
            for obj in sample.objects
        }
        for obj in sample.objects:
            self._write_line(
                self._streams["objects"],
                {
                    "sim_step": sample.sim_step,
                    "sim_time_s": sample.sim_time_s,
                    "model_tick": sample.model_tick,
                    "env_id": sample.env_id,
                    "state": obj,
                    "future_object_states": labels_by_object[obj.instance_id],
                },
            )
            self._object_record_count += 1

        self._metrics.update(sample)
        self._step_count += 1
        self._last_sim_step = sample.sim_step
        self._last_sim_time_s = sample.sim_time_s
        self._last_model_tick = sample.model_tick

    def record_event(self, event: Event) -> None:
        self._ensure_open()
        event.validate_against(self.manifest.task)
        if (
            self._last_event_time_s is not None
            and event.time_s < self._last_event_time_s
        ):
            raise ValueError("event time_s cannot decrease")
        self._write_line(self._streams["events"], event)
        self._event_count += 1
        self._last_event_time_s = event.time_s

    def finalize(
        self, evaluation: EpisodeEvaluation | None = None
    ) -> tuple[Path, EpisodeEvaluation]:
        self._ensure_open()
        result = evaluation or self._metrics.finalize()
        self.record_event(
            Event(
                kind=EventKind.EPISODE_END,
                time_s=max(
                    self._last_sim_time_s or 0.0,
                    self._last_event_time_s or 0.0,
                ),
                sim_step=self._last_sim_step,
                payload={
                    "success": result.success,
                    "failure_reason": result.failure_reason.value,
                },
            )
        )

        for name in self._STREAM_NAMES:
            _close_and_publish(
                self._streams[name],
                self._temporary_paths[name],
                self._staging_path / f"{name}.jsonl",
            )
        summary = {
            "episode_id": self.manifest.episode_id,
            "task_id": self.manifest.task.task_id,
            "task_type": self.manifest.task.task_type,
            "robot_mode": self.manifest.task.robot_mode,
            "status": (
                EpisodeStatus.SUCCESS if result.success else EpisodeStatus.FAILURE
            ),
            "success": result.success,
            "failure_reason": result.failure_reason,
            "sample_count": self._step_count,
            "object_record_count": self._object_record_count,
            "action_chunk_count": self._action_chunk_count,
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
        result = self._metrics.failure_evaluation(reason, metadata)
        return self.finalize(result)

    def _validate_action_reference(self, sample: StepSample) -> None:
        if sample.action_chunk_id is None:
            return
        trace = self._active_action_chunks.get(sample.action_chunk_id)
        if trace is None:
            raise ValueError("sample references an unknown action chunk")
        assert sample.action_index_in_chunk is not None
        if sample.action_index_in_chunk >= len(trace.actions):
            raise ValueError("action_index_in_chunk exceeds the referenced chunk")
        if sample.action != trace.actions[sample.action_index_in_chunk]:
            raise ValueError("sample action does not match the referenced chunk action")
        expected_tick = trace.valid_from_tick + sample.action_index_in_chunk
        if sample.model_tick != expected_tick:
            raise ValueError("sample model_tick does not match its action chunk index")
        if trace.execute_from_tick is None or trace.execute_until_tick is None or not (
            trace.execute_from_tick <= sample.model_tick < trace.execute_until_tick
        ):
            raise ValueError("sample references an action outside the execute window")

    def _evict_finished_action_chunks(self, model_tick: int) -> None:
        finished = tuple(
            chunk_id
            for chunk_id, trace in self._active_action_chunks.items()
            if trace.execute_until_tick is not None
            and trace.execute_until_tick <= model_tick
        )
        for chunk_id in finished:
            del self._active_action_chunks[chunk_id]

    def _ensure_open(self) -> None:
        if self._finalized:
            raise RuntimeError("episode recorder has already been finalized")

    @staticmethod
    def _write_line(stream: TextIO, value: Any) -> None:
        json.dump(
            to_jsonable(value),
            stream,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        stream.write("\n")
