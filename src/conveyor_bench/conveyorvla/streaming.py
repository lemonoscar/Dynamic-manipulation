"""Latest-observation streaming scheduler for ConveyorVLA action chunks."""

from __future__ import annotations

import math
import queue
from dataclasses import dataclass
from typing import Any, Sequence

from .temporal import ACTION_DIM, ACTION_HORIZON, CONTROL_HZ, MODEL_HZ


@dataclass(frozen=True)
class StreamChunk:
    episode_id: str
    generation_id: str
    observation_model_tick: int
    observation_control_tick: int
    actions: tuple[tuple[float, ...], ...]
    inference_started_s: float
    inference_finished_s: float

    def __post_init__(self) -> None:
        if not self.episode_id or not self.generation_id:
            raise ValueError("stream chunks require episode and generation IDs")
        for name in ("observation_model_tick", "observation_control_tick"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if len(self.actions) != ACTION_HORIZON:
            raise ValueError(f"stream chunk must contain {ACTION_HORIZON} actions")
        if any(
            len(row) != ACTION_DIM
            or any(not math.isfinite(float(component)) for component in row)
            for row in self.actions
        ):
            raise ValueError("stream chunk actions must be finite action10 rows")
        if (
            not math.isfinite(self.inference_started_s)
            or not math.isfinite(self.inference_finished_s)
            or self.inference_started_s < 0.0
            or self.inference_finished_s < self.inference_started_s
        ):
            raise ValueError("stream chunk inference timestamps are invalid")

    def scheduled_actions(self) -> tuple[tuple[int, tuple[float, ...]], ...]:
        stride = CONTROL_HZ // MODEL_HZ
        return tuple(
            (
                self.observation_control_tick + stride * (index + 1),
                tuple(float(component) for component in action),
            )
            for index, action in enumerate(self.actions)
        )


@dataclass(frozen=True)
class StreamMerge:
    accepted: bool
    reason: str
    skipped_actions: int
    remaining_actions: int
    first_target_control_tick: int | None
    last_target_control_tick: int | None


class ActionStreamBuffer:
    """Merge chunks by exact target tick and reject stale episode generations."""

    def __init__(self, *, min_remaining_actions: int = 2) -> None:
        if (
            isinstance(min_remaining_actions, bool)
            or not isinstance(min_remaining_actions, int)
            or not 1 <= min_remaining_actions <= ACTION_HORIZON
        ):
            raise ValueError("min_remaining_actions must be within [1, 20]")
        self.min_remaining_actions = min_remaining_actions
        self.episode_id: str | None = None
        self.generation_id: str | None = None
        self._last_observation_control_tick = -1
        self._waypoints: dict[int, tuple[float, ...]] = {}

    def reset(self, episode_id: str, generation_id: str) -> None:
        if not episode_id or not generation_id:
            raise ValueError("reset requires non-empty episode and generation IDs")
        self.episode_id = episode_id
        self.generation_id = generation_id
        self._last_observation_control_tick = -1
        self._waypoints.clear()

    def accept(self, chunk: StreamChunk, current_control_tick: int) -> StreamMerge:
        if (
            isinstance(current_control_tick, bool)
            or not isinstance(current_control_tick, int)
            or current_control_tick < 0
        ):
            raise ValueError("current_control_tick must be a non-negative integer")
        scheduled = chunk.scheduled_actions()
        future = tuple(item for item in scheduled if item[0] > current_control_tick)
        skipped = len(scheduled) - len(future)

        if (
            chunk.episode_id != self.episode_id
            or chunk.generation_id != self.generation_id
        ):
            return _merge(False, "generation_mismatch", skipped, future)
        if chunk.observation_control_tick <= self._last_observation_control_tick:
            return _merge(False, "out_of_order_observation", skipped, future)
        self._last_observation_control_tick = chunk.observation_control_tick
        if not future:
            self.prune(current_control_tick)
            return _merge(False, "fully_stale", skipped, future)
        if len(future) < self.min_remaining_actions:
            self.prune(current_control_tick)
            return _merge(False, "insufficient_future", skipped, future)

        first_new_tick = future[0][0]
        self._waypoints = {
            tick: action
            for tick, action in self._waypoints.items()
            if current_control_tick < tick < first_new_tick
        }
        self._waypoints.update(future)
        return _merge(True, "accepted", skipped, future)

    def prune(self, current_control_tick: int) -> None:
        self._waypoints = {
            tick: action
            for tick, action in self._waypoints.items()
            if tick > current_control_tick
        }

    def next_waypoint(
        self, current_control_tick: int
    ) -> tuple[int, tuple[float, ...]] | None:
        self.prune(current_control_tick)
        if not self._waypoints:
            return None
        tick = min(self._waypoints)
        return tick, self._waypoints[tick]

    def waypoints(self) -> tuple[tuple[int, tuple[float, ...]], ...]:
        return tuple(sorted(self._waypoints.items()))


def put_latest(target: Any, value: Any) -> None:
    """Replace a size-one queue entry so the newest result always wins."""

    while True:
        try:
            target.put_nowait(value)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                continue


def _merge(
    accepted: bool,
    reason: str,
    skipped: int,
    future: Sequence[tuple[int, tuple[float, ...]]],
) -> StreamMerge:
    return StreamMerge(
        accepted=accepted,
        reason=reason,
        skipped_actions=skipped,
        remaining_actions=len(future),
        first_target_control_tick=future[0][0] if future else None,
        last_target_control_tick=future[-1][0] if future else None,
    )


__all__ = ["ActionStreamBuffer", "StreamChunk", "StreamMerge", "put_latest"]
