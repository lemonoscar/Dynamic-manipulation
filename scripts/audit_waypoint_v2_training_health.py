#!/usr/bin/env python3
"""Audit a live Waypoint-v2 run for consecutive four-H20 training health."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "conveyorvla-waypoint-v2-training-health-audit-v1"
RUN_SCHEMA = "conveyorvla-waypoint-training-run-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--resolved-run", required=True, type=Path)
    parser.add_argument("--run-state", required=True, type=Path)
    parser.add_argument("--minimum-consecutive-steps", type=int, default=20)
    parser.add_argument("--gpu-memory-total-mib", type=float, default=97_871.0)
    parser.add_argument("--formal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_training_health(
        _read_events(args.events),
        _read_json(args.resolved_run),
        _read_json(args.run_state),
        events_path=args.events.expanduser().resolve(),
        minimum=args.minimum_consecutive_steps,
        gpu_memory_total_mib=args.gpu_memory_total_mib,
        formal=args.formal,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def audit_training_health(
    events: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    events_path: Path,
    minimum: int,
    gpu_memory_total_mib: float,
    formal: bool,
) -> dict[str, Any]:
    problems = []
    if minimum <= 0:
        problems.append("minimum consecutive steps must be positive")
    if not math.isfinite(gpu_memory_total_mib) or gpu_memory_total_mib <= 0.0:
        problems.append("GPU memory capacity must be finite and positive")
    if resolved.get("schema_version") != RUN_SCHEMA:
        problems.append("resolved run is not Waypoint-v2")
    if any(event.get("event") == "failed" for event in events):
        problems.append("run contains a failed event")

    train = [event for event in events if event.get("event") == "train_step"]
    window = train[-minimum:] if minimum > 0 else []
    steps = [int(event.get("step", -1)) for event in window]
    if len(window) != minimum:
        problems.append("not enough train_step events")
    elif steps != list(range(steps[0], steps[0] + minimum)):
        problems.append("train_step events are not consecutive")

    auxiliary_enabled = _auxiliary_enabled(resolved)
    scalar_fields = [
        "loss",
        "answer_loss",
        "route_loss",
        "navigation_loss",
        "manipulation_loss",
        "gradient_norm",
        "vlm_gradient_norm",
        "navigation_gradient_norm",
        "manipulation_gradient_norm",
        "optimizer_step_time_s",
        "samples_per_second",
        "gpu_hours_per_step",
        "peak_allocated_memory_mib",
        "peak_reserved_memory_mib",
    ]
    if auxiliary_enabled:
        scalar_fields.append("auxiliary_gradient_norm")
    positive_fields = {
        "gradient_norm",
        "vlm_gradient_norm",
        "navigation_gradient_norm",
        "manipulation_gradient_norm",
        "optimizer_step_time_s",
        "samples_per_second",
        "gpu_hours_per_step",
    }
    if auxiliary_enabled:
        positive_fields.add("auxiliary_gradient_norm")
    for event in window:
        step = event.get("step")
        if event.get("valid_optimizer_step") is not True:
            problems.append(f"step {step} is not a valid optimizer step")
        for field in scalar_fields:
            value = event.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"step {step} has invalid {field}")
            elif field in positive_fields and float(value) <= 0.0:
                problems.append(f"step {step} has non-positive {field}")
        learning_rates = event.get("learning_rates")
        if (
            not isinstance(learning_rates, list)
            or not learning_rates
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in learning_rates
            )
        ):
            problems.append(f"step {step} has invalid learning rates")

    peak_reserved = max(
        (float(event.get("peak_reserved_memory_mib", math.inf)) for event in window),
        default=math.inf,
    )
    if math.isfinite(gpu_memory_total_mib) and peak_reserved >= 0.95 * gpu_memory_total_mib:
        problems.append("peak reserved memory leaves less than 5% H20 headroom")
    timing = [
        float(event["optimizer_step_time_s"])
        for event in window
        if isinstance(event.get("optimizer_step_time_s"), (int, float))
        and math.isfinite(float(event["optimizer_step_time_s"]))
    ]
    throughput = [
        float(event["samples_per_second"])
        for event in window
        if isinstance(event.get("samples_per_second"), (int, float))
        and math.isfinite(float(event["samples_per_second"]))
    ]
    if _persistent_collapse(throughput):
        problems.append("samples/s has a persistent three-step collapse")

    arguments = _mapping(resolved.get("arguments"))
    output_dir = Path(str(arguments.get("output_dir", ""))).expanduser()
    if output_dir and events_path.parent != output_dir.resolve():
        problems.append("events path does not match resolved output directory")
    world_size = int(resolved.get("world_size", 0))
    gpu_uuids = _gpu_uuids(resolved.get("visible_gpu_uuids"))
    if formal:
        if world_size != 4:
            problems.append("formal run does not use world_size=4")
        if len(gpu_uuids) != 4:
            problems.append("formal run does not bind four unique GPU UUIDs")
        if int(arguments.get("save_interval_steps", 0)) != 500:
            problems.append("formal checkpoint interval is not 500 optimizer steps")
        if int(arguments.get("save_first_checkpoint_step", 0)) != 500:
            problems.append("formal first checkpoint is not step 500")
        if int(resolved.get("max_steps", 0)) <= minimum:
            problems.append("formal max_steps is only the health-check horizon")
        if resolved.get("training_subset") is not False:
            problems.append("formal run unexpectedly uses an episode/row subset")
    state_step = int(state.get("global_step", -1))
    if state.get("status") != "running":
        problems.append("run state is not live/running")
    if steps and state_step < steps[-1]:
        problems.append("run_state global_step lags the health window")

    report = {
        "schema_version": REPORT_SCHEMA,
        "ok": not problems,
        "formal": formal,
        "events": str(events_path),
        "observed_steps": steps,
        "world_size": world_size,
        "gpu_uuids": gpu_uuids,
        "run_state": {"status": state.get("status"), "global_step": state_step},
        "metrics": {
            "loss": _series_summary(window, "loss"),
            "gradient_norm": _series_summary(window, "gradient_norm"),
            "optimizer_step_time_s": _summary(timing),
            "samples_per_second": _summary(throughput),
            "peak_reserved_memory_mib": peak_reserved,
            "peak_memory_fraction": peak_reserved / gpu_memory_total_mib,
            "last_learning_rates": (
                window[-1].get("learning_rates") if window else None
            ),
        },
        "problems": problems,
    }
    return report


def _auxiliary_enabled(resolved: Mapping[str, Any]) -> bool:
    config = _mapping(resolved.get("resolved_policy_config"))
    auxiliary = _mapping(config.get("auxiliary"))
    return any(
        bool(auxiliary.get(field))
        for field in ("enable_boundary_progress", "enable_prefix", "enable_crl")
    )


def _persistent_collapse(values: Sequence[float]) -> bool:
    if len(values) < 6:
        return False
    baseline = statistics.median(values)
    if baseline <= 0.0:
        return True
    collapsed = [value < baseline / 3.0 for value in values]
    return any(all(collapsed[index : index + 3]) for index in range(len(values) - 2))


def _series_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
    ]
    return _summary(values)


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "cv": None}
    mean = statistics.fmean(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "cv": (
            None
            if len(values) < 2 or mean == 0.0
            else statistics.pstdev(values) / abs(mean)
        ),
    }


def _gpu_uuids(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    normalized = value.replace(";", ",").replace(" ", ",")
    return sorted({item for item in normalized.split(",") if item})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return value


def _read_events(path: Path) -> list[Mapping[str, Any]]:
    with path.expanduser().resolve().open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
