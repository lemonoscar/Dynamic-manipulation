#!/usr/bin/env python3
"""Verify consecutive finite training events and all three gradient paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--minimum-consecutive-steps", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)
    events = _read_jsonl(args.events.expanduser().resolve())
    train = [event for event in events if event.get("event") == "train_step"]
    window = train[-args.minimum_consecutive_steps :]
    problems = []
    steps = [int(event["step"]) for event in window]
    if len(window) != args.minimum_consecutive_steps:
        problems.append("not enough train_step events")
    elif steps != list(range(steps[0], steps[0] + len(steps))):
        problems.append("train_step events are not consecutive")
    numeric_fields = (
        "subtask_loss",
        "action_loss",
        "navigation_loss",
        "manipulation_loss",
        "gradient_norm",
        "vlm_gradient_norm",
        "navigation_gradient_norm",
        "manipulation_gradient_norm",
        "teacher_forcing_probability",
    )
    for event in window:
        for field in numeric_fields:
            value = event.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                problems.append(f"step {event.get('step')} has non-finite {field}")
        for field in (
            "vlm_gradient_norm",
            "navigation_gradient_norm",
            "manipulation_gradient_norm",
        ):
            if float(event.get(field, 0.0)) <= 0.0:
                problems.append(f"step {event.get('step')} has zero {field}")
        rates = event.get("learning_rates")
        if not isinstance(rates, list) or not rates or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in rates
        ):
            problems.append(f"step {event.get('step')} has invalid learning rates")
    if any(event.get("event") == "failed" for event in events):
        problems.append("run contains a failed event")
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
            problems.append("checkpoint is missing trainer_state.json")
    report = {
        "schema_version": "conveyor-vla-al0-training-health-audit-1",
        "ok": not problems,
        "events": str(args.events.expanduser().resolve()),
        "observed_steps": steps,
        "representative": window[-1] if window else None,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "problems": problems,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
