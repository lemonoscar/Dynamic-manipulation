#!/usr/bin/env python3
"""Stream AL0 legacy-profile JSONL into auditable state28 statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_SCHEMA_VERSIONS = frozenset(
    {"conveyor-bench-m0-mobile-v1", "conveyor-bench-v2-export-1"}
)
PROFILE = "m0_mobile_v1"
STATS_SCHEMA_VERSION = "conveyor-bench-m0-mobile-state-stats-v1"
STATE_LAYOUT = (
    "root_linear_velocity_body.x",
    "root_linear_velocity_body.y",
    "root_linear_velocity_body.z",
    "root_angular_velocity_body.x",
    "root_angular_velocity_body.y",
    "root_angular_velocity_body.z",
    "projected_gravity_body.x",
    "projected_gravity_body.y",
    "projected_gravity_body.z",
    *(f"arm_joint_position.{index}" for index in range(1, 7)),
    *(f"arm_joint_velocity.{index}" for index in range(1, 7)),
    "tcp_position_base.x",
    "tcp_position_base.y",
    "tcp_position_base.z",
    "tcp_rotation_vector_base.x",
    "tcp_rotation_vector_base.y",
    "tcp_rotation_vector_base.z",
    "gripper_open_fraction",
)


class StatisticsError(ValueError):
    """Raised when input cannot be audited as train-split AL0 data."""


def _record(path: Path, line_number: int, raw_line: bytes) -> Mapping[str, Any]:
    location = f"{path}:{line_number}"
    if not raw_line.strip():
        raise StatisticsError(f"{location} is blank")
    try:
        value = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StatisticsError(f"{location} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise StatisticsError(f"{location} must contain a JSON object")
    if value.get("schema_version") not in SOURCE_SCHEMA_VERSIONS:
        raise StatisticsError(f"{location} has an unsupported schema_version")
    if value.get("profile") != PROFILE:
        raise StatisticsError(f"{location} must use profile {PROFILE!r}")
    if value.get("split") != "train":
        raise StatisticsError(f"{location} is not a train-split record")
    if value.get("source_task_outcome") != "success":
        raise StatisticsError(f"{location} is not a successful episode record")
    if value.get("source_assisted") is not False:
        raise StatisticsError(
            f"{location} source_assisted must be explicitly false"
        )
    if value.get("object_curriculum_split") != "train":
        raise StatisticsError(
            f"{location} object_curriculum_split must be train"
        )
    if value.get("state_layout") != list(STATE_LAYOUT):
        raise StatisticsError(f"{location} has a non-canonical state_layout")
    return value


def _state28(record: Mapping[str, Any], path: Path, line_number: int) -> tuple[float, ...]:
    value = record.get("state28")
    location = f"{path}:{line_number}"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StatisticsError(f"{location} state28 must be an array")
    if len(value) != len(STATE_LAYOUT):
        raise StatisticsError(
            f"{location} state28 must contain {len(STATE_LAYOUT)} values"
        )
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise StatisticsError(f"{location} state28[{index}] is not numeric")
        number = float(item)
        if not math.isfinite(number):
            raise StatisticsError(f"{location} state28[{index}] is not finite")
        result.append(number)
    return tuple(result)


def compute_statistics(paths: Sequence[Path]) -> dict[str, Any]:
    """Compute population moments in one pass over local JSONL files."""

    if not paths:
        raise StatisticsError("at least one train JSONL input is required")
    resolved_paths = tuple(path.expanduser().resolve() for path in paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise StatisticsError("the same input path was supplied more than once")

    count = 0
    mean = [0.0] * len(STATE_LAYOUT)
    squared_deviation = [0.0] * len(STATE_LAYOUT)
    sources: list[dict[str, Any]] = []
    for path in resolved_paths:
        if not path.is_file():
            raise StatisticsError(f"input is not a local file: {path}")
        digest = hashlib.sha256()
        file_count = 0
        try:
            with path.open("rb") as stream:
                for line_number, raw_line in enumerate(stream, start=1):
                    digest.update(raw_line)
                    record = _record(path, line_number, raw_line)
                    state = _state28(record, path, line_number)
                    count += 1
                    file_count += 1
                    for index, value in enumerate(state):
                        delta = value - mean[index]
                        mean[index] += delta / count
                        squared_deviation[index] += delta * (value - mean[index])
        except OSError as error:
            raise StatisticsError(f"cannot read {path}: {error}") from error
        sources.append(
            {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "record_count": file_count,
            }
        )

    if count == 0:
        raise StatisticsError("train JSONL inputs contain no records")
    source_set_payload = "\n".join(
        sorted(source["sha256"] for source in sources)
    ).encode("ascii")
    layout_payload = json.dumps(
        STATE_LAYOUT, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": STATS_SCHEMA_VERSION,
        "accepted_source_schema_versions": sorted(SOURCE_SCHEMA_VERSIONS),
        "split": "train",
        "state_key": "state28",
        "state_dimension": len(STATE_LAYOUT),
        "state_layout": list(STATE_LAYOUT),
        "state_layout_sha256": hashlib.sha256(layout_payload).hexdigest(),
        "count": count,
        "mean": mean,
        "std": [math.sqrt(value / count) for value in squared_deviation],
        "std_definition": "population",
        "source_files": sources,
        "source_set_sha256": hashlib.sha256(source_set_payload).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train_jsonl",
        nargs="+",
        type=Path,
        help="local m0_mobile.jsonl files belonging to the train split",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new JSON file to create (existing files are never overwritten)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        statistics = compute_statistics(args.train_jsonl)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(statistics, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, StatisticsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
