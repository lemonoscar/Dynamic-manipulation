#!/usr/bin/env python3
"""Fail closed unless AL0 predicts the demonstrated descend/close boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from conveyor_bench.m0_online import M0OnlineClient  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--state-statistics", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18765")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--approach-candidates", type=int, default=3)
    parser.add_argument("--approach-intent-threshold", type=float, default=0.08)
    parser.add_argument("--index-tolerance", type=int, default=4)
    parser.add_argument("--executed-prefix", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--report", type=Path)
    return parser


def _first_index(values: list[bool]) -> int | None:
    return next((index for index, value in enumerate(values) if value), None)


def _is_transition_candidate(record: dict) -> bool:
    chunk = record.get("canonical_action10_chunk")
    return bool(
        isinstance(chunk, list)
        and len(chunk) == 16
        and chunk[0][5] > 0.003
        and any(action[5] < -0.003 for action in chunk)
        and any(action[9] < 0.0 for action in chunk)
    )


def _spread(records: list[dict], count: int) -> list[dict]:
    if len(records) <= count:
        return records
    if count == 1:
        return [records[len(records) // 2]]
    return [
        records[round(index * (len(records) - 1) / (count - 1))]
        for index in range(count)
    ]


def _timing_ok(
    target_descent: int | None,
    predicted_descent: int | None,
    target_close: int | None,
    predicted_close: int | None,
    *,
    index_tolerance: int,
) -> bool:
    pairs = (
        (target_descent, predicted_descent),
        (target_close, predicted_close),
    )
    if any(target is None or predicted is None for target, predicted in pairs):
        return False
    return bool(
        all(
            abs(predicted - target) <= index_tolerance
            for target, predicted in pairs
        )
    )


def _transition_ok(
    target_descent: int | None,
    predicted_descent: int | None,
    target_close: int | None,
    predicted_close: int | None,
    *,
    index_tolerance: int,
    executed_prefix: int,
) -> bool:
    pairs = (
        (target_descent, predicted_descent),
        (target_close, predicted_close),
    )
    return bool(
        _timing_ok(
            target_descent,
            predicted_descent,
            target_close,
            predicted_close,
            index_tolerance=index_tolerance,
        )
        and all(
            target is not None
            and predicted is not None
            and (target >= executed_prefix or predicted < executed_prefix)
            for target, predicted in pairs
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.max_candidates <= 0
        or args.approach_candidates <= 0
        or args.approach_intent_threshold <= 0.0
        or args.index_tolerance < 0
        or not 1 <= args.executed_prefix <= 16
    ):
        raise SystemExit(
            "candidate counts and approach threshold must be positive, "
            "index-tolerance non-negative, and executed-prefix within [1, 16]"
        )
    episode = args.episode_root.expanduser().resolve()
    export = episode / "exports" / "m0_mobile.jsonl"
    records = [
        json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()
    ]
    candidates = [record for record in records if _is_transition_candidate(record)][
        : args.max_candidates
    ]
    if not candidates:
        raise SystemExit("export contains no pregrasp-to-close transition candidates")

    client = M0OnlineClient.from_files(
        args.endpoint, args.state_statistics.expanduser().resolve()
    )
    health = client.health()
    approach_records = _spread(
        [
            record
            for record in records
            if record["observation_time_s"]
            < candidates[0]["observation_time_s"]
            if record["canonical_action10_chunk"][0][0] >= 0.16
        ],
        args.approach_candidates,
    )
    if not approach_records:
        raise SystemExit("export contains no mobile-approach candidates")
    approach_checks = []
    for offset, record in enumerate(approach_records):
        frames = record["policy_camera_frames"]
        images = [
            np.asarray(Image.open(episode / frame["relative_path"]).convert("RGB"))
            for frame in frames
        ]
        sequence_id = int(record["model_tick"])
        result = client.infer(
            images[0],
            images[1],
            record["instruction"],
            record["state28"],
            sequence_id=sequence_id,
            request_id=f"approach-intent-gate:{offset}",
            seed=args.seed + sequence_id,
        )
        mean_prefix_vx = sum(
            action[0] for action in result.physical_actions[:2]
        ) / 2.0
        approach_checks.append(
            {
                "observation_time_s": record["observation_time_s"],
                "mean_prefix_vx_mps": mean_prefix_vx,
                "server_inference_ms": result.server_inference_ms,
                "round_trip_ms": result.round_trip_ms,
                "ok": mean_prefix_vx >= args.approach_intent_threshold,
            }
        )
    checks = []
    for offset, record in enumerate(candidates):
        sequence_id = int(record["model_tick"])
        frames = record["policy_camera_frames"]
        images = [
            np.asarray(Image.open(episode / frame["relative_path"]).convert("RGB"))
            for frame in frames
        ]
        result = client.infer(
            images[0],
            images[1],
            record["instruction"],
            record["state28"],
            sequence_id=sequence_id,
            request_id=f"grasp-transition-gate:{offset}",
            seed=args.seed + sequence_id,
        )
        target = record["canonical_action10_chunk"]
        predicted = result.physical_actions
        target_descent = _first_index([action[5] < -0.003 for action in target])
        predicted_descent = _first_index(
            [action[5] < -0.003 for action in predicted]
        )
        target_close = _first_index([action[9] < 0.0 for action in target])
        predicted_close = _first_index([action[9] < 0.5 for action in predicted])
        timing_within_tolerance = _timing_ok(
            target_descent,
            predicted_descent,
            target_close,
            predicted_close,
            index_tolerance=args.index_tolerance,
        )
        ok = _transition_ok(
            target_descent,
            predicted_descent,
            target_close,
            predicted_close,
            index_tolerance=args.index_tolerance,
            executed_prefix=args.executed_prefix,
        )
        checks.append(
            {
                "observation_time_s": record["observation_time_s"],
                "target_descent_index": target_descent,
                "predicted_descent_index": predicted_descent,
                "target_close_index": target_close,
                "predicted_close_index": predicted_close,
                "target_min_dz_m": min(action[5] for action in target),
                "predicted_min_dz_m": min(action[5] for action in predicted),
                "timing_within_tolerance": timing_within_tolerance,
                "target_descent_within_executed_prefix": bool(
                    target_descent is not None
                    and target_descent < args.executed_prefix
                ),
                "predicted_descent_within_executed_prefix": bool(
                    predicted_descent is not None
                    and predicted_descent < args.executed_prefix
                ),
                "target_close_within_executed_prefix": bool(
                    target_close is not None
                    and target_close < args.executed_prefix
                ),
                "predicted_close_within_executed_prefix": bool(
                    predicted_close is not None
                    and predicted_close < args.executed_prefix
                ),
                "server_inference_ms": result.server_inference_ms,
                "round_trip_ms": result.round_trip_ms,
                "ok": ok,
            }
        )
    report = {
        "schema_version": "conveyor-bench-m0-grasp-transition-gate-1",
        "ok": all(check["ok"] for check in checks)
        and all(check["ok"] for check in approach_checks),
        "model": health["model"],
        "candidate_count": len(checks),
        "approach_candidate_count": len(approach_checks),
        "approach_intent_threshold_mps": args.approach_intent_threshold,
        "index_tolerance": args.index_tolerance,
        "executed_prefix": args.executed_prefix,
        "checks": checks,
        "approach_checks": approach_checks,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        destination = args.report.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
