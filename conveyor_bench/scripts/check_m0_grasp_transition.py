#!/usr/bin/env python3
"""Fail closed unless M0 predicts the demonstrated descend/close boundary."""

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
    parser.add_argument("--index-tolerance", type=int, default=4)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_candidates <= 0 or args.index_tolerance < 0:
        raise SystemExit("max-candidates must be positive and index-tolerance non-negative")
    episode = args.episode_root.expanduser().resolve()
    export = episode / "exports" / "m0_mobile.jsonl"
    candidates = [
        record
        for record in (
            json.loads(line) for line in export.read_text(encoding="utf-8").splitlines()
        )
        if _is_transition_candidate(record)
    ][: args.max_candidates]
    if not candidates:
        raise SystemExit("export contains no pregrasp-to-close transition candidates")

    client = M0OnlineClient.from_files(
        args.endpoint, args.state_statistics.expanduser().resolve()
    )
    health = client.health()
    checks = []
    for offset, record in enumerate(candidates):
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
            sequence_id=offset,
            request_id=f"grasp-transition-gate:{offset}",
            seed=args.seed + offset,
        )
        target = record["canonical_action10_chunk"]
        predicted = result.physical_actions
        target_descent = _first_index([action[5] < -0.003 for action in target])
        predicted_descent = _first_index(
            [action[5] < -0.003 for action in predicted]
        )
        target_close = _first_index([action[9] < 0.0 for action in target])
        predicted_close = _first_index([action[9] < 0.5 for action in predicted])
        ok = bool(
            target_descent is not None
            and predicted_descent is not None
            and target_close is not None
            and predicted_close is not None
            and abs(predicted_descent - target_descent) <= args.index_tolerance
            and abs(predicted_close - target_close) <= args.index_tolerance
        )
        checks.append(
            {
                "observation_time_s": record["observation_time_s"],
                "target_descent_index": target_descent,
                "predicted_descent_index": predicted_descent,
                "target_close_index": target_close,
                "predicted_close_index": predicted_close,
                "server_inference_ms": result.server_inference_ms,
                "round_trip_ms": result.round_trip_ms,
                "ok": ok,
            }
        )
    report = {
        "schema_version": "conveyor-bench-m0-grasp-transition-gate-1",
        "ok": all(check["ok"] for check in checks),
        "model": health["model"],
        "candidate_count": len(checks),
        "index_tolerance": args.index_tolerance,
        "checks": checks,
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
