#!/usr/bin/env python3
"""Audit one canonical V1 episode and write quality_report.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from conveyor_bench.v1.quality import (  # noqa: E402
    DataStatus,
    FrameStatsProvider,
    audit_episode,
)

_CANONICAL_FILENAMES = {
    "manifest.json",
    "summary.json",
    "steps.jsonl",
    "objects.jsonl",
    "action_chunks.jsonl",
    "events.jsonl",
    "camera_frames.jsonl",
}


def _quality_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _normalize_quality(
    quality: Mapping[str, Any],
    frame_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    visibility = quality.get("object_visibility")
    if not isinstance(visibility, Mapping) and frame_entry is not None:
        visibility = frame_entry.get("object_visibility")
    return {
        "black_fraction": quality.get(
            "black_fraction", quality.get("dark_fraction")
        ),
        "blur_score": quality.get(
            "blur_score", quality.get("laplacian_variance")
        ),
        "object_visibility": (
            dict(visibility) if isinstance(visibility, Mapping) else {}
        ),
    }


def load_jsonl_frame_stats(
    camera_index_path: str | Path,
) -> FrameStatsProvider | None:
    """Build a provider from quality fields already stored in camera JSONL."""

    path = Path(camera_index_path)
    if not path.is_file():
        return None
    stats: dict[tuple[str, int], Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} contains invalid JSON: {error}"
                ) from error
            if not isinstance(row, Mapping):
                continue
            frame_index = row.get("frame_index")
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                continue
            frames = row.get("frames")
            if isinstance(frames, Mapping):
                for camera_id, raw_entry in frames.items():
                    if not isinstance(camera_id, str) or not isinstance(
                        raw_entry, Mapping
                    ):
                        continue
                    quality = _quality_mapping(raw_entry.get("quality"))
                    if quality is not None:
                        stats[(camera_id, frame_index)] = _normalize_quality(
                            quality, raw_entry
                        )
                continue
            camera_id = row.get("camera_id")
            quality = _quality_mapping(row.get("quality"))
            if isinstance(camera_id, str) and quality is not None:
                stats[(camera_id, frame_index)] = _normalize_quality(
                    quality, row
                )
    if not stats:
        return None

    def provider(
        _frame_path: Path, frame_reference: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        camera_id = frame_reference.get("camera_id")
        frame_index = frame_reference.get("frame_index")
        if not isinstance(camera_id, str) or not isinstance(frame_index, int):
            return None
        return stats.get((camera_id, frame_index))

    return provider


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_to_file(
    episode_directory: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    episode = Path(episode_directory)
    destination = (
        Path(output_path)
        if output_path is not None
        else episode / "quality_report.json"
    )
    canonical_paths = {
        (episode / name).resolve() for name in _CANONICAL_FILENAMES
    }
    if destination.resolve() in canonical_paths:
        raise ValueError("quality report cannot overwrite a canonical episode file")

    camera_index = episode / "camera_frames.jsonl"
    provider = load_jsonl_frame_stats(camera_index)
    report = audit_episode(episode, frame_stats_provider=provider)
    payload = report.to_dict()
    payload["frame_stats_source"] = (
        "camera_frames.jsonl" if provider is not None else None
    )
    _atomic_write_json(destination, payload)
    return destination, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path; defaults to <episode>/quality_report.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path, report = audit_to_file(args.episode, args.output)
    except (OSError, ValueError) as error:
        print(f"audit_v1_episode: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "episode_id": report["episode_id"],
                "task_outcome": report["task_outcome"],
                "data_status": report["data_status"],
                "quality_report": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if report["data_status"] == DataStatus.CORRUPT.value else 0


if __name__ == "__main__":
    raise SystemExit(main())
