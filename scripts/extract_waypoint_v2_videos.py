#!/usr/bin/env python3
"""Render auditable Waypoint-v2 phase-progress and boundary videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint import WaypointRoute  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_v2 import (  # noqa: E402
    BOUNDARY_EVENTS,
    DATASET_SCHEMA_VERSION_V2,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--clip-radius", type=int, default=10)
    args = parser.parse_args(argv)
    if args.clip_radius <= 0:
        parser.error("--clip-radius must be positive")
    root = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION_V2:
        parser.error("dataset is not immutable Waypoint-v2")
    records = {
        split: _read_jsonl(
            root / manifest["records"][split]["relative_path"]
        )
        for split in ("train", "val", "test")
    }
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        clips = []
        full_episodes = []
        for split, rows in records.items():
            by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_episode[str(row["source_episode_id"])].append(row)
            for episode_rows in by_episode.values():
                episode_rows.sort(key=lambda row: float(row["timestamp"]))
            for transition in BOUNDARY_EVENTS:
                selected = _select_transition(rows, transition)
                episode_rows = by_episode[str(selected["source_episode_id"])]
                position = next(
                    index
                    for index, row in enumerate(episode_rows)
                    if row["source_row_id"] == selected["source_row_id"]
                )
                clip_rows = episode_rows[
                    max(0, position - args.clip_radius) : position + args.clip_radius + 1
                ]
                path = staging / "boundaries" / f"{split}_{transition.replace('->', '_to_')}.mp4"
                _write_video(path, [_render_frame(row) for row in clip_rows])
                clips.append(
                    {
                        "split": split,
                        "transition": transition,
                        "transition_id": selected["transition_id"],
                        "source_episode_id": selected["source_episode_id"],
                        "source_row_ids": [row["source_row_id"] for row in clip_rows],
                        "relative_path": path.relative_to(staging).as_posix(),
                        "sha256": _sha256(path),
                    }
                )
            episode_id, episode_rows = _select_full_episode(by_episode)
            path = staging / "full_episodes" / f"{split}_{episode_id.replace(':', '_')}.mp4"
            _write_video(path, [_render_frame(row) for row in episode_rows])
            full_episodes.append(
                {
                    "split": split,
                    "source_episode_id": episode_id,
                    "frame_count": len(episode_rows),
                    "route_order": list(dict.fromkeys(str(row["route"]) for row in episode_rows)),
                    "untruncated": True,
                    "relative_path": path.relative_to(staging).as_posix(),
                    "sha256": _sha256(path),
                }
            )
        payload = {
            "schema_version": "conveyorvla-waypoint-v2-phase-video-manifest-v1",
            "dataset_schema_version": DATASET_SCHEMA_VERSION_V2,
            "dataset_manifest_sha256": _sha256(manifest_path),
            "fps": 5,
            "frame_layout": "head_current_left_wrist_current_right",
            "overlay_fields": [
                "route",
                "phase_progress",
                "boundary_class",
                "boundary_transition",
                "boundary_signed_time_s",
                "original_valid_prefix_k",
                "suffix_reason",
                "terminal_hold_applied",
            ],
            "clips": clips,
            "full_episodes": full_episodes,
        }
        temporary = staging / ".manifest.json.tmp"
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, staging / "manifest.json")
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({**payload, "output_root": str(output)}, indent=2, sort_keys=True))
    return 0


def _select_transition(
    rows: Sequence[Mapping[str, Any]], transition: str
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("transition_window")
        and row.get("boundary_transition") == transition
        and row.get("boundary_signed_time_s") is not None
    ]
    if not candidates:
        raise RuntimeError(f"no Waypoint-v2 transition candidate: {transition}")
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["boundary_signed_time_s"])),
            0 if row["boundary_class"] == "BEFORE" else 1,
            str(row["source_episode_id"]),
            int(row["source_row_id"]),
        ),
    )


def _select_full_episode(
    by_episode: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[str, Sequence[Mapping[str, Any]]]:
    required_routes = {route.value for route in WaypointRoute}
    required_transitions = set(BOUNDARY_EVENTS)
    candidates = [
        (episode_id, rows)
        for episode_id, rows in by_episode.items()
        if {str(row["route"]) for row in rows} == required_routes
        and {
            str(row["boundary_transition"])
            for row in rows
            if row.get("boundary_transition") is not None
        }
        == required_transitions
    ]
    if not candidates:
        raise RuntimeError("no complete Waypoint-v2 episode for phase video")
    return min(candidates, key=lambda item: item[0])


def _render_frame(record: Mapping[str, Any]) -> np.ndarray:
    from PIL import Image, ImageDraw

    head = _rgb_path(Path(str(record["head_images"][-1])))
    wrist = _rgb_path(Path(str(record["wrist_images"][-1])))
    canvas = Image.new("RGB", (448, 304), "black")
    canvas.paste(head, (0, 64))
    canvas.paste(wrist, (224, 64))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 4),
        f"{record['split']} | {record['route']} | progress={float(record['phase_progress']):.3f}",
        fill="white",
    )
    signed = record.get("boundary_signed_time_s")
    signed_text = "n/a" if signed is None else f"{float(signed):+.2f}s"
    draw.text(
        (6, 23),
        f"boundary={record['boundary_class']} {record.get('boundary_transition')} dt={signed_text}",
        fill="white",
    )
    draw.text(
        (6, 42),
        f"K*={record.get('original_valid_prefix_k')} suffix={record['suffix_reason']} hold={record['terminal_hold_applied']}",
        fill="white",
    )
    draw.text((6, 290), "HEAD", fill="white")
    draw.text((230, 290), "WRIST", fill="white")
    return np.asarray(canvas, dtype=np.uint8)


def _rgb_path(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").resize((224, 224))


def _write_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    import av

    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width = int(frames[0].shape[1])
        stream.height = int(frames[0].shape[0])
        stream.pix_fmt = "yuv420p"
        for value in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(value, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
