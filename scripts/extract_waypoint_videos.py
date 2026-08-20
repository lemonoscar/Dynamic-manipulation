#!/usr/bin/env python3
"""Extract split x route waypoint-v1 review clips with GT target overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.waypoint import WaypointRoute  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args(argv)
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    root = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for split in ("train", "val", "test"):
        path = root / manifest["records"][split]["relative_path"]
        selected = _best_segments(path, args.max_frames)
        for route in WaypointRoute:
            rows = selected.get(route.value)
            if not rows:
                raise RuntimeError(f"no {split}/{route.value} video segment")
            frames = [_render_frame(row) for row in rows]
            clip_path = output / split / f"{route.value}.mp4"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            _write_video(clip_path, frames)
            clips.append(
                {
                    "split": split,
                    "route": route.value,
                    "source_episode_id": rows[0]["source_episode_id"],
                    "source_row_ids": [row["source_row_id"] for row in rows],
                    "frame_count": len(rows),
                    "relative_path": clip_path.relative_to(output).as_posix(),
                    "sha256": _sha256(clip_path),
                }
            )
    payload = {
        "schema_version": "conveyorvla-waypoint-data-clips-v1",
        "dataset_manifest_sha256": _sha256(root / "manifest.json"),
        "fps": 5,
        "frame_layout": "head_current_left_wrist_current_right",
        "source_cameras": ["front/head", "wrist"],
        "third_person_available": False,
        "gt_overlay": "first_valid_NAV_body_waypoint_or_ARM_absolute_TCP_target",
        "clips": clips,
    }
    temporary = output / ".clip_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "clip_manifest.json")
    print(json.dumps({**payload, "output_root": str(output)}, indent=2, sort_keys=True))
    return 0


def _best_segments(path: Path, max_frames: int) -> dict[str, list[Mapping[str, Any]]]:
    best: dict[str, list[Mapping[str, Any]]] = {}
    current: list[Mapping[str, Any]] = []
    identity: tuple[str, str] | None = None

    def finish() -> None:
        if not current or identity is None:
            return
        route = identity[1]
        if len(current) > len(best.get(route, ())):
            best[route] = list(current[:max_frames])

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            row_identity = (str(row["source_episode_id"]), str(row["route"]))
            if row_identity != identity:
                finish()
                current = []
                identity = row_identity
            if len(current) < max_frames:
                current.append(row)
        finish()
    return best


def _render_frame(row: Mapping[str, Any]) -> np.ndarray:
    from PIL import Image, ImageDraw

    with Image.open(row["head_images"][1]) as image:
        head = image.convert("RGB").resize((320, 240))
    with Image.open(row["wrist_images"][1]) as image:
        wrist = image.convert("RGB").resize((320, 240))
    canvas = Image.new("RGB", (640, 300), "black")
    canvas.paste(head, (0, 60))
    canvas.paste(wrist, (320, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{row['split']} | {row['route']} | t={row['timestamp']:.2f}s", fill="white")
    valid = [index for index, value in enumerate(row["action_valid_mask"]) if value]
    if row["route"] == WaypointRoute.DONE.value:
        target = "DONE: no action"
    elif row["action_domain"] == "NAVIGATION":
        index = valid[0] if valid else None
        target = "NAV RECOVER" if index is None else f"GT NAV[{index}] body={_rounded(row['nav_waypoints_body'][index])}"
    else:
        index = valid[0] if valid else None
        target = "ARM RECOVER" if index is None else f"GT ARM[{index}] TCP_Bt={_rounded(row['arm_targets_base'][index])}"
    draw.text((6, 22), target, fill="white")
    draw.text((6, 40), f"valid={len(valid)}/20 boundary={row['boundary_transition']}", fill="white")
    return np.asarray(canvas, dtype=np.uint8)


def _rounded(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _write_video(path: Path, frames: Sequence[np.ndarray]) -> None:
    import av

    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width = int(frames[0].shape[1])
        stream.height = int(frames[0].shape[0])
        stream.pix_fmt = "yuv420p"
        for value in frames:
            frame = av.VideoFrame.from_ndarray(value, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
