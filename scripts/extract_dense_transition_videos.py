#!/usr/bin/env python3
"""Extract train/val/test videos for all four dense-transition phases."""

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

from conveyor_bench.conveyorvla.lerobot_v3 import VIDEO_FEATURE_KEYS  # noqa: E402
from conveyor_bench.conveyorvla.subtasks import PHASE_ORDER  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.hierarchy_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    annotations = _read_jsonl(root / manifest["annotations_relative_path"])
    base = (root / manifest["base_dataset_relative_path"]).resolve()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=manifest["base_repo_id"],
        root=base,
        video_backend="pyav",
    )
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in annotations:
        by_episode.setdefault(str(row["source_episode_id"]), []).append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda item: int(item["base_index"]))

    clips = []
    phase_frames: dict[str, list[np.ndarray]] = {
        phase.name: [] for phase in PHASE_ORDER
    }
    for phase in PHASE_ORDER:
        for split in ("train", "val", "test"):
            selected = _select_annotation(annotations, split, phase.name)
            rows = by_episode[str(selected["source_episode_id"])]
            position = next(
                index
                for index, row in enumerate(rows)
                if row["sample_id"] == selected["sample_id"]
            )
            clip_rows = rows[max(0, position - 10) : position + 6]
            frames = [_render_frame(dataset[int(row["base_index"])], row) for row in clip_rows]
            path = output / "clips" / f"{phase.name}_{split}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_video(path, frames)
            clips.append(
                {
                    "split": split,
                    "phase": phase.name,
                    "sample_id": selected["sample_id"],
                    "source_episode_id": selected["source_episode_id"],
                    "base_indices": [int(row["base_index"]) for row in clip_rows],
                    "relative_path": path.relative_to(output).as_posix(),
                    "sha256": _sha256(path),
                }
            )
            phase_frames[phase.name].extend(frames)
        montage = output / f"{phase.name}_train_val_test.mp4"
        _write_video(montage, phase_frames[phase.name])

    full_episodes = []
    full_episode_frames: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        episode_id, rows = _select_full_episode(by_episode, split)
        frames = [
            _render_frame(dataset[int(row["base_index"])], row) for row in rows
        ]
        path = output / "full_episodes" / f"{split}_{episode_id.replace(':', '_')}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_video(path, frames)
        full_episode_frames.extend(frames)
        full_episodes.append(
            {
                "split": split,
                "source_episode_id": episode_id,
                "frame_count": len(frames),
                "phase_order": list(dict.fromkeys(row["phase_name"] for row in rows)),
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": _sha256(path),
            }
        )
    full_montage = output / "full_episode_train_val_test.mp4"
    _write_video(full_montage, full_episode_frames)

    payload = {
        "schema_version": "conveyor-vla-al0-dense-transition-clips-2",
        "hierarchy_manifest_sha256": _sha256(root / "manifest.json"),
        "fps": 5,
        "frame_layout": "head_current_left_wrist_current_right",
        "source_cameras": ["front/head", "wrist"],
        "third_person_available": False,
        "clips": clips,
        "full_episodes": full_episodes,
        "full_episode_montage": {
            "relative_path": full_montage.relative_to(output).as_posix(),
            "sha256": _sha256(full_montage),
        },
    }
    temporary = output / ".clip_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output / "clip_manifest.json")
    print(json.dumps({**payload, "output_root": str(output)}, indent=2, sort_keys=True))
    return 0


def _select_annotation(
    annotations: Sequence[Mapping[str, Any]],
    split: str,
    phase: str,
) -> Mapping[str, Any]:
    candidates = [
        row
        for row in annotations
        if row["split"] == split
        and row["phase_name"] == phase
        and row["is_boundary_window"]
    ]
    if not candidates:
        raise RuntimeError(f"no boundary clip candidate for {split}/{phase}")
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["seconds_to_boundary"])),
            str(row["source_episode_id"]),
            int(row["base_index"]),
        ),
    )


def _select_full_episode(
    by_episode: Mapping[str, Sequence[Mapping[str, Any]]],
    split: str,
) -> tuple[str, Sequence[Mapping[str, Any]]]:
    required = {phase.name for phase in PHASE_ORDER}
    candidates = [
        (episode_id, rows)
        for episode_id, rows in by_episode.items()
        if rows
        and rows[0]["split"] == split
        and {row["phase_name"] for row in rows} == required
    ]
    if not candidates:
        raise RuntimeError(f"no complete four-phase episode for {split}")
    return min(candidates, key=lambda item: item[0])


def _render_frame(frame: Mapping[str, Any], annotation: Mapping[str, Any]) -> np.ndarray:
    from PIL import Image, ImageDraw

    head = _rgb(frame[VIDEO_FEATURE_KEYS[1]])
    wrist = _rgb(frame[VIDEO_FEATURE_KEYS[3]])
    canvas = Image.new("RGB", (448, 280), "black")
    canvas.paste(Image.fromarray(head), (0, 56))
    canvas.paste(Image.fromarray(wrist), (224, 56))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{annotation['split']} | {annotation['phase_name']}", fill="white")
    draw.text(
        (6, 22),
        f"boundary={annotation['boundary_transition']} dt={annotation['seconds_to_boundary']:.2f}s",
        fill="white",
    )
    draw.text(
        (6, 40),
        f"valid={annotation['valid_action_steps']}/20 ref={annotation['navigation_reference_mode']}",
        fill="white",
    )
    return np.asarray(canvas, dtype=np.uint8)


def _rgb(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape == (3, 224, 224):
        array = np.transpose(array, (1, 2, 0))
    if array.shape != (224, 224, 3):
        raise RuntimeError(f"unexpected decoded video shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
    return array.astype(np.uint8)


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
