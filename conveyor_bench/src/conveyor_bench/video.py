"""Small synchronous MP4 writer used by the single-environment V0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EpisodeVideoWriter:
    def __init__(self, directory: str | Path, *, fps: float, frame_size: tuple[int, int]):
        output_dir = Path(directory)
        width, height = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._head = cv2.VideoWriter(
            str(output_dir / "head_rgb.mp4"),
            fourcc,
            fps,
            (width, height),
        )
        self._wrist = cv2.VideoWriter(
            str(output_dir / "wrist_rgb.mp4"),
            fourcc,
            fps,
            (width, height),
        )
        self._overview = cv2.VideoWriter(
            str(output_dir / "overview_rgb.mp4"),
            fourcc,
            fps,
            (width, height),
        )
        if (
            not self._head.isOpened()
            or not self._wrist.isOpened()
            or not self._overview.isOpened()
        ):
            self.close()
            raise RuntimeError("OpenCV could not initialize the episode MP4 writers")
        self._metadata = (output_dir / "camera_frames.jsonl").open(
            "x",
            encoding="utf-8",
        )
        self.frame_count = 0

    def add(
        self,
        *,
        sim_step: int,
        sim_time_s: float,
        head_rgb: Any,
        wrist_rgb: Any,
        overview_rgb: Any,
    ) -> None:
        self._head.write(_to_bgr(head_rgb))
        self._wrist.write(_to_bgr(wrist_rgb))
        self._overview.write(_to_bgr(overview_rgb))
        json.dump(
            {
                "frame_index": self.frame_count,
                "sim_step": sim_step,
                "sim_time_s": sim_time_s,
            },
            self._metadata,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._metadata.write("\n")
        self.frame_count += 1

    def close(self) -> None:
        head = getattr(self, "_head", None)
        wrist = getattr(self, "_wrist", None)
        overview = getattr(self, "_overview", None)
        metadata = getattr(self, "_metadata", None)
        if head is not None:
            head.release()
            self._head = None
        if wrist is not None:
            wrist.release()
            self._wrist = None
        if overview is not None:
            overview.release()
            self._overview = None
        if metadata is not None:
            metadata.flush()
            metadata.close()
            self._metadata = None

    def __enter__(self) -> "EpisodeVideoWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _to_bgr(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Expected one camera environment, got shape {array.shape}")
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., ::-1])
