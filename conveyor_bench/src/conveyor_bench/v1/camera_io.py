"""Lossless, per-camera frame storage for ConveyorBench V1 episodes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from .protocol import CameraFrameRef


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    width: int
    height: int
    role: str

    def __post_init__(self) -> None:
        if not self.camera_id or "/" in self.camera_id or ".." in self.camera_id:
            raise ValueError("camera_id must be a safe directory name")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        if self.role not in {"policy_observation", "observer_only"}:
            raise ValueError(f"unsupported camera role: {self.role!r}")


class MultiCameraFrameWriter:
    """Write synchronized PNG images and an auditable JSONL frame index.

    The observer camera may use a different resolution from policy cameras.
    Each encoded image is published with ``os.replace`` before its index row is
    made visible, so an interrupted collection cannot reference a partial PNG.
    """

    def __init__(
        self,
        episode_directory: str | Path,
        specs: tuple[CameraSpec, ...],
    ) -> None:
        if not specs:
            raise ValueError("at least one camera spec is required")
        if len({spec.camera_id for spec in specs}) != len(specs):
            raise ValueError("camera ids must be unique")
        self._root = Path(episode_directory)
        self._specs = {spec.camera_id: spec for spec in specs}
        self._camera_root = self._root / "cameras"
        self._camera_root.mkdir(parents=True, exist_ok=False)
        for camera_id in self._specs:
            (self._camera_root / camera_id).mkdir()
        self._index = (self._root / "camera_frames.jsonl").open(
            "x", encoding="utf-8"
        )
        self.frame_count = 0
        self._closed = False

    def add(
        self,
        *,
        sim_step: int,
        capture_time_s: float,
        images: Mapping[str, Any],
    ) -> tuple[CameraFrameRef, ...]:
        if self._closed:
            raise RuntimeError("camera writer is closed")
        if set(images) != set(self._specs):
            missing = sorted(set(self._specs) - set(images))
            extra = sorted(set(images) - set(self._specs))
            raise ValueError(
                f"camera image ids do not match specs; missing={missing}, extra={extra}"
            )
        if sim_step < 0 or capture_time_s < 0:
            raise ValueError("sim_step and capture_time_s must be non-negative")

        cv2 = _import_cv2()
        references: list[CameraFrameRef] = []
        quality: dict[str, dict[str, float]] = {}
        for camera_id, spec in self._specs.items():
            bgr = _to_bgr(images[camera_id])
            if bgr.shape[:2] != (spec.height, spec.width):
                raise ValueError(
                    f"{camera_id} expected {(spec.height, spec.width)}, "
                    f"got {bgr.shape[:2]}"
                )
            success, encoded = cv2.imencode(
                ".png",
                bgr,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not success:
                raise RuntimeError(f"OpenCV could not encode {camera_id}")
            relative_path = (
                Path("cameras")
                / camera_id
                / f"{self.frame_count:06d}.png"
            )
            final_path = self._root / relative_path
            temporary = final_path.with_name(
                f".{final_path.stem}.{uuid4().hex}.tmp.png"
            )
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, final_path)
            references.append(
                CameraFrameRef(
                    camera_id=camera_id,
                    frame_index=self.frame_count,
                    capture_time_s=float(capture_time_s),
                    relative_path=relative_path.as_posix(),
                )
            )
            quality[camera_id] = _frame_quality(bgr, cv2)

        json.dump(
            {
                "frame_index": self.frame_count,
                "sim_step": int(sim_step),
                "capture_time_s": float(capture_time_s),
                "frames": {
                    reference.camera_id: {
                        "relative_path": reference.relative_path,
                        "resolution": [
                            self._specs[reference.camera_id].width,
                            self._specs[reference.camera_id].height,
                        ],
                        "role": self._specs[reference.camera_id].role,
                        "quality": quality[reference.camera_id],
                    }
                    for reference in references
                },
            },
            self._index,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self._index.write("\n")
        self.frame_count += 1
        return tuple(references)

    def close(self) -> None:
        if self._closed:
            return
        self._index.flush()
        os.fsync(self._index.fileno())
        self._index.close()
        self._closed = True

    def __enter__(self) -> "MultiCameraFrameWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _to_bgr(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"expected one camera environment, got shape {array.shape}"
            )
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected RGB/RGBA image, got shape {array.shape}")
    rgb = array[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    elif rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])


def _frame_quality(bgr: np.ndarray, cv2: Any) -> dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "mean_luma": float(gray.mean()),
        "luma_std": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "dark_fraction": float(np.mean(gray <= 5)),
        "bright_fraction": float(np.mean(gray >= 250)),
    }


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("OpenCV is required to write V1 camera frames") from error
    return cv2
