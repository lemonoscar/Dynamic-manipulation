"""Raw LiDAR recording and headless three-panel probe rendering.

This module deliberately has no Isaac imports.  Simulator and hardware
adapters produce :class:`LidarScan`; the evidence contract and renderer remain
identical across backends.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class UnitreeL2ProvisionalConfig:
    """Explicit first-pass approximation of Unitree L2 normal mode."""

    profile_id: str = "unitree_l2_normal_uniform_provisional_v1"
    frame_id: str = "unilidar_lidar"
    horizontal_fov_deg: float = 360.0
    vertical_fov_deg: float = 90.0
    channels: int = 90
    columns_per_revolution: int = 128
    rotation_hz: float = 5.55
    raw_min_range_m: float = 0.05
    raw_max_range_m: float = 30.0
    display_max_range_m: float = 10.0
    accumulation_seconds: float = 1.0
    mount_parent_frame: str = "base"
    mount_position_xyz_m: tuple[float, float, float] = (
        0.28945,
        0.0,
        -0.046825,
    )
    mount_orientation_wxyz: tuple[float, float, float, float] = (
        0.131314,
        0.0,
        0.991341,
        0.0,
    )
    mount_extrinsic_provisional: bool = True
    deskew: bool = False
    range_noise_model: str = "ideal"

    def __post_init__(self) -> None:
        if self.channels < 2 or self.columns_per_revolution < 2:
            raise ValueError("LiDAR pattern must have at least 2x2 rays")
        if self.rotation_hz <= 0.0:
            raise ValueError("rotation_hz must be positive")
        if not 0.0 < self.raw_min_range_m < self.raw_max_range_m:
            raise ValueError("raw range must satisfy 0 < min < max")
        if not self.raw_min_range_m < self.display_max_range_m <= self.raw_max_range_m:
            raise ValueError("display range must lie inside the raw range")
        if self.accumulation_seconds <= 0.0:
            raise ValueError("accumulation_seconds must be positive")
        norm = float(np.linalg.norm(self.mount_orientation_wxyz))
        if abs(norm - 1.0) > 2.0e-3:
            raise ValueError("mount quaternion must be normalized")

    @property
    def scan_period_s(self) -> float:
        return 1.0 / self.rotation_hz

    @property
    def emitted_points_per_scan(self) -> int:
        return self.channels * self.columns_per_revolution

    @property
    def emitted_points_per_second(self) -> float:
        return self.emitted_points_per_scan * self.rotation_hz

    @property
    def horizontal_resolution_deg(self) -> float:
        return self.horizontal_fov_deg / self.columns_per_revolution

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "scan_period_s": self.scan_period_s,
                "emitted_points_per_scan": self.emitted_points_per_scan,
                "emitted_points_per_second": self.emitted_points_per_second,
                "horizontal_resolution_deg": self.horizontal_resolution_deg,
                "profile_status": "provisional_not_measured",
            }
        )
        return payload


def quaternion_wxyz_to_matrix(quaternion: np.ndarray | tuple[float, ...]) -> np.ndarray:
    """Return a 3x3 active rotation matrix for a normalized WXYZ quaternion."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion is invalid")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(
    position_xyz: np.ndarray | tuple[float, ...],
    orientation_wxyz: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_wxyz_to_matrix(orientation_wxyz)
    matrix[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    return matrix


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz)
    matrix = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected points shape (N, 3), got {points.shape}")
    if matrix.shape != (4, 4):
        raise ValueError(f"expected transform shape (4, 4), got {matrix.shape}")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class LidarScan:
    """One causal revolution with primary raw fields and optional audit IDs."""

    scan_index: int
    sim_time_s: float
    xyz_sensor_m: np.ndarray
    xyz_world_m: np.ndarray
    intensity: np.ndarray
    relative_time_s: np.ndarray
    ring: np.ndarray
    sensor_to_world: np.ndarray
    emitted_point_count: int
    backend: str
    object_id_audit: np.ndarray | None = None
    intensity_synthetic: bool = False

    def __post_init__(self) -> None:
        point_count = len(self.xyz_sensor_m)
        if self.xyz_sensor_m.shape != (point_count, 3):
            raise ValueError("xyz_sensor_m must have shape (N, 3)")
        if self.xyz_world_m.shape != (point_count, 3):
            raise ValueError("xyz_world_m must have shape (N, 3)")
        for name, values in (
            ("intensity", self.intensity),
            ("relative_time_s", self.relative_time_s),
            ("ring", self.ring),
        ):
            if np.asarray(values).shape != (point_count,):
                raise ValueError(f"{name} must have shape (N,)")
        if np.asarray(self.sensor_to_world).shape != (4, 4):
            raise ValueError("sensor_to_world must have shape (4, 4)")
        if self.object_id_audit is not None and np.asarray(self.object_id_audit).shape != (
            point_count,
        ):
            raise ValueError("object_id_audit must have shape (N,)")
        if self.emitted_point_count < point_count:
            raise ValueError("emitted point count cannot be smaller than valid count")

    @property
    def valid_point_count(self) -> int:
        return int(self.xyz_sensor_m.shape[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


class ProbeRecorder:
    """Write immutable scan evidence while keeping simulator truth separate."""

    def __init__(self, root: Path, resolved_config: Mapping[str, Any]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scan_dir = self.root / "raw" / "scans"
        self.audit_dir = self.root / "audit" / "object_ids"
        self.scan_dir.mkdir(parents=True, exist_ok=False)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "resolved_config.json").write_text(
            json.dumps(_jsonable(resolved_config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._scans = (self.root / "raw" / "scans.jsonl").open("x", encoding="utf-8")
        self._tf = (self.root / "raw" / "tf.jsonl").open("x", encoding="utf-8")
        self._clock = (self.root / "raw" / "clock.jsonl").open("x", encoding="utf-8")
        self._scan_count = 0
        self._valid_points = 0
        self._audit_scan_count = 0
        self._closed = False

    def record_scan(self, scan: LidarScan) -> Path:
        if self._closed:
            raise RuntimeError("recorder is closed")
        stem = f"scan_{scan.scan_index:06d}"
        relative_path = Path("raw") / "scans" / f"{stem}.npz"
        np.savez_compressed(
            self.root / relative_path,
            xyz=np.asarray(scan.xyz_sensor_m, dtype=np.float32),
            intensity=np.asarray(scan.intensity, dtype=np.float32),
            time=np.asarray(scan.relative_time_s, dtype=np.float32),
            ring=np.asarray(scan.ring, dtype=np.uint16),
        )
        audit_path: str | None = None
        if scan.object_id_audit is not None:
            audit_relative = Path("audit") / "object_ids" / f"{stem}.npz"
            np.savez_compressed(
                self.root / audit_relative,
                object_id=np.asarray(scan.object_id_audit, dtype=np.uint32),
            )
            audit_path = audit_relative.as_posix()
            self._audit_scan_count += 1
        metadata = {
            "scan_index": scan.scan_index,
            "sim_time_s": scan.sim_time_s,
            "frame_id": "unilidar_lidar",
            "raw_path": relative_path.as_posix(),
            "object_id_audit_path": audit_path,
            "emitted_point_count": scan.emitted_point_count,
            "valid_point_count": scan.valid_point_count,
            "backend": scan.backend,
            "intensity_synthetic": scan.intensity_synthetic,
            "deskewed": False,
        }
        self._scans.write(json.dumps(metadata, sort_keys=True) + "\n")
        self._tf.write(
            json.dumps(
                {
                    "scan_index": scan.scan_index,
                    "sim_time_s": scan.sim_time_s,
                    "parent_frame": "world",
                    "child_frame": "unilidar_lidar",
                    "matrix_row_major": np.asarray(scan.sensor_to_world).reshape(-1).tolist(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._clock.write(
            json.dumps({"scan_index": scan.scan_index, "sim_time_s": scan.sim_time_s}) + "\n"
        )
        for handle in (self._scans, self._tf, self._clock):
            handle.flush()
        self._scan_count += 1
        self._valid_points += scan.valid_point_count
        return self.root / relative_path

    def close(self, extra_summary: Mapping[str, Any] | None = None) -> Path:
        if self._closed:
            return self.root / "summary.json"
        for handle in (self._scans, self._tf, self._clock):
            handle.close()
        summary = {
            "scan_count": self._scan_count,
            "total_valid_points": self._valid_points,
            "primary_raw_contains_object_ids": False,
            "audit_object_ids_separate": True,
            "audit_object_id_record_count": self._audit_scan_count,
        }
        if extra_summary:
            summary.update(_jsonable(extra_summary))
        path = self.root / "summary.json"
        path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.root / "audit" / "manifest.json").write_text(
            json.dumps(
                {
                    "primary_raw_contains_object_ids": False,
                    "object_id_storage": "audit/object_ids/*.npz",
                    "object_id_record_count": self._audit_scan_count,
                    "object_id_availability": (
                        "available"
                        if self._audit_scan_count
                        else "unavailable_for_backend"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._closed = True
        return path

    def __enter__(self) -> "ProbeRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _to_bgr(image: np.ndarray) -> np.ndarray:
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected RGB image shape (H, W, >=3), got {array.shape}")
    rgb = np.clip(array[..., :3], 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(rgb[..., ::-1])


class ThreePanelVideoWriter:
    """Encode synchronized RGB, current-scan and one-second-map diagnostics."""

    def __init__(
        self,
        path: Path,
        config: UnitreeL2ProvisionalConfig,
        *,
        fps: float = 10.0,
        frame_size_wh: tuple[int, int] = (1920, 720),
    ) -> None:
        import cv2

        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.frame_size_wh = frame_size_wh
        self._history: deque[LidarScan] = deque()
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size_wh,
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer: {self.path}")
        self.frame_count = 0

    @staticmethod
    def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
        import cv2

        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((height, width, 3), 24, dtype=np.uint8)
        x0 = (width - resized.shape[1]) // 2
        y0 = (height - resized.shape[0]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    @staticmethod
    def _colorize(values: np.ndarray, minimum: float, maximum: float, colormap: int) -> np.ndarray:
        import cv2

        normalized = np.clip((values - minimum) / max(maximum - minimum, 1.0e-6), 0.0, 1.0)
        return cv2.applyColorMap((normalized * 255).astype(np.uint8), colormap).reshape(-1, 3)

    def _point_panel(
        self,
        points: np.ndarray,
        values: np.ndarray,
        *,
        title: str,
        value_label: str,
        value_limits: tuple[float, float],
        axes: str,
        colormap: int,
        width: int,
        height: int,
        ground_z: float,
    ) -> np.ndarray:
        import cv2

        canvas = np.full((height, width, 3), (17, 20, 24), dtype=np.uint8)
        plot_center = np.asarray((width * 0.50, height * 0.59))
        extent = self.config.display_max_range_m
        scale = min((width - 74) / (2.0 * math.sqrt(2.0) * extent), (height - 94) / (1.45 * extent))

        def project(values_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            horizontal = (values_xyz[:, 0] - values_xyz[:, 1]) / math.sqrt(2.0)
            depth = (values_xyz[:, 0] + values_xyz[:, 1]) / math.sqrt(2.0)
            vertical = 0.43 * depth + 1.35 * values_xyz[:, 2]
            pixels = np.stack(
                (
                    plot_center[0] + horizontal * scale,
                    plot_center[1] - vertical * scale,
                ),
                axis=1,
            )
            return pixels, depth

        grid_values = np.arange(-8.0, 8.1, 2.0)
        for value in grid_values:
            for endpoints in (
                np.asarray([[-8.0, value, ground_z], [8.0, value, ground_z]]),
                np.asarray([[value, -8.0, ground_z], [value, 8.0, ground_z]]),
            ):
                pixels, _ = project(endpoints)
                cv2.line(
                    canvas,
                    tuple(np.rint(pixels[0]).astype(int)),
                    tuple(np.rint(pixels[1]).astype(int)),
                    (47, 52, 58),
                    1,
                    cv2.LINE_AA,
                )

        origin = np.asarray([[0.0, 0.0, 0.0]])
        origin_pixel, _ = project(origin)
        for endpoint, color, label in (
            ((2.0, 0.0, 0.0), (70, 90, 235), "+X"),
            ((0.0, 2.0, 0.0), (75, 205, 90), "+Y"),
            ((0.0, 0.0, 2.0), (235, 130, 60), "+Z"),
        ):
            endpoint_pixel, _ = project(np.asarray([endpoint]))
            start = tuple(np.rint(origin_pixel[0]).astype(int))
            finish = tuple(np.rint(endpoint_pixel[0]).astype(int))
            cv2.arrowedLine(canvas, start, finish, color, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(
                canvas,
                label,
                (finish[0] + 3, finish[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )
        if len(points):
            spatial_range = np.linalg.norm(points, axis=1)
            mask = (
                np.isfinite(points).all(axis=1)
                & (spatial_range >= self.config.raw_min_range_m)
                & (spatial_range <= self.config.display_max_range_m)
            )
            selected = points[mask]
            selected_values = values[mask]
            if len(selected):
                pixels, depth = project(selected)
                px = np.rint(pixels[:, 0]).astype(np.int32)
                py = np.rint(pixels[:, 1]).astype(np.int32)
                colors = self._colorize(
                    selected_values,
                    value_limits[0],
                    value_limits[1],
                    colormap,
                )
                valid_pixels = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                order = np.argsort(depth[valid_pixels])[::-1]
                visible_x = px[valid_pixels][order]
                visible_y = py[valid_pixels][order]
                visible_colors = colors[valid_pixels][order]
                canvas[visible_y, visible_x] = visible_colors
                neighbor_y = np.minimum(visible_y + 1, height - 1)
                canvas[neighbor_y, visible_x] = visible_colors
        cv2.putText(canvas, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 240, 244), 1, cv2.LINE_AA)
        cv2.putText(canvas, axes, (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 174, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, value_label, (width - 168, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 174, 180), 1, cv2.LINE_AA)
        return canvas

    def write(
        self,
        *,
        head_rgb: np.ndarray,
        overview_rgb: np.ndarray,
        scan: LidarScan,
        sim_time_s: float,
        phase: str,
    ) -> None:
        import cv2

        width, height = self.frame_size_wh
        panel_width = width // 3
        if not self._history or self._history[-1].scan_index != scan.scan_index:
            self._history.append(scan)
        cutoff = sim_time_s - self.config.accumulation_seconds
        while self._history and self._history[0].sim_time_s < cutoff:
            self._history.popleft()

        head = self._letterbox(_to_bgr(head_rgb), panel_width, height)
        overview_width = 320
        overview_height = 214
        overview_x = panel_width - overview_width - 18
        overview_y = 42
        overview = self._letterbox(
            _to_bgr(overview_rgb), overview_width, overview_height
        )
        head[
            overview_y : overview_y + overview_height,
            overview_x : overview_x + overview_width,
        ] = overview
        cv2.rectangle(
            head,
            (overview_x - 2, overview_y - 2),
            (overview_x + overview_width + 2, overview_y + overview_height + 2),
            (240, 240, 240),
            2,
        )
        cv2.putText(
            head,
            "OVERVIEW | BOX1+COKE + BOX2",
            (overview_x + 8, overview_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (238, 240, 244),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(head, "HEAD RGB + OVERVIEW", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (238, 240, 244), 1, cv2.LINE_AA)
        cv2.putText(head, f"sim {sim_time_s:7.3f}s | {phase}", (18, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (238, 240, 244), 1, cv2.LINE_AA)
        cv2.putText(head, "Liangzhu box1 -> box2 coke | no conveyor", (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (238, 240, 244), 1, cv2.LINE_AA)

        center_world = scan.sensor_to_world[:3, 3]
        current_world_centered = scan.xyz_world_m - center_world
        ranges = np.linalg.norm(scan.xyz_sensor_m, axis=1)
        current_panel = self._point_panel(
            current_world_centered,
            ranges,
            title=f"CURRENT RAW SCAN #{scan.scan_index} | OBLIQUE 3D",
            value_label="color: range",
            value_limits=(self.config.raw_min_range_m, self.config.display_max_range_m),
            axes="gravity aligned | grid: 2m",
            colormap=cv2.COLORMAP_TURBO,
            width=panel_width,
            height=height,
            ground_z=-float(center_world[2]),
        )

        world_points = np.concatenate([item.xyz_world_m for item in self._history], axis=0)
        recency = np.concatenate(
            [
                np.full(
                    item.valid_point_count,
                    max(0.0, self.config.accumulation_seconds - (sim_time_s - item.sim_time_s)),
                    dtype=np.float32,
                )
                for item in self._history
            ],
            axis=0,
        )
        local_world = world_points - center_world
        world_panel = self._point_panel(
            local_world,
            recency,
            title=f"WORLD RAW | LAST {self.config.accumulation_seconds:.1f}s | OBLIQUE 3D",
            value_label="color: recency (red=new)",
            value_limits=(0.0, self.config.accumulation_seconds),
            axes="world axes | grid: 2m",
            colormap=cv2.COLORMAP_TURBO,
            width=width - 2 * panel_width,
            height=height,
            ground_z=-float(center_world[2]),
        )
        frame = np.concatenate([head, current_panel, world_panel], axis=1)
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "ThreePanelVideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
