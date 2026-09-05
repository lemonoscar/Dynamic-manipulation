"""Localhost-only HTTP backend for interactive LiDAR scan inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


POINT_RECORD_DTYPE = np.dtype(
    [("position", "<f4", (3,)), ("color", "u1", (4,))], align=False
)
AUDIT_COLORS = {
    1: (105, 115, 125, 105),
    2: (255, 75, 55, 255),
    3: (70, 225, 105, 255),
    4: (65, 145, 255, 255),
}


@dataclass(frozen=True)
class PointPayload:
    body: bytes
    point_count: int
    scan_index: int
    sim_time_s: float
    object_counts: dict[int, int]
    sam_selected_count: int = 0


class ScanRepository:
    """Read immutable probe scans without materializing the full episode."""

    def __init__(
        self,
        evidence_root: Path,
        *,
        display_min_range_m: float = 0.05,
        display_max_range_m: float = 10.0,
        sam_label_dir: Path | None = None,
    ) -> None:
        self.root = Path(evidence_root).expanduser().resolve(strict=True)
        self.display_min_range_m = float(display_min_range_m)
        self.display_max_range_m = float(display_max_range_m)
        if not 0.0 <= self.display_min_range_m < self.display_max_range_m:
            raise ValueError("display range must satisfy 0 <= min < max")
        manifest = self.root / "raw" / "scans.jsonl"
        self._rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self._rows:
            raise ValueError(f"no scans in {manifest}")
        self.sam_label_dir = (
            None
            if sam_label_dir is None
            else Path(sam_label_dir).expanduser().resolve(strict=True)
        )
        self._sam_positions = [
            position
            for position, row in enumerate(self._rows)
            if self._sam_path(int(row["scan_index"])).is_file()
        ]

    @property
    def scan_count(self) -> int:
        return len(self._rows)

    def metadata(self) -> dict[str, Any]:
        return {
            "scan_count": self.scan_count,
            "first_scan_index": int(self._rows[0]["scan_index"]),
            "last_scan_index": int(self._rows[-1]["scan_index"]),
            "first_sim_time_s": float(self._rows[0]["sim_time_s"]),
            "last_sim_time_s": float(self._rows[-1]["sim_time_s"]),
            "display_min_range_m": self.display_min_range_m,
            "display_max_range_m": self.display_max_range_m,
            "point_record_bytes": POINT_RECORD_DTYPE.itemsize,
            "frame_id": self._rows[0].get("frame_id", "unknown"),
            "sam_positions": self._sam_positions,
        }

    def _sam_path(self, scan_index: int) -> Path:
        if self.sam_label_dir is None:
            return Path("/__sam_labels_unavailable__")
        return self.sam_label_dir / f"scan_{scan_index:06d}_sam3d.npz"

    def _resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError(f"scan path escapes evidence root: {relative_path}")
        return path

    def point_payload(self, position: int, color_mode: str) -> PointPayload:
        if not 0 <= position < self.scan_count:
            raise IndexError(f"scan position out of range: {position}")
        if color_mode not in {"range", "audit", "sam"}:
            raise ValueError(f"unsupported color mode: {color_mode}")
        row = self._rows[position]
        with np.load(self._resolve(row["raw_path"])) as raw:
            xyz = np.asarray(raw["xyz"], dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"invalid xyz shape: {xyz.shape}")
        ranges = np.linalg.norm(xyz, axis=1)
        valid = (
            np.isfinite(xyz).all(axis=1)
            & (ranges >= self.display_min_range_m)
            & (ranges <= self.display_max_range_m)
        )
        object_ids: np.ndarray | None = None
        sam_selected: np.ndarray | None = None
        audit_path = row.get("object_id_audit_path")
        if color_mode == "audit":
            if not audit_path:
                raise ValueError("audit colors are unavailable for this scan")
            with np.load(self._resolve(audit_path)) as audit:
                object_ids = np.asarray(audit["object_id"], dtype=np.uint32)
            if object_ids.shape != (len(xyz),):
                raise ValueError("object_id and xyz point counts differ")
        if color_mode == "sam":
            sam_path = self._sam_path(int(row["scan_index"]))
            if not sam_path.is_file():
                raise ValueError("SAM 3D labels are unavailable for this scan")
            with np.load(sam_path) as labels:
                sam_selected = np.asarray(labels["selected"], dtype=bool)
            if sam_selected.shape != (len(xyz),):
                raise ValueError("SAM selection and xyz point counts differ")
        xyz = xyz[valid]
        ranges = ranges[valid]
        if object_ids is not None:
            object_ids = object_ids[valid]
        if sam_selected is not None:
            sam_selected = sam_selected[valid]
        records = np.empty(len(xyz), dtype=POINT_RECORD_DTYPE)
        records["position"] = xyz
        object_counts: dict[int, int] = {}
        sam_selected_count = 0
        if color_mode == "sam" and sam_selected is not None:
            colors = np.tile(np.asarray((105, 115, 125, 65), dtype=np.uint8), (len(xyz), 1))
            colors[sam_selected] = np.asarray((255, 35, 210, 255), dtype=np.uint8)
            records["color"] = colors
            sam_selected_count = int(np.count_nonzero(sam_selected))
        elif color_mode == "audit" and object_ids is not None:
            colors = np.tile(np.asarray(AUDIT_COLORS[1], dtype=np.uint8), (len(xyz), 1))
            for object_id, color in AUDIT_COLORS.items():
                selected = object_ids == object_id
                colors[selected] = color
                object_counts[object_id] = int(np.count_nonzero(selected))
            records["color"] = colors
        else:
            normalized = np.clip(
                (ranges - self.display_min_range_m)
                / (self.display_max_range_m - self.display_min_range_m),
                0.0,
                1.0,
            )
            colors = np.empty((len(xyz), 4), dtype=np.uint8)
            colors[:, 0] = np.clip(255.0 * (1.5 * normalized - 0.25), 0, 255)
            colors[:, 1] = np.clip(255.0 * (1.0 - abs(2.0 * normalized - 1.0)), 0, 255)
            colors[:, 2] = np.clip(255.0 * (1.25 - 1.5 * normalized), 0, 255)
            colors[:, 3] = 255
            records["color"] = colors
        return PointPayload(
            body=records.tobytes(),
            point_count=len(records),
            scan_index=int(row["scan_index"]),
            sim_time_s=float(row["sim_time_s"]),
            object_counts=object_counts,
            sam_selected_count=sam_selected_count,
        )


def make_server(
    repository: ScanRepository,
    html_path: Path,
    *,
    port: int,
) -> ThreadingHTTPServer:
    html = Path(html_path).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send(html, "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/meta":
                    body = json.dumps(repository.metadata(), sort_keys=True).encode()
                    self._send(body, "application/json")
                    return
                if parsed.path == "/api/scan":
                    query = parse_qs(parsed.query)
                    position = int(query.get("position", ["0"])[0])
                    color_mode = query.get("color", ["range"])[0]
                    payload = repository.point_payload(position, color_mode)
                    headers = {
                        "X-Point-Count": str(payload.point_count),
                        "X-Scan-Index": str(payload.scan_index),
                        "X-Sim-Time-S": str(payload.sim_time_s),
                        "X-Object-Counts": json.dumps(payload.object_counts),
                        "X-SAM-Selected-Count": str(payload.sam_selected_count),
                    }
                    self._send(
                        payload.body,
                        "application/octet-stream",
                        headers=headers,
                    )
                    return
                self._send_error(404, "not found")
            except (IndexError, ValueError, KeyError) as error:
                self._send_error(400, str(error))

        def _send(
            self,
            body: bytes,
            content_type: str,
            *,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: int, message: str) -> None:
            body = json.dumps({"error": message}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"viewer: {format % args}")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


__all__ = ["AUDIT_COLORS", "POINT_RECORD_DTYPE", "PointPayload", "ScanRepository", "make_server"]
