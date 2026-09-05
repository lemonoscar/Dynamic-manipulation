#!/usr/bin/env python3
"""Serve an interactive Liangzhu LiDAR viewer through an SSH tunnel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.perception.lidar_web_viewer import ScanRepository, make_server


DEFAULT_EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "artifacts/runs/liangzhu-lidar-oblique-boxpair-corrected-final-20260830/evidence"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sam-label-dir", type=Path)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    repository = ScanRepository(args.evidence_root, sam_label_dir=args.sam_label_dir)
    html_path = PROJECT_ROOT / "src/conveyor_bench/perception/lidar_viewer.html"
    server = make_server(repository, html_path, port=args.port)
    print(
        f"LiDAR viewer ready: http://127.0.0.1:{args.port} "
        f"({repository.scan_count} scans, localhost only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
