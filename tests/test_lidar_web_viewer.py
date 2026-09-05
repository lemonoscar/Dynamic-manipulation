from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from conveyor_bench.perception.lidar_web_viewer import (
    AUDIT_COLORS,
    POINT_RECORD_DTYPE,
    ScanRepository,
    make_server,
)


def _evidence(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    (root / "raw/scans").mkdir(parents=True)
    (root / "audit/object_ids").mkdir(parents=True)
    xyz = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.5, 0.25], [11.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.savez_compressed(
        root / "raw/scans/scan_000000.npz",
        xyz=xyz,
        intensity=np.ones(3, dtype=np.float32),
        time=np.zeros(3, dtype=np.float32),
        ring=np.arange(3, dtype=np.uint16),
    )
    np.savez_compressed(
        root / "audit/object_ids/scan_000000.npz",
        object_id=np.asarray([1, 2, 3], dtype=np.uint32),
    )
    row = {
        "scan_index": 7,
        "sim_time_s": 1.25,
        "frame_id": "unilidar_lidar",
        "raw_path": "raw/scans/scan_000000.npz",
        "object_id_audit_path": "audit/object_ids/scan_000000.npz",
    }
    (root / "raw/scans.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root


def test_repository_filters_display_range_and_colors_audit(tmp_path: Path) -> None:
    repository = ScanRepository(_evidence(tmp_path))
    payload = repository.point_payload(0, "audit")
    records = np.frombuffer(payload.body, dtype=POINT_RECORD_DTYPE)

    assert repository.metadata()["scan_count"] == 1
    assert payload.point_count == 2
    assert payload.scan_index == 7
    assert payload.object_counts == {1: 1, 2: 1, 3: 0, 4: 0}
    np.testing.assert_allclose(records["position"], [[1.0, 0.0, 0.0], [2.0, 0.5, 0.25]])
    np.testing.assert_array_equal(records["color"][1], AUDIT_COLORS[2])


def test_repository_colors_external_sam_selection(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    label_dir = tmp_path / "sam_labels"
    label_dir.mkdir()
    np.savez_compressed(
        label_dir / "scan_000007_sam3d.npz",
        selected=np.asarray([False, True, False]),
    )
    repository = ScanRepository(evidence, sam_label_dir=label_dir)

    payload = repository.point_payload(0, "sam")
    records = np.frombuffer(payload.body, dtype=POINT_RECORD_DTYPE)

    assert repository.metadata()["sam_positions"] == [0]
    assert payload.sam_selected_count == 1
    np.testing.assert_array_equal(records["color"][1], [255, 35, 210, 255])


def test_http_server_is_localhost_only_and_serves_binary_scan(tmp_path: Path) -> None:
    html = tmp_path / "viewer.html"
    html.write_text("<html>viewer</html>", encoding="utf-8")
    server = make_server(ScanRepository(_evidence(tmp_path)), html, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        with urlopen(f"http://127.0.0.1:{port}/api/meta") as response:
            assert json.load(response)["scan_count"] == 1
        with urlopen(f"http://127.0.0.1:{port}/api/scan?position=0&color=range") as response:
            assert response.headers["X-Point-Count"] == "2"
            assert len(response.read()) == 2 * POINT_RECORD_DTYPE.itemsize
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
