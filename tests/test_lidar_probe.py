from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conveyor_bench.perception import (
    LidarScan,
    ProbeRecorder,
    ThreePanelVideoWriter,
    UnitreeL2ProvisionalConfig,
    quaternion_wxyz_to_matrix,
    transform_points,
)


def _scan(index: int = 0, sim_time_s: float = 0.2) -> LidarScan:
    sensor_to_world = np.eye(4, dtype=np.float64)
    sensor_to_world[:3, 3] = (1.0, 2.0, 0.5)
    points_sensor = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.5, 0.4], [3.0, -0.5, -0.2]],
        dtype=np.float32,
    )
    return LidarScan(
        scan_index=index,
        sim_time_s=sim_time_s,
        xyz_sensor_m=points_sensor,
        xyz_world_m=transform_points(points_sensor, sensor_to_world).astype(np.float32),
        intensity=np.ones(3, dtype=np.float32),
        relative_time_s=np.asarray([0.0, 0.05, 0.10], dtype=np.float32),
        ring=np.asarray([0, 1, 2], dtype=np.uint16),
        sensor_to_world=sensor_to_world,
        emitted_point_count=4,
        backend="test",
        object_id_audit=np.asarray([10, 20, 30], dtype=np.uint32),
        intensity_synthetic=True,
    )


def test_unitree_l2_provisional_profile_is_explicit() -> None:
    config = UnitreeL2ProvisionalConfig()

    assert config.emitted_points_per_scan == 11_520
    assert config.emitted_points_per_second == pytest.approx(63_936.0)
    assert config.scan_period_s == pytest.approx(1.0 / 5.55)
    assert config.to_dict()["profile_status"] == "provisional_not_measured"


def test_quaternion_and_point_transform() -> None:
    rotation = quaternion_wxyz_to_matrix((2**-0.5, 0.0, 0.0, 2**-0.5))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (1.0, 2.0, 3.0)

    result = transform_points(np.asarray([[1.0, 0.0, 0.0]]), transform)

    np.testing.assert_allclose(result, [[1.0, 3.0, 3.0]], atol=1.0e-7)


def test_recorder_separates_primary_raw_from_object_id_audit(tmp_path: Path) -> None:
    recorder = ProbeRecorder(tmp_path / "probe", {"contract": "test"})
    raw_path = recorder.record_scan(_scan())
    summary_path = recorder.close({"done": True})

    with np.load(raw_path) as primary:
        assert set(primary.files) == {"xyz", "intensity", "time", "ring"}
    with np.load(tmp_path / "probe/audit/object_ids/scan_000000.npz") as audit:
        np.testing.assert_array_equal(audit["object_id"], [10, 20, 30])
    scan_metadata = json.loads(
        (tmp_path / "probe/raw/scans.jsonl").read_text(encoding="utf-8")
    )
    assert scan_metadata["object_id_audit_path"] == "audit/object_ids/scan_000000.npz"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["done"] is True
    assert summary["audit_object_id_record_count"] == 1
    audit_manifest = json.loads(
        (tmp_path / "probe/audit/manifest.json").read_text(encoding="utf-8")
    )
    assert audit_manifest["object_id_availability"] == "available"


def test_three_panel_writer_produces_expected_resolution(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "probe.mp4"
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 1] = 80
    overview = np.zeros((320, 480, 3), dtype=np.uint8)
    overview[..., 0] = 120
    writer = ThreePanelVideoWriter(
        path,
        UnitreeL2ProvisionalConfig(),
        fps=5.0,
    )
    writer.write(
        head_rgb=rgb,
        overview_rgb=overview,
        scan=_scan(),
        sim_time_s=0.2,
        phase="static",
    )
    writer.write(
        head_rgb=rgb,
        overview_rgb=overview,
        scan=_scan(),
        sim_time_s=0.3,
        phase="static",
    )
    assert len(writer._history) == 1
    writer.close()

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok is True
    assert frame.shape == (720, 1920, 3)
