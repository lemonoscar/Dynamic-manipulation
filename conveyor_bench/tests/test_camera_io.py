import json

import numpy as np
import pytest

from conveyor_bench.schema.camera_io import (
    CameraSpec,
    MultiCameraFrameWriter,
)


def test_writer_accepts_mixed_camera_resolutions(tmp_path):
    specs = (
        CameraSpec("head_rgb", 8, 6, "policy_observation"),
        CameraSpec("wrist_rgb", 8, 6, "policy_observation"),
        CameraSpec("overview_rgb", 12, 7, "observer_only"),
    )
    writer = MultiCameraFrameWriter(tmp_path, specs)
    references = writer.add(
        sim_step=16,
        capture_time_s=0.04,
        images={
            "head_rgb": np.zeros((1, 6, 8, 4), dtype=np.uint8),
            "wrist_rgb": np.full((6, 8, 3), 127, dtype=np.uint8),
            "overview_rgb": np.ones((7, 12, 3), dtype=np.float32),
        },
    )
    writer.close()

    assert [reference.camera_id for reference in references] == [
        "head_rgb",
        "wrist_rgb",
        "overview_rgb",
    ]
    assert all((tmp_path / reference.relative_path).is_file() for reference in references)
    record = json.loads((tmp_path / "camera_frames.jsonl").read_text())
    assert record["frame_index"] == 0
    assert record["frames"]["overview_rgb"]["resolution"] == [12, 7]
    assert record["frames"]["head_rgb"]["quality"]["dark_fraction"] == 1.0


def test_writer_rejects_missing_camera(tmp_path):
    writer = MultiCameraFrameWriter(
        tmp_path,
        (CameraSpec("head_rgb", 8, 6, "policy_observation"),),
    )
    with pytest.raises(ValueError, match="missing"):
        writer.add(sim_step=0, capture_time_s=0.0, images={})
    writer.close()


def test_camera_spec_rejects_path_traversal():
    with pytest.raises(ValueError):
        CameraSpec("../head", 8, 6, "policy_observation")
