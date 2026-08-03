from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "compute_m0_mobile_state_stats.py"
CONFIG = PROJECT_ROOT / "configs" / "v1.json"


STATE_LAYOUT = [
    "root_linear_velocity_body.x",
    "root_linear_velocity_body.y",
    "root_linear_velocity_body.z",
    "root_angular_velocity_body.x",
    "root_angular_velocity_body.y",
    "root_angular_velocity_body.z",
    "projected_gravity_body.x",
    "projected_gravity_body.y",
    "projected_gravity_body.z",
    *(f"arm_joint_position.{index}" for index in range(1, 7)),
    *(f"arm_joint_velocity.{index}" for index in range(1, 7)),
    "tcp_position_base.x",
    "tcp_position_base.y",
    "tcp_position_base.z",
    "tcp_rotation_vector_base.x",
    "tcp_rotation_vector_base.y",
    "tcp_rotation_vector_base.z",
    "gripper_open_fraction",
]


def _record(state: list[float]) -> dict:
    return {
        "schema_version": "conveyor-bench-m0-mobile-v1",
        "profile": "m0_mobile_v1",
        "split": "train",
        "state28": state,
        "state_layout": STATE_LAYOUT,
    }


def _run(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(input_path), "--output", str(output_path)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v1_config_freezes_m0_mobile_export_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["exports"]["m0_mobile"]

    assert config["schema_version"] == "conveyor-bench-m0-mobile-v1"
    assert config["observation_rate_hz"] == 25
    assert config["action_rate_hz"] == 50
    assert config["action_horizon"] == 16
    assert config["state_dimension"] == 28
    assert config["state_layout"] == STATE_LAYOUT
    assert config["action_dimension_mask"] == [
        True,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert config["policy_camera_ids"] == ["head_rgb", "wrist_rgb"]


def test_cli_streams_population_statistics_and_hashes_source(tmp_path: Path) -> None:
    input_path = tmp_path / "m0_mobile.jsonl"
    output_path = tmp_path / "state_stats.json"
    rows = [_record([float(index) for index in range(28)])]
    rows.append(_record([float(index + 2) for index in range(28)]))
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    input_path.write_bytes(payload)

    completed = _run(input_path, output_path)

    assert completed.returncode == 0, completed.stderr
    stats = json.loads(output_path.read_text(encoding="utf-8"))
    assert stats["split"] == "train"
    assert stats["count"] == 2
    assert stats["mean"] == [float(index + 1) for index in range(28)]
    assert stats["std"] == [1.0] * 28
    assert stats["std_definition"] == "population"
    assert stats["state_layout"] == STATE_LAYOUT
    layout_payload = json.dumps(
        STATE_LAYOUT, ensure_ascii=False, separators=(",", ":")
    ).encode()
    source_hash = hashlib.sha256(payload).hexdigest()
    assert stats["state_layout_sha256"] == hashlib.sha256(layout_payload).hexdigest()
    assert stats["source_files"] == [
        {
            "path": str(input_path.resolve()),
            "sha256": source_hash,
            "record_count": 2,
        }
    ]
    assert stats["source_set_sha256"] == hashlib.sha256(source_hash.encode()).hexdigest()


def test_cli_rejects_non_train_or_non_finite_records(tmp_path: Path) -> None:
    for name, mutation, message in (
        ("val", {"split": "val"}, "not a train-split record"),
        ("nan", {"state28": [math.nan] * 28}, "is not finite"),
    ):
        input_path = tmp_path / f"{name}.jsonl"
        output_path = tmp_path / f"{name}.json"
        row = _record([0.0] * 28)
        row.update(mutation)
        input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        completed = _run(input_path, output_path)

        assert completed.returncode == 2
        assert message in completed.stderr
        assert not output_path.exists()
