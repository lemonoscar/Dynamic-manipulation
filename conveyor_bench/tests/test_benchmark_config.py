import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_benchmark_config_has_one_runtime_contract() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8")
    )

    assert config["schema_version"] == "conveyor-bench-config-1"
    assert config["episode_protocol_version"] == "conveyor-bench-v1"
    assert config["scene"]["renderer"] == "isaac_rtx_native_nurec"
    assert config["scene"]["runtime_downloads_allowed"] is False
    assert config["assets"]["sidecar_delivery"] == "ssh_only"
    assert config["assets"]["sidecar_root_environment"] == (
        "CONVEYOR_BENCH_ASSET_ROOT"
    )


def test_robot_camera_and_conveyor_contract_match_runtime() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8")
    )

    assert config["conveyor"]["size_xyz_m"] == [0.252, 1.56, 0.06]
    assert config["conveyor"]["training_speed_mps"] == 0.01
    assert config["cameras"]["policy_cameras"] == ["head_rgb", "wrist_rgb"]
    assert config["cameras"]["observer_cameras"] == ["overview_rgb"]
    assert config["rates_hz"] == {
        "physics": 400,
        "control": 50,
        "camera": 25,
        "model": 25,
        "policy_query": 5,
    }


def test_collection_status_does_not_overclaim_mobile_readiness() -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "benchmark.json").read_text(encoding="utf-8")
    )

    assert config["task"]["subtasks"] == [
        "navigation",
        "dynamic_pick_and_place",
    ]
    assert config["status"]["fixed_base_dynamic_pick_place"] == "passed"
    assert config["status"]["whole_body_navigation"] == (
        "blocked_on_locomotion_gate"
    )
    assert config["status"]["formal_mobile_collection"] == "not_started"
