from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "collect.py"


def _load():
    spec = importlib.util.spec_from_file_location("collect", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        asset_root=tmp_path / "assets",
        output_root=tmp_path / "output",
        python=Path("/usr/bin/python3"),
        isaaclab_source=None,
        kit_cache_root=None,
        physical_gpu=3,
        robot_mode="fixed_base",
        episodes=3,
        seed=10,
        belt_speed=0.005,
        target_intercept_lead_time=5.0,
        max_duration=60.0,
        require_all_success=True,
        dry_run=False,
    )


def test_command_is_real_cola_three_camera_collection(tmp_path: Path) -> None:
    module = _load()
    args = _args(tmp_path)
    command = module.build_collection_command(args)

    assert command[0] == "/usr/bin/python3"
    assert str(PROJECT_ROOT / "scripts" / "run_benchmark.py") in command
    assert command[command.index("--target-asset") + 1] == "cola"
    assert command[command.index("--active-objects") + 1] == "1"
    assert command[command.index("--destination") + 1] == "sort_bin_blue"
    assert command[command.index("--belt-speed") + 1] == "0.005"
    assert command[command.index("--target-intercept-lead-time") + 1] == "5.0"
    assert "--save-camera-frames" in command
    kit_args = command[command.index("--kit_args") + 1]
    assert "--/renderer/activeGpu=3" in kit_args
    assert "--/renderer/multiGpu/enabled=false" in kit_args


def test_collection_defaults_to_dynamic_belt() -> None:
    module = _load()

    args = module.build_parser().parse_args(
        (
            "--asset-root",
            "/tmp/assets",
            "--output-root",
            "/tmp/output",
            "--physical-gpu",
            "3",
        )
    )

    assert args.belt_speed == pytest.approx(0.01)
    assert args.robot_mode == "whole_body_policy"
    assert args.max_duration == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("physical_gpu", 4),
        ("episodes", 9),
        ("belt_speed", 0.005),
        ("belt_speed", 0.02),
    ),
)
def test_collection_rejects_out_of_contract_work(
    tmp_path: Path, field: str, value: object
) -> None:
    module = _load()
    args = _args(tmp_path)
    args.asset_root.mkdir()
    setattr(args, field, value)

    with pytest.raises(module.CollectionError):
        module._resolve(args)


def test_collection_accepts_gpu_zero(tmp_path: Path) -> None:
    module = _load()
    args = _args(tmp_path)
    args.asset_root.mkdir()
    args.physical_gpu = 0

    assert module._resolve(args).physical_gpu == 0


def test_stationary_collection_requires_registered_seed(tmp_path: Path) -> None:
    module = _load()
    args = _args(tmp_path)
    args.asset_root.mkdir()
    args.belt_speed = 0.0
    args.seed = 0

    with pytest.raises(module.CollectionError, match="registered scenario seeds"):
        module._resolve(args)
