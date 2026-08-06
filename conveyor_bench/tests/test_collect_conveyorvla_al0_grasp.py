from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "collect_conveyorvla_al0_grasp.py"
SPEC = importlib.util.spec_from_file_location(
    "collect_conveyorvla_al0_grasp", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


def test_grasp_curriculum_is_two_speeds_four_shapes_and_unique_seeds() -> None:
    cells = collector.cells()
    seeds = [
        seed
        for cell in cells
        for phase in ("pilot", "production")
        for seed in cell.seeds(phase)
    ]

    assert len(cells) == 8
    assert {(cell.target, cell.belt_speed_mps) for cell in cells} == {
        (target, speed)
        for target in collector.TARGETS
        for speed in collector.BELT_SPEEDS_MPS
    }
    assert len(seeds) == len(set(seeds)) == 584
    assert all(len(cell.seeds("production")) == 72 for cell in cells)


def test_collection_command_is_single_target_slow_and_temporally_balanced(
    tmp_path: Path,
) -> None:
    cell = collector.cells()[0]
    command = collector.build_collection_command(
        cell,
        "pilot",
        cell.base_seed,
        1,
        tmp_path,
        Path(sys.executable),
        2,
    )

    assert command[command.index("--active-objects") + 1] == "1"
    assert command[command.index("--task-family") + 1] == "single_target"
    assert command[command.index("--belt-speed") + 1] == str(cell.belt_speed_mps)
    assert command[command.index("--target-intercept-lead-time") + 1] == "5.0"
    assert command[command.index("--max-duration") + 1] == "40.0"
    assert command[command.index("--destination") + 1] == "sort_bin_blue"
    assert "--require-all-success" in command
    kit_args = command[command.index("--kit_args") + 1]
    assert "--/renderer/activeGpu=2" in kit_args
    assert "--/renderer/multiGpu/enabled=false" in kit_args


def test_production_has_a_resumable_success_quota_and_seed_reserve() -> None:
    cell = collector.cells()[0]
    batches = collector._contiguous_batches(cell.seeds("production"))

    assert batches == tuple(
        (cell.base_seed + 1 + offset, 8) for offset in range(0, 72, 8)
    )
    assert collector.PRODUCTION_SUCCESS_TARGET_PER_CELL == 48


def test_dry_run_and_runtime_reject_gpu_zero_one(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--phase",
            "pilot",
            "--output-root",
            str(tmp_path),
            "--physical-gpu",
            "2",
            "--physical-gpu",
            "3",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["physical_gpus"] == [2, 3]
    assert len(payload["commands"]) == 8

    with pytest.raises(collector.CollectionError, match="only physical GPUs 2 and 3"):
        collector.run_phase(
            tmp_path / "invalid",
            "pilot",
            Path(sys.executable),
            (0, 1),
            workers=2,
        )

    with pytest.raises(collector.CollectionError, match="cannot exceed"):
        collector.run_phase(
            tmp_path / "oversubscribed",
            "pilot",
            Path(sys.executable),
            (2,),
            workers=2,
        )


def test_worker_environment_prepends_explicit_isaaclab_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "IsaacLab" / "source" / "isaaclab"
    package = source / "isaaclab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    resolved = collector._resolve_isaaclab_source(source)
    environment = collector._worker_environment(
        tmp_path / "output",
        "pilot",
        collector.cells()[0],
        200000,
        3,
        resolved,
    )

    assert environment["PYTHONPATH"].split(collector.os.pathsep) == [
        str(source.resolve()),
        "/existing/pythonpath",
    ]
