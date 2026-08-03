from __future__ import annotations

import json
from pathlib import Path

import pytest

from conveyor_bench.v2 import exporters


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_export_source(tmp_path: Path) -> Path:
    episode = tmp_path / "ep-v2-export"
    episode.mkdir()
    suite = {
        "schema_version": "conveyor-bench-v2-task-context-1",
        "benchmark_suite_version": "conveyor-bench-v2",
        "canonical_protocol_version": "conveyor-bench-v1",
        "scene_id": "transverse_near_sort_v2",
        "task_family": "continuous_multi_target",
        "robot_mode": "fixed_base",
        "target_sequence_ids": ["target-a", "target-b"],
        "destination_zone_by_target": {
            "target-a": "zone-a",
            "target-b": "zone-b",
        },
    }
    _write_json(
        episode / "manifest.json",
        {
            "episode": {
                "task": {
                    "metadata": {"benchmark_suite": suite},
                }
            }
        },
    )
    _write_jsonl(
        episode / "steps.jsonl",
        [
            {
                "model_tick": 0,
                "sim_step": 0,
                "selected_object_id": "target-a",
            },
            {
                "model_tick": 0,
                "sim_step": 1,
                "selected_object_id": "target-a",
            },
            {
                "model_tick": 1,
                "sim_step": 2,
                "selected_object_id": "target-b",
            },
        ],
    )
    return episode


@pytest.mark.parametrize(
    ("iterator_name", "profile_field"),
    (
        ("iter_dynamicvla_records", "delta_action7_chunk"),
        ("iter_m0_records", "world_delta_arm7_chunk"),
        ("iter_m0_mobile_records", "model_action10_chunk"),
    ),
)
def test_v2_export_is_a_lossless_v1_projection_with_task_supervision(
    tmp_path: Path,
    monkeypatch,
    iterator_name: str,
    profile_field: str,
) -> None:
    episode = _make_export_source(tmp_path)
    base_records = [
        {
            "schema_version": "conveyor-bench-v1-export-1",
            "profile": (
                "dynamicvla"
                if "dynamic" in iterator_name
                else "m0_mobile_v1"
                if "mobile" in iterator_name
                else "m0"
            ),
            "model_tick": tick,
            "sim_step": tick * 2 + 1,
            "state6": (1.0, 2.0, 3.0, 0.0, 0.0, 0.0),
            "canonical_action10_chunk": ((0.1,) * 10,),
            profile_field: ((0.2,) * 7,),
        }
        for tick in range(2)
    ]
    v1_name = f"_{iterator_name}_v1"
    monkeypatch.setattr(
        exporters,
        v1_name,
        lambda _episode, _config=None: iter(base_records),
    )

    records = list(getattr(exporters, iterator_name)(episode))

    for index, (base, record) in enumerate(zip(base_records, records, strict=True)):
        assert record["schema_version"] == "conveyor-bench-v2-export-1"
        for key, value in base.items():
            if key != "schema_version":
                assert record[key] == value
        assert record["scene_id"] == "transverse_near_sort_v2"
        assert record["task_family"] == "continuous_multi_target"
        assert record["target_sequence_ids"] == ("target-a", "target-b")
        assert record["destination_zone_by_target"] == {
            "target-a": "zone-a",
            "target-b": "zone-b",
        }
        assert record["current_target_id"] == ("target-a", "target-b")[index]
        assert record["current_subtask_index"] == index
        assert record["supervision_only_fields"] == (
            "current_target_id",
            "current_subtask_index",
        )
