from __future__ import annotations

import json
from pathlib import Path

import pytest


pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from conveyor_bench.conveyorvla.dataset import M0MobileDataset, make_m0_mobile_loader
from conveyor_bench.conveyorvla.config import M0MobileError


ACTION_MASK = [True, False, True, True, True, True, True, True, True, True]


def _record(name: str, *, mode: str = "whole_body_policy") -> dict:
    return {
        "schema_version": "conveyor-bench-m0-mobile-v1",
        "profile": "m0_mobile_v1",
        "split": "train",
        "source_task_outcome": "success",
        "source_task_type": "dynamic_sort",
        "source_assisted": False,
        "object_curriculum_split": "train",
        "robot_mode": mode,
        "belt_speed_mps": 0.08,
        "sample_id": name,
        "instruction": "pick the moving part",
        "policy_camera_frames": [
            {
                "camera_id": camera,
                "relative_path": f"cameras/{camera}/{name}.png",
            }
            for camera in ("head_rgb", "wrist_rgb")
        ],
        "state28": [2.0] * 28,
        "model_action10_chunk": [[0.0, 9.0] + [0.0] * 8] * 16,
        "action_dimension_mask": ACTION_MASK,
        "action_horizon": 16,
        "action_rate_hz": 50,
        "causal_offset_control_steps": 1,
    }


def _write_source(root: Path, name: str, record: dict) -> Path:
    for frame in record["policy_camera_frames"]:
        image_path = root / frame["relative_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2), 128).save(image_path)
    path = root / f"{name}.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def _statistics() -> dict:
    return {"split": "train", "mean": [0.0] * 28, "std": [1.0] * 28}


def test_dataset_is_lazy_normalized_and_policy_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [
        _write_source(tmp_path, name, _record(name))
        for name in ("first", "second")
    ]
    stats_path = tmp_path / "state_stats.json"
    stats_path.write_text(
        json.dumps({"split": "train", "mean": [1.0] * 28, "std": [1.0] * 28}),
        encoding="utf-8",
    )
    opened: list[Path] = []
    real_open = Image.open

    def tracked_open(path, *args, **kwargs):
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked_open)
    dataset = M0MobileDataset(paths, tmp_path, stats_path)
    assert opened == []

    example = dataset[0]

    assert set(example) == {"image", "lang", "state", "action", "action_mask"}
    assert [image.mode for image in example["image"]] == ["RGB", "RGB"]
    assert opened == [
        tmp_path / "cameras/head_rgb/first.png",
        tmp_path / "cameras/wrist_rgb/first.png",
    ]
    assert example["state"] == ((1.0,) * 28,)
    assert example["action"][0][1] == 0.0

    batch = next(
        iter(
            make_m0_mobile_loader(
                dataset,
                batch_size=2,
                shuffle=False,
            )
        )
    )
    assert isinstance(batch, list)
    assert len(batch) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("split", "val", "train-split"),
        ("source_task_outcome", "failure", "successful episode"),
        ("robot_mode", "fixed_base", "robot_mode"),
    ),
)
def test_dataset_rejects_non_training_records_by_default(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    record = _record("sample")
    record[field] = value
    path = _write_source(tmp_path, "sample", record)

    with pytest.raises(M0MobileError, match=message):
        M0MobileDataset(path, tmp_path, _statistics())


def test_dataset_can_explicitly_allow_fixed_base(tmp_path: Path) -> None:
    path = _write_source(tmp_path, "fixed", _record("fixed", mode="fixed_base"))

    dataset = M0MobileDataset(
        path,
        tmp_path,
        _statistics(),
        allow_fixed_base=True,
    )

    assert len(dataset) == 1


def test_dataset_can_freeze_initial_belt_speed(tmp_path: Path) -> None:
    record = _record("speed")
    record["belt_speed_mps"] = 0.06
    path = _write_source(tmp_path, "speed", record)

    with pytest.raises(M0MobileError, match="belt_speed_mps"):
        M0MobileDataset(
            path,
            tmp_path,
            _statistics(),
            expected_belt_speed_mps=0.08,
        )


def test_dataset_can_select_stationary_records_fail_closed(tmp_path: Path) -> None:
    record = _record("stationary")
    record["belt_speed_mps"] = 0.0
    record["source_task_type"] = "stationary_sort"
    path = _write_source(tmp_path, "stationary", record)

    dataset = M0MobileDataset(
        path,
        tmp_path,
        _statistics(),
        expected_belt_speed_mps=0.0,
        expected_task_types=("stationary_sort",),
    )

    assert len(dataset) == 1


def test_dataset_rejects_wrong_task_type_filter(tmp_path: Path) -> None:
    record = _record("dynamic")
    record["source_task_type"] = "dynamic_sort"
    path = _write_source(tmp_path, "dynamic", record)

    with pytest.raises(M0MobileError, match="source_task_type"):
        M0MobileDataset(
            path,
            tmp_path,
            _statistics(),
            expected_task_types=("stationary_sort",),
        )


@pytest.mark.parametrize(
    ("task_type", "speed"),
    (
        ("stationary_sort", 0.08),
        ("dynamic_sort", 0.0),
        (None, 0.08),
    ),
)
def test_dataset_always_rejects_task_speed_contract_mismatch(
    tmp_path: Path,
    task_type: str | None,
    speed: float,
) -> None:
    record = _record("mismatch")
    if task_type is None:
        record.pop("source_task_type")
    else:
        record["source_task_type"] = task_type
    record["belt_speed_mps"] = speed
    path = _write_source(tmp_path, "mismatch", record)

    with pytest.raises(M0MobileError, match="source_task_type"):
        M0MobileDataset(
            path,
            tmp_path,
            _statistics(),
        )


@pytest.mark.parametrize("value", (True, None))
def test_dataset_rejects_assisted_or_legacy_provenance(
    tmp_path: Path, value: bool | None
) -> None:
    record = _record("assisted")
    if value is None:
        record.pop("source_assisted")
    else:
        record["source_assisted"] = value
    path = _write_source(tmp_path, "assisted", record)

    with pytest.raises(M0MobileError, match="source_assisted"):
        M0MobileDataset(path, tmp_path, _statistics())


@pytest.mark.parametrize(
    ("task_type", "object_split"),
    (("dynamic_sort", "val"), ("stationary_sort", "val"), ("dynamic_sort", None)),
)
def test_dataset_rejects_missing_or_mismatched_object_curriculum(
    tmp_path: Path, task_type: str, object_split: str | None
) -> None:
    record = _record("curriculum")
    record["source_task_type"] = task_type
    if task_type == "stationary_sort":
        record["belt_speed_mps"] = 0.0
    if object_split is None:
        record.pop("object_curriculum_split")
    else:
        record["object_curriculum_split"] = object_split
    path = _write_source(tmp_path, "curriculum", record)

    with pytest.raises(M0MobileError, match="object_curriculum_split"):
        M0MobileDataset(path, tmp_path, _statistics())


def test_dataset_rejects_statistics_without_explicit_train_split(
    tmp_path: Path,
) -> None:
    path = _write_source(tmp_path, "sample", _record("sample"))
    statistics = _statistics()
    del statistics["split"]

    with pytest.raises(M0MobileError, match="train split"):
        M0MobileDataset(path, tmp_path, statistics)
