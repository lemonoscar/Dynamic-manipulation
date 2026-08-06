"""Lazy PyTorch dataset for policy-only ConveyorVLA AL0 examples."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from torch.utils.data import DataLoader, Dataset

from conveyor_bench.m0_mobile import (
    M0MobileError,
    M0MobileNormalizer,
    load_m0_mobile_config,
    sample_from_record,
)


class M0MobileDataset(Dataset[dict[str, Any]]):
    """Index local JSONL records and open policy images only on item access."""

    def __init__(
        self,
        jsonl_paths: str | Path | Iterable[str | Path],
        episode_root: str | Path,
        state_statistics: Mapping[str, Any] | str | Path,
        *,
        config: Mapping[str, Any] | None = None,
        allow_fixed_base: bool = False,
        expected_belt_speed_mps: float | None = None,
        expected_task_types: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(allow_fixed_base, bool):
            raise M0MobileError("allow_fixed_base must be a boolean")
        self.episode_root = Path(episode_root).expanduser().resolve()
        if not self.episode_root.is_dir():
            raise M0MobileError(
                f"episode root is not a directory: {self.episode_root}"
            )
        self.config = config if config is not None else load_m0_mobile_config()
        statistics = _load_statistics(state_statistics)
        if statistics.get("split") != "train":
            raise M0MobileError("state statistics must come from the train split")
        self.normalizer = M0MobileNormalizer.from_config(self.config, statistics)
        self.allow_fixed_base = allow_fixed_base
        if expected_belt_speed_mps is not None and (
            isinstance(expected_belt_speed_mps, bool)
            or not isinstance(expected_belt_speed_mps, (int, float))
            or not math.isfinite(expected_belt_speed_mps)
            or expected_belt_speed_mps < 0.0
        ):
            raise M0MobileError(
                "expected_belt_speed_mps must be non-negative and finite"
            )
        self.expected_belt_speed_mps = (
            float(expected_belt_speed_mps)
            if expected_belt_speed_mps is not None
            else None
        )
        allowed_task_types = {"stationary_sort", "dynamic_sort"}
        if expected_task_types is None:
            resolved_task_types = None
        elif isinstance(expected_task_types, str):
            resolved_task_types = frozenset((expected_task_types,))
        else:
            resolved_task_types = frozenset(expected_task_types)
        if resolved_task_types is not None and (
            not resolved_task_types
            or not resolved_task_types <= allowed_task_types
        ):
            raise M0MobileError(
                "expected_task_types must contain stationary_sort and/or "
                "dynamic_sort"
            )
        self.expected_task_types = resolved_task_types
        self._records: list[tuple[Path, int, int]] = []

        paths = _jsonl_paths(jsonl_paths)
        for path in paths:
            try:
                with path.open("rb") as stream:
                    line_number = 0
                    while True:
                        offset = stream.tell()
                        raw_line = stream.readline()
                        if not raw_line:
                            break
                        line_number += 1
                        record = _decode_record(path, line_number, raw_line)
                        self._validate_record(record, path, line_number)
                        sample_from_record(
                            record,
                            self.episode_root,
                            self.config,
                            require_images=False,
                        )
                        self._records.append((path, offset, line_number))
            except OSError as error:
                raise M0MobileError(f"cannot read {path}: {error}") from error
        if not self._records:
            raise M0MobileError("ConveyorVLA AL0 dataset contains no records")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, offset, line_number = self._records[index]
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                raw_line = stream.readline()
        except OSError as error:
            raise M0MobileError(f"cannot read {path}: {error}") from error
        record = _decode_record(path, line_number, raw_line)
        self._validate_record(record, path, line_number)
        sample = sample_from_record(
            record,
            self.episode_root,
            self.config,
            require_images=True,
        )
        return sample.as_model_example(self.normalizer, _load_rgb)

    def _validate_record(
        self,
        record: Mapping[str, Any],
        path: Path,
        line_number: int,
    ) -> None:
        location = f"{path}:{line_number}"
        if record.get("split") != "train":
            raise M0MobileError(f"{location} is not a train-split record")
        if record.get("source_task_outcome") != "success":
            raise M0MobileError(f"{location} is not a successful episode record")
        if record.get("source_assisted") is not False:
            raise M0MobileError(
                f"{location} source_assisted must be false; regenerate legacy "
                "exports that omit this field"
            )
        task_type = record.get("source_task_type")
        if task_type not in {"stationary_sort", "dynamic_sort"}:
            raise M0MobileError(
                f"{location} source_task_type must be stationary_sort or "
                "dynamic_sort; regenerate legacy exports that omit this field"
            )
        speed = record.get("belt_speed_mps")
        if (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not math.isfinite(speed)
            or speed < 0.0
        ):
            raise M0MobileError(
                f"{location} belt_speed_mps must be finite and non-negative"
            )
        stationary_speed = math.isclose(
            float(speed), 0.0, rel_tol=0.0, abs_tol=1.0e-9
        )
        if (task_type == "stationary_sort") != stationary_speed:
            raise M0MobileError(
                f"{location} source_task_type and belt_speed_mps disagree"
            )
        object_curriculum_split = record.get("object_curriculum_split")
        if object_curriculum_split not in {"train", "val", "unseen"}:
            raise M0MobileError(
                f"{location} object_curriculum_split must be train, val, or "
                "unseen; regenerate legacy exports that omit this field"
            )
        if task_type == "dynamic_sort" and object_curriculum_split != record.get(
            "split"
        ):
            raise M0MobileError(
                f"{location} dynamic_sort object_curriculum_split must match split"
            )
        if task_type == "stationary_sort" and object_curriculum_split != "train":
            raise M0MobileError(
                f"{location} stationary_sort object_curriculum_split must be train"
            )
        if (
            self.expected_task_types is not None
            and task_type not in self.expected_task_types
        ):
            raise M0MobileError(
                f"{location} source_task_type must be one of "
                f"{sorted(self.expected_task_types)}"
            )
        allowed_modes = {"whole_body_policy"}
        if self.allow_fixed_base:
            allowed_modes.add("fixed_base")
        if record.get("robot_mode") not in allowed_modes:
            raise M0MobileError(
                f"{location} robot_mode must be one of {sorted(allowed_modes)}"
            )
        if self.expected_belt_speed_mps is not None:
            if not math.isclose(
                float(speed),
                self.expected_belt_speed_mps,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise M0MobileError(
                    f"{location} belt_speed_mps must be "
                    f"{self.expected_belt_speed_mps}"
                )


def make_m0_mobile_loader(
    dataset: M0MobileDataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Return policy-ready batches without trying to tensorize PIL images."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=list,
    )


def _jsonl_paths(
    value: str | Path | Iterable[str | Path],
) -> tuple[Path, ...]:
    raw_paths = (value,) if isinstance(value, (str, Path)) else tuple(value)
    paths = tuple(Path(path).expanduser().resolve() for path in raw_paths)
    if not paths:
        raise M0MobileError("at least one AL0 legacy-profile JSONL path is required")
    if len(set(paths)) != len(paths):
        raise M0MobileError("AL0 legacy-profile JSONL paths must be unique")
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise M0MobileError(f"AL0 legacy-profile JSONL is not a file: {missing}")
    return paths


def _load_statistics(
    value: Mapping[str, Any] | str | Path,
) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    path = Path(value).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            statistics = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read state statistics {path}: {error}") from error
    if not isinstance(statistics, Mapping):
        raise M0MobileError("state statistics must be a JSON object")
    return statistics


def _decode_record(
    path: Path,
    line_number: int,
    raw_line: bytes,
) -> Mapping[str, Any]:
    try:
        record = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M0MobileError(
            f"{path}:{line_number} is not valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(record, Mapping):
        raise M0MobileError(f"{path}:{line_number} must contain a JSON object")
    return record


def _load_rgb(path: Path) -> Any:
    try:
        from PIL import Image
    except ImportError as error:
        raise M0MobileError("Pillow is required to load policy images") from error
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except OSError as error:
        raise M0MobileError(f"cannot load policy image {path}: {error}") from error


__all__ = ["M0MobileDataset", "make_m0_mobile_loader"]
