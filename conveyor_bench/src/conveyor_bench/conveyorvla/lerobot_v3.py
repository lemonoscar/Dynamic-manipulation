"""LeRobot v3 storage contract for ConveyorVLA AL0 temporal demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from conveyor_bench.m0_mobile import M0MobileError, M0MobileNormalizer

from .temporal import (
    ACTION_DIM,
    ACTION_DIMENSION_MASK,
    ACTION_HORIZON,
    CAMERA_IDS,
    CONTROL_HZ,
    HISTORY_OFFSETS_MODEL_TICKS,
    MODEL_HZ,
    STATE_DIM,
    TEMPORAL_PROFILE,
    TEMPORAL_SCHEMA_VERSION,
    temporal_sample_from_record,
)


LEROBOT_V3_CONFIG_SCHEMA_VERSION = "conveyor-vla-al0-lerobot-v3-config-1"
DEFAULT_LEROBOT_V3_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "conveyorvla_al0_lerobot_v3.json"
)
VIDEO_FEATURE_KEYS = (
    "observation.images.head_tminus2",
    "observation.images.head",
    "observation.images.wrist_tminus2",
    "observation.images.wrist",
)


class ConveyorVLAAL0LeRobotDataset:
    """Official LeRobot v3 rows adapted to the temporal AL0 policy input."""

    def __init__(
        self,
        root: str | Path,
        normalizer_config: Mapping[str, Any],
    ) -> None:
        dataset_root = Path(root).expanduser().resolve()
        manifest = _load_manifest(dataset_root)
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise M0MobileError("lerobot is required to train from LeRobot v3") from error
        self.dataset = LeRobotDataset(
            repo_id=str(manifest["repo_id"]),
            root=dataset_root,
            video_backend="pyav",
        )
        if len(self.dataset) != int(manifest["frame_count"]):
            raise M0MobileError("LeRobot training frame count disagrees with manifest")
        self.state_statistics = _state_statistics(
            self.dataset.meta.stats,
            count=len(self.dataset),
        )
        self.normalizer = M0MobileNormalizer.from_config(
            normalizer_config,
            self.state_statistics,
        )
        self.root = dataset_root
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return lerobot_model_example(self.dataset[index], self.normalizer)


def lerobot_model_example(
    frame: Mapping[str, Any],
    normalizer: M0MobileNormalizer,
) -> dict[str, Any]:
    """Convert one decoded LeRobot row into the existing temporal-policy API."""

    state = _numeric_array(frame.get("observation.state"), "observation.state")
    action = _numeric_array(frame.get("action"), "action")
    if state.shape != (STATE_DIM,) or action.shape != (ACTION_HORIZON * ACTION_DIM,):
        raise M0MobileError("decoded LeRobot state/action shape mismatch")
    instruction = frame.get("task")
    if not isinstance(instruction, str) or not instruction.strip():
        raise M0MobileError("decoded LeRobot task must be a non-empty string")
    return {
        "video": (
            (frame[VIDEO_FEATURE_KEYS[0]], frame[VIDEO_FEATURE_KEYS[1]]),
            (frame[VIDEO_FEATURE_KEYS[2]], frame[VIDEO_FEATURE_KEYS[3]]),
        ),
        "lang": instruction.strip(),
        "state": (normalizer.normalize_state(state.tolist()),),
        "action": tuple(
            normalizer.normalize_action(row)
            for row in action.reshape(ACTION_HORIZON, ACTION_DIM).tolist()
        ),
        "action_mask": ACTION_DIMENSION_MASK,
    }


def load_lerobot_v3_config(
    path: str | Path = DEFAULT_LEROBOT_V3_CONFIG_PATH,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read LeRobot v3 config {source}: {error}") from error
    if not isinstance(value, dict):
        raise M0MobileError("LeRobot v3 config must be a JSON object")
    _validate_config(value)
    return value


def lerobot_features(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the feature declaration accepted by LeRobotDataset.create()."""

    _validate_config(config)
    features = _mapping(config["features"], "features")
    result: dict[str, dict[str, Any]] = {}
    for video in _sequence(features["videos"], "features.videos"):
        spec = _mapping(video, "video feature")
        result[str(spec["key"])] = {
            "dtype": "video",
            "shape": tuple(spec["shape"]),
            "names": ["height", "width", "channels"],
        }
    state = _mapping(features["state"], "features.state")
    result[str(state["key"])] = {
        "dtype": str(state["dtype"]),
        "shape": tuple(state["shape"]),
        "names": list(state["names"]),
    }
    action = _mapping(features["action"], "features.action")
    action_names = [
        f"t+{tick:02d}.{name}"
        for tick in range(1, ACTION_HORIZON + 1)
        for name in action["step_names"]
    ]
    result[str(action["key"])] = {
        "dtype": str(action["dtype"]),
        "shape": tuple(action["shape"]),
        "names": action_names,
    }
    return result


def iter_query_records(
    jsonl_path: str | Path,
    config: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Select one model query every five 25 Hz source ticks."""

    _validate_config(config)
    path = Path(jsonl_path).expanduser().resolve()
    stride = int(_mapping(config["sampling"], "sampling")["query_stride_model_ticks"])
    anchor: int | None = None
    previous_tick: int | None = None
    episode_id: Any = None
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise M0MobileError(f"{path}:{line_number} is invalid JSON: {error}") from error
                record = _mapping(record, f"{path}:{line_number}")
                _validate_source_record(record, config, f"{path}:{line_number}")
                tick = _integer(record.get("observation_model_tick"), "observation_model_tick")
                if previous_tick is not None and tick <= previous_tick:
                    raise M0MobileError(f"{path}:{line_number} model ticks are not strictly increasing")
                current_episode = record.get("source_episode_id")
                if episode_id is None:
                    episode_id = current_episode
                elif current_episode != episode_id:
                    raise M0MobileError(f"{path} contains more than one source episode")
                if anchor is None:
                    anchor = tick
                previous_tick = tick
                if (tick - anchor) % stride == 0:
                    yield record
    except OSError as error:
        raise M0MobileError(f"cannot read temporal export {path}: {error}") from error


def lerobot_frame_from_record(
    record: Mapping[str, Any],
    episode_root: str | Path,
    config: Mapping[str, Any],
    *,
    image_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Map one selected temporal record to the four-video LeRobot row."""

    _validate_source_record(record, config, "temporal record")
    sample = temporal_sample_from_record(record, episode_root)
    load_image = image_loader or _load_rgb
    ordered_paths = (
        sample.camera_paths[0][0],
        sample.camera_paths[0][1],
        sample.camera_paths[1][0],
        sample.camera_paths[1][1],
    )
    frame = {
        key: _rgb_array(load_image(path), key, _video_shape(config, key))
        for key, path in zip(VIDEO_FEATURE_KEYS, ordered_paths, strict=True)
    }
    state = np.asarray(sample.state, dtype=np.float32)
    action = np.asarray(sample.actions, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,) or action.shape != (ACTION_HORIZON * ACTION_DIM,):
        raise M0MobileError("AL0 LeRobot state/action shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise M0MobileError("AL0 LeRobot state/action must be finite")
    frame["observation.state"] = state
    frame["action"] = action
    frame["task"] = sample.instruction
    return frame


def write_lerobot_episodes(
    dataset: Any,
    episode_roots: Iterable[str | Path],
    config: Mapping[str, Any],
    *,
    image_loader: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Populate an already-created LeRobotDataset and return source provenance."""

    _validate_config(config)
    roots = _episode_roots(episode_roots)
    relative_export = Path(_mapping(config["source"], "source")["export_relative_path"])
    episode_reports = []
    total_frames = 0
    for root in roots:
        jsonl_path = (root / relative_export).resolve()
        try:
            jsonl_path.relative_to(root)
        except ValueError as error:
            raise M0MobileError("temporal export path escapes episode root") from error
        frame_count = 0
        first_tick = None
        last_tick = None
        source_episode_id = None
        for record in iter_query_records(jsonl_path, config):
            tick = int(record["observation_model_tick"])
            first_tick = tick if first_tick is None else first_tick
            last_tick = tick
            source_episode_id = record["source_episode_id"]
            dataset.add_frame(
                lerobot_frame_from_record(
                    record,
                    root,
                    config,
                    image_loader=image_loader,
                )
            )
            frame_count += 1
        if frame_count == 0:
            raise M0MobileError(f"temporal episode produced no 5 Hz queries: {jsonl_path}")
        dataset.save_episode()
        total_frames += frame_count
        episode_reports.append(
            {
                "source_episode_id": source_episode_id,
                "source_episode_root": str(root),
                "source_temporal_export": str(jsonl_path),
                "source_temporal_sha256": _sha256(jsonl_path),
                "query_frames": frame_count,
                "first_observation_model_tick": first_tick,
                "last_observation_model_tick": last_tick,
            }
        )
    return {
        "episode_count": len(episode_reports),
        "frame_count": total_frames,
        "episodes": episode_reports,
    }


def materialize_lerobot_v3(
    episode_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_LEROBOT_V3_CONFIG_PATH,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Create an immutable official LeRobot v3 derivative beside the raw data."""

    config_source = Path(config_path).expanduser().resolve()
    config = load_lerobot_v3_config(config_source)
    expected_version = str(_mapping(config["format"], "format")["lerobot_package_version"])
    try:
        installed_version = version("lerobot")
    except PackageNotFoundError as error:
        raise M0MobileError(
            f"lerobot=={expected_version} is required for AL0 conversion"
        ) from error
    if installed_version != expected_version:
        raise M0MobileError(
            f"AL0 conversion requires lerobot=={expected_version}, got {installed_version}"
        )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise M0MobileError("cannot import LeRobotDataset") from error

    roots = _episode_roots(episode_roots)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise M0MobileError(f"LeRobot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    resolved_repo_id = repo_id or str(config["format"]["repo_id"])
    encoding = _mapping(config["encoding"], "encoding")
    sampling = _mapping(config["sampling"], "sampling")
    try:
        dataset = LeRobotDataset.create(
            repo_id=resolved_repo_id,
            root=staging,
            robot_type=str(config["format"]["robot_type"]),
            fps=int(sampling["query_fps"]),
            features=lerobot_features(config),
            use_videos=True,
            vcodec=str(encoding["vcodec"]),
            image_writer_threads=int(encoding["image_writer_threads"]),
            metadata_buffer_size=int(encoding["metadata_buffer_size"]),
            batch_encoding_size=int(encoding["batch_encoding_size"]),
            encoder_threads=int(encoding["encoder_threads"]),
            streaming_encoding=bool(encoding["streaming_encoding"]),
        )
        report = write_lerobot_episodes(dataset, roots, config)
        dataset.finalize()
        loaded = LeRobotDataset(
            repo_id=resolved_repo_id,
            root=staging,
            video_backend="pyav",
        )
        _validate_official_reload(loaded, report, config)
        manifest = {
            "schema_version": "conveyor-vla-al0-lerobot-v3-manifest-1",
            "dataset_version": "v3.0",
            "repo_id": resolved_repo_id,
            "robot_type": config["format"]["robot_type"],
            "lerobot_package_version": installed_version,
            "config_sha256": _sha256(config_source),
            "query_fps": sampling["query_fps"],
            "action_rate_hz": sampling["action_rate_hz"],
            "control_hz": sampling["control_hz"],
            "video_feature_keys": list(VIDEO_FEATURE_KEYS),
            "state_shape": [STATE_DIM],
            "action_storage_shape": [ACTION_HORIZON * ACTION_DIM],
            "action_logical_shape": [ACTION_HORIZON, ACTION_DIM],
            **report,
        }
        manifest_path = staging / "meta" / "conveyorvla_al0_conversion.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return {**manifest, "dataset_root": str(output)}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_official_reload(
    dataset: Any,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if int(dataset.meta.total_episodes) != int(report["episode_count"]):
        raise M0MobileError("official LeRobot reload episode count mismatch")
    if len(dataset) != int(report["frame_count"]):
        raise M0MobileError("official LeRobot reload frame count mismatch")
    if set(dataset.meta.video_keys) != set(VIDEO_FEATURE_KEYS):
        raise M0MobileError("official LeRobot reload video schema mismatch")
    if config["validation"]["require_first_frame_decode"]:
        first = dataset[0]
        if tuple(first["observation.state"].shape) != (STATE_DIM,):
            raise M0MobileError("decoded LeRobot state shape mismatch")
        if tuple(first["action"].shape) != (ACTION_HORIZON * ACTION_DIM,):
            raise M0MobileError("decoded LeRobot action shape mismatch")
        for key in VIDEO_FEATURE_KEYS:
            if tuple(first[key].shape) != (3, 224, 224):
                raise M0MobileError(f"decoded LeRobot video shape mismatch: {key}")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != LEROBOT_V3_CONFIG_SCHEMA_VERSION:
        raise M0MobileError("LeRobot v3 config has the wrong schema_version")
    format_config = _mapping(config.get("format"), "format")
    expected_format = {
        "dataset_version": "v3.0",
        "lerobot_package_version": "0.4.4",
        "robot_type": "go2_x5_mobile_manipulator",
        "use_videos": True,
    }
    for key, expected in expected_format.items():
        if format_config.get(key) != expected:
            raise M0MobileError(f"format.{key} must be {expected!r}")
    if not isinstance(format_config.get("repo_id"), str) or not format_config["repo_id"]:
        raise M0MobileError("format.repo_id must be a non-empty string")
    source = _mapping(config.get("source"), "source")
    expected_source = {
        "profile": TEMPORAL_PROFILE,
        "schema_version": TEMPORAL_SCHEMA_VERSION,
        "required_task_scope": "grasp_only",
        "required_outcome": "success",
        "require_unassisted": True,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise M0MobileError(f"source.{key} must be {expected!r}")
    relative_export = Path(str(source.get("export_relative_path", "")))
    if not relative_export.parts or relative_export.is_absolute() or ".." in relative_export.parts:
        raise M0MobileError("source.export_relative_path must stay inside an episode")
    sampling = _mapping(config.get("sampling"), "sampling")
    expected_sampling = {
        "control_hz": CONTROL_HZ,
        "source_model_hz": MODEL_HZ,
        "query_fps": 5,
        "query_stride_model_ticks": 5,
        "query_anchor": "first_eligible_record",
        "history_offsets_model_ticks": list(HISTORY_OFFSETS_MODEL_TICKS),
        "action_rate_hz": MODEL_HZ,
        "action_horizon": ACTION_HORIZON,
    }
    for key, expected in expected_sampling.items():
        if sampling.get(key) != expected:
            raise M0MobileError(f"sampling.{key} must be {expected!r}")
    if not math.isclose(float(sampling.get("history_span_s", -1.0)), 0.08):
        raise M0MobileError("sampling.history_span_s must be 0.08")
    if not math.isclose(float(sampling.get("action_horizon_s", -1.0)), 0.8):
        raise M0MobileError("sampling.action_horizon_s must be 0.8")
    features = _mapping(config.get("features"), "features")
    videos = [_mapping(value, "video feature") for value in _sequence(features.get("videos"), "features.videos")]
    expected_video_sources = (
        (VIDEO_FEATURE_KEYS[0], CAMERA_IDS[0], HISTORY_OFFSETS_MODEL_TICKS[0]),
        (VIDEO_FEATURE_KEYS[1], CAMERA_IDS[0], HISTORY_OFFSETS_MODEL_TICKS[1]),
        (VIDEO_FEATURE_KEYS[2], CAMERA_IDS[1], HISTORY_OFFSETS_MODEL_TICKS[0]),
        (VIDEO_FEATURE_KEYS[3], CAMERA_IDS[1], HISTORY_OFFSETS_MODEL_TICKS[1]),
    )
    actual_video_sources = tuple(
        (video.get("key"), video.get("source_camera_id"), video.get("history_offset_model_ticks"))
        for video in videos
    )
    if actual_video_sources != expected_video_sources or any(video.get("shape") != [224, 224, 3] for video in videos):
        raise M0MobileError("features.videos must be the four ordered 224x224 AL0 streams")
    state = _mapping(features.get("state"), "features.state")
    if state.get("key") != "observation.state" or state.get("dtype") != "float32" or state.get("shape") != [STATE_DIM]:
        raise M0MobileError("features.state must be observation.state float32[28]")
    state_names = _sequence(state.get("names"), "features.state.names")
    if len(state_names) != STATE_DIM or len(set(state_names)) != STATE_DIM:
        raise M0MobileError("features.state.names must contain 28 unique names")
    action = _mapping(features.get("action"), "features.action")
    if (
        action.get("key") != "action"
        or action.get("dtype") != "float32"
        or action.get("shape") != [ACTION_HORIZON * ACTION_DIM]
        or action.get("logical_shape") != [ACTION_HORIZON, ACTION_DIM]
        or action.get("flatten_order") != "time_major"
        or len(_sequence(action.get("step_names"), "features.action.step_names")) != ACTION_DIM
        or action.get("dimension_mask") != list(ACTION_DIMENSION_MASK)
    ):
        raise M0MobileError("features.action must preserve the AL0 20x10 time-major contract")
    if features.get("observer_camera_is_model_input") is not False or features.get("object_state_is_model_input") is not False:
        raise M0MobileError("observer camera and object state must not be model inputs")
    if features.get("task_key") != "task":
        raise M0MobileError("features.task_key must be 'task'")
    encoding = _mapping(config.get("encoding"), "encoding")
    if encoding.get("vcodec") != "h264" or encoding.get("streaming_encoding") is not False:
        raise M0MobileError("AL0 offline conversion must use non-streaming H.264")
    for key in (
        "image_writer_threads",
        "metadata_buffer_size",
        "batch_encoding_size",
        "encoder_threads",
    ):
        if isinstance(encoding.get(key), bool) or not isinstance(encoding.get(key), int) or encoding[key] <= 0:
            raise M0MobileError(f"encoding.{key} must be a positive integer")
    validation = _mapping(config.get("validation"), "validation")
    validation_keys = {
        "require_official_reload",
        "require_first_frame_decode",
        "require_exact_episode_count",
        "require_exact_frame_count",
        "require_finite_state_and_action",
        "require_all_four_video_features",
        "never_overwrite_output",
    }
    if set(validation) != validation_keys or any(
        validation.get(key) is not True for key in validation_keys
    ):
        raise M0MobileError("all LeRobot v3 validation gates must be enabled")


def _validate_source_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    location: str,
) -> None:
    source = _mapping(config["source"], "source")
    expected = {
        "schema_version": source["schema_version"],
        "profile": source["profile"],
        "policy_task_scope": source["required_task_scope"],
        "source_task_outcome": source["required_outcome"],
        "source_assisted": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise M0MobileError(f"{location} {key} must be {value!r}")
    if not isinstance(record.get("source_episode_id"), str) or not record["source_episode_id"].strip():
        raise M0MobileError(f"{location} source_episode_id must be a non-empty string")
    state_names = config["features"]["state"]["names"]
    state_layout = _sequence(record.get("state_layout"), f"{location} state_layout")
    if list(state_layout) != state_names:
        raise M0MobileError(f"{location} state_layout disagrees with LeRobot config")


def _video_shape(config: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    for video in config["features"]["videos"]:
        if video["key"] == key:
            return tuple(video["shape"])
    raise M0MobileError(f"unknown video feature: {key}")


def _rgb_array(value: Any, key: str, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.uint8:
        raise M0MobileError(f"{key} must be uint8 HWC with shape {shape}, got {array.shape}/{array.dtype}")
    return np.ascontiguousarray(array)


def _load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:
        raise M0MobileError("Pillow is required for LeRobot conversion") from error
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise M0MobileError(f"cannot load temporal image {path}: {error}") from error


def _episode_roots(values: Iterable[str | Path]) -> tuple[Path, ...]:
    roots = tuple(Path(value).expanduser().resolve() for value in values)
    if not roots:
        raise M0MobileError("at least one temporal episode root is required")
    if len(set(roots)) != len(roots):
        raise M0MobileError("temporal episode roots must be unique")
    missing = next((root for root in roots if not root.is_dir()), None)
    if missing is not None:
        raise M0MobileError(f"temporal episode root is not a directory: {missing}")
    return roots


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M0MobileError(f"{name} must be a sequence")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M0MobileError(f"{name} must be a non-negative integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(root: Path) -> Mapping[str, Any]:
    path = root / "meta" / "conveyorvla_al0_conversion.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read AL0 LeRobot manifest {path}: {error}") from error
    if not isinstance(manifest, Mapping):
        raise M0MobileError("AL0 LeRobot manifest must be an object")
    expected = {
        "schema_version": "conveyor-vla-al0-lerobot-v3-manifest-1",
        "dataset_version": "v3.0",
        "lerobot_package_version": "0.4.4",
        "query_fps": 5,
        "action_rate_hz": MODEL_HZ,
        "control_hz": CONTROL_HZ,
        "state_shape": [STATE_DIM],
        "action_storage_shape": [ACTION_HORIZON * ACTION_DIM],
        "action_logical_shape": [ACTION_HORIZON, ACTION_DIM],
        "video_feature_keys": list(VIDEO_FEATURE_KEYS),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise M0MobileError(f"AL0 LeRobot manifest {key} must be {value!r}")
    if not isinstance(manifest.get("repo_id"), str) or not manifest["repo_id"]:
        raise M0MobileError("AL0 LeRobot manifest repo_id is missing")
    if not isinstance(manifest.get("frame_count"), int) or manifest["frame_count"] <= 0:
        raise M0MobileError("AL0 LeRobot manifest frame_count must be positive")
    return manifest


def _state_statistics(stats: Any, *, count: int) -> dict[str, Any]:
    mapping = _mapping(stats, "LeRobot statistics")
    state = _mapping(mapping.get("observation.state"), "LeRobot state statistics")
    mean = _numeric_array(state.get("mean"), "LeRobot state mean").reshape(-1)
    std = _numeric_array(state.get("std"), "LeRobot state std").reshape(-1)
    if mean.shape != (STATE_DIM,) or std.shape != (STATE_DIM,):
        raise M0MobileError("LeRobot state statistics must contain 28 values")
    if np.any(std < 0.0):
        raise M0MobileError("LeRobot state standard deviations must be non-negative")
    return {
        "schema_version": "conveyor-vla-al0-state-statistics-1",
        "split": "train",
        "count": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _numeric_array(value: Any, name: str) -> np.ndarray:
    if all(hasattr(value, attribute) for attribute in ("detach", "cpu", "numpy")):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise M0MobileError(f"{name} must be numeric") from error
    if not np.isfinite(array).all():
        raise M0MobileError(f"{name} must be finite")
    return array


__all__ = [
    "ConveyorVLAAL0LeRobotDataset",
    "DEFAULT_LEROBOT_V3_CONFIG_PATH",
    "LEROBOT_V3_CONFIG_SCHEMA_VERSION",
    "VIDEO_FEATURE_KEYS",
    "iter_query_records",
    "lerobot_features",
    "lerobot_frame_from_record",
    "lerobot_model_example",
    "load_lerobot_v3_config",
    "materialize_lerobot_v3",
    "write_lerobot_episodes",
]
