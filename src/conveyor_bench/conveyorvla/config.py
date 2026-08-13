"""Pure-Python contracts for the ConveyorVLA AL0 compatibility stack."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


CONFIG_SCHEMA_VERSION = "conveyor-bench-m0-mobile-train-config-1"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/model.json"
MODEL_FAMILY = "ConveyorVLA"
MODEL_VARIANT = "AL0"
MODEL_NAME = f"{MODEL_FAMILY} {MODEL_VARIANT}"
MODEL_SLUG = "conveyorvla_al0"
CANONICAL_MODEL_ROOT_ENV = "CONVEYORVLA_AL0_MODEL_ROOT"
LEGACY_MODEL_ROOT_ENV = "DYNAMIC_M0_MODEL_ROOT"


class M0MobileError(ValueError):
    """Raised when a model artifact or exported training sample is invalid."""


@dataclass(frozen=True)
class ArtifactCheck:
    artifact_id: str
    path: Path
    size: int
    sha256: str | None


@dataclass(frozen=True)
class M0MobileSample:
    sample_id: str
    instruction: str
    image_paths: tuple[Path, Path]
    state: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    action_mask: tuple[bool, ...]

    def as_model_example(
        self,
        normalizer: M0MobileNormalizer,
        image_loader: Callable[[Path], Any] | None = None,
    ) -> dict[str, Any]:
        """Return policy fields in the frozen checkpoint-compatible convention."""

        images = [
            image_loader(path) if image_loader is not None else path
            for path in self.image_paths
        ]
        return {
            "image": images,
            "lang": self.instruction,
            "state": (normalizer.normalize_state(self.state),),
            "action": tuple(
                normalizer.normalize_action(action) for action in self.actions
            ),
            "action_mask": self.action_mask,
        }


@dataclass(frozen=True)
class M0MobileNormalizer:
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    state_std_floor: float
    state_clip: float
    action_scale: tuple[float, ...]
    action_clip: tuple[float, float]
    hard_zero_indices: frozenset[int]
    passthrough_indices: frozenset[int]
    gripper_range: tuple[float, float]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        state_statistics: Mapping[str, Any],
    ) -> M0MobileNormalizer:
        data = _mapping(config.get("data"), "config.data")
        normalization = _mapping(config.get("normalization"), "normalization")
        state_cfg = _mapping(normalization.get("state"), "normalization.state")
        action_cfg = _mapping(normalization.get("action"), "normalization.action")
        state_dim = _positive_integer(data.get("state_dim"), "data.state_dim")
        action_dim = _positive_integer(data.get("action_dim"), "data.action_dim")
        mean = _finite_vector(state_statistics.get("mean"), state_dim, "state mean")
        std = _finite_vector(state_statistics.get("std"), state_dim, "state std")
        if any(value < 0.0 for value in std):
            raise M0MobileError("state standard deviations must be non-negative")
        scale = _finite_vector(action_cfg.get("scale"), action_dim, "action scale")
        if any(value <= 0.0 for value in scale):
            raise M0MobileError("action scales must be positive")
        clip = _finite_vector(action_cfg.get("clip_range"), 2, "action clip")
        gripper_range = _finite_vector(
            action_cfg.get("gripper_range"), 2, "gripper range"
        )
        if clip[0] >= clip[1] or gripper_range[0] >= gripper_range[1]:
            raise M0MobileError("normalization ranges must be strictly increasing")
        hard_zero = _index_set(
            action_cfg.get("hard_zero_indices"), action_dim, "hard zero indices"
        )
        passthrough = _index_set(
            action_cfg.get("passthrough_indices"), action_dim, "passthrough indices"
        )
        if hard_zero & passthrough:
            raise M0MobileError("hard-zero and passthrough action indices overlap")
        return cls(
            state_mean=mean,
            state_std=std,
            state_std_floor=_positive_number(
                state_cfg.get("standard_deviation_floor"), "state std floor"
            ),
            state_clip=_positive_number(
                state_cfg.get("clip_standard_deviations"), "state clip"
            ),
            action_scale=scale,
            action_clip=(clip[0], clip[1]),
            hard_zero_indices=hard_zero,
            passthrough_indices=passthrough,
            gripper_range=(gripper_range[0], gripper_range[1]),
        )

    def normalize_state(self, state: Sequence[float]) -> tuple[float, ...]:
        values = _finite_vector(state, len(self.state_mean), "state")
        return tuple(
            min(
                self.state_clip,
                max(
                    -self.state_clip,
                    (value - mean) / max(std, self.state_std_floor),
                ),
            )
            for value, mean, std in zip(
                values, self.state_mean, self.state_std, strict=True
            )
        )

    def normalize_action(self, action: Sequence[float]) -> tuple[float, ...]:
        values = _finite_vector(action, len(self.action_scale), "action")
        low, high = self.action_clip
        result: list[float] = []
        for index, (value, scale) in enumerate(
            zip(values, self.action_scale, strict=True)
        ):
            if index in self.hard_zero_indices:
                result.append(0.0)
            elif index in self.passthrough_indices:
                result.append(
                    min(self.gripper_range[1], max(self.gripper_range[0], value))
                )
            else:
                result.append(min(high, max(low, value / scale)))
        return tuple(result)

    def denormalize_action(self, action: Sequence[float]) -> tuple[float, ...]:
        values = _finite_vector(action, len(self.action_scale), "action")
        low, high = self.action_clip
        result: list[float] = []
        for index, (value, scale) in enumerate(
            zip(values, self.action_scale, strict=True)
        ):
            if index in self.hard_zero_indices:
                result.append(0.0)
            elif index in self.passthrough_indices:
                result.append(
                    min(self.gripper_range[1], max(self.gripper_range[0], value))
                )
            else:
                result.append(min(high, max(low, value)) * scale)
        return tuple(result)


def load_m0_mobile_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(
            f"cannot read {MODEL_NAME} config {config_path}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise M0MobileError(f"{MODEL_NAME} config must be a JSON object")
    _validate_config(config)
    return config


def resolve_model_root(
    config: Mapping[str, Any],
    explicit_root: str | Path | None = None,
) -> Path:
    raw_root = explicit_root
    if raw_root is None:
        variable = config.get("model_root_env")
        if not isinstance(variable, str) or not variable:
            raise M0MobileError("model_root_env must be a non-empty string")
        variables = [variable]
        legacy_variables = config.get("legacy_model_root_envs", ())
        if isinstance(legacy_variables, Sequence) and not isinstance(
            legacy_variables, (str, bytes)
        ):
            variables.extend(
                item
                for item in legacy_variables
                if isinstance(item, str) and item and item not in variables
            )
        raw_root = next(
            (os.environ[name] for name in variables if os.environ.get(name)),
            None,
        )
        if not raw_root:
            raise M0MobileError(
                "pass --model-root or set one of: " + ", ".join(variables)
            )
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise M0MobileError(f"model root is not a directory: {root}")
    return root


def audit_model_artifacts(
    config: Mapping[str, Any],
    model_root: str | Path,
    *,
    verify_hashes: bool = False,
) -> tuple[ArtifactCheck, ...]:
    root = Path(model_root).expanduser().resolve()
    if not root.is_dir():
        raise M0MobileError(f"model root is not a directory: {root}")
    checks: list[ArtifactCheck] = []
    for artifact in _sequence(config.get("artifacts"), "artifacts"):
        artifact = _mapping(artifact, "artifact")
        artifact_id = _nonempty_string(artifact.get("id"), "artifact.id")
        for entry in _sequence(artifact.get("files"), f"{artifact_id}.files"):
            entry = _mapping(entry, f"{artifact_id}.file")
            relative = Path(_nonempty_string(entry.get("path"), "artifact file path"))
            path = (root / relative).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise M0MobileError(f"artifact escapes model root: {relative}") from error
            if not path.is_file():
                raise M0MobileError(f"missing model artifact: {path}")
            expected_size = _positive_integer(entry.get("size"), f"{relative}.size")
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                raise M0MobileError(
                    f"model artifact size mismatch for {path}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            expected_hash = entry.get("sha256")
            if expected_hash is not None:
                expected_hash = _sha256_string(expected_hash, f"{relative}.sha256")
            actual_hash = _sha256(path) if verify_hashes and expected_hash else None
            if actual_hash is not None and actual_hash != expected_hash:
                raise M0MobileError(f"model artifact SHA-256 mismatch: {path}")
            checks.append(
                ArtifactCheck(artifact_id, path, actual_size, actual_hash)
            )
    return tuple(checks)


def iter_m0_mobile_samples(
    jsonl_path: str | Path,
    episode_root: str | Path,
    config: Mapping[str, Any],
    *,
    require_images: bool = True,
) -> Iterator[M0MobileSample]:
    path = Path(jsonl_path)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise M0MobileError(
                        f"{path}:{line_number} is not valid JSON: {error}"
                    ) from error
                yield sample_from_record(
                    _mapping(record, f"{path}:{line_number}"),
                    episode_root,
                    config,
                    require_images=require_images,
                )
    except OSError as error:
        raise M0MobileError(f"cannot read {path}: {error}") from error


def sample_from_record(
    record: Mapping[str, Any],
    episode_root: str | Path,
    config: Mapping[str, Any],
    *,
    require_images: bool = True,
) -> M0MobileSample:
    data = _mapping(config.get("data"), "config.data")
    accepted = tuple(
        _nonempty_string(value, "accepted schema version")
        for value in _sequence(
            data.get("accepted_schema_versions"), "accepted_schema_versions"
        )
    )
    if record.get("schema_version") not in accepted:
        raise M0MobileError(
            f"unsupported AL0 legacy schema: {record.get('schema_version')!r}"
        )
    if record.get("profile") != data.get("profile"):
        raise M0MobileError(f"record profile must be {data.get('profile')!r}")

    sample_id = _nonempty_string(record.get("sample_id"), "sample_id")
    instruction = _nonempty_string(record.get("instruction"), "instruction")
    expected_cameras = tuple(
        _nonempty_string(value, "camera_order")
        for value in _sequence(data.get("camera_order"), "camera_order")
    )
    frames = _sequence(record.get("policy_camera_frames"), "policy_camera_frames")
    if len(frames) != len(expected_cameras):
        raise M0MobileError("policy_camera_frames has the wrong length")
    root = Path(episode_root).expanduser().resolve()
    image_paths: list[Path] = []
    for expected_camera, raw_frame in zip(expected_cameras, frames, strict=True):
        frame = _mapping(raw_frame, "policy camera frame")
        if frame.get("camera_id") != expected_camera:
            raise M0MobileError(
                f"policy camera order must be {expected_cameras}; "
                f"got {frame.get('camera_id')!r}"
            )
        relative = Path(
            _nonempty_string(frame.get("relative_path"), "camera relative_path")
        )
        image_path = (root / relative).resolve()
        try:
            image_path.relative_to(root)
        except ValueError as error:
            raise M0MobileError(f"camera path escapes episode root: {relative}") from error
        if require_images and not image_path.is_file():
            raise M0MobileError(f"missing policy image: {image_path}")
        image_paths.append(image_path)

    state_dim = _positive_integer(data.get("state_dim"), "data.state_dim")
    state = _finite_vector(record.get("state28"), state_dim, "state28")
    action_dim = _positive_integer(data.get("action_dim"), "data.action_dim")
    action_horizon = _positive_integer(
        data.get("action_horizon"), "data.action_horizon"
    )
    raw_actions = _sequence(record.get("model_action10_chunk"), "model actions")
    if len(raw_actions) != action_horizon:
        raise M0MobileError("model_action10_chunk has the wrong horizon")
    actions = tuple(
        _finite_vector(action, action_dim, "model action") for action in raw_actions
    )
    expected_mask = tuple(
        _boolean(value, "action_dimension_mask")
        for value in _sequence(
            data.get("action_dimension_mask"), "data.action_dimension_mask"
        )
    )
    actual_mask = tuple(
        _boolean(value, "action_dimension_mask")
        for value in _sequence(
            record.get("action_dimension_mask"), "record.action_dimension_mask"
        )
    )
    if actual_mask != expected_mask:
        raise M0MobileError("record action_dimension_mask disagrees with config")
    if record.get("action_horizon") != action_horizon:
        raise M0MobileError("record action_horizon disagrees with config")
    if record.get("action_rate_hz") != data.get("action_rate_hz"):
        raise M0MobileError("record action_rate_hz disagrees with config")
    if record.get("causal_offset_control_steps") != data.get(
        "causal_offset_control_steps"
    ):
        raise M0MobileError("record is not causally aligned")
    return M0MobileSample(
        sample_id=sample_id,
        instruction=instruction,
        image_paths=(image_paths[0], image_paths[1]),
        state=state,
        actions=actions,
        action_mask=actual_mask,
    )


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise M0MobileError(f"config schema must be {CONFIG_SCHEMA_VERSION!r}")
    identity = config.get("model_identity")
    if identity is not None:
        identity = _mapping(identity, "config.model_identity")
        expected_identity = {
            "family": MODEL_FAMILY,
            "variant": MODEL_VARIANT,
            "name": MODEL_NAME,
        }
        if dict(identity) != expected_identity:
            raise M0MobileError(
                f"model_identity must identify {MODEL_NAME!r} exactly"
            )
    data = _mapping(config.get("data"), "config.data")
    action = _mapping(config.get("action_model"), "config.action_model")
    for name in ("state_dim", "action_dim", "action_horizon"):
        if _positive_integer(data.get(name), f"data.{name}") != _positive_integer(
            action.get(name), f"action_model.{name}"
        ):
            raise M0MobileError(f"data.{name} and action_model.{name} disagree")
    cameras = tuple(_sequence(data.get("camera_order"), "data.camera_order"))
    if cameras != ("head_rgb", "wrist_rgb"):
        raise M0MobileError("camera_order must be head_rgb then wrist_rgb")
    mask = tuple(_sequence(data.get("action_dimension_mask"), "action mask"))
    if len(mask) != action["action_dim"] or any(
        not isinstance(value, bool) for value in mask
    ):
        raise M0MobileError("action_dimension_mask is invalid")
    if mask[1] or not all(mask[index] for index in range(len(mask)) if index != 1):
        raise M0MobileError("only base_vy may be masked for the AL0 legacy profile")
    if data.get("observer_camera_allowed") is not False:
        raise M0MobileError("observer camera must not be a model input")
    if data.get("supervision_only_fields_are_model_inputs") is not False:
        raise M0MobileError("supervision-only fields must not be model inputs")
    transfer = _mapping(config.get("checkpoint_transfer"), "checkpoint_transfer")
    expected_reinitialized = {
        "state_encoder.layer1.weight",
        "state_encoder.layer1.bias",
        "action_encoder.layer1.weight",
        "action_encoder.layer1.bias",
        "action_decoder.layer2.weight",
        "action_decoder.layer2.bias",
    }
    if set(_sequence(transfer.get("reinitialize_action_keys"), "reinitialize keys")) != expected_reinitialized:
        raise M0MobileError("checkpoint boundary-layer reinitialization set is invalid")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M0MobileError(f"{name} must be a sequence")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise M0MobileError(f"{name} must be a non-empty string")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M0MobileError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise M0MobileError(f"{name} must be a positive finite number")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise M0MobileError(f"{name} must be a boolean")
    return value


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    sequence = _sequence(value, name)
    if len(sequence) != size:
        raise M0MobileError(f"{name} must contain exactly {size} values")
    result: list[float] = []
    for component in sequence:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise M0MobileError(f"{name} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise M0MobileError(f"{name} must contain only finite numbers")
        result.append(number)
    return tuple(result)


def _sha256_string(value: Any, name: str) -> str:
    value = _nonempty_string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise M0MobileError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_set(value: Any, size: int, name: str) -> frozenset[int]:
    indices = _sequence(value, name)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= size
        for index in indices
    ):
        raise M0MobileError(f"{name} contains an invalid index")
    if len(indices) != len(set(indices)):
        raise M0MobileError(f"{name} contains duplicate indices")
    return frozenset(indices)


__all__ = [
    "CANONICAL_MODEL_ROOT_ENV",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "LEGACY_MODEL_ROOT_ENV",
    "MODEL_FAMILY",
    "MODEL_NAME",
    "MODEL_SLUG",
    "MODEL_VARIANT",
    "ArtifactCheck",
    "M0MobileError",
    "M0MobileNormalizer",
    "M0MobileSample",
    "audit_model_artifacts",
    "iter_m0_mobile_samples",
    "load_m0_mobile_config",
    "resolve_model_root",
    "sample_from_record",
]
