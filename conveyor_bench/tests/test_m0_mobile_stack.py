from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conveyor_bench.m0_mobile import (
    CANONICAL_MODEL_ROOT_ENV,
    LEGACY_MODEL_ROOT_ENV,
    MODEL_NAME,
    M0MobileError,
    M0MobileNormalizer,
    audit_model_artifacts,
    load_m0_mobile_config,
    resolve_model_root,
    sample_from_record,
)


def _normalizer(config):
    return M0MobileNormalizer.from_config(
        config,
        {"mean": [0.0] * 28, "std": [1.0] * 28},
    )


def _record() -> dict:
    return {
        "schema_version": "conveyor-bench-m0-mobile-v1",
        "profile": "m0_mobile_v1",
        "sample_id": "episode:sim-step-16",
        "instruction": "pick the moving part",
        "policy_camera_frames": [
            {
                "camera_id": "head_rgb",
                "relative_path": "cameras/head_rgb/000001.png",
            },
            {
                "camera_id": "wrist_rgb",
                "relative_path": "cameras/wrist_rgb/000001.png",
            },
        ],
        "state28": [0.25] * 28,
        "model_action10_chunk": [[0.01] * 9 + [1.0] for _ in range(16)],
        "action_dimension_mask": [
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
        ],
        "action_horizon": 16,
        "action_rate_hz": 50,
        "causal_offset_control_steps": 1,
        "overview_rgb": "must-not-leak",
        "current_target_id": "supervision-only",
        "future_object_states": [{"must": "not leak"}],
    }


def test_default_m0_mobile_config_freezes_go2_x5_contract() -> None:
    config = load_m0_mobile_config()

    assert config["model_identity"]["name"] == MODEL_NAME
    assert config["model_root_env"] == CANONICAL_MODEL_ROOT_ENV
    assert LEGACY_MODEL_ROOT_ENV in config["legacy_model_root_envs"]
    assert config["action_model"]["state_dim"] == 28
    assert config["action_model"]["action_dim"] == 10
    assert config["action_model"]["action_horizon"] == 16
    assert config["data"]["camera_order"] == ["head_rgb", "wrist_rgb"]
    assert config["spatial_model"]["enabled"] is False


def test_model_root_prefers_canonical_env_and_supports_legacy_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_m0_mobile_config()
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    canonical.mkdir()
    legacy.mkdir()
    monkeypatch.delenv(CANONICAL_MODEL_ROOT_ENV, raising=False)
    monkeypatch.setenv(LEGACY_MODEL_ROOT_ENV, str(legacy))

    assert resolve_model_root(config) == legacy.resolve()

    monkeypatch.setenv(CANONICAL_MODEL_ROOT_ENV, str(canonical))
    assert resolve_model_root(config) == canonical.resolve()


def test_sample_adapter_exposes_only_policy_inputs(tmp_path: Path) -> None:
    config = load_m0_mobile_config()
    for camera in ("head_rgb", "wrist_rgb"):
        path = tmp_path / "cameras" / camera / "000001.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"png")

    sample = sample_from_record(_record(), tmp_path, config)
    example = sample.as_model_example(_normalizer(config), lambda path: path.name)

    assert set(example) == {"image", "lang", "state", "action", "action_mask"}
    assert example["image"] == ["000001.png", "000001.png"]
    assert len(example["state"]) == 1
    assert len(example["state"][0]) == 28
    assert len(example["action"]) == 16
    assert len(example["action"][0]) == 10
    assert example["action"][0][1] == 0.0


def test_action_normalization_round_trip_and_safety_projection() -> None:
    config = load_m0_mobile_config()
    normalizer = _normalizer(config)
    action = (0.15, 0.7, -0.175, 0.0125, -0.0125, 0.0, 0.06, -0.06, 0.0, 1.0)

    normalized = normalizer.normalize_action(action)
    restored = normalizer.denormalize_action(normalized)

    assert normalized == pytest.approx(
        (0.5, 0.0, -0.5, 0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 1.0)
    )
    assert restored == pytest.approx(
        (0.15, 0.0, -0.175, 0.0125, -0.0125, 0.0, 0.06, -0.06, 0.0, 1.0)
    )


def test_artifact_audit_checks_size_hash_and_root_boundary(tmp_path: Path) -> None:
    payload = b"model"
    model_file = tmp_path / "weights.bin"
    model_file.write_bytes(payload)
    config = {
        "artifacts": [
            {
                "id": "tiny",
                "files": [
                    {
                        "path": "weights.bin",
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ]
    }

    checks = audit_model_artifacts(config, tmp_path, verify_hashes=True)

    assert len(checks) == 1
    assert checks[0].artifact_id == "tiny"
    assert checks[0].sha256 == hashlib.sha256(payload).hexdigest()

    config["artifacts"][0]["files"][0]["path"] = "../weights.bin"
    with pytest.raises(M0MobileError, match="escapes model root"):
        audit_model_artifacts(config, tmp_path)


def test_sample_adapter_rejects_reversed_camera_order(tmp_path: Path) -> None:
    config = load_m0_mobile_config()
    record = _record()
    record["policy_camera_frames"].reverse()

    with pytest.raises(M0MobileError, match="camera order"):
        sample_from_record(record, tmp_path, config, require_images=False)
