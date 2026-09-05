"""Auditable loading and source identity for formal joint-trajectory evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .joint_trajectory import MODEL_CONTRACT_ID, DATASET_SCHEMA_VERSION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity(root: Path, *, scope: str = "all") -> dict:
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").glob("*.py")])
    if scope == "open_loop":
        files = [p for p in files if ("src" in p.relative_to(root).parts and p.name != "formal_physics.py")
                 or p.name.startswith("train_") or p.name == "evaluate_joint_trajectory_formal.py"]
    hashes = {str(path.relative_to(root)): sha256(path) for path in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "scope": scope, "files": hashes}


def validate_formal_checkpoint(checkpoint: Path, config_path: Path) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    run = checkpoint.parent.parent
    manifest = read_json(checkpoint / "joint_trajectory_checkpoint_manifest.json")
    resolved = read_json(run / "resolved_run.json")
    step = manifest.get("global_step")
    if not isinstance(step, int) or isinstance(step, bool) or checkpoint.name != f"step_{step:06d}":
        raise ValueError("checkpoint directory/step binding differs")
    keys = ("run_kind", "model_contract_id", "dataset_schema_version", "dataset_manifest_sha256",
            "normalization_sha256", "normalizer_id", "policy_config_sha256", "stage_a_steps", "max_steps")
    for key in keys:
        if manifest.get(key) is None or manifest[key] != resolved.get(key):
            raise ValueError(f"checkpoint/resolved binding differs: {key}")
    if manifest["run_kind"] != "formal" or step != resolved["max_steps"]:
        raise ValueError("formal evaluation is fixed to the final checkpoint")
    if manifest["model_contract_id"] != MODEL_CONTRACT_ID or manifest["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported formal model/data contract")
    checksums = {}
    for line in (run / "CHECKSUMS.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest
    # Validate the saved run metadata before using the paths and hashes it supplies.
    for path in (run / "resolved_run.json", run / "resolved_policy_config.json", run / "source.patch",
                 checkpoint / "joint_trajectory_checkpoint_manifest.json", checkpoint / "model.safetensors"):
        relative = str(path.relative_to(run))
        if checksums.get(relative) != sha256(path):
            raise ValueError(f"saved checksum mismatch: {relative}")
    config = read_json(config_path)
    if sha256(config_path) != manifest["policy_config_sha256"]:
        raise ValueError("original policy config hash differs")
    if config != resolved["resolved_policy_config"] or config != read_json(run / "resolved_policy_config.json"):
        raise ValueError("resolved policy config content differs")
    from .joint_trajectory_training import validate_joint_trajectory_config
    validate_joint_trajectory_config(config)
    root = Path(resolved["dataset_root"]).resolve()
    if sha256(root / "manifest.json") != manifest["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash differs")
    dataset = read_json(root / "manifest.json")
    normalizer_path = (root / dataset["normalization_relative_path"]).resolve()
    if not normalizer_path.is_relative_to(root) or sha256(normalizer_path) != manifest["normalization_sha256"]:
        raise ValueError("normalization hash differs")
    from .joint_trajectory_data import JointTrajectoryNormalizer
    normalizer = JointTrajectoryNormalizer.from_path(normalizer_path)
    if normalizer.payload["normalizer_id"] != manifest["normalizer_id"] or dataset["normalizer_id"] != manifest["normalizer_id"]:
        raise ValueError("normalizer identity differs")
    return {"checkpoint": str(checkpoint), "checkpoint_id": checkpoint.name,
            **manifest, "weights_sha256": checksums[str((checkpoint / "model.safetensors").relative_to(run))],
            "resolved_run_sha256": checksums["resolved_run.json"], "resolved": resolved,
            "dataset_root": str(root), "config": config}


def load_formal_policy(binding: dict, model_root: Path, attention: str = "sdpa", device: str = "cuda:0"):
    import torch
    from safetensors.torch import load_model
    from scripts import train_joint_trajectory as training
    policy, token_ids = training._build_model(binding["config"], model_root.resolve(), attention)
    if dict(token_ids) != binding["resolved"]["special_token_ids"]:
        raise ValueError("processor special token IDs differ")
    missing, unexpected = load_model(policy, str(Path(binding["checkpoint"]) / "model.safetensors"), strict=True)
    if missing or unexpected:
        raise ValueError(f"strict load failed: {missing}, {unexpected}")
    policy.to(device=torch.device(device), dtype=torch.bfloat16).eval()
    return policy


def public_identity(binding: dict) -> dict:
    return {key: value for key, value in binding.items() if key not in {"resolved", "config"}}
