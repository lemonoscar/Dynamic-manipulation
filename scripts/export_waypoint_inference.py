#!/usr/bin/env python3
"""Consolidate a bound Waypoint v1/v2 ZeRO checkpoint for CUDA inference."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import check_waypoint_checkpoint as checkpoint_gate  # noqa: E402
from scripts import train_waypoint as training  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_v2 import MODEL_CONTRACT_ID_V2  # noqa: E402


EXPORT_SCHEMA = "conveyorvla-waypoint-inference-export-v1"
EXPORT_SCHEMA_V2 = "conveyorvla-waypoint-inference-export-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest, resolved, dataset_root = checkpoint_gate._validate_binding(checkpoint)
    output = args.output_dir.expanduser().resolve()
    _reserve_export(output)
    weights = output / "weights"
    pytorch_weights = output / ".pytorch-weights"
    converter = checkpoint / "zero_to_fp32.py"
    if not converter.is_file():
        raise M0MobileError("checkpoint has no generated ZeRO consolidation script")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(checkpoint),
            str(pytorch_weights),
            "--max_shard_size",
            str(args.max_shard_size),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )
    _convert_pytorch_shards_to_safetensors(pytorch_weights, weights)
    shutil.rmtree(pytorch_weights)
    weight_files = _safe_weight_identity(weights)
    config_source = Path(str(resolved["config"])).resolve()
    normalization = checkpoint_gate._mapping(
        manifest["normalization"], "normalization binding"
    )
    normalization_source = dataset_root / str(normalization["relative_path"])
    processor_source = (
        checkpoint / str(manifest["processor_relative_path"])
    ).resolve()
    shutil.copy2(config_source, output / "policy_config.json")
    shutil.copy2(normalization_source, output / "normalization.json")
    shutil.copy2(
        checkpoint / "waypoint_checkpoint_manifest.json",
        output / "source_checkpoint_manifest.json",
    )
    shutil.copy2(
        checkpoint.parents[1] / "resolved_run.json",
        output / "source_resolved_run.json",
    )
    shutil.copytree(processor_source, output / "processor")
    export = {
        "schema_version": (
            EXPORT_SCHEMA_V2
            if manifest["model_contract_id"] == MODEL_CONTRACT_ID_V2
            else EXPORT_SCHEMA
        ),
        "status": "complete",
        "source_checkpoint": str(checkpoint),
        "global_step": int(manifest["global_step"]),
        "model_contract_id": manifest["model_contract_id"],
        "source_git": manifest["source_git"],
        "special_token_ids": manifest["special_token_ids"],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "normalization_sha256": manifest["normalization_sha256"],
        "camera_contract": manifest["camera_contract"],
        "dataset_action_contract": manifest["dataset_action_contract"],
        "dataset_transition_contract": manifest.get("dataset_transition_contract"),
        "dataset_crl_contract": manifest.get("dataset_crl_contract"),
        "auxiliary_contract": manifest.get("auxiliary_contract"),
        "loss_contract": manifest.get("loss_contract"),
        "qwen_base": manifest["qwen_base"],
        "model_root": resolved["model_root"],
        "attention_implementation": checkpoint_gate._mapping(
            resolved["arguments"], "resolved arguments"
        )["attention_implementation"],
        "source_checkpoint_manifest_sha256": training.common._sha256(
            checkpoint / "waypoint_checkpoint_manifest.json"
        ),
        "source_resolved_run_sha256": training.common._sha256(
            checkpoint.parents[1] / "resolved_run.json"
        ),
        "policy_config_sha256": training.common._sha256(
            output / "policy_config.json"
        ),
        "processor_files": _directory_identity(output / "processor"),
        "weights": {
            "format": "safetensors-fp32-zero3-consolidated",
            "relative_path": "weights",
            "files": weight_files,
        },
    }
    training.common._write_json_atomic(output / "inference_manifest.json", export)
    print(
        json.dumps(
            {
                "status": "pass",
                "output_dir": str(output),
                "global_step": export["global_step"],
                "weight_files": len(weight_files),
                "weight_bytes": sum(item["size"] for item in weight_files.values()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _reserve_export(output: Path) -> None:
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise M0MobileError("inference exports must stay outside the Git worktree")
    if output.exists():
        raise M0MobileError(f"inference export output already exists: {output}")
    output.mkdir(parents=True)


def _safe_weight_identity(root: Path) -> dict[str, dict[str, Any]]:
    index = root / "model.safetensors.index.json"
    shards = sorted(root.glob("model-*.safetensors"))
    if not index.is_file() or not shards:
        raise M0MobileError("ZeRO consolidation did not produce sharded safetensors")
    paths = [index, *shards]
    if any(path.stat().st_size <= 0 for path in paths):
        raise M0MobileError("ZeRO consolidation produced an empty weight file")
    return {
        path.name: {
            "size": path.stat().st_size,
            "sha256": training.common._sha256(path),
        }
        for path in paths
    }


def _convert_pytorch_shards_to_safetensors(source: Path, destination: Path) -> None:
    import torch
    from safetensors.torch import save_file

    index_path = source / "pytorch_model.bin.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise M0MobileError("ZeRO consolidation did not produce a valid PyTorch index") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise M0MobileError("ZeRO PyTorch weight map is empty")

    destination.mkdir()
    safe_weight_map: dict[str, str] = {}
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str) or not shard_name.endswith(".bin"):
            raise M0MobileError("ZeRO PyTorch shard name is invalid")
        shard_path = source / shard_name
        expected = {name for name, value in weight_map.items() if value == shard_name}
        state = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or set(state) != expected:
            raise M0MobileError(f"ZeRO PyTorch shard index mismatch: {shard_name}")
        safe_state = {}
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise M0MobileError(f"ZeRO weight is not a tensor: {name}")
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                raise M0MobileError(f"ZeRO consolidated weight is not fp32: {name}")
            # Qwen ties embed_tokens and lm_head. Clone every tensor so the
            # safetensors writer receives independent, contiguous storage.
            safe_state[name] = tensor.detach().contiguous().clone()
        safe_name = shard_name.replace("pytorch_model", "model", 1).replace(
            ".bin", ".safetensors"
        )
        save_file(safe_state, destination / safe_name, metadata={"format": "pt"})
        safe_weight_map.update({name: safe_name for name in expected})
        del state, safe_state

    training.common._write_json_atomic(
        destination / "model.safetensors.index.json",
        {
            "metadata": index.get("metadata", {}),
            "weight_map": safe_weight_map,
        },
    )


def _directory_identity(root: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise M0MobileError(f"artifact directory is empty: {root}")
    return {
        path.relative_to(root).as_posix(): {
            "size": path.stat().st_size,
            "sha256": training.common._sha256(path),
        }
        for path in paths
    }


if __name__ == "__main__":
    raise SystemExit(main())
