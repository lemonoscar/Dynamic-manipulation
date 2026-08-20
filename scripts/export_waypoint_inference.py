#!/usr/bin/env python3
"""Consolidate a bound Waypoint v1 ZeRO checkpoint for single-GPU inference."""

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


EXPORT_SCHEMA = "conveyorvla-waypoint-inference-export-v1"


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
            str(weights),
            "--safe_serialization",
            "--max_shard_size",
            str(args.max_shard_size),
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )
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
        "schema_version": EXPORT_SCHEMA,
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
