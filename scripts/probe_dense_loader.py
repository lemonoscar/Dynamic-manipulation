#!/usr/bin/env python3
"""Decode one row per split/phase from a dense-transition Liangzhu view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_m0_mobile_config,
)
from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    ConveyorVLAAL0HierarchicalDataset,
)
from conveyor_bench.conveyorvla.subtasks import PHASE_ORDER  # noqa: E402
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    DEFAULT_TEMPORAL_CONFIG_PATH,
    build_temporal_policy_config,
    load_temporal_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG_PATH)
    args = parser.parse_args(argv)
    config = build_temporal_policy_config(
        load_m0_mobile_config(args.config),
        load_temporal_config(args.temporal_config),
    )
    manifest_path = args.hierarchy_root.expanduser().resolve() / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    action_scale_check = _check_action_scales(config, manifest_payload)
    finalization = manifest_payload.get("finalized_training_contract", {})
    if finalization.get("temporal_config_sha256") != _sha256(
        args.temporal_config.expanduser().resolve()
    ):
        action_scale_check["ok"] = False
        action_scale_check["problems"].append(
            "finalized view is not bound to the selected temporal config"
        )
    if not action_scale_check["ok"]:
        raise RuntimeError(
            "temporal action scale is smaller than the new train-split "
            "recommendation: " + ", ".join(action_scale_check["problems"])
        )
    decoded = []
    for split in ("train", "val", "test"):
        dataset = ConveyorVLAAL0HierarchicalDataset(
            args.hierarchy_root,
            config,
            split=split,
            component="joint",
        )
        for phase in PHASE_ORDER:
            index = next(
                index
                for index, row in enumerate(dataset.annotations)
                if int(row["phase_id"]) == int(phase)
                and row["is_boundary_window"]
            )
            example = dataset[index]
            if (
                "Completed subtasks" in example["lang"]
                or "Previous model prediction" in example["lang"]
            ):
                raise RuntimeError("loader prompt contains externally supplied history")
            if any(
                key in example
                for key in (
                    "subtask_history",
                    "previous_subtask_label",
                    "previous_subtask_text",
                )
            ):
                raise RuntimeError("loader example exposes annotation semantic history")
            if len(example["video"]) != 2 or any(
                len(clip) != 2 for clip in example["video"]
            ):
                raise RuntimeError("loader did not decode two cameras at two times")
            if len(example["action_valid_mask"]) != 20:
                raise RuntimeError("loader action_valid_mask does not cover the horizon")
            decoded.append(
                {
                    "split": split,
                    "phase": phase.name,
                    "sample_id": example["sample_id"],
                    "action_dimension": len(example["action"][0]),
                    "valid_action_steps": sum(example["action_valid_mask"]),
                    "navigation_reference_mode": example["navigation_reference_mode"],
                }
            )
    report = {
        "schema_version": "conveyor-vla-al0-dense-loader-probe-2",
        "ok": True,
        "manifest_sha256": _sha256(manifest_path),
        "action_scale_check": action_scale_check,
        "decoded": decoded,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _check_action_scales(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    scales = [float(value) for value in config["normalization"]["action"]["scale"]]
    recommendations: dict[str, float] = {}
    problems = []
    finalized = manifest.get("finalized_training_contract", {})
    if finalized.get("resolved_training_action_scale") != scales:
        problems.append("finalized manifest action scale differs from temporal config")
    for key in (
        "train_navigation_action_statistics",
        "train_manipulation_action_statistics",
    ):
        statistics = manifest[key]
        indices = statistics["action_indices_in_action10"]
        values = statistics["recommended_physical_scale"]
        for index, recommendation in zip(indices, values, strict=True):
            if recommendation is None:
                continue
            value = float(recommendation)
            recommendations[str(index)] = value
            if scales[int(index)] + 1.0e-12 < value:
                problems.append(
                    f"action[{index}]={scales[int(index)]} < recommended={value}"
                )
    return {
        "ok": not problems,
        "configured_action_scale": scales,
        "recommended_minimum_by_action_index": recommendations,
        "problems": problems,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
