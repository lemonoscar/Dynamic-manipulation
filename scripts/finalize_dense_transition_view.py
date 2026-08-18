#!/usr/bin/env python3
"""Finalize a statistics-probe view after freezing the train-derived config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.hierarchical_data import (  # noqa: E402
    HIERARCHY_VIEW_SCHEMA_VERSION,
    audit_dense_transition_view,
)
from conveyor_bench.conveyorvla.subtasks import (  # noqa: E402
    NAVIGATION_ARM_JOINT_REFERENCES,
    NAVIGATION_GRIPPER_REFERENCES,
    NAVIGATION_REFERENCE_MODES,
    Phase,
)
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    DEFAULT_TEMPORAL_CONFIG_PATH,
    load_temporal_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--temporal-config", type=Path, default=DEFAULT_TEMPORAL_CONFIG_PATH
    )
    args = parser.parse_args(argv)
    source = args.source_view.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    temporal_path = args.temporal_config.expanduser().resolve()
    if source.parent != output.parent:
        raise ValueError("source and finalized views must be sibling directories")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite finalized view: {output}")

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != HIERARCHY_VIEW_SCHEMA_VERSION:
        raise ValueError("source view has the wrong schema version")
    annotations_relative = str(source_manifest["annotations_relative_path"])
    source_annotations = source / annotations_relative
    if _sha256(source_annotations) != source_manifest.get("annotations_sha256"):
        raise ValueError("source annotations checksum is invalid")
    temporal = load_temporal_config(temporal_path)

    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        target_annotations = staging / annotations_relative
        target_annotations.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_annotations, target_annotations)
        manifest = dict(source_manifest)
        manifest["navigation_references"] = {
            phase.name: {
                "mode": NAVIGATION_REFERENCE_MODES[phase],
                "arm_joint_reference": list(NAVIGATION_ARM_JOINT_REFERENCES[phase]),
                "gripper_open_fraction": NAVIGATION_GRIPPER_REFERENCES[phase],
                "tcp_delta_used": False,
            }
            for phase in (Phase.NAV_TO_SOURCE, Phase.NAV_TO_TARGET)
        }
        manifest["finalized_training_contract"] = {
            "schema_version": "conveyor-vla-al0-dense-view-finalization-1",
            "source_view_manifest_sha256": _sha256(source_manifest_path),
            "annotations_copied_without_reselection": True,
            "resolved_training_action_scale": list(
                temporal["normalization"]["action"]["scale"]
            ),
            "temporal_config_sha256": _sha256(temporal_path),
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit = audit_dense_transition_view(staging)
        if audit.get("ok") is not True:
            raise RuntimeError("finalized view audit failed: " + "; ".join(audit["problems"]))
        os.replace(staging, output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    report = {
        "ok": True,
        "dataset_root": str(output),
        "schema_version": manifest["schema_version"],
        "manifest_sha256": _sha256(output / "manifest.json"),
        "annotations_sha256": _sha256(output / annotations_relative),
        "source_view_manifest_sha256": _sha256(source_manifest_path),
        "resolved_training_action_scale": manifest["finalized_training_contract"][
            "resolved_training_action_scale"
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
