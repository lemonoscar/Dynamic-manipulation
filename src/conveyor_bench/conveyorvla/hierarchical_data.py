"""Audit PCT demonstrations against the hierarchical AL0 data contract."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from conveyor_bench.conveyorvla.config import M0MobileError, M0MobileNormalizer
from conveyor_bench.conveyorvla.lerobot_v3 import (
    ACTION_DIM,
    ACTION_HORIZON,
    VIDEO_FEATURE_KEYS,
    lerobot_model_example,
    load_lerobot_v3_config,
)
from conveyor_bench.conveyorvla.pct_dataset import (
    audit_pct_episode,
    iter_pct_temporal_records,
)
from conveyor_bench.conveyorvla.subtasks import (
    FULL_INSTRUCTION,
    NAVIGATION_ARM_JOINT_REFERENCES,
    NAVIGATION_GRIPPER_REFERENCES,
    NAVIGATION_REFERENCE_MODES,
    PCT_PHASES,
    PHASE_ORDER,
    ActionDomain,
    Phase,
    action_domain,
    phase_instruction,
    subtask_prompt,
    subtask_solution,
    project_action10,
)


@dataclass(frozen=True)
class HierarchyAuditThresholds:
    """Conservative physical invariants, evaluated at the recorded 5 Hz rate."""

    nav_tcp_position_drift_m: float = 0.08
    nav_tcp_orientation_drift_rad: float = 0.35
    nav_gripper_range_m: float = 0.01
    manipulation_base_command: float = 0.02
    manipulation_base_position_drift_m: float = 0.05
    manipulation_base_yaw_drift_rad: float = 0.10
    target_navigation_displacement_m: float = 0.20
    target_navigation_cumulative_turn_rad: float = 0.50
    minimum_queries_per_phase: int = 3


DEFAULT_HIERARCHY_AUDIT_THRESHOLDS = HierarchyAuditThresholds()
HIERARCHY_VIEW_SCHEMA_VERSION = (
    "conveyor-vla-al0-liangzhu-seen-dense-transition-view-4"
)
HIERARCHY_SPLIT_SEED = "conveyor-vla-al0-liangzhu-seen-split-v2"
BOUNDARY_WINDOW_S = 1.0
NAVIGATION_DENSE_TERMINAL_WINDOW_S = 4.0


class ConveyorVLAAL0HierarchicalDataset:
    """Phase-labelled view over an immutable official LeRobot v3 dataset."""

    def __init__(
        self,
        root: str | Path,
        normalizer_config: Mapping[str, Any],
        *,
        split: str = "train",
        component: str = "joint",
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise M0MobileError("hierarchical split must be train, val, or test")
        if component not in {"joint", "navigation", "manipulation"}:
            raise M0MobileError(
                "hierarchical component must be joint, navigation, or manipulation"
            )
        view_root = Path(root).expanduser().resolve()
        manifest = _load_view_manifest(view_root)
        base_root = (view_root / str(manifest["base_dataset_relative_path"])).resolve()
        if base_root.parent != view_root.parent or not base_root.is_dir():
            raise M0MobileError("hierarchy view base dataset must be an existing sibling")
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise M0MobileError("lerobot is required for hierarchical training") from error
        self.dataset = LeRobotDataset(
            repo_id=str(manifest["base_repo_id"]),
            root=base_root,
            video_backend="pyav",
        )
        annotations = _read_jsonl(view_root / str(manifest["annotations_relative_path"]))
        domain_filter = {
            "joint": None,
            "navigation": ActionDomain.NAVIGATION,
            "manipulation": ActionDomain.MANIPULATION,
        }[component]
        self.annotations = tuple(
            annotation
            for annotation in annotations
            if annotation.get("split") == split
            and (
                domain_filter is None
                or int(annotation.get("action_domain_id", -1)) == int(domain_filter)
            )
        )
        if not self.annotations:
            raise M0MobileError(
                f"hierarchy view has no {split}/{component} training rows"
            )
        if max(int(row["base_index"]) for row in self.annotations) >= len(self.dataset):
            raise M0MobileError("hierarchy annotation references a missing base row")
        self.state_statistics = _mapping(
            manifest.get("train_state_statistics"),
            "train_state_statistics",
        )
        self.normalizer = M0MobileNormalizer.from_config(
            normalizer_config,
            self.state_statistics,
        )
        self.root = view_root
        self.base_root = base_root
        self.manifest = manifest
        self.split = split
        self.component = component
        counts = Counter(int(row["phase_id"]) for row in self.annotations)
        self.phase_counts = {Phase(key).name: value for key, value in sorted(counts.items())}
        self.sample_weights = tuple(
            (
                len(self.annotations)
                / (len(counts) * counts[int(row["phase_id"])])
                * float(row["sampling_weight"])
            )
            for row in self.annotations
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        annotation = self.annotations[index]
        phase = Phase(int(annotation["phase_id"]))
        domain = action_domain(phase)
        example = lerobot_model_example(
            self.dataset[int(annotation["base_index"])],
            self.normalizer,
        )
        if self.component != "joint":
            expected = (
                ActionDomain.NAVIGATION
                if self.component == "navigation"
                else ActionDomain.MANIPULATION
            )
            if domain is not expected:
                raise M0MobileError("hierarchy component/action-domain mismatch")
        valid_mask = tuple(bool(value) for value in annotation["action_valid_mask"])
        if len(valid_mask) != ACTION_HORIZON or any(
            not earlier and later
            for earlier, later in zip(valid_mask, valid_mask[1:])
        ):
            raise M0MobileError("hierarchy action_valid_mask must be a valid prefix")
        example["lang"] = subtask_prompt(FULL_INSTRUCTION)
        example["solution"] = str(annotation["assistant_solution"])
        example["action"] = tuple(
            project_action10(row, domain) for row in example["action"]
        )
        example["action_mask"] = tuple(True for _ in example["action"][0])
        example["action_valid_mask"] = valid_mask
        example.update(
            {
                "phase_id": int(phase),
                "phase_name": phase.name,
                "action_domain_id": int(domain),
                "action_domain_name": domain.name,
                "subtask_text": str(annotation["subtask_text"]),
                "previous_subtask_label": annotation.get("previous_subtask_label"),
                "previous_subtask_text": (
                    None
                    if annotation.get("previous_subtask_label") is None
                    else phase_instruction(Phase[str(annotation["previous_subtask_label"])])
                ),
                "next_subtask_label": str(annotation["next_subtask_label"]),
                "seconds_to_boundary": float(annotation["seconds_to_boundary"]),
                "is_boundary_window": bool(annotation["is_boundary_window"]),
                "boundary_transition": annotation.get("boundary_transition"),
                "transition_reason": str(annotation["transition_reason"]),
                "navigation_reference_mode": annotation.get(
                    "navigation_reference_mode"
                ),
                "dataset_scope": "seen",
                "sample_id": str(annotation["sample_id"]),
                "base_index": int(annotation["base_index"]),
                "split": self.split,
            }
        )
        return example


def materialize_pct_hierarchy_view(
    episode_roots: Iterable[str | Path],
    base_lerobot_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Create an immutable phase/split sidecar without duplicating LeRobot videos."""

    roots = tuple(Path(value).expanduser().resolve() for value in episode_roots)
    if not roots or len(roots) != len(set(roots)):
        raise M0MobileError("hierarchy source episode roots must be non-empty and unique")
    root_by_id = {_source_episode_id(root): root for root in roots}
    if len(root_by_id) != len(roots):
        raise M0MobileError("hierarchy source episode IDs must be unique")
    base_root = Path(base_lerobot_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if base_root.parent != output.parent:
        raise M0MobileError("hierarchy view and base LeRobot dataset must be siblings")
    if output.exists():
        raise M0MobileError(f"hierarchy output already exists: {output}")
    base_manifest_path = base_root / "meta" / "conveyorvla_al0_conversion.json"
    base_manifest = _read_json(base_manifest_path)
    if (
        base_manifest.get("history_offsets_model_ticks") != [-5, 0]
        or not math.isclose(
            float(base_manifest.get("history_span_s", -1.0)),
            0.20,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise M0MobileError(
            "Liangzhu base manifest must use [-5, 0] / 0.20 s visual history"
        )
    expected_aliases = {
        raw: phase.name for raw, phase in sorted(PCT_PHASES.items())
    }
    if (
        base_manifest.get("transition_observations_included") is not True
        or base_manifest.get("source_phase_aliases") != expected_aliases
    ):
        raise M0MobileError(
            "dense-transition view requires an expanded base with explicit "
            "planning-state aliases"
        )
    base_episodes = _sequence(base_manifest.get("episodes"), "base episodes")
    if int(base_manifest.get("episode_count", -1)) != len(base_episodes):
        raise M0MobileError("base LeRobot episode manifest is inconsistent")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise M0MobileError("lerobot is required to build a hierarchy view") from error
    base_dataset = LeRobotDataset(
        repo_id=str(base_manifest["repo_id"]),
        root=base_root,
        video_backend="pyav",
    )
    if len(base_dataset) != int(base_manifest["frame_count"]):
        raise M0MobileError("base LeRobot frame count disagrees with its manifest")

    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    annotations_path = staging / "annotations.jsonl"
    annotations: list[dict[str, Any]] = []
    base_offset = 0
    episode_reports = []
    try:
        for episode_index, raw_report in enumerate(base_episodes):
            report = _mapping(raw_report, "base episode report")
            episode_id = str(report.get("source_episode_id", ""))
            source_root = root_by_id.get(episode_id)
            if source_root is None:
                raise M0MobileError(f"base episode has no supplied raw source: {episode_id}")
            hierarchy_audit = audit_pct_hierarchy_episode(source_root)
            if not hierarchy_audit["eligible"]:
                raise M0MobileError(
                    f"base episode fails hierarchy audit: {episode_id}: "
                    + "; ".join(hierarchy_audit["problems"])
                )
            expected_rows = int(report["query_frames"])
            candidates = list(iter_pct_temporal_records(source_root))
            if len(candidates) != expected_rows:
                raise M0MobileError(
                    f"cannot align base rows for {episode_id}: "
                    f"candidates={len(candidates)} expected={expected_rows}"
                )
            selected = 0
            split = _episode_split(episode_id)
            for row_in_episode, candidate in enumerate(candidates):
                phase = Phase(int(candidate["phase_id"]))
                valid_mask = [bool(value) for value in candidate["action_valid_mask"]]
                if len(valid_mask) != ACTION_HORIZON or any(
                    not earlier and later
                    for earlier, later in zip(valid_mask, valid_mask[1:])
                ):
                    raise M0MobileError(
                        f"invalid action prefix mask for {candidate['sample_id']}"
                    )
                previous_label = candidate.get("previous_subtask_label")
                annotations.append(
                    {
                        "episode_id": episode_id,
                        "base_index": base_offset + row_in_episode,
                        "base_episode_index": episode_index,
                        "source_episode_id": episode_id,
                        "sample_id": candidate["sample_id"],
                        "timestamp_s": (
                            int(candidate["observation_control_tick"]) / 50.0
                        ),
                        "phase_id": int(phase),
                        "phase_name": phase.name,
                        "subtask_label": phase.name,
                        "previous_subtask_label": previous_label,
                        "next_subtask_label": candidate["next_subtask_label"],
                        "action_domain_id": int(action_domain(phase)),
                        "action_domain_name": action_domain(phase).name,
                        "dataset_scope": "seen",
                        "subtask_text": phase_instruction(phase),
                        "assistant_solution": subtask_solution(phase),
                        "seconds_to_boundary": float(
                            candidate["seconds_to_boundary"]
                        ),
                        "seconds_to_next_boundary_s": float(
                            candidate["seconds_to_next_boundary_s"]
                        ),
                        "seconds_since_previous_boundary_s": float(
                            candidate["seconds_since_previous_boundary_s"]
                        ),
                        "is_boundary_window": bool(
                            candidate["is_boundary_window"]
                        ),
                        "boundary_transition": candidate["boundary_transition"],
                        "transition_reason": candidate["transition_reason"],
                        "action_valid_mask": valid_mask,
                        "valid_action_steps": sum(valid_mask),
                        "sampling_weight": _dense_sampling_weight(
                            phase,
                            float(candidate["seconds_to_next_boundary_s"]),
                            float(candidate["seconds_since_previous_boundary_s"]),
                            bool(candidate["is_boundary_window"]),
                        ),
                        "navigation_reference_mode": (
                            NAVIGATION_REFERENCE_MODES[phase]
                            if action_domain(phase) is ActionDomain.NAVIGATION
                            else None
                        ),
                        "split": split,
                    }
                )
                selected += 1
            episode_reports.append(
                {
                    "base_episode_index": episode_index,
                    "source_episode_id": episode_id,
                    "base_query_frames": expected_rows,
                    "selected_dense_transition_frames": selected,
                    "split": split,
                }
            )
            base_offset += expected_rows
        if base_offset != len(base_dataset):
            raise M0MobileError("hierarchy base-row alignment does not cover the dataset")
        if not annotations:
            raise M0MobileError("hierarchy view contains no dense-transition rows")
        split_counts = Counter(str(row["split"]) for row in annotations)
        if set(split_counts) != {"train", "val", "test"}:
            raise M0MobileError("hierarchy episode hash did not produce all three splits")
        with annotations_path.open("w", encoding="utf-8") as stream:
            for row in annotations:
                stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        train_indices = [
            int(row["base_index"]) for row in annotations if row["split"] == "train"
        ]
        train_states = np.stack(
            [
                np.asarray(
                    value.detach().cpu().numpy()
                    if hasattr(value, "detach")
                    else value,
                    dtype=np.float64,
                )
                for value in base_dataset.hf_dataset.select(train_indices)[
                    "observation.state"
                ]
            ]
        )
        if train_states.shape != (len(train_indices), 28) or not np.isfinite(train_states).all():
            raise M0MobileError("hierarchy train states are missing or non-finite")
        state_statistics = _state_statistics(train_states)
        phase_counts = Counter(str(row["phase_name"]) for row in annotations)
        domain_counts = Counter(str(row["action_domain_name"]) for row in annotations)
        boundary_counts = Counter(
            str(row["boundary_transition"])
            for row in annotations
            if row["is_boundary_window"] and row["boundary_transition"] is not None
        )
        boundary_split_counts = Counter(
            f"{row['split']}:{row['boundary_transition']}"
            for row in annotations
            if row["is_boundary_window"] and row["boundary_transition"] is not None
        )
        valid_action_prefix_counts = Counter(
            int(row["valid_action_steps"]) for row in annotations
        )
        manifest = {
            "schema_version": HIERARCHY_VIEW_SCHEMA_VERSION,
            "base_dataset_relative_path": os.path.relpath(base_root, output),
            "base_repo_id": base_manifest["repo_id"],
            "base_manifest_sha256": _sha256(base_manifest_path),
            "base_transition_observations_included": bool(
                base_manifest.get("transition_observations_included")
            ),
            "source_phase_aliases": base_manifest.get("source_phase_aliases"),
            "annotations_relative_path": "annotations.jsonl",
            "annotations_sha256": _sha256(annotations_path),
            "full_instruction": FULL_INSTRUCTION,
            "dataset_scope": "seen",
            "subtask_languages": {
                phase.name: phase_instruction(phase) for phase in PHASE_ORDER
            },
            "subtask_solutions": {
                phase.name: subtask_solution(phase) for phase in PHASE_ORDER
            },
            "prompt_history_contract": {
                "ground_truth_subtask_history_allowed": False,
                "inference_memory": "previous_model_prediction_only",
                "training_memory": "single_previous_label_with_dropout_corruption_and_teacher_forcing_decay",
            },
            "phase_order": [phase.name for phase in PHASE_ORDER],
            "phase_action_domains": {
                phase.name: action_domain(phase).name for phase in PHASE_ORDER
            },
            "action_spaces": {
                "NAVIGATION": {"indices_in_action10": [0, 2], "dimension": 2},
                "MANIPULATION": {
                    "indices_in_action10": list(range(3, 10)),
                    "dimension": 7,
                },
            },
            "action_valid_mask_semantics": (
                "prefix of future 25 Hz actions belonging to the current expert; "
                "cross-expert suffix is false"
            ),
            "boundary_window_s": BOUNDARY_WINDOW_S,
            "navigation_dense_terminal_window_s": (
                NAVIGATION_DENSE_TERMINAL_WINDOW_S
            ),
            "navigation_references": {
                phase.name: {
                    "mode": NAVIGATION_REFERENCE_MODES[phase],
                    "arm_joint_reference": list(
                        NAVIGATION_ARM_JOINT_REFERENCES[phase]
                    ),
                    "gripper_open_fraction": NAVIGATION_GRIPPER_REFERENCES[phase],
                    "tcp_delta_used": False,
                }
                for phase in (Phase.NAV_TO_SOURCE, Phase.NAV_TO_TARGET)
            },
            "action_horizon": ACTION_HORIZON,
            "base_action_dimension": ACTION_DIM,
            "history_offsets_model_ticks": [-5, 0],
            "history_span_s": 0.20,
            "video_feature_keys": list(VIDEO_FEATURE_KEYS),
            "split_seed": HIERARCHY_SPLIT_SEED,
            "split_unit": "source_episode_id",
            "split_percentages": {"train": 90, "val": 5, "test": 5},
            "episode_count": len(episode_reports),
            "selected_frame_count": len(annotations),
            "split_counts": dict(sorted(split_counts.items())),
            "phase_counts": dict(sorted(phase_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "boundary_counts": dict(sorted(boundary_counts.items())),
            "boundary_split_counts": dict(sorted(boundary_split_counts.items())),
            "valid_action_prefix_counts": {
                str(key): value
                for key, value in sorted(valid_action_prefix_counts.items())
            },
            "train_state_statistics": state_statistics,
            "episodes": episode_reports,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return {**manifest, "dataset_root": str(output)}
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def audit_pct_hierarchy_episode(
    episode_root: str | Path,
    *,
    thresholds: HierarchyAuditThresholds = DEFAULT_HIERARCHY_AUDIT_THRESHOLDS,
) -> dict[str, Any]:
    """Reject demonstrations that violate four-phase or action-domain invariants."""

    root = Path(episode_root).expanduser().resolve()
    source = audit_pct_episode(root)
    problems = list(source["problems"])
    try:
        samples = _read_jsonl(root / "samples.jsonl")
        metrics = _episode_metrics(samples)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        problems.append(str(error))
        return {
            **source,
            "eligible": False,
            "problems": problems,
            "hierarchy_metrics": None,
            "hierarchy_thresholds": asdict(thresholds),
        }

    if tuple(metrics["collapsed_phase_order"]) != tuple(
        phase.name for phase in PHASE_ORDER
    ):
        problems.append("execution phases are missing, repeated, or out of order")
    for phase in PHASE_ORDER:
        count = metrics["phase_query_counts"].get(phase.name, 0)
        if count < thresholds.minimum_queries_per_phase:
            problems.append(
                f"{phase.name} has {count} phase-pure queries; "
                f"requires {thresholds.minimum_queries_per_phase}"
            )
    for phase in (Phase.NAV_TO_SOURCE, Phase.NAV_TO_TARGET):
        phase_metrics = metrics["phase_metrics"][phase.name]
        _maximum_problem(
            problems,
            phase,
            "TCP position drift",
            phase_metrics["tcp_position_drift_m"],
            thresholds.nav_tcp_position_drift_m,
        )
        _maximum_problem(
            problems,
            phase,
            "TCP orientation drift",
            phase_metrics["tcp_orientation_drift_rad"],
            thresholds.nav_tcp_orientation_drift_rad,
        )
        _maximum_problem(
            problems,
            phase,
            "gripper range",
            phase_metrics["gripper_range_m"],
            thresholds.nav_gripper_range_m,
        )
    for phase in (Phase.PICK, Phase.PLACE):
        phase_metrics = metrics["phase_metrics"][phase.name]
        _maximum_problem(
            problems,
            phase,
            "base command",
            phase_metrics["base_command_abs_max"],
            thresholds.manipulation_base_command,
        )
        _maximum_problem(
            problems,
            phase,
            "base position drift",
            phase_metrics["base_position_drift_m"],
            thresholds.manipulation_base_position_drift_m,
        )
        _maximum_problem(
            problems,
            phase,
            "base yaw drift",
            phase_metrics["base_yaw_drift_rad"],
            thresholds.manipulation_base_yaw_drift_rad,
        )
    target = metrics["phase_metrics"][Phase.NAV_TO_TARGET.name]
    if target["base_displacement_m"] < thresholds.target_navigation_displacement_m:
        problems.append("NAV_TO_TARGET does not translate far enough to be navigation")
    if target["cumulative_turn_rad"] < thresholds.target_navigation_cumulative_turn_rad:
        problems.append("NAV_TO_TARGET does not contain a meaningful turn")
    return {
        **source,
        "eligible": not problems,
        "problems": problems,
        "hierarchy_metrics": metrics,
        "hierarchy_thresholds": asdict(thresholds),
    }


def audit_dense_transition_view(root: str | Path) -> dict[str, Any]:
    """Audit split, boundary, temporal, masking, and navigation contracts."""

    view_root = Path(root).expanduser().resolve()
    manifest = _load_view_manifest(view_root)
    annotations = _read_jsonl(
        view_root / str(manifest["annotations_relative_path"])
    )
    problems: list[str] = []
    if any("subtask_history" in row for row in annotations):
        problems.append("annotations contain forbidden ground-truth subtask_history")
    if len({int(row["base_index"]) for row in annotations}) != len(annotations):
        problems.append("base_index is not unique")
    if len({str(row["sample_id"]) for row in annotations}) != len(annotations):
        problems.append("sample_id is not unique")

    episode_splits: dict[str, set[str]] = {}
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    phase_split_counts: Counter[str] = Counter()
    boundary_window_counts: Counter[str] = Counter()
    masked_boundary_counts: Counter[str] = Counter()
    for row in annotations:
        episode_id = str(row["source_episode_id"])
        split = str(row["split"])
        phase = Phase(int(row["phase_id"]))
        episode_splits.setdefault(episode_id, set()).add(split)
        by_episode.setdefault(episode_id, []).append(row)
        phase_split_counts[f"{split}:{phase.name}"] += 1
        if row["is_boundary_window"] and row.get("boundary_transition"):
            boundary_window_counts[
                f"{split}:{row['boundary_transition']}"
            ] += 1
        valid = tuple(bool(value) for value in row["action_valid_mask"])
        if len(valid) != ACTION_HORIZON or any(
            not earlier and later
            for earlier, later in zip(valid, valid[1:])
        ):
            problems.append(f"invalid action_valid_mask: {row['sample_id']}")
        if (
            row["is_boundary_window"]
            and row.get("next_subtask_label") in {phase.name for phase in PHASE_ORDER}
            and sum(valid) < ACTION_HORIZON
        ):
            masked_boundary_counts[
                f"{split}:{phase.name}->{row['next_subtask_label']}"
            ] += 1
        reference = row.get("navigation_reference_mode")
        if action_domain(phase) is ActionDomain.NAVIGATION:
            if reference != NAVIGATION_REFERENCE_MODES[phase]:
                problems.append(f"wrong navigation reference: {row['sample_id']}")
        elif reference is not None:
            problems.append(f"manipulation row has navigation reference: {row['sample_id']}")

    leaked = sorted(
        episode_id
        for episode_id, splits in episode_splits.items()
        if len(splits) != 1
    )
    if leaked:
        problems.append(f"source episodes cross splits: {leaked[:3]}")
    for split in ("train", "val", "test"):
        for phase in PHASE_ORDER:
            if phase_split_counts[f"{split}:{phase.name}"] == 0:
                problems.append(f"missing {split}/{phase.name} rows")
        for left, right in zip(PHASE_ORDER, PHASE_ORDER[1:]):
            transition = f"{left.name}->{right.name}"
            if boundary_window_counts[f"{split}:{transition}"] == 0:
                problems.append(f"missing {split}/{transition} boundary window")
            if masked_boundary_counts[f"{split}:{transition}"] == 0:
                problems.append(f"missing {split}/{transition} masked action suffix")

    transition_counts: Counter[str] = Counter()
    discontinuities = 0
    for episode_id, rows in by_episode.items():
        rows.sort(key=lambda row: int(row["base_index"]))
        phases = [Phase(int(row["phase_id"])) for row in rows]
        collapsed = [
            phase
            for index, phase in enumerate(phases)
            if index == 0 or phase is not phases[index - 1]
        ]
        if tuple(collapsed) != PHASE_ORDER:
            problems.append(f"episode phase order is incomplete: {episode_id}")
            continue
        for index in range(1, len(rows)):
            previous = phases[index - 1]
            current = phases[index]
            if previous is current:
                continue
            transition = f"{previous.name}->{current.name}"
            transition_counts[transition] += 1
            delta = float(rows[index]["timestamp_s"]) - float(
                rows[index - 1]["timestamp_s"]
            )
            if not math.isclose(delta, 0.20, rel_tol=0.0, abs_tol=1.0e-6):
                discontinuities += 1
    for left, right in zip(PHASE_ORDER, PHASE_ORDER[1:]):
        transition = f"{left.name}->{right.name}"
        if transition_counts[transition] != len(by_episode):
            problems.append(
                f"{transition} occurs in {transition_counts[transition]}/"
                f"{len(by_episode)} episodes"
            )
    if discontinuities:
        problems.append(f"{discontinuities} phase boundaries are not 0.20 s continuous")

    navigation_trace = []
    for split in ("train", "val", "test"):
        for phase in (Phase.NAV_TO_SOURCE, Phase.NAV_TO_TARGET):
            row = next(
                item
                for item in annotations
                if item["split"] == split and int(item["phase_id"]) == int(phase)
            )
            navigation_trace.append(
                {
                    "split": split,
                    "sample_id": row["sample_id"],
                    "phase": phase.name,
                    "reference_mode": row["navigation_reference_mode"],
                    "arm_joint_reference": list(
                        NAVIGATION_ARM_JOINT_REFERENCES[phase]
                    ),
                    "gripper_open_fraction": NAVIGATION_GRIPPER_REFERENCES[phase],
                    "tcp_delta_used": False,
                }
            )
    return {
        "schema_version": "conveyor-vla-al0-dense-transition-audit-1",
        "ok": not problems,
        "dataset_root": str(view_root),
        "manifest_sha256": _sha256(view_root / "manifest.json"),
        "annotations_sha256": _sha256(
            view_root / str(manifest["annotations_relative_path"])
        ),
        "episode_count": len(by_episode),
        "row_count": len(annotations),
        "split_episode_counts": dict(
            sorted(
                Counter(next(iter(splits)) for splits in episode_splits.values()).items()
            )
        ),
        "phase_split_counts": dict(sorted(phase_split_counts.items())),
        "transition_counts": dict(sorted(transition_counts.items())),
        "boundary_window_counts": dict(sorted(boundary_window_counts.items())),
        "masked_boundary_counts": dict(sorted(masked_boundary_counts.items())),
        "boundary_discontinuities": discontinuities,
        "navigation_trace": navigation_trace,
        "problems": problems,
    }


def _episode_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples.jsonl is empty")
    by_phase: dict[Phase, list[Mapping[str, Any]]] = {
        phase: [] for phase in PHASE_ORDER
    }
    collapsed: list[Phase] = []
    phase_values: list[Phase | None] = []
    previous_step: int | None = None
    for sample in samples:
        step = _integer(sample.get("simulation_step"), "simulation_step")
        if previous_step is not None and step <= previous_step:
            raise ValueError("sample simulation steps must increase")
        previous_step = step
        raw_phase = str(sample.get("pipeline_state", ""))
        phase = PCT_PHASES.get(raw_phase)
        phase_values.append(phase)
        if phase is None:
            continue
        by_phase[phase].append(sample)
        if not collapsed or collapsed[-1] is not phase:
            collapsed.append(phase)

    query_counts = {phase.name: 0 for phase in PHASE_ORDER}
    # A 20 x 25 Hz chunk spans exactly four 5 Hz sample intervals. Requiring
    # both visual-history frames and the complete future interval to retain the
    # label prevents chunks from leaking across a phase boundary.
    for index in range(1, len(samples)):
        phase = phase_values[index]
        target_index = index + 4
        if phase is None or target_index >= len(samples):
            continue
        history_and_future = phase_values[index - 1 : target_index + 1]
        if all(value is phase for value in history_and_future):
            query_counts[phase.name] += 1

    phase_metrics = {
        phase.name: _phase_metrics(by_phase[phase]) for phase in PHASE_ORDER
    }
    return {
        "sample_count": len(samples),
        "collapsed_phase_order": [phase.name for phase in collapsed],
        "phase_sample_counts": {
            phase.name: len(by_phase[phase]) for phase in PHASE_ORDER
        },
        "phase_query_counts": query_counts,
        "phase_metrics": phase_metrics,
    }


def _source_episode_id(root: Path) -> str:
    collection = next(
        (
            parent.name
            for parent in root.parents
            if parent.name.startswith("liangzhu_0729_n")
        ),
        "liangzhu_pct",
    )
    return f"{collection}:{root.name}"


def _dense_sampling_weight(
    phase: Phase,
    seconds_to_next_boundary: float,
    seconds_since_previous_boundary: float,
    is_boundary_window: bool,
) -> float:
    """Retain every row while emphasizing navigation endpoints and switches."""

    if action_domain(phase) is ActionDomain.NAVIGATION:
        weight = 0.5
        if seconds_to_next_boundary <= NAVIGATION_DENSE_TERMINAL_WINDOW_S:
            weight = 2.0
        if seconds_to_next_boundary <= 2.0:
            weight = 4.0
    else:
        weight = 3.0 if seconds_since_previous_boundary <= BOUNDARY_WINDOW_S else 1.0
    return max(weight, 4.0 if is_boundary_window else weight)


def _episode_split(episode_id: str) -> str:
    digest = hashlib.sha256(
        f"{HIERARCHY_SPLIT_SEED}:{episode_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def _state_statistics(states: np.ndarray) -> dict[str, Any]:
    config = load_lerobot_v3_config()
    layout = list(config["features"]["state"]["names"])
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    layout_sha256 = hashlib.sha256(
        json.dumps(layout, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source_payload = json.dumps(
        {
            "count": len(states),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": "conveyor-bench-m0-mobile-state-stats-v1",
        "accepted_source_schema_versions": [HIERARCHY_VIEW_SCHEMA_VERSION],
        "split": "train",
        "state_key": "observation.state",
        "state_dimension": 28,
        "state_layout": layout,
        "state_layout_sha256": layout_sha256,
        "count": len(states),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "std_definition": "population_std_over_dense_transition_train_rows",
        "source_files": ["LeRobotDataset.hf_dataset/observation.state"],
        "source_set_sha256": hashlib.sha256(source_payload).hexdigest(),
    }


def _load_view_manifest(root: Path) -> Mapping[str, Any]:
    manifest = _read_json(root / "manifest.json")
    expected = {
        "schema_version": HIERARCHY_VIEW_SCHEMA_VERSION,
        "dataset_scope": "seen",
        "action_horizon": ACTION_HORIZON,
        "base_action_dimension": ACTION_DIM,
        "video_feature_keys": list(VIDEO_FEATURE_KEYS),
        "phase_order": [phase.name for phase in PHASE_ORDER],
        "history_offsets_model_ticks": [-5, 0],
        "history_span_s": 0.20,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise M0MobileError(f"hierarchy manifest {key} must be {value!r}")
    annotations = root / str(manifest.get("annotations_relative_path", ""))
    if not annotations.is_file() or _sha256(annotations) != manifest.get(
        "annotations_sha256"
    ):
        raise M0MobileError("hierarchy annotation sidecar is missing or corrupt")
    return manifest


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read hierarchy JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise M0MobileError(f"{name} must be a sequence")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_metrics(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not samples:
        return {
            "tcp_position_drift_m": math.inf,
            "tcp_orientation_drift_rad": math.inf,
            "gripper_range_m": math.inf,
            "base_command_abs_max": math.inf,
            "base_position_drift_m": math.inf,
            "base_yaw_drift_rad": math.inf,
            "base_displacement_m": 0.0,
            "cumulative_turn_rad": 0.0,
        }
    base_poses = [_vector(sample.get("base_pose"), 7, "base_pose") for sample in samples]
    tcp_poses = [_vector(sample.get("tcp_pose"), 7, "tcp_pose") for sample in samples]
    relative_tcp = [
        _relative_pose(base, tcp) for base, tcp in zip(base_poses, tcp_poses, strict=True)
    ]
    base_yaws = [_yaw(pose[3:]) for pose in base_poses]
    first_tcp_position, first_tcp_quaternion = relative_tcp[0]
    first_base = base_poses[0]
    gripper = [_number(sample.get("gripper_position"), "gripper_position") for sample in samples]
    base_commands = [
        _vector(sample.get("base_velocity"), 3, "base_velocity") for sample in samples
    ]
    return {
        "tcp_position_drift_m": max(
            _distance(position, first_tcp_position) for position, _ in relative_tcp
        ),
        "tcp_orientation_drift_rad": max(
            _quaternion_angle(quaternion, first_tcp_quaternion)
            for _, quaternion in relative_tcp
        ),
        "gripper_range_m": max(gripper) - min(gripper),
        "base_command_abs_max": max(abs(value) for row in base_commands for value in row),
        "base_position_drift_m": max(
            math.hypot(pose[0] - first_base[0], pose[1] - first_base[1])
            for pose in base_poses
        ),
        "base_yaw_drift_rad": max(
            abs(_wrap_angle(value - base_yaws[0])) for value in base_yaws
        ),
        "base_displacement_m": math.hypot(
            base_poses[-1][0] - first_base[0],
            base_poses[-1][1] - first_base[1],
        ),
        "cumulative_turn_rad": sum(
            abs(_wrap_angle(current - previous))
            for previous, current in zip(base_yaws, base_yaws[1:])
        ),
    }


def _relative_pose(
    base: Sequence[float],
    tcp: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    inverse = _conjugate(_unit_quaternion(base[3:]))
    position = _rotate(
        inverse,
        (tcp[0] - base[0], tcp[1] - base[1], tcp[2] - base[2]),
    )
    orientation = _unit_quaternion(_multiply(inverse, _unit_quaternion(tcp[3:])))
    return position, orientation


def _maximum_problem(
    problems: list[str],
    phase: Phase,
    name: str,
    value: float,
    threshold: float,
) -> None:
    if value > threshold:
        problems.append(
            f"{phase.name} {name} {value:.6g} exceeds {threshold:.6g}"
        )


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _vector(value: object, size: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    return tuple(_number(item, name) for item in value)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _unit_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = _vector(value, 4, "quaternion")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm <= 1.0e-8:
        raise ValueError("quaternion has zero norm")
    return tuple(item / norm for item in quaternion)  # type: ignore[return-value]


def _conjugate(value: Sequence[float]) -> tuple[float, float, float, float]:
    w, x, y, z = value
    return (w, -x, -y, -z)


def _multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = left
    bw, bx, by, bz = right
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _rotate(
    quaternion: Sequence[float],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    rotated = _multiply(
        _multiply(quaternion, (0.0, *vector)),
        _conjugate(quaternion),
    )
    return rotated[1], rotated[2], rotated[3]


def _quaternion_angle(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(left, right, strict=True)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = _unit_quaternion(quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "ConveyorVLAAL0HierarchicalDataset",
    "DEFAULT_HIERARCHY_AUDIT_THRESHOLDS",
    "HIERARCHY_SPLIT_SEED",
    "HIERARCHY_VIEW_SCHEMA_VERSION",
    "HierarchyAuditThresholds",
    "audit_pct_hierarchy_episode",
    "audit_dense_transition_view",
    "materialize_pct_hierarchy_view",
]
