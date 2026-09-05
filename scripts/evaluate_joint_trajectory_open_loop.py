#!/usr/bin/env python3
"""Run a deterministic 32-episode overfit check for Joint-Trajectory v1."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from transformers.modeling_utils import load_sharded_checkpoint  # noqa: E402

from scripts import train_joint_trajectory as training  # noqa: E402
from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.joint_trajectory import (  # noqa: E402
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import (  # noqa: E402
    ConveyorVLAJointTrajectoryDataset,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (  # noqa: E402
    DirectJointTrajectoryExecutor,
    JointSafetyLimits,
)


ROUTES = tuple(JointTrajectoryRoute)
ARM_LOWER = (-2.618, 0.0, 0.0, -1.5708, -1.5708, -1.5708)
ARM_UPPER = (3.14, 3.14, 3.14, 1.5708, 1.5708, 1.5708)
ARM_MAX_RATE_RAD_S = (3.0,) * 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise M0MobileError("open-loop batch size must be positive")
    if not torch.cuda.is_available():
        raise M0MobileError("Joint-Trajectory open-loop evaluation requires CUDA")
    checkpoint, manifest, resolved, dataset_root = _validate_binding(args)
    dataset = ConveyorVLAJointTrajectoryDataset(dataset_root, split="train")
    selected = _select_one_row_per_episode(
        dataset, _strings(resolved.get("overfit_episode_ids"), "overfit_episode_ids")
    )

    config = _mapping(resolved.get("resolved_policy_config"), "resolved policy config")
    started = time.perf_counter()
    policy, token_ids = training._build_model(
        config,
        args.model_root.expanduser().resolve(),
        args.attention_implementation,
    )
    if dict(token_ids) != _mapping(resolved.get("special_token_ids"), "special token IDs"):
        raise M0MobileError("current processor token IDs differ from the checkpoint")
    incompatible = load_sharded_checkpoint(
        policy,
        str(args.weights.expanduser().resolve()),
        strict=True,
        prefer_safe=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise M0MobileError("consolidated step-250 weights did not load strictly")
    device = torch.device("cuda:0")
    policy.to(device=device, dtype=torch.bfloat16).eval()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows: list[dict[str, Any]] = []
    for offset in range(0, len(selected), args.batch_size):
        batch_indices = selected[offset : offset + args.batch_size]
        examples = [dataset[index] for index in batch_indices]
        decisions = policy.predict_routes(examples)
        actions: list[Sequence[Sequence[float]] | None] = [None] * len(examples)
        correct = [
            index
            for index, (example, decision) in enumerate(zip(examples, decisions, strict=True))
            if decision.valid
            and decision.route is not None
            and decision.route.value == example["route"]
        ]
        if correct:
            torch.manual_seed(args.seed + offset * 1009)
            torch.cuda.manual_seed_all(args.seed + offset * 1009)
            sampled = policy.predict_actions(
                [examples[index] for index in correct],
                [decisions[index] for index in correct],
            )
            for local_index, action in zip(correct, sampled, strict=True):
                actions[local_index] = action
        for index, example, decision, action in zip(
            batch_indices, examples, decisions, actions, strict=True
        ):
            rows.append(_evaluation_row(dataset, index, example, decision, action))
        print(
            json.dumps(
                {
                    "event": "open_loop_progress",
                    "completed": len(rows),
                    "total": len(selected),
                }
            ),
            flush=True,
        )

    report = _build_report(
        checkpoint=checkpoint,
        weights=args.weights.expanduser().resolve(),
        manifest=manifest,
        resolved=resolved,
        selected=selected,
        rows=rows,
        seed=args.seed,
        elapsed_s=time.perf_counter() - started,
    )
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["report"] = str(report_path)
    training.common._write_json_atomic(report_path, report)
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True), flush=True)
    return 0


def _validate_binding(
    args: argparse.Namespace,
) -> tuple[Path, Mapping[str, Any], Mapping[str, Any], Path]:
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = checkpoint / "joint_trajectory_checkpoint_manifest.json"
    resolved_path = checkpoint.parents[1] / "resolved_run.json"
    if not manifest_path.is_file() or not resolved_path.is_file():
        raise M0MobileError("checkpoint lacks its Joint-Trajectory binding files")
    manifest = _mapping(json.loads(manifest_path.read_text()), "checkpoint manifest")
    resolved = _mapping(json.loads(resolved_path.read_text()), "resolved run")
    if manifest.get("global_step") != training.common._checkpoint_step(checkpoint):
        raise M0MobileError("checkpoint directory and manifest step differ")
    for key in (
        "run_kind",
        "model_contract_id",
        "dataset_schema_version",
        "dataset_manifest_sha256",
        "normalization_sha256",
        "normalizer_id",
        "policy_config_sha256",
        "stage_a_steps",
        "max_steps",
    ):
        if manifest.get(key) != resolved.get(key):
            raise M0MobileError(f"checkpoint/resolved-run binding differs: {key}")
    if manifest.get("run_kind") != "disposable_32_episode_overfit":
        raise M0MobileError("this evaluator requires the exact 32-episode overfit run")
    dataset_root = Path(str(resolved.get("dataset_root", ""))).expanduser().resolve()
    dataset_manifest = dataset_root / "manifest.json"
    if (
        not dataset_manifest.is_file()
        or training.common._sha256(dataset_manifest)
        != manifest.get("dataset_manifest_sha256")
    ):
        raise M0MobileError("dataset manifest no longer matches the checkpoint")
    weights = args.weights.expanduser().resolve()
    index_path = weights / "pytorch_model.bin.index.json"
    if not index_path.is_file():
        raise M0MobileError("consolidated inference weights are incomplete")
    index = _mapping(json.loads(index_path.read_text()), "weight index")
    weight_map = _mapping(index.get("weight_map"), "weight map")
    shards = {weights / str(name) for name in weight_map.values()}
    if len(weight_map) != 1210 or len(shards) != 4 or any(not path.is_file() for path in shards):
        raise M0MobileError("consolidated inference weight index is incomplete")
    return checkpoint, manifest, resolved, dataset_root


def _select_one_row_per_episode(
    dataset: ConveyorVLAJointTrajectoryDataset,
    episode_ids: Sequence[str],
) -> list[int]:
    if len(episode_ids) != 32 or len(set(episode_ids)) != 32:
        raise M0MobileError("overfit run must bind exactly 32 unique episodes")
    by_episode_route: dict[tuple[str, str], list[int]] = defaultdict(list)
    interior: dict[tuple[str, str], list[int]] = defaultdict(list)
    eligible = set(episode_ids)
    for index, (episode, route, transition) in enumerate(
        zip(dataset.episode_ids, dataset.routes, dataset.transition_ids, strict=True)
    ):
        if episode not in eligible:
            continue
        key = (episode, route)
        by_episode_route[key].append(index)
        if transition is None:
            interior[key].append(index)
    selected = []
    for episode_index, episode in enumerate(episode_ids):
        route = ROUTES[episode_index % len(ROUTES)].value
        candidates = interior[(episode, route)] or by_episode_route[(episode, route)]
        if not candidates:
            raise M0MobileError(f"overfit episode lacks route {route}: {episode}")
        selected.append(candidates[len(candidates) // 2])
    counts = Counter(dataset.routes[index] for index in selected)
    if counts != Counter({route.value: 8 for route in ROUTES}):
        raise M0MobileError("32-episode selection is not exactly eight rows per route")
    return selected


def _evaluation_row(
    dataset: ConveyorVLAJointTrajectoryDataset,
    index: int,
    example: Mapping[str, Any],
    decision: Any,
    normalized_action: Sequence[Sequence[float]] | None,
) -> dict[str, Any]:
    target_route = JointTrajectoryRoute(str(example["route"]))
    predicted_route = (
        decision.route.value
        if decision.valid and decision.route is not None
        else "RECOVER"
    )
    row: dict[str, Any] = {
        "index": index,
        "sample_id": example["sample_id"],
        "episode_id": example["episode_id"],
        "target_route": target_route.value,
        "predicted_route": predicted_route,
        "route_correct": predicted_route == target_route.value,
        "route_valid": bool(decision.valid),
        "recover_reason": decision.recover_reason,
        "route_confidence": float(decision.route_confidence),
        "route_probs": dict(decision.route_probs),
        "subtask": decision.subtask_text,
        "action_evaluated": normalized_action is not None,
    }
    if normalized_action is None:
        return row
    predicted = dataset.normalizer.denormalize_action(target_route, normalized_action)
    target = dataset.normalizer.denormalize_action(target_route, example["action"])
    flat_normalized = [float(value) for values in normalized_action for value in values]
    row["normalized_oob_fraction"] = sum(abs(value) > 1.0 for value in flat_normalized) / len(
        flat_normalized
    )
    if action_domain(target_route) is JointTrajectoryDomain.NAVIGATION:
        xy = [math.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(predicted, target)]
        yaw = [abs(_wrap(p[2] - t[2])) for p, t in zip(predicted, target)]
        row.update(
            {
                "domain": "NAVIGATION",
                "xy_ade_m": statistics.fmean(xy),
                "xy_fde_m": xy[-1],
                "yaw_mae_rad": statistics.fmean(yaw),
                "yaw_final_error_rad": yaw[-1],
                "predicted_action": [list(values) for values in predicted],
                "target_action": [list(values) for values in target],
            }
        )
        return row

    joint_error = [
        abs(p[axis] - t[axis])
        for p, t in zip(predicted, target)
        for axis in range(6)
    ]
    gripper_error = [abs(p[6] - t[6]) for p, t in zip(predicted, target)]
    state = dataset.normalizer.denormalize_mani_state(example["mani_state"])
    executor = DirectJointTrajectoryExecutor(
        JointSafetyLimits(ARM_LOWER, ARM_UPPER, ARM_MAX_RATE_RAD_S)
    )
    predicted_chunk = executor.prepare(state[:6], predicted)
    target_chunk = executor.prepare(state[:6], target)
    row.update(
        {
            "domain": "MANIPULATION",
            "joint_mae_rad": statistics.fmean(joint_error),
            "joint_final_l2_rad": math.sqrt(
                sum((predicted[-1][axis] - target[-1][axis]) ** 2 for axis in range(6))
            ),
            "gripper_mae": statistics.fmean(gripper_error),
            "gripper_binary_accuracy": statistics.fmean(
                float((p[6] >= 0.5) == (t[6] >= 0.5))
                for p, t in zip(predicted, target)
            ),
            "predicted_saturation_rate": predicted_chunk.saturation_rate,
            "target_saturation_rate": target_chunk.saturation_rate,
            "predicted_action": [list(values) for values in predicted],
            "target_action": [list(values) for values in target],
        }
    )
    return row


def _build_report(**values: Any) -> dict[str, Any]:
    rows = values["rows"]
    route_accuracy = statistics.fmean(float(row["route_correct"]) for row in rows)
    evaluated = [row for row in rows if row["action_evaluated"]]
    nav = [row for row in evaluated if row["domain"] == "NAVIGATION"]
    mani = [row for row in evaluated if row["domain"] == "MANIPULATION"]
    metrics = {
        "route_accuracy": route_accuracy,
        "route_recover_count": sum(row["predicted_route"] == "RECOVER" for row in rows),
        "route_confusion": dict(
            sorted(Counter(f'{row["target_route"]}->{row["predicted_route"]}' for row in rows).items())
        ),
        "actions_evaluated": len(evaluated),
        "navigation_samples": len(nav),
        "navigation_xy_ade_mean_m": _mean(nav, "xy_ade_m"),
        "navigation_xy_fde_mean_m": _mean(nav, "xy_fde_m"),
        "navigation_yaw_mae_mean_rad": _mean(nav, "yaw_mae_rad"),
        "manipulation_samples": len(mani),
        "manipulation_joint_mae_mean_rad": _mean(mani, "joint_mae_rad"),
        "manipulation_joint_final_l2_mean_rad": _mean(mani, "joint_final_l2_rad"),
        "manipulation_gripper_mae_mean": _mean(mani, "gripper_mae"),
        "manipulation_gripper_binary_accuracy": _mean(mani, "gripper_binary_accuracy"),
        "manipulation_predicted_saturation_rate": _weighted_mean(
            mani, "predicted_saturation_rate"
        ),
        "normalized_oob_fraction": _mean(evaluated, "normalized_oob_fraction"),
    }
    finite = all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for key, value in metrics.items()
        if key not in {"route_confusion"}
    )
    status = "pass" if finite and len(evaluated) == len(rows) else "fail"
    return {
        "schema_version": "conveyorvla-joint-trajectory-open-loop-overfit-v1",
        "status": status,
        "scope": "32 bound training episodes only; no validation or test rows",
        "checkpoint": str(values["checkpoint"]),
        "global_step": int(values["manifest"]["global_step"]),
        "weights": str(values["weights"]),
        "run_kind": values["manifest"]["run_kind"],
        "model_contract_id": values["manifest"]["model_contract_id"],
        "dataset_manifest_sha256": values["manifest"]["dataset_manifest_sha256"],
        "selection": {
            "strategy": "one interior row per overfit episode; routes round-robin",
            "rows": len(values["selected"]),
            "episodes": 32,
            "route_counts": dict(sorted(Counter(row["target_route"] for row in rows).items())),
            "indices": list(values["selected"]),
        },
        "diffusion_seed": int(values["seed"]),
        "joint_safety_limits": {
            "lower_rad": list(ARM_LOWER),
            "upper_rad": list(ARM_UPPER),
            "max_rate_rad_s": list(ARM_MAX_RATE_RAD_S),
            "source": "approved arm-vla 388b681 Go2-X5 URDF",
        },
        "metrics": metrics,
        "elapsed_s": float(values["elapsed_s"]),
        "rows": rows,
    }


def _console_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "status": report["status"],
        "global_step": report["global_step"],
        "selection": report["selection"],
        "metrics": report["metrics"],
        "elapsed_s": report["elapsed_s"],
        "report": str(report.get("report", "written by --report")),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return math.nan if not rows else statistics.fmean(float(row[key]) for row in rows)


def _weighted_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return _mean(rows, key)


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise M0MobileError(f"{name} must be a string list")
    return tuple(value)


if __name__ == "__main__":
    raise SystemExit(main())
