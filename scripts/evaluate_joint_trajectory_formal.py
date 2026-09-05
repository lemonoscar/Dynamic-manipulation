#!/usr/bin/env python3
"""Full held-out evaluation of the final formal checkpoint with resumable evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from conveyor_bench.conveyorvla.formal_checkpoint import (
    validate_formal_checkpoint, load_formal_policy, public_identity, source_identity, read_json, write_json,
)
from conveyor_bench.conveyorvla.formal_metrics import ROUTES, trajectory_metrics, summarize
from conveyor_bench.conveyorvla.joint_trajectory import (
    JointTrajectoryRoute, action_domain, canonical_solution, ROUTE_SUBTASKS,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import ConveyorVLAJointTrajectoryDataset


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/manipulation_navi_v1.json")
    p.add_argument("--model-root", type=Path, default=ROOT / "artifacts/models/base")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=[17])
    p.add_argument("--max-rows", type=int, default=0, help="0=full split; positive=stratified smoke only")
    p.add_argument("--freeze-test-from", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true")
    return p


def select_indices(dataset, maximum):
    if maximum == 0 or maximum >= len(dataset):
        return list(range(len(dataset)))
    # Equal route allocation for diagnostic smoke, never for the full score.
    selected = []
    for index, route in enumerate(ROUTES):
        candidates = [i for i, r in enumerate(dataset.routes) if r == route]
        count = maximum // 4 + int(index < maximum % 4)
        if count:
            selected.extend(candidates[min(len(candidates)-1, int((k+.5)*len(candidates)/count))] for k in range(count))
    return sorted(set(selected))


def oracle_decision(example):
    from conveyor_bench.conveyorvla.joint_trajectory_model import JointTrajectoryRouteDecision
    route = JointTrajectoryRoute(example["route"])
    return JointTrajectoryRouteDecision(route, canonical_solution(route), ROUTE_SUBTASKS[route], 1.,
                                        {r: float(r == route.value) for r in ROUTES}, True)


def evaluation_row(dataset, index, example, decision, action, oracle, seed):
    route = example["route"]
    prediction = decision.route.value if decision.valid and decision.route is not None else "RECOVER"
    raw_route = max(decision.route_probs, key=decision.route_probs.get)
    row = {k: example[k] for k in ("sample_id", "episode_id", "transition_id", "boundary_transition",
                                    "boundary_signed_time_s", "transition_window", "gripper_transition")}
    row.update(index=index, diffusion_seed=seed, target_route=route, predicted_route=prediction,
               query_timestamp_s=dataset._record(index)["query_timestamp_s"],
               route_correct=prediction == route, raw_route_correct=raw_route == route,
               route_valid=decision.valid, recover_reason=decision.recover_reason,
               route_probs=dict(decision.route_probs), subtask=decision.subtask_text,
               real_future_points=example["terminal_hold_start_index"], predicted=None, oracle=None,
               action_failure=None, oracle_failure=None)
    if prediction == "RECOVER":
        row["action_failure"] = "invalid_route_or_subtask"
    elif action_domain(prediction) != action_domain(route):
        row["action_failure"] = "cross_domain_route_error_no_comparable_action"
    target = dataset.normalizer.denormalize_action(route, example["action"])
    state = None if example["mani_state"] is None else dataset.normalizer.denormalize_mani_state(example["mani_state"])
    baseline = [[0., 0., 0.]] * 10 if state is None else [[0.] * 6 + [state[12]]] * 10
    for name, normalized in (("predicted", action), ("oracle", oracle), ("baseline", baseline)):
        if normalized is None:
            continue
        try:
            physical = normalized if name == "baseline" else dataset.normalizer.denormalize_action(route, normalized)
            row[name] = trajectory_metrics(route, physical, target, state, row["real_future_points"])
            row[name + "_action"] = [list(p) for p in physical]
            if name != "baseline":
                row[name]["normalized_oob_fraction"] = sum(abs(float(v)) > 1 for p in normalized for v in p) / sum(map(len, normalized))
        except (ValueError, TypeError) as error:
            row["action_failure" if name == "predicted" else "oracle_failure"] = f"{name}_invalid_action:{error}"
    row["target_action"] = [list(p) for p in target]
    return row


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size <= 0 or args.max_rows < 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("invalid batch, row limit or duplicate seeds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    binding = validate_formal_checkpoint(args.checkpoint, args.config)
    source = source_identity(ROOT, scope="open_loop")
    protocol = {"schema": "conveyorvla-formal-evaluation-v1", "identity": public_identity(binding),
                "source_sha256": source["sha256"], "batch_size": args.batch_size, "seeds": args.seeds,
                "attention": "sdpa", "dtype": "bfloat16", "saturation_limit": .005,
                "oracle": "ground_truth_route_and_canonical_subtask_diagnostic_only",
                "ci": "2000 episode-cluster bootstrap draws; episode-weighted mean; seed 20260905"}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.split == "test":
        if args.freeze_test_from is None or args.max_rows:
            raise ValueError("test requires a frozen full validation report; test smoke is forbidden")
        validation = read_json(args.freeze_test_from)
        if not (validation.get("status") == "complete" and validation.get("split") == "val"
                and validation.get("full_split") is True and validation.get("protocol_sha256") == protocol_hash):
            raise ValueError("test protocol differs from completed full validation")
    dataset = ConveyorVLAJointTrajectoryDataset(binding["dataset_root"], split=args.split)
    indices = select_indices(dataset, args.max_rows)
    selection = {"split": args.split, "indices": indices, "sample_count": len(indices),
                 "episode_count": len({dataset.episode_ids[i] for i in indices}),
                 "route_counts": dict(Counter(dataset.routes[i] for i in indices)),
                 "full_split": len(indices) == len(dataset)}
    configuration = {"protocol": protocol, "protocol_sha256": protocol_hash, "selection": selection}
    config_file = output / "evaluation_config.json"
    if config_file.exists() and read_json(config_file) != configuration:
        raise ValueError("output directory belongs to another evaluation protocol or selection")
    write_json(config_file, configuration)
    write_json(output / "source_identity.json", source)
    write_json(output / "preflight.json", {"status": "passed", **configuration})
    print(json.dumps({"event": "preflight_passed", "split": args.split, "rows": len(indices),
                      "episodes": selection["episode_count"]}), flush=True)
    if args.preflight_only:
        return 0
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; preflight passed but model evaluation has not run")
    rows_file = output / "rows.jsonl"
    rows = [json.loads(line) for line in rows_file.read_text().splitlines()] if rows_file.exists() else []
    expected = [(seed, i) for seed in args.seeds for i in indices]
    if [(r["diffusion_seed"], r["index"]) for r in rows] != expected[:len(rows)]:
        raise ValueError("resume rows are not an exact prefix of the frozen selection")
    if any(r["sample_id"] != dataset._record(r["index"])["sample_id"] for r in rows):
        raise ValueError("resume sample identity differs")
    started = time.perf_counter()
    policy = load_formal_policy(binding, args.model_root, device=args.device)
    write_json(output / "strict_load.json", {"strict": True, "missing_keys": [], "unexpected_keys": [],
        "parameters": sum(p.numel() for p in policy.parameters()), "elapsed_s": time.perf_counter()-started,
        "identity": public_identity(binding)})
    with rows_file.open("a") as stream:
        for seed_index, seed in enumerate(args.seeds):
            for offset in range(0, len(indices), args.batch_size):
                batch_indices = indices[offset:offset + args.batch_size]
                begin = seed_index * len(indices) + offset
                end = begin + len(batch_indices)
                if end <= len(rows):
                    continue
                examples = [dataset[i] for i in batch_indices]
                tic = time.perf_counter()
                decisions = policy.predict_routes(examples)
                eligible = [i for i, (e, d) in enumerate(zip(examples, decisions))
                            if d.valid and d.route is not None and action_domain(d.route) == action_domain(e["route"])]
                actions = [None] * len(examples)
                torch.manual_seed(seed + offset * 1009)
                if eligible:
                    predicted = policy.predict_actions([examples[i] for i in eligible], [decisions[i] for i in eligible])
                    for i, action in zip(eligible, predicted, strict=True):
                        actions[i] = action
                torch.manual_seed(seed + offset * 1009)
                oracle = policy.predict_actions(examples, [oracle_decision(e) for e in examples])
                elapsed = time.perf_counter() - tic
                for local, (i, e, d, a, o) in enumerate(zip(batch_indices, examples, decisions, actions, oracle, strict=True)):
                    if begin + local < len(rows):
                        continue
                    row = evaluation_row(dataset, i, e, d, a, o, seed)
                    row["batch_elapsed_s"] = elapsed
                    stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
                progress = {"event": "progress", "completed": len(rows), "total": len(expected),
                            "elapsed_s": time.perf_counter()-started, "last_batch_s": elapsed}
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    report = {"status": "complete", "split": args.split, "full_split": selection["full_split"],
              "protocol_sha256": protocol_hash, "protocol": protocol, "selection": selection,
              "elapsed_this_process_s": time.perf_counter()-started,
              "metrics": summarize(rows), "rows_path": str(rows_file),
              "oracle_action_failures": sum(r["oracle_failure"] is not None for r in rows),
              "interpretation": "Open-loop action errors are not closed-loop task success; oracle is teacher-forced."}
    write_json(output / "report.json", report)
    print(json.dumps({"event": "complete", "report": str(output / "report.json"),
                      "route_accuracy": report["metrics"]["route_accuracy"],
                      "saturation_gate": report["metrics"]["saturation_gate"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
