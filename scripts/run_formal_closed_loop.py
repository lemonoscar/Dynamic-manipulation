#!/usr/bin/env python3
"""Prepare source-bound migration tasks, execute both physics profiles, aggregate all attempts."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from conveyor_bench.conveyorvla.formal_checkpoint import (
    validate_formal_checkpoint, public_identity, source_identity, sha256, read_json, write_json,
)
from conveyor_bench.conveyorvla.formal_metrics import cluster_mean

REFERENCE = ROOT / "artifacts/sources/checkouts/arm-vla-388b681"
ASSETS = ROOT / "artifacts/assets/conveyorvla-v3/liangzhu"
ISAAC_PYTHON = ROOT / "artifacts/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311/bin/python"
LOCOMOTION = ROOT / "artifacts/runs/conveyorvla-al0-liangzhu-closed-loop-20260813-r1/pct_runtime/checkpoints/go2_x5/pct_multifloor/model_26000.pt"


def bind_scene(output):
    source = ASSETS / "liangzhu.usda"
    replacements = []
    def replace_asset(match):
        original = match.group(1)
        file, bracket, suffix = original.partition("[")
        if file.startswith("../../robot/"):
            path = REFERENCE / "source" / file.removeprefix("../../")
        else:
            path = (source.parent / file).resolve()
        if not path.is_file():
            raise ValueError(f"missing scene dependency: {path}")
        replacements.append({"source": original, "local": str(path), "sha256": sha256(path)})
        # The approved materializer matches these two *fallback* paths before
        # replacing them with LIANGZHU_* environment bindings. They are tokens,
        # not assets to load directly from this intermediate layer.
        arc = path
        if file in {"./usdz/liangzhu.usdz", "./usd/liangzhu_collision.usda"}:
            arc = REFERENCE / "source/scene/liangzhu" / file.removeprefix("./")
        return "@" + str(arc) + (bracket + suffix if bracket else "") + "@"
    text = re.sub(r"@([^@]+)@", replace_asset, source.read_text())
    path = output / "scene_bound.usda"
    path.write_text(text)
    write_json(output / "scene_binding.json", {"source_sha256": sha256(source), "bound_sha256": sha256(path),
                                               "dependencies": replacements, "environment": "5.1 migration"})
    return path


def prepare(args):
    binding = validate_formal_checkpoint(args.checkpoint, ROOT / "configs/manipulation_navi_v1.json")
    if args.split == "test":
        if args.freeze_validation is None:
            raise ValueError("test tasks require --freeze-validation with a completed full validation report")
        validation = read_json(args.freeze_validation)
        if not (validation.get("status") == "complete" and validation.get("split") == "val"
                and validation.get("full_split") is True
                and validation["protocol"]["identity"]["weights_sha256"] == binding["weights_sha256"]
                and validation["protocol"]["source_sha256"] == source_identity(ROOT, scope="open_loop")["sha256"]):
            raise ValueError("closed-loop test is not bound to the frozen completed validation")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = Path(binding["dataset_root"])
    dataset_manifest = read_json(dataset_root / "manifest.json")
    record_path = dataset_root / f"{args.split}.jsonl"
    if sha256(record_path) != dataset_manifest["records"][args.split]["sha256"]:
        raise ValueError("closed-loop split record identity differs")
    episodes = sorted({json.loads(line)["episode_id"] for line in record_path.open()})
    if args.limit:
        episodes = episodes[:args.limit]
    source = args.source_root.resolve()
    if sha256(source / "manifest.ndjson") != dataset_manifest["source_snapshot_manifest_sha256"]:
        raise ValueError("source snapshot identity differs")
    entries = {r["index"]: r for r in (json.loads(line) for line in (source / "manifest.ndjson").open())}
    scene = bind_scene(output)
    grouped = defaultdict(list)
    for episode in episodes:
        entry = entries[int(episode.rsplit("-", 1)[1])]
        grouped[entry["archive"]].append((episode, entry))
    tasks = []
    for archive, pairs in grouped.items():
        wanted = {entry["member_path"] + "/" + name for _, entry in pairs for name in ("task.json", "summary.json")}
        payloads = {}
        archive_path = (source / archive).resolve()
        if not archive_path.is_relative_to(source):
            raise ValueError("source archive escapes snapshot")
        expected = dataset_manifest["source_archive_sha256"]
        archive_hash = sha256(archive_path)
        if archive_hash != expected.get(archive, expected.get(Path(archive).name)):
            raise ValueError(f"source archive checksum mismatch: {archive}")
        with tarfile.open(archive_path, "r:") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if name in wanted:
                    if not member.isfile() or member.size > 16*1024*1024:
                        raise ValueError("invalid source JSON member")
                    payloads[name] = json.load(tar.extractfile(member))
                    if len(payloads) == len(wanted):
                        break
        for episode, entry in pairs:
            task = payloads[entry["member_path"] + "/task.json"]
            summary = payloads[entry["member_path"] + "/summary.json"]
            if not summary.get("success"):
                raise ValueError(f"source episode not successful: {episode}")
            directory = output / "tasks" / episode
            directory.mkdir(parents=True)
            write_json(directory / "source_task.json", task)
            write_json(directory / "source_summary.json", summary)
            local = copy.deepcopy(task)
            local["scene_usd"] = str(scene)
            for key in ("annotation_config", "annotation_config_report", "scene_asset_binding_runtime"):
                local.pop(key, None)
            write_json(directory / "task.json", local)
            tasks.append({"episode_id": episode, "split": args.split, "source_seed": entry["seed"],
                          "task": str(directory / "task.json"), "task_sha256": sha256(directory / "task.json"),
                          "source_task_sha256": sha256(directory / "source_task.json"),
                          "source_summary_sha256": sha256(directory / "source_summary.json"),
                          "source_archive_sha256": archive_hash,
                          "source_success_semantics": summary.get("success_semantics"),
                          "source_fixed_joint": summary.get("simulation_report", {}).get("grasp_fixed_joint_report")})
        print(json.dumps({"event": "tasks_materialized", "archive": archive, "tasks": len(tasks)}), flush=True)
    identity = {**public_identity(binding), "source_sha256": source_identity(ROOT)["sha256"]}
    write_json(output / "expected_identity.json", identity)
    manifest = {"schema": "conveyorvla-formal-closed-loop-v1", "identity": identity,
                "runner_source_sha256": source_identity(ROOT)["sha256"],
                "scene": str(scene), "scene_sha256": sha256(scene),
                "scene_binding": str(output / "scene_binding.json"),
                "scene_binding_sha256": sha256(output / "scene_binding.json"),
                "runtime_resources": {str(path): sha256(path) for path in (
                    ASSETS / "pct/liangzhu_single_floor.pickle", ASSETS / "pct/liangzhu_single_floor_walkable.npy",
                    ASSETS / "ply/liangzhu_collision.ply", LOCOMOTION)},
                "split": args.split, "full_split": len(episodes) == 50, "tasks": sorted(tasks, key=lambda t: t["episode_id"]),
                "environment": "IsaacSim5.1_migration_from_Sim6", "inference_freezes_simulation": True,
                "profiles": ["source_assisted", "no_grasp_assist"], "diffusion_seeds": args.seeds}
    write_json(output / "manifest.json", manifest)


def command(task, args, output, profile, seed):
    return [str(ISAAC_PYTHON), "-u", "-B", str(ROOT / "scripts/run_joint_trajectory_rollout.py"),
        "--reference-root", str(REFERENCE), "--model-endpoint", args.endpoint,
        "--expected-identity", str(args.manifest.resolve().parent / "expected_identity.json"),
        "--physics-profile", profile, "--diffusion-seed", str(seed), "--isaac-device", args.isaac_device,
        "--max-queries", str(args.max_queries), "--max-control-steps", "12000", "--no-require-initial-source-visible", "--",
        "--scene-profile", "liangzhu", "--navigation-visual-mode", "full", "--task-json", task["task"], "--output-dir", str(output),
        "--num-episodes", "1", "--seed", str(task["source_seed"]), "--headless", "--overview", "--record-video",
        "--no-randomize-task", "--no-randomize-base-goal", "--global-planner", "pct",
        "--pct-server-python", str(ISAAC_PYTHON), "--pct-server-script", str(REFERENCE / "scripts/navigation/pct_grid_server.py"),
        "--pct-tomogram-path", str(ASSETS / "pct/liangzhu_single_floor.pickle"),
        "--pct-walkable-path", str(ASSETS / "pct/liangzhu_single_floor_walkable.npy"),
        "--pct-collision-ply-path", str(ASSETS / "ply/liangzhu_collision.ply"), "--pct-no-fallback",
        "--pct-coord-mode", "identity", "--policy-profile", "pct_multifloor", "--locomotion-checkpoint", str(LOCOMOTION),
        "--locomotion-task", "RobotLab-Isaac-Velocity-Rough-Go2-X5-DogOnly-v0"]


def recover_partial_trace(runtime):
    """Keep observed model/physics evidence when termination prevented a summary.

    Only a truncated final JSON line may be skipped. Missing success evidence
    remains failure; a partial trace never manufactures a completed episode.
    """
    result = {"query_count": 0, "physics_evidence": {}, "final_route": None,
              "evidence_source": "partial_trace_without_summary"}
    counts = dict(position_events=0, rate_events=0, gripper_events=0, denominator=0)
    events = {"physical_pick_verified": "pick_verified", "physical_carry_verified": "carry_verified",
              "physical_release_observed": "release_observed", "physical_drop_proxy": "drop_detected",
              "grasp_assistance_attach": "grasp_constraint_created"}
    traces = list(runtime.rglob("joint_trajectory_trace.jsonl"))
    if len(traces) > 1:
        raise ValueError("ambiguous partial episode traces")
    for path in traces:
        with path.open() as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise
                    result["truncated_final_trace_line"] = True
                    continue
                event = item.get("event")
                if event == "model_query":
                    result["query_count"] += 1
                    response = item["response"]
                    result["final_route"] = response.get("committed_route") or "PENDING"
                    chunk = response.get("manipulation")
                    if chunk:
                        counts["denominator"] += 70
                        for name in ("position", "rate", "gripper"):
                            counts[f"{name}_events"] += int(chunk[f"{name}_saturation_count"])
                elif event in events:
                    result["physics_evidence"][events[event]] = True
    result["saturation"] = counts
    return result


def run(args):
    manifest = read_json(args.manifest)
    if manifest.get("runner_source_sha256", manifest["identity"]["source_sha256"]) != source_identity(ROOT)["sha256"]:
        raise ValueError("closed-loop code changed after task protocol freeze; prepare a new manifest")
    if "scene" in manifest:
        if sha256(Path(manifest["scene"])) != manifest["scene_sha256"] or sha256(Path(manifest["scene_binding"])) != manifest["scene_binding_sha256"]:
            raise ValueError("closed-loop scene binding changed")
        for item in read_json(Path(manifest["scene_binding"]))["dependencies"]:
            if sha256(Path(item["local"])) != item["sha256"]:
                raise ValueError("closed-loop scene dependency changed")
        for path, digest in manifest["runtime_resources"].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"closed-loop runtime resource changed: {path}")
    tasks = manifest["tasks"][:args.limit] if args.limit else manifest["tasks"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    env = dict(os.environ, LIANGZHU_VISUAL_USDZ=str(ASSETS / "usdz/liangzhu.usdz"),
               LIANGZHU_COLLISION_USD=str(ASSETS / "usd/liangzhu_collision.usda"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    for task in tasks:
        if sha256(Path(task["task"])) != task["task_sha256"]:
            raise ValueError("materialized task changed")
        for profile in manifest["profiles"]:
            for seed in manifest["diffusion_seeds"]:
                directory = output / f'{task["episode_id"]}-{profile}-seed{seed}'
                result_file = directory / "attempt.json"
                if result_file.exists():
                    result = read_json(result_file)
                    if result.get("manifest_sha256") != sha256(args.manifest):
                        raise ValueError("attempt belongs to another manifest")
                    if result.get("status") != "complete":
                        raise ValueError("an unfinished attempt exists; inspect its process/log before resuming")
                    attempts.append(result)
                    continue
                directory.mkdir(parents=True, exist_ok=False)
                cmd = command(task, args, directory / "runtime", profile, seed)
                write_json(directory / "command.json", {"argv": cmd, "cwd": str(REFERENCE)})
                result = {"episode_id": task["episode_id"], "profile": profile, "diffusion_seed": seed,
                          "manifest_sha256": sha256(args.manifest), "status": "running", "success": False,
                          "transfer_chain_success": False, "failure_reason": None}
                write_json(result_file, result)
                start = time.perf_counter()
                with (directory / "process.log").open("w") as log:
                    try:
                        process = subprocess.Popen(cmd, cwd=REFERENCE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        result["returncode"] = process.wait(timeout=args.timeout_s)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        result["failure_reason"] = "wall_clock_timeout"
                    except OSError as error:
                        result["failure_reason"] = f"runtime_launch_failed:{type(error).__name__}:{error}"
                summaries = list((directory / "runtime").rglob("summary.json"))
                summaries = [p for p in summaries if read_json(p).get("schema_version") == "conveyorvla-joint-trajectory-rollout-summary-v1"]
                if len(summaries) == 1:
                    summary = read_json(summaries[0])
                    result.update(success=bool(summary["success"]), transfer_chain_success=bool(summary["transfer_chain_success"]),
                                  failure_reason=result["failure_reason"] or summary["failure_reason"],
                                  summary=str(summaries[0]), physics_evidence=summary["physics_evidence"],
                                  timing=summary["timing"], saturation=summary["predicted_chunk_saturation"],
                                  query_count=summary["query_count"],
                                  final_route=summary["state_trace"][-1] if summary["state_trace"] else None)
                else:
                    result.update(recover_partial_trace(directory / "runtime"))
                    result["failure_reason"] = result["failure_reason"] or "runtime_failed_before_episode_summary"
                result.update(status="complete", wall_s=time.perf_counter()-start)
                write_json(result_file, result)
                attempts.append(result)
                write_json(output / "report.json", aggregate(attempts, manifest, args))
                print(json.dumps({"event": "attempt_completed", **result}), flush=True)
    write_json(output / "report.json", aggregate(attempts, manifest, args))


def aggregate(attempts, manifest, args):
    import numpy as np
    report = {"schema": "formal-closed-loop-report-v1", "environment": manifest["environment"],
              "manifest_sha256": sha256(args.manifest), "smoke": bool(args.limit or args.max_queries < 96),
              "allocated_attempts": (min(args.limit, len(manifest["tasks"])) if args.limit else len(manifest["tasks"])) * len(manifest["profiles"]) * len(manifest["diffusion_seeds"]),
              "completed_attempts": len(attempts), "profiles": {}, "attempts": attempts}
    report["status"] = "complete" if report["completed_attempts"] == report["allocated_attempts"] else "running"
    for profile in manifest["profiles"]:
        rows = [r for r in attempts if r["profile"] == profile]
        episodes = [r["episode_id"] for r in rows]
        failures = Counter(r["failure_reason"] for r in rows if r["failure_reason"])
        latency = [v for r in rows for v in r.get("timing", {}).get("inference_wall_s", [])]
        valid = [r for r in rows if r.get("query_count", 0) > 0]
        saturation = [r["saturation"] for r in rows if r.get("saturation", {}).get("denominator", 0)]
        denominator = sum(r["denominator"] for r in saturation)
        clip_events = sum(r["position_events"] + r["rate_events"] + r["gripper_events"] for r in saturation)
        report["profiles"][profile] = {
            "contract_success": cluster_mean([r["success"] for r in rows], episodes),
            "transfer_chain_success": cluster_mean([r["transfer_chain_success"] for r in rows], episodes),
            "pick_success": cluster_mean([r.get("physics_evidence", {}).get("pick_verified", False) for r in rows], episodes),
            "carry_success": cluster_mean([r.get("physics_evidence", {}).get("carry_verified", False) for r in rows], episodes),
            "release_success": cluster_mean([r.get("physics_evidence", {}).get("release_observed", False) for r in rows], episodes),
            "drop_rate": cluster_mean([r.get("physics_evidence", {}).get("drop_detected", False) for r in rows], episodes),
            "failure_counts": dict(failures),
            "failure_route_counts": dict(Counter(r.get("final_route") or "before_first_query" for r in rows if not r["success"])),
            "attempts_with_model_queries": len(valid), "attempts_without_model_queries": len(rows)-len(valid),
            "contract_success_on_queried_attempts": cluster_mean([r["success"] for r in valid], [r["episode_id"] for r in valid]),
            "predicted_chunk_saturation_rate": clip_events / denominator if denominator else None,
            "saturation_gate": {"threshold": .005, "events": clip_events, "denominator": denominator,
                                "passed": clip_events / denominator <= .005 if denominator else None},
            "interpretation": "Allocated-attempt rates include startup failures. No model capability conclusion when no model queries ran.",
            "inference_latency_s_p50_p95_p99": np.quantile(latency, [.5, .95, .99]).tolist() if latency else None,
            "pure_physics": False, "inference_freezes_simulation": True}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--checkpoint", type=Path, required=True)
    prep.add_argument("--source-root", type=Path, default=ROOT / "artifacts/datasets/modelscope-liangzhuNeW_500")
    prep.add_argument("--split", choices=("val", "test"), default="val")
    prep.add_argument("--freeze-validation", type=Path)
    prep.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    for sub in (prep,):
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--limit", type=int, default=0)
    execute = subs.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--endpoint", default="http://127.0.0.1:18082")
    execute.add_argument("--isaac-device", default="cuda:0")
    execute.add_argument("--limit", type=int, default=0)
    execute.add_argument("--max-queries", type=int, default=96)
    execute.add_argument("--timeout-s", type=float, default=1200.)
    args = p.parse_args()
    if args.limit < 0:
        raise ValueError("limit must be nonnegative")
    (prepare if args.mode == "prepare" else run)(args)


if __name__ == "__main__":
    main()
