#!/usr/bin/env python3
"""Paired source/model endpoint probes on validation observations; no robot motion."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, write_json, source_identity
from conveyor_bench.conveyorvla.formal_metrics import cluster_mean
from conveyor_bench.conveyorvla.execution_consistency import navigation_decomposition, validate_dwa_inputs
from conveyor_bench.conveyorvla.waypoint import nav_waypoint_world, yaw_from_quaternion
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (
    ArmVLAPCTPlannerAdapter, ArmVLADWAControllerAdapter, APPROVED_ARM_VLA_COMMIT,
)
from scripts.run_waypoint_rollout import _reference_identity


def uncertainty_report(results):
    """Episode-weighted CIs; frame pairs stay together within each episode."""
    report = {}
    for final_only in (False, True):
        subset = [r for r in results if not final_only or r["final_route_row"]]
        strata = {}
        for kind in ("source", "model"):
            selected = [r for r in subset if r["kind"] == kind]
            returned = [r for r in selected if "snap_m" in r]
            episodes = [r["episode_id"] for r in returned]
            strata[kind] = {
                "coverage": cluster_mean(["snap_m" in r for r in selected], [r["episode_id"] for r in selected]),
                "snap_failure_given_plan": cluster_mean([r["snap_m"] > .10 for r in returned], episodes),
                "snap_distance_m": cluster_mean([r["snap_m"] for r in returned], episodes),
                "snap_fail_grid_search_distance_counts": dict(Counter(
                    str(r["metadata"].get("snap_end_dist")) for r in returned if r["snap_m"] > .10)),
            }
        pairs = defaultdict(dict)
        for row in subset:
            if "snap_m" in row:
                pairs[(row["episode_id"], row["sample_id"])][row["kind"]] = row
        complete = [(ep, pair) for (ep, _), pair in pairs.items() if len(pair) == 2]
        strata["paired_model_minus_source_snap_m"] = cluster_mean(
            [p["model"]["snap_m"] - p["source"]["snap_m"] for _, p in complete],
            [ep for ep, _ in complete])
        report["last_NAV_row_per_route" if final_only else "all_selected_rows"] = strata
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--validation-rows",type=Path,required=True)
    p.add_argument("--runtime-config",type=Path,required=True)
    p.add_argument("--reference-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--rows-per-route",type=int,default=5,help="evenly spaced incl. final row; 0=all")
    args=p.parse_args()
    if args.rows_per_route < 0:
        raise ValueError("rows-per-route must be nonnegative")
    manifest=json.loads((args.prepared/"manifest.json").read_text())
    if manifest["split"] != "val":
        raise ValueError("only validation probes are supported")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.reference_root.resolve()))
    _reference_identity(args.reference_root.resolve())
    from source.interfaces import SimulationState,NavGoal
    from source.tasks import episode_spec_from_dict
    from source.pipeline.navigation_smoke import create_navigation_components
    from source.navigation.navlib import DWAController
    config_raw=json.loads(args.runtime_config.read_text())
    config=SimpleNamespace(**{k:SimpleNamespace(**v) if isinstance(v,dict) else v for k,v in config_raw.items()})
    groups=defaultdict(list)
    for line in args.validation_rows.open():
        row=json.loads(line)
        if row["episode_id"] not in manifest["episode_ids"]:
            raise ValueError("evaluation rows contain a non-validation episode")
        if row["target_route"] in {"NAV_TO_SOURCE","NAV_TO_TARGET"}:
            groups[(row["episode_id"],row["target_route"])].append(row)
    results=[]
    with (args.output_dir/"rows.jsonl").open("x") as output:
        for episode in manifest["episode_ids"]:
            source=args.prepared/"source"/episode
            for name in ("samples.jsonl","migration_task.json"):
                if sha256(source/name) != manifest["files"][f"source/{episode}/{name}"]:
                    raise ValueError("prepared source changed")
            samples={int(x["frame_index"]):x for x in (json.loads(line) for line in (source/"samples.jsonl").open())}
            task=json.loads((source/"migration_task.json").read_text())
            spec=episode_spec_from_dict(task)
            planner,executor,_=create_navigation_components(config=config,episode_spec=spec)
            pct=ArmVLAPCTPlannerAdapter(planner,simulation_state_factory=SimulationState,nav_goal_factory=NavGoal,reference_commit=APPROVED_ARM_VLA_COMMIT)
            try:
                for route in ("NAV_TO_SOURCE","NAV_TO_TARGET"):
                    rows=groups[(episode,route)]
                    indices=range(len(rows)) if not args.rows_per_route else sorted(set(int(i) for i in np.linspace(0,len(rows)-1,min(len(rows),args.rows_per_route),dtype=int)))
                    raw=executor._single_floor_raw_map if route=="NAV_TO_SOURCE" else executor._carry_single_floor_raw_map
                    local={"grid_map":raw.inflate(float(executor.local_clearance_radius)),"raw_grid_map":raw}
                    goal=task["pick" if route=="NAV_TO_SOURCE" else "place"]["base_goal"]
                    nominal=(goal["x"],goal["y"],goal["yaw"])
                    for idx in indices:
                        row=rows[idx]; sample=samples[int(row["sample_id"].rsplit("-",1)[1])]
                        base=sample["base_pose"]
                        start=(*base[:3],yaw_from_quaternion(base[3:]))
                        for kind,key in (("source","target_action"),("model","predicted_action")):
                            result={"episode_id":episode,"sample_id":row["sample_id"],"route":route,"kind":kind,
                                    "diffusion_seed":row["diffusion_seed"],"final_route_row":idx==len(rows)-1,
                                    "query_pose_xyzyaw":start,"success":False,"C_available":False}
                            try:
                                action=row.get(key)
                                if action is None or np.shape(action)!=(10,3):
                                    raise ValueError("no_comparable_NAV_prediction")
                                a=nav_waypoint_world(base,action[-1]); requested=(a[0],a[1],base[2],a[2])
                                plan=pct.plan(start,requested)
                                b=plan.snapped_goal_world
                                result.update(navigation_decomposition(nominal=nominal,requested=a,planned=(b[0],b[1],b[3])))
                                result.update(success=plan.snap_distance_m<=.10,snap_m=plan.snap_distance_m,
                                              path=plan.path_world,metadata=dict(plan.metadata),snap_limit_m=.10,
                                              gate_failure=None if plan.snap_distance_m<=.10 else "pct_endpoint_snap_exceeds_limit")
                                dwa=ArmVLADWAControllerAdapter(DWAController,executor.dwa_config,reference_commit=APPROVED_ARM_VLA_COMMIT)
                                try:
                                    # Probe the current adapter; its input guards retain explicit failure states.
                                    command=dwa.command(plan.path_world,(start[0],start[1],start[3]),(0,0,0),local)
                                    result["dwa_probe"]={"command":command,"trace":dwa.last_trace,"velocity_semantics":"zero_velocity_diagnostic_not_source_velocity"}
                                except Exception as error:
                                    import traceback
                                    result["dwa_probe"]={"error":f"{type(error).__name__}:{error}","traceback":traceback.format_exc()}
                                try:
                                    result["input_validation"]=validate_dwa_inputs(plan.path_world,local)
                                except ValueError as error:
                                    result["input_validation"]={"error":str(error)}
                            except Exception as error:
                                result["error"]=f"{type(error).__name__}:{error}"
                            output.write(json.dumps(result)+"\n");output.flush();results.append(result)
            finally:
                planner.close()
            print(json.dumps({"event":"episode_probed","episode":episode,"rows":len(results)}),flush=True)
    report={"schema":"validation-paired-pct-probe-v1","status":"complete","split":"val",
            "robot_motion":False,"C_available":False,"rows_per_route":args.rows_per_route,
            "source_identity":source_identity(ROOT),"runtime_config_sha256":sha256(args.runtime_config),
            "validation_rows_sha256":sha256(args.validation_rows),"prepared_manifest_sha256":sha256(args.prepared/"manifest.json"),
            "groups":{}}
    for kind in ("source","model"):
        values=[r for r in results if r["kind"]==kind]
        snaps=[r["snap_m"] for r in values if "snap_m" in r]
        report["groups"][kind]={"rows":len(values),"returned_plans":len(snaps),"accepted":sum(r["success"] for r in values),
            "snap_quantiles_m":np.quantile(snaps,[0,.5,.95,.99,1]).tolist() if snaps else None,
            "snap_gate_failures":sum(r.get("gate_failure") is not None for r in values),
            "planning_errors":dict(Counter(r["error"] for r in values if "error" in r)),
            "input_errors":dict(Counter(r["input_validation"]["error"] for r in values if "error" in r.get("input_validation",{}))),
            "dwa_errors":dict(Counter(r["dwa_probe"]["error"] for r in values if "error" in r.get("dwa_probe",{})))}
    report["uncertainty"] = uncertainty_report(results)
    write_json(args.output_dir/"report.json",report)


if __name__=="__main__":
    main()
