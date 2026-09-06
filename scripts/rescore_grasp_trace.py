#!/usr/bin/env python3
"""Rescore archived sampled replay without changing trajectories or assistance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.physical_events import RelativeGraspEvaluator
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, write_json, source_identity
from conveyor_bench.isaac.finger_contact_tensor_probe import support_coverage_result


def rescore(path):
    evaluator=None;events=[];count=0;contacts=None;contact_rows=0
    external_witness=articulated_witness=False
    for line in path.open():
        row=json.loads(line)
        if row['event']=='source_state_initialized':
            evaluator=RelativeGraspEvaluator(row['actual_observation']['object_pose'][2],
                                             lambda event,data:events.append({'event':event,**data}))
        elif evaluator is not None and row['event']=='grasp_contact_measurement':
            contacts=dict(row);contact_rows+=1
            support=row.get('object_support_evidence')
            if support is not None:
                loaded=support.get('loaded_contacts_by_filter')
                if loaded is None:
                    contacts['external_support']=None  # Unvalidated wildcard aggregation.
                else:
                    finger=[any(v>1.e-5 for v in row['fingers'][n]['normal_forces_N']) for n in ('arm_link7','arm_link8')]
                    reverse=[any(v>0 for path,v in loaded.items() if path.endswith('/'+n)) for n in ('arm_link7','arm_link8')]
                    verified,external_witness,articulated_witness,_=support_coverage_result(
                        support['nonfinger_loaded_contacts'],finger,reverse,
                        external_witness=external_witness,articulated_witness=articulated_witness)
                    contacts['external_support']=verified
        elif evaluator is not None and row['event']=='control_step':
            state=SimpleNamespace(**row['state_after'])
            command=row['action']['metadata']['gripper_open_fraction_requested']
            evaluator.observe(state,command,contacts=contacts);count+=1;contacts=None
    if evaluator is None or not count:raise ValueError(f'missing initialized physical trace: {path}')
    return {'trace':str(path.resolve()),'trace_sha256':sha256(path),'observed_ticks':count,
            'evaluation':evaluator.evidence(),'events':events,
            'assistance_reexecuted':False,'contact_measurement_rows':contact_rows,
            'support_coverage_rechecked':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=[rescore(p) for p in sorted(args.trace_root.rglob('replay_trace.jsonl'))]
    if not rows:raise ValueError('no replay traces')
    write_json(args.output,{'schema':'read-only-grasp-rescore-v2','source_identity':source_identity(ROOT),
                           'rows':rows,'trials':len(rows),
                           'geometry_proxy_holds':sum(r['evaluation']['ever_geometry_hold_proxy'] for r in rows),
                           'contact_success_rate':None,'interpretation':'paired traces are not independent tasks'})


if __name__=='__main__':main()
