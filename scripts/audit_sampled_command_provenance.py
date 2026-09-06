#!/usr/bin/env python3
"""Audit explicit versus recorder-cached targets without relabelling a release."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.joint_trajectory_data import _align_sampled_5hz_rows
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json
from conveyor_bench.conveyorvla.formal_metrics import LIMITS
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointTrajectoryExecutor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--validation-records',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    manifest=json.loads((args.prepared/'manifest.json').read_text());aligned={}
    if manifest['split']!='val':raise ValueError('validation only')
    for ep in manifest['episode_ids']:
        source=args.prepared/'source'/ep
        for name in ['frames.jsonl','samples.jsonl']:
            if sha256(source/name)!=manifest['files'][f'source/{ep}/{name}']:raise ValueError('source changed')
        aligned[ep]=_align_sampled_5hz_rows([json.loads(x) for x in (source/'samples.jsonl').open()],
                                           [json.loads(x) for x in (source/'frames.jsonl').open()])
    counts=Counter();rates=Counter();rows=[]
    decoder=DirectJointTrajectoryExecutor(LIMITS)
    for line in args.validation_records.open():
        r=json.loads(line)
        if r['split']!='val' or r['episode_id'] not in aligned:raise ValueError('unexpected record')
        if r['mani_state'] is None:continue
        sequence=aligned[r['episode_id']]
        index=next(i for i,v in enumerate(sequence) if v[0]['frame_index']==int(r['sample_id'].rsplit('-',1)[1]))
        s,f,o=sequence[index];phase=s['pipeline_state']
        chunk=decoder.prepare(r['mani_state'][:6],r['mani_delta_q_gripper'])
        counts[phase]+=1;rates[phase]+=chunk.rate_saturation_count
        future=sequence[index+1:index+1+r["terminal_hold_start_index"]]
        unavailable=sum(v[1]['action'].get('arm_joint_positions') is None for v in future)
        applied=np.array([c.joint_position for c in chunk.commands])
        raw=np.array(r['mani_delta_q_gripper'])[:,:6]+r['mani_state'][:6]
        h,axis=np.unravel_index(np.argmax(np.abs(applied-raw)),applied.shape)
        row={'sample_id':r['sample_id'],'phase':phase,'route':r['route'],
             'future_samples_without_explicit_arm_command':unavailable,
             'rate_events':chunk.rate_saturation_count,'position_events':chunk.position_saturation_count,
             'max_change_rad':float(np.abs(applied-raw).max()),'horizon_1based':int(h)+1,'joint_1based':int(axis)+1}
        if row['max_change_rad']>.5:
            target_index = index + min(int(h)+1, r['terminal_hold_start_index'])
            target_sample,target_frame,target_state=sequence[target_index]
            if not np.allclose(target_sample['action'][3:9], raw[int(h)], atol=1.e-6, rtol=0):
                raise ValueError('source target provenance disagrees with published action')
            names=target_state['metadata']['joint_names']
            row['evidence']={'query_q':r['mani_state'][:6],'raw_absolute_target':raw[int(h)].tolist(),
                'applied_target':applied[int(h)].tolist(),'source_frame_action':target_frame['action'],
                'target_sample_frame':target_sample['frame_index'],
                'target_time_measured_q':[target_state['joint_positions'][names.index(f'arm_joint{i}')] for i in range(1,7)]}
        rows.append(row)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'rows.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    report={'schema':'sampled-command-provenance-audit-v1','input_sha256':sha256(args.validation_records),
            'prepared_sha256':sha256(args.prepared/'manifest.json'),'rows_by_phase':dict(counts),
            'rate_events_by_query_phase':dict(rates),'rows_with_missing_explicit_future_commands':sum(r['future_samples_without_explicit_arm_command']>0 for r in rows),
            'largest_changes':sorted(rows,key=lambda r:r['max_change_rad'],reverse=True)[:5],
            'interpretation':'A cached target is not necessarily an explicitly issued or applied command. No source labels were rewritten.'}
    write_json(args.output_dir/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='largest_changes'},indent=2))


if __name__=='__main__':main()
