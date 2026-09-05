#!/usr/bin/env python3
"""Apply the current pick criterion to archived validation pre-action samples."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.execution_consistency import sampled_phase,replay_schedule
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    manifest=json.loads((args.prepared/'manifest.json').read_text())
    if manifest['split']!='val':
        raise ValueError('only validation evidence is allowed')
    rows=[]
    for ep in manifest['episode_ids']:
        source=args.prepared/'source'/ep
        for name in ['samples.jsonl','frames.jsonl']:
            if sha256(source/name)!=manifest['files'][f'source/{ep}/{name}']:
                raise ValueError('prepared source changed')
        phase=sampled_phase([json.loads(x) for x in (source/'samples.jsonl').open()],
                            [json.loads(x) for x in (source/'frames.jsonl').open()])
        initial_z=phase[0][2]['object_pose'][2]
        ticks=[];closed=False
        for (_,_,state),command in zip(phase,replay_schedule(phase,0)):
            names=state['metadata']['joint_names'];q=state['joint_positions']
            fraction=sum(q[names.index(n)] for n in ['arm_joint7','arm_joint8'])/.08
            lift=state['object_pose'][2]-initial_z
            distance=math.dist(state['object_pose'][:3],state['tcp_pose'][:3])
            speed=math.sqrt(sum(v*v for v in state['object_velocity'][:3]))
            geometry=closed and lift>=.04 and distance<=.08 and speed<=.30
            ticks.append({'fraction':fraction,'geometry':geometry,'formal':geometry and fraction<=.5})
            # This sample is pre-action; its own command cannot establish prior closure.
            closed |= command['gripper_fraction']<=.5
        rows.append({'episode_id':ep,'sampled_pick_observations':len(ticks),
                     'geometry_witness':any(t['geometry'] for t in ticks),
                     'formal_witness':any(t['formal'] for t in ticks),
                     'witness_fractions':[t['fraction'] for t in ticks if t['geometry']]})
    report={'schema':'source-sampled-pick-criterion-audit-v1','source_environment_reexecuted':False,
            'prepared_manifest_sha256':sha256(args.prepared/'manifest.json'),
            'observations':'Archived pre-action 5Hz samples, prior commands only; no conclusion about unsampled ticks or false positives.',
            'episodes':len(rows),'geometry_witness_episodes':sum(r['geometry_witness'] for r in rows),
            'formal_witness_episodes':sum(r['formal_witness'] for r in rows),'rows':rows}
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
