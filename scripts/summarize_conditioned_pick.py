#!/usr/bin/env python3
"""Report paired conditional PICK evidence, uncertainty, and first-input differences."""
from __future__ import annotations
import argparse,base64,io,json,math
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json
from conveyor_bench.conveyorvla.formal_metrics import saturation_gate


def wilson(successes,n):
    z=1.959963984540054
    center=(successes/n+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt((successes/n)*(1-successes/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-half),min(1.,center+half)]


def image_mae(x,y):
    def pixels(value):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))),dtype=float)
    return float(np.abs(pixels(x)-pixels(y)).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-source-episodes',type=int,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    groups={};identities=set()
    for path in sorted(args.root.rglob('summary.json')):
        summary=json.loads(path.read_text())
        if summary.get('schema')!='conditioned-pick-diagnostic-v1':continue
        if summary['status']!='complete':raise ValueError(f'incomplete diagnostic: {path}')
        key=(summary['source_episode'],summary['execute_points'])
        if key in groups:raise ValueError('duplicate source/period; do not count reruns as independent tasks')
        identities.add(summary['model_identity']['weights_sha256'])
        trace=path.parent/'trace.jsonl';requests=[];responses=[]
        for line in trace.open():
            row=json.loads(line)
            if row['event']=='model_request':requests.append(row)
            if row['event']=='conditioned_model_response':responses.append(row)
        if not requests or len(requests)!=len(responses):raise ValueError('unpaired model requests/responses')
        events=sum(sum(r['chunk'][k] for k in ('position_saturation_count','rate_saturation_count','gripper_saturation_count')) for r in responses)
        closures=[[c['gripper_open_fraction']<=.5 for c in r['chunk']['commands']] for r in responses]
        e=summary['physics_evidence'];v=e['evaluation_v2']
        item={'source_episode':key[0],'execute_points':key[1],
              'summary_relative_path':str(path.relative_to(args.root)), 'summary_sha256':sha256(path),
              'trace_sha256':sha256(trace),'queries':len(responses),
              'geometry_hold_proxy':v['ever_geometry_hold_proxy'],
              'contact_grasp_verified':(True if v['ever_contact_grasp_verified'] is True else
                  False if v.get('unknown_contact_window_samples') == 0 else None),
              'legacy_pick_verified':e['pick_verified'],'final_lift_m':e['latest']['lift_m'],
              'peak_lift_m':e['latest']['peak_lift_m'],'final_tcp_distance_m':e['latest']['object_tcp_distance_m'],
              'grasp_constraint_created':e['grasp_constraint_created'],
              'executed_closed_targets':sum(sum(c[:key[1]]) for c in closures),
              'executed_target_count':len(responses)*key[1],
              'queries_with_closure_only_in_unexecuted_tail':sum(not any(c[:key[1]]) and any(c[key[1]:]) for c in closures),
              'full_predicted_chunk_saturation_gate':saturation_gate({'sample_mean':events/(len(responses)*70)}),
              'saturation_denominator_semantics':'all ten predicted points per query, including unexecuted tails'}
        groups[key]=(item,summary,requests[0],responses[0])
    episodes=sorted({key[0] for key in groups})
    if len(episodes)!=args.expected_source_episodes or len(identities)!=1:
        raise ValueError('source count or fixed checkpoint identity mismatch')
    pairs=[]
    for ep in episodes:
        a,b=groups[(ep,10)],groups[(ep,2)]
        physical={k:float(np.max(np.abs(np.array(a[1]['first_query_state'][k])-b[1]['first_query_state'][k])))
                  for k in ('robot_root_pose','joint_positions','joint_velocities','object_pose','object_velocity','tcp_pose')}
        pairs.append({'source_episode':ep,'first_query_physical_max_abs_differences':physical,
                      'first_query_jpeg_exact_match':{k:a[2][k]==b[2][k] for k in ('head_images','wrist_images')},
                      'first_query_pixel_mae_0to255':{k:[image_mae(x,y) for x,y in zip(a[2][k],b[2][k])] for k in ('head_images','wrist_images')},
                      'first_prediction_max_abs_difference':float(np.max(np.abs(np.array(a[3]['physical_relative_action'])-b[3]['physical_relative_action']))),
                      'geometry_success_difference_short_minus_long':int(b[0]['geometry_hold_proxy'])-int(a[0]['geometry_hold_proxy'])})
    rates={}
    for period in (10,2):
        selected=[v[0] for key,v in groups.items() if key[1]==period];n=len(selected)
        k=sum(row['geometry_hold_proxy'] for row in selected)
        rates[str(period)]={'source_episodes':n,'proxy_holds':k,'proxy_hold_rate':k/n,
                            'wilson95':wilson(k,n),'strict_contact_rate':None}
    write_json(args.output,{'schema':'conditioned-pick-paired-report-v1','weights_sha256':identities.pop(),
                           'rows':[v[0] for v in groups.values()],'pairs':pairs,'rates':rates,
                           'interpretation':'Convenience-selected validation pilot. Wilson intervals are descriptive binomial intervals, not population capability certification. No autonomous routing or full-task claim; rendering differences retained.'})


if __name__=='__main__':main()
