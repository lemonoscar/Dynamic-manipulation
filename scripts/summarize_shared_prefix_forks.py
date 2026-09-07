#!/usr/bin/env python3
"""Report fork matching, commands, geometry and timing; no pilot success-rate inference."""
import argparse,json,math,sys,base64,io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.physical_events import RelativeGraspEvaluator,rotation_wxyz
from conveyor_bench.conveyorvla.formal_checkpoint import write_json,sha256
from conveyor_bench.conveyorvla.formal_metrics import LIMITS

def geometry(state):
    obj=np.asarray(state['object_pose']);tcp=np.asarray(state['tcp_pose']);return {'tcp_distance_m':float(np.linalg.norm(obj[:3]-tcp[:3])),'relative_position_E_m':(rotation_wxyz(tcp[3:]).T@(obj[:3]-tcp[:3])).tolist()}

def clipping(response,q):
    raw=np.asarray(response['physical_relative_action'],float);previous=np.asarray(q,float);out=[]
    for i,row in enumerate(raw):
        absolute=np.asarray(q)+row[:6];pos=np.clip(absolute,LIMITS.lower,LIMITS.upper);lim=np.clip(pos,previous-np.asarray(LIMITS.max_rate_rad_s)*.2,previous+np.asarray(LIMITS.max_rate_rad_s)*.2)
        counts=[int(np.sum(np.abs(pos-absolute)>1e-12)),int(np.sum(np.abs(lim-pos)>1e-12)),int(abs(np.clip(row[6],0,1)-row[6])>1e-12)]
        if not np.allclose(lim,response['chunk']['commands'][i]['joint_position'],atol=1e-10,rtol=0):raise ValueError('clipping reconstruction differs')
        previous=lim;out.append(counts)
    total=np.sum(out,axis=0).tolist()
    if total!=[response['chunk'][k] for k in ['position_saturation_count','rate_saturation_count','gripper_saturation_count']]:raise ValueError('clipping totals differ')
    return out

def compare(a,b):
    out={};sa=a['state'];sb=b['state']
    for key,tol in [('robot_root_pose',1e-5),('joint_positions',1e-5),('joint_velocities',1e-4),('object_pose',1e-5),('object_velocity',1e-4),('tcp_pose',1e-5),('robot_root_velocity',1e-4)]:
        x=np.array(sa[key]);y=np.array(sb[key])
        if key.endswith('pose') and x.shape==(7,) and x[3:]@y[3:]<0:y[3:]*=-1
        err=float(np.max(np.abs(x-y)));out[key]={'max_abs_difference':err,'tolerance':tol,'matched':err<=tol}
    out['last_action']={'matched':a['last_action']==b['last_action']}
    diff=float(np.max(np.abs(np.asarray(a['articulation_position_target'])-b['articulation_position_target'])));out['articulation_position_target']={'max_abs_difference':diff,'matched':diff<=1e-5}
    pixels=[];steps=[]
    for x,y in zip(a['camera_history'],b['camera_history']):
        steps.append(x['step_index']==y['step_index'])
        for key in ['head_jpeg_base64','wrist_jpeg_base64']:
            xx=np.asarray(Image.open(io.BytesIO(base64.b64decode(x[key])))).astype(float);yy=np.asarray(Image.open(io.BytesIO(base64.b64decode(y[key])))).astype(float);pixels.append(float(np.mean(np.abs(xx-yy))))
    out['camera_history']={'matched':all(steps) and len(pixels)==4 and max(pixels)<=.25,'pixel_mean_absolute_differences':pixels,'step_indices_match':all(steps)}
    out['all_observed_fields_match']=all(v['matched'] for v in out.values());out['hidden_controller_contact_equivalence']='not_proven'
    return out

def summarize(path):
    s=json.loads((path/'summary.json').read_text());rows=list(map(json.loads,(path/'trace.jsonl').open()));start=s['fork_start_time_s'];basez=s['initialization']['actual_observation']['object_pose'][2]
    evaluator=RelativeGraspEvaluator(basez,lambda *args:None);measurements=[];commands=[];predictions=[];shared=next(x for x in rows if x['event']=='shared_first_plan');plans={'shared-first':clipping(shared['response'],shared['request']['joint_position'])};allplans=[plans['shared-first']]
    for r in rows:
        if r['event']=='fork_prediction':
            result=r['result'];plan=r['plan_id'];q=r['query_state'];names=q['joint_names'];anchor=[q['joint_positions'][names.index(f'arm_joint{i}')] for i in range(1,7)];plans[plan]=clipping(result,anchor);allplans.append(plans[plan])
            raw=result['physical_relative_action'];k=next((i for i,v in enumerate(raw) if v[6]<=.5),None)
            predictions.append({k:v for k,v in r.items() if k not in ['result','query_state','event','timestamp_unix_s']}|{'raw_closure_index':k,'raw_closure_time_s':None if k is None else r['time_s']+k*.2,**geometry(q),'joint_position':anchor,'absolute_commands':result['chunk']['commands']})
        if r['event']=='executed_plan_point' and r['time_s']>=start-1e-9:
            c=r['command'];q=r['state_before'];names=q['joint_names'];measured=np.array([q['joint_positions'][names.index(f'arm_joint{i}')] for i in range(1,7)])
            commands.append({'plan_id':r['plan_id'],'plan_index':r['plan_index'],'relative_time_s':r['time_s']-start,'time_s':r['time_s'],'command':c,'command_minus_measured_q':(np.array(c['joint_position'])-measured).tolist(),**geometry(q),'clipping_events':plans[r['plan_id']][r['plan_index']]})
        if r['event']=='control_step' and r['state_after']['timestamp']>start+1e-9:
            state=SimpleNamespace(**r['state_after']);g=r['action']['metadata']['gripper_open_fraction_requested'];evaluator.observe(state,g,contacts=None);measurements.append(dict(evaluator.evidence()['latest']))
    closings=[r for r in commands if r['command']['gripper_open_fraction']<=.5];e=evaluator.evidence()
    base_candidates=[r for r in measurements if r['lift_m']>=.04 and r['tcp_distance_m']<=.08 and r['command_fraction']<=.5]
    latest=e['latest'];available=None if not base_candidates else measurements[-1]['time_s']-base_candidates[0]['time_s']
    outcome='sustained_geometry_proxy' if e['ever_geometry_hold_proxy'] else ('geometry_appeared_observation_insufficient' if base_candidates and available<1 else 'geometry_appeared_not_sustained' if base_candidates else 'no_lift_proximity_close_evidence_in_window')
    executed=np.sum([c['clipping_events'] for c in commands],axis=0).tolist();full=np.sum([v for plan in allplans for v in plan],axis=0).tolist()
    return {'episode':s['source_episode'],'branch':s['branch'],'summary_sha256':sha256(path/'summary.json'),'trace_sha256':sha256(path/'trace.jsonl'),'shared_plan_sha256':s['first_plan_sha256'],'context_prefix_points':s.get('context_prefix_points',0),'fork_start_time_s':start,'window_duration_s':s['duration_after_fork_s'],'commands':commands,'predictions':predictions,'first_actual_close':closings[0] if closings else None,'closed_executed_points':len(closings),'executed_points':len(commands),'geometry_window_outcome':outcome,'peak_lift_m':max(r['lift_m'] for r in measurements),'final_lift_m':latest['lift_m'],'final_tcp_distance_m':latest['tcp_distance_m'],'geometry_evaluation':e,'clipping':{'window_executed_events':executed,'window_executed_denominator':len(commands)*7,'window_executed_rate':sum(executed)/(len(commands)*7),'full_shared_plus_generated_prediction_events':full,'full_prediction_denominator':len(allplans)*70,'full_prediction_rate':sum(full)/(len(allplans)*70),'full_prediction_gate_passed':sum(full)/(len(allplans)*70)<=.005,'shadow_predictions_counted_as_predictions_not_execution':True},'fork_state':s['fork_state']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[summarize(x.parent) for x in sorted(a.root.glob('*/runtime/episode_*/summary.json'))];groups={}
    for ep in sorted({r['episode'] for r in rows}):
        g={r['branch']:r for r in rows if r['episode']==ep};groups[ep]={'available_branches':list(g),'pairs':{}}
        for aa,bb in [('A','B'),('B','C')]:
            if aa in g and bb in g:groups[ep]['pairs'][aa+'-'+bb]=compare(g[aa]['fork_state'],g[bb]['fork_state'])
    for r in rows:del r['fork_state']
    write_json(a.output,{'schema':'fork-mechanism-evidence-v1','runs':rows,'fork_matching':groups,'interpretation':'Convenience mechanism diagnostic. No overall grasp success rate. Missing contact is unknown. Shared first-plan clipping and generated full chunks are distinct from window-executed clipping.'})
if __name__=='__main__':main()
