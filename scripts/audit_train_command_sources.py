#!/usr/bin/env python3
"""Read-only point/channel provenance audit. No label filling or release changes."""
import argparse,json,tarfile,sys,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.joint_trajectory_data import _align_sampled_5hz_rows
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json
from conveyor_bench.conveyorvla.formal_metrics import LIMITS
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointTrajectoryExecutor

def close(a,b):
    return a is not None and b is not None and np.shape(a)==np.shape(b) and bool(np.allclose(a,b,atol=1e-6,rtol=0))

def explicit(action,ch):
    if ch=='arm':return action.get('arm_joint_positions')
    if ch=='base_velocity':return action.get('base_velocity')
    q=(action.get('metadata') or {}).get('gripper_joint_positions')
    if q:return [q[0],q[1] if len(q)>1 else q[0]]
    return {'open':[.04,.04],'close':[0.,0.]}.get(action.get('gripper_command'))

def classify(target,command,*,applied_evidence,old_cache=None,ever_current_command=False,hold_evidence=False):
    if command is not None:
        if not close(target,command):return 'unknown','explicit_target_disagrees_with_sample'
        return ('valid_new','explicit_dispatch_poststep') if applied_evidence else ('unknown','explicit_without_application_evidence')
    if hold_evidence:return 'valid_hold','fresh_current_cycle_target_report'
    if not ever_current_command and close(target,old_cache):return 'invalid_cache','pre_reset_value_without_current_cycle_command'
    return 'unknown','no_proven_effective_current_cycle_target'

def measured(o,ch):
    names=o.get('metadata',{}).get('joint_names',[]);q=o.get('joint_positions',[])
    needed=[f'arm_joint{i}' for i in (range(1,7) if ch=='arm' else (7,8))]
    return [q[names.index(n)] for n in needed] if all(n in names for n in needed) else None

def audit_episode(ep,samples,frames):
    aligned=_align_sampled_5hz_rows(samples,frames);old={ch:None for ch in ('arm','gripper','base_velocity')};reset=[]
    for i,f in enumerate(frames):
        pre=f.get('observation') or {};post=f.get('post_step_observation') or {}
        if pre.get('step_index',0)>post.get('step_index',0) and post.get('step_index')==0:
            for ch in ('arm','gripper'):old[ch]=measured(pre,ch)
            reset.append(i+1)
    seen={ch:False for ch in old};last={ch:None for ch in old};points=[];lines={id(f):i+1 for i,f in enumerate(frames)}
    first_line=lines[id(aligned[0][1])]
    if any(line>=first_line for line in reset):raise ValueError('reset inside sampled cycle needs explicit multi-cycle audit')
    cursor=max(reset,default=0)
    for s,f,o in aligned:
        # Include diagnostic/non-sampled frames when deciding whether a current-cycle command existed.
        for prior in frames[cursor:lines[id(f)]-1]:
            for ch in seen:
                if explicit(prior['action'],ch) is not None:seen[ch]=True;last[ch]=None
        cursor=lines[id(f)]
        action=f['action'];pre=f.get('observation') or {};post=f.get('post_step_observation') or {};pm=post.get('metadata') or {};prm=pre.get('metadata') or {}
        advanced=post.get('step_index',-1)>pre.get('step_index',-1)
        dispatch=advanced and pm.get('last_action_source')==action.get('source') and not action.get('metadata',{}).get('skip_physics_step',False)
        row={'episode_id':ep,'source_frame':s['frame_index'],'time_s':s['timestamp'],'simulation_step':s['simulation_step'],'phase':s['pipeline_state'],'frames_jsonl_line':lines[id(f)],'reset_evidence_lines':reset,'channels':{}}
        for ch,sl in [('base_velocity',slice(0,3)),('arm',slice(3,9)),('gripper',slice(9,11))]:
            target=s['action'][sl];cmd=explicit(action,ch);key='last_arm_joint_position_target_report' if ch=='arm' else 'last_gripper_joint_position_target_report'
            report=pm.get(key) or {};previous=prm.get(key) or {};fresh=report.get('applied') is True and report.get('apply_count',0)>previous.get('apply_count',0);reported=report.get('target_positions')
            matches=close(target,reported) or bool(ch=='gripper' and reported and close(target,[reported[0]]*2))
            evidence=(fresh and matches) or (dispatch and bool(action.get('metadata',{}).get('world_step_owned_by_pipeline')))
            if ch=='base_velocity':evidence=dispatch
            hold=seen[ch] and fresh and matches and close(last[ch],target)
            label,reason=classify(target,cmd,applied_evidence=evidence,old_cache=old[ch],ever_current_command=seen[ch],hold_evidence=hold)
            row['channels'][ch]={'class':label,'reason':reason,'target':target,'explicit_command':cmd,'poststep_advanced':advanced,'poststep_source_matches':dispatch,'fresh_target_report_matches':bool(fresh and matches),'pre_reset_cache':old[ch] if label=='invalid_cache' else None,'application_evidence_scope':'pipeline_dispatch_not_actuator_readback' if label=='valid_new' and not(fresh and matches) else 'target_report_or_provenance'}
            if cmd is not None:seen[ch]=True;last[ch]=cmd if label=='valid_new' else None
        points.append(row)
    return aligned,points

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('dataset-root','source-root','output-dir'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);m=json.loads((a.dataset_root/'manifest.json').read_text())
    if sha256(a.dataset_root/'train.jsonl')!=m['records']['train']['sha256']:raise ValueError('train hash differs')
    if sha256(a.source_root/'manifest.ndjson')!=m['source_snapshot_manifest_sha256']:raise ValueError('source manifest differs')
    records=defaultdict(list)
    for l in (a.dataset_root/'train.jsonl').open():
        r=json.loads(l)
        if r['split']!='train':raise ValueError('nontrain record')
        records[r['episode_id']].append(r)
    entries={int(x['index']):x for x in map(json.loads,(a.source_root/'manifest.ndjson').open())};groups=defaultdict(list)
    for ep in records:groups[entries[int(ep.rsplit('-',1)[1])]['archive']].append(ep)
    counts={ch:Counter() for ch in ('arm','gripper','base_velocity')};chunks={ch:Counter() for ch in ('arm','gripper')};refs={ch:Counter() for ch in chunks};affected=Counter();phases=Counter();hashes={};episodes=[];nall=nmani=nnav=position=rate=grip=0;decoder=DirectJointTrajectoryExecutor(LIMITS)
    with (a.output_dir/'points.jsonl').open('x') as pf,(a.output_dir/'chunks.jsonl').open('x') as cf:
        for archive,eps in sorted(groups.items()):
            path=(a.source_root/archive).resolve()
            if not path.is_relative_to(a.source_root.resolve()):raise ValueError('unsafe path')
            print('hashing',archive,flush=True);digest=sha256(path)
            if digest!=m['source_archive_sha256'][archive]:raise ValueError('archive hash differs')
            hashes[archive]=digest;wanted={entries[int(ep.rsplit('-',1)[1])]['member_path']+'/'+name:(ep,name) for ep in eps for name in ('samples.jsonl','frames.jsonl','events.jsonl')};data=defaultdict(dict)
            with tarfile.open(path,'r:') as tar:
                for member in tar:
                    key=member.name.removeprefix('./')
                    if key not in wanted:continue
                    ep,name=wanted.pop(key)
                    if not member.isfile() or member.size>64*1024*1024:raise ValueError('invalid evidence')
                    raw=tar.extractfile(member).read();data[ep][name]=[json.loads(l) for l in raw.splitlines()];hashes[key]=hashlib.sha256(raw).hexdigest()
                    if not wanted:break
            if wanted:raise ValueError('missing evidence')
            for ep in eps:
                aligned,points=audit_episode(ep,data[ep]['samples.jsonl'],data[ep]['frames.jsonl']);idx={x['source_frame']:i for i,x in enumerate(points)};ec=Counter()
                for r in points:
                    pf.write(json.dumps(r)+'\n');nall+=1
                    for ch,v in r['channels'].items():counts[ch][v['class']]+=1
                for r in records[ep]:
                    if r['mani_state'] is None:nnav+=1;continue
                    nmani+=1;i=idx[int(r['sample_id'].rsplit('-',1)[1])];real=r['terminal_hold_start_index'];future=points[i+1:i+1+real];raw=np.asarray(r['mani_delta_q_gripper']);absolute=raw[:,:6]+r['mani_state'][:6]
                    for j,v in enumerate(future):
                        if not close(absolute[j],v['channels']['arm']['target']):raise ValueError('arm horizon differs')
                        if abs(raw[j,6]-np.clip(np.mean(v['channels']['gripper']['target'])/.04,0,1))>1e-6:raise ValueError('gripper horizon differs')
                    presence={ch:sorted({v['channels'][ch]['class'] for v in future}) for ch in chunks};bad={ch:[v['source_frame'] for v in future if v['channels'][ch]['class'] in ('invalid_cache','unknown')] for ch in chunks}
                    invalid=any(v['channels'][ch]['class']=='invalid_cache' for v in future for ch in chunks);unknown=any(v['channels'][ch]['class']=='unknown' for v in future for ch in chunks)
                    for ch in chunks:chunks[ch].update(presence[ch]);refs[ch].update(v['channels'][ch]['class'] for v in future)
                    affected['invalid_cache']+=invalid;affected['unknown']+=unknown;affected['isolate_union']+=invalid or unknown;ec['isolate_union']+=invalid or unknown
                    if invalid or unknown:phases[points[i]['phase']]+=1
                    dec=decoder.prepare(r['mani_state'][:6],r['mani_delta_q_gripper']);position+=dec.position_saturation_count;rate+=dec.rate_saturation_count;grip+=dec.gripper_saturation_count
                    cf.write(json.dumps({'sample_id':r['sample_id'],'episode_id':ep,'query_frame':points[i]['source_frame'],'real_future_points':real,'padding_points':10-real,'classes_by_channel':presence,'isolate':invalid or unknown,'affected_source_frames':bad,'rate_events':dec.rate_saturation_count,'position_events':dec.position_saturation_count,'gripper_events':dec.gripper_saturation_count})+'\n')
                episodes.append({'episode_id':ep,'sampled_source_points':len(points),**ec});print('audited',ep,flush=True)
            pf.flush();cf.flush()
    report={'schema':'train-command-provenance-v1','status':'complete','train_episodes':len(records),'distinct_sampled_source_points':nall,'source_point_counts_by_channel':counts,'mani_chunks':nmani,'nav_chunks_not_mani_command_supervision':nnav,'chunk_class_presence_by_channel_nonexclusive':chunks,'horizon_reference_counts_not_unique_source_points':refs,'affected_mani_chunks_nonexclusive':affected,'isolated_chunks_by_query_phase':phases,'episodes':episodes,'saturation':{'position':position,'rate':rate,'gripper':grip,'denominator':nmani*10*7,'sample_mean':(position+rate+grip)/(nmani*10*7),'threshold':.005},'evidence_limits':['Sampled logs do not reveal all intervening high-frequency commands.','valid_new can use explicit matching target, world-step ownership, poststep source and advancement: pipeline dispatch evidence, not independent actuator target readback.','valid_hold requires fresh matching target report; gaps without target evidence remain unknown.','Base velocity provenance is distinct from NAV pose supervision.','No labels, normalizer, dataset or model changed.'],'input_hashes':hashes,'train_records_sha256':m['records']['train']['sha256'],'audit_script_sha256':sha256(Path(__file__))};write_json(a.output_dir/'report.json',report)

if __name__=='__main__':main()
