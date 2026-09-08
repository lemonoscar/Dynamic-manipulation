#!/usr/bin/env python3
"""Complete recorded-episode fit for one preregistered training family; not autonomous physics."""
import argparse, hashlib, json, math, os, sys, time, subprocess
from collections import Counter
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def select_train_family(actions, transitions, normalizer):
    """Choose by source identity/coverage/images only, before any model outputs."""
    from PIL import Image
    routes = {'NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE'}
    train_families = sorted({r['task_family_id'] for r in actions if r['split'] == 'train'})
    exclusions = []
    for family in train_families:
        pool = [r for r in actions if r['task_family_id'] == family]
        tr = [r for r in transitions if r['task_family_id'] == family]
        if any(r['split'] != 'train' for r in [*pool, *tr]):
            raise ValueError('selected family crosses a data split')
        if {r['route'] for r in pool} != routes:
            exclusions.append({'family': family, 'reason': 'missing_action_route'})
            continue
        if family not in normalizer['families'] or normalizer['fit_split'] != 'train':
            raise ValueError('training family is not bound to the frozen train normalizer')
        if len({r['episode_uuid'] for r in [*pool, *tr]}) != 1:
            raise ValueError('single-family diagnostic requires one unambiguous episode')
        image_hashes = {}
        try:
            for row in [*pool, *tr]:
                value = row.get('input', row)
                root = Path(row['episode_root']).resolve()
                for name in value['images']:
                    path = (root / name).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError('image escapes selected source episode')
                    if str(path) not in image_hashes:
                        with Image.open(path) as frame:
                            frame.convert('RGB').load()
                        image_hashes[str(path)] = sha(path)
        except (OSError, ValueError) as error:
            exclusions.append({'family': family, 'reason': 'image_unavailable', 'error': str(error)})
            continue
        if not tr or {r['label']['operation'] for r in tr} != {'CONTINUE', 'ADVANCE'}:
            raise ValueError('selected complete family lacks qualified transition queries')
        return family, dict(family=family, split='train',
            selection='lexicographically_first_train_family_with_all_four_routes_and_decodable_RGB',
            exclusions=exclusions, action_rows=len(pool), transition_rows=len(tr),
            routes=dict(Counter(r['route'] for r in pool)),
            episode_uuids=sorted({r['episode_uuid'] for r in pool}),
            episode_roots=sorted({r['episode_root'] for r in pool}),
            query_time_range_s=[min(r['query_time_s'] for r in pool), max(r['query_time_s'] for r in pool)],
            normalizer_train_family_member=True, image_hashes=image_hashes)
    raise ValueError('no complete training family with available RGB')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','release','checkpoint','legacy-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--wall-seconds',type=int,default=900)
    p.add_argument('--expected-checkpoint-sha256',default='474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0')
    a=p.parse_args(); start=time.monotonic()
    sys.path[:0]=[str(a.repo),str(a.repo/'src')]
    import numpy as np
    import torch
    from conveyor_bench.conveyorvla.full_episode_backend import load_full_episode_backend,propose_transition
    from conveyor_bench.conveyorvla.staged_rgb_backend import encode_rgb_conditions
    from conveyor_bench.conveyorvla.full_episode_context import public_context_text
    from conveyor_bench.conveyorvla.formal_metrics import cluster_mean,LIMITS
    from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointTrajectoryExecutor
    from scripts.train_full_episode_vla import images,validate_rows
    torch.set_num_threads(4)
    a.output.mkdir(parents=True,exist_ok=False)
    def event(kind,**data):
        v=dict(event=kind,elapsed_s=time.monotonic()-start,time_unix_s=time.time(),**data)
        print(json.dumps(v),flush=True)
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    def deadline():
        if time.monotonic()-start>a.wall_seconds:raise TimeoutError('frozen open-loop wall limit')
    manifest=json.loads((a.release/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if sha(a.release/name)!=expected:raise ValueError('release file identity changed: '+name)
    actions=[json.loads(l) for l in (a.release/'actions.jsonl').read_text().splitlines()]
    transitions=[json.loads(l) for l in (a.release/'transitions.jsonl').read_text().splitlines()]
    validate_rows(manifest,actions,transitions)
    family,selection=select_train_family(actions,transitions,json.loads((a.release/'normalization.json').read_text()))
    write(a.output/'selection.json',selection)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if saved['steps']!=1700:raise ValueError('requested best step is not 1700')
    if saved['training_binding']['release_sha256']!=sha(a.release/'manifest.json'):raise ValueError('checkpoint release mismatch')
    source_files={str(p.relative_to(a.repo)):sha(p) for p in (a.repo/'src/conveyor_bench/conveyorvla').rglob('*.py')}
    identity=dict(source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip(),source_files=source_files,checkpoint_sha256=sha(a.checkpoint),step=saved['steps'],normalizer_id=saved['normalizer']['normalizer_id'],
        release_sha256=sha(a.release/'manifest.json'),code_sha256=sha(__file__),time_profile=saved['candidate']['time_profile'])
    if saved['normalizer']!=json.loads((a.release/'normalization.json').read_text()):raise ValueError('normalizer payload differs from frozen release')
    identity['normalization_sha256']=sha(a.release/'normalization.json')
    if subprocess.check_output(['git','status','--porcelain'],cwd=a.repo,text=True).strip():raise ValueError('evaluation source tree must be clean')
    if identity['checkpoint_sha256']!=a.expected_checkpoint_sha256:raise ValueError('checkpoint differs from approved step1700 bytes')
    selection['checkpoint_training_binding']=saved['training_binding']
    selection['checkpoint_release_binding_verified']=True
    write(a.output/'selection.json',selection)
    del saved
    protocol=dict(schema='full-episode-rgb-train-family-fit-v1',identity=identity,seeds=[17],
        selected_family=family,split='train',selection_sha256=sha(a.output/'selection.json'),
        autonomous_closed_loop=False,training_weights_modified=False,normalizer_refit=False,
        action_condition='teacher committed active task and prior memory; conditional action diagnostic',
        transition_condition='RGB and causal prior memory only; labels/evaluator never supplied',
        precision='FP32 stored weights, BF16 autocast, FP32 expert sampling tensors',
        target_scoring='native supported true mask only; no terminal hold labels',
        saturation_gate=dict(limit=.005,definition='full prediction (position+rate+gripper events)/(10*7), sample mean'),
        real_time=False,depth=False,wall_seconds=a.wall_seconds)
    write(a.output/'protocol.json',protocol);event('preflight_passed',step=identity['step'],checkpoint_sha256=identity['checkpoint_sha256'])
    backend=load_full_episode_backend(a.checkpoint,legacy_checkpoint=a.legacy_checkpoint,repo_root=a.repo,device=a.device)
    if backend.model_id!=identity['checkpoint_sha256']:raise ValueError('strict loader weight identity mismatch')
    write(a.output/'strict_load.json',dict(strict=True,identity=identity,parameters=sum(v.numel() for v in backend.policy.parameters())+sum(v.numel() for v in backend.model.parameters())))
    event('strict_load_passed')
    safety=DirectJointTrajectoryExecutor(LIMITS)
    boundaries={}
    for row in transitions:
        boundaries.setdefault(row['episode_uuid'],set()).add(row['label_evidence']['event_time_s'])
    def metrics(row,pred,norm):
        target=np.asarray(row['actions']);mask=np.asarray(row['action_valid_mask']);n=int(mask.sum());mani=row['route'] in ('PICK','PLACE')
        if pred.shape!=target.shape or not np.isfinite(pred).all():raise ValueError('nonfinite/malformed sampled action')
        baseline=np.zeros_like(target)
        if mani:baseline[:,6]=row['mani_state'][12]
        normalized_target=backend.normalizer.normalize_action(row['route'],target)
        result={'finite':True,'valid_points':n,'normalized_oob_fraction':float((abs(norm)>1).mean())}
        for label,count in [('first',1),('executed_prefix',min(2,n)),('full_valid',n)]:
            d=pred[:count]-target[:count];base=baseline[:count]-target[:count]
            nd=norm[:count]-normalized_target[:count]
            v=dict(mae_by_dimension=np.abs(d).mean(0).tolist(),rmse_by_dimension=np.sqrt((d*d).mean(0)).tolist(),
                normalized_rmse=float(np.sqrt((nd*nd).mean())),zero_hold_sse=float((base*base).sum()),model_sse=float((d*d).sum()))
            if mani:v.update(joint_mae_rad=float(abs(d[:,:6]).mean()),gripper_mae=float(abs(d[:,6]).mean()),gripper_accuracy=float(((pred[:count,6]>=.5)==(target[:count,6]>=.5)).mean()))
            else:
                yaw=(d[:,2]+np.pi)%(2*np.pi)-np.pi
                v.update(xy_ade_m=float(np.linalg.norm(d[:,:2],axis=1).mean()),xy_fde_m=float(np.linalg.norm(d[-1,:2])),yaw_mae_rad=float(abs(yaw).mean()))
            result[label]=v
        if mani:
            chunk=safety.prepare(row['mani_state'][:6],pred)
            applied=np.array([[*c.joint_position,c.gripper_open_fraction] for c in chunk.commands])
            result.update(saturation_rate=chunk.saturation_rate,position_events=chunk.position_saturation_count,
                rate_events=chunk.rate_saturation_count,gripper_events=chunk.gripper_saturation_count,
                applied_absolute=applied.tolist())
            truth=safety.prepare(row['mani_state'][:6],target)
            # Invalid target padding is never counted as real target clipping evidence.
            vp=target[:n].copy();vp[:,:6]+=np.array(row['mani_state'][:6])
            # Valid-prefix comparison is independent of the masked tail.
            ta=np.array([[*c.joint_position,c.gripper_open_fraction] for c in truth.commands])[:n]
            result['target_valid_unique_modified_fraction']=float((abs(ta-vp)>1e-12).mean())
        return result
    def aggregate(rows,transition_rows):
        result={'rows':len(rows),'queries':len({r['observation_id'] for r in rows}),'families':len({r['family'] for r in rows}),'routes':{}}
        for route in ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'):
            rs=[r for r in rows if r['route']==route]; out={'rows':len(rs),'strata':{}}
            for stratum,selected in [('all',rs),('boundary_1s',[r for r in rs if r['boundary_1s']]),('interior',[r for r in rs if not r['boundary_1s']]),('partial',[r for r in rs if sum(r['mask'])<10])]:
                block={'rows':len(selected)}
                for window in ('first','executed_prefix','full_valid'):
                    keys=('normalized_rmse','joint_mae_rad','gripper_mae','gripper_accuracy') if route in ('PICK','PLACE') else ('normalized_rmse','xy_ade_m','xy_fde_m','yaw_mae_rad')
                    block[window]={k:cluster_mean([r['metrics'][window][k] for r in selected],[r['family'] for r in selected]) for k in keys}
                    sse=sum(r['metrics'][window]['model_sse'] for r in selected);zero=sum(r['metrics'][window]['zero_hold_sse'] for r in selected)
                    block[window]['skill_vs_zero_hold']=1-sse/zero if zero else None
                out['strata'][stratum]=block
            result['routes'][route]=out
        mani=[r for r in rows if 'saturation_rate' in r['metrics']]
        sm=cluster_mean([r['metrics']['saturation_rate'] for r in mani],[r['family'] for r in mani])
        result['saturation_gate']={'limit':.005,'statistics':sm,'passed':sm['sample_mean'] is not None and sm['sample_mean']<=.005}
        result['finite_rate']=sum(r['metrics']['finite'] for r in rows)/len(rows)
        result['transition']={'queries':len(transition_rows),'schema_valid':sum(r['schema_valid'] for r in transition_rows),
            'accuracy':cluster_mean([r['correct'] for r in transition_rows],[r['family'] for r in transition_rows]),
            'confusion':{t:{p:sum(r['label']['operation']==t and r['predicted_operation']==p for r in transition_rows) for p in ('CONTINUE','ADVANCE','INVALID')} for t in ('CONTINUE','ADVANCE')}}
        return result
    for split in ('train',):
        pool=[r for r in actions if r['split']==split and r['task_family_id']==family];tr=[r for r in transitions if r['split']==split and r['task_family_id']==family]
        pool.sort(key=lambda r:(r['episode_uuid'],r['query_time_s']));tr.sort(key=lambda r:(r['query_time_s'],r['observation_id']));rows=[];trs=[]
        event('split_started',split=split,action_queries=len(pool),transition_queries=len(tr))
        with (a.output/(split+'_predictions.jsonl')).open('x') as stream:
            for index,row in enumerate(pool):
                deadline();tick=time.monotonic()
                for name in row['images']:
                    path=(Path(row['episode_root'])/name).resolve()
                    if sha(path)!=selection['image_hashes'][str(path)]:raise ValueError('source RGB changed after selection')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    conditions=encode_rgb_conditions(backend.policy,row['original_instruction'],row['route'],images(row),backend.task_cache,context=public_context_text(row['task_context']))
                head=backend.model.manipulation if row['route'] in ('PICK','PLACE') else backend.model.navigation
                parameter=next(head.parameters())
                conditions={k:v.clone().to(device=parameter.device,dtype=parameter.dtype if v.is_floating_point() else v.dtype) for k,v in conditions.items()}
                state=torch.tensor(backend.normalizer.normalize_mani_state(row['mani_state']),device=parameter.device,dtype=parameter.dtype)[None] if row['route'] in ('PICK','PLACE') else None
                for seed in protocol['seeds']:
                    torch.manual_seed(seed)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        normalized=backend.model.sample('MANIPULATION' if state is not None else 'NAVIGATION',state,conditions)
                    norm=normalized[0].float().cpu().numpy();prediction=backend.normalizer.denormalize_action(row['route'],norm)
                    r=dict(observation_id=row['observation_id'],family=row['task_family_id'],episode_uuid=row['episode_uuid'],query_time_s=row['query_time_s'],route=row['route'],diffusion_seed=seed,mask=row['action_valid_mask'],
                        boundary_1s=any(abs(row['query_time_s']-t)<=1 for t in boundaries.get(row['episode_uuid'],())),
                        normalized_prediction=norm.tolist(),prediction=prediction.tolist(),target=row['actions'],images=row['images'],image_times_s=row['image_times_s'],
                        metrics=metrics(row,prediction,norm),
                        measured_gripper=row['mani_state'][12],target_gripper=[x[6] for x in row['actions']] if state is not None else None,
                        predicted_gripper=prediction[:,6].tolist() if state is not None else None)
                    rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n')
                stream.flush()
                if index<2 or (index+1)%25==0:event('action_progress',split=split,queries=index+1,total=len(pool),query_wall_s=time.monotonic()-tick)
        with (a.output/(split+'_transitions.jsonl')).open('x') as stream:
            for index,row in enumerate(tr):
                deadline();valid=True;error=None;prediction=None
                for name in row['input']['images']:
                    path=(Path(row['episode_root'])/name).resolve()
                    if sha(path)!=selection['image_hashes'][str(path)]:raise ValueError('transition RGB changed after selection')
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=propose_transition(backend.policy.qwen,images(row),instruction=row['input']['original_instruction'],context=row['input']['task_context'])['proposal']
                except (ValueError,TypeError,KeyError) as exc:valid=False;error=repr(exc)
                r=dict(observation_id=row['observation_id'],family=row['task_family_id'],parent_memory_id=row['parent_memory_id'],label=row['label'],prediction=prediction,schema_valid=valid,error=error,
                    predicted_operation=prediction['operation'] if valid else 'INVALID',correct=valid and prediction==row['label'])
                trs.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
                if (index+1)%10==0:event('transition_progress',split=split,queries=index+1,total=len(tr))
        report=dict(status='complete',split=split,full_split=False,full_selected_family=True,protocol=protocol,metrics=aggregate(rows,trs),
            action_coverage=len(rows)==len(pool)*len(protocol['seeds']),transition_coverage=len(trs)==len(tr),
            predictions_sha256=sha(a.output/(split+'_predictions.jsonl')),transitions_sha256=sha(a.output/(split+'_transitions.jsonl')))
        write(a.output/(split+'_report.json'),report);event('split_complete',split=split,metrics=report['metrics'])
    if source_files!={str(p.relative_to(a.repo)):sha(p) for p in (a.repo/'src/conveyor_bench/conveyorvla').rglob('*.py')}:raise ValueError('model/codec source changed during fit')
    event('finished',interpretation='fit on recorded training episode observations; not free-running physical success')

if __name__=='__main__':main()
