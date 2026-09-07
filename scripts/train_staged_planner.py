#!/usr/bin/env python3
"""Prepare or explicitly train a separate task planner from causal task-view-v2 labels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from conveyor_bench.conveyorvla.staged_data import jsonl,digest,RawEpisode
from conveyor_bench.conveyorvla.rolling_planner import planner_prompt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=ROOT/'artifacts/models/base');p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--effective-batch',type=int,default=16);p.add_argument('--learning-rate',type=float,default=2e-6)
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=20260907)
    a=p.parse_args(argv)
    if min(a.steps,a.effective_batch,a.learning_rate)<=0:raise ValueError('positive training budgets required')
    manifest=json.loads((a.release/'manifest.json').read_text())
    if manifest.get('schema')!='staged-release-v2' or manifest.get('synthetic'):
        raise ValueError('planner training requires a nonsynthetic verified task release')
    if digest(a.release/'tasks.jsonl')!=manifest['files']['tasks.jsonl']:
        raise ValueError('task view checksum mismatch')
    rows=[r for r in jsonl(a.release/'tasks.jsonl') if r['split']=='train']
    if not rows or any(r.get('synthetic') for r in rows):raise ValueError('no nonsynthetic planner train labels')
    episodes={r['episode_root']:RawEpisode(r['episode_root']) for r in rows}
    # Re-derive labels to catch edited/stale contexts, not just tensor shape.
    for root,episode in episodes.items():
        derived={r['observation_id']:r for r in episode.task_view()}
        for row in (r for r in rows if r['episode_root']==root):
            expected=derived[row['observation_id']]
            if row['input']!=expected['input'] or row['label']!=expected['label']:
                raise ValueError('task view differs from causal replay')
    if not a.execute:
        print(json.dumps({'status':'preflight_passed','rows':len(rows),'mode':'H2_adaptation_shared_phiP',
            'training_started':False,'steps':a.steps,'effective_batch':a.effective_batch}));return
    if not a.output or a.output.exists() or a.output.resolve().is_relative_to(ROOT):
        raise ValueError('planner training output must be NEW and outside worktree')
    import torch
    from PIL import Image
    from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint,load_formal_policy
    binding=validate_formal_checkpoint(a.checkpoint,ROOT/'configs/manipulation_navi_v1.json')
    policy=load_formal_policy(binding,a.model_root,device=a.device)
    qwen=policy.qwen
    del policy  # Separate planner service; no action experts are trained here.
    qwen.train();torch.manual_seed(a.seed)
    optimizer=torch.optim.AdamW(qwen.parameters(),lr=a.learning_rate)
    a.output.mkdir(parents=True)
    (a.output/'resolved.json').write_text(json.dumps({'schema':'planner-adaptation-v2','steps':a.steps,
        'effective_batch':a.effective_batch,'seed':a.seed,'learning_rate':a.learning_rate,
        'release_sha256':digest(a.release/'manifest.json'),'initialization_sha256':binding['weights_sha256'],
        'mode':'H2_adaptation_shared_phiP','low_level_checkpoint_unchanged':True},indent=2)+'\n')
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True);total=0.
        for index in torch.randint(len(rows),(a.effective_batch,)).tolist():
            row=rows[index];episode=episodes[row['episode_root']]
            obs=episode.observations[row['observation_id']][0];images=[]
            for name in obs.images:
                with Image.open(episode.root/name) as im:images.append(im.convert('RGB'))
            request={'mode':'H2','context':row['input'],
                'allowed_operations':['CONTINUE','ADVANCE','REPAIR_SUFFIX','FINISH','UNRESOLVED']}
            inputs=dict(qwen.build_joint_trajectory_inputs([{'video':(images[:2],images[2:]),
                'lang':planner_prompt(request)}],solutions=[json.dumps(row['label'])],supervise_solutions=True))
            output=qwen(**inputs,return_dict=True)
            loss=output.loss/a.effective_batch
            if not torch.isfinite(loss):raise ValueError('nonfinite planner CE loss')
            loss.backward();total+=float(loss)
        torch.nn.utils.clip_grad_norm_(qwen.parameters(),1.);optimizer.step()
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'loss':total})+'\n')
    qwen.model.save_pretrained(a.output/'planner');qwen.processor.save_pretrained(a.output/'planner')

if __name__=='__main__':main()
