#!/usr/bin/env python3
"""Freeze current RGB and task-only Qwen tokens for candidate action training."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from PIL import Image
from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint,load_formal_policy
from conveyor_bench.conveyorvla.staged_data import jsonl,digest
from conveyor_bench.conveyorvla.joint_trajectory import canonical_solution


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=ROOT/'artifacts/models/base')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--max-rows',type=int,default=0)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('condition cache output must be new')
    manifest=json.loads((a.release/'manifest.json').read_text())
    if manifest['schema']!='staged-release-v2':raise ValueError('unsupported release')
    for name,expected in manifest['files'].items():
        if digest(a.release/name)!=expected:raise ValueError('release checksum mismatch')
    binding=validate_formal_checkpoint(a.checkpoint,ROOT/'configs/manipulation_navi_v1.json')
    policy=load_formal_policy(binding,a.model_root,device=a.device)
    a.output.mkdir(parents=True);entries={};task_cache={}
    rows=jsonl(a.release/'actions.jsonl')
    if a.max_rows:rows=rows[:a.max_rows]
    with torch.inference_mode():
        for index,row in enumerate(rows):
            episode=Path(row['episode_root']);task=json.loads((episode/'task.json').read_text())
            instruction=task['original_instruction'];primitive=row['route']
            # New data does not supervise nonexistent PLACE retreat.
            subtask='Lower, release, and verify placement.' if primitive=='PLACE' else primitive
            semantic=instruction+'\nCurrent task: '+subtask
            task_key=(row['episode_uuid'],row['active_task_epoch'],semantic)
            if task_key not in task_cache:
                tokens=policy.qwen.processor.tokenizer(semantic,return_tensors='pt')
                tokens={k:v.to(a.device) for k,v in tokens.items()}
                out=policy.qwen(**tokens,output_hidden_states=True,return_dict=True,use_cache=False)
                task_cache[task_key]=(out.hidden_states[-1].float().cpu(),tokens['attention_mask'].bool().cpu())
            images=[]
            for name in row['images']:
                path=(episode/name).resolve()
                if not path.is_relative_to(episode.resolve()):raise ValueError('image reference escapes episode')
                with Image.open(path) as im:images.append(im.convert('RGB'))
            inputs=dict(policy.qwen.build_joint_trajectory_inputs([{'video':(images[:2],images[2:]),
                'lang':semantic}],solutions=[canonical_solution(primitive).replace(
                    'Lower, release, and retract from the destination box.','Lower, release, and verify placement.')],
                supervise_solutions=False))
            inputs.pop('labels',None)
            out=policy.qwen(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
            task_tokens,task_mask=task_cache[task_key]
            cache={k:row[k] for k in ('observation_id','active_task_id','active_task_epoch','query_time_s')}
            cache.update(encoder_model_id=binding['weights_sha256'],calibration_id=row.get('calibration_id'),
                task_tokens=task_tokens,task_mask=task_mask,live_tokens=out.hidden_states[-1].float().cpu(),
                live_mask=inputs['attention_mask'].bool().cpu())
            if row.get('depth') is not None:
                import numpy as np
                from conveyor_bench.conveyorvla.geometry_encoder import project_depth
                d=row['depth']
                def load(name):
                    path=(episode/name).resolve()
                    if not path.is_relative_to(episode.resolve()):raise ValueError('depth path escapes episode')
                    return np.load(path,allow_pickle=False)
                z=torch.as_tensor(load(d['path']),dtype=torch.float32)[None]
                valid=torch.as_tensor(load(d['valid_path']),dtype=torch.bool)[None]
                points,valid=project_depth(z,valid,torch.tensor(d['intrinsics'],dtype=torch.float32)[None],
                    torch.tensor(d['camera_to_base'],dtype=torch.float32)[None],definition=d['definition'],unit_scale=d['unit_scale'])
                cache.update(points=points,depth_valid=valid)
            name=f'condition-{index:08d}.pt';torch.save(cache,a.output/name)
            entries[row['episode_uuid']+':'+row['observation_id']]={'file':name,'sha256':digest(a.output/name)}
    (a.output/'manifest.json').write_text(json.dumps({'schema':'staged-condition-cache-v2',
        'release_sha256':digest(a.release/'manifest.json'),'encoder_model_id':binding['weights_sha256'],
        'entries':entries},indent=2)+'\n')
    print(json.dumps({'rows':len(entries),'encoder_model_id':binding['weights_sha256'],'training_started':False}))

if __name__=='__main__':main()
