#!/usr/bin/env python3
"""Freeze current RGB and task-only Qwen tokens for candidate action training."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from PIL import Image
from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint,load_formal_policy
from conveyor_bench.conveyorvla.staged_data import jsonl,digest
from conveyor_bench.conveyorvla.staged_rgb_backend import encode_rgb_conditions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model-root',type=Path,default=ROOT/'artifacts/models/base')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--max-rows',type=int,default=0)
    p.add_argument('--modalities',choices=('rgb','rgbd'),default='rgbd')
    p.add_argument('--splits',nargs='+',choices=('train','validation','test'),default=['train','validation'])
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('condition cache output must be new')
    if not 0<=a.shard_index<a.shard_count:raise ValueError('invalid cache shard')
    manifest=json.loads((a.release/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='staged-release-v2':raise ValueError('unsupported release')
    if manifest.get('input_modalities',a.modalities)!=a.modalities:raise ValueError('release/cache input modalities mismatch')
    for name,expected in manifest['files'].items():
        if digest(a.release/name)!=expected:raise ValueError('release checksum mismatch')
    binding=validate_formal_checkpoint(a.checkpoint,ROOT/'configs/manipulation_navi_v1.json')
    policy=load_formal_policy(binding,a.model_root,device=a.device)
    a.output.mkdir(parents=True);entries={};task_cache={}
    rows=[r for r in jsonl(a.release/'actions.jsonl') if r['split'] in a.splits]
    if a.max_rows:rows=rows[:a.max_rows]
    rows=rows[a.shard_index::a.shard_count]
    with torch.inference_mode():
        for index,row in enumerate(rows):
            started=time.perf_counter()
            episode=Path(row['episode_root']);task=json.loads((episode/'task.json').read_text(encoding='utf-8'))
            instruction=task.get('original_instruction',task.get('base_instruction',task.get('instruction')));primitive=row['route']
            if not instruction:raise ValueError('missing source instruction')
            images=[]
            for name in row['images']:
                path=(episode/name).resolve()
                if not path.is_relative_to(episode.resolve()):raise ValueError('image reference escapes episode')
                with Image.open(path) as im:images.append(im.convert('RGB'))
            encoded=encode_rgb_conditions(policy,instruction,primitive,images,task_cache)
            cache={k:row[k] for k in ('observation_id','active_task_id','active_task_epoch','query_time_s')}
            cache.update(encoder_model_id=binding['weights_sha256'],calibration_id=row.get('calibration_id'),**encoded)
            if a.modalities=='rgbd' and row.get('depth') is not None:
                import numpy as np
                from conveyor_bench.conveyorvla.geometry_encoder import project_depth
                points_all=[];valid_all=[]
                for d in (row['depth'] if isinstance(row['depth'],list) else [row['depth']]):
                    def load(name):
                        path=(episode/name).resolve()
                        if not path.is_relative_to(episode.resolve()):raise ValueError('depth path escapes episode')
                        if path.suffix.lower()=='.png':
                            with Image.open(path) as im:
                                array=np.asarray(im)
                            if array.dtype!=np.uint16:raise ValueError('depth PNG must be uint16')
                            return array.copy()
                        return np.load(path,allow_pickle=False)
                    raw_depth=load(d['path'])
                    valid_array=load(d['valid_path']) if d.get('valid_path') else raw_depth>0
                    z=torch.as_tensor(raw_depth.astype(np.float32),dtype=torch.float32)[None]
                    valid=torch.as_tensor(valid_array,dtype=torch.bool)[None]
                    points,valid=project_depth(z,valid,torch.tensor(d['intrinsics'],dtype=torch.float32)[None],
                        torch.tensor(d['camera_to_base'],dtype=torch.float32)[None],definition=d['definition'],
                        unit_scale=d['unit_scale'],pixel_offset=d.get('pixel_offset',0.))
                    points_all.append(points);valid_all.append(valid)
                cache.update(points=torch.cat(points_all,dim=1),depth_valid=torch.cat(valid_all,dim=1))
            name=f'condition-{index:08d}.pt';torch.save(cache,a.output/name)
            entries[row['episode_uuid']+':'+row['observation_id']]={'file':name,'sha256':digest(a.output/name)}
            print(json.dumps({'cached_row':index+1,'route':row['route'],'wall_s':time.perf_counter()-started,
                'live_token_shape':list(cache['live_tokens'].shape),'depth_points':None if 'points' not in cache else list(cache['points'].shape)}),flush=True)
    (a.output/'manifest.json').write_text(json.dumps({'schema':'staged-condition-cache-v2',
        'release_sha256':digest(a.release/'manifest.json'),'encoder_model_id':binding['weights_sha256'],
        'input_modalities':a.modalities,'splits':a.splits,'shard_count':a.shard_count,'shard_index':a.shard_index,
        'entries':entries},indent=2)+'\n')
    print(json.dumps({'rows':len(entries),'encoder_model_id':binding['weights_sha256'],'training_started':False}))

if __name__=='__main__':main()
