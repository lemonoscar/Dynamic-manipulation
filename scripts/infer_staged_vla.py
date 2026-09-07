#!/usr/bin/env python3
"""Offline candidate inference from an identity-bound live condition cache.

Outputs are normalized proposals, not authorized robot commands. Use the same
normalizer and the versioned queue/safety boundary before any execution.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.staged_experts import StagedConfig,StagedExperts
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer,validate_condition_cache


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--condition',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True,help='one action-view-v2 record, only observation metadata/state consumed')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--prefix',type=Path,help='normalized already aligned clean prefix for trained RTC only')
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('new proposal output required')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if saved.get('training_steps',0)<=0:raise ValueError('candidate needs trained checkpoint identity')
    config=StagedConfig(**saved['candidate']);normalizer=StagedNormalizer(saved['normalizer'])
    row=json.loads(a.record.read_text());cache=torch.load(a.condition,map_location='cpu',weights_only=True)
    conditions=validate_condition_cache(cache,row,saved['encoder_model_id'])
    if not config.depth:conditions={k:v for k,v in conditions.items() if k not in {'points','depth_valid'}}
    model=StagedExperts(config,M0DiTConfig(vlm_hidden_dim=2560,hidden_size=1024,action_horizon=10))
    model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
    conditions={k:v.to(a.device) for k,v in conditions.items()}
    mani=row['route'] in {'PICK','PLACE'}
    state=torch.tensor(normalizer.normalize_mani_state(row['mani_state']),dtype=torch.float32,device=a.device)[None] if mani else None
    kwargs={}
    if a.prefix:
        prefix=torch.tensor(json.loads(a.prefix.read_text()),dtype=torch.float32,device=a.device)[None]
        kwargs=dict(prefix=prefix,prefix_length=prefix.shape[1])
    torch.manual_seed(a.seed)
    normalized=model.sample_prefix('MANIPULATION' if mani else 'NAVIGATION',state,conditions,**kwargs)
    physical=normalizer.denormalize_action(row['route'],normalized)[0]
    output={'schema':'staged-action-proposal-v2','observation_id':row['observation_id'],
        'active_task_epoch':row['active_task_epoch'],'active_task_id':row['active_task_id'],
        'normalizer_id':normalizer.payload['normalizer_id'],'time_profile':row['time_profile'],
        'query_time_s':row['query_time_s'],'physical_relative_proposal':physical.cpu().tolist(),
        'execution_authorized':False,'safety_limits_applied':False}
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
