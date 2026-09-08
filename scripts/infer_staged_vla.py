#!/usr/bin/env python3
"""Offline candidate inference from an identity-bound live condition cache.

Outputs are normalized proposals, not authorized robot commands. Use the same
normalizer and the versioned queue/safety boundary before any execution.
"""
import argparse,json,sys,math
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.staged_experts import StagedConfig,StagedExperts
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer,validate_condition_cache
from conveyor_bench.conveyorvla.staged_data import digest
from conveyor_bench.conveyorvla.contracts.action import TIME_PROFILES,encode_mani,finite_array


def validate_rtc_context(context,row,config,normalizer,checkpoint_sha256):
    """Bind prior effective absolute targets before reanchoring for VJP guidance.

    The queue/controller owns provenance: this verifies the supplied identity,
    available intervals and codec, and never invents an unavailable terminal hold.
    """
    if config.training_rtc or config.depth or config.coupling!='C0' or config.bounded_gripper:
        raise ValueError('RTC context requires RGB C0 inference-VJP candidate')
    if row.get('route') not in {'PICK','PLACE'}:
        raise ValueError('inference VJP RTC is only defined for Mani')
    expected={'schema':'staged-rtc-context-v1','checkpoint_sha256':checkpoint_sha256,
        'normalizer_id':normalizer.payload['normalizer_id'],'episode_uuid':row['episode_uuid'],
        'active_task_id':row['active_task_id'],'active_task_epoch':row['active_task_epoch'],
        'observation_id':row['observation_id'],'query_time_s':row['query_time_s'],
        'time_profile':config.time_profile}
    if row.get('time_profile')!=config.time_profile or any(context.get(k)!=v for k,v in expected.items()):
        raise ValueError('stale/foreign RTC checkpoint, task, observation or time profile')
    query=float(row['query_time_s']);source_time=float(context['source_query_time_s'])
    if not math.isfinite(query) or not math.isfinite(source_time) or not 0<=source_time<query or not context.get('source_action_id'):
        raise ValueError('RTC prior action needs earlier query and source action identity')
    profile=TIME_PROFILES[config.time_profile];horizon=profile.horizon
    applies=finite_array(context['target_apply_times_s'],(horizon,),'RTC apply times')
    if not np.allclose(applies,profile.apply_times(query),atol=1e-7,rtol=0):
        raise ValueError('RTC prior targets do not align with current actual apply intervals')
    absolute=finite_array(context['previous_absolute'],(horizon,7),'RTC effective absolute targets')
    weights=finite_array(context['weights'],(horizon,),'RTC weights')
    available=np.asarray(context['available_previous'])
    if available.shape!=(horizon,) or available.dtype!=bool or (weights<0).any() or (weights>1).any() or (weights[~available]!=0).any():
        raise ValueError('unavailable RTC targets need zero guidance')
    if ((absolute[available,6]<0)|(absolute[available,6]>1)).any():
        raise ValueError('RTC effective gripper must be calibrated open_fraction')
    gain=float(context.get('max_guidance_weight',5.))
    if not math.isfinite(gain) or not 0<=gain<=5:
        raise ValueError('RTC gain exceeds frozen bound')
    previous=normalizer.normalize_action(row['route'],encode_mani(absolute,row['mani_state'][:6]))
    return {'previous':np.asarray(previous), 'weights':weights,
        'max_guidance_weight':gain}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--condition',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True,help='one action-view-v2 record, only observation metadata/state consumed')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--prefix',type=Path,help='reserved; unbound prefix files are rejected')
    p.add_argument('--rtc-context',type=Path,help='identity-bound effective absolute targets for inference VJP RTC')
    a=p.parse_args(argv)
    if a.prefix:
        raise ValueError('unbound prefix file: task, normalizer and apply-time identity required; CLI prefix deployment is unavailable')
    if a.output.exists():raise ValueError('new proposal output required')
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if saved.get('training_steps',0)<=0:raise ValueError('candidate needs trained checkpoint identity')
    config=StagedConfig(**saved['candidate']);normalizer=StagedNormalizer(saved['normalizer'])
    row=json.loads(a.record.read_text())
    if row.get('time_profile')!=config.time_profile:
        raise ValueError('record/checkpoint action time profile mismatch')
    if row.get('route') not in {'PICK','PLACE','NAV_TO_SOURCE','NAV_TO_TARGET'}:
        raise ValueError('unknown action route')
    cache=torch.load(a.condition,map_location='cpu',weights_only=True)
    conditions=validate_condition_cache(cache,row,saved['encoder_model_id'])
    if not config.depth:conditions={k:v for k,v in conditions.items() if k not in {'points','depth_valid'}}
    rtc_context=None
    if a.rtc_context:
        rtc_context=validate_rtc_context(json.loads(a.rtc_context.read_text()),row,config,normalizer,digest(a.checkpoint))
    model=StagedExperts(config,M0DiTConfig(vlm_hidden_dim=2560,hidden_size=1024,action_horizon=10))
    model.load_state_dict(saved['model'],strict=True);model.to(a.device).eval()
    conditions={k:v.to(a.device) for k,v in conditions.items()}
    mani=row['route'] in {'PICK','PLACE'}
    state=torch.tensor(normalizer.normalize_mani_state(row['mani_state']),dtype=torch.float32,device=a.device)[None] if mani else None
    torch.manual_seed(a.seed)
    if config.training_rtc:
        if rtc_context is not None:raise ValueError('inference VJP and training-prefix RTC are separate contracts')
        normalized=model.sample_prefix('MANIPULATION' if mani else 'NAVIGATION',state,conditions)
    else:
        normalized=model.sample('MANIPULATION' if mani else 'NAVIGATION',state,conditions,rtc_context=rtc_context)
    physical=normalizer.denormalize_action(row['route'],normalized)[0]
    output={'schema':'staged-action-proposal-v2','observation_id':row['observation_id'],
        'active_task_epoch':row['active_task_epoch'],'active_task_id':row['active_task_id'],
        'normalizer_id':normalizer.payload['normalizer_id'],'time_profile':config.time_profile,
        'query_time_s':row['query_time_s'],'physical_relative_proposal':physical.cpu().tolist(),
        'rtc_applied':rtc_context is not None,'execution_authorized':False,'safety_limits_applied':False}
    a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
