#!/usr/bin/env python3
"""Loopback-only fixed-PICK RTC proposal protocol, distinct from autonomous policy."""
import json,math,time
from http.server import HTTPServer
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from scripts import serve_joint_trajectory as server
from scripts.serve_conditioned_pick import ConditionedPickService,PROTOCOL as BASE_PROTOCOL
from conveyor_bench.conveyorvla.contracts.action import encode_mani,finite_array

PROTOCOL='conveyorvla-conditioned-pick-rtc-diagnostic/v1'
BASE_FIELDS={'request_id','episode_id','sequence_id','instruction','head_images','wrist_images',
    'joint_position','joint_velocity','gripper_open_fraction'}
RTC_FIELDS={'protocol_version','observation_id','query_time_s','image_times_s','active_task_id',
    'active_task_epoch','time_profile','rtc_context'}


def validate_context(payload,session,weights_sha256):
    if set(payload)-(BASE_FIELDS|RTC_FIELDS) or payload.get('protocol_version')!=PROTOCOL:
        raise ValueError('invalid RTC diagnostic fields/protocol')
    if payload.get('time_profile')!='legacy_future_5hz':
        raise ValueError('frozen checkpoint requires its legacy time profile')
    query=float(payload['query_time_s']);times=finite_array(payload['image_times_s'],(4,),'image times')
    if not math.isfinite(query) or query<.2 or not np.allclose(times,(query-.2,query)*2,atol=1e-7,rtol=0):
        raise ValueError('RTC requires exact causal 0.2s camera history')
    if type(payload.get('sequence_id')) is not int or payload['sequence_id']<0:
        raise ValueError('invalid query sequence')
    if not payload.get('observation_id') or not payload.get('active_task_id') or type(payload['active_task_epoch']) is not int:
        raise ValueError('missing observation/task identity')
    context=payload.get('rtc_context')
    if context is None:return None
    allowed={'identity','source_action_id','source_query_time_s','source_sequence_id','previous_absolute',
        'target_apply_times_s','available_previous','weights','max_guidance_weight'}
    if set(context)-allowed or not context.get('source_action_id'):
        raise ValueError('invalid prior action provenance')
    identity={'episode_id':payload['episode_id'],'active_task_id':payload['active_task_id'],
        'active_task_epoch':payload['active_task_epoch'],'weights_sha256':weights_sha256,
        'normalization_sha256':session.normalization_sha256}
    if context.get('identity')!=identity:raise ValueError('foreign/stale RTC action identity')
    if type(context.get('source_sequence_id')) is not int or context['source_sequence_id']<0:
        raise ValueError('invalid source sequence')
    source_time=float(context['source_query_time_s'])
    if not math.isfinite(source_time) or not 0<=source_time<query or not context['source_sequence_id']<payload['sequence_id']:
        raise ValueError('prior request is not earlier than current query')
    applies=finite_array(context['target_apply_times_s'],(10,),'actual target apply times')
    if not np.allclose(applies,query+np.arange(10)*.2,atol=1e-7,rtol=0):
        raise ValueError('RTC old targets must align to current actual apply intervals')
    absolute=finite_array(context['previous_absolute'],(10,7),'absolute prior targets')
    weights=finite_array(context['weights'],(10,),'RTC weights')
    available=np.asarray(context['available_previous'])
    if available.shape!=(10,) or available.dtype!=bool or (weights<0).any() or (weights>1).any() or (weights[~available]!=0).any():
        raise ValueError('unavailable old targets must have zero RTC guidance')
    if ((absolute[available,6]<0)|(absolute[available,6]>1)).any():
        raise ValueError('available prior gripper targets exceed calibrated range')
    gain=float(context.get('max_guidance_weight',5.))
    if not math.isfinite(gain) or not 0<=gain<=5:raise ValueError('RTC gain exceeds frozen diagnostic bound')
    q=server._vector(payload['joint_position'],6,'joint_position')
    previous=session.normalizer.normalize_action('PICK',encode_mani(absolute,q))
    return {'previous':np.asarray(previous).tolist(),'weights':weights.tolist(),'max_guidance_weight':gain}


class ConditionedPickRTCService:
    def __init__(self,service):
        self.service=service;self.base=ConditionedPickService(service)

    def health(self):
        return {**self.base.health(),'protocol_version':PROTOCOL,'rtc':'VJP_sampling',
            'time_profile':'legacy_future_5hz','previous_input':'absolute_targets_aligned_by_actual_apply_time'}

    def infer(self,payload):
        started=time.perf_counter()
        context=validate_context(payload,self.service.session,self.service.identity['weights_sha256'])
        basic={key:payload[key] for key in BASE_FIELDS};basic['protocol_version']=BASE_PROTOCOL
        response=self.base.infer(basic,rtc_context=context)
        return {**response,'protocol_version':PROTOCOL,'rtc_applied':context is not None,
            'request_id':payload['request_id'],'sequence_id':payload['sequence_id'],
            'observation_id':payload['observation_id'],'query_time_s':payload['query_time_s'],
            'active_task_id':payload['active_task_id'],'active_task_epoch':payload['active_task_epoch'],
            'time_profile':'legacy_future_5hz','service_elapsed_s':time.perf_counter()-started,
            'execution_authorized':False}


def main():
    args=server.build_parser().parse_args();service,_=server.load_service(args)
    http=HTTPServer(('127.0.0.1',args.port),server._Handler);http.service=ConditionedPickRTCService(service)
    print(json.dumps(http.service.health()),flush=True)
    try:http.serve_forever()
    finally:http.server_close()

if __name__=='__main__':main()
