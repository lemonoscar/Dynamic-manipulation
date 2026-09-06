#!/usr/bin/env python3
"""Diagnostic-only fixed PICK/canonical-prefix service for a frozen checkpoint.

This protocol is deliberately distinct from the autonomous route service. It
accepts RGB and measurable joints only, never object/base/task truth or targets.
"""
from __future__ import annotations
import json
from dataclasses import asdict
from http.server import HTTPServer
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts import serve_joint_trajectory as server
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute, joint_trajectory_prompt
from scripts.evaluate_joint_trajectory_formal import oracle_decision

PROTOCOL='conveyorvla-conditioned-pick-diagnostic/v1'


class ConditionedPickService:
    def __init__(self, service):
        self.service=service

    def health(self):
        return {**self.service.health(),'protocol_version':PROTOCOL,'diagnostic_only':True,
                'forced_route':'PICK','prefix':'canonical','seed_semantics':'same diffusion seed at each query'}

    def infer(self,payload):
        allowed={'protocol_version','request_id','episode_id','sequence_id','instruction',
                 'head_images','wrist_images','joint_position','joint_velocity','gripper_open_fraction'}
        if set(payload)-allowed or payload.get('protocol_version')!=PROTOCOL:
            raise ValueError('invalid diagnostic request fields/protocol')
        q=server._vector(payload['joint_position'],6,'joint_position')
        dq=server._vector(payload['joint_velocity'],6,'joint_velocity')
        grip=float(payload['gripper_open_fraction'])
        if not 0 <= grip <= 1:
            raise ValueError('invalid measured gripper fraction')
        session=self.service.session
        example={'video':(server._decode_pair(payload['head_images'],'head_images'),
                          server._decode_pair(payload['wrist_images'],'wrist_images')),
                 'lang':joint_trajectory_prompt(str(payload['instruction'])),
                 'mani_state':session.normalizer.normalize_mani_state((*q,*dq,grip))}
        decision=oracle_decision({'route':'PICK'})
        torch.manual_seed(self.service.seed);torch.cuda.manual_seed_all(self.service.seed)
        normalized=session.policy.predict_actions([example],[decision])[0]
        if normalized is None:
            raise ValueError('PICK expert returned no action')
        physical=session.normalizer.denormalize_action(JointTrajectoryRoute.PICK,normalized)
        chunk=session.joint_executor.prepare(q,physical)
        return {'protocol_version':PROTOCOL,'diagnostic_only':True,'forced_route':'PICK',
                'assistant_prefix':decision.assistant_prefix,'subtask':decision.subtask_text,
                'normalized_action':normalized,'physical_relative_action':physical,
                'chunk':asdict(chunk),'checkpoint_id':session.checkpoint_id,
                'normalization_sha256':session.normalization_sha256}


def main():
    args=server.build_parser().parse_args()
    service,_=server.load_service(args)
    diagnostic=ConditionedPickService(service)
    http=HTTPServer(('127.0.0.1',args.port),server._Handler)
    http.service=diagnostic
    print(json.dumps(diagnostic.health()),flush=True)
    try:http.serve_forever()
    finally:http.server_close()


if __name__=='__main__':main()
