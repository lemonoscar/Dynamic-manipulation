#!/usr/bin/env python3
"""Loopback RGB full-checkpoint proposals; execution and task evidence stay client-side."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
PROTOCOL = 'full-episode-rgb-diagnostic-v1'


def validate_payload(payload):
    required = {'protocol_version', 'request', 'instruction', 'head_images', 'wrist_images',
                'diffusion_seed', 'predict_transition'}
    if set(payload) != required or payload['protocol_version'] != PROTOCOL:
        raise ValueError('explicit full RGB protocol required; undeclared fields forbidden')
    r = payload['request']
    if set(r) != {'request_id', 'identity', 'plan_version', 'observation', 'task',
                  'rtc_context', 'time_profile', 'plan_context'}:
        raise ValueError('action request allowlist differs')
    if r['time_profile'] != 'causal_command_5hz':
        raise ValueError('full checkpoint requires causal_command_5hz')
    o = r['observation']
    if set(o) != {'mission_id', 'observation_id', 'time_s', 'q', 'dq', 'gripper',
                  'image_times_s', 'base_xyyaw'}:
        raise ValueError('observation allowlist differs')
    t = float(o['time_s'])
    if not math.isfinite(t) or t < .2 or len(o['image_times_s']) != 4:
        raise ValueError('invalid camera history')
    if any(not math.isfinite(a) or abs(a-b) > 1e-7 for a, b in zip(o['image_times_s'], (t-.2,t,t-.2,t))):
        raise ValueError('camera history must be real query and query-0.2s')
    seed = payload['diffusion_seed']
    if type(seed) is not int or not 0 <= seed < 2**32 or type(payload['predict_transition']) is not bool:
        raise ValueError('invalid seed/transition flag')
    return r


class FullEpisodeService:
    def __init__(self, backend, identity):
        self.backend, self.identity = backend, identity

    def health(self):
        return {'ok': True, 'protocol_version': PROTOCOL, **self.identity,
                'normalizer': self.backend.normalizer.payload}

    def infer(self, payload):
        import torch
        from scripts.serve_joint_trajectory import _decode_pair
        from conveyor_bench.conveyorvla.contracts.action import ActionIdentity
        from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
        from conveyor_bench.conveyorvla.task_memory import Task
        from conveyor_bench.conveyorvla.rolling_runtime import ActionRequest
        from conveyor_bench.conveyorvla.full_episode_backend import propose_transition
        r = validate_payload(payload)
        images = (*_decode_pair(payload['head_images'], 'head_images'),
                  *_decode_pair(payload['wrist_images'], 'wrist_images'))
        request = ActionRequest(**{**r, 'identity': ActionIdentity(**r['identity']),
            'task': Task(**r['task']), 'observation': CurrentObservation(**r['observation'], images=images)})
        torch.manual_seed(payload['diffusion_seed'])
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actions = self.backend(request, payload['instruction'])
            proposal = None
            if payload['predict_transition']:
                proposal = propose_transition(self.backend.policy.qwen, images,
                    instruction=payload['instruction'], context=request.plan_context)
        return {'request_id': request.request_id, 'identity': r['identity'],
                'observation_id': request.observation.observation_id,
                'query_time_s': request.observation.time_s, 'time_profile': request.time_profile,
                'physical_actions': actions.tolist(), 'transition': proposal,
                'rtc_applied': request.rtc_context is not None,
                'server_wall_s': time.perf_counter()-started,
                'weights_sha256': self.identity['weights_sha256']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--legacy-checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--port', type=int, default=18170)
    p.add_argument('--identity-output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from http.server import HTTPServer
    from scripts.serve_joint_trajectory import _Handler
    from conveyor_bench.conveyorvla.full_episode_backend import load_full_episode_backend
    from conveyor_bench.conveyorvla.staged_data import digest
    if args.identity_output.exists():
        raise ValueError('identity output must be new')
    if digest(args.checkpoint) != args.expected_sha256:
        raise ValueError('checkpoint hash changed')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    step = saved.get('steps')
    if saved.get('schema') != 'full-episode-vla-weights-v1' or step != 1700:
        raise ValueError(f'expected step 1700, got {step}')
    del saved
    backend = load_full_episode_backend(args.checkpoint, legacy_checkpoint=args.legacy_checkpoint,
                                       repo_root=ROOT, device='cuda:0')
    import subprocess
    identity = {'checkpoint_id': backend.model_id, 'weights_sha256': args.expected_sha256,
                'global_step': 1700, 'strict_load': True, 'time_profile': 'causal_command_5hz',
                'master_weight_dtype': 'float32', 'autocast': 'bfloat16', 'depth': False,
                'source_head': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                'service_sha256': digest(Path(__file__))}
    service = FullEpisodeService(backend, identity)
    with args.identity_output.open('x') as out:
        json.dump(service.health(), out, indent=2)
    server = HTTPServer(('127.0.0.1', args.port), _Handler)
    server.service = service
    print(json.dumps({'event': 'ready', 'identity': identity, 'port': args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
