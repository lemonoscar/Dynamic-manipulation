#!/usr/bin/env python3
"""Paired real-RGB/state interventions; no updates and no physical execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess


def camera_variants(rgb, donor):
    """Intervene on complete temporal pairs; never mix t-.2 with the other camera."""
    from PIL import Image
    if len(rgb) != 4 or len(donor) != 4:
        raise ValueError('four ordered RGB frames required')
    rgb,donor=list(rgb),list(donor)
    black=[Image.new('RGB', im.size) for im in rgb]
    return dict(baseline=rgb, black_front=black[:2]+rgb[2:], black_wrist=rgb[:2]+black[2:],
                swap_front=donor[:2]+rgb[2:], swap_wrist=rgb[:2]+donor[2:],
                black_rgb=black, exchange_camera_roles=rgb[2:]+rgb[:2])


def camera_token_report(policy, instruction, route, rgb, context):
    from conveyor_bench.conveyorvla.staged_rgb_backend import _rgb_live_inputs, rgb_condition_semantic
    semantic=rgb_condition_semantic(instruction,route,rgb,context=context)
    inputs=_rgb_live_inputs(policy,semantic,route,rgb)
    grids=inputs['video_grid_thw'].cpu().tolist()
    merge=policy.qwen.model.config.vision_config.spatial_merge_size
    patch_id=policy.qwen.model.config.video_token_id
    actual=int((inputs['input_ids']==patch_id).sum().item())
    counts=[int(t*h*w)//(merge*merge) for t,h,w in grids]
    if len(grids)!=2 or sum(counts)!=actual:
        raise ValueError('two-camera visual-token count mismatch')
    return dict(camera_order=['front','wrist'],frame_order=['t-0.2','t'],
                original_frame_sizes=[list(im.size) for im in rgb],video_grid_thw=grids,
                spatial_merge_size=merge,visual_tokens_per_camera=counts,
                actual_video_token_count=actual,live_sequence_tokens=int(inputs['attention_mask'].sum().item()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--legacy-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--study', choices=['state','cameras'], default='state')
    p.add_argument('--max-wall-seconds', type=int, default=1200)
    args = p.parse_args()
    if not 0 < args.max_wall_seconds <= 1200:
        raise ValueError('diagnostic wall budget must be in (0,1200] seconds')
    sys.path[:0] = [str(args.repo), str(args.repo / 'src')]
    import numpy as np
    import torch
    from PIL import Image
    from conveyor_bench.conveyorvla.full_episode_backend import load_full_episode_backend
    from conveyor_bench.conveyorvla.staged_rgb_backend import encode_rgb_conditions
    from conveyor_bench.conveyorvla.full_episode_context import public_context_text
    from scripts.train_full_episode_vla import images

    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def deadline():
        if time.monotonic() - start > args.max_wall_seconds:
            raise TimeoutError('diagnostic wall budget exhausted')
    pool = [json.loads(s) for s in (args.release / 'actions.jsonl').open()]
    selected = []
    for family in ('liangzhu_seed_16100021', 'liangzhu_seed_16100026'):
        for route in ('PICK', 'PLACE'):
            rows = sorted([r for r in pool if r['split'] == 'train' and r['task_family_id'] == family
                           and r['route'] == route and r['training_eligible']], key=lambda r: r['query_time_s'])
            if len(rows) < 4:
                raise ValueError('insufficient source queries')
            selected.extend([rows[len(rows)//4], rows[3*len(rows)//4]])
    identity = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    source_head=subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),
                    diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), families=[16100021, 16100026],
                    max_queries=8, max_wall_s=args.max_wall_seconds, optimization_steps=0,study=args.study,
                    interpretation='paired input interventions, not causal modality-importance percentages',
                    variants=(['baseline', 'swap_rgb', 'black_rgb', 'swap_qpos', 'swap_dq', 'swap_state']
                              if args.study=='state' else ['baseline','black_front','black_wrist','swap_front','swap_wrist','black_rgb','exchange_camera_roles']),
                    queries=[{k:r[k] for k in ('task_family_id','route','observation_id','images','query_time_s')} for r in selected])
    (args.output/'protocol.json').write_text(json.dumps(identity, indent=2))
    assert torch.cuda.device_count() == 1
    backend = load_full_episode_backend(args.checkpoint, legacy_checkpoint=args.legacy_checkpoint,
                                        repo_root=args.repo, device='cuda')
    backend.model.eval()
    assert not backend.model.training
    result = []
    for index, row in enumerate(selected):
        deadline()
        donor = selected[(index+4)%8]
        assert donor['route'] == row['route'] and donor['task_family_id'] != row['task_family_id']
        rgb = images(row)
        rgb_variants = dict(baseline=rgb, swap_rgb=images(donor),
                            black_rgb=[Image.new('RGB', im.size) for im in rgb])
        if args.study=='cameras':rgb_variants=camera_variants(rgb,images(donor))
        token_report=camera_token_report(backend.policy,row['original_instruction'],row['route'],rgb,
            public_context_text(row['task_context'])) if args.study=='cameras' else None
        conditions = {}
        for mode, frames in rgb_variants.items():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                bank = encode_rgb_conditions(backend.policy, row['original_instruction'], row['route'], frames,
                    backend.task_cache, context=public_context_text(row['task_context']))
            conditions[mode] = {k:v.clone().to('cuda') for k,v in bank.items()}
        state = torch.tensor(backend.normalizer.normalize_mani_state(row['mani_state']), device='cuda', dtype=torch.float32)[None]
        other = torch.tensor(backend.normalizer.normalize_mani_state(donor['mani_state']), device='cuda', dtype=torch.float32)[None]
        states = {'baseline':state, 'swap_state':other}
        for name, interval in [('swap_qpos',slice(0,6)), ('swap_dq',slice(6,12))]:
            altered=state.clone();altered[:,interval]=other[:,interval];states[name]=altered
        target=np.asarray(row['actions']);valid=np.asarray(row['action_valid_mask'], dtype=bool)
        record={'observation_id':row['observation_id'], 'family':row['task_family_id'], 'route':row['route'],
                'query_time_s':row['query_time_s'], 'donor_observation_id':donor['observation_id'],
                'state_delta_normalized':(other-state).cpu().tolist(), 'camera_token_report':token_report,'draws':[]}
        for seed in (171, 172):
            deadline()
            noise=torch.randn((1,10,7), generator=torch.Generator(device='cuda').manual_seed(seed), device='cuda')
            predictions={}
            for mode in identity['variants']:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    norm=backend.model.sample('MANIPULATION', states.get(mode,state),
                        conditions.get(mode,conditions['baseline']), noise=noise)
                pred=backend.normalizer.denormalize_action(row['route'], norm[0].float().cpu().numpy())
                if not np.isfinite(pred).all(): raise ValueError('nonfinite proposal')
                predictions[mode]=pred
            draw={'seed':seed,'variants':{}}
            for mode,pred in predictions.items():
                err=pred[valid]-target[valid];delta=pred[valid]-predictions['baseline'][valid]
                draw['variants'][mode]=dict(joint_target_mae_rad=float(np.abs(err[:,:6]).mean()),
                    gripper_target_mae_fraction=float(np.abs(err[:,6]).mean()),
                    joint_change_mae_rad=float(np.abs(delta[:,:6]).mean()),
                    gripper_change_mae_fraction=float(np.abs(delta[:,6]).mean()), prediction=pred.tolist())
            record['draws'].append(draw)
        result.append(record)
        with (args.output/'queries.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps({'closed_queries':len(result),'elapsed_s':time.monotonic()-start}),flush=True)
    report={**identity,'status':'complete','elapsed_s':time.monotonic()-start,
            'by_route':{},
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    for route in ('PICK','PLACE','ALL'):
        draws=[d for r in result if route=='ALL' or r['route']==route for d in r['draws']]
        report['by_route'][route]={mode:{metric:float(np.mean([d['variants'][mode][metric] for d in draws]))
            for metric in ('joint_target_mae_rad','gripper_target_mae_fraction','joint_change_mae_rad','gripper_change_mae_fraction')}
            for mode in identity['variants']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
