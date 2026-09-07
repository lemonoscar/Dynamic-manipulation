#!/usr/bin/env python3
"""Staged frozen-encoder action training; preflight by default, never auto-train.

--smoke uses labeled synthetic tensors and tiny random networks, writes no weights.
Real training requires a verified release, matching frozen condition cache and an
explicit initialization; no dataset or weights are placed in Git.
"""
import argparse,json,sys,time
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.staged_experts import StagedConfig,StagedExperts
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer,validate_condition_cache
from conveyor_bench.conveyorvla.staged_data import jsonl,digest


def smoke(config):
    torch.manual_seed(17)
    base=M0DiTConfig(vlm_hidden_dim=8,input_embedding_dim=8,hidden_size=12,num_attention_heads=2,
        attention_head_dim=4,num_layers=4,dropout=0.,max_seq_len=64,num_target_vision_tokens=2)
    model=StagedExperts(config,base)
    conditions=dict(task_tokens=torch.randn(1,2,8),live_tokens=torch.randn(1,4,8),
        task_mask=torch.ones(1,2,dtype=torch.bool),live_mask=torch.ones(1,4,dtype=torch.bool))
    if config.depth:
        conditions.update(points=torch.randn(1,32,3),depth_valid=torch.ones(1,32,dtype=torch.bool))
    losses={}
    for domain,head in [('NAVIGATION',model.navigation),('MANIPULATION',model.manipulation)]:
        state=None if domain=='NAVIGATION' else torch.randn(1,13)
        actions=torch.randn(1,head.config.action_horizon,head.config.action_dim)
        kwargs={}
        if config.training_rtc and domain=='MANIPULATION':kwargs['prefix_lengths']=[2]
        if config.fk_loss_weight and domain=='MANIPULATION':
            # Explicit toy kinematics checks gradient plumbing only.
            kwargs.update(fk=lambda q:(q[...,:3],torch.eye(3).expand(*q.shape[:-1],3,3)),
                anchor_q=torch.zeros(1,6),denormalize=lambda x:x)
        loss=model.loss(domain,actions,state,conditions,**kwargs)
        loss.backward();losses[domain]=float(loss)
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite candidate loss/gradients')
        model.zero_grad(set_to_none=True)
    model.eval()
    sampled=model.sample_prefix('MANIPULATION',torch.zeros(1,13),conditions)
    return {'candidate':asdict(config),'synthetic':True,'random_initialization':True,
        'losses':losses,'sample_shape':list(sampled.shape),'finite_sample':bool(torch.isfinite(sampled).all()),
        'parameters':sum(p.numel() for p in model.parameters()),'physical_capability_evidence':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--release',type=Path)
    p.add_argument('--condition-cache',type=Path)
    p.add_argument('--encoder-model-id')
    p.add_argument('--initialization',type=Path,help='staged checkpoint with exact config identity')
    p.add_argument('--legacy-checkpoint',type=Path,help='explicit old formal dual-DiT migration, geometry newly initialized')
    p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true',help='perform reviewed training; otherwise preflight only')
    p.add_argument('--device',default='cpu')
    a=p.parse_args(argv)
    payload=json.loads(a.config.read_text());config=StagedConfig(**payload['candidate'])
    if a.smoke:
        print(json.dumps(smoke(config),allow_nan=False));return 0
    if not a.release or not a.condition_cache or not a.encoder_model_id:
        p.error('real preflight requires release, condition-cache and encoder-model-id')
    manifest=json.loads((a.release/'manifest.json').read_text())
    if manifest.get('schema')!='staged-release-v2' or manifest.get('synthetic'):
        raise ValueError('real training needs nonsynthetic verified staged release')
    if manifest['time_profile']!=config.time_profile:
        raise ValueError('dataset/action time profile mismatch')
    for name,expected in manifest['files'].items():
        if digest(a.release/name)!=expected:raise ValueError('release file checksum mismatch: '+name)
    normalizer=StagedNormalizer(json.loads((a.release/'normalization.json').read_text()))
    if normalizer.payload['normalizer_id']!=manifest['normalizer_id']:
        raise ValueError('release/normalizer binding mismatch')
    rows=[r for r in jsonl(a.release/'actions.jsonl') if r['split']=='train']
    if not rows or any(not r['training_eligible'] or r.get('synthetic') for r in rows):
        raise ValueError('no valid train rows')
    if config.fk_loss_weight:
        raise ValueError('FK is a tested callback interface; calibrated differentiable robot FK must be supplied before training this candidate')
    cache_manifest=json.loads((a.condition_cache/'manifest.json').read_text())
    if cache_manifest['release_sha256']!=digest(a.release/'manifest.json'):
        raise ValueError('condition cache belongs to another release')
    for row in rows:
        key=row['episode_uuid']+':'+row['observation_id']
        entry=cache_manifest['entries'][key]
        path=a.condition_cache/entry['file']
        if not path.resolve().is_relative_to(a.condition_cache.resolve()) or digest(path)!=entry['sha256']:
            raise ValueError('condition cache path/hash mismatch')
        cache=torch.load(path,map_location='cpu',weights_only=True)
        validate_condition_cache(cache,row,a.encoder_model_id)
        if config.depth and ('points' not in cache or 'depth_valid' not in cache):
            raise ValueError('RGB-D training cache lacks calibrated depth/mask')
    if bool(a.initialization)==bool(a.legacy_checkpoint):
        raise ValueError('choose exactly one explicit staged initialization or legacy migration')
    report={'schema':'staged-training-preflight-v2','candidate':asdict(config),
        'train_rows':len(rows),'families':len({r['task_family_id'] for r in rows}),
        'normalizer_id':manifest['normalizer_id'],'encoder_model_id':a.encoder_model_id,
        'training_started':False,'steps':payload['training']['steps'],
        'effective_batch':payload['training']['effective_batch']}
    if not a.execute:
        print(json.dumps(report));return 0
    if not a.output or a.output.exists() or a.output.resolve().is_relative_to(ROOT):
        raise ValueError('training needs a NEW output directory outside the Git worktree')
    torch.manual_seed(payload['training']['seed'])
    # Match the actual old backbone; architecture differences are declared below.
    base=M0DiTConfig(vlm_hidden_dim=2560,hidden_size=1024,action_horizon=10)
    model=StagedExperts(config,base)
    initialization={}
    if a.initialization:
        saved=torch.load(a.initialization,map_location='cpu',weights_only=True)
        if saved['candidate']!=asdict(config):raise ValueError('staged checkpoint config mismatch')
        model.load_state_dict(saved['model'],strict=True)
        initialization={'kind':'strict_staged','sha256':digest(a.initialization)}
    else:
        from safetensors import safe_open
        from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint
        binding=validate_formal_checkpoint(a.legacy_checkpoint,ROOT/'configs/manipulation_navi_v1.json')
        path=a.legacy_checkpoint/'model.safetensors'
        with safe_open(path,framework='pt',device='cpu') as f:
            for prefix,head in [('navigation_expert.',model.navigation),('manipulation_expert.',model.manipulation)]:
                state={k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
                head.load_state_dict(state,strict=True)
        initialization={'kind':'strict_old_trunks_new_semantics','weights_sha256':binding['weights_sha256'],
            'new_random_parameters':[n for n,_ in model.named_parameters() if n.startswith('geometry.')],
            'old_normalizer_replaced_only_for_new_training':True,
            'time_profile_and_token_usage_require_training':True}
    model.to(a.device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=payload['training']['learning_rate'])
    a.output.mkdir(parents=True)
    (a.output/'resolved.json').write_text(json.dumps({**report,'initialization':initialization},indent=2)+'\n')
    generator=torch.Generator().manual_seed(payload['training']['seed'])
    count=payload['training']['effective_batch']
    if count % 2: raise ValueError('effective batch must split equally across domains')
    nav_indices=[i for i,r in enumerate(rows) if r['route'] in {'NAV_TO_SOURCE','NAV_TO_TARGET'}]
    mani_indices=[i for i,r in enumerate(rows) if r['route'] in {'PICK','PLACE'}]
    if not nav_indices or not mani_indices: raise ValueError('training pool requires both domains')
    for step in range(payload['training']['steps']):
        optimizer.zero_grad(set_to_none=True);total=0.
        selected=[]
        for pool in (nav_indices,mani_indices):
            selected.extend(pool[i] for i in torch.randint(len(pool),(count//2,),generator=generator).tolist())
        for idx in selected:
            row=rows[idx];entry=cache_manifest['entries'][row['episode_uuid']+':'+row['observation_id']]
            cache=torch.load(a.condition_cache/entry['file'],map_location='cpu',weights_only=True)
            conditions=validate_condition_cache(cache,row,a.encoder_model_id)
            conditions={k:v.to(a.device) for k,v in conditions.items()}
            if not config.depth:
                conditions.pop('points',None);conditions.pop('depth_valid',None)
            route=row['route'];mani=route in {'PICK','PLACE'}
            actions=torch.tensor(normalizer.normalize_action(route,row['actions']),dtype=torch.float32,device=a.device)[None]
            state=torch.tensor(normalizer.normalize_mani_state(row['mani_state']),dtype=torch.float32,device=a.device)[None] if mani else None
            kwargs={'prefix_lengths':[int(torch.randint(0,actions.shape[1],(),generator=generator))]} if config.training_rtc and mani else {}
            loss=model.loss('MANIPULATION' if mani else 'NAVIGATION',actions,state,conditions,**kwargs)/count
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();total+=float(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'loss':total})+'\n')
    torch.save({'candidate':asdict(config),'model':model.state_dict(),'normalizer':normalizer.payload,
        'encoder_model_id':a.encoder_model_id,'training_steps':payload['training']['steps']},a.output/'model.pt')
    return 0

if __name__=='__main__':raise SystemExit(main())
