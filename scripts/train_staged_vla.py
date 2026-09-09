#!/usr/bin/env python3
"""Train frozen-Qwen dual action experts from a verified, family-isolated release.

Preflight is the default. --execute requires a new external run directory.
--resume restores optimizer and all RNGs used here into a new run directory.
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import torch
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.staged_experts import StagedConfig, StagedExperts
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer, validate_condition_cache
from conveyor_bench.conveyorvla.staged_data import jsonl, digest

ROUTES = ('NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE')


def smoke(config):
    torch.manual_seed(17)
    base = M0DiTConfig(vlm_hidden_dim=8, input_embedding_dim=8, hidden_size=12,
        num_attention_heads=2, attention_head_dim=4, num_layers=4, dropout=0.,
        max_seq_len=64, num_target_vision_tokens=2)
    model = StagedExperts(config, base)
    conditions = dict(task_tokens=torch.randn(1, 2, 8), live_tokens=torch.randn(1, 4, 8),
        task_mask=torch.ones(1, 2, dtype=torch.bool), live_mask=torch.ones(1, 4, dtype=torch.bool))
    if config.depth:
        conditions.update(points=torch.randn(1, 32, 3), depth_valid=torch.ones(1, 32, dtype=torch.bool))
    losses = {}
    for domain, head in [('NAVIGATION', model.navigation), ('MANIPULATION', model.manipulation)]:
        state = None if domain == 'NAVIGATION' else torch.randn(1, 13)
        actions = torch.randn(1, head.config.action_horizon, head.config.action_dim)
        kwargs = {'prefix_lengths': [2]} if config.training_rtc and domain == 'MANIPULATION' else {}
        if config.fk_loss_weight and domain == 'MANIPULATION':
            kwargs.update(fk=lambda q: (q[..., :3], torch.eye(3).expand(*q.shape[:-1], 3, 3)),
                anchor_q=torch.zeros(1, 6), denormalize=lambda x: x)
        loss = model.loss(domain, actions, state, conditions, **kwargs)
        loss.backward()
        losses[domain] = float(loss)
        if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite candidate loss/gradients')
        model.zero_grad(set_to_none=True)
    model.eval()
    sampled = model.sample_prefix('MANIPULATION', torch.zeros(1, 13), conditions)
    return {'candidate': asdict(config), 'synthetic': True, 'random_initialization': True,
        'losses': losses, 'sample_shape': list(sampled.shape), 'finite_sample': bool(torch.isfinite(sampled).all()),
        'parameters': sum(p.numel() for p in model.parameters()), 'physical_capability_evidence': False}


def validate_release_rows(manifest, rows, normalizer):
    if manifest.get('schema') != 'staged-release-v2' or manifest.get('synthetic'):
        raise ValueError('training requires nonsynthetic verified staged release')
    if manifest.get('not_formal_training_release') or manifest.get('purpose') == 'development_smoke_only':
        raise ValueError('development smoke release cannot be used for qualified training')
    family_splits = {}
    for row in rows:
        split, family = row['split'], row['task_family_id']
        if split not in {'train', 'validation', 'test'}:
            raise ValueError('invalid split')
        if family in family_splits and family_splits[family] != split:
            raise ValueError('task family crosses train/validation/test')
        family_splits[family] = split
        if not row.get('training_eligible') or row.get('synthetic') or row['route'] not in ROUTES:
            raise ValueError('invalid action row in release')
        if row['time_profile'] != manifest['time_profile']:
            raise ValueError('action row time profile mismatch')
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    for name, pool in [('train', train), ('validation', validation)]:
        if {r['route'] for r in pool} != set(ROUTES):
            raise ValueError(name + ' must contain all four action routes')
    if normalizer.payload['normalizer_id'] != manifest['normalizer_id']:
        raise ValueError('release/normalizer binding mismatch')
    if (set(normalizer.payload['families']) != {r['task_family_id'] for r in train}
            or normalizer.payload['fit_row_count'] != len(train)
            or normalizer.payload['time_profiles'] != [manifest['time_profile']]):
        raise ValueError('normalizer must be fitted on exactly the full eligible train pool')
    return train, validation


def select_validation(rows, limit):
    """Deterministic round-robin across route/family strata, never train rows."""
    strata = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r['episode_uuid'], r['query_time_s'], r['observation_id'])):
        if row['split'] != 'validation':
            raise ValueError('validation selector received non-validation row')
        strata[(row['route'], row['task_family_id'])].append(row)
    if limit < len(strata):
        raise ValueError('validation row budget must cover every route/family stratum')
    quotas = {key: 0 for key in strata}
    for _ in range(limit):
        available = [key for key in sorted(strata) if quotas[key] < len(strata[key])]
        if not available:
            break
        key = min(available, key=lambda key: quotas[key])
        quotas[key] += 1
    selected = []
    for key in sorted(strata):
        pool, count = strata[key], quotas[key]
        indices = [round(i * (len(pool) - 1) / max(count - 1, 1)) for i in range(count)]
        selected.extend(pool[i] for i in indices)
    return selected


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng(generator):
    return {'torch_cpu': torch.get_rng_state(), 'sample_generator': generator.get_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(saved, generator):
    torch.set_rng_state(saved['torch_cpu'])
    generator.set_state(saved['sample_generator'])
    if saved['torch_cuda']:
        if len(saved['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('resume CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(saved['torch_cuda'])


def check_resume(saved, binding):
    def canonical(value):
        if isinstance(value, dict) and isinstance(value.get('candidate'), dict):
            return {**value, 'candidate': asdict(StagedConfig(**value['candidate']))}
        return value
    if saved.get('schema') != 'staged-training-state-v1' or canonical(saved.get('training_binding')) != canonical(binding):
        raise ValueError('resume release/cache/encoder/normalizer/config identity mismatch')
    for key in ('optimizer', 'rng', 'model', 'training_steps', 'elapsed_wall_s', 'best_validation_loss'):
        if key not in saved:
            raise ValueError('incomplete resume checkpoint: ' + key)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--release', type=Path)
    p.add_argument('--condition-cache', type=Path)
    p.add_argument('--encoder-model-id')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--initialization', type=Path, help='exact staged weights; starts a new optimizer')
    group.add_argument('--legacy-checkpoint', type=Path, help='explicit strict old dual-DiT migration')
    group.add_argument('--resume', type=Path, help='last.pt; preserves optimizer/RNG/budget in a NEW output')
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--device', default='cpu')
    a = p.parse_args(argv)
    payload = json.loads(a.config.read_text(encoding='utf-8'))
    config = StagedConfig(**payload['candidate'])
    if a.smoke:
        print(json.dumps(smoke(config), allow_nan=False))
        return 0
    if not a.release or not a.condition_cache or not a.encoder_model_id:
        p.error('real preflight requires release, condition-cache and encoder-model-id')
    if not (a.initialization or a.legacy_checkpoint or a.resume):
        p.error('choose explicit initialization, legacy-checkpoint or resume')
    training = dict(payload['training'])
    steps, count = training['steps'], training['effective_batch']
    interval = training.get('checkpoint_every', 100)
    eval_interval = training.get('validation_every', 100)
    validation_limit = training.get('validation_rows', 64)
    wall_limit = training.get('max_wall_seconds', 7200)
    if (steps <= 0 or count <= 0 or count % 2 or interval <= 0 or eval_interval <= 0
            or validation_limit <= 0 or wall_limit <= 0 or not math.isfinite(wall_limit)):
        raise ValueError('invalid finite training budget')
    training.update(checkpoint_every=interval, validation_every=eval_interval,
        validation_rows=validation_limit, max_wall_seconds=wall_limit)
    manifest = json.loads((a.release / 'manifest.json').read_text())
    if manifest['time_profile'] != config.time_profile:
        raise ValueError('dataset/action time profile mismatch')
    for name, expected in manifest['files'].items():
        path = (a.release / name).resolve()
        if not path.is_relative_to(a.release.resolve()) or digest(path) != expected:
            raise ValueError('release path/checksum mismatch: ' + name)
    normalizer = StagedNormalizer(json.loads((a.release / 'normalization.json').read_text()))
    rows, val_rows = validate_release_rows(manifest, jsonl(a.release / 'actions.jsonl'), normalizer)
    selected_validation = select_validation(val_rows, validation_limit)
    if config.fk_loss_weight:
        raise ValueError('calibrated differentiable FK must be provided before training')
    cache_manifest = json.loads((a.condition_cache / 'manifest.json').read_text())
    release_hash = digest(a.release / 'manifest.json')
    if cache_manifest['release_sha256'] != release_hash or cache_manifest['encoder_model_id'] != a.encoder_model_id:
        raise ValueError('condition cache release/encoder mismatch')
    for row in rows + val_rows:
        entry = cache_manifest['entries'][row['episode_uuid'] + ':' + row['observation_id']]
        path = a.condition_cache / entry['file']
        if not path.resolve().is_relative_to(a.condition_cache.resolve()) or digest(path) != entry['sha256']:
            raise ValueError('condition cache path/hash mismatch')
        cache = torch.load(path, map_location='cpu', weights_only=True)
        validate_condition_cache(cache, row, a.encoder_model_id)
        if config.depth and ('points' not in cache or 'depth_valid' not in cache):
            raise ValueError('RGB-D cache lacks calibrated geometry')
        if not config.depth and any(k in cache for k in ('points', 'depth_valid')):
            raise ValueError('RGB-only training requires a cache without depth tensors')
    binding = {'candidate': asdict(config), 'training': training, 'release_sha256': release_hash,
        'cache_manifest_sha256': digest(a.condition_cache / 'manifest.json'),
        'encoder_model_id': a.encoder_model_id, 'normalizer_id': manifest['normalizer_id']}
    torch.manual_seed(training['seed'])
    base = M0DiTConfig(vlm_hidden_dim=2560, hidden_size=1024, action_horizon=10)
    model = StagedExperts(config, base)
    saved = None
    if a.resume or a.initialization:
        checkpoint = a.resume or a.initialization
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if StagedConfig(**saved['candidate']) != config:
            raise ValueError('staged checkpoint candidate mismatch')
        if saved['encoder_model_id'] != a.encoder_model_id or saved['normalizer'] != normalizer.payload:
            raise ValueError('staged initialization encoder/normalizer mismatch')
        if a.resume:
            check_resume(saved, binding)
        model.load_state_dict(saved['model'], strict=True)
        initialization = {'kind': 'resume' if a.resume else 'strict_staged', 'sha256': digest(checkpoint),
            'path': str(checkpoint.resolve())}
    else:
        from safetensors import safe_open
        from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint
        old = validate_formal_checkpoint(a.legacy_checkpoint, ROOT / 'configs/manipulation_navi_v1.json')
        if old['weights_sha256'] != a.encoder_model_id:
            raise ValueError('legacy initialization and frozen Qwen checkpoint differ')
        with safe_open(a.legacy_checkpoint / 'model.safetensors', framework='pt', device='cpu') as f:
            for prefix, head in [('navigation_expert.', model.navigation), ('manipulation_expert.', model.manipulation)]:
                head.load_state_dict({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}, strict=True)
        initialization = {'kind': 'strict_old_trunks_new_semantics', 'weights_sha256': old['weights_sha256'],
            'new_random_parameters': [n for n, _ in model.named_parameters() if n.startswith('geometry.')],
            'old_normalizer_replaced_only_for_new_training': True, 'time_profile_and_token_usage_require_training': True}
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError('nonfinite initialization parameter')
    report = {'schema': 'staged-training-preflight-v3', **binding, 'train_rows': len(rows),
        'train_families': sorted({r['task_family_id'] for r in rows}), 'validation_rows': len(val_rows),
        'validation_families': sorted({r['task_family_id'] for r in val_rows}),
        'selected_validation_rows': len(selected_validation), 'initialization': initialization,
        'initialization_strict_loaded': True, 'training_started': False}
    print(json.dumps(report, allow_nan=False), flush=True)
    if not a.execute:
        return 0
    if not a.output or a.output.exists() or a.output.resolve().is_relative_to(ROOT):
        raise ValueError('training needs a NEW output directory outside the Git worktree')
    model.to(a.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'])
    generator = torch.Generator().manual_seed(training['seed'])
    completed, elapsed_before, best_loss = 0, 0., None
    if a.resume:
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], generator)
        completed, elapsed_before, best_loss = saved['training_steps'], saved['elapsed_wall_s'], saved['best_validation_loss']
        if completed >= steps or elapsed_before >= wall_limit:
            raise ValueError('resume budget is already exhausted')
    a.output.mkdir(parents=True)
    (a.output / 'resolved.json').write_text(json.dumps(report, indent=2) + '\n')
    (a.output / 'validation_selection.json').write_text(json.dumps([
        {k: r[k] for k in ('episode_uuid', 'observation_id', 'task_family_id', 'route')}
        for r in selected_validation], indent=2) + '\n')
    start = time.monotonic()
    def elapsed():
        return elapsed_before + time.monotonic() - start
    def event(record):
        record.update(elapsed_wall_s=elapsed(), timestamp_unix_s=time.time())
        text = json.dumps(record, allow_nan=False)
        print(text, flush=True)
        with (a.output / 'events.jsonl').open('a') as stream:
            stream.write(text + '\n')
    def weights():
        return {'candidate': asdict(config), 'model': model.state_dict(), 'normalizer': normalizer.payload,
            'encoder_model_id': a.encoder_model_id, 'training_steps': completed, 'training_binding': binding}
    def checkpoint(reason):
        atomic_save({**weights(), 'schema': 'staged-training-state-v1', 'optimizer': optimizer.state_dict(),
            'rng': capture_rng(generator), 'elapsed_wall_s': elapsed(), 'best_validation_loss': best_loss}, a.output / 'last.pt')
        event({'event': 'checkpoint', 'step': completed, 'reason': reason})
    def row_loss(row, validation=False):
        entry = cache_manifest['entries'][row['episode_uuid'] + ':' + row['observation_id']]
        cache = torch.load(a.condition_cache / entry['file'], map_location='cpu', weights_only=True)
        conditions = {k: v.to(a.device) for k, v in validate_condition_cache(cache, row, a.encoder_model_id).items()}
        route = row['route']
        mani = route in {'PICK', 'PLACE'}
        actions = torch.tensor(normalizer.normalize_action(route, row['actions']), dtype=torch.float32, device=a.device)[None]
        state = torch.tensor(normalizer.normalize_mani_state(row['mani_state']), dtype=torch.float32, device=a.device)[None] if mani else None
        kwargs = {}
        if config.training_rtc and mani:
            kwargs['prefix_lengths'] = [2 if validation else int(torch.randint(0, actions.shape[1], (), generator=generator))]
        return model.loss('MANIPULATION' if mani else 'NAVIGATION', actions, state, conditions, **kwargs)
    def evaluate():
        nonlocal best_loss
        by_route = defaultdict(list)
        model.eval()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(training['seed'] + 1)
            for row in selected_validation:
                if elapsed() >= wall_limit:
                    model.train()
                    event({'event': 'validation_incomplete', 'step': completed, 'reason': 'wall_budget'})
                    return
                loss = float(row_loss(row, validation=True))
                if not math.isfinite(loss):
                    raise ValueError('nonfinite validation loss')
                by_route[row['route']].append(loss)
        model.train()
        metrics = {key: sum(value) / len(value) for key, value in by_route.items()}
        score = sum(metrics.values()) / len(metrics)
        improved = best_loss is None or score < best_loss
        if improved:
            best_loss = score
            atomic_save({**weights(), 'validation_loss': score}, a.output / 'best.pt')
        event({'event': 'validation', 'step': completed, 'loss': score, 'by_route': metrics,
            'rows': len(selected_validation), 'new_best': improved, 'metric': 'fixed_rng_flow_loss_not_task_success'})
    pools = [[i for i, r in enumerate(rows) if r['route'] in routes]
        for routes in ({'NAV_TO_SOURCE', 'NAV_TO_TARGET'}, {'PICK', 'PLACE'})]
    evaluate()
    checkpoint('initial_or_resume')
    stop_reason = 'step_budget'
    try:
        for step in range(completed, steps):
            if elapsed() >= wall_limit:
                stop_reason = 'wall_budget'
                break
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            selected = [pool[i] for pool in pools for i in torch.randint(len(pool), (count // 2,), generator=generator).tolist()]
            incomplete = False
            for idx in selected:
                if elapsed() >= wall_limit:
                    incomplete = True
                    break
                loss = row_loss(rows[idx]) / count
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                total += float(loss)
            if incomplete:
                optimizer.zero_grad(set_to_none=True)
                stop_reason = 'wall_budget_partial_batch_discarded'
                break
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            event({'event': 'train', 'step': completed, 'loss': total, 'gradient_norm': float(norm),
                'step_wall_s': time.monotonic() - step_start, 'effective_batch': count})
            if completed % eval_interval == 0 or completed == steps:
                evaluate()
            if completed % interval == 0:
                checkpoint('periodic')
    except Exception as error:
        event({'event': 'failed', 'step': completed, 'error': str(error)})
        # Preserve last validated state; do not overwrite it with potentially nonfinite tensors.
        raise
    checkpoint(stop_reason)
    atomic_save(weights(), a.output / 'model.pt')
    event({'event': 'finished', 'step': completed, 'reason': stop_reason,
        'best_validation_loss': best_loss, 'physical_capability_evidence': False})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
