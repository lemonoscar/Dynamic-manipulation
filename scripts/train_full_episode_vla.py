#!/usr/bin/env python3
"""Bounded RGB full-model adaptation over complete ordered episode query pools.

A separate 2-step --preflight-steps run checks true full-model gradients and memory.
It never supplies initialization for the production run. No automatic resume.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
from torch import nn
from PIL import Image
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint, load_formal_policy
from conveyor_bench.conveyorvla.staged_experts import StagedConfig, StagedExperts
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer
from conveyor_bench.conveyorvla.staged_data import digest, jsonl
from conveyor_bench.conveyorvla.staged_rgb_backend import encode_rgb_training_conditions
from scripts.train_staged_vla import atomic_save
from conveyor_bench.conveyorvla import full_episode_data
from conveyor_bench.conveyorvla.full_episode_context import public_context_text

ROUTES = ('NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))



def images(row):
    root = Path(row['episode_root']).resolve()
    names = row.get('images', row.get('input', {}).get('images'))
    if not isinstance(names, list) or len(names) != 4:
        raise ValueError('four causal RGB references required')
    result = []
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('RGB reference escapes episode')
        with Image.open(path) as frame:
            result.append(frame.convert('RGB'))
    return result


def validate_rows(manifest, actions, transitions):
    if manifest.get('schema') != 'full-episode-rgb-derived-v1' or manifest.get('input_modalities') != 'rgb':
        raise ValueError('a versioned RGB full-episode release is required')
    if manifest.get('time_profile') != 'causal_command_5hz' or manifest.get('synthetic'):
        raise ValueError('wrong full-episode source/profile')
    seen = {}
    keys = set()
    for kind, pool in [('action', actions), ('transition', transitions)]:
        for row in pool:
            family, split = row['task_family_id'], row['split']
            if split not in {'train', 'validation', 'test'} or (family in seen and seen[family] != split):
                raise ValueError('family crosses split or invalid split')
            seen[family] = split
            key = (kind, row['episode_uuid'], row['observation_id'],
                row.get('parent_memory_id') if kind == 'transition' else None)
            if key in keys:
                raise ValueError('duplicate query in full release')
            keys.add(key)
            if row.get('synthetic') or row.get('depth') is not None:
                raise ValueError('synthetic/depth input forbidden')
            value = row if kind == 'action' else row['input']
            expected_times = [row['query_time_s'] - .2, row['query_time_s']] * 2
            observed_times = value['image_times_s']
            if (len(value['images']) != 4 or len(observed_times) != 4
                    or not np.allclose(observed_times, expected_times, rtol=0, atol=1e-7)):
                raise ValueError('full release must bind exact causal four-RGB timestamps')
            public_context_text(value['task_context'])
            if kind == 'action':
                if value['task_context']['active_task'] != row['route']:
                    raise ValueError('action task context/route mismatch')
                mask = row['action_valid_mask']
                if len(mask) != 10 or not any(mask) or any(type(v) is not bool for v in mask):
                    raise ValueError('invalid action validity mask')
                if mask != sorted(mask, reverse=True):
                    raise ValueError('only native-supported prefix masks are legal')
                values = np.asarray(row['actions'], dtype=float)
                if values.shape != (10, 7 if row['route'] in {'PICK', 'PLACE'} else 3) or not np.isfinite(values).all():
                    raise ValueError('invalid action tensor')
                if not row.get('training_eligible') or row['time_profile'] != manifest['time_profile']:
                    raise ValueError('ineligible action row')
            else:
                if not row.get('transition_training_eligible'):
                    raise ValueError('transition lacks offline training eligibility')
                operation = row['label']['operation']
                if operation not in {'CONTINUE', 'ADVANCE'}:
                    raise ValueError('unsupported or fabricated transition supervision')
                active = value['task_context']['active_task']
                index = ROUTES.index(active)
                if operation == 'ADVANCE' and index == len(ROUTES) - 1:
                    raise ValueError('no invented terminal transition')
                expected = active if operation == 'CONTINUE' else ROUTES[index + 1]
                if row['label']['next_task'] != expected:
                    raise ValueError('transition target disagrees with public suffix')
    for split in ('train', 'validation'):
        if {r['route'] for r in actions if r['split'] == split} != set(ROUTES):
            raise ValueError('all four action routes required in ' + split)
        if {r['label']['operation'] for r in transitions if r['split'] == split} != {'CONTINUE', 'ADVANCE'}:
            raise ValueError('both transition operations required in ' + split)


def validation_subset(rows, limit, transition=False):
    pools = defaultdict(list)
    for row in rows:
        if row['split'] == 'validation':
            route = row['label']['operation'] if transition else row['route']
            pools[route].append(row)
    if limit < len(pools):
        raise ValueError('validation budget cannot cover categories')
    selected = []
    for key in pools:
        pools[key].sort(key=lambda r: (r['task_family_id'], r['episode_uuid'], r['query_time_s']))
    quota = {key: 0 for key in pools}
    for _ in range(min(limit, sum(map(len, pools.values())))):
        available = [key for key in sorted(pools) if quota[key] < len(pools[key])]
        quota[min(available, key=lambda key: quota[key])] += 1
    for key in sorted(pools):
        pool, count = pools[key], quota[key]
        selected.extend(pool[round(i * (len(pool) - 1) / max(count - 1, 1))] for i in range(count))
    return selected


def episode_order(rows, seed):
    families = defaultdict(lambda: defaultdict(list))
    for index, item in enumerate(rows):
        row = item[1]
        families[row['task_family_id']][row['episode_uuid']].append(index)
    rng = random.Random(seed)
    family_order = sorted(families)
    rng.shuffle(family_order)
    order = []
    for family in family_order:
        episodes = sorted(families[family])
        rng.shuffle(episodes)
        for episode in episodes:
            order.extend(sorted(families[family][episode],
                key=lambda i: (rows[i][1]['query_time_s'], rows[i][0], rows[i][1]['observation_id'])))
    if sorted(order) != list(range(len(rows))):
        raise ValueError('episode sampler lost/duplicated rows')
    return order


class FullModel(nn.Module):
    def __init__(self, qwen, experts):
        super().__init__()
        self.qwen = qwen
        self.experts = experts


def embedding_parameters(qwen):
    core = getattr(qwen, 'model', qwen)
    input_module = core.get_input_embeddings() if hasattr(core, 'get_input_embeddings') else core.embed_tokens
    output_module = core.get_output_embeddings() if hasattr(core, 'get_output_embeddings') else core.lm_head
    if input_module is None or output_module is None:
        raise ValueError('Qwen must expose input/output embedding modules')
    return input_module.weight, output_module.weight


def optimizer_groups(model, config):
    groups = defaultdict(list)
    names = defaultdict(list)
    input_weight, output_weight = embedding_parameters(model.qwen)
    tied = input_weight is output_weight
    for name, parameter in model.named_parameters():
        if name.startswith('experts.navigation.'):
            key = 'navigation'
        elif name.startswith('experts.manipulation.'):
            key = 'manipulation'
        elif tied and parameter is input_weight:
            key = 'qwen_embeddings_lm_head'
        elif parameter is input_weight:
            key = 'qwen_embeddings'
        elif parameter is output_weight or '.lm_head.' in name:
            key = 'qwen_lm_head'
        elif any(v in name.split('.') for v in ('visual', 'vision_model', 'vision_tower')):
            key = 'qwen_vision'
        else:
            key = 'qwen_core'
        groups[key].append(parameter)
        names[key].append(name)
    expected = {'navigation', 'manipulation', 'qwen_vision', 'qwen_core'}
    expected |= {'qwen_embeddings_lm_head'} if tied else {'qwen_embeddings', 'qwen_lm_head'}
    if set(groups) != expected:
        raise ValueError('full model parameter groups missing: ' + str(expected - set(groups)))
    flat = [p for values in groups.values() for p in values]
    if len(flat) != len({id(p) for p in flat}) or {id(p) for p in flat} != {id(p) for p in model.parameters()}:
        raise ValueError('optimizer does not uniquely cover every deployed parameter')
    if not all(p.requires_grad and p.dtype == torch.float32 for p in flat):
        raise ValueError('all deployed master parameters must be trainable FP32')
    rate = {key: config['action_learning_rate'] if key in {'navigation', 'manipulation'}
        else config['vision_learning_rate'] if key == 'qwen_vision' else config['qwen_learning_rate'] for key in groups}
    result = [{'name': key, 'params': groups[key], 'lr': rate[key]} for key in sorted(groups)]
    return result, names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--legacy-checkpoint', type=Path, required=True)
    parser.add_argument('--staged-checkpoint', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, default=ROOT / 'artifacts/models/base')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--preflight-steps', type=int, default=0)
    parser.add_argument('--qwen-device', default='cuda:0')
    parser.add_argument('--action-device', default='cuda:1')
    args = parser.parse_args(argv)
    config = read(args.config)
    budget = config['training']
    required = ('effective_batch', 'epochs', 'max_wall_seconds', 'seed', 'action_learning_rate',
        'qwen_learning_rate', 'vision_learning_rate', 'transition_loss_weight')
    if any(k not in budget for k in required):
        raise ValueError('explicit full-model training budget and rates required')
    count, epochs, wall_limit = budget['effective_batch'], budget['epochs'], budget['max_wall_seconds']
    if count != 8 or not 0 < epochs <= 3 or not 0 < wall_limit <= 8 * 3600 or args.preflight_steps not in (0, 2):
        raise ValueError('unsupported or unbounded approved full-model recipe')
    manifest = read(args.release / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = (args.release / name).resolve()
        if not path.is_relative_to(args.release.resolve()) or digest(path) != expected:
            raise ValueError('release file path/hash mismatch: ' + name)
    actions, transitions = jsonl(args.release / 'actions.jsonl'), jsonl(args.release / 'transitions.jsonl')
    validate_rows(manifest, actions, transitions)
    old = validate_formal_checkpoint(args.legacy_checkpoint, ROOT / 'configs/manipulation_navi_v1.json')
    saved = torch.load(args.staged_checkpoint, map_location='cpu', weights_only=True)
    candidate = StagedConfig(**saved['candidate'])
    if candidate.depth or candidate.training_rtc or candidate.time_profile != manifest['time_profile']:
        raise ValueError('staged initializer must be compatible RGB causal inference-RTC weights')
    normalizer = StagedNormalizer(saved['normalizer'])
    if saved['encoder_model_id'] != old['weights_sha256']:
        raise ValueError('staged weights and old frozen Qwen identity differ')
    normalizer_path = args.release / 'normalization.json'
    if not normalizer_path.is_file() or read(normalizer_path) != normalizer.payload:
        raise ValueError('full release must preserve exact completed staged normalizer')
    if manifest.get('normalizer_id') != normalizer.payload['normalizer_id']:
        raise ValueError('full release normalizer identity mismatch')
    if not set(normalizer.payload['families']).issubset({r['task_family_id'] for r in actions if r['split'] == 'train'}):
        raise ValueError('staged normalizer fit families cannot leave full train pool')
    rows = [('action', r) for r in actions if r['split'] == 'train'] + [('transition', r) for r in transitions if r['split'] == 'train']
    validation = [('action', r) for r in validation_subset(actions, budget.get('validation_action_rows', 16))]
    validation += [('transition', r) for r in validation_subset(transitions, budget.get('validation_transition_rows', 16), True)]
    binding = {'release_sha256': digest(args.release / 'manifest.json'), 'data_helper_sha256': digest(Path(full_episode_data.__file__)),
        'config_sha256': digest(args.config), 'normalizer_id': normalizer.payload['normalizer_id'],
        'base_encoder_model_id': old['weights_sha256'], 'staged_initialization_sha256': digest(args.staged_checkpoint),
        'candidate': asdict(candidate), 'code_commit': config.get('code_commit')}
    report = {'schema': 'full-episode-training-preflight-v1', **binding, 'training': budget,
        'train_action_rows': sum(k == 'action' for k, _ in rows),
        'train_transition_rows': sum(k == 'transition' for k, _ in rows),
        'train_families': sorted({r['task_family_id'] for _, r in rows}),
        'validation_families': sorted({r['task_family_id'] for _, r in validation}),
        'validation_rows': len(validation), 'depth': False, 'tensor_cache_used': False,
        'teacher_forced_offline_transitions': True, 'deployment_capability_evidence': False,
        'preflight_steps': args.preflight_steps, 'training_started': False}
    print(json.dumps(report), flush=True)
    if not args.execute:
        return 0
    if not args.output or args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        raise ValueError('a new external output directory is required')
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 2
            or args.qwen_device != 'cuda:0' or args.action_device != 'cuda:1'):
        raise ValueError('requires exactly two visible GPUs: Qwen cuda:0, experts cuda:1')
    torch.manual_seed(budget['seed'])
    random.seed(budget['seed'])
    start = time.monotonic()
    policy = load_formal_policy(old, args.model_root, device='cpu')
    qwen = policy.qwen
    input_weight, output_weight = embedding_parameters(qwen)
    sharing = {'input_output_embeddings_tied': input_weight is output_weight,
        'checked_on_cpu': input_weight.device.type == output_weight.device.type == 'cpu',
        'language_objective': 'explicit_transition_answer_cross_entropy'}
    if not sharing['checked_on_cpu']:
        raise ValueError('Qwen tied-weight identity must be checked before GPU placement')
    print(json.dumps({'event': 'qwen_parameter_sharing', **sharing}), flush=True)
    qwen.enable_full_finetuning()
    qwen.to(device=args.qwen_device, dtype=torch.float32)
    del policy
    experts = StagedExperts(candidate, M0DiTConfig(vlm_hidden_dim=2560, hidden_size=1024, action_horizon=10))
    experts.load_state_dict(saved['model'], strict=True)
    experts.to(device=args.action_device, dtype=torch.float32)
    del saved
    model = FullModel(qwen, experts).train()
    groups, parameter_names = optimizer_groups(model, budget)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, weight_decay=1e-8, foreach=False)
    args.output.mkdir(parents=True)
    report.update(initialization_strict_loaded=True, qwen_parameter_sharing=sharing, optimizer_parameters=sum(p.numel() for p in model.parameters()),
        optimizer_groups={g['name']: {'parameters': sum(p.numel() for p in g['params']), 'lr': g['lr'],
            'names': parameter_names[g['name']]} for g in groups})
    (args.output / 'resolved.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'validation_selection.json').write_text(json.dumps([
        {'kind': k, 'episode_uuid': r['episode_uuid'], 'observation_id': r['observation_id'], 'task_family_id': r['task_family_id']}
        for k, r in validation], indent=2) + '\n')
    steps, epoch, cursor, best_score = 0, 0, 0, None
    order = []
    gradient_groups_seen = set()
    def elapsed():
        return time.monotonic() - start
    def event(record):
        record.update(elapsed_wall_s=elapsed(), timestamp_unix_s=time.time())
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (args.output / 'events.jsonl').open('a') as stream:
            stream.write(line + '\n')
    def weights():
        return {'schema': 'full-episode-vla-weights-v1', 'qwen_model': qwen.state_dict(),
            'experts_model': experts.state_dict(), 'candidate': asdict(candidate), 'normalizer': normalizer.payload,
            'base_encoder_model_id': binding['base_encoder_model_id'],
            'staged_initialization_sha256': binding['staged_initialization_sha256'],
            'training_binding': binding, 'steps': steps, 'epoch': epoch, 'cursor': cursor}
    def checkpoint(reason):
        if args.preflight_steps:
            return
        atomic_save({**weights(), 'schema': 'full-episode-vla-state-v1', 'optimizer': optimizer.state_dict(),
            'rng': {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'python': random.getstate()},
            'order': order, 'elapsed_wall_s': elapsed(), 'best_score': best_score}, args.output / 'last.pt')
        event({'event': 'checkpoint', 'step': steps, 'reason': reason})
    def loss(kind, row):
        rgb = images(row)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if kind == 'action':
                context = public_context_text(row['task_context'])
                conditions = encode_rgb_training_conditions(model, row['original_instruction'], row['route'], rgb, context=context)
                conditions = {k: v.to(args.action_device) for k, v in conditions.items()}
                action = torch.tensor(normalizer.normalize_action(row['route'], row['actions']),
                    device=args.action_device, dtype=torch.float32)[None]
                mani = row['route'] in {'PICK', 'PLACE'}
                state = torch.tensor(normalizer.normalize_mani_state(row['mani_state']),
                    device=args.action_device, dtype=torch.float32)[None] if mani else None
                mask = torch.tensor(row['action_valid_mask'], device=args.action_device, dtype=torch.bool)[None]
                return experts.loss('MANIPULATION' if mani else 'NAVIGATION', action, state,
                    conditions, action_valid_mask=mask)
            prompt, response = full_episode_data.transition_prompt(row), full_episode_data.transition_response(row)
            inputs = dict(qwen.build_joint_trajectory_inputs(
                [{'video': (rgb[:2], rgb[2:]), 'lang': prompt}], solutions=[response], supervise_solutions=True))
            labels = inputs.pop('labels')
            output = qwen(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)
            valid = labels[:, 1:] != -100
            if not valid.any():
                raise ValueError('transition answer has no supervised tokens')
            return nn.functional.cross_entropy(output.logits[:, :-1][valid].float(), labels[:, 1:][valid])
    def evaluate():
        nonlocal best_score
        metrics = defaultdict(list)
        model.eval()
        # The differentiable helper intentionally rejects no_grad. Validation
        # enables graph construction but never calls backward; each row is freed.
        with torch.random.fork_rng(devices=[0, 1]):
            torch.manual_seed(budget['seed'] + 1)
            for kind, row in validation:
                if elapsed() >= wall_limit:
                    event({'event': 'validation_incomplete', 'step': steps})
                    model.train()
                    return
                value = float(loss(kind, row).detach())
                if not math.isfinite(value):
                    raise ValueError('nonfinite validation loss')
                category = row['route'] if kind == 'action' else row['label']['operation']
                metrics[kind + ':' + category].append(value)
        model.train()
        means = {k: sum(v) / len(v) for k, v in metrics.items()}
        score = sum(means['action:' + route] for route in ROUTES) / 4
        score += budget['transition_loss_weight'] * sum(v for k, v in means.items() if k.startswith('transition:')) / 2
        improved = best_score is None or score < best_score
        if improved:
            best_score = score
            if not args.preflight_steps:
                atomic_save({**weights(), 'validation_score': score}, args.output / 'best.pt')
        event({'event': 'validation', 'step': steps, 'score': score, 'by_category': means,
            'new_best': improved, 'physical_capability_evidence': False})
    def grad_report(require_all):
        result = {}
        for group in groups:
            parameters = group['params']
            active = [p for p in parameters if p.grad is not None]
            finite = all(torch.isfinite(p.grad).all() for p in active)
            norm_sq = sum(float(torch.linalg.vector_norm(p.grad.detach())) ** 2 for p in active)
            result[group['name']] = {'parameter_tensors': len(parameters), 'gradient_tensors': len(active),
                'finite': bool(finite), 'norm': math.sqrt(norm_sq), 'sample_update': None}
            if not finite or (require_all and (len(active) != len(parameters) or norm_sq <= 0)):
                missing = [name for name, p in zip(parameter_names[group['name']], parameters) if p.grad is None]
                raise ValueError('missing/nonfinite full-model gradient group: ' + group['name'] + '; missing=' + str(missing[:10]))
            if norm_sq > 0:
                gradient_groups_seen.add(group['name'])
        return result
    if args.preflight_steps:
        # Explicit diagnostic cover, separate from complete episode production order.
        preflight = [next(i for i, (k, r) in enumerate(rows) if k == 'action' and r['route'] == route) for route in ROUTES]
        transition_indices = [i for i, (k, _) in enumerate(rows) if k == 'transition']
        preflight.extend(transition_indices[i % len(transition_indices)] for i in range(count - len(preflight)))
        orders = [preflight * args.preflight_steps]
    else:
        orders = [episode_order(rows, budget['seed'] + i) for i in range(epochs)]
    stop_reason = 'epochs_complete'
    try:
        if not args.preflight_steps:
            evaluate()  # Initial validation is a baseline; it is not a completed training step.
        for epoch, order in enumerate(orders):
            cursor = 0
            while cursor < len(order):
                if elapsed() >= wall_limit:
                    stop_reason = 'wall_budget'
                    break
                step_start = time.monotonic()
                selected = order[cursor:cursor + count]
                optimizer.zero_grad(set_to_none=True)
                total = 0.
                kinds = Counter()
                incomplete = False
                for index in selected:
                    if elapsed() >= wall_limit:
                        incomplete = True
                        break
                    kind, row = rows[index]
                    value = loss(kind, row)
                    scale = budget['transition_loss_weight'] if kind == 'transition' else 1.
                    weighted = value * scale / len(selected)
                    if not torch.isfinite(weighted):
                        raise ValueError('nonfinite training loss')
                    weighted.backward()
                    total += float(weighted.detach())
                    kinds[kind] += 1
                if incomplete:
                    optimizer.zero_grad(set_to_none=True)
                    stop_reason = 'wall_budget_partial_batch_discarded'
                    break
                gradients = grad_report(bool(args.preflight_steps))
                # clip_grad_norm_ cannot combine scalar norms on different devices.
                norm = math.sqrt(sum(g['norm'] ** 2 for g in gradients.values()))
                if not math.isfinite(norm):
                    raise ValueError('nonfinite global gradient norm')
                multiplier = min(1., 1. / (norm + 1e-6))
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(multiplier)
                probes = {}
                if args.preflight_steps:
                    for group in groups:
                        parameter = next(p for p in group['params'] if p.grad is not None and p.grad.abs().max() > 0)
                        index = int(parameter.grad.detach().abs().argmax())
                        probes[group['name']] = (parameter, index, float(parameter.detach().reshape(-1)[index]))
                optimizer.step()
                for name, (parameter, index, previous) in probes.items():
                    changed = float(parameter.detach().reshape(-1)[index]) != previous
                    gradients[name]['sample_update'] = changed
                    if not changed:
                        raise ValueError('full-model optimizer group did not change: ' + name)
                cursor += len(selected)
                steps += 1
                torch.cuda.synchronize(0); torch.cuda.synchronize(1)
                event({'event': 'train', 'step': steps, 'epoch': epoch + 1, 'cursor': cursor,
                    'epoch_rows': len(order), 'loss': total, 'rows': dict(kinds), 'gradient_norm': norm,
                    'gradient_groups': gradients, 'step_wall_s': time.monotonic() - step_start,
                    'peak_cuda_bytes': {str(i): torch.cuda.max_memory_allocated(i) for i in (0, 1)}})
                if not args.preflight_steps and (steps % budget.get('validation_every', 100) == 0):
                    evaluate()
                if not args.preflight_steps and (steps % budget.get('checkpoint_every', 100) == 0
                        or steps == budget.get('first_checkpoint_step', 20)):
                    checkpoint('periodic')
            if stop_reason.startswith('wall_budget'):
                break
            if not args.preflight_steps:
                event({'event': 'epoch_complete', 'epoch': epoch + 1, 'rows_seen_once': len(order)})
        if not args.preflight_steps:
            evaluate()
            checkpoint(stop_reason)
            atomic_save(weights(), args.output / 'model.pt')
        result = {'schema': 'full-episode-training-result-v1', 'steps': steps, 'reason': stop_reason,
            'preflight_only': bool(args.preflight_steps), 'all_optimizer_groups_received_gradients': gradient_groups_seen == {g['name'] for g in groups},
            'best_validation_score': best_score, 'elapsed_wall_s': elapsed(), 'physical_capability_evidence': False}
        if args.preflight_steps and (steps != args.preflight_steps or not result['all_optimizer_groups_received_gradients']):
            raise ValueError('full-model preflight incomplete')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        event({'event': 'finished', **result})
        return 0
    except Exception as error:
        event({'event': 'failed', 'step': steps, 'error': repr(error)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
