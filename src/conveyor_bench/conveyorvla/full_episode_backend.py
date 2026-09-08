"""Strict reload of the jointly adapted RGB backbone and both action experts.

Transition predictions are proposals only. They are not completion evidence and
do not bypass TaskMemory's independent feedback requirements.
"""
import json
from pathlib import Path
import torch
from torch import nn
from .formal_checkpoint import load_formal_policy,validate_formal_checkpoint
from .staged_data import digest
from .staged_experts import StagedConfig,StagedExperts
from .staged_training import StagedNormalizer
from .staged_rgb_backend import StagedRGBBackend
from .dit import M0DiTConfig


def load_full_episode_backend(checkpoint, *, legacy_checkpoint, repo_root, device='cpu'):
    checkpoint,repo_root=Path(checkpoint),Path(repo_root)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved.get('schema') not in {'full-episode-vla-weights-v1','full-episode-vla-state-v1'}:
        raise ValueError('explicit joint full-episode weights required')
    config=StagedConfig(**saved['candidate'])
    if config.depth or config.training_rtc or config.time_profile!='causal_command_5hz':
        raise ValueError('unsupported full-episode deployment contract')
    binding=validate_formal_checkpoint(Path(legacy_checkpoint),repo_root/'configs/manipulation_navi_v1.json')
    if saved['base_encoder_model_id'] != binding['weights_sha256']:
        raise ValueError('full checkpoint/base tokenizer architecture identity mismatch')
    source=load_formal_policy(binding,repo_root/'artifacts/models/base',device='cpu')
    policy=nn.Module();policy.qwen=source.qwen
    del source
    policy.qwen.to(dtype=torch.float32)
    policy.qwen.load_state_dict(saved['qwen_model'],strict=True)
    model=StagedExperts(config,M0DiTConfig(vlm_hidden_dim=2560,hidden_size=1024,action_horizon=10))
    model.load_state_dict(saved['experts_model'],strict=True)
    normalizer=StagedNormalizer(saved['normalizer'])
    for key,value in saved['qwen_model'].items():
        if not torch.isfinite(value).all():raise ValueError('nonfinite Qwen weight: '+key)
    for key,value in saved['experts_model'].items():
        if not torch.isfinite(value).all():raise ValueError('nonfinite expert weight: '+key)
    del saved
    policy.to(device=device).eval();model.to(device=device).eval()
    checkpoint_id=digest(checkpoint)
    # The adapted Qwen receives a new identity; pretraining caches cannot be reused.
    return StagedRGBBackend(policy,model,normalizer,checkpoint_id,checkpoint_id,use_plan_context=True)


def validate_transition_text(raw_text, context, *, allow_invalid=False):
    """Reject invalid model text without inventing a task decision."""
    response = None
    expected = None
    try:
        response = json.loads(raw_text)
        if not isinstance(response, dict) or set(response) != {'operation', 'next_task'} or response['operation'] not in ('CONTINUE', 'ADVANCE'):
            raise ValueError('untrained or malformed transition operation')
        expected = context['active_task'] if response['operation'] == 'CONTINUE' else (
            context['remaining_tasks'][1] if len(context['remaining_tasks']) > 1 else None)
        if expected is None or response['next_task'] != expected:
            raise ValueError('transition proposal does not preserve the remaining plan')
    except ValueError as exc:
        if not allow_invalid:
            raise
        return {'proposal': None, 'valid': False, 'raw_text': raw_text,
                'parsed': response, 'expected_next_task': expected, 'error': str(exc),
                'completion_evidence': False, 'execution_authorized': False}
    result = {'proposal': response, 'completion_evidence': False, 'execution_authorized': False}
    if allow_invalid:
        result.update(valid=True, raw_text=raw_text)
    return result


def propose_transition(qwen, images, *, instruction, context, allow_invalid=False):
    from .full_episode_data import transition_prompt
    if len(images)!=4:raise ValueError('four current/history RGB frames required')
    from .full_episode_context import public_context_text
    public_context_text(context)
    prompt=transition_prompt({'input':{'original_instruction':instruction,'task_context':context}})
    inputs=dict(qwen.build_joint_trajectory_inputs([{'video':(images[:2],images[2:]),'lang':prompt}],supervise_solutions=False))
    inputs.pop('labels',None)
    with torch.inference_mode():
        tokens=qwen.model.generate(**inputs,max_new_tokens=96,do_sample=False)
    raw_text=qwen.processor.tokenizer.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    return validate_transition_text(raw_text, context, allow_invalid=allow_invalid)
