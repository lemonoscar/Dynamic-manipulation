"""Shared frozen RGB encoder and the staged action backend for RollingRuntime.

The caller loads the encoder and candidate with their verified checkpoint IDs.
This adapter generates proposals; the runtime still owns epoch, queue and safety.
"""
import numpy as np
import torch
from .joint_trajectory import canonical_solution


def rgb_condition_semantic(instruction, primitive, images, *, context=None):
    if not isinstance(instruction, str) or not instruction:
        raise ValueError('missing source instruction')
    if primitive not in {'PICK', 'PLACE', 'NAV_TO_SOURCE', 'NAV_TO_TARGET'}:
        raise ValueError('unknown action route')
    if len(images) != 4:
        raise ValueError('staged RGB requires four current/history images')
    subtask = 'Lower, release, and verify placement.' if primitive == 'PLACE' else primitive
    if context is not None and (not isinstance(context, str) or not context.strip()):
        raise ValueError('task context must be nonempty public text or None')
    semantic = instruction + '\nCurrent task: ' + subtask
    return semantic if context is None else semantic + '\nTask context:\n' + context


def _rgb_task_inputs(policy, semantic):
    device = next(policy.qwen.parameters()).device
    tokens = policy.qwen.processor.tokenizer(semantic, return_tensors='pt')
    return {k: v.to(device) for k, v in tokens.items()}


def _rgb_live_inputs(policy, semantic, primitive, images):
    images = list(images)
    inputs = dict(policy.qwen.build_joint_trajectory_inputs(
        [{'video': (images[:2], images[2:]), 'lang': semantic}],
        solutions=[canonical_solution(primitive).replace(
            'Lower, release, and retract from the destination box.',
            'Lower, release, and verify placement.')], supervise_solutions=False))
    inputs.pop('labels', None)
    return inputs


@torch.inference_mode()
def encode_rgb_conditions(policy, instruction, primitive, images, task_cache=None, *, context=None):
    """Frozen cache/live inference path; returns the original CPU tensor banks.

    Images are head[t-.2,t], wrist[t-.2,t]. The optional cache belongs to this
    frozen policy; live RGB tokens are always recomputed. No depth is consumed.
    """
    semantic = rgb_condition_semantic(instruction, primitive, images, context=context)
    key = (id(policy.qwen), semantic)
    cached = None if task_cache is None else task_cache.get(key)
    if cached is None:
        tokens = _rgb_task_inputs(policy, semantic)
        out = policy.qwen(**tokens, output_hidden_states=True, return_dict=True, use_cache=False)
        cached = (out.hidden_states[-1].float().cpu(), tokens['attention_mask'].bool().cpu())
        if task_cache is not None:
            task_cache[key] = cached
    inputs = _rgb_live_inputs(policy, semantic, primitive, images)
    out = policy.qwen(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    task_tokens, task_mask = cached
    return dict(task_tokens=task_tokens, task_mask=task_mask,
        live_tokens=out.hidden_states[-1].float().cpu(),
        live_mask=inputs['attention_mask'].bool().cpu())


def encode_rgb_training_conditions(policy, instruction, primitive, images, *, context=None):
    """Differentiable RGB action conditions, with no tensor cache or CPU detach.

    This retains exactly the staged action prompt and bank construction. It is
    action conditioning on a known active task, NOT a causal route-prediction
    prompt: use a separate query without the teacher route for route/answer CE.
    Qwen must be unfrozen by the caller; precision/autocast belong to the runner.
    """
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise ValueError('training RGB encoding requires an enabled autograd context')
    if not any(parameter.requires_grad for parameter in policy.qwen.parameters()):
        raise ValueError('training RGB encoding requires trainable Qwen parameters')
    semantic = rgb_condition_semantic(instruction, primitive, images, context=context)
    tokens = _rgb_task_inputs(policy, semantic)
    task_out = policy.qwen(**tokens, output_hidden_states=True, return_dict=True, use_cache=False)
    inputs = _rgb_live_inputs(policy, semantic, primitive, images)
    live_out = policy.qwen(**inputs, output_hidden_states=True, return_dict=True, use_cache=False)
    task_hidden, live_hidden = task_out.hidden_states[-1], live_out.hidden_states[-1]
    if not task_hidden.requires_grad or not live_hidden.requires_grad:
        raise ValueError('Qwen detached its training condition graph')
    return dict(task_tokens=task_hidden.float(), task_mask=tokens['attention_mask'].bool(),
        live_tokens=live_hidden.float(), live_mask=inputs['attention_mask'].bool())


class StagedRGBBackend:
    """Callable compatible with run_rolling_episode's action_backend boundary.

    model_id is the staged checkpoint identity; encoder_model_id is the frozen
    encoder identity used by that checkpoint's condition cache. The caller must
    verify these when loading weights, exactly as for FrozenActionBackend.
    """
    def __init__(self, policy, model, normalizer, model_id, encoder_model_id, *, use_plan_context=False):
        if not model_id or not encoder_model_id:
            raise ValueError('explicit staged and encoder checkpoint identities required')
        if model.config.depth or model.config.coupling != 'C0' or model.config.training_rtc or model.config.bounded_gripper:
            raise ValueError('live staged backend requires RGB C0 inference-VJP contract')
        self.policy = policy.eval()
        self.model = model.eval()
        self.normalizer = normalizer
        self.model_id = model_id
        self.encoder_model_id = encoder_model_id
        self.task_cache = {}
        self.use_plan_context = use_plan_context

    def __call__(self, request, original_instruction):
        identity = request.identity
        if identity.model_id != self.model_id or identity.normalizer_id != self.normalizer.payload['normalizer_id']:
            raise ValueError('stale/foreign staged model or normalizer identity')
        if getattr(request, 'time_profile', None) != self.model.config.time_profile:
            raise ValueError('request/staged checkpoint action time profile mismatch')
        obs, task = request.observation, request.task
        if identity.mission_id != obs.mission_id or identity.active_task_id != task.task_id:
            raise ValueError('cross-mission/task action request')
        if task.target_ref != 'cola' or task.destination_ref != 'destination':
            raise ValueError('staged RGB contract cannot represent a changed target identity')
        if len(obs.images) != 4 or not np.allclose(obs.image_times_s,
                (obs.time_s-.2, obs.time_s, obs.time_s-.2, obs.time_s), atol=1e-7, rtol=0):
            raise ValueError('staged RGB requires exact causal 0.2s camera history')
        mani = task.primitive in {'PICK', 'PLACE'}
        context = None
        if self.use_plan_context:
            from .full_episode_context import public_context_text
            context = public_context_text(request.plan_context)
            if request.plan_context['active_task'] != task.primitive:
                raise ValueError('action context differs from canonical active task')
        conditions = encode_rgb_conditions(self.policy, original_instruction,
            task.primitive, obs.images, self.task_cache, context=context)
        head = self.model.manipulation if mani else self.model.navigation
        parameter = next(head.parameters())
        # inference_mode tensors from Qwen/cache cannot enter a VJP backward.
        with torch.inference_mode(False):
            conditions = {k: v.detach().clone().to(device=parameter.device,
                dtype=parameter.dtype if v.is_floating_point() else v.dtype)
                for k, v in conditions.items()}
            state = torch.as_tensor(self.normalizer.normalize_mani_state(obs.mani_state),
                device=parameter.device, dtype=parameter.dtype)[None].clone() if mani else None
            normalized = self.model.sample('MANIPULATION' if mani else 'NAVIGATION',
                state, conditions, rtc_context=request.rtc_context)
            return self.normalizer.denormalize_action(task.primitive, normalized)[0].detach().float().cpu().numpy()
