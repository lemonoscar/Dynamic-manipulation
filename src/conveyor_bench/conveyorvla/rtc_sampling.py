# Algorithm reference: MIT, Copyright (c) 2025 Physical Intelligence.
# See docs/third_party/rtc_MIT.txt; pinned commit documented in the delivery report.
"""Inference-time inpainting VJP for M0's noise=0 -> clean=1 convention.

Equations follow Physical-Intelligence/real-time-chunking-kinetix realtime_action.
This is a PyTorch clean-action adaptation, not a claim of physical replication.
"""
import math
import torch


def prefix_weights(committed, overlap, horizon, schedule='exp', *, device=None):
    if not 0 <= committed <= overlap <= horizon:
        raise ValueError('require 0 <= committed <= overlap <= horizon')
    index = torch.arange(horizon, device=device)
    weights = ((committed - 1 - index) / (overlap - committed + 1) + 1).clamp(0, 1)
    if schedule == 'exp':
        weights = weights * torch.expm1(weights) / math.expm1(1.)
    elif schedule != 'linear':
        raise ValueError('unsupported overlap schedule')
    return torch.where(index < overlap, weights, 0.)


def guidance_gain(tau, maximum):
    if not 0 <= tau < 1 or not math.isfinite(maximum) or maximum < 0:
        raise ValueError('invalid RTC time/gain')
    if tau == 0:
        return maximum
    return min(((1-tau)/tau) * (tau*tau + (1-tau)**2)/(1-tau)**2, maximum)


def vjp_velocity(predict_clean, x, tau, previous, weights):
    # Caller disables inference_mode and clones tensors. No parameter .grad accumulation.
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        clean = predict_clean(x, tau)
        velocity = (clean-x)/(1-tau)  # Match legacy sample: no training denominator clamp.
        estimate = x + (1-tau)*velocity
        residual = (weights * (previous-estimate)).detach()
        correction = torch.autograd.grad(estimate, x, grad_outputs=residual, create_graph=False)[0]
    return velocity.detach(), correction.detach()


def sample_rtc(head, vl_embeddings, state, *, previous, weights,
               encoder_attention_mask=None, noise=None, steps=None, max_guidance_weight=5.):
    if head.training:
        raise ValueError('inference RTC requires eval mode')
    count = head.config.num_inference_timesteps if steps is None else steps
    if count <= 0:
        raise ValueError('steps must be positive')
    guidance_gain(0., max_guidance_weight)
    # enable_grad alone does not undo outer policy @inference_mode.
    with torch.inference_mode(False):
        vl = vl_embeddings.detach().clone()
        s = None if state is None else state.detach().clone()
        if s is not None and s.ndim == 2:
            s = s[:, None]
        mask = None if encoder_attention_mask is None else encoder_attention_mask.detach().clone()
        previous = previous.detach().to(vl).clone()
        weights = weights.detach().to(vl).clone()
        shape = (vl.shape[0], head.config.action_horizon, head.config.action_dim)
        if previous.shape != shape or weights.shape not in (shape, (*shape[:2], 1)):
            raise ValueError('RTC previous/weights shape mismatch')
        if not torch.isfinite(previous).all() or not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
            raise ValueError('invalid RTC targets/weights')
        x = torch.randn(shape, device=vl.device, dtype=vl.dtype) if noise is None else noise.detach().to(vl).clone()
        head._validate_inputs(vl, x, s)
        if not torch.isfinite(x).all():
            raise ValueError('nonfinite RTC noise')
        def predict(value, tau):
            time = torch.full((shape[0],), tau, device=value.device, dtype=value.dtype)
            return head._predict_clean(vl, s, value, time, mask)
        for index in range(count):
            tau = index / count
            velocity, correction = vjp_velocity(predict, x, tau, previous, weights)
            x = x.detach() + (velocity + guidance_gain(tau, max_guidance_weight)*correction)/count
            if not torch.isfinite(x).all():
                raise ValueError('nonfinite RTC sample')
        return x.detach()


def prefix_conditioned_loss(head, vl_embeddings, actions, state, *, prefix_lengths,
                            noise=None, time=None, encoder_attention_mask=None):
    """Clean prefix is seen by self-attention; only legal, nonempty suffix trains.

    The caller must quarantine unknown/stale chunks before this function.
    Token-time has no new parameters, but its use needs a separately trained identity.
    """
    batch, horizon, dim = actions.shape
    head._validate_inputs(vl_embeddings, actions, state)
    if not torch.isfinite(actions).all():
        raise ValueError('unknown/nonfinite labels must be quarantined')
    lengths = torch.as_tensor(prefix_lengths, device=actions.device)
    if lengths.shape != (batch,) or (lengths < 0).any() or (lengths >= horizon).any() or (lengths != lengths.long()).any():
        raise ValueError('each sample needs a legal prefix and nonempty suffix')
    prefix = torch.arange(horizon, device=actions.device)[None] < lengths[:, None]
    noise = torch.randn_like(actions) if noise is None else noise.to(actions)
    time = torch.rand(batch, device=actions.device, dtype=actions.dtype) if time is None else time.to(actions)
    if noise.shape != actions.shape or time.shape != (batch,) or not torch.isfinite(noise).all() or not torch.isfinite(time).all() or (time < 0).any() or (time >= 1).any():
        raise ValueError('invalid training noise/time')
    token_time = torch.where(prefix, 1., time[:, None])
    x = token_time[..., None]*actions + (1-token_time[..., None])*noise
    state = None if state is None else state[:, None] if state.ndim == 2 else state
    predicted = head._predict_clean(vl_embeddings, state, x, token_time, encoder_attention_mask)
    denominator = (1-token_time[..., None]).clamp_min(head.config.time_epsilon)
    error = ((predicted-actions)/denominator).square()
    suffix = ~prefix
    return (error * suffix[..., None]).sum() / (suffix.sum()*dim)
