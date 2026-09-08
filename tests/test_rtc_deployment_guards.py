"""CPU regression for stop semantics and candidate deployment identity guards."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from conveyor_bench.conveyorvla.joint_trajectory_runtime import JointSafetyLimits
from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
from conveyor_bench.conveyorvla.task_memory import TaskMemory, Task
from conveyor_bench.conveyorvla.staged_experts import StagedConfig, StagedExperts
from conveyor_bench.conveyorvla.dit import M0DiTConfig


class Controller:
    time = 0.
    calls = None

    def __init__(self):
        self.calls = []

    def observe(self):
        return CurrentObservation('m', str(self.time), self.time, (0.,)*6, (0.,)*6, .25)

    def apply(self, target, observation):
        self.calls.append(('apply', target)); self.time += .02
        return target

    def hold(self, observation, reason):
        self.calls.append(('hold', reason)); self.time += .02

    def stop(self, observation, reason):
        self.calls.append(('stop', reason)); self.time += .02


def runtime(controller, register=True):
    class Norm:
        payload = {'normalizer_id': 'test'}
        def normalize_action(self, route, actions): return actions
    memory = TaskMemory('m', 'transfer', (Task('p', 'a', 'PICK', 'cola'),))
    if register: memory.observe(controller.observe(), [])
    return RollingRuntime(memory, model_id='test', normalizer=Norm(), safety_context_id='test',
        limits=JointSafetyLimits((-2.,)*6, (2.,)*6, (1.,)*6))


def test_unsafe_target_latches_stop_rejects_inflight_and_new_actions():
    c = Controller(); r = runtime(c); o = c.observe()
    request = r.prepare_request(o)
    actions = [(0.1,)*6 + (.25,)]*10
    r.complete_request(request.request_id, actions, now_s=0.)
    pending = r.prepare_request(o)
    r.tick(o, safety=lambda target, obs: (False, 'collision'), controller=c)
    r.tick(c.observe(), safety=lambda target, obs: (True, ''), controller=c)
    assert [kind for kind, _ in c.calls] == ['stop', 'stop']
    assert r.safety_stop_reason == 'collision' and not r.queue.history
    with pytest.raises(ValueError, match='safety_stop_latched'):
        r.complete_request(pending.request_id, actions, now_s=c.time)
    with pytest.raises(ValueError, match='safety_stop_latched'):
        r.prepare_request(o)


def test_driver_finalization_never_reissues_hold_after_stop():
    from conveyor_bench.conveyorvla.rolling_episode import run_rolling_episode
    class Planner:
        def prepare(self, observation, feedback): return ({},)
        def backend(self, request): return None
        def commit_response(self, prepared, response): return False
    c = Controller(); r = runtime(c, register=False)
    result = run_rolling_episode(r, Planner(), lambda request, instruction: [(0.1,)*6+(.25,)]*10,
        observer=c.observe, feedback_estimator=lambda obs: (), controller=c,
        safety=lambda target, obs: (False, 'collision'), max_control_ticks=3)
    assert result['control_ticks'] == 1 and result['safety_stop_reason'] == 'collision'
    assert c.calls == [('stop', 'collision'), ('stop', 'collision')]


@pytest.mark.parametrize('domain,prefix_length', [('TYPO', 0), ('NAVIGATION', 2)])
def test_staged_rejects_unknown_domain_and_untrained_nav_prefix(domain, prefix_length):
    base = M0DiTConfig(vlm_hidden_dim=8, input_embedding_dim=8, hidden_size=12,
        num_attention_heads=2, attention_head_dim=4, num_layers=2, dropout=0.,
        max_seq_len=64, num_target_vision_tokens=2)
    model = StagedExperts(StagedConfig(training_rtc=True), base).eval()
    with pytest.raises(ValueError, match='unknown expert domain|NAV clean prefix'):
        model.sample_prefix(domain, None, {}, prefix_length=prefix_length)


def test_cli_rejects_unbound_prefix_before_loading_weights(tmp_path):
    from scripts import infer_staged_vla as cli
    with pytest.raises(ValueError, match='unbound prefix'):
        cli.main(['--checkpoint', 'missing', '--condition', 'missing', '--record', 'missing',
            '--output', str(tmp_path/'out.json'), '--prefix', 'unbound.json'])


def test_cli_rejects_profile_mismatch_before_loading_condition(tmp_path, monkeypatch):
    from scripts import infer_staged_vla as cli
    record = tmp_path/'record.json'
    record.write_text(json.dumps({'route': 'PICK', 'time_profile': 'legacy_future_5hz'}))
    monkeypatch.setattr(cli.torch, 'load', lambda *args, **kwargs: {
        'training_steps': 1, 'candidate': asdict(StagedConfig()), 'normalizer': {}})
    monkeypatch.setattr(cli, 'StagedNormalizer', lambda payload: None)
    with pytest.raises(ValueError, match='record/checkpoint action time profile mismatch'):
        cli.main(['--checkpoint', 'missing', '--condition', 'missing', '--record', str(record),
            '--output', str(tmp_path/'out.json')])


def test_staged_vjp_off_exact_guidance_finite_and_no_parameter_grad():
    torch.manual_seed(31)
    base = M0DiTConfig(vlm_hidden_dim=8, input_embedding_dim=8, hidden_size=12,
        num_attention_heads=2, attention_head_dim=4, num_layers=2, dropout=0.,
        max_seq_len=64, num_target_vision_tokens=2)
    model = StagedExperts(StagedConfig(), base).eval()
    conditions = dict(task_tokens=torch.randn(1,2,8), live_tokens=torch.randn(1,4,8),
        task_mask=torch.ones(1,2,dtype=torch.bool), live_mask=torch.ones(1,4,dtype=torch.bool))
    state = torch.zeros(1,13); noise = torch.randn(1,10,7)
    legacy = model.sample_prefix('MANIPULATION',state,conditions,noise=noise)
    off = model.sample('MANIPULATION',state,conditions,noise=noise)
    context = {'previous':torch.zeros(10,7),'weights':torch.ones(10),'max_guidance_weight':0.}
    with torch.inference_mode():
        zero = model.sample('MANIPULATION',state,conditions,noise=noise,rtc_context=context)
        context['max_guidance_weight'] = 1.
        guided = model.sample('MANIPULATION',state,conditions,noise=noise,rtc_context=context)
    torch.testing.assert_close(off,legacy,rtol=0,atol=0)
    torch.testing.assert_close(zero,legacy,rtol=0,atol=0)
    assert torch.isfinite(guided).all() and not torch.equal(guided,off)
    assert all(p.grad is None for p in model.parameters())


def test_staged_rtc_context_codec_and_identity_rejections():
    from scripts.infer_staged_vla import validate_rtc_context
    class Norm:
        payload = {'normalizer_id':'n'}
        def normalize_action(self, route, actions): return actions
    row = dict(episode_uuid='ep', active_task_id='pick', active_task_epoch=3,
        observation_id='q2', query_time_s=1., time_profile='causal_command_5hz',
        route='PICK', mani_state=[.1]*6+[0.]*6+[.25])
    context = dict(schema='staged-rtc-context-v1', checkpoint_sha256='model',
        normalizer_id='n', **{k:row[k] for k in ('episode_uuid','active_task_id',
        'active_task_epoch','observation_id','query_time_s','time_profile')},
        source_query_time_s=.6, source_action_id='a1', previous_absolute=[[.2]*6+[.25]]*10,
        target_apply_times_s=(1.+np.arange(10)*.2).tolist(), available_previous=[True]*8+[False]*2,
        weights=[1.]*8+[0.]*2, max_guidance_weight=5.)
    result = validate_rtc_context(context,row,StagedConfig(),Norm(),'model')
    np.testing.assert_allclose(result['previous'][:,:6],.1)
    np.testing.assert_allclose(result['previous'][:,6],.25)
    for patch in ({'checkpoint_sha256':'other'}, {'normalizer_id':'other'},
                  {'active_task_epoch':2}, {'time_profile':'legacy_future_5hz'},
                  {'target_apply_times_s':(1.2+np.arange(10)*.2).tolist()}, {'weights':[1.]*10}):
        with pytest.raises(ValueError):
            validate_rtc_context({**context,**patch},row,StagedConfig(),Norm(),'model')
