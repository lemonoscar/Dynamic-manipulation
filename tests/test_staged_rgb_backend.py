"""Shared RGB cache/live encoding and RollingRuntime backend contract."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from conveyor_bench.conveyorvla.contracts.action import ActionIdentity
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from conveyor_bench.conveyorvla.joint_trajectory import canonical_solution
from conveyor_bench.conveyorvla.task_memory import Task
from conveyor_bench.conveyorvla.staged_experts import StagedConfig
from conveyor_bench.conveyorvla import staged_rgb_backend as module


class Qwen(nn.Module):
    def __init__(self):
        super().__init__(); self.anchor = nn.Parameter(torch.zeros(1)); self.calls = 0; self.prompts = []
        self.processor = SimpleNamespace(tokenizer=self.tokenize)

    def tokenize(self, text, return_tensors):
        return {'input_ids':torch.tensor([[len(text), sum(map(ord,text)) % 100]]),
            'attention_mask':torch.ones(1,2,dtype=torch.long)}

    def build_joint_trajectory_inputs(self, examples, solutions, supervise_solutions):
        self.prompts.append((examples, solutions, supervise_solutions))
        frames = examples[0]['video'][0] + examples[0]['video'][1]
        return {'input_ids':torch.tensor([[len(examples[0]['lang'])]+list(frames)]),
            'attention_mask':torch.ones(1,5,dtype=torch.long),'labels':torch.tensor([[-1]])}

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict, use_cache):
        self.calls += 1
        return SimpleNamespace(hidden_states=(input_ids.float()[...,None].expand(-1,-1,8).clone(),))


class Policy(nn.Module):
    def __init__(self):
        super().__init__(); self.qwen = Qwen()


def reference_original_cache(policy, instruction, primitive, images):
    # The old cache script's exact prompts and task/live construction.
    subtask='Lower, release, and verify placement.' if primitive=='PLACE' else primitive
    semantic=instruction+'\nCurrent task: '+subtask
    tokens=policy.qwen.processor.tokenizer(semantic,return_tensors='pt')
    out=policy.qwen(**tokens,output_hidden_states=True,return_dict=True,use_cache=False)
    task_tokens,task_mask=out.hidden_states[-1].float().cpu(),tokens['attention_mask'].bool().cpu()
    inputs=dict(policy.qwen.build_joint_trajectory_inputs([{'video':(images[:2],images[2:]),
        'lang':semantic}],solutions=[canonical_solution(primitive).replace(
        'Lower, release, and retract from the destination box.','Lower, release, and verify placement.')],
        supervise_solutions=False))
    inputs.pop('labels',None)
    out=policy.qwen(**inputs,output_hidden_states=True,return_dict=True,use_cache=False)
    return dict(task_tokens=task_tokens,task_mask=task_mask,live_tokens=out.hidden_states[-1].float().cpu(),
        live_mask=inputs['attention_mask'].bool().cpu())


@pytest.mark.parametrize('primitive',['PICK','PLACE','NAV_TO_SOURCE','NAV_TO_TARGET'])
def test_shared_encoder_exact_original_cache_prompts_and_four_rgb(primitive):
    policy=Policy().eval(); images=[11,12,21,22]
    expected=reference_original_cache(policy,'transfer cola',primitive,images)
    cache={}; actual=module.encode_rgb_conditions(policy,'transfer cola',primitive,tuple(images),cache)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    assert policy.qwen.prompts[-1] == policy.qwen.prompts[-2]
    assert set(actual)=={'task_tokens','task_mask','live_tokens','live_mask'}
    before=policy.qwen.calls
    refreshed=module.encode_rgb_conditions(policy,'transfer cola',primitive,[13,14,23,24],cache)
    assert policy.qwen.calls==before+1  # Reuse task tokens, recompute live RGB.
    torch.testing.assert_close(refreshed['task_tokens'],actual['task_tokens'],rtol=0,atol=0)
    assert not torch.equal(refreshed['live_tokens'],actual['live_tokens'])


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.config=StagedConfig(); self.manipulation=nn.Linear(1,1); self.navigation=nn.Linear(1,1)
        self.called=None

    def sample(self, domain, state, conditions, rtc_context):
        assert all(not value.is_inference() for value in conditions.values())
        assert state is None or not state.is_inference()
        self.called=(domain,state,conditions,rtc_context)
        return torch.zeros(1,10,7 if domain=='MANIPULATION' else 3)


class Norm:
    payload={'normalizer_id':'norm'}
    def normalize_mani_state(self, value):return np.asarray(value)*.5
    def denormalize_action(self, route, value):return value+1.


def request():
    identity=ActionIdentity('mission',0,2,'pick','model','norm','safety')
    obs=CurrentObservation('mission','obs',.4,(.2,)*6,(.4,)*6,.3,
        (11,12,21,22),(.2,.4,.2,.4))
    return SimpleNamespace(identity=identity, observation=obs,
        task=Task('pick','attempt','PICK','cola','destination'),
        time_profile='causal_command_5hz',rtc_context={'previous':[[0.]*7]*10,'weights':[1.]*8+[0.]*2})


def test_backend_shared_encoding_identity_state_and_rtc_passthrough(monkeypatch):
    policy=Policy();model=Model();backend=module.StagedRGBBackend(policy,model,Norm(),'model','encoder')
    original=module.encode_rgb_conditions;calls=[]
    def spy(*args,**kwargs):calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(module,'encode_rgb_conditions',spy)
    r=request()
    with torch.inference_mode():result=backend(r,'original mission')
    assert not model.training and not policy.training
    assert len(calls)==1 and calls[0][1:4]==('original mission','PICK',r.observation.images)
    domain,state,conditions,context=model.called
    assert domain=='MANIPULATION' and context is r.rtc_context
    torch.testing.assert_close(state,torch.tensor([r.observation.mani_state])*.5)
    np.testing.assert_array_equal(result,np.ones((10,7)))


@pytest.mark.parametrize('field,value', [('model_id','foreign'),('normalizer_id','foreign'),('mission_id','foreign')])
def test_backend_rejects_foreign_identity(field,value):
    backend=module.StagedRGBBackend(Policy(),Model(),Norm(),'model','encoder');r=request()
    r.identity=replace(r.identity,**{field:value})
    with pytest.raises(ValueError,match='identity|cross-mission'):
        backend(r,'mission')


def test_backend_rejects_profile_history_and_target_changes():
    backend=module.StagedRGBBackend(Policy(),Model(),Norm(),'model','encoder')
    r=request();r.time_profile='legacy_future_5hz'
    with pytest.raises(ValueError,match='time profile'):backend(r,'mission')
    r=request();r.observation=replace(r.observation,image_times_s=(.1,.4,.2,.4))
    with pytest.raises(ValueError,match='camera history'):backend(r,'mission')
    r=request();r.task=replace(r.task,target_ref='other')
    with pytest.raises(ValueError,match='target identity'):backend(r,'mission')
