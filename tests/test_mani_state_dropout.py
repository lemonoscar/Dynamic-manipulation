"""State regularization must not alter old weights, evaluation or execution."""
from dataclasses import asdict, replace
import json
import pytest
import torch
from conveyor_bench.conveyorvla.dit import M0DiTConfig
from conveyor_bench.conveyorvla.staged_experts import StagedConfig, StagedExperts


def inputs(p=0.):
    torch.manual_seed(73)
    head = M0DiTConfig(vlm_hidden_dim=8, input_embedding_dim=8, hidden_size=16,
        num_attention_heads=2, attention_head_dim=4, num_layers=4,
        dropout=0., max_seq_len=32, num_target_vision_tokens=2)
    model = StagedExperts(StagedConfig(mani_state_dropout=p), head)
    conditions = dict(task_tokens=torch.randn(2,3,8), task_mask=torch.ones(2,3,dtype=torch.bool),
        live_tokens=torch.randn(2,5,8,requires_grad=True), live_mask=torch.ones(2,5,dtype=torch.bool))
    state=torch.randn(2,13,requires_grad=True)
    action=torch.randn(2,10,7);noise=torch.randn_like(action);time=torch.tensor([.3,.6])
    return model,conditions,state,action,dict(noise=noise,time=time)


def test_old_default_no_rng_or_gradient_change():
    model,c,s,a,kw=inputs()
    legacy=asdict(model.config);legacy.pop('mani_state_dropout')
    assert StagedConfig(**legacy).mani_state_dropout==0
    before=torch.random.get_rng_state().clone()
    loss=model.loss('MANIPULATION',a,s,c,**kw)
    assert torch.equal(before,torch.random.get_rng_state())
    grad=torch.autograd.grad(loss,(s,c['live_tokens']),retain_graph=True)
    # The old loss called the ordinary head directly with the same flow sample.
    vl,mask,banks=model.banks(**c)
    t=kw['time'][:,None,None];x=t*a+(1-t)*kw['noise']
    pred=model.manipulation._predict_clean(vl,s[:,None],x,kw['time'],mask,condition_banks=banks)
    reference=((pred-a)/(1-t).clamp_min(model.manipulation.config.time_epsilon)).square().sum()/a.numel()
    torch.testing.assert_close(loss,reference,rtol=0,atol=0)
    for lhs,rhs in zip(grad,torch.autograd.grad(reference,(s,c['live_tokens']))):
        torch.testing.assert_close(lhs,rhs,rtol=0,atol=0)
    assert grad[0].abs().sum()>0 and grad[1].abs().sum()>0


def test_training_drop_removes_state_and_bias_but_keeps_visual_gradient():
    model,c,s,a,kw=inputs(1.)
    loss=model.loss('MANIPULATION',a,s,c,**kw)
    other=model.loss('MANIPULATION',a,s+100,c,**kw)
    torch.testing.assert_close(loss,other,rtol=0,atol=0)
    loss.backward()
    assert s.grad.abs().sum()==0
    assert all(p.grad is not None and p.grad.abs().sum()==0 for p in model.manipulation.state_encoder.parameters())
    assert c['live_tokens'].grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.manipulation.model.parameters())


def test_eval_sample_rtc_and_nav_keep_legacy_behavior():
    model,c,s,a,kw=inputs(1.)
    reference=StagedExperts(replace(model.config,mani_state_dropout=0),model.manipulation.config)
    reference.load_state_dict(model.state_dict(),strict=True)
    model.eval();reference.eval()
    before=torch.random.get_rng_state().clone()
    torch.testing.assert_close(model.loss('MANIPULATION',a,s,c,**kw),
        reference.loss('MANIPULATION',a,s,c,**kw),rtol=0,atol=0)
    for rtc in [None,dict(previous=torch.zeros_like(a),weights=torch.ones(2,10,1),max_guidance_weight=1.)]:
        torch.testing.assert_close(model.sample('MANIPULATION',s,c,noise=kw['noise'],rtc_context=rtc),
            reference.sample('MANIPULATION',s,c,noise=kw['noise'],rtc_context=rtc),rtol=0,atol=0)
    model.train();reference.train()
    nav=a[:,:,:3];nav_kw=dict(noise=kw['noise'][:,:,:3],time=kw['time'])
    torch.testing.assert_close(model.loss('NAVIGATION',nav,None,c,**nav_kw),
        reference.loss('NAVIGATION',nav,None,c,**nav_kw),rtol=0,atol=0)
    assert torch.equal(before,torch.random.get_rng_state())
    restored=StagedConfig(**json.loads(json.dumps(asdict(model.config))))
    assert restored.mani_state_dropout==1.


def test_mask_is_per_query_and_retained_token_not_rescaled():
    model,c,s,a,kw=inputs()
    vl,mask,banks=model.banks(**c)
    keep=torch.tensor([False,True])
    head=model.manipulation
    masked=head._predict_clean(vl,s[:,None],a,kw['time'],mask,state_token_keep=keep)
    original=head._predict_clean(vl,s[:,None],a,kw['time'],mask)
    changed=head._predict_clean(vl,(s+100)[:,None],a,kw['time'],mask,state_token_keep=keep)
    torch.testing.assert_close(masked[1],original[1],rtol=0,atol=0)
    torch.testing.assert_close(masked[0],changed[0],rtol=0,atol=0)
    assert not torch.equal(masked[1],changed[1])
    with pytest.raises(ValueError):head._predict_clean(vl,s[:,None],a,kw['time'],mask,state_token_keep=torch.ones(2))


@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),float('inf')])
def test_probability_is_validated(value):
    with pytest.raises(ValueError):StagedConfig(mani_state_dropout=value)
