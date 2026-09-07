"""Independent expert/input candidates, requiring explicitly adapted weights."""
from dataclasses import dataclass, replace
import torch
from torch import nn
from .dit import M0DiTActionHead, M0DiTConfig
from .contracts.action import TIME_PROFILES
from .geometry_encoder import GeometryEncoder


@dataclass(frozen=True)
class StagedConfig:
    name: str = 'B0-learned'
    time_profile: str = 'causal_command_5hz'
    nav_output: str = 'trajectory'
    coupling: str = 'C0'
    depth: bool = False
    training_rtc: bool = False
    bounded_gripper: bool = False
    fk_loss_weight: float = 0.
    update_period_s: float = .4
    depth_tokens: int = 16

    def __post_init__(self):
        if self.time_profile not in TIME_PROFILES or self.nav_output not in {'trajectory','endpoint'} or self.coupling not in {'C0','C1'}:
            raise ValueError('unsupported staged configuration')
        if self.update_period_s!=.4 or self.depth_tokens<=0 or self.fk_loss_weight<0:
            raise ValueError('invalid staged timing/loss budget')


class StagedExperts(nn.Module):
    def __init__(self, config, head_config=None):
        super().__init__()
        self.config=config
        base=head_config or M0DiTConfig(vlm_hidden_dim=2560)
        horizon=TIME_PROFILES[config.time_profile].horizon
        self.navigation=M0DiTActionHead(replace(base,action_dim=3,state_dim=0,
            action_horizon=1 if config.nav_output=='endpoint' else 10))
        self.manipulation=M0DiTActionHead(replace(base,action_dim=7,state_dim=13,action_horizon=horizon))
        self.geometry=GeometryEncoder(base.vlm_hidden_dim,config.depth_tokens) if config.depth else None

    def banks(self, task_tokens, live_tokens, task_mask, live_mask, *, points=None, depth_valid=None):
        if self.geometry is not None:
            if points is None or depth_valid is None:
                raise ValueError('RGB-D needs explicit points and valid mask, including all-missing depth')
            geom,mask=self.geometry(points,depth_valid)
            live_tokens=torch.cat((live_tokens,geom),1)
            live_mask=torch.cat((live_mask,mask),1)
        elif points is not None:
            raise ValueError('RGB candidate must not receive depth')
        combined=torch.cat((task_tokens,live_tokens),1)
        mask=torch.cat((task_mask,live_mask),1)
        if not task_mask.any(1).all() or not live_mask.any(1).all():
            raise ValueError('each task/live bank needs a valid token')
        # C0: every cross block sees one concatenated bank.
        # C1: successive cross blocks alternate task and current live banks.
        banks=None if self.config.coupling=='C0' else ((task_tokens,task_mask),(live_tokens,live_mask))
        return combined,mask,banks

    def clean(self, head, vl, state, x, time, mask, banks):
        predicted=head._predict_clean(vl,state,x,time,mask,condition_banks=banks)
        if head is self.manipulation and self.config.bounded_gripper:
            predicted=torch.cat((predicted[:,:,:6],2*torch.sigmoid(predicted[:,:,6:])-1),-1)
        return predicted

    def loss(self, domain, actions, state, conditions, *, prefix_lengths=None, noise=None, time=None,
             fk=None, anchor_q=None, denormalize=None):
        head=self.manipulation if domain=='MANIPULATION' else self.navigation if domain=='NAVIGATION' else None
        if head is None:
            raise ValueError('unknown expert domain')
        if domain=='NAVIGATION' and self.config.nav_output=='endpoint' and actions.shape[1]==10:
            actions=actions[:,-1:]
        vl,mask,banks=self.banks(**conditions)
        head._validate_inputs(vl,actions,state)
        if not torch.isfinite(actions).all():
            raise ValueError('quarantine invalid action chunks before training')
        batch,horizon,dim=actions.shape
        state=None if state is None else state[:,None] if state.ndim==2 else state
        noise=torch.randn_like(actions) if noise is None else noise.to(actions)
        if time is None:
            draw=head.beta_dist.sample((batch,)).to(actions)
            time=(head.config.noise_s-draw)/head.config.noise_s
        else:
            time=time.to(actions)
        suffix=torch.ones((batch,horizon),device=actions.device,dtype=torch.bool)
        token_time=time[:,None].expand(-1,horizon)
        if self.config.training_rtc and domain=='MANIPULATION':
            if prefix_lengths is None:
                raise ValueError('training RTC requires explicitly sampled delay/prefix lengths')
            lengths=torch.as_tensor(prefix_lengths,device=actions.device)
            if lengths.shape!=(batch,) or (lengths<0).any() or (lengths>=horizon).any() or (lengths!=lengths.long()).any():
                raise ValueError('RTC needs nonempty suffix per sample')
            suffix=torch.arange(horizon,device=actions.device)[None]>=lengths[:,None]
            token_time=torch.where(suffix,token_time,1.)
        x=token_time[...,None]*actions+(1-token_time[...,None])*noise
        predicted=self.clean(head,vl,state,x,token_time if self.config.training_rtc and domain=='MANIPULATION' else time,mask,banks)
        error=((predicted-actions)/(1-token_time[...,None]).clamp_min(head.config.time_epsilon)).square()
        loss=(error*suffix[...,None]).sum()/(suffix.sum()*dim)
        if domain=='MANIPULATION' and self.config.fk_loss_weight:
            if fk is None or anchor_q is None or denormalize is None:
                raise ValueError('FK needs differentiable kinematics, measured anchor and bound normalizer')
            pred=denormalize(predicted)[...,:6]+anchor_q[:,None]
            truth=denormalize(actions)[...,:6]+anchor_q[:,None]
            # fk returns matched Cartesian position and rotation matrix; no object truth.
            pp,pr=fk(pred);tp,tr=fk(truth)
            pos=(pp-tp).square().sum(-1)
            cosine=((pr.transpose(-1,-2)@tr).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
            # chordal SO(3) metric is differentiable at exact agreement.
            rotation=2*(1-cosine.clamp(-1,1))
            loss=loss+self.config.fk_loss_weight*((pos+rotation)*suffix).sum()/suffix.sum()
        return loss

    @torch.no_grad()
    def sample(self, domain, state, conditions, *, noise=None, steps=None):
        if self.config.training_rtc:
            raise ValueError('training-time RTC deployment requires an explicit committed prefix; use sample_prefix')
        return self.sample_prefix(domain,state,conditions,noise=noise,steps=steps)

    @torch.no_grad()
    def sample_prefix(self, domain, state, conditions, *, prefix=None, prefix_length=0, noise=None, steps=None):
        head=self.manipulation if domain=='MANIPULATION' else self.navigation
        vl,mask,banks=self.banks(**conditions)
        state=None if state is None else state[:,None] if state.ndim==2 else state
        shape=(vl.shape[0],head.config.action_horizon,head.config.action_dim)
        x=torch.randn(shape,device=vl.device,dtype=vl.dtype) if noise is None else noise.to(vl).clone()
        head._validate_inputs(vl,x,state)
        count=head.config.num_inference_timesteps if steps is None else steps
        if count<=0 or not 0<=prefix_length<shape[1]:
            raise ValueError('invalid sampling steps/prefix length')
        if prefix_length and (not self.config.training_rtc or prefix is None or prefix.shape!=x[:,:prefix_length].shape):
            raise ValueError('clean prefix requires trained token-time contract and aligned shape')
        for i in range(count):
            time=torch.full((shape[0],),i/count,device=x.device,dtype=x.dtype)
            if prefix_length:
                x[:,:prefix_length]=prefix
                token_time=time[:,None].expand(-1,shape[1]).clone();token_time[:,:prefix_length]=1.
            else:
                token_time=time
            pred=self.clean(head,vl,state,x,token_time,mask,banks)
            x=x+(pred-x)/(1-i/count)/count
        if prefix_length:
            x[:,:prefix_length]=prefix  # explicit training-time conditioning, not inference VJP
        return x
