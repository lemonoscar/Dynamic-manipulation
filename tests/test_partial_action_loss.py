import unittest
import torch
from conveyor_bench.conveyorvla.staged_experts import StagedExperts, StagedConfig
from conveyor_bench.conveyorvla.dit import M0DiTConfig

class PartialActionLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        c=M0DiTConfig(vlm_hidden_dim=8,input_embedding_dim=8,hidden_size=12,
            num_attention_heads=2,attention_head_dim=4,num_layers=4,dropout=0.,
            max_seq_len=64,num_target_vision_tokens=2)
        self.model=StagedExperts(StagedConfig(),c)
        self.conditions=dict(task_tokens=torch.randn(1,2,8),live_tokens=torch.randn(1,4,8),
            task_mask=torch.ones(1,2,dtype=torch.bool),live_mask=torch.ones(1,4,dtype=torch.bool))
    def test_padding_is_not_a_target_or_condition(self):
        for domain,dim in [('NAVIGATION',3),('MANIPULATION',7)]:
            actions=torch.randn(1,10,dim);noise=torch.randn_like(actions)
            state=None if dim==3 else torch.zeros(1,13)
            mask=torch.tensor([[True]*3+[False]*7]);other=actions.clone();other[:,3:]=99999
            kwargs=dict(action_valid_mask=mask,noise=noise,time=torch.tensor([.3]))
            one=self.model.loss(domain,actions,state,self.conditions,**kwargs)
            two=self.model.loss(domain,other,state,self.conditions,**kwargs)
            self.assertTrue(torch.equal(one,two))
            one.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.parameters()))
    def test_full_mask_preserves_original_loss(self):
        a=torch.randn(1,10,3);kwargs=dict(noise=torch.randn_like(a),time=torch.tensor([.4]))
        old=self.model.loss('NAVIGATION',a,None,self.conditions,**kwargs)
        new=self.model.loss('NAVIGATION',a,None,self.conditions,action_valid_mask=torch.ones(1,10,dtype=torch.bool),**kwargs)
        self.assertTrue(torch.equal(old,new))
    def test_empty_or_discontinuous_mask_rejected(self):
        for mask in [torch.zeros(1,10,dtype=torch.bool),torch.tensor([[True,False,True]+[False]*7])]:
            with self.assertRaises(ValueError):
                self.model.loss('NAVIGATION',torch.zeros(1,10,3),None,self.conditions,action_valid_mask=mask)

if __name__=='__main__':unittest.main()
