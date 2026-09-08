import tempfile,unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from conveyor_bench.conveyorvla import full_episode_backend as backend
from conveyor_bench.conveyorvla.staged_experts import StagedConfig

class Expert(nn.Module):
    def __init__(self,config,unused=None):
        super().__init__();self.config=config;self.weight=nn.Parameter(torch.zeros(1))

class FullReloadPrecisionTest(unittest.TestCase):
    def test_fp32_update_is_not_rounded_by_legacy_bf16_architecture(self):
        policy=nn.Module();policy.qwen=nn.Linear(1,1,bias=False).bfloat16()
        value=torch.tensor([[1.0001]],dtype=torch.float32)
        self.assertFalse(torch.equal(value,value.bfloat16().float()))
        saved=dict(schema='full-episode-vla-weights-v1',candidate=asdict(StagedConfig()),
            base_encoder_model_id='old',qwen_model={'weight':value},experts_model={'weight':torch.ones(1)},normalizer={})
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.pt';torch.save(saved,path)
            with patch.object(backend,'validate_formal_checkpoint',return_value={'weights_sha256':'old'}), \
                 patch.object(backend,'load_formal_policy',return_value=policy), \
                 patch.object(backend,'StagedExperts',Expert), \
                 patch.object(backend,'StagedNormalizer',return_value=object()):
                loaded=backend.load_full_episode_backend(path,legacy_checkpoint='old',repo_root=temporary)
            self.assertTrue(torch.equal(loaded.policy.qwen.weight.detach(),value))
            self.assertEqual(loaded.policy.qwen.weight.dtype,torch.float32)

if __name__=='__main__':unittest.main()
