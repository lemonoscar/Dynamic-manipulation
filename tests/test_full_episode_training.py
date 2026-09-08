import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts.train_full_episode_vla import episode_order,optimizer_groups,FullModel
from conveyor_bench.conveyorvla.full_episode_context import public_context_text

class Core(nn.Module):
    def __init__(self,tied):
        super().__init__();self.embed_tokens=nn.Embedding(7,3);self.lm_head=nn.Linear(3,7,bias=False)
        if tied:self.lm_head.weight=self.embed_tokens.weight
        self.visual=nn.Linear(3,3);self.language=nn.Linear(3,3)
    def get_input_embeddings(self):return self.embed_tokens
    def get_output_embeddings(self):return self.lm_head

class FullEpisodeTrainingTest(unittest.TestCase):
    def test_optimizer_covers_tied_and_untied_parameters_once(self):
        for tied in (True,False):
            qwen=nn.Module();qwen.model=Core(tied)
            experts=nn.Module();experts.navigation=nn.Linear(3,3);experts.manipulation=nn.Linear(3,7)
            model=FullModel(qwen,experts)
            groups,names=optimizer_groups(model,dict(action_learning_rate=1e-5,vision_learning_rate=5e-7,qwen_learning_rate=2e-6))
            parameters=[p for g in groups for p in g['params']]
            self.assertEqual(len(parameters),len({id(p) for p in parameters}))
            self.assertEqual({id(p) for p in parameters},{id(p) for p in model.parameters()})
            self.assertEqual('qwen_embeddings_lm_head' in names,tied)
    def test_epoch_visits_each_query_once_in_episode_order(self):
        rows=[('action',dict(task_family_id=f,episode_uuid=e,query_time_s=t,observation_id=str(t)))
              for f,e in [('f1','a'),('f2','b')] for t in [2.,0.,1.]]
        order=episode_order(rows,3)
        self.assertEqual(sorted(order),list(range(6)))
        sequence=[rows[i][1] for i in order]
        self.assertEqual([r['query_time_s'] for r in sequence],[0.,1.,2.]*2)
    def test_context_rejects_truth_and_preserves_unknown(self):
        c=dict(completed_tasks=['NAV_TO_SOURCE'],active_task='PICK',remaining_tasks=['PICK','NAV_TO_TARGET','PLACE'],current_facts={})
        self.assertIn('"current_facts":{}',public_context_text(c))
        for bad in [dict(c,teacher_phase='place'),dict(c,current_facts={'carrying':True})]:
            with self.assertRaises(ValueError):public_context_text(bad)

if __name__=='__main__':unittest.main()
