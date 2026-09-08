"""CPU-only audit of the actual runner's research scheduler and NAV wiring."""
import ast
from dataclasses import dataclass,asdict
import math
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'run_full_episode_diagnostic.py').read_text())
advance=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='advance_model_claim')
pipeline=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pipeline_type')
cls=next(n for n in pipeline.body if isinstance(n,ast.ClassDef))
navigation=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_navigation_window')
ns={'asdict':asdict}
exec(compile(ast.Module(body=[advance],type_ignores=[]),'actual_advance','exec'),ns)

@dataclass
class Task:
    primitive:str

class Memory:
    def __init__(self):
        self.tasks=tuple(Task(n) for n in ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'))
        self.events=[];self.cursor=0;self.active_task_epoch=0;self.plan_version=0
    @property
    def active_task(self):return self.tasks[self.cursor]

class DiagnosticAudit(unittest.TestCase):
    def test_model_claim_is_not_completion_evidence(self):
        m=Memory();context={'completed_tasks':[],'active_task':'NAV_TO_SOURCE',
            'remaining_tasks':[t.primitive for t in m.tasks],'current_facts':{}}
        result=ns['advance_model_claim'](m,context,SimpleNamespace(time_s=1.,observation_id='o'))
        self.assertEqual((m.cursor,m.active_task_epoch,m.plan_version),(1,1,1))
        self.assertEqual(result['active_task'],'PICK')
        self.assertEqual(result['completed_tasks'],['NAV_TO_SOURCE'])
        self.assertEqual(m.events[0]['kind'],'MODEL_ADVANCE_CLAIM')
        self.assertFalse(m.events[0]['completion_verified'])
        self.assertEqual(result['current_facts'],{})

    def test_wrong_task_and_last_task_cannot_advance(self):
        m=Memory()
        with self.assertRaises(ValueError):ns['advance_model_claim'](m,{'active_task':'PICK'},SimpleNamespace())
        m.cursor=3
        with self.assertRaises(ValueError):ns['advance_model_claim'](m,{'active_task':'PLACE'},SimpleNamespace())
        self.assertEqual(m.events,[])

    def test_navigation_reuses_one_map_for_stateful_dwa(self):
        # Structural guard for the actual stateful DWA map identity contract.
        calls=[n for n in ast.walk(navigation) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='_local_map']
        self.assertEqual(len(calls),1)
        loop=next(n for n in navigation.body if isinstance(n,ast.For))
        self.assertNotIn(calls[0],list(ast.walk(loop)),
            'Recreating inflated map every tick resets ArmVLADWAControllerAdapter state')

    def test_advance_branch_discards_actions_before_execution(self):
        run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_episode')
        branches=[n for n in ast.walk(run) if isinstance(n,ast.If)
                  and 'transition' in ast.unparse(n.test) and "'ADVANCE'" in ast.unparse(n.test)]
        self.assertEqual(len(branches),1)
        nested=branches[0].body[0]
        code=ast.unparse(nested)
        self.assertIn('runtime.pending.pop(request.request_id)',code)
        self.assertIn('runtime.synchronize_task()',code)
        self.assertIsInstance(nested.body[-1],ast.Continue)

if __name__=='__main__':unittest.main()
