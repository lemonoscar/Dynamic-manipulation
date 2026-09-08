"""CPU tests of actual parser/proposal/service code; only model and tensor APIs are doubles."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SERVICE_PATH=HERE/'serve_full_episode_diagnostic.py'
BACKEND_PATH=HERE/'full_episode_backend.py'
if not BACKEND_PATH.exists():
    BACKEND_PATH=HERE.parent/'src/conveyor_bench/conveyorvla/full_episode_backend.py'

def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(selected)==len(names),(path,names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),namespace)

package='conveyor_bench.conveyorvla'
# These two pure shared sources are the real training/deployment serializers.
context_path=HERE.parent/'src/conveyor_bench/conveyorvla/full_episode_context.py'
data_path=HERE.parent/'src/conveyor_bench/conveyorvla/full_episode_data.py'
if not context_path.exists():
    source=Path('/home/lemon/research/Issac/doc/evaluation_step1700_20260908/source/conveyorvla')
    context_path=source/'full_episode_context.py';data_path=source/'full_episode_data.py'
context=types.ModuleType(package+'.full_episode_context');exec(context_path.read_text(),context.__dict__)
data=types.ModuleType(package+'.full_episode_data');exec(data_path.read_text(),data.__dict__)

class Record(types.SimpleNamespace):pass
class Tokens:
    def __getitem__(self,key):return [901,902]
class Qwen:
    def __init__(self,text):
        self.text=text;self.calls=0;self.failure=None
        self.model=types.SimpleNamespace(generate=self.generate)
        self.processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(decode=lambda *a,**kw:self.text))
    def build_joint_trajectory_inputs(self,*a,**kw):return {'input_ids':types.SimpleNamespace(shape=(1,4)),'labels':'removed'}
    def generate(self,**kw):
        self.calls+=1
        if self.failure:raise self.failure
        return Tokens()
class Actions:
    def __init__(self):self.values=[[i*.01+j*.001 for j in range(7)] for i in range(10)]
    def tolist(self):return copy.deepcopy(self.values)
class Backend:
    def __init__(self,text):
        self.actions=Actions();self.calls=0;self.failure=None;self.policy=types.SimpleNamespace(qwen=Qwen(text))
    def __call__(self,request,instruction):
        # Mirror the actual StagedRGBBackend input validation boundary before any mocked model work.
        context.public_context_text(request.plan_context)
        if request.plan_context['active_task']!=request.task.primitive:raise ValueError('action context differs from canonical active task')
        self.calls+=1
        if self.failure:raise self.failure
        return self.actions

CTX={'completed_tasks':['NAV_TO_SOURCE'],'active_task':'PICK','remaining_tasks':['PICK','NAV_TO_TARGET','PLACE'],'current_facts':{}}

def payload():
    return {'protocol_version':'full-episode-rgb-diagnostic-v2','instruction':'frozen release instruction',
        'head_images':['head-history','head-now'],'wrist_images':['wrist-history','wrist-now'],
        'diffusion_seed':17,'predict_transition':True,
        'request':{'request_id':3,'identity':{'mission_id':'mission','active_task_epoch':1},'plan_version':1,
            'observation':{'mission_id':'mission','observation_id':'tick50','time_s':1.,'q':[0.]*6,'dq':[0.]*6,'gripper':.5,
                'image_times_s':[.8,1.,.8,1.],'base_xyyaw':[0.,0.,0.]},
            'task':{'task_id':'pick','primitive':'PICK'},'rtc_context':None,'time_profile':'causal_command_5hz','plan_context':copy.deepcopy(CTX)}}

class Tests(unittest.TestCase):
    def setUp(self):
        torch=types.ModuleType('torch');torch.bfloat16='bf16';torch.inference_mode=nullcontext
        torch.autocast=lambda *a,**kw:nullcontext();torch.manual_seed=lambda seed:None
        backend=types.ModuleType(package+'.full_episode_backend');backend.__package__=package
        backend.__dict__.update(json=json,torch=torch)
        load_functions(BACKEND_PATH,{'validate_transition_text','propose_transition'},backend.__dict__)
        modules={'torch':torch,package+'.full_episode_backend':backend,
            package+'.full_episode_context':context,package+'.full_episode_data':data}
        for name,items in {
            'scripts.serve_joint_trajectory':{'_decode_pair':lambda values,name:tuple(values)},
            package+'.contracts.action':{'ActionIdentity':Record},
            package+'.contracts.observation':{'CurrentObservation':Record},
            package+'.task_memory':{'Task':Record},package+'.rolling_runtime':{'ActionRequest':Record},
        }.items():
            m=types.ModuleType(name);m.__dict__.update(items);modules[name]=m
        self.patcher=patch.dict(sys.modules,modules);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.backend_module=backend
        spec=importlib.util.spec_from_file_location('actual_demo_service_test',SERVICE_PATH)
        self.service_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.service_module)
    def test_legal_continue_and_advance(self):
        for operation,next_task in [('CONTINUE','PICK'),('ADVANCE','NAV_TO_TARGET')]:
            raw=json.dumps({'operation':operation,'next_task':next_task})
            old=self.backend_module.validate_transition_text(raw,CTX)
            self.assertEqual(set(old),{'proposal','completion_evidence','execution_authorized'})
            new=self.backend_module.validate_transition_text(raw,CTX,allow_invalid=True)
            self.assertTrue(new['valid']);self.assertEqual(new['proposal'],old['proposal']);self.assertEqual(new['raw_text'],raw)
    def test_invalid_text_saved_and_old_strict_rejects(self):
        for raw in ['not JSON','[]','{"operation":[],"next_task":"PICK"}',
                    '{"operation":"CONTINUE","next_task":"NAV_TO_SOURCE"}',
                    '{"operation":"ADVANCE","next_task":"PLACE"}']:
            with self.subTest(raw=raw):
                result=self.backend_module.validate_transition_text(raw,CTX,allow_invalid=True)
                self.assertFalse(result['valid']);self.assertIsNone(result['proposal']);self.assertEqual(result['raw_text'],raw)
                with self.assertRaises(ValueError):self.backend_module.validate_transition_text(raw,CTX)
    def service(self,text):
        backend=Backend(text)
        return self.service_module.FullEpisodeService(backend,{'weights_sha256':'frozen-sha'}),backend
    def test_actual_infer_preserves_actions_on_invalid_transition(self):
        raw='{"operation":"ADVANCE","next_task":"PLACE"}'
        service,backend=self.service(raw);original=copy.deepcopy(backend.actions.values);p=payload()
        response=service.infer(p)
        self.assertEqual(response['physical_actions'],original);self.assertEqual(backend.calls,1)
        self.assertFalse(response['transition']['valid']);self.assertIsNone(response['transition']['proposal'])
        self.assertEqual(response['transition']['raw_text'],raw);self.assertEqual(response['transition']['expected_next_task'],'NAV_TO_TARGET')
        self.assertEqual(response['request_id'],3);self.assertEqual(response['identity'],p['request']['identity'])
        self.assertEqual(p['request']['plan_context'],CTX);self.assertEqual(response['weights_sha256'],'frozen-sha')
    def test_bad_request_not_swallowed(self):
        service,backend=self.service('{}');p=payload();p['request']['time_profile']='legacy_future_5hz'
        with self.assertRaises(ValueError):service.infer(p)
        self.assertEqual(backend.calls,0)
    def test_bad_context_not_swallowed_with_transition_disabled(self):
        service,backend=self.service('{}');p=payload();p['predict_transition']=False;p['request']['plan_context']['remaining_tasks']=['PLACE']
        with self.assertRaises(ValueError):service.infer(p)
        self.assertEqual(backend.calls,0)
    def test_low_level_gpu_failure_not_swallowed(self):
        service,backend=self.service('{}');backend.failure=RuntimeError('mock CUDA failure')
        with self.assertRaisesRegex(RuntimeError,'CUDA failure'):service.infer(payload())
        self.assertEqual(backend.policy.qwen.calls,0)
    def test_high_level_gpu_failure_not_swallowed(self):
        service,backend=self.service('{}');backend.policy.qwen.failure=RuntimeError('mock generation CUDA failure')
        with self.assertRaisesRegex(RuntimeError,'generation CUDA failure'):service.infer(payload())
    def test_actual_propose_validates_context_before_generate(self):
        qwen=Qwen('{}');bad=copy.deepcopy(CTX);bad['current_facts']={'GT_grasped':True}
        with self.assertRaises(ValueError):self.backend_module.propose_transition(qwen,[1,2,3,4],instruction='x',context=bad,allow_invalid=True)
        self.assertEqual(qwen.calls,0)
    def test_actual_propose_old_mode_strict(self):
        qwen=Qwen('{"operation":"CONTINUE","next_task":"PLACE"}')
        with self.assertRaises(ValueError):self.backend_module.propose_transition(qwen,[1,2,3,4],instruction='x',context=CTX)

if __name__=='__main__':unittest.main()
