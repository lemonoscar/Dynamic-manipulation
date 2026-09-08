"""CPU guards; run directly from this directory without Isaac or model weights."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_service',HERE/'serve_full_episode_diagnostic.py')
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
tree=ast.parse((HERE/'run_full_episode_diagnostic.py').read_text())
camera=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='camera_times')
ns={'math':math};exec(compile(ast.Module(body=[camera],type_ignores=[]),'camera_times','exec'),ns)


class ContractGuards(unittest.TestCase):
    def packet(self):
        return {'protocol_version':service.PROTOCOL,'instruction':'transfer','head_images':['a','b'],
            'wrist_images':['c','d'],'diffusion_seed':17,'predict_transition':False,
            'request':{'request_id':0,'identity':{},'plan_version':0,'task':{},'rtc_context':None,
                'time_profile':'causal_command_5hz','plan_context':{},
                'observation':{'mission_id':'m','observation_id':'o','time_s':1.,'q':[0]*6,'dq':[0]*6,
                    'gripper':.5,'base_xyyaw':[0]*3,'image_times_s':[.8,1.,.8,1.]}}}

    def test_protocol_and_time_semantics(self):
        service.validate_payload(self.packet())
        p=self.packet();p['request']['time_profile']='legacy_future_5hz'
        with self.assertRaises(ValueError):service.validate_payload(p)
        p=self.packet();p['request']['observation']['image_times_s'][0]=float('nan')
        with self.assertRaises(ValueError):service.validate_payload(p)

    def test_truth_depth_and_undeclared_inputs_rejected(self):
        for key in ('depth','evaluator_truth','teacher_phase'):
            p=self.packet();p['request']['observation'][key]={}
            with self.assertRaises(ValueError):service.validate_payload(p)

    def test_real_capture_identity(self):
        states={}
        for tick,t in ((40,.8),(50,1.)):
            states[tick]=SimpleNamespace(step_index=tick,timestamp=t,metadata={'camera_capture_report':{
                'accepted':True,'capture_step_index':tick,'render_step_index':tick,
                'capture_timestamp':t,'available_camera_keys':['front','wrist']}})
        pair=[SimpleNamespace(step_index=40),SimpleNamespace(step_index=50)]
        self.assertEqual(ns['camera_times'](pair,states,states[50]),[.8,1.,.8,1.])
        states[40].metadata['camera_capture_report']['render_step_index']=30
        with self.assertRaises(ValueError):ns['camera_times'](pair,states,states[50])


if __name__=='__main__':unittest.main()
