#!/usr/bin/env python3
"""Bounded mechanism/candidate/real-checkpoint numerics; no training or physics claims."""
import argparse,json,sys,time,subprocess,importlib.util
from pathlib import Path
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from conveyor_bench.conveyorvla.formal_checkpoint import source_identity


def mechanisms():
    from conveyor_bench.conveyorvla.task_memory import TaskMemory,Task,Evidence,PlannerEdit
    from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
    from conveyor_bench.conveyorvla.rolling_planner import RollingPlanner
    from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
    from conveyor_bench.conveyorvla.joint_trajectory_runtime import JointSafetyLimits
    from conveyor_bench.conveyorvla.dit import M0DiTActionHead,M0DiTConfig
    from conveyor_bench.conveyorvla.rtc_sampling import sample_rtc
    results=[]
    for mode in ('H0','H1','H2'):
        for rtc in (False,True):
            torch.manual_seed(17)
            head=M0DiTActionHead(M0DiTConfig(action_dim=7,state_dim=13,action_horizon=10,
                vlm_hidden_dim=8,input_embedding_dim=8,hidden_size=12,num_attention_heads=2,
                attention_head_dim=4,num_layers=2,dropout=0.,max_seq_len=64,num_target_vision_tokens=2)).eval()
            memory=TaskMemory('synthetic','test only',(Task('pick','attempt','PICK','cola'),),mode=mode)
            class Norm:
                payload={'normalizer_id':'synthetic_identity'}
                def normalize_action(self,route,x):return np.asarray(x)
            limits=JointSafetyLimits((-2.,)*6,(2.,)*6,(1.,)*6)
            runtime=RollingRuntime(memory,model_id='tiny_random_test',normalizer=Norm(),safety_context_id='synthetic_safety',limits=limits,rtc=rtc)
            class Plant:
                q=np.zeros(6);grip=1.;applies=0
                def hold(self,obs,reason):pass
                def apply(self,target,obs):
                    self.q=self.q+.1*(np.asarray(target[:6])-self.q)
                    self.grip=target[6];self.applies+=1
                    return target
            plant=Plant();contexts=[]
            def planner_backend(request):
                contexts.append(request['context'])
                if mode=='H0':
                    return {'current_task_candidate':asdict(Task('repair','repair-attempt','PICK','changed'))}
                if mode=='H1':
                    return asdict(PlannerEdit(memory.plan_version,memory.active_task_epoch,'o40','CONTINUE'))
                return asdict(PlannerEdit(memory.plan_version,memory.active_task_epoch,'o40','REPAIR_SUFFIX',('changed',),
                    (Task('repair','repair-attempt','PICK','changed'),),True,'observed_target_change'))
            planner=RollingPlanner(memory,planner_backend,model_id='same_synthetic_rule_backend')
            started=time.perf_counter()
            for tick in range(100):
                t=tick*.02
                observation=CurrentObservation('synthetic',f'o{tick}',t,tuple(plant.q),(0.,)*6,plant.grip)
                feedback=(Evidence('changed','synthetic','o40',t,'sensor_estimator','target_changed',True),) if tick==40 else ()
                memory.observe(observation,feedback)
                if tick==40:planner.request(observation,feedback)
                if tick%20==0:
                    request=runtime.prepare_request(observation,expected_delay_s=0.)
                    vl=torch.zeros(1,3,8);state=torch.tensor([observation.mani_state],dtype=torch.float32)
                    noise=torch.randn(1,10,7)
                    if request.rtc_context:
                        c=request.rtc_context
                        action=sample_rtc(head,vl,state,previous=torch.tensor(c['previous'],dtype=torch.float32)[None],
                            weights=torch.tensor(c['weights'],dtype=torch.float32)[None,:,None],noise=noise)
                    else:action=head.sample(vl,state,noise=noise)
                    physical=action[0].numpy();physical[:,6]=(physical[:,6]+1)/2
                    runtime.complete_request(request.request_id,physical,now_s=t)
                runtime.tick(observation,safety=lambda target,obs:(bool(np.isfinite(target).all()),'finite_synthetic_only'),controller=plant)
            if mode=='H0' and any('remaining_plan' in c for c in contexts):raise ValueError('H0 plan leakage')
            results.append({'mode':mode,'rtc':rtc,'control_ticks':100,'applied':plant.applies,
                'requests':runtime.sequence,'epoch':memory.active_task_epoch,'events':len(memory.events),
                'wall_s':time.perf_counter()-started,'synthetic':True,'plant':'first_order_joint_test_only',
                'geometry_transfer_success':None,'strict_full_success':None,'deployment_gate_passed':False})
    return results


def checkpoint_numerics(checkpoint,output):
    from safetensors import safe_open
    from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint
    from conveyor_bench.conveyorvla.dit import M0DiTActionHead,M0DiTConfig
    from conveyor_bench.conveyorvla.joint_trajectory_data import JointTrajectoryNormalizer
    from conveyor_bench.conveyorvla.rtc_sampling import sample_rtc,prefix_weights
    binding=validate_formal_checkpoint(checkpoint,ROOT/'configs/manipulation_navi_v1.json')
    config=binding['config']['action_model']
    cfg={k:config[k] for k in M0DiTConfig.__dataclass_fields__ if k in config}
    cfg.update(action_dim=7,state_dim=13,vlm_hidden_dim=config['cross_attention_dim'])
    head=M0DiTActionHead(M0DiTConfig(**cfg)).eval()
    with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as f:
        weights={k.removeprefix('manipulation_expert.'):f.get_tensor(k) for k in f.keys() if k.startswith('manipulation_expert.')}
    head.load_state_dict(weights,strict=True)
    # Load the original implementation separately for a genuine before/after check.
    old_path=output/'legacy_dit_reference.py'
    old_path.write_bytes(subprocess.check_output(['git','show','bf5d5ab:src/conveyor_bench/conveyorvla/dit.py'],cwd=ROOT))
    spec=importlib.util.spec_from_file_location('legacy_dit_reference',old_path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    old=module.M0DiTActionHead(module.M0DiTConfig(**cfg)).eval();old.load_state_dict(weights,strict=True)
    torch.manual_seed(17)
    vl=torch.randn(1,8,config['cross_attention_dim']);state=torch.zeros(1,13);noise=torch.randn(1,10,7)
    original=old.sample(vl,state,noise=noise);current=head.sample(vl,state,noise=noise)
    zero=sample_rtc(head,vl,state,previous=current,weights=torch.ones(1,10,1),noise=noise,max_guidance_weight=0.)
    torch.testing.assert_close(original,current,rtol=0,atol=0);torch.testing.assert_close(current,zero,rtol=0,atol=0)
    results=[]
    for steps in (4,8):
        start=time.perf_counter()
        with torch.inference_mode():
            guided=sample_rtc(head,vl.clone(),state.clone(),previous=current.clone(),
                weights=prefix_weights(2,8,10)[None,:,None],noise=noise,steps=steps)
        results.append({'steps':steps,'wall_s':time.perf_counter()-start,'finite':bool(torch.isfinite(guided).all()),
            'mean_abs_delta_from_4step_independent':float((guided-current).abs().mean()),
            'parameter_grad_accumulated':any(p.grad is not None for p in head.parameters())})
    normalizer=JointTrajectoryNormalizer.from_path(Path(binding['dataset_root'])/'normalization.json')
    rows=[]
    with (Path(binding['dataset_root'])/'val.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line)
            if r['route']=='PICK':rows.append(r)
            if len(rows)==8:break
    errors=[]
    for row in rows:
        a=np.asarray(row['mani_delta_q_gripper'])
        recovered=normalizer.denormalize_action('PICK',normalizer.normalize_action('PICK',a))
        errors.append(float(np.max(np.abs(a-recovered))))
    return {'weights_sha256':binding['weights_sha256'],'normalizer_id':binding['normalizer_id'],
        'strict_load':True,'old_sampler_bitwise_equal':True,'rtc_zero_gain_bitwise_equal':True,
        'real_data_codec_rows':len(rows),'max_codec_roundtrip_error':max(errors),'rtc':results,
        'vlm_condition':'synthetic_8_token_numerical_probe','physical_capability_evidence':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('mechanism','candidates','checkpoint'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('new experiment output required')
    a.output.mkdir(parents=True)
    if a.mode=='mechanism':result=mechanisms()
    elif a.mode=='checkpoint':
        if not a.checkpoint:p.error('--checkpoint required')
        result=checkpoint_numerics(a.checkpoint,a.output)
    else:
        from scripts.train_staged_vla import smoke
        from conveyor_bench.conveyorvla.staged_experts import StagedConfig
        result=[]
        for path in sorted((ROOT/'configs/vla_staged').glob('*.json')):
            value=json.loads(path.read_text())
            if 'candidate' in value:result.append(smoke(StagedConfig(**value['candidate'])))
    report={'schema':'staged-validation-v2','mode':a.mode,'source_identity':source_identity(ROOT),
        'results':result,'formal_training_started':False,'physical_rollout':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'mode':a.mode,'report':str(a.output/'report.json'),'result':result},allow_nan=False))

if __name__=='__main__':main()
