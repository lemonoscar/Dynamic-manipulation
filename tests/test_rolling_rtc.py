"""Synthetic mechanism checks, never physical task evidence."""
from dataclasses import replace, asdict
import numpy as np
import pytest
import torch
from conveyor_bench.conveyorvla.task_memory import TaskMemory, Task, Evidence, PlannerEdit, transfer_skeleton
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from conveyor_bench.conveyorvla.contracts.action import ActionIdentity, TIME_PROFILES, reanchor_mani, decode_mani, nav_goal_world
from conveyor_bench.conveyorvla.action_queue import AbsoluteActionQueue, ActionPlan
from conveyor_bench.conveyorvla.rtc_sampling import prefix_weights, sample_rtc, vjp_velocity, prefix_conditioned_loss
from conveyor_bench.conveyorvla.dit import M0DiTConfig, M0DiTActionHead


def obs(t=0., name='o', **kw):
    return CurrentObservation('m', name, t, (0.,)*6, (0.,)*6, 1., **kw)


def evidence(predicate, value=True, t=0., name='o', eid='e', **kw):
    return Evidence(eid, 'm', name, t, 'sensor_estimator', predicate, value, **kw)


def edit(memory, operation, refs=(), **kw):
    return PlannerEdit(memory.plan_version, memory.active_task_epoch,
        next(reversed(memory.observations)), operation, refs, **kw)


def test_event_is_not_current_carrying_or_model_claim():
    m=TaskMemory('m','transfer',transfer_skeleton())
    m.observe(obs(), [evidence('source_reached')])
    m.apply(edit(m,'ADVANCE',('e',)))
    m.last_model_output={'claim':'PICK completed'}
    with pytest.raises(ValueError,match='unsupported'):
        m.apply(edit(m,'ADVANCE'))
    m.observe(obs(.2,'o2'), [evidence('carrying',t=.2,name='o2',eid='e2')])
    m.apply(edit(m,'ADVANCE',('e2',)))
    m.observe(obs(.4,'o3'), [evidence('carrying',False,t=.4,name='o3',eid='e3')])
    assert m.last_completed_task_id == 'task-1'
    assert m.fact_value('carrying',.4) is False
    assert len(m.events)==2
    retry=Task('retry-pick','retry-attempt','PICK','cola',completion_conditions=('carrying',))
    m.apply(edit(m,'REPAIR_SUFFIX',('e3',),new_suffix=(retry,),changes_active_task_semantics=True))
    assert m.active_task == retry and m.last_completed_task_id=='task-1'
    assert len(m.events)==3


def test_future_repair_preserves_epoch_and_cas_is_atomic():
    m=TaskMemory('m','transfer',transfer_skeleton())
    m.observe(obs(),[evidence('path_blocked')])
    stale=edit(m,'CONTINUE')
    m.apply(edit(m,'REPAIR_SUFFIX',('e',),new_suffix=(Task('future','future-a','PICK','cola'),)))
    assert m.plan_version==1 and m.active_task_epoch==0
    with pytest.raises(ValueError,match='compare-and-swap'):
        m.apply(stale)
    assert m.plan_version==1
    with pytest.raises(ValueError,match='fresh'):
        m.apply(edit(m,'REPAIR_SUFFIX',('e',),new_suffix=(Task('task-0','attempt-0','PICK','other'),),changes_active_task_semantics=True))
    assert m.plan_version==1 and m.active_task_epoch==0


@pytest.mark.parametrize('source',['evaluator','teacher','oracle'])
def test_truth_evidence_rejected(source):
    with pytest.raises(ValueError):
        replace(evidence('carrying'),source=source)


def test_expiry_future_and_h0_allowlist():
    m=TaskMemory('m','transfer',transfer_skeleton(),mode='H0')
    m.observe(obs(),[evidence('carrying',validity_until_s=.1)])
    assert m.fact_value('carrying',.2) is None
    assert set(m.planner_context(obs()))=={'original_instruction','observation_id','time_s','feedback'}
    record=asdict(obs());record['evaluator_truth']={'carrying':True}
    assert CurrentObservation.from_record(record)==obs()
    with pytest.raises(ValueError,match='future'):
        m.observe(obs(.2,'o2'),[evidence('carrying',t=.3,name='o2',eid='e2')])


def identity(epoch=0):
    return ActionIdentity('m',0,epoch,'task','model','normalizer','safety')


def plan(request=0, t=0., epoch=0, value=.1):
    targets=tuple((value,)*6+(1.,) for _ in range(10))
    return ActionPlan(request,identity(epoch),0,f'o{request}',t,(0.,)*6,
        'legacy_future_5hz',t,targets,targets,t+2.)


def test_commit_late_arrival_and_epoch_cancellation():
    q=AbsoluteActionQueue(identity())
    q.accept(plan(),now_s=0.,plan_version=0)
    assert q.next_for_control_time(0.).action_index==0
    q.commit_through(.4,.3)
    old=q.next_for_control_time(.5)
    assert old.request_id==0
    q.accept(plan(1,.4,value=.8),now_s=.65,plan_version=0)
    assert q.next_for_control_time(.68).request_id==0
    c=q.next_for_control_time(.72)
    assert c.request_id==1 and c.action_index==1 and c.target[0]==.8
    with pytest.raises(ValueError,match='out_of_order'):
        q.accept(plan(),now_s=.8,plan_version=0)
    q.cancel(identity(1))
    with pytest.raises(ValueError,match='stale'):
        q.accept(plan(2,.8),now_s=.9,plan_version=1)
    assert q.next_for_control_time(.9).target is None


def test_queue_coverage_exhaustion_no_replay_and_expiry():
    q=AbsoluteActionQueue(identity())
    with pytest.raises(ValueError,match='coverage'):
        q.commit_through(0.,.2)
    q.accept(plan(),now_s=.45,plan_version=0)
    assert q.next_for_control_time(.46).action_index==2
    with pytest.raises(ValueError,match='advance'):
        q.next_for_control_time(.46)
    assert q.next_for_control_time(2.).target is None
    with pytest.raises(ValueError,match='expired'):
        q.accept(plan(1),now_s=2.,plan_version=0)


def test_anchor_and_label_apply_clock_are_distinct():
    a=np.arange(70).reshape(10,7)/100
    old=np.ones(6);new=np.ones(6)*2
    np.testing.assert_allclose(decode_mani(a,old),decode_mani(reanchor_mani(a,old,new),new))
    p=TIME_PROFILES['legacy_future_5hz']
    assert p.target_times(1.)[0]==1.2 and p.apply_times(1.)[0]==1.
    np.testing.assert_allclose(nav_goal_world((1,0,0),(2,3,np.pi/2)),(2,4,np.pi/2))
    with pytest.raises(ValueError,match='10D'):
        decode_mani(np.zeros((10,10)),old)


def tiny(horizon=4):
    return M0DiTActionHead(M0DiTConfig(action_dim=7,state_dim=13,action_horizon=horizon,
        vlm_hidden_dim=8,input_embedding_dim=8,hidden_size=12,num_attention_heads=2,
        attention_head_dim=4,num_layers=2,dropout=0.,max_seq_len=64,num_target_vision_tokens=2))


def test_vjp_matches_finite_difference_and_not_hard_copy():
    torch.manual_seed(10)
    x=torch.randn(1,4,7,dtype=torch.float64)
    previous=torch.randn_like(x);weights=torch.ones_like(x)
    def clean(x,t):return .3*x + .07*x.square()
    velocity, correction=vjp_velocity(clean,x,.5,previous,weights)
    residual=(previous-clean(x,.5)).detach()
    direction=torch.randn_like(x);h=1e-6
    fd=((clean(x+h*direction,.5)-clean(x-h*direction,.5))*residual).sum()/(2*h)
    torch.testing.assert_close((correction*direction).sum(),fd,rtol=1e-6,atol=1e-7)
    assert correction.abs().sum()>0
    assert not torch.equal(clean(x,.5),previous)


def test_rtc_zero_gain_old_path_exact_and_outer_inference_mode():
    torch.manual_seed(2)
    head=tiny().eval();vl=torch.randn(1,3,8);state=torch.randn(1,13);noise=torch.randn(1,4,7)
    expected=head.sample(vl,state,noise=noise)
    with torch.inference_mode():
        actual=sample_rtc(head,vl.clone(),state.clone(),previous=torch.zeros_like(noise),
            weights=torch.ones(1,4,1),noise=noise,max_guidance_weight=0.)
        guided=sample_rtc(head,vl.clone(),state.clone(),previous=torch.zeros_like(noise),
            weights=torch.ones(1,4,1),noise=noise,max_guidance_weight=1.)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.isfinite(guided).all() and not torch.equal(guided,expected)
    assert all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize('horizon',[10,50])
def test_token_time_clean_prefix_seen_and_gradients(horizon):
    torch.manual_seed(3)
    head=tiny(horizon);vl=torch.randn(2,3,8);state=torch.randn(2,13);actions=torch.randn(2,horizon,7)
    captured={};original=head._predict_clean
    def record(v,s,x,t,m):
        captured.update(x=x.detach().clone(),t=t.clone())
        return original(v,s,x,t,m)
    head._predict_clean=record
    loss=prefix_conditioned_loss(head,vl,actions,state,prefix_lengths=[2,0],time=torch.tensor([.4,.3]))
    loss.backward()
    torch.testing.assert_close(captured['x'][0,:2],actions[0,:2])
    assert (captured['t'][0,:2]==1).all()
    assert torch.isfinite(loss) and head.action_encoder.layer1.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='nonempty suffix'):
        prefix_conditioned_loss(head,vl,actions,state,prefix_lengths=[horizon,0])


def test_prefix_weights_reference_example():
    torch.testing.assert_close(prefix_weights(2,6,10,'linear'),torch.tensor([1,1,.8,.6,.4,.2,0,0,0,0]))


def test_finish_requires_current_placement_and_not_model_claim():
    task=Task('place','attempt','PLACE','cola',completion_conditions=('placed',))
    m=TaskMemory('m','transfer',(task,))
    m.observe(obs(),[evidence('placed',None)])
    with pytest.raises(ValueError,match='unsupported'):m.apply(edit(m,'FINISH',('e',)))
    m.observe(obs(.2,'o2'),[evidence('placed',t=.2,name='o2',eid='e2')])
    m.apply(edit(m,'FINISH',('e2',)))
    assert m.finished and m.active_task is None and m.last_completed_task_id=='place'


def test_runtime_invalid_fact_cancels_actions_and_controller_apply_is_evidence():
    from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
    from conveyor_bench.conveyorvla.joint_trajectory_runtime import JointSafetyLimits
    class Norm:
        payload={'normalizer_id':'test'}
        def normalize_action(self,r,a):return a
    task=Task('place','a','PLACE','cola',preconditions=('carrying',))
    m=TaskMemory('m','transfer',(task,));m.observe(obs(),[evidence('carrying')])
    runtime=RollingRuntime(m,model_id='test',normalizer=Norm(),safety_context_id='test',limits=JointSafetyLimits((-2.,)*6,(2.,)*6,(1.,)*6))
    request=runtime.prepare_request(obs());runtime.complete_request(request.request_id,[(.1,)*6+(1.,)]*10,now_s=0.)
    class Controller:
        reason=None
        def apply(self,target,observation):return target
        def hold(self,observation,reason):self.reason=reason
    controller=Controller()
    runtime.tick(obs(),safety=lambda t,o:(True,'test_only'),controller=controller)
    assert len(runtime.queue.history)==1
    m.observe(obs(.2,'o2'),[evidence('carrying',False,t=.2,name='o2',eid='e2')])
    runtime.tick(obs(.2,'o2'),safety=lambda t,o:(True,'test_only'),controller=controller)
    assert controller.reason=='current_task_precondition_invalid' and m.active_task_epoch==1
    assert len(runtime.queue.history)==1


def test_frozen_backend_uses_only_canonical_task_current_images_and_bound_normalizer():
    from conveyor_bench.conveyorvla.rolling_runtime import FrozenActionBackend,ActionRequest
    from conveyor_bench.conveyorvla.joint_trajectory import canonical_solution
    class Norm:
        def normalize_mani_state(self,v):return v
        def denormalize_action(self,r,v):return v
    class Policy:
        def eval(self):pass
        def predict_actions(self,examples,decisions,**kw):
            assert set(examples[0])=={'video','lang','mani_state'}
            assert decisions[0].assistant_prefix==canonical_solution('PICK')
            assert examples[0]['video']==(('h0','h1'),('w0','w1'))
            return [((0.,)*6+(1.,),)*10]
    observation=obs(.2,images=('h0','h1','w0','w1'),image_times_s=(0.,.2,0.,.2))
    request=ActionRequest(0,identity(),0,observation,Task('p','a','PICK','cola','destination'),None)
    backend=FrozenActionBackend(Policy(),Norm())
    assert len(backend(request,'original task'))==10
    with pytest.raises(ValueError,match='cannot represent'):
        backend(replace(request,task=Task('q','b','PICK','other','destination')),'task')


def test_episode_driver_routes_through_actual_memory_queue_and_safety():
    from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
    from conveyor_bench.conveyorvla.rolling_planner import RollingPlanner
    from conveyor_bench.conveyorvla.rolling_episode import run_rolling_episode
    from conveyor_bench.conveyorvla.joint_trajectory_runtime import JointSafetyLimits
    class Norm:
        payload={'normalizer_id':'test'}
        def normalize_action(self,r,a):return a
    class Plant:
        t=.2;applies=0;holds=0
        def observe(self):return obs(self.t,f'o{self.t}')
        def apply(self,target,o):self.t+=.02;self.applies+=1;return target
        def hold(self,o,reason):self.t+=.02;self.holds+=1
    m=TaskMemory('m','transfer',(Task('p','a','PICK','cola'),))
    runtime=RollingRuntime(m,model_id='test',normalizer=Norm(),safety_context_id='test',
        limits=JointSafetyLimits((-2.,)*6,(2.,)*6,(1.,)*6),rtc=True)
    planner=RollingPlanner(m,lambda r:asdict(PlannerEdit(r['context']['plan_version'],
        r['context']['active_task_epoch'],r['context']['observation_id'],'CONTINUE')),model_id='test_rule')
    plant=Plant()
    result=run_rolling_episode(runtime,planner,lambda req,instruction:[(0.,)*6+(1.,)]*10,
        observer=plant.observe,feedback_estimator=lambda o:(),controller=plant,
        safety=lambda target,o:(True,'synthetic_only'),max_control_ticks=25)
    assert plant.applies==25 and plant.holds==1
    assert not result['failures'] and result['geometry_transfer_success'] is None


def test_isaac_binding_only_reads_sensors_and_uses_existing_action_adapter():
    from types import SimpleNamespace
    from conveyor_bench.conveyorvla.rolling_isaac import IsaacRollingAdapter
    from conveyor_bench.conveyorvla.joint_trajectory_system import IsaacJointActionAdapter
    class Sim:
        time=0.;tick=0;actions=[]
        def read(self):
            return SimpleNamespace(timestamp=self.time,step_index=self.tick,
                metadata={'joint_names':[f'arm_joint{i}' for i in range(1,9)]},
                joint_positions=(0.,)*6+(.04,.04),joint_velocities=(0.,)*8,
                robot_root_pose=(0.,0.,.4,1.,0.,0.,0.))
        def apply(self,action):self.actions.append(action)
        def step(self,render):self.time+=.02;self.tick+=1
    m=TaskMemory('m','transfer',(Task('p','a','PICK','cola'),));sim=Sim();records=[]
    adapter=IsaacRollingAdapter(sim,IsaacJointActionAdapter(lambda **kw:SimpleNamespace(**kw)),m,
        camera_reader=lambda s:(s.timestamp,np.zeros((2,2,3),np.uint8),np.zeros((2,2,3),np.uint8)),
        navigation_executor=None,local_map=lambda r:None,record=records.append,
        effective_target_reader=lambda:{'mission_id':'m','verified':True,'joint_position':(0.,)*6,
            'gripper_open_fraction':.25})
    for _ in range(10):
        o=adapter.observe();adapter.hold(o,'warmup')
        assert sim.actions[-1].metadata['gripper_joint_positions']==(.01,.01)
    o=adapter.observe();assert len(o.images)==4
    np.testing.assert_allclose(o.image_times_s,(0.,.2,0.,.2))
    target=(.01,)*6+(.5,);assert adapter.apply(target,o)==target
    assert sim.actions[-1].base_velocity==(0.,0.,0.)
    assert sim.actions[-1].metadata['gripper_open_fraction_requested']==.5
    assert records[-1]['provenance']=='controller_target_applied_not_motor_torque'
