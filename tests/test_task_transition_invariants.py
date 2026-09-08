"""Task entry versus continuous invariants at complete-transfer handoffs."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import numpy as np
import pytest
from conveyor_bench.conveyorvla.task_memory import TaskMemory, Task, Evidence, PlannerEdit, transfer_skeleton
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
from conveyor_bench.conveyorvla.joint_trajectory_runtime import JointSafetyLimits


def observe(memory, t, **facts):
    obs=CurrentObservation('m',str(t),t,(0.,)*6,(0.,)*6,.9)
    evidence=[Evidence(f'{t}:{k}','m',str(t),t,'sensor_estimator',k,v) for k,v in facts.items()]
    memory.observe(obs,evidence)
    return obs


def edit(memory, operation, predicate):
    return PlannerEdit(memory.plan_version,memory.active_task_epoch,next(reversed(memory.observations)),
        operation,(memory.current_facts[predicate].evidence_id,))


def runtime(memory):
    class Norm:
        payload={'normalizer_id':'n'}
        def normalize_action(self,route,actions):return actions
    return RollingRuntime(memory,model_id='model',normalizer=Norm(),safety_context_id='safe',
        limits=JointSafetyLimits((-2.,)*6,(2.,)*6,(1.,)*6))


def test_place_requires_carrying_entry_but_release_can_settle_and_finish():
    memory=TaskMemory('m','transfer',transfer_skeleton()[2:]);rt=runtime(memory)
    o=observe(memory,.2,carrying=False,target_reached=True)
    with pytest.raises(ValueError,match='next task preconditions'):
        memory.apply(edit(memory,'ADVANCE','target_reached'))
    o=observe(memory,.4,carrying=True,target_reached=True)
    memory.apply(edit(memory,'ADVANCE','target_reached'))
    request=rt.prepare_request(o);epoch=memory.active_task_epoch
    observe(memory,.6,carrying=False,placed=None)
    rt.synchronize_task()
    assert memory.active_task_epoch==epoch and memory.active_task.primitive=='PLACE'
    rt.complete_request(request.request_id,[(0.,)*6+(1.,)]*10,now_s=.6)
    with pytest.raises(ValueError,match='completion unsupported'):
        memory.apply(edit(memory,'FINISH','placed'))
    observe(memory,.8,carrying=False,placed=True)
    memory.apply(edit(memory,'FINISH','placed'))
    assert memory.finished and memory.events[-1]['task']['primitive']=='PLACE'


def test_direct_place_initialization_still_requires_entry_evidence():
    memory=TaskMemory('m','transfer',(transfer_skeleton()[-1],));rt=runtime(memory)
    o=observe(memory,.2,carrying=None)
    with pytest.raises(ValueError,match='precondition_invalid'):rt.prepare_request(o)
    o=observe(memory,.4,carrying=True);rt.prepare_request(o);epoch=memory.active_task_epoch
    o=observe(memory,.6,carrying=False);rt.prepare_request(o)
    assert memory.active_task_epoch==epoch


def test_pick_completion_is_history_and_nav_still_requires_current_carrying():
    memory=TaskMemory('m','transfer',transfer_skeleton()[1:]);rt=runtime(memory)
    o=observe(memory,.2,carrying=True)
    request=rt.prepare_request(o);rt.complete_request(request.request_id,[(0.,)*6+(.25,)]*10,now_s=.2)
    memory.apply(edit(memory,'ADVANCE','carrying'));rt.synchronize_task()
    assert rt.queue.next_for_control_time(.2).target is None
    rt.prepare_request(o);epoch=memory.active_task_epoch
    o=observe(memory,.4,carrying=False);rt.synchronize_task()
    assert memory.active_task_epoch==epoch+1
    assert memory.events[0]['task']['primitive']=='PICK' and memory.last_completed_task_id=='task-1'
    assert memory.fact_value('carrying',.4) is False
    with pytest.raises(ValueError,match='precondition_invalid'):rt.prepare_request(o)


def test_pick_to_nav_wait_and_navigation_keep_effective_gripper_not_measured_aperture():
    from conveyor_bench.conveyorvla.rolling_isaac import IsaacRollingAdapter
    from conveyor_bench.conveyorvla.joint_trajectory_system import IsaacJointActionAdapter
    class Sim:
        time=.2;tick=10
        def __init__(self):self.actions=[]
        def read(self):
            return SimpleNamespace(timestamp=self.time,step_index=self.tick,
                metadata={'joint_names':[f'arm_joint{i}' for i in range(1,9)],'body_velocity':(0.,0.,0.)},
                joint_positions=(0.,)*6+(.036,.036),joint_velocities=(0.,)*8,
                robot_root_pose=(0.,0.,.4,1.,0.,0.,0.))
        def apply(self,action):self.actions.append(action)
        def step(self,render):self.time+=.02;self.tick+=1
    class Nav:
        require_online_safety=True
        def begin(self,*args,**kwargs):pass
        def command(self,*args,**kwargs):
            return SimpleNamespace(base_velocity=(.2,0.,0.),requires_requery=False,reached_local_goal=False,reason=None,trace={})
        def reset(self):pass
    memory=TaskMemory('m','transfer',transfer_skeleton()[1:]);rt=runtime(memory)
    o=observe(memory,.2,carrying=True);memory.apply(edit(memory,'ADVANCE','carrying'));rt.synchronize_task()
    sim=Sim();records=[]
    adapter=IsaacRollingAdapter(sim,IsaacJointActionAdapter(lambda **kwargs:SimpleNamespace(**kwargs)),memory,
        camera_reader=lambda state:None,navigation_executor=Nav(),local_map=lambda route:None,record=records.append,
        effective_target_reader=lambda:{'mission_id':'m','verified':True,'joint_position':(.1,)*6,'gripper_open_fraction':.25})
    adapter.hold(o,'waiting_for_first_NAV_proposal')
    proposal={'identity':rt.identity(),'query_time_s':.2,'query_base_xyyaw':(0.,0.,0.),
        'reference_query_body':[(.1,0.,0.)]*10,'valid_until_s':2.2}
    adapter.navigation(proposal,adapter.observe())
    assert len(sim.actions)==2 and sim.actions[-1].base_velocity==(.2,0.,0.), (sim.actions,records)
    for action in sim.actions:
        assert action.arm_joint_positions==(.1,)*6
        assert action.metadata['gripper_joint_positions']==(.01,.01)
    assert memory.last_completed_task_id=='task-1'


def test_legacy_tasks_preserve_invariants_and_h0_semantic_comparison_includes_them():
    from conveyor_bench.conveyorvla.rolling_planner import RollingPlanner
    original=Task('place','attempt','PLACE','cola','destination',('carrying',),('placed',))
    assert original.invariants==('carrying',)
    memory=TaskMemory('m','transfer',(original,),mode='H0');o=observe(memory,.2,carrying=True)
    candidate=replace(original,task_id='new-place',attempt_id='new-attempt',invariants=())
    planner=RollingPlanner(memory,lambda request:{'current_task_candidate':asdict(candidate)},model_id='planner')
    assert planner.request(o) is True
    assert memory.active_task.invariants==() and memory.active_task_epoch==1


def test_request_plan_context_only_has_committed_history_and_current_plan():
    memory=TaskMemory('m','transfer',transfer_skeleton());rt=runtime(memory)
    o=observe(memory,.2,source_reached=True)
    before=rt.prepare_request(o).plan_context
    assert before=={'completed_tasks':[],'active_task':'NAV_TO_SOURCE',
        'remaining_tasks':['NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'],'current_facts':{}}
    memory.apply(edit(memory,'ADVANCE','source_reached'))
    after=rt.prepare_request(o).plan_context
    assert after=={'completed_tasks':['NAV_TO_SOURCE'],'active_task':'PICK',
        'remaining_tasks':['PICK','NAV_TO_TARGET','PLACE'],'current_facts':{}}
    assert before['completed_tasks']==[] and before['active_task']=='NAV_TO_SOURCE'
    assert 'last_model_output' not in after and 'source_reached' not in after['current_facts']
