from types import SimpleNamespace
import math
import numpy as np
import pytest
from conveyor_bench.conveyorvla.physical_events import RelativeGraspEvaluator
from conveyor_bench.conveyorvla.formal_physics import FormalPhysics


def state(t, *, offset=(.02,0,0), height=.1, rotation=0., speed=1.):
    q=(math.cos(rotation/2),0,0,math.sin(rotation/2))
    tcp=(speed*t,0,height)
    x,y,z=offset
    d=(math.cos(rotation)*x-math.sin(rotation)*y, math.sin(rotation)*x+math.cos(rotation)*y,z)
    return SimpleNamespace(timestamp=t, tcp_pose=(*tcp,*q),
                           object_pose=(*(tcp[i]+d[i] for i in range(3)),*q),
                           object_velocity=(speed,0,0,0,0,0))


def run_case(kind):
    evaluator=RelativeGraspEvaluator(0.,lambda *args:None)
    contact={'bilateral_finger_contact':True,'opposing_finger_normals':True,'external_support':False}
    for i in range(101):
        t=i*.02; kwargs={}; c=contact; command=0.
        if kind=='empty': kwargs['offset']=(.2,0,0); c={**contact,'bilateral_finger_contact':False}
        if kind=='table_supported': kwargs['height']=0.;c={**contact,'external_support':True}
        if kind=='bumped': kwargs['height']=.1 if .7<t<1.1 else 0.; c={**contact,'bilateral_finger_contact':False}
        if kind=='tray_lift': c={**contact,'bilateral_finger_contact':False,'external_support':True}
        if kind=='two_finger_support': c={**contact,'opposing_finger_normals':False}
        if kind=='slip': kwargs['offset']=(.02+.03*t,0,0)
        if kind=='opening': command=1.
        if kind=='unknown_contacts': c=None
        evaluator.observe(state(t,**kwargs),command,contacts=c)
    return evaluator


def test_fast_rigid_grasp_is_stable_but_world_speed_is_a_separate_safety_failure():
    e=run_case('positive')
    assert e.latest['geometry_hold_proxy'] and e.latest['contact_grasp_verified']
    assert not e.latest['world_speed_safe']


@pytest.mark.parametrize('kind',['empty','table_supported','bumped','tray_lift','two_finger_support','slip','opening'])
def test_contact_grasp_rejects_explicit_negative_controls(kind):
    e=run_case(kind)
    assert e.latest['contact_grasp_verified'] is False
    assert e.ever_contact_grasp is False
    if kind=='tray_lift':
        assert e.latest['geometry_hold_proxy']  # Known ambiguity, never hide this proxy false positive.


def test_missing_contacts_stay_unknown_even_for_a_stable_geometric_hold():
    e=run_case('unknown_contacts')
    assert e.latest['geometry_hold_proxy']
    assert e.latest['contact_grasp_verified'] is None
    assert not e.ever_contact_grasp


def test_end_effector_frame_removes_common_rigid_rotation():
    e=RelativeGraspEvaluator(0.,lambda *args:None)
    for i in range(101):
        e.observe(state(i*.02,rotation=i*.04),0.,contacts=None)
    assert e.latest['relative_position_drift_m'] < 1e-12
    assert e.latest['relative_rotation_drift_rad'] < 1e-6
    assert e.latest['geometry_hold_proxy']


def test_gaps_cannot_be_counted_as_continuous_holding():
    e=RelativeGraspEvaluator(0.,lambda *args:None)
    for i in range(10): e.observe(state(i*.2),0.)
    assert not e.latest['window_complete']


def test_evaluator_outputs_cannot_activate_or_disable_legacy_assistance(monkeypatch):
    class Sim:
        def __init__(self):self.attaches=0;self.s=state(0.,height=0.,speed=0.)
        def read(self):return self.s
        def create_verified_grasp_constraint(self):self.attaches+=1;return {'active':True}
    sim=Sim();proxy=FormalPhysics(sim,'source_assisted',lambda *args:None);proxy.arm()
    monkeypatch.setattr('conveyor_bench.conveyorvla.formal_physics.measured_named_joint_state',lambda s:SimpleNamespace(gripper_open_fraction=.77))
    proxy.closed_command_seen=True
    proxy.evaluator.pick=True
    proxy.observe(sim.s)
    assert sim.attaches==0
    proxy.evaluator.pick=False
    proxy.evaluator.observe=lambda **kw:None
    monkeypatch.setattr('conveyor_bench.conveyorvla.formal_physics.measured_named_joint_state',lambda s:SimpleNamespace(gripper_open_fraction=.2))
    sim.s=state(.02,height=.1,speed=0.)
    proxy.observe(sim.s)
    assert sim.attaches==1 and proxy.evaluator.pick is False
