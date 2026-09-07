from pathlib import Path
import importlib.util
from types import SimpleNamespace

def load(name):
    p=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py';s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_none_command_does_not_establish_invalid_cache():
    m=load('audit_train_command_sources')
    assert m.classify([1],None,applied_evidence=False)[0]=='unknown'
    assert m.classify([1],None,applied_evidence=False,old_cache=[1])[0]=='invalid_cache'
    assert m.classify([1],None,applied_evidence=False,old_cache=[1],ever_current_command=True)[0]=='unknown'

def test_explicit_needs_application_and_matching_target():
    m=load('audit_train_command_sources')
    assert m.classify([1],[1],applied_evidence=False)[0]=='unknown'
    assert m.classify([1],[1],applied_evidence=True)[0]=='valid_new'
    assert m.classify([1],[2],applied_evidence=True)[0]=='unknown'
    assert m.classify([1],None,applied_evidence=False,hold_evidence=True)[0]=='valid_hold'

def test_gripper_interpolated_joint_target_precedes_close_intent():
    m=load('audit_train_command_sources')
    assert m.explicit({'gripper_command':'close','metadata':{'gripper_joint_positions':[.039,.039]}},'gripper')==[.039,.039]

def test_closure_index_reports_absence_and_first_intent_only():
    m=load('run_shared_prefix_fork')
    assert m.closure_index([SimpleNamespace(gripper_open_fraction=1.)]*10) is None
    assert m.closure_index([SimpleNamespace(gripper_open_fraction=v) for v in [1,.7,.5,0]])==2

def test_triangle_union_covers_rectangle_but_does_not_fill_holes():
    import numpy as np
    from conveyor_bench.conveyorvla.nav_geometry_audit import uncovered_rectangle
    tris=np.array([[[0,0],[1,0],[0,1]],[[1,0],[1,1],[0,1]]],float)
    assert uncovered_rectangle([0,0],[1,1],tris)<1e-12
    assert abs(uncovered_rectangle([0,0],[1,1],tris[:1])-.5)<1e-12
    assert uncovered_rectangle([0,0],[1,1],[np.zeros((3,2))])==1

def test_clipping_reconstruction_matches_actual_decoder():
    from dataclasses import asdict
    from conveyor_bench.conveyorvla.formal_metrics import LIMITS
    from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointTrajectoryExecutor
    m=load('summarize_shared_prefix_forks');q=[0.]*6;raw=[[10.,10.,10.,10.,10.,10.,-1.]]*10
    response={'physical_relative_action':raw,'chunk':asdict(DirectJointTrajectoryExecutor(LIMITS).prepare(q,raw))}
    counts=m.clipping(response,q)
    assert len(counts)==10
    assert sum(c[2] for c in counts)==10

def test_connected_sweep_requires_complete_ground_and_obstacle_clearance():
    import numpy as np
    from conveyor_bench.conveyorvla.nav_geometry_audit import connected_sweep_certificate
    mesh=SimpleNamespace(triangles=np.array([[[-3,-3,0],[3,-3,0],[-3,3,0]],[[3,-3,0],[3,3,0],[-3,3,0]]]),face_normals=np.array([[0,0,1],[0,0,1]]),face_adjacency=np.array([[0,1]]))
    assert connected_sweep_certificate(mesh,[[0,0],[.2,0]],.5,0,[])['valid']
    obstacle={'path':'box','minimum':[-.1,-.1,.1],'maximum':[.1,.1,.2]}
    assert not connected_sweep_certificate(mesh,[[0,0],[.2,0]],.5,0,[obstacle])['valid']
    mesh.face_adjacency=np.empty((0,2),dtype=int)
    result=connected_sweep_certificate(mesh,[[0,0],[.2,0]],.5,0,[])
    assert not result['valid'] and result['uncovered_support_area_m2']>0
