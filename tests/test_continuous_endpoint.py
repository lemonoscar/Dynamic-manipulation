import math
import numpy as np
import pytest
from conveyor_bench.conveyorvla.continuous_endpoint import (
    SweptDiskEvidence, continuous_endpoint_candidate, classify_degenerate_path,
)
from conveyor_bench.conveyorvla.waypoint_execution import PCTPlan


def evidence(cells=None, yaw=0.):
    return SweptDiskEvidence(np.ones((100,100),bool) if cells is None else cells,
                             .05,(0.,0.,yaw),.3,'a'*64,'b'*64)


def test_continuous_segment_preserves_coarse_bias_and_is_not_deployment_approval():
    p=PCTPlan(((1.,1.),(2.,2.)),(2.,2.,0.,0.),math.sqrt(.02))
    result=continuous_endpoint_candidate(p,(2.1,2.1,0.,.3),evidence())
    assert result['candidate_path_world'][-1]==[2.1,2.1]
    assert result['coarse_endpoint_B']==[2.,2.,0.,0.]
    assert result['coarse_snap_distance_m'] > .1
    assert result['geometry_certificate']['valid']
    assert not result['deployment_approved']


def test_free_endpoints_do_not_hide_obstacle_between_them_or_beside_centerline():
    cells=np.ones((100,100),bool)
    cells[100-1-24,30]=False  # x=1.525,y=1.225, within footprint, off centerline.
    assert not evidence(cells).check((1.,1.),(2.,1.))['valid']


def test_turn_sweep_support_hole_and_out_of_bounds_fail_closed():
    cells=np.ones((100,100),bool);cells[100-1-24,20]=False
    assert not evidence(cells).check((1.,1.),(1.,1.))['valid']
    assert not evidence().check((.1,1.),(.2,1.))['valid']


def test_rotated_map_and_vertical_connection():
    assert evidence(yaw=math.pi/2).check((-1.,1.),(-1.,2.))['valid']
    p=PCTPlan(((1.,1.),(2.,2.)),(2.,2.,1.,0.),0.)
    assert not continuous_endpoint_candidate(p,(2.,2.,0.,0.),evidence())['geometry_certificate']['valid']


@pytest.mark.parametrize('current,reason',[
    ((0.,0.,0.),'local_goal_reached'),
    ((0.,0.,1.),'validated_in_place_turn_required'),
    ((.2,0.,0.),'reconnect_from_measured_pose_required'),
])
def test_three_degenerate_path_semantics(current,reason):
    assert classify_degenerate_path(current,(0.,0.,0.))==reason
