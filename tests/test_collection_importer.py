"""Regression checks for the importer contract; real corpus checks use the CLI."""
import torch
import numpy as np
from conveyor_bench.conveyorvla.collection_importer import pose_matrix, action_route
from conveyor_bench.conveyorvla.geometry_encoder import project_depth
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from test_staged_data_experts import fixture, mutate
from conveyor_bench.conveyorvla.staged_data import RawEpisode


def test_each_held_channel_uses_its_own_parent(tmp_path):
    root=fixture(tmp_path/'ep')
    def parents(rows):
        r=rows[20]['resolved']
        r['source_class']['arm_target']='held_valid';r['source_class']['gripper_target']='held_valid'
        r['arm_target']=rows[18]['resolved']['arm_target']
        r['gripper_target']=rows[19]['resolved']['gripper_target']
        r['parent_command_ids']={'arm_target':'c18','gripper_target':'c19'}
        r['parent_command_id']='wrong_common_parent'
    mutate(root/'control_effective_50hz.jsonl',parents)
    ep=RawEpisode(root)
    assert all(ep.controls[20]['_valid'].values())
    ep.action_view('o10')


def test_pixel_center_offset_is_explicit_and_legacy_unchanged():
    depth=torch.tensor([[[1000.,0.]]]);valid=depth>0
    k=torch.eye(3)[None];t=torch.eye(4)[None]
    old,mask=project_depth(depth,valid,k,t,definition='z_depth',unit_scale=.001)
    new,_=project_depth(depth,valid,k,t,definition='z_depth',unit_scale=.001,pixel_offset=.5)
    torch.testing.assert_close(old[0,0],torch.tensor([0.,0.,1.]))
    torch.testing.assert_close(new[0,0],torch.tensor([.5,.5,1.]))
    assert not mask[0,1] and torch.count_nonzero(new[0,1])==0


def test_measured_state_does_not_use_command_gate():
    obs=CurrentObservation('m','o',0.,(0.,)*6,(0.,)*6,1.0001)
    assert obs.gripper==1.0001
    np.testing.assert_allclose(pose_matrix([1,2,3,1,0,0,0])[:3,3],[1,2,3])
    assert action_route('exec_nav_to_place')=='NAV_TO_TARGET'
