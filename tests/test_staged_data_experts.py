"""All generated episodes in this file are synthetic contract fixtures."""
import json
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import pytest
import torch
from conveyor_bench.conveyorvla.staged_data import RawEpisode,validate_family_splits
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer
from conveyor_bench.conveyorvla.geometry_encoder import project_depth,GeometryEncoder
from conveyor_bench.conveyorvla.navigation_safety import ReachConfig,ReachMonitor,OnlineNavigationSafety
from conveyor_bench.conveyorvla.continuous_endpoint import SweptDiskEvidence
from conveyor_bench.conveyorvla.task_memory import transfer_skeleton


def fixture(root, *, primitive='PICK', uuid='synthetic', family='family'):
    root.mkdir()
    manifest={'schema':'raw-control-v2','action_contract':'joint-command-v2','episode_uuid':uuid,
        'task_family_id':family,'source_commit':'a'*40,'resolved_config_sha256':'b'*64,
        'assets_sha256':'c'*64,'assistance_profile':'synthetic_no_physics','synthetic':True,
        'observation_contract':'named-q6-dq6-gripper-rgb-v2','joint_unit':'rad','gripper_unit':'open_fraction',
        'joint_names':[f'arm_joint{i}' for i in range(6,0,-1)]}
    (root/'manifest.json').write_text(json.dumps(manifest))
    observations=[];controls=[]
    for i in range(131):
        t=round(i*.02,8)
        observations.append({'mission_id':uuid,'observation_id':f'o{i}','time_s':t,'q':[.01*j for j in range(6,0,-1)],
            'dq':[0]*6,'gripper':1.,'base_xyyaw':[0,0,0],'images':['h0','h1','w0','w1'],
            'image_times_s':[max(0,t-.2),t,max(0,t-.2),t],'primitive':primitive,
            'active_task_id':'task','active_task_epoch':0,'action_query':i==10,
            'nav_goal_world_xyyaw':[t,0,0],'nav_goal_source_class':'issued'})
        if i<130:
            controls.append({'schema':'raw-control-v2','identity':{k:manifest[k] for k in
                ('episode_uuid','task_family_id','source_commit','resolved_config_sha256')}|{'reset_generation':0},
                'clock':{'control_tick':i,'sim_time_s':t,'wall_monotonic_ns':1000+i,'apply_sequence':0,'physics_advanced':True},
                'observation_ref':f'o{i}','post_observation_ref':f'o{i+1}',
                'requested':{},'resolved':{'command_id':f'c{i}','arm_target':[t+j*.01 for j in range(6,0,-1)],
                    'gripper_target':[1 if t<1 else 0],'base_twist':[0,0,0],
                    'source_class':dict(arm_target='issued',gripper_target='issued',base_twist='issued'),
                    'parent_command_id':None,'actuator_mode':'synthetic_position','limiting_report':{}},'intervention_refs':[]})
    def write(name,rows):(root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('observations.jsonl',observations);write('control_effective_50hz.jsonl',controls)
    (root/'task.json').write_text(json.dumps({'original_instruction':'synthetic transfer','initial_plan':[asdict(t) for t in transfer_skeleton()]}))
    write('task_events.jsonl',[{'observation_id':'o10','feedback':[],'label':{'parent_plan_version':0,
        'observed_active_task_epoch':0,'observation_id':'o10','operation':'CONTINUE'}}])
    return root


def mutate(path,fn):
    rows=[json.loads(r) for r in path.read_text().splitlines()];fn(rows)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_50hz_time_named_codec_and_task_view(tmp_path):
    ep=RawEpisode(fixture(tmp_path/'ep'))
    a=ep.action_view('o10',profile='legacy_future_5hz')
    assert a['first_target_offset_s']==.2 and a['target_times_s'][0]==.4
    np.testing.assert_allclose(a['actions'][0][:6],[.4]*6)
    b=ep.action_view('o10',profile='causal_command_25hz')
    assert np.shape(b['actions'])==(50,7)
    np.testing.assert_allclose(b['target_times_s'][-1],2.16)
    assert b['target_times_s'][0]==b['first_apply_time_s']==.2
    assert len(ep.task_view())==1 and ep.task_view()[0]['synthetic']


@pytest.mark.parametrize('mutation,pattern',[
    (lambda r:r.pop(30),'missing 50Hz'),
    (lambda r:r[30]['identity'].update(reset_generation=1),'reset'),
    (lambda r:r[30]['clock'].update(wall_monotonic_ns=0),'nonmonotonic'),
    (lambda r:r[30].update(post_observation_ref='o29'),'timing'),
])
def test_raw_invalid_clock_and_reset_rejected(tmp_path,mutation,pattern):
    root=fixture(tmp_path/'ep');mutate(root/'control_effective_50hz.jsonl',mutation)
    with pytest.raises(ValueError,match=pattern):RawEpisode(root)


@pytest.mark.parametrize('source',['stale','unknown','held_valid'])
def test_bad_provenance_quarantines_entire_chunk(tmp_path,source):
    root=fixture(tmp_path/'ep')
    mutate(root/'control_effective_50hz.jsonl',lambda r:r[20]['resolved']['source_class'].update(arm_target=source))
    with pytest.raises(ValueError,match='quarantined'):RawEpisode(root).action_view('o10')


def test_same_tick_only_final_apply_and_truth_does_not_enter_view(tmp_path):
    root=fixture(tmp_path/'ep')
    def add(rows):
        extra=json.loads(json.dumps(rows[20]));extra['clock']['physics_advanced']=False
        rows[20]['clock']['apply_sequence']=1;rows.insert(20,extra)
    mutate(root/'control_effective_50hz.jsonl',add)
    before=RawEpisode(root).action_view('o10')
    (root/'evaluator_truth.jsonl').write_text('{"carrying":true}\n')
    after=RawEpisode(root).action_view('o10')
    assert before==after
    mutate(root/'control_effective_50hz.jsonl',lambda r:r[20]['clock'].update(physics_advanced=True))
    with pytest.raises(ValueError,match='only final'):RawEpisode(root)


def test_split_leakage_and_synthetic_normalizer_rejected(tmp_path):
    a=RawEpisode(fixture(tmp_path/'a',uuid='a'));b=RawEpisode(fixture(tmp_path/'b',uuid='b'))
    with pytest.raises(ValueError,match='family'):validate_family_splits([a,b],{'a':'train','b':'test'})
    row=a.action_view('o10');row['split']='train'
    with pytest.raises(ValueError,match='nonsynthetic'):StagedNormalizer.fit([row])


def test_depth_definition_masks_and_calibration():
    depth=torch.ones(1,2,2);valid=torch.ones_like(depth,dtype=torch.bool)
    k=torch.eye(3)[None];t=torch.eye(4)[None]
    z,m=project_depth(depth,valid,k,t,definition='z_depth')
    rays,_=project_depth(depth,valid,k,t,definition='ray_range')
    torch.testing.assert_close(z[0,-1],torch.ones(3))
    torch.testing.assert_close(torch.linalg.vector_norm(rays,dim=-1),torch.ones(1,4))
    depth[0,0,0]=float('nan')
    z,m=project_depth(depth,valid,k,t,definition='z_depth')
    assert not m[0,0] and (z[0,0]==0).all()
    encoder=GeometryEncoder(8,2)
    tokens,mask=encoder(z,torch.zeros_like(m))
    assert not mask.any() and (tokens==0).all()
    t[0,0,0]=2
    with pytest.raises(ValueError,match='rigid'):project_depth(depth,valid,k,t,definition='z_depth')


def test_reach_gap_yaw_recheck_and_stability():
    m=ReachMonitor(ReachConfig())
    assert m.update(.13,0,(0,0,0),0)=='TRANSLATE'
    assert m.update(.11,.3,(0,0,0),.2)=='VALIDATED_TURN_REQUIRED'
    assert m.update(.13,0,(0,0,0),.4)=='TRANSLATE'
    assert m.update(.11,.1,(0,0,0),.6)=='SETTLE'
    assert m.update(.11,.1,(0,0,0),.8)=='REACHED'
    assert m.update(.11,.1,(.2,0,0),1.)=='SETTLE'


def test_online_current_envelope_and_stop_coverage():
    evidence=SweptDiskEvidence(np.ones((100,100),bool),.1,(0,0,0),.3,'a'*64,'b'*64)
    monitor=OnlineNavigationSafety(evidence,'safety',10.,.1,1.,.5)
    assert monitor.check((5,5,0),(0,0,0),(.3,0,0),1.)['valid']
    assert not monitor.check((.2,.2,0),(0,0,0),(.3,0,0),1.)['valid']
    assert monitor.check((5,5,0),(0,0,0),(1,0,0),1.)['reason']=='speed_outside_stop_model'
    assert not monitor.check((5,5,0),(0,0,0),(.3,0,0),11.)['valid']


@pytest.mark.parametrize('path',sorted((Path(__file__).parents[1]/'configs/vla_staged').glob('*.json')))
def test_independent_candidate_gradient_smoke(path):
    payload=json.loads(path.read_text())
    if 'candidate' not in payload:return
    from scripts.train_staged_vla import smoke
    from conveyor_bench.conveyorvla.staged_experts import StagedConfig
    result=smoke(StagedConfig(**payload['candidate']))
    assert result['finite_sample'] and result['synthetic']


def test_25hz_changes_only_mani_and_unsampled_bad_tick_quarantines(tmp_path):
    root=fixture(tmp_path/'nav',primitive='NAV_TO_SOURCE')
    row=RawEpisode(root).action_view('o10',profile='causal_command_25hz')
    assert len(row['actions'])==10 and row['time_profile']=='causal_command_5hz'
    root=fixture(tmp_path/'mani')
    mutate(root/'control_effective_50hz.jsonl',lambda r:r[21]['resolved']['source_class'].update(arm_target='stale'))
    with pytest.raises(ValueError,match='inside real interval'):RawEpisode(root).action_view('o10')


def test_explicit_paired_gripper_calibration(tmp_path):
    root=fixture(tmp_path/'pair')
    manifest=json.loads((root/'manifest.json').read_text());manifest.update(gripper_unit='rad',
        gripper_calibration={'closed_joint_positions':[0,0],'open_joint_positions':[.04,-.04],'pair_fraction_tolerance':.01})
    (root/'manifest.json').write_text(json.dumps(manifest))
    def convert(rows):
        for r in rows:
            g=r['resolved']['gripper_target'][0];r['resolved']['gripper_target']=[.04*g,-.04*g]
    mutate(root/'control_effective_50hz.jsonl',convert)
    row=RawEpisode(root).action_view('o10')
    assert row['actions'][0][-1]==1 and row['actions'][-1][-1]==0


def test_prepare_cli_creates_only_marked_synthetic_view(tmp_path):
    from scripts.prepare_staged_data import main
    root=fixture(tmp_path/'episode');split=tmp_path/'split.json'
    split.write_text(json.dumps({'synthetic':'train'}));output=tmp_path/'view'
    assert main(['--episodes',str(root),'--split-manifest',str(split),'--output',str(output)])==0
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['synthetic'] and manifest['action_rows']==1 and manifest['task_rows']==1
    assert manifest['normalizer_id'] is None
    with pytest.raises(ValueError,match='overwrite'):
        main(['--episodes',str(root),'--split-manifest',str(split),'--output',str(output)])
