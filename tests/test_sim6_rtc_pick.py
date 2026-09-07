import copy
from types import SimpleNamespace
import pytest
from scripts.run_sim6_rtc_pick import prior_context,request_payload,PROTOCOL


def test_prior_uses_actual_remaining_interval_without_terminal_hold():
    payload={'episode_id':'episode','active_task_id':'PICK','active_task_epoch':0,'query_time_s':4.8}
    commands=[{'joint_position':[i]*6,'gripper_open_fraction':i/10} for i in range(10)]
    prior={'query_time_s':4.4,'commands':commands,'request_id':'r0','sequence_id':0}
    health={'weights_sha256':'w','normalization_sha256':'n'}
    c=prior_context(payload,prior,health)
    assert c['previous_absolute'][:8]==[[i]*6+[i/10] for i in range(2,10)]
    assert c['available_previous']==[True]*8+[False]*2
    assert all(0 < w < 1 for w in c['weights'][:8])
    assert c['weights'][8:]==[0.,0.]
    assert c['weights'][0]>c['weights'][7]
    assert c['target_apply_times_s']==pytest.approx([4.8+i*.2 for i in range(10)])
    assert c['identity']['normalization_sha256']=='n'
    with pytest.raises(ValueError,match='unaligned'):
        prior_context({**payload,'query_time_s':4.81},prior,health)


def test_wire_preserves_true_images_and_excludes_initializer_truth():
    p={'episode_id':'e','head_images':['h0','h1'],'wrist_images':['w0','w1']}
    state=SimpleNamespace(timestamp=4.4,step_index=220,object_pose='evaluator_only')
    wire=request_payload(p,state,[4.2,4.4,4.2,4.4])
    assert wire['image_times_s']==pytest.approx([4.2,4.4,4.2,4.4])
    assert wire['head_images']==p['head_images']
    assert wire['protocol_version']==PROTOCOL
    assert 'object_pose' not in wire


def test_camera_evidence_rejects_stale_or_relabelled_times():
    from scripts.run_sim6_rtc_pick import verified_image_times
    def state(tick):
        return SimpleNamespace(step_index=tick,timestamp=tick*.02,metadata={'camera_capture_report':{
            'accepted':True,'capture_step_index':tick,'render_step_index':tick,'capture_timestamp':tick*.02,
            'available_camera_keys':['front','wrist']}})
    states={210:state(210),220:state(220)};pair=[SimpleNamespace(step_index=i) for i in (210,220)]
    assert verified_image_times(pair,states,states[220])==pytest.approx([4.2,4.4,4.2,4.4])
    with pytest.raises(ValueError,match='older'):
        verified_image_times(pair,states,state(221))
    states[210].metadata['camera_capture_report']['render_step_index']=200
    with pytest.raises(ValueError,match='unverified'):
        verified_image_times(pair,states,states[220])


def test_response_cannot_change_epoch_duration_or_hide_saturation():
    from scripts.run_sim6_rtc_pick import validate_response
    request={'request_id':'r','observation_id':'o','sequence_id':1,'active_task_id':'PICK','active_task_epoch':2,'query_time_s':4.4,'time_profile':'legacy_future_5hz','rtc_context':{}}
    health={'checkpoint_id':'old','normalization_sha256':'old-normalizer'}
    response={k:v for k,v in request.items() if k!='rtc_context'}
    response.update(health,rtc_applied=True,chunk={'commands':[{'index':i,'duration_s':.2,'base_velocity':[0.,0.,0.],
        'joint_position':[0.]*6,'gripper_open_fraction':.5} for i in range(10)],'position_saturation_count':1,'rate_saturation_count':0,'gripper_saturation_count':0})
    report=validate_response(request,response,health)
    assert report['rate']==pytest.approx(1/70) and not report['gate_passed']
    bad=copy.deepcopy(response);bad['active_task_epoch']=3
    with pytest.raises(ValueError,match='identity'):
        validate_response(request,bad,health)
    bad=copy.deepcopy(response);bad['chunk']['commands'][0]['duration_s']=.04
    with pytest.raises(ValueError,match='interval'):
        validate_response(request,bad,health)
