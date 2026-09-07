from types import SimpleNamespace
import copy
import numpy as np
import pytest
from scripts.serve_conditioned_pick_rtc import validate_context,PROTOCOL
from conveyor_bench.conveyorvla.joint_trajectory_model import ConveyorVLAJointTrajectoryPolicy


def request():
    identity={'episode_id':'e','active_task_id':'p','active_task_epoch':2,
        'weights_sha256':'weights','normalization_sha256':'norm'}
    return {'protocol_version':PROTOCOL,'request_id':'r','episode_id':'e','sequence_id':3,
        'instruction':'transfer','head_images':[],'wrist_images':[],'joint_position':[.1]*6,
        'joint_velocity':[0.]*6,'gripper_open_fraction':1.00001,'observation_id':'o',
        'query_time_s':1.4,'image_times_s':[1.2,1.4,1.2,1.4],'active_task_id':'p',
        'active_task_epoch':2,'time_profile':'legacy_future_5hz','rtc_context':{
            'identity':identity,'source_action_id':'a','source_query_time_s':1.,'source_sequence_id':2,
            'previous_absolute':[[.4]*6+[.5]]*10,'target_apply_times_s':(1.4+np.arange(10)*.2).tolist(),
            'available_previous':[True]*8+[False]*2,'weights':[1.]*8+[0.]*2}}


def test_rtc_service_reanchors_absolute_previous_and_masks_unknown_tail():
    session=SimpleNamespace(normalization_sha256='norm',normalizer=SimpleNamespace(normalize_action=lambda route,a:a))
    p=request();c=validate_context(p,session,'weights')
    np.testing.assert_allclose(np.asarray(c['previous'])[:,:6],.3)
    np.testing.assert_allclose(np.asarray(c['previous'])[:,6],.5)
    for mutation in (lambda p:p['rtc_context']['identity'].update(active_task_epoch=1),
                     lambda p:p['rtc_context']['target_apply_times_s'].__setitem__(0,1.3),
                     lambda p:p['rtc_context']['weights'].__setitem__(9,1.),
                     lambda p:p.update(evaluator_truth=True),
                     lambda p:p.update(image_times_s=[1.4]*4)):
        bad=copy.deepcopy(p);mutation(bad)
        with pytest.raises(ValueError):validate_context(bad,session,'weights')
    p['rtc_context']=None
    assert validate_context(p,session,'weights') is None


def test_regular_predict_does_not_reference_undefined_rtc_contexts():
    decision=SimpleNamespace(valid=False)
    fake=SimpleNamespace(predict_routes=lambda examples:[decision])
    result=ConveyorVLAJointTrajectoryPolicy.predict(fake,[{}])
    assert len(result)==1 and result[0].decision is decision
