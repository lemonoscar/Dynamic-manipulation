import unittest
from types import SimpleNamespace
from scripts.prepare_full_episode_rgb import native_prefix, task_context, transition_prompt


def episode():
    raw={'primitive':'PICK','active_task_id':'pick','active_task_epoch':1}
    controls=[];observations={}
    for tick in range(30):
        observations[str(tick)]=(None,dict(raw))
        controls.append({'clock':{'control_tick':tick,'sim_time_s':tick*.02},
            'observation_ref':str(tick),'_valid':{'arm_target':True,'gripper_target':True,'base_twist':True},
            'intervention_refs':[]})
    return SimpleNamespace(controls=controls,observations=observations),raw


class PrefixTests(unittest.TestCase):
    def test_task_boundary_masks_whole_bin_and_suffix(self):
        ep,raw=episode();ep.observations['15'][1]['active_task_epoch']=2
        mask,reason,_=native_prefix(ep,raw,0)
        self.assertEqual(mask,[True]+[False]*9)
        self.assertEqual(reason,'task_identity_or_epoch_boundary')

    def test_unknown_or_out_of_range_not_relaxed(self):
        for mutation in ('unknown','range','intervention'):
            ep,raw=episode()
            if mutation=='unknown':ep.controls[7]['_valid']['gripper_target']=False
            elif mutation=='range':ep.controls[7]['_action_exclusion_reason']='effective_gripper_outside_calibrated_range'
            else:ep.controls[7]['intervention_refs']=['non_bc']
            self.assertFalse(any(native_prefix(ep,raw,0)[0]))

    def test_tail_is_masked_not_held(self):
        ep,raw=episode();mask,reason,_=native_prefix(ep,raw,0)
        self.assertEqual(mask,[True]*3+[False]*7)
        self.assertEqual(reason,'missing_native_application_interval')

    def test_prompt_never_contains_label_or_truth(self):
        context=task_context('PICK',['NAV_TO_SOURCE'])
        self.assertEqual(context['remaining_tasks'],['PICK','NAV_TO_TARGET','PLACE'])
        self.assertEqual(context['current_facts'],{})
        row={'input':{'original_instruction':'Move coke','task_context':context},
            'label':{'operation':'ADVANCE','secret':'future-success-secret'},
            'label_evidence':{'object_pose':'GT-secret'}}
        prompt=transition_prompt(row)
        self.assertNotIn('future-success-secret',prompt);self.assertNotIn('GT-secret',prompt)


if __name__=='__main__':unittest.main()
