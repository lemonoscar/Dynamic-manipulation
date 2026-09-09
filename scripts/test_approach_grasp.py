"""CPU regression of the actual two-task runner and its independent score."""
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import test_timer_only_diagnostic as runner
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.physical_events import RelativeGraspEvaluator
from conveyor_bench.conveyorvla.task_memory import Task, TaskMemory
from conveyor_bench.conveyorvla.full_episode_context import public_context_text

tree = ast.parse(runner.PATH.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
    'initial_task_context', 'predicts_transition', 'approach_grasp_result', 'advance_model_claim'}]
ns = {'asdict': asdict}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(runner.PATH), 'exec'), ns)


class ApproachGraspTests(unittest.TestCase):
    def test_two_tasks_and_terminal_pick(self):
        context = ns['initial_task_context']('approach_grasp')
        frozen = json.loads((runner.PATH.parents[1] / 'configs/approach_grasp/task_context.json').read_text())
        self.assertEqual(context, frozen)
        public_context_text(context)
        tasks = tuple(Task(str(i), str(i), name, 'cola') for i, name in enumerate(context['remaining_tasks']))
        memory = TaskMemory('m', 'Approach and grasp', tasks, mode='H1')
        self.assertTrue(ns['predicts_transition']('approach_grasp', context))
        context = ns['advance_model_claim'](memory, context, NS(observation_id='o', time_s=1.))
        self.assertEqual(context['remaining_tasks'], ['PICK'])
        self.assertFalse(ns['predicts_transition']('approach_grasp', context))
        self.assertFalse(memory.events[-1]['completion_verified'])
        with self.assertRaises(ValueError):
            ns['advance_model_claim'](memory, context, NS(observation_id='o2', time_s=2.))

    def test_old_full_transfer_context_preserved(self):
        context = ns['initial_task_context']('train_seed_full')
        self.assertEqual(context['remaining_tasks'], ['NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE'])
        for i in range(4):
            current = dict(context, remaining_tasks=context['remaining_tasks'][i:])
            self.assertEqual(ns['predicts_transition']('train_seed_full', current), i < 3)

    def test_real_runner_requires_continuous_measured_closed_lift(self):
        # Execute the actual _physical_step method with CPU simulation doubles.
        runner.ns['JointTrajectoryRoute'] = JointTrajectoryRoute
        runner.ns['measured_named_joint_state'] = lambda state: NS(gripper_open_fraction=state.measured)
        c = runner.instance()
        c.clock_start_s = 0.
        c.approach_grasp_evaluator = RelativeGraspEvaluator(0., lambda *args: None)
        c.approach_grasp_first_success_s = None
        c.physics = NS(command_fraction=0.)
        raw_read = c.raw_sim.read
        measured = 1.
        lift = .05
        c.raw_sim.read = lambda: NS(**vars(raw_read()),
            object_pose=(0., 0., lift, 1., 0., 0., 0.),
            tcp_pose=(0., 0., lift, 1., 0., 0., 0.),
            object_velocity=(0.,)*6, measured=measured)
        def steps(n, route=JointTrajectoryRoute.PICK):
            for _ in range(n):
                c._physical_step(None, route=route, command_index=None)
        steps(51)  # Closed command alone cannot pass with an open measured gripper.
        self.assertFalse(c.approach_grasp_evaluator.ever_proxy_hold)
        measured = 0.
        lift = .001
        steps(51)  # Touching the can without lifting cannot pass.
        self.assertFalse(c.approach_grasp_evaluator.ever_proxy_hold)
        lift = .05
        steps(50)
        self.assertFalse(c.approach_grasp_evaluator.ever_proxy_hold)
        steps(1)
        self.assertTrue(c.approach_grasp_evaluator.ever_proxy_hold)
        self.assertAlmostEqual(c.approach_grasp_first_success_s, 3.06)
        self.assertIsNone(c.approach_grasp_evaluator.evidence()['ever_contact_grasp_verified'])

    def test_missing_false_or_assisted_evidence_never_passes(self):
        summary = dict(scoring_valid=True, status='complete', clock_elapsed_s=60.,
            physics_evidence=dict(profile='no_grasp_assist', pick_verified=True,
                grasp_constraint_created=False, mid_episode_object_resets=0))
        grasp = {'ever_geometry_hold_proxy': True}
        result = ns['approach_grasp_result'](summary, grasp, 10.)
        self.assertTrue(result['success'])
        self.assertIsNone(result['strict_contact_success'])
        self.assertFalse(ns['approach_grasp_result'](summary, {}, None)['success'])
        for key, value in [('scoring_valid', False), ('clock_elapsed_s', 59.),
                           ('grasp_constraint_created', True), ('pick_verified', False),
                           ('mid_episode_object_resets', 1)]:
            changed = deepcopy(summary)
            (changed if key in changed else changed['physics_evidence'])[key] = value
            self.assertFalse(ns['approach_grasp_result'](changed, grasp, 10.)['success'])


if __name__ == '__main__':
    unittest.main()
