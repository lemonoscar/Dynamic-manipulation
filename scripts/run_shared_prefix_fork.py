#!/usr/bin/env python3
"""A/B/C shared-prefix diagnostic. Reuse one archived, hash-bound first prediction."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import run_conditioned_pick as base
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute


def closure_index(commands):
    return next((i for i,c in enumerate(commands) if c.gripper_open_fraction<=.5),None)


def factory(options, fork):
    parent=base.pipeline_type(options)
    class Fork(parent):
        def _execute_policy(self, options, summary, health, payload, state):
            saved=json.loads(fork.first_plan.read_text())
            if saved['weights_sha256']!=health['weights_sha256'] or saved['source_episode']!=options.source_episode.name:
                raise ValueError('first-plan identity differs')
            old=[base.runner._command(c) for c in saved['response']['chunk']['commands']]
            if len(old)!=10:raise ValueError('first plan must contain 10 commands')
            summary.update(execution_mode='shared-prefix-fork-v1',branch=fork.branch,
                           first_plan_sha256=sha256(fork.first_plan),window_s=1.6,extension=None)
            self._record('shared_first_plan',saved)
            self._query_count=0;last_action=None
            start=float(state.timestamp)
            def execute(command,plan_id,index):
                nonlocal last_action
                action=self.action_adapter.manipulation(command,route=JointTrajectoryRoute.PICK,sequence_id=self._query_count)
                before=self.simulation.read()
                self._record('executed_plan_point',{'plan_id':plan_id,'plan_index':index,
                    'time_s':before.timestamp,'command':asdict(command),
                    'state_before':base.runner.waypoint_runner._state_snapshot(before)})
                for _ in range(10):self._physical_step(action,route=JointTrajectoryRoute.PICK,command_index=command.index)
                last_action=action
            for i,c in enumerate(saved.get('prefix_commands',[])):
                execute(base.runner._command(c),'shared-context-prefix',i)
            start=float(self.simulation.read().timestamp)
            summary['shared_plan_query_state']=base.runner.waypoint_runner._state_snapshot(self.simulation.read())
            summary['context_prefix_points']=len(saved.get('prefix_commands',[]))
            for i,c in enumerate(old[:2]):execute(c,'shared-first',i)
            fork_state=self.simulation.read()
            pair=self.frames.pair_after(None)
            if pair is None:raise ValueError('fork camera history missing')
            summary['fork_state']={'state':base.runner.waypoint_runner._state_snapshot(fork_state),
                'last_action':base.runner.waypoint_runner._jsonable(asdict(last_action)),
                'camera_history':[asdict(f) for f in pair],
                'articulation_position_target':self.simulation.simulation._adapter.robot.data.joint_pos_target[0].tolist(),
                'controller_hidden_state':'not_completely_observable',
                'contact_coverage':'unknown'}
            self._record('fork_state',summary['fork_state'])
            fork_start=float(fork_state.timestamp)
            for block in range(4):
                selected=old[2+block*2:4+block*2];plan_id='shared-first'
                if fork.branch in ('B','C'):
                    before=self.simulation.read();payload,qstate=self._next_request()
                    if self.simulation.read().step_index!=before.step_index:
                        raise ValueError('query unexpectedly advanced physics')
                    payload['protocol_version']=base.PROTOCOL
                    self._record('model_request',payload)
                    self._record('query_camera_evidence',qstate.metadata.get('camera_capture_report',{}))
                    wall=time.time();result=self.client.infer(payload)
                    after=self.simulation.read()
                    if after.step_index!=before.step_index:raise ValueError('inference advanced physics')
                    if result['checkpoint_id']!=health['checkpoint_id']:raise ValueError('checkpoint changed')
                    self._query_count+=1
                    proposed=[base.runner._command(c) for c in result['chunk']['commands']]
                    k=closure_index(proposed)
                    self._record('fork_prediction',{'branch':fork.branch,'plan_id':f'new-{block}',
                        'time_s':qstate.timestamp,'fork_relative_s':qstate.timestamp-fork_start,
                        'wall_inference_s':time.time()-wall,'query_state':base.runner.waypoint_runner._state_snapshot(qstate),
                        'result':result,'limited_closure_index':k,
                        'limited_closure_time_s':None if k is None else qstate.timestamp+k*.2,
                        'old_remaining_index':2+block*2,'replacement_applied':fork.branch=='C',
                        'cancel_reason':'scheduled_prefix_replacement' if fork.branch=='C' else None})
                    if fork.branch=='C':selected=proposed[:2];plan_id=f'new-{block}'
                for j,c in enumerate(selected):execute(c,plan_id,2+block*2+j if plan_id=='shared-first' else j)
            summary['window_physics_evidence']=self.physics.evidence()
            summary['fork_start_time_s']=fork_start
            summary['policy_start_time_s']=start
            summary['model_queries_after_fork']=self._query_count
            summary['duration_after_fork_s']=self.simulation.read().timestamp-fork_start
            summary['grasp_success_rate_not_estimated']=True
    return Fork


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch',choices=('A','B','C'),required=True)
    p.add_argument('--first-plan',type=Path,required=True)
    fork,rest=p.parse_known_args()
    original=base.pipeline_type
    def patched(options):
        base.pipeline_type=original
        return factory(options,fork)
    base.pipeline_type=patched
    sys.argv=[sys.argv[0],*rest]
    return base.main()

if __name__=='__main__':raise SystemExit(main())
