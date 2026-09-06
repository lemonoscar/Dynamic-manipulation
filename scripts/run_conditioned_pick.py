#!/usr/bin/env python3
"""Fixed checkpoint PICK diagnosis: shared initial state, 10- vs 2-point feedback.

Source initialization is evaluator-only. Requests contain RGB and measured
joints; route/canonical-prefix conditioning is explicit, not autonomous routing.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import run_joint_trajectory_rollout as runner
from scripts.replay_sampled_joint_targets import initialize_source_state
from conveyor_bench.conveyorvla.execution_consistency import sampled_phase
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,source_identity,write_json
from conveyor_bench.conveyorvla.formal_physics import FormalPhysics
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.joint_trajectory_system import measured_named_joint_state
from conveyor_bench.isaac.grasp_contact_probe import IsaacGraspContactProbe
PROTOCOL='conveyorvla-conditioned-pick-diagnostic/v1'
WEIGHTS_SHA='d86360e96d97f45467281ca77a006eba85c085c737e4156170efbf8a58a351b9'


def pipeline_type(options):
    class ConditionedPickPipeline(runner.JointTrajectoryRolloutPipeline):
        def _measured_hold(self, source):
            if self.physics is None or not self.physics.armed:
                return super()._measured_hold(source)
            joints = measured_named_joint_state(self.simulation.read())
            command = runner.DirectJointCommand(index=0, joint_position=joints.joint_position,
                                               gripper_open_fraction=joints.gripper_open_fraction)
            return self.action_adapter.hold(command, route=JointTrajectoryRoute.PICK,
                                            sequence_id=self._query_count, source=source)

        def run_episode(self):
            period=options.execute_points
            self.episode_dir.mkdir(parents=True,exist_ok=True)
            self._trace_stream=(self.episode_dir/'trace.jsonl').open('x')
            self.client=runner.JointTrajectoryHTTPClient(options.endpoint,timeout_s=120.)
            base_simulation=self.simulation;probe=None
            summary={'schema':'conditioned-pick-diagnostic-v1','status':'running','success':False,
                     'execution_mode':'fixed_PICK_canonical_subtask_diagnostic','pure_physics_success':False,
                     'state_trace':['PICK'],'full_task_success':None,'autonomous_routing':False,
                     'source_episode':options.source_episode.name,'execute_points':period,
                     'initial_state_kind':'source_PICK',
                     'simulation_budget_s':options.simulation_seconds,'source_identity':source_identity(ROOT)}
            started=time.time()
            try:
                health=self.client.health()
                if health.get('protocol_version')!=PROTOCOL or health.get('weights_sha256')!=WEIGHTS_SHA or not health.get('strict_load'):
                    raise ValueError('diagnostic service checkpoint/protocol mismatch')
                summary['model_identity']=health
                source=json.loads((options.source_episode/'summary.json').read_text())
                phase=sampled_phase([json.loads(x) for x in (options.source_episode/'samples.jsonl').open()],
                                    [json.loads(x) for x in (options.source_episode/'frames.jsonl').open()])
                observation=phase[0][2]
                self.episode_seed=source['seed']
                self.config=replace(self.config,video=replace(self.config.video,fps=5.))
                write_json(self.episode_dir/'resolved_config.json',runner.waypoint_runner._jsonable(asdict(self.config)))
                self._start_video();self._prepare_episode()
                if options.record_contacts:
                    probe=IsaacGraspContactProbe(base_simulation,self._record)
                    base_simulation.read_grasp_contacts=probe.read
                summary['initialization']=initialize_source_state(base_simulation,observation)
                base_simulation._runtime.sim.forward()
                summary['source_state_render_sync']=base_simulation._render_without_physics(
                    valid_state_step=int(base_simulation.read().step_index),
                    reason='conditioned_pick_source_state_sync', force=True)
                robot=base_simulation._adapter.robot
                summary['articulation_limits']={'joint_names':list(robot.joint_names),
                    'position':robot.root_physx_view.get_dof_limits()[0].tolist(),
                    'max_velocity':robot.root_physx_view.get_dof_max_velocities()[0].tolist()}
                self.physics=FormalPhysics(base_simulation,'no_grasp_assist',self._record)
                self.simulation=self.physics;self.physics.arm()
                self.physics.previous_fraction=self.physics.command_fraction=measured_named_joint_state(self.simulation.read()).gripper_open_fraction
                self.frames=runner.TemporalJPEGBuffer(separation_steps=10,jpeg_quality=self.jpeg_quality)
                self._camera_states.clear();self._last_query_camera_step=None
                # Build a real t-.2,t history after initialization; no old navigation image is reused.
                history_start=float(self.simulation.read().timestamp)
                payload,state=self._next_request()
                summary['first_query_state']=runner.waypoint_runner._state_snapshot(state)
                summary['first_query_camera_report']=state.metadata.get('camera_capture_report')
                summary['initial_history_hold_s']=float(state.timestamp)-history_start
                control_start=self._control_steps
                while self._control_steps-control_start < round(options.simulation_seconds/.02):
                    if self._query_count:
                        payload,state=self._next_request()
                    payload['protocol_version']=PROTOCOL
                    self._record('query_camera_evidence', state.metadata.get('camera_capture_report', {}))
                    self._record('model_request',payload)
                    result=self.client.infer(payload)
                    if result.get('checkpoint_id')!=health['checkpoint_id']:
                        raise ValueError('model response checkpoint changed')
                    self._record('conditioned_model_response',result)
                    commands=[runner._command(c) for c in result['chunk']['commands']]
                    self._query_count+=1
                    for command in commands[:period]:
                        action=self.action_adapter.manipulation(command,route=JointTrajectoryRoute.PICK,sequence_id=self._query_count)
                        for _ in range(10):
                            if self._control_steps-control_start >= round(options.simulation_seconds/.02):break
                            self._physical_step(action,route=JointTrajectoryRoute.PICK,command_index=command.index)
                summary.update(status='complete',success=True,failure_reason=None)
            except Exception as error:
                summary.update(status='failed',failure_reason=f'{type(error).__name__}:{error}',traceback=traceback.format_exc())
            finally:
                summary.update(wall_s=time.time()-started,model_queries=self._query_count,
                               success_semantics='diagnostic_control_budget_completed_not_grasp_success',
                               physics_evidence=None if self.physics is None else self.physics.evidence())
                try:summary['video']=self._close_video(summary['status'])
                except Exception as error:summary['video_error']=str(error)
                if probe:
                    probe.close();del base_simulation.read_grasp_contacts
                write_json(self.episode_dir/'summary.json',runner.waypoint_runner._jsonable(summary))
                self._trace_stream.close();self._trace_stream=None
                if self.close_simulation_on_exit:base_simulation.close()
            return summary
    return ConditionedPickPipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-episode',type=Path,required=True)
    parser.add_argument('--validation-manifest',type=Path,required=True)
    parser.add_argument('--endpoint',default='http://127.0.0.1:18086')
    parser.add_argument('--execute-points',type=int,choices=(1,2,10),required=True,
                        help='one fresh Isaac process per feedback condition; stage reuse is not a validated pair')
    parser.add_argument('--simulation-seconds',type=float,default=12.)
    parser.add_argument('--record-contacts',action='store_true')
    options,runtime=parser.parse_known_args()
    if not 0 < options.simulation_seconds <= 60:raise ValueError('diagnostic budget must be 0..60s')
    manifest=json.loads(options.validation_manifest.read_text())
    if manifest['split']!='val' or options.source_episode.name not in manifest['episode_ids']:
        raise ValueError('only prepared validation sources are accepted')
    for name in ['samples.jsonl','frames.jsonl','summary.json','migration_task.json']:
        if sha256(options.source_episode/name)!=manifest['files'][f'source/{options.source_episode.name}/{name}']:
            raise ValueError('prepared source changed')
    if '--expected-identity' in runtime:raise ValueError('conditional diagnosis is not autonomous formal evaluation')
    if runtime[runtime.index('--num-episodes')+1] != '1':
        raise ValueError('conditioned PICK requires one fresh process per condition')
    reference=Path(runtime[runtime.index('--reference-root')+1]).resolve()
    runner.waypoint_runner._reference_identity(reference);sys.path.insert(0,str(reference))
    import source.simulation as simulation
    original=simulation.IsaacLabNavigationRuntimeConfig
    def config(*args,**kwargs):
        kwargs['camera_render_interval_control_steps']=10
        kwargs['enable_verified_grasp_fixed_joint']=False
        return original(*args,**kwargs)
    simulation.IsaacLabNavigationRuntimeConfig=config
    runner.JointTrajectoryRolloutPipeline=pipeline_type(options)
    result=runner.main(runtime)
    output_dir=Path(runtime[runtime.index('--output-dir')+1])
    summaries=list(output_dir.glob('episode_*/summary.json'))
    complete=len(summaries)==1 and json.loads(summaries[0].read_text()).get('status')=='complete'
    return result or (0 if complete else 1)


if __name__=='__main__':raise SystemExit(main())
