#!/usr/bin/env python3
"""Bounded full1700 physical RGB diagnosis using the existing causal rolling queue.

Sim5.1 state installation is an explicit new condition, not an exact source-solver
replay. Fixed-task diagnostics do not prove autonomous task completion. Missing
ordinary feedback or navigation certificates fail closed and retain artifacts.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import run_joint_trajectory_rollout as old
from scripts.run_conditioned_pick import first_array_row, PolicyCameraGrid
from scripts.serve_full_episode_diagnostic import PROTOCOL
from conveyor_bench.conveyorvla.contracts.observation import CurrentObservation
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, write_json
from conveyor_bench.conveyorvla.formal_physics import FormalPhysics
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.joint_trajectory_system import measured_named_joint_state
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointCommand
from conveyor_bench.conveyorvla.formal_metrics import LIMITS
from conveyor_bench.conveyorvla.rolling_runtime import RollingRuntime
from conveyor_bench.conveyorvla.staged_training import StagedNormalizer
from conveyor_bench.conveyorvla.task_memory import TaskMemory, Task, transfer_skeleton
from conveyor_bench.conveyorvla.full_episode_context import public_context_text


def source_query(root, tick):
    rows = {r['control_tick']: r for r in map(json.loads, (root/'control_effective_50hz.jsonl').open())}
    row, parent = rows[tick], rows[tick-1]
    if parent['reset_generation'] != row['reset_generation'] or not parent['verified']:
        raise ValueError('initial target lacks same-reset verified provenance')
    if any(not parent['channels'][k]['valid'] for k in ('arm','gripper')):
        raise ValueError('initial arm/gripper target invalid')
    observation = next(r for r in map(json.loads,(root/'observations.jsonl').open()) if r['observation_id']==row['observation_ref'])
    sample = next(r for r in map(json.loads,(root/'samples.jsonl').open()) if r['effective_command']['command_id']==row['command_id'])
    if sample['camera_capture_step'] != tick or sample['simulation_step'] != tick:
        raise ValueError('initial raw sample/tick binding differs')
    if observation['units'] != {'arm_q':'rad','arm_dq':'rad/s','gripper':'m'}:
        raise ValueError('source state units differ')
    return observation, sample, parent


def install_sim51(sim, obs, sample, parent):
    import numpy as np
    import torch
    robot = sim._adapter.robot
    names, source_names = list(robot.joint_names), obs['joint_names']
    if set(names) != set(source_names):
        raise ValueError('source/current joint names differ')
    q = [obs['q'][source_names.index(n)] for n in names]
    dq = [obs['dq'][source_names.index(n)] for n in names]
    tensor = lambda v: torch.tensor([v], device=robot.device, dtype=torch.float32)
    sim._adapter.set_base_pose_lock(False); sim._adapter.set_support_joint_lock(False)
    robot.write_root_pose_to_sim(tensor(obs['base_pose']))  # IsaacSim 5.1 WXYZ only.
    robot.write_root_velocity_to_sim(tensor(obs['base_velocity']))
    robot.write_joint_state_to_sim(tensor(q), tensor(dq))
    targets = list(q)
    for i, v in enumerate(parent['channels']['arm']['target'], 1):
        targets[names.index(f'arm_joint{i}')] = v
    targets[names.index('arm_joint7')] = parent['channels']['gripper']['target'][0]
    robot.set_joint_position_target(tensor(targets))
    obj = sample['object_state']
    result = sim._write_object_physics_state(position_xyz=tuple(obj[:3]),
        quaternion_wxyz=tuple(obj[3:7]), velocity_xyz_rpy=tuple(obj[7:]))
    if result.get('applied') is not True:
        raise ValueError('source object initialization unavailable')
    sim._runtime.scene.write_data_to_sim(); sim._runtime.sim.forward()
    actual = sim.read()
    def error(a,b):
        a,b = np.asarray(a,float),np.asarray(b,float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('initial state nonfinite or wrong shape')
        return float(np.max(np.abs(a-b)))
    def quaternion(a,b):
        return min(error(a,b),error(a,[-v for v in b]))
    errors = {'base_xyz':error(actual.robot_root_pose[:3],obs['base_pose'][:3]),
              'base_quaternion':quaternion(actual.robot_root_pose[3:],obs['base_pose'][3:]),
              'base_velocity':error(actual.robot_root_velocity,obs['base_velocity']),
              'q':error(actual.joint_positions,q), 'dq':error(actual.joint_velocities,dq),
              'object_xyz':error(actual.object_pose[:3],obj[:3]),
              'object_quaternion':quaternion(actual.object_pose[3:],obj[3:7]),
              'object_velocity':error(actual.object_velocity,obj[7:])}
    if max(errors.values()) > 1e-5:
        raise ValueError(f'source initialization readback failed: {errors}')
    applied = first_array_row(robot.data.joint_pos_target)
    if any(abs(applied[names.index(f'arm_joint{i}')]-targets[names.index(f'arm_joint{i}')])>1e-7 for i in range(1,8)):
        raise ValueError('effective arm/joint7 target install failed')
    return {'errors':errors, 'new_condition_not_solver_restore':True,
            'source_parent_command_id':parent['command_id'], 'object_write':result}


def camera_times(pair, states, live):
    if pair[1].step_index != live.step_index:
        raise ValueError('query camera is not live physics state')
    times = []
    for frame in pair:
        state = states[frame.step_index]
        report = state.metadata.get('camera_capture_report',{})
        if not report.get('accepted') or report.get('capture_step_index') != state.step_index or report.get('render_step_index') != state.step_index:
            raise ValueError('camera lacks accepted current render evidence')
        stamp = float(report['capture_timestamp'])
        if not math.isfinite(stamp) or abs(stamp-state.timestamp)>1e-7:
            raise ValueError('camera timestamp differs from physics')
        if not {'front','wrist'}.issubset(report.get('available_camera_keys',[])):
            raise ValueError('missing dual camera capture')
        times.append(stamp)
    if abs(times[1]-live.timestamp)>1e-7 or abs(times[1]-times[0]-.2)>1e-7:
        raise ValueError('camera history is not query/query-0.2')
    return times*2


def compare_shared_state(saved, current):
    if saved['step_index']!=current['step_index'] or abs(saved['timestamp']-current['timestamp'])>1e-7:
        raise ValueError('shared first physics clock differs')
    for key in ('joint_positions','joint_velocities','robot_root_velocity','object_velocity'):
        a,b=saved[key],current[key]
        if len(a)!=len(b) or not all(math.isfinite(v) for v in (*a,*b)) or max(abs(x-y) for x,y in zip(a,b))>1e-5:
            raise ValueError(f'shared first physical {key} differs')
    for key in ('robot_root_pose','object_pose'):
        a,b=saved[key],current[key]
        if len(a)!=7 or len(b)!=7 or not all(math.isfinite(v) for v in (*a,*b)):
            raise ValueError('shared first physical pose missing/nonfinite')
        rotation=min(max(abs(x-y) for x,y in zip(a[3:],b[3:])),max(abs(x+y) for x,y in zip(a[3:],b[3:])))
        if max(abs(x-y) for x,y in zip(a[:3],b[:3]))>1e-5 or rotation>1e-5:
            raise ValueError(f'shared first physical {key} differs')


def pipeline_type(options):
    class FullDiagnostic(old.JointTrajectoryRolloutPipeline):
        def _prepare_episode(self):
            self._record('stage_prepared', self.simulation.prepare_episode(self.episode_spec))
            self.simulation.reset(self.episode_spec, seed=self.episode_seed)

        def _measured_hold(self, source):
            if not hasattr(self,'trusted'):
                raise ValueError('same-episode effective target not initialized')
            return self.action_adapter.hold(self.trusted, route=self.route,
                sequence_id=self._query_count, source=source)

        def _read_trusted(self):
            robot = self.raw_sim._adapter.robot
            names = list(robot.joint_names); q = first_array_row(robot.data.joint_pos_target)
            fraction = q[names.index('arm_joint7')]/.04
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError('effective joint7 outside calibrated range')
            self.trusted = DirectJointCommand(0,tuple(q[names.index(f'arm_joint{i}')] for i in range(1,7)),fraction)

        def _obs(self, state, image_times=()):
            j = measured_named_joint_state(state)
            return CurrentObservation(self.memory.mission_id,f'{self.memory.mission_id}:tick{state.step_index}',
                float(state.timestamp),j.joint_position,j.joint_velocity,j.gripper_open_fraction,
                images=(None,)*len(image_times),image_times_s=tuple(image_times),
                base_xyyaw=(*state.robot_root_pose[:2],old.yaw_from_quaternion(state.robot_root_pose[3:])))

        def hold(self, observation, reason):
            self._physical_step(self._measured_hold('full1700_'+reason),route=self.route,command_index=None)

        def stop(self, observation, reason):
            # Stop advances no further physical tick; task-local simulation then closes.
            self._record('safety_stop',{'reason':reason,'time_s':observation.time_s})
            self.simulation.apply(self._measured_hold('full1700_safety_stop'))

        def apply(self, target, observation):
            action=self.action_adapter.manipulation(DirectJointCommand(0,tuple(target[:6]),target[6]),
                route=self.route,sequence_id=self._query_count)
            self._physical_step(action,route=self.route,command_index=None)
            self._read_trusted()
            applied=(*self.trusted.joint_position,self.trusted.gripper_open_fraction)
            if max(abs(a-b) for a,b in zip(applied,target))>1e-6:
                raise ValueError('effective controller target differs from queued target')
            return applied

        def safety(self, target, observation):
            if not all(math.isfinite(x) for x in (*observation.q,*observation.dq,*target)):
                return False,'nonfinite_online_joint_state_or_target'
            if any(q<lo-1e-5 or q>hi+1e-5 for q,lo,hi in zip(observation.q,LIMITS.lower,LIMITS.upper)):
                return False,'measured_joint_position_outside_limits'
            if any(abs(v)>bound for v,bound in zip(observation.dq,LIMITS.max_rate_rad_s)):
                return False,'measured_joint_velocity_above_bound'
            if any(q<lo or q>hi for q,lo,hi in zip(target[:6],LIMITS.lower,LIMITS.upper)) or not 0<=target[6]<=1:
                return False,'queued_joint_or_gripper_target_outside_limits'
            return True,'bounded_joint_diagnostic_only'

        def run_episode(self):
            self.episode_dir.mkdir(parents=True,exist_ok=True)
            self._trace_stream=(self.episode_dir/'trace.jsonl').open('x')
            self.raw_sim=self.simulation
            summary={'schema':'full1700-physical-diagnostic-v1','status':'running','success':False,
                'full_task_success':None,'deployment_gate_passed':False,'strict_full_success':None,
                'execution_mode':options.mode,'rtc':options.rtc,'time_profile':'causal_command_5hz',
                'replan_period_s':.4,'control_period_s':.02,'inference_pauses_simulation':True,
                'new_condition':options.condition_label,'source_solver_reproduction':False,
                'source_query_tick':options.query_tick,'source_episode':str(options.source_episode),
                'task_context_source':str(options.task_context),'task_context_sha256':sha256(options.task_context),
                'runner_sha256':sha256(Path(__file__)),'saturation_gate_threshold':.005,
                'training_weights_modified':False,'depth':False,
                'online_safety':{'measured_joint_limits_tolerance_rad':1e-5,'measured_joint_speed_limit_rad_s':3.,
                    'predicted_target_rate_limit_rad_s':3.,'target_rate_interval_s':.2,
                    'target_rate_is_not_50hz_actual_velocity_guarantee':True,'collision_certificate':False}}
            started=time.perf_counter(); probe=None
            try:
                self.client=old.JointTrajectoryHTTPClient(options.endpoint,timeout_s=180.)
                health=self.client.health()
                if health.get('protocol_version')!=PROTOCOL or health.get('global_step')!=1700 or health.get('weights_sha256')!=options.expected_sha256 or not health.get('strict_load'):
                    raise ValueError('full1700 service identity mismatch')
                summary['model_identity']=health
                context=json.loads(options.task_context.read_text());public_context_text(context)
                self.route=JointTrajectoryRoute(context['active_task'])
                if options.mode=='autonomous' and context != {'completed_tasks':[],'active_task':'NAV_TO_SOURCE','remaining_tasks':['NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'],'current_facts':{}}:
                    raise ValueError('autonomous attempt must start at complete four-task skeleton')
                obs,sample,parent=source_query(options.source_episode,options.query_tick)
                summary['source_hashes']={n:sha256(options.source_episode/n) for n in ('manifest.json','control_effective_50hz.jsonl','observations.jsonl','samples.jsonl')}
                fraction=parent['channels']['gripper']['target'][0]/.04
                if not 0<=fraction<=1:raise ValueError('trusted initial joint7 command invalid')
                self.trusted=DirectJointCommand(0,tuple(parent['channels']['arm']['target']),fraction)
                self.action_adapter.gripper_joint_names=('arm_joint7',)
                self._prepare_episode()
                summary['initialization']=install_sim51(self.raw_sim,obs,sample,parent)
                self._read_trusted()
                self.physics=FormalPhysics(self.raw_sim,'no_grasp_assist',self._record)
                self.simulation=self.physics;self.physics.arm()
                self.physics.previous_fraction=self.physics.command_fraction=fraction
                self.config=replace(self.config,video=replace(self.config.video,fps=5.))
                self._start_video()
                self.frames=PolicyCameraGrid(separation_steps=10,jpeg_quality=self.jpeg_quality)
                self._camera_states.clear();self._last_query_camera_step=None
                self.raw_sim._render_without_physics(valid_state_step=self.raw_sim.read().step_index,
                    reason='full1700_initial_state_sync',force=True)
                if options.record_contacts:
                    from conveyor_bench.isaac.grasp_contact_probe import IsaacGraspContactProbe
                    probe=IsaacGraspContactProbe(self.raw_sim,self._record);self.raw_sim.read_grasp_contacts=probe.read
                mission=f'full1700:{self.episode_seed}:{hashlib.sha256(str(self.episode_dir).encode()).hexdigest()[:12]}'
                tasks=transfer_skeleton() if options.mode=='autonomous' else (Task('fixed-diagnostic','fixed-attempt',self.route.value,'cola','destination'),)
                self.memory=TaskMemory(mission,str(self.episode_spec.instruction),tasks,mode='H1')
                runtime=RollingRuntime(self.memory,model_id=health['checkpoint_id'],
                    normalizer=StagedNormalizer(health['normalizer']),safety_context_id='bounded-diagnostic-joint-limits',
                    limits=LIMITS,rtc=options.rtc,time_profile='causal_command_5hz')
                self.runtime=runtime
                # PLACE/carry initialization does not turn teacher events into ordinary current facts.
                if self.route in {JointTrajectoryRoute.PLACE,JointTrajectoryRoute.NAV_TO_TARGET}:
                    summary.update(status='blocked',failure_reason='independent_current_carrying_feedback_unavailable')
                    return summary
                control_start=None
                for query in range(options.max_queries):
                    payload,state=self._next_request()
                    if control_start is None:control_start=self._control_steps
                    pair=self.frames.pair_after(None)
                    stamps=camera_times(pair,self._camera_states,state)
                    observation=self._obs(state,stamps);self.memory.observe(observation)
                    request=runtime.prepare_request(observation)
                    request=replace(request,plan_context=context)
                    runtime.pending[request.request_id]=request
                    wire=asdict(request);wire['observation'].pop('images')
                    packet={'protocol_version':PROTOCOL,'request':wire,
                        'instruction':str(self.episode_spec.instruction),
                        'head_images':payload['head_images'],'wrist_images':payload['wrist_images'],
                        'diffusion_seed':(options.diffusion_seed+query*1009)%(2**32),
                        'predict_transition':options.mode=='autonomous'}
                    packet=old.waypoint_runner._jsonable(packet)
                    self._record('model_request',packet)
                    before=self.simulation.read().step_index;request_started=time.perf_counter()
                    if query==0 and options.rtc and options.first_plan is not None:
                        saved=json.loads(options.first_plan.read_text())
                        prior=saved['request']; prior_obs=prior['request']['observation']
                        compare_shared_state(saved['query_state'],old.waypoint_runner._state_snapshot(state))
                        if saved['response']['weights_sha256']!=options.expected_sha256 or prior['request']['plan_context']!=context or prior['instruction']!=packet['instruction']:
                            raise ValueError('shared first plan model/task changed')
                        if prior_obs['time_s']!=wire['observation']['time_s'] or prior_obs['image_times_s']!=list(stamps):
                            raise ValueError('shared first plan time differs')
                        for key in ('q','dq','base_xyyaw'):
                            if max(abs(a-b) for a,b in zip(prior_obs[key],wire['observation'][key]))>1e-5:
                                raise ValueError(f'shared first plan measured {key} differs')
                        if abs(prior_obs['gripper']-observation.gripper)>1e-5:
                            raise ValueError('shared first measured gripper differs')
                        summary['shared_first_rgb_hash_equal']={k:[a==b for a,b in zip(prior[k],packet[k])] for k in ('head_images','wrist_images')}
                        summary['shared_first_plan_sha256']=sha256(options.first_plan)
                        response={**saved['response'],'request_id':request.request_id,'identity':wire['identity'],
                            'observation_id':observation.observation_id,'query_time_s':observation.time_s}
                        self._record('adopted_shared_first_proposal',{'original_request':prior['request'],
                            'first_plan_sha256':summary['shared_first_plan_sha256'],'new_identity':wire['identity']})
                    else:
                        response=self.client.infer(packet)
                        if query==0 and options.first_plan is not None:
                            with options.first_plan.open('x') as out:json.dump({'request':packet,'response':response,
                                'query_state':old.waypoint_runner._jsonable(old.waypoint_runner._state_snapshot(state))},out)
                    if self.simulation.read().step_index!=before:raise ValueError('paused inference moved physics')
                    self._record('model_response',{'response':response,'client_roundtrip_s':time.perf_counter()-request_started})
                    if response['identity']!=wire['identity'] or response['request_id']!=request.request_id or response['observation_id']!=observation.observation_id or response['query_time_s']!=observation.time_s or response['time_profile']!='causal_command_5hz' or response['weights_sha256']!=options.expected_sha256 or response['rtc_applied']!=(request.rtc_context is not None):
                        raise ValueError('foreign/stale model proposal')
                    self._query_count+=1
                    transition=response.get('transition')
                    if transition is not None and transition['proposal']['operation']=='ADVANCE':
                        summary.update(status='blocked',failure_reason='ADVANCE_without_independent_completion_feedback',blocked_proposal=transition)
                        break
                    plan=runtime.complete_request(request.request_id,response['physical_actions'],now_s=observation.time_s)
                    if self.route.value.startswith('NAV_'):
                        self._record('navigation_proposal',old.waypoint_runner._jsonable(plan))
                        summary.update(status='blocked',failure_reason='navigation_online_geometry_and_stop_certificate_unavailable')
                        break
                    for _ in range(20):
                        live=self.simulation.read(); current=self._obs(live)
                        runtime.tick(current,safety=self.safety,controller=self)
                        if runtime.safety_stop_reason is not None:
                            summary.update(status='failed',failure_reason='safety_stop_'+runtime.safety_stop_reason)
                            break
                    if runtime.safety_stop_reason is not None:break
                    if self._control_steps-control_start>=round(options.simulation_seconds/.02):
                        summary.update(status='complete',failure_reason=None);break
                else:
                    summary.update(status='complete',failure_reason='query_budget_reached')
            except Exception as error:
                summary.update(status='failed',failure_reason=f'{type(error).__name__}:{error}',traceback=traceback.format_exc())
            finally:
                if hasattr(self,'runtime'):
                    summary['runtime_events']=self.runtime.events
                    summary['queue_applied_points']=len(self.runtime.queue.history)
                    counts=sum(sum(e.get('saturation_events',{}).values()) for e in self.runtime.events)
                    denom=sum(e.get('denominator',0) for e in self.runtime.events)
                    summary['saturation_rate']=counts/denom if denom else None
                    summary['saturation_gate_passed']=bool(denom and counts/denom<=.005)
                    self.runtime.queue.cancel(reason='diagnostic_end')
                summary.update(wall_s=time.perf_counter()-started,model_queries=self._query_count,
                    control_steps=self._control_steps,physics_evidence=None if self.physics is None else self.physics.evidence(),
                    success=False,success_semantics='diagnostic_completion_is_not_task_success')
                try:
                    if hasattr(self,'trusted'):self.simulation.apply(self._measured_hold('full1700_final_hold'))
                    summary['video']=self._close_video(summary['status'])
                except Exception as error:summary['close_error']=str(error)
                if probe is not None:probe.close();del self.raw_sim.read_grasp_contacts
                write_json(self.episode_dir/'summary.json',old.waypoint_runner._jsonable(summary))
                self._trace_stream.close();self._trace_stream=None
                if self.close_simulation_on_exit:self.raw_sim.close()
            return summary
    return FullDiagnostic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-episode',type=Path,required=True)
    p.add_argument('--query-tick',type=int,required=True)
    p.add_argument('--task-context',type=Path,required=True)
    p.add_argument('--condition-label',required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:18170')
    p.add_argument('--mode',choices=['fixed_task','autonomous'],required=True)
    p.add_argument('--rtc',action='store_true')
    p.add_argument('--record-contacts',action='store_true')
    p.add_argument('--first-plan',type=Path,help='RTC-off creates; paired RTC-on verifies state and adopts exact first prediction')
    p.add_argument('--simulation-seconds',type=float,default=4.)
    p.add_argument('--max-queries',type=int,default=10)
    p.add_argument('--diffusion-seed',type=int,default=17)
    options,runtime=p.parse_known_args()
    if not 0<options.simulation_seconds<=60 or not 0<options.max_queries<=150:
        raise ValueError('bounded diagnostic only: <=60 physics seconds and <=150 queries')
    if options.rtc and options.mode=='fixed_task' and (options.first_plan is None or not options.first_plan.is_file()):
        raise ValueError('paired RTC-on requires prior frozen RTC-off first plan')
    if not options.rtc and options.first_plan is not None and options.first_plan.exists():
        raise ValueError('RTC-off first plan path must be new')
    if '--expected-identity' in runtime:raise ValueError('new diagnostic must not impersonate old formal protocol')
    if runtime[runtime.index('--num-episodes')+1]!='1':raise ValueError('one fresh process per condition required')
    reference=Path(runtime[runtime.index('--reference-root')+1]).resolve()
    old.waypoint_runner._reference_identity(reference);sys.path.insert(0,str(reference))
    import source.simulation as simulation
    original=simulation.IsaacLabNavigationRuntimeConfig
    def config(*args,**kwargs):
        kwargs['camera_render_interval_control_steps']=10
        kwargs['enable_verified_grasp_fixed_joint']=False
        return original(*args,**kwargs)
    simulation.IsaacLabNavigationRuntimeConfig=config
    # CUDA sees only physical GPU3; RTX/Vulkan retains the physical ordinal.
    # Verified against this bundled IsaacLab5.1 _resolve_device_settings method.
    import isaaclab.app as isaac_app
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('this frozen H20 diagnostic requires physical GPU3 exclusively')
    base = isaac_app.AppLauncher
    class _SingleGPUAppLauncher(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The bundled converter hardcodes /tmp when usd_dir is absent.
            # Bind only this process's conversion output to its authorized run.
            from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
            original_init = AssetConverterBase.__init__
            cache = Path(os.environ['TMPDIR']).resolve() / 'converted_assets'
            if not cache.is_relative_to(Path('/diff/wallx_workspace/dzb').resolve()):
                raise ValueError('asset conversion cache escapes authorized work root')
            def run_scoped_conversion(instance, cfg):
                if cfg.usd_dir is None:
                    cfg.usd_dir = str(cache / hashlib.sha256(str(cfg.asset_path).encode()).hexdigest()[:12])
                return original_init(instance, cfg)
            AssetConverterBase.__init__ = run_scoped_conversion
            print(json.dumps({'event':'run_scoped_asset_conversion','cache':str(cache)}),flush=True)
        def _resolve_device_settings(self, launcher_args):
            super()._resolve_device_settings(launcher_args)
            if self.device_id != 0:
                raise ValueError('single-visible-GPU simulation requires logical cuda:0')
            launcher_args.update(physics_gpu=0, active_gpu=3, multi_gpu=False)
            launcher_args['kit_args'] = (launcher_args.get('kit_args','') +
                ' --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false'
                ' --/renderer/multiGpu/maxGpuCount=1').strip()
            print(json.dumps({'event':'explicit_gpu_binding','cuda_visible_devices':'3',
                'physics_logical_gpu':0,'render_physical_gpu':3,'multi_gpu':False}),flush=True)
    isaac_app.AppLauncher = _SingleGPUAppLauncher
    old.JointTrajectoryRolloutPipeline=pipeline_type(options)
    return old.main(runtime)


if __name__=='__main__':
    raise SystemExit(main())
