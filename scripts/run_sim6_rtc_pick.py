#!/usr/bin/env python3
"""Bounded new-Sim6 PICK diagnosis. Source initialization truth never enters HTTP."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys, time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import run_conditioned_pick as base
from conveyor_bench.conveyorvla import waypoint_planner_adapters as adapters
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.rtc_sampling import prefix_weights
PROTOCOL='conveyorvla-conditioned-pick-rtc-diagnostic/v1'
NORMALIZER='a8805b21718f7d8bbff6df17b361e1fd2cdf8f15ec9f70294add188500d22725'


def source_query(root,tick):
    controls=[json.loads(l) for l in (root/'control_effective_50hz.jsonl').open()]
    rows={r['control_tick']:r for r in controls}
    row=rows[tick];previous=rows[tick-1]
    if previous['reset_generation']!=row['reset_generation'] or not previous['verified']:
        raise ValueError('no trustworthy same-reset initial effective target')
    for name in ('arm','gripper'):
        if not previous['channels'][name]['valid']:raise ValueError('invalid initial target channel')
    gripper=previous['channels']['gripper']['target'][0]/.04
    if not 0<=gripper<=1:raise ValueError('initial commanded gripper outside calibrated range')
    obs=next(json.loads(l) for l in (root/'observations.jsonl').open()
             if json.loads(l)['observation_id']==row['observation_ref'])
    sample=next(json.loads(l) for l in (root/'samples.jsonl').open()
                if json.loads(l)['effective_command']['command_id']==row['command_id'])
    if sample['camera_capture_step']!=tick or sample['simulation_step']!=tick:
        raise ValueError('source initialization sample is not same tick')
    if obs['units']!={'arm_q':'rad','arm_dq':'rad/s','gripper':'m'}:raise ValueError('source unit contract differs')
    for values in (obs['q'],obs['dq'],obs['base_pose'],obs['base_velocity'],sample['object_state'],previous['channels']['arm']['target']):
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite source initialization')
    # Evaluator-only initialization. object_state is never returned to policy requests.
    return obs,sample,previous,gripper


def source_reference(root,expected):
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
    allowed={
        'source/scene/liangzhu':Path('/hdd1/dzb_xhq/VLA/conveyorvla-v3/liangzhu'),
        'source/scene/objects':Path('/hdd1/dzb_xhq/VLA/conveyorvla-v3/objects')}
    for line in dirty.splitlines():
        name=line[3:]
        if line[:3]!='?? ' or name not in allowed:raise ValueError(f'unapproved source modification: {line}')
        path=root/name
        if not path.is_symlink() or path.resolve()!=allowed[name] or not path.is_dir():raise ValueError('source asset link binding differs')
    if len(expected)!=40 or actual!=expected:
        raise ValueError('new Sim6 reference must match explicit SHA')
    return actual


def verified_image_times(pair,states,live):
    if pair[1].step_index!=live.step_index:raise ValueError('policy image query is older than live physics')
    times=[]
    for frame in pair:
        state=states[frame.step_index];report=state.metadata.get('camera_capture_report',{})
        if not report.get('accepted') or report.get('capture_step_index')!=state.step_index or report.get('render_step_index')!=state.step_index:
            raise ValueError('unverified rendered camera state')
        stamp=float(report['capture_timestamp'])
        if not math.isfinite(stamp) or abs(stamp-state.timestamp)>1e-7:raise ValueError('camera capture time differs from state')
        if not {'front','wrist'}.issubset(report.get('available_camera_keys',[])):raise ValueError('dual camera image missing')
        times.append(stamp)
    if abs(times[1]-live.timestamp)>1e-7 or abs(times[1]-times[0]-.2)>1e-7:raise ValueError('camera pair is not real current/t-0.2')
    return times*2


def quaternion_error(a,b):
    if len(a)!=4 or len(b)!=4 or not all(math.isfinite(v) for v in [*a,*b]):raise ValueError('invalid quaternion')
    return min(max(abs(x-y) for x,y in zip(a,b)),max(abs(x+y) for x,y in zip(a,b)))


def compare_physical_states(left,right):
    if left['step_index']!=right['step_index'] or not all(math.isfinite(v) for v in (left['timestamp'],right['timestamp'])) or abs(left['timestamp']-right['timestamp'])>1e-7:raise ValueError('shared physical clock differs')
    for key in ('joint_positions','joint_velocities','robot_root_velocity','object_velocity'):
        a,b=left[key],right[key]
        if len(a)!=len(b) or not all(math.isfinite(v) for v in [*a,*b]) or max(abs(x-y) for x,y in zip(a,b))>1e-5:
            raise ValueError(f'shared physical {key} differs')
    for key in ('robot_root_pose','object_pose'):
        a,b=left[key],right[key]
        if not all(math.isfinite(v) for v in [*a,*b]) or max(abs(x-y) for x,y in zip(a[:3],b[:3]))>1e-5 or quaternion_error(a[3:],b[3:])>1e-5:
            raise ValueError(f'shared physical {key} differs')


def validate_response(request,response,health):
    for key in ('request_id','observation_id','sequence_id','active_task_id','active_task_epoch','query_time_s','time_profile'):
        if response.get(key)!=request[key]:raise ValueError(f'response identity differs: {key}')
    if response.get('checkpoint_id')!=health['checkpoint_id'] or response.get('normalization_sha256')!=health['normalization_sha256']:
        raise ValueError('response model identity differs')
    if response.get('rtc_applied')!=(request.get('rtc_context') is not None):raise ValueError('response RTC branch differs')
    commands=response['chunk']['commands']
    if len(commands)!=10:raise ValueError('legacy prediction must contain10points')
    for index,c in enumerate(commands):
        if c['index']!=index or not math.isfinite(c['duration_s']) or abs(c['duration_s']-.2)>1e-9 or list(c['base_velocity'])!=[0.,0.,0.]:
            raise ValueError('proposal execution interval or PICK base command differs')
        values=[*c['joint_position'],c['gripper_open_fraction']]
        if len(values)!=7 or not all(math.isfinite(v) for v in values) or not 0<=values[-1]<=1:
            raise ValueError('invalid proposal targets')
    counts={k:int(response['chunk'][k]) for k in ('position_saturation_count','rate_saturation_count','gripper_saturation_count')}
    if any(v<0 for v in counts.values()):raise ValueError('negative saturation counter')
    rate=sum(counts.values())/70
    return {'request_id':request['request_id'],'counts':counts,'denominator':70,'rate':rate,'gate_passed':rate<=.005}


def request_payload(payload,state,image_times,epoch=0):
    query=float(state.timestamp)
    return {**payload,'protocol_version':PROTOCOL,'observation_id':f"{payload['episode_id']}:t{state.step_index}",
        'query_time_s':query,'image_times_s':image_times,
        'active_task_id':'canonical_PICK_diagnostic','active_task_epoch':epoch,'time_profile':'legacy_future_5hz'}


def prior_context(payload,prior,health):
    # Existing queue targets are selected by actual application time, never by label time.
    query=payload['query_time_s'];start=prior['query_time_s']
    offset=round((query-start)/.2)
    if offset<1 or abs(start+offset*.2-query)>1e-7:raise ValueError('unaligned prior execution queue')
    commands=prior['commands'];available=[];absolute=[]
    for i in range(10):
        index=offset+i;valid=index<len(commands);available.append(valid)
        c=commands[index] if valid else None
        absolute.append([*c['joint_position'],c['gripper_open_fraction']] if valid else [0.]*7)
    return {'identity':{k:payload[k] for k in ('episode_id','active_task_id','active_task_epoch')}|
                {'weights_sha256':health['weights_sha256'],'normalization_sha256':health['normalization_sha256']},
        'source_action_id':prior['request_id'],'source_query_time_s':start,'source_sequence_id':prior['sequence_id'],
        'previous_absolute':absolute,'target_apply_times_s':[query+i*.2 for i in range(10)],
        'available_previous':available,'weights':prefix_weights(0,sum(available),10).tolist(),'max_guidance_weight':5.}


def fingerprint(payload):
    # Exact RGB identity; floating readback tolerance is separately frozen below.
    return {k:[hashlib.sha256(v.encode()).hexdigest() for v in payload[k]]
            for k in ('head_images','wrist_images')}


def initialize_sim6(simulation,obs,sample,previous):
    import torch,numpy as np
    robot=simulation._adapter.robot;names=list(robot.joint_names);source_names=obs['joint_names']
    if set(names)!=set(source_names):raise ValueError('articulation joint names differ')
    tensor=lambda x:torch.tensor([x],dtype=torch.float32,device=robot.device)
    q=[obs['q'][source_names.index(n)] for n in names];dq=[obs['dq'][source_names.index(n)] for n in names]
    simulation._adapter.set_base_pose_lock(False);simulation._adapter.set_support_joint_lock(False)
    p=obs['base_pose'];robot.write_root_pose_to_sim(tensor([*p[:3],*p[4:7],p[3]])) # Sim6 articulation XYZW
    robot.write_root_velocity_to_sim(tensor(obs['base_velocity']));robot.write_joint_state_to_sim(tensor(q),tensor(dq))
    targets=list(q)
    for i,v in enumerate(previous['channels']['arm']['target'],1):targets[names.index(f'arm_joint{i}')]=v
    targets[names.index('arm_joint7')]=previous['channels']['gripper']['target'][0]
    robot.set_joint_position_target(tensor(targets))
    obj=sample['object_state'] # Physics object wrapper retains WXYZ.
    report=simulation._write_object_physics_state(position_xyz=tuple(obj[:3]),quaternion_wxyz=tuple(obj[3:7]),velocity_xyz_rpy=tuple(obj[7:]))
    if report.get('applied') is not True:raise ValueError('object initialization unsupported')
    simulation._runtime.scene.write_data_to_sim();simulation._runtime.sim.forward();actual=simulation.read()
    errors={'base_xyz':math.dist(actual.robot_root_pose[:3],p[:3]),
        'base_quaternion':quaternion_error(actual.robot_root_pose[3:],p[3:]),
        'base_velocity':max(abs(a-b) for a,b in zip(actual.robot_root_velocity,obs['base_velocity'])),
        'q':float(np.max(np.abs(np.array(actual.joint_positions)-q))),
        'dq':float(np.max(np.abs(np.array(actual.joint_velocities)-dq))),
        'object_xyz':math.dist(actual.object_pose[:3],obj[:3]),
        'object_quaternion':quaternion_error(actual.object_pose[3:],obj[3:7]),
        'object_velocity':max(abs(a-b) for a,b in zip(actual.object_velocity,obj[7:]))}
    if not all(math.isfinite(v) for v in errors.values()) or max(errors.values())>1e-5:raise ValueError(f'initialization readback failed: {errors}')
    actual_target=base.first_array_row(robot.data.joint_pos_target)
    for n in [f'arm_joint{i}' for i in range(1,8)]:
        if abs(actual_target[names.index(n)]-targets[names.index(n)])>1e-7:raise ValueError('trusted target install failed')
    return {'initialization_only':True,'new_condition_not_solver_state_restore':True,'errors':errors,
        'trusted_parent_command_id':previous['command_id'],'trusted_channels':previous['channels'],
        'joint7_target_m':targets[names.index('arm_joint7')],'object_write':report}


def pipeline_type(options):
    parent=base.pipeline_type(options)
    class Sim6RTC(parent):
        def _prepare_episode(self):
            # Restore the audited source state before any simulated warmup step.
            report=self.simulation.prepare_episode(self.episode_spec)
            self._record('stage_prepared',report)
            self.simulation.reset(self.episode_spec,seed=self.episode_seed)
            self._record('episode_reset',{'seed':self.episode_seed,'source_restore_follows':True})
        def _next_request(self):
            payload,state=super()._next_request();pair=self.frames.pair_after(None)
            if pair is None:raise ValueError('camera pair missing')
            live=self.simulation.read()
            if state.step_index!=live.step_index or abs(state.timestamp-live.timestamp)>1e-7:raise ValueError('query state is stale')
            self._verified_image_times=verified_image_times(pair,self._camera_states,live)
            payload={**payload,'episode_id':self.diagnostic_episode_id,'request_id':f'{self.diagnostic_episode_id}:request{self._query_count}'}
            return payload,state
        def _source_initialization(self):
            health=self.client.health()
            if health.get('normalization_sha256')!=NORMALIZER or health.get('time_profile')!='legacy_future_5hz':raise ValueError('legacy model identity mismatch before physics')
            self.raw_obs,self.raw_sample,self.initial_parent,self.initial_fraction=source_query(options.source_episode,options.query_tick)
            self.trusted_parent=self.initial_parent['command_id']
            self.trusted_sequence=0
            self.trusted=base.runner.DirectJointCommand(index=0,joint_position=tuple(self.initial_parent['channels']['arm']['target']),gripper_open_fraction=self.initial_fraction)
            manifest=json.loads((options.source_episode/'manifest.json').read_text())
            self.diagnostic_episode_id=f"{manifest['episode_uuid']}:q{options.query_tick}:{options.branch}:{hashlib.sha256(str(self.episode_dir).encode()).hexdigest()[:12]}"
            if manifest['task_family_id']!=f'liangzhu_seed_{options.source_seed}':raise ValueError('source family and explicit seed differ')
            return options.source_seed,self.raw_obs
        def _initialize_state(self,simulation,observation):
            return initialize_sim6(simulation,observation,self.raw_sample,self.initial_parent)
        def _initial_gripper_fraction(self):return self.initial_fraction
        def _measured_hold(self,source):
            if not hasattr(self,'trusted'):raise ValueError('hold requires explicit same-episode initial target')
            action=self.action_adapter.hold(self.trusted,route=JointTrajectoryRoute.PICK,sequence_id=self.trusted_sequence,source=source)
            return replace(action,metadata={**action.metadata,'trusted_target_parent':self.trusted_parent})
        def _execute_policy(self,options,summary,health,payload,state):
            if health['normalization_sha256']!=NORMALIZER:raise ValueError('legacy normalizer identity differs')
            payload=request_payload(payload,state,self._verified_image_times)
            first_path=options.first_plan
            identity={'source_reference_sha':options.source_sha,'source_manifest_sha':sha256(options.source_episode/'manifest.json'),
                      'query_tick':options.query_tick,'vla_code_sha':options.code_sha,'weights_sha256':health['weights_sha256'],'normalization_sha256':health['normalization_sha256']}
            saturation=[]
            def infer(request):
                self._record('model_request',request);before=self.simulation.read().step_index;started=time.perf_counter()
                response=self.client.infer(request);elapsed=time.perf_counter()-started
                if self.simulation.read().step_index!=before:raise ValueError('paused diagnostic unexpectedly advanced physics')
                saturation.append(validate_response(request,response,health))
                summary['saturation_by_query']=saturation
                summary['saturation_gate_passed']=all(r['gate_passed'] for r in saturation)
                self._record('model_response',{'response':response,'client_roundtrip_s':elapsed,'clock':'source_perf_counter','paused_physics':True})
                return response
            if options.branch=='old_tail':
                if first_path.exists():raise ValueError('first plan path must be new')
                result=infer(payload)
                write_json(first_path,{'identity':identity,'request':payload,'response':result,'image_hashes':fingerprint(payload),'query_state':base.runner.waypoint_runner._state_snapshot(state)})
            else:
                saved=json.loads(first_path.read_text());result=saved['response']
                if saved['identity']!=identity or saved['image_hashes']!=fingerprint(payload):raise ValueError('new Sim6 shared first condition/RGB differs')
                snapshot=base.runner.waypoint_runner._state_snapshot(state)
                compare_physical_states(saved['query_state'],snapshot)
                saturation.append(validate_response(saved['request'],result,health))
                if abs(saved['request']['gripper_open_fraction']-payload['gripper_open_fraction'])>1e-5:raise ValueError('shared measured gripper differs')
                for key in ('joint_position','joint_velocity'):
                    if max(abs(a-b) for a,b in zip(saved['request'][key],payload[key]))>1e-5:raise ValueError('shared initial proprioception differs')
            first=result['chunk']['commands'];prior={**payload,'commands':first}
            if options.branch!='old_tail':
                prior['request_id']='adopted-first-plan:'+sha256(first_path)
                self._record('adopted_first_plan',{'first_plan_sha256':sha256(first_path),'original_request_id':saved['request']['request_id'],'adopted_action_id':prior['request_id'],'episode_id':payload['episode_id']})
            summary['saturation_by_query']=saturation
            summary['saturation_gate_passed']=all(r['gate_passed'] for r in saturation)
            if len(first)!=10:raise ValueError('expected legacy10point prediction')
            summary.update(schema='sim6-rtc-pick-diagnostic-v1',branch=options.branch,first_plan_sha256=sha256(first_path),
                source_reference_sha=options.source_sha,condition='new_Sim6_raw_N_initialization',paused_simulation=True,
                realtime_pass=False,window_s=1.6,extra_hold_s=0.,saturation_gate_threshold=.005)
            first_action_id=prior['request_id']
            def execute(command,origin,owner_sequence,owner_action_id):
                c=base.runner._command(command);action=self.action_adapter.manipulation(c,route=JointTrajectoryRoute.PICK,sequence_id=owner_sequence)
                action=replace(action,metadata={**action.metadata,'rtc_diagnostic_action_id':owner_action_id})
                self._record('plan_point',{'origin':origin,'owner_sequence_id':owner_sequence,'owner_action_id':owner_action_id,'command':command,'actual_apply_start_s':self.simulation.read().timestamp,'duration_s':.2})
                for _ in range(10):self._physical_step(action,route=JointTrajectoryRoute.PICK,command_index=c.index)
                # Read back the effective actuator targets, not measured finger opening.
                robot=self.simulation.simulation._adapter.robot;names=list(robot.joint_names);target=base.first_array_row(robot.data.joint_pos_target)
                grip=target[names.index('arm_joint7')]/.04
                if not 0<=grip<=1:raise ValueError('effective joint7 target outside calibrated range')
                self.trusted_sequence=owner_sequence;self.trusted_parent=owner_action_id
                self.trusted=base.runner.DirectJointCommand(index=c.index,joint_position=tuple(target[names.index(f'arm_joint{i}')] for i in range(1,7)),gripper_open_fraction=grip)
            self._query_count=1
            for c in first[:2]:execute(c,'shared_first2',0,first_action_id)
            summary['fork_state']=base.runner.waypoint_runner._state_snapshot(self.simulation.read())
            fork_file=first_path.with_suffix('.fork.json')
            if options.branch=='old_tail':
                if fork_file.exists():raise ValueError('fork evidence path must be new')
                write_json(fork_file,summary['fork_state'])
            else:
                saved_fork=json.loads(fork_file.read_text())
                compare_physical_states(saved_fork,summary['fork_state'])
            start=self.simulation.read().timestamp
            for block in range(4):
                selected=first[2+block*2:4+block*2];owner_sequence=0;owner_action_id=first_action_id
                if options.branch!='old_tail':
                    before=self.simulation.read().step_index;p,s=self._next_request()
                    if self.simulation.read().step_index!=before:raise ValueError('query history unavailable without extra physics')
                    p=request_payload(p,s,self._verified_image_times)
                    if options.branch=='rtc':p['rtc_context']=prior_context(p,prior,health)
                    response=infer(p);selected=response['chunk']['commands'][:2];owner_sequence=p['sequence_id'];owner_action_id=p['request_id']
                    prior={**p,'commands':response['chunk']['commands']};self._query_count+=1
                for c in selected:execute(c,f'{options.branch}_{block}',owner_sequence,owner_action_id)
            summary['duration_after_fork_s']=self.simulation.read().timestamp-start
            summary['window_physics_evidence']=self.physics.evidence()
    return Sim6RTC


def run_and_check_summary(run,output):
    try:
        result=run()
    except SystemExit as exit_status:
        result=exit_status.code if isinstance(exit_status.code,int) else (0 if exit_status.code is None else 1)
    summaries=list(output.glob('episode_*/summary.json'))
    complete=len(summaries)==1 and json.loads(summaries[0].read_text()).get('status')=='complete'
    return result or (0 if complete else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-episode',type=Path,required=True);p.add_argument('--query-tick',type=int,required=True)
    p.add_argument('--source-sha',required=True);p.add_argument('--branch',choices=['old_tail','replacement','rtc'],required=True)
    p.add_argument('--first-plan',type=Path,required=True);p.add_argument('--endpoint',default='http://127.0.0.1:18092')
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--preflight-output',type=Path)
    p.add_argument('--code-manifest',type=Path,required=True)
    options,runtime=p.parse_known_args();options.execute_points=2;options.simulation_seconds=2.;options.video_fps=5;options.record_contacts=True
    options.source_seed=int(runtime[runtime.index('--seed')+1])
    reference=Path(runtime[runtime.index('--reference-root')+1]).resolve()
    code_manifest=json.loads(options.code_manifest.read_text())
    if len(code_manifest['commit'])!=40:raise ValueError('VLA snapshot commit missing')
    for name,expected in code_manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path)!=expected:raise ValueError(f'VLA snapshot file mismatch: {name}')
    for name in ('scripts/run_sim6_rtc_pick.py','scripts/run_conditioned_pick.py'):
        if name not in code_manifest['files']:raise ValueError('VLA manifest omits required entry')
    options.code_sha=code_manifest['commit']
    task_path=Path(runtime[runtime.index('--task-json')+1]).resolve()
    task=json.loads(task_path.read_text(encoding='utf-8'));asset_hashes={}
    for key in ('scene_usd','nav_map'):
        if not task.get(key):
            asset_hashes[key]={'task_value':task.get(key),'resolution':'source scene-profile runtime; verified during pipeline construction'}
            continue
        path=Path(task[key]);path=path if path.is_absolute() else reference/path
        if not path.is_file():raise ValueError(f'actual task asset unavailable: {key}={path}')
        asset_hashes[key]={'path':str(path.resolve()),'sha256':sha256(path)}
    source_reference(reference,options.source_sha);obs,sample,previous,grip=source_query(options.source_episode,options.query_tick)
    if runtime[runtime.index('--num-episodes')+1]!='1':raise ValueError('one fresh process per branch required')
    if '--expected-identity' in runtime:raise ValueError('diagnostic cannot claim autonomous formal identity')
    sys.path.insert(0,str(reference))
    source_main=base.runner.waypoint_runner._load_reference_main(reference)
    downstream=runtime[runtime.index('--')+1:]
    from urllib.parse import urlparse
    endpoint_port=urlparse(runtime[runtime.index('--model-endpoint')+1]).port
    parsed_source=source_main._parse_args([*downstream,'--remote-vla-eval','--vla-endpoint',f'ws://127.0.0.1:{endpoint_port}'])
    if options.preflight_only:
        report={'status':'cpu_contract_preflight_only','source_sha':options.source_sha,'query_tick':options.query_tick,
                'trusted_parent_command':previous['command_id'],'joint7_initial_target_m':grip*.04,
                'source_cli_parse':{'mode':parsed_source.mode,'device':parsed_source.device,'actual_downstream_argc':len(downstream)+3},'physics_run':False,'vla_code_sha':options.code_sha,'actual_task_sha256':sha256(task_path),'task_assets':asset_hashes,'missing_runtime_checks':['Sim6 root readback','effective motor target readback','shared first RGB/state match','source HTTP roundtrip'],
                'source_hashes':{n:sha256(options.source_episode/n) for n in ('manifest.json','observations.jsonl','samples.jsonl','control_effective_50hz.jsonl','task.json')}}
        if options.preflight_output:write_json(options.preflight_output,report)
        print(json.dumps(report));return 0
    # Explicit new reference selection in this separate process; both old pins remain unchanged elsewhere.
    base.runner.waypoint_runner._reference_identity=lambda root:source_reference(root,options.source_sha)
    base.runner.waypoint_runner.APPROVED_ARM_VLA_COMMIT=options.source_sha
    adapters.APPROVED_ARM_VLA_COMMIT=options.source_sha
    base.runner.APPROVED_ARM_VLA_COMMIT=options.source_sha
    base.PROTOCOL=PROTOCOL
    sys.path.insert(0,str(reference));import source.simulation as simulation
    original=simulation.IsaacLabNavigationRuntimeConfig
    def config(*args,**kwargs):
        kwargs['camera_render_interval_control_steps']=10;kwargs['enable_verified_grasp_fixed_joint']=False
        kwargs['enable_front_camera']=True;kwargs['enable_wrist_camera']=True
        return original(*args,**kwargs)
    simulation.IsaacLabNavigationRuntimeConfig=config
    base.runner.JointTrajectoryRolloutPipeline=pipeline_type(options)
    output=Path(runtime[runtime.index('--output-dir')+1])
    return run_and_check_summary(lambda:base.runner.main(runtime),output)


if __name__=='__main__':raise SystemExit(main())
