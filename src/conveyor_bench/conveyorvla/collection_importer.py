"""Read collection release v2 in place; derived views never read evaluator truth.

This version imports actuator witnesses and samples, not similarly named proposed
wire records. Teacher phase is an offline action route label, never planner input.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .staged_data import RawEpisode, ARM_NAMES, jsonl, digest
from .contracts.observation import CurrentObservation

IMPORTER_VERSION = 'collection-to-staged-v1'


def pose_matrix(pose):
    x,y,z,w,qx,qy,qz = map(float,pose)
    q=np.array([w,qx,qy,qz]); norm=np.linalg.norm(q)
    if not np.isfinite(q).all() or abs(norm-1)>1e-4:
        raise ValueError('invalid world pose quaternion')
    w,qx,qy,qz=q/norm
    t=np.eye(4)
    t[:3,:3]=[[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)],
        [2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)],
        [2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]]
    t[:3,3]=[x,y,z]
    if not np.isfinite(t).all():raise ValueError('nonfinite world pose')
    return t


def base_xyyaw(pose):
    t=pose_matrix(pose)
    return (float(t[0,3]),float(t[1,3]),math.atan2(t[1,0],t[0,0]))


def action_route(phase):
    return ('NAV_TO_SOURCE' if 'nav_to_pick' in phase else 'NAV_TO_TARGET' if 'nav_to_place' in phase
        else 'PICK' if 'pick' in phase else 'PLACE' if 'place' in phase else 'VERIFY')


def sample_depth(sample, observation, camera):
    frame=sample['camera_frames'][camera]
    if frame.get('depth_contract_version')!='1.1' or frame.get('depth_source')!='isaac_dynamic_z_plus_static_mesh_raycast_v1':
        raise ValueError('unsupported collection depth composition; version explicitly required')
    if frame.get('depth_semantics')!='z_depth' or frame.get('depth_unit')!='millimeter' or frame.get('depth_scale')!=.001:
        raise ValueError('collection depth units/definition mismatch')
    k=observation['camera_calibration']['cameras'][camera]['effective_intrinsics']
    K=[[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]]
    pose=frame['camera_pose_world']
    if pose['control_tick']!=sample['camera_capture_step']:
        raise ValueError('camera pose/capture tick mismatch')
    world_camera=pose_matrix([*pose['position_xyz'],*pose['quaternion_wxyz']])
    world_base=pose_matrix(observation['base_pose'])
    transform=np.linalg.inv(world_base)@world_camera
    geometry=frame['depth_geometry']
    if geometry.get('pixel_coordinate')!='pixel_center_index_plus_half':
        raise ValueError('unknown collection depth pixel convention')
    calibration={'intrinsics':K,'camera_to_base':transform.tolist(),
        'world_camera':world_camera.tolist(),'world_base':world_base.tolist(),
        'geometry_source':frame['depth_source'],'mesh_world_sha256':geometry['mesh_world_sha256'],
        'pixel_offset':.5}
    return {'camera':camera,'path':frame['depth_raw_path'],'invalid_mask_source':'depth_gt_zero',
        'storage':'uint16_png','definition':'z_depth','unit_scale':.001,'pixel_offset':.5,
        'capture_time_s':frame['camera_capture_timestamp'],**calibration,
        'calibration_id':hashlib.sha256(json.dumps(calibration,sort_keys=True).encode()).hexdigest(),
        'source_contract_version':'1.1','rgb_geometry_alignment':geometry['geometric_alignment_to_3dgs']}


class CollectionEpisode(RawEpisode):
    def __init__(self, root):
        self.root=Path(root); source=json.loads((self.root/'manifest.json').read_text())
        if source.get('schema')!='raw-control-v2' or source.get('action_contract')!='joint6_gripper2_effective_and_base_twist3':
            raise ValueError('unsupported collection actuator contract')
        if source.get('time_profile')!={'control_dt_s':.02,'raw_hz':50}:
            raise ValueError('collection requires native 50Hz')
        self.source_manifest=source
        task=json.loads((self.root/'task.json').read_text())
        if task['training_action']['source_gripper_joint_range_m']!=[0.,.04]:
            raise ValueError('unrecognized physical gripper calibration')
        self.task={'original_instruction':task.get('base_instruction',task['instruction']),'initial_plan':[]}
        self.manifest={'schema':'raw-control-v2','action_contract':'joint-command-v2',
            'episode_uuid':source['episode_uuid'],'task_family_id':source['task_family_id'],
            'source_commit':source['source']['commit'],'resolved_config_sha256':source['resolved_task_sha256'],
            'assets_sha256':hashlib.sha256(json.dumps(source['source']['assets'],sort_keys=True).encode()).hexdigest(),
            'assistance_profile':'source_manifest_and_interventions_bound',
            'observation_contract':'named-q6-dq6-gripper-rgb-v2','joint_unit':'rad',
            'gripper_unit':'open_fraction','joint_names':list(ARM_NAMES),'importer_version':IMPORTER_VERSION,
            'source_manifest_sha256':digest(self.root/'manifest.json'),'planner_eligible':False,
            'nav_label_semantics':'future_measured_reference_not_goal_command'}
        self.gripper_size=1; self.order=list(range(6))
        source_observations=jsonl(self.root/'observations.jsonl')
        raw_obs={r['observation_id']:r for r in source_observations}
        if len(raw_obs)!=len(source_observations):raise ValueError('duplicate source observation ID')
        controls=jsonl(self.root/'control_effective_50hz.jsonl')
        by_command={r['command_id']:r for r in controls}
        if len(by_command)!=len(controls):raise ValueError('duplicate source command ID')
        samples={}; self.import_audit={'version':IMPORTER_VERSION,'source_depth_metadata_conflicts':0,
            'measured_gripper_out_of_range':0,'command_gripper_out_of_range':0,'command_gripper_out_of_range_ticks':[],'task_status':'unresolved_not_used_for_training'}
        # Explicit command identity join. Only allowlisted sample fields survive.
        for s in jsonl(self.root/'samples.jsonl'):
            witness=s['effective_command']; cid=witness['command_id']; c=by_command.get(cid)
            if c is None or any(witness[k]!=c[k] for k in ('reset_generation','control_tick','apply_sequence','channels')):
                raise ValueError('sample/control actuator identity mismatch')
            tick=c['control_tick']; stamp=c['sim_time_s']
            if tick in samples:raise ValueError('duplicate camera sample tick')
            if not s.get('camera_state_synchronized') or s['camera_capture_step']!=tick or abs(s['camera_capture_timestamp']-stamp)>1e-7:
                raise ValueError('sample camera/state is not synchronous')
            for name in ('front','wrist'):
                f=s['camera_frames'][name]
                if f['camera_capture_step']!=tick or abs(f['camera_capture_timestamp']-stamp)>1e-7:
                    raise ValueError('camera frame capture mismatch')
            samples[tick]={'camera_frames':s['camera_frames'],'camera_capture_step':tick,'command_id':cid}
        if set(samples)!={c['control_tick'] for c in controls}:
            raise ValueError('collection camera samples do not cover every physical control tick')
        task_events=jsonl(self.root/'task_events.jsonl')
        retry_ticks={e['control_tick']:e.get('metadata',{}).get('attempt_id','retry') for e in task_events
            if e.get('name')=='collection_recovery_replan'}
        interventions=jsonl(self.root/'interventions.jsonl')
        unknown_interventions={x['control_tick'] for x in interventions if x.get('kind') not in {'runtime_assistance_state','command_inhibition'}}
        self.import_audit['intervention_kinds']=sorted({str(x.get('kind')) for x in interventions})
        self.import_audit['unknown_intervention_ticks']=sorted(unknown_interventions)
        self.import_audit['declared_physical_assistance_events']=[x for x in interventions if x.get('kind')=='runtime_assistance_state']
        self.observations={}; normalized=[]; epoch=-1; last_route=None;attempt='initial'
        for c in controls:
            if any(c[k]!=source[k] for k in ('episode_uuid','task_family_id')):
                raise ValueError('source control episode/family mismatch')
            if c.get('schema')!='raw-control-v2' or not c.get('physics_advanced'):
                raise ValueError('source record is not a raw physical tick')
            if abs(raw_obs[c['post_observation_ref']]['sim_time_s']-c['post_sim_time_s'])>1e-7:
                raise ValueError('source post timestamp mismatch')
            route=action_route(c['controller_phase'])
            retry=c['control_tick'] in retry_ticks or c['requested'].get('source')=='collection_recovery_replan'
            if retry:attempt=retry_ticks.get(c['control_tick'],'pick_retry_1')
            if route!=last_route or retry:epoch+=1;last_route=route
            for ref in (c['observation_ref'],c['post_observation_ref']):
                o=raw_obs[ref]; names=o['joint_names']
                if len(set(names))!=len(names) or any(n not in names for n in (*ARM_NAMES,'arm_joint7','arm_joint8')):
                    raise ValueError('missing or duplicate named physical joints')
                if o['units']!={'arm_q':'rad','arm_dq':'rad/s','gripper':'m'}:
                    raise ValueError('source physical units missing')
                finger=np.array([o['q'][names.index(n)] for n in ('arm_joint7','arm_joint8')])
                fraction=float(finger.mean()/.04)
                # Physical sensor overshoot is retained explicitly, not misread as fraction.
                if not 0<=fraction<=1:self.import_audit['measured_gripper_out_of_range']+=1
                raw={'mission_id':source['episode_uuid'],'observation_id':ref,'time_s':o['sim_time_s'],
                    'q':[o['q'][names.index(n)] for n in ARM_NAMES],'dq':[o['dq'][names.index(n)] for n in ARM_NAMES],
                    'gripper':fraction,'measured_finger_m':finger.tolist(),
                    'measured_gripper_unclipped_fraction':fraction,'base_xyyaw':base_xyyaw(o['base_pose']),
                    'primitive':route,'active_task_id':f'offline-{epoch}-{route}','active_task_epoch':epoch,'offline_attempt_id':attempt,
                    'task_identity_source':'offline_teacher_phase_not_planner_condition','images':[],'image_times_s':[],
                    'action_query':False,'nav_reference_world_xyyaw':base_xyyaw(o['base_pose']),
                    'nav_reference_source':'future_measured_reference'}
                if ref==c['observation_ref'] and c['control_tick'] in samples:
                    tick=c['control_tick'];s=samples[tick];previous=samples.get(tick-10)
                    raw['depth']=[sample_depth(s,o,camera) for camera in ('front','wrist')]
                    if not o.get('depth',{}).get('valid',False):self.import_audit['source_depth_metadata_conflicts']+=1
                    if previous is not None:
                        frames=[previous['camera_frames']['front'],s['camera_frames']['front'],
                            previous['camera_frames']['wrist'],s['camera_frames']['wrist']]
                        raw['images']=[f['raw_image_path'] for f in frames]
                        raw['image_times_s']=[f['camera_capture_timestamp'] for f in frames]
                    raw['action_query']=route!='VERIFY'
                self.observations[ref]=(CurrentObservation.from_record(raw),raw)
            resolved={'command_id':c['command_id'],'actuator_mode':c['actuator_mode'],
                'limiting_report':c['limiting_report'],'source_class':{},'parent_command_ids':{}}
            for source_channel,target in [('arm','arm_target'),('gripper','gripper_target'),('base','base_twist')]:
                ch=c['channels'][source_channel]; value=ch['target']
                size={'arm':6,'gripper':2,'base':3}[source_channel]
                if value is not None and (np.asarray(value).shape!=(size,) or not np.isfinite(value).all()):
                    raise ValueError('malformed source actuator target')
                if source_channel=='gripper':
                    if c.get('gripper_actuated_mask')!=[True,False]:raise ValueError('unknown gripper actuator mask')
                    value=None if value is None else [value[0]/.04]
                resolved[target]=value
                resolved['source_class'][target]=ch['source_class'] if ch['valid'] else 'unknown'
                resolved['parent_command_ids'][target]=ch['parent_command_id']
            normalized.append({'schema':'raw-control-v2','identity':{k:self.manifest[k] for k in
                ('episode_uuid','task_family_id','source_commit','resolved_config_sha256')}|{'reset_generation':c['reset_generation']},
                'clock':{'control_tick':c['control_tick'],'sim_time_s':c['sim_time_s'],
                    'wall_monotonic_ns':c['first_apply_wall_monotonic_ns'],'apply_sequence':c['apply_sequence'],
                    'physics_advanced':c['physics_advanced']},'observation_ref':c['observation_ref'],
                'post_observation_ref':c['post_observation_ref'],'resolved':resolved,'requested':c['requested'],
                'intervention_refs':(['unclassified_intervention'] if c['control_tick'] in unknown_interventions else
                    ['declared_non_bc_perturbation'] if (c.get('action_metadata',{}).get('bc_eligible') is False
                    or c.get('action_metadata',{}).get('collection_intervention',{}).get('bc_eligible') is False) else []), 'source_channels':c['channels']})
        self.controls=self._controls(normalized)
        for control in self.controls:
            target=control['resolved']['gripper_target']
            if target is not None and not 0<=target[0]<=1:
                control['_action_exclusion_reason']='effective_gripper_outside_calibrated_range'
                self.import_audit['command_gripper_out_of_range']+=1
                self.import_audit['command_gripper_out_of_range_ticks'].append({'tick':control['clock']['control_tick'],'master_m':control['source_channels']['gripper']['target'][0],'fraction':target[0]})
        self.times=np.array([r['clock']['sim_time_s'] for r in self.controls])
        self.import_audit.update(controls=len(self.controls),camera_samples=len(samples),
            observation_rows=len(self.observations),source_file_hashes={n:digest(self.root/n) for n in
                ('manifest.json','task.json','observations.jsonl','control_effective_50hz.jsonl','samples.jsonl','interventions.jsonl','task_events.jsonl')})

    def task_view(self):
        # Source events are retained by hash; unresolved teacher skeleton is not a label.
        return []


def load_episode(root):
    m=json.loads((Path(root)/'manifest.json').read_text())
    return CollectionEpisode(root) if m.get('action_contract')=='joint6_gripper2_effective_and_base_twist3' else RawEpisode(root)
