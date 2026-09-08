#!/usr/bin/env python3
"""Derive RGB partial-action and offline transition labels; never read evaluator truth."""
import argparse
import collections
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    from conveyor_bench.conveyorvla.full_episode_data import STAGES, task_context, transition_prompt, transition_response
except ModuleNotFoundError:
    # Local draft/fixture use; the production copy imports the shared package above.
    from full_episode_data import STAGES, task_context, transition_prompt, transition_response


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_prefix(ep, raw, query_time, horizon=10):
    """Each scored point requires its entire 0.2s physical application bin."""
    channels = ('arm_target', 'gripper_target') if raw['primitive'] in {'PICK', 'PLACE'} else ('base_twist',)
    start = round(query_time / .02)
    by_tick = {c['clock']['control_tick']: c for c in ep.controls}
    identity = (raw['active_task_id'], raw['active_task_epoch'])
    valid, reason = [], None
    for point in range(horizon):
        if reason:
            valid.append(False)
            continue
        for tick in range(start + 10 * point, start + 10 * (point + 1)):
            c = by_tick.get(tick)
            if c is None or abs(c['clock']['sim_time_s'] - tick * .02) > 1e-7:
                reason = 'missing_native_application_interval'
                break
            state = ep.observations[c['observation_ref']][1]
            if (state['active_task_id'], state['active_task_epoch']) != identity:
                reason = 'task_identity_or_epoch_boundary'
                break
            if not all(c['_valid'][key] for key in channels):
                reason = 'unknown_or_stale_effective_command'
                break
            if c.get('intervention_refs'):
                reason = 'non_bc_or_unclassified_intervention'
                break
            if 'gripper_target' in channels and c.get('_action_exclusion_reason'):
                reason = c['_action_exclusion_reason']
                break
        valid.append(reason is None)
    return valid, reason, by_tick


def rgb_binding(ep, oid):
    obs, raw = ep.observations[oid]
    expected = [obs.time_s - .2, obs.time_s] * 2
    if len(obs.images) != 4 or len(obs.image_times_s) != 4 or any(abs(a-b) > 1e-7 for a,b in zip(expected,obs.image_times_s)):
        raise ValueError('missing_exact_causal_four_rgb')
    for name in obs.images:
        p = (ep.root / name).resolve()
        if not p.is_relative_to(ep.root.resolve()) or not p.is_file():
            raise ValueError('missing_or_escaping_rgb')
    return obs, raw


def partial_action(ep, oid):
    obs, raw = rgb_binding(ep, oid)
    mask, reason, controls = native_prefix(ep, raw, obs.time_s)
    if not any(mask):
        raise ValueError('no_valid_full_application_bin:' + str(reason))
    mani = raw['primitive'] in {'PICK','PLACE'}
    actions, sources = [], []
    for i, valid in enumerate(mask):
        if not valid:
            actions.append([0.] * (7 if mani else 3)); sources.append(None)
            continue
        c = controls[round(obs.time_s/.02) + i*10]
        if mani:
            target = c['resolved']['arm_target']
            grip = c['resolved']['gripper_target'][0]
            if not 0 <= grip <= 1:
                raise ValueError('effective_gripper_outside_calibrated_range')
            action = [float(q)-float(anchor) for q,anchor in zip(target,obs.q)] + [float(grip)]
        else:
            state = ep.observations[c['observation_ref']][1]
            gx,gy,gyaw = state['nav_reference_world_xyyaw']
            x,y,yaw = obs.base_xyyaw
            action = [math.cos(yaw)*(gx-x)+math.sin(yaw)*(gy-y),
                      -math.sin(yaw)*(gx-x)+math.cos(yaw)*(gy-y), (gyaw-yaw+math.pi)%(2*math.pi)-math.pi]
        if not all(math.isfinite(v) for v in action):
            raise ValueError('nonfinite_action')
        actions.append(action); sources.append(c['resolved']['command_id'])
    return dict(schema='action-view-v3-partial-rgb', episode_uuid=ep.manifest['episode_uuid'],
        task_family_id=ep.manifest['task_family_id'], observation_id=oid, query_time_s=obs.time_s,
        time_profile='causal_command_5hz', first_target_offset_s=0., first_apply_time_s=obs.time_s,
        target_times_s=[obs.time_s+i*.2 for i in range(10)], sample_period_s=.2,
        active_task_id=raw['active_task_id'], active_task_epoch=raw['active_task_epoch'],
        route=raw['primitive'], actions=actions, action_valid_mask=mask,
        action_valid_duration_s=[.2 if v else 0. for v in mask], partial_window_reason=reason,
        mani_state=list(obs.mani_state), images=list(obs.images), image_times_s=list(obs.image_times_s),
        source_command_ids=sources, target_kind=['real_native_supported' if v else 'padding_not_command' for v in mask],
        nav_label_semantics='future_measured_reference_not_goal_command', depth=None, calibration_id=None,
        input_modalities='rgb', synthetic=False, training_eligible=True)


EVENTS = ('nav_to_pick_success','pick_success','nav_to_place_success')


def transition_rows(ep, stride=20):
    events = [json.loads(line) for line in (ep.root/'task_events.jsonl').open()]
    source_obs = {o['observation_id']:o for o in map(json.loads,(ep.root/'observations.jsonl').open())}
    queries = []
    for oid,(obs,raw) in ep.observations.items():
        if not oid.endswith(':pre') or round(obs.time_s/.02)%stride:
            continue
        try: rgb_binding(ep,oid)
        except ValueError: continue
        o = source_obs[oid]
        queries.append((obs.time_s, o['frozen_wall_monotonic_ns'],oid))
    queries.sort()
    rows, missing, completed = [], [], []
    for index,name in enumerate(EVENTS):
        matches = [e for e in events if e.get('name')==name]
        if len(matches)!=1:
            missing.append({'event':name,'reason':'no_unique_success_event'})
            break
        event = matches[0]; t=event['sim_time_s']; wall=event['wall_monotonic_ns']
        if index==2 and not any(e.get('name')=='carry_control_success' and e.get('sim_time_s')==t for e in events):
            missing.append({'event':name,'reason':'missing_carry_success_pair'})
            break
        before=[q for q in queries if q[0]<t and (not completed or q[0]>completed[-1]['time_s'])]
        after=[q for q in queries if q[0]>=t and q[1]>=wall]
        if not before or not after or t-before[-1][0]>.4+1e-7 or after[0][0]-t>.4+1e-7:
            missing.append({'event':name,'reason':'missing_nearby_causal_rgb_pair'})
            break
        previous=ep.observations[before[-1][2]][1]
        if previous['primitive']!=STAGES[index]:
            missing.append({'event':name,'reason':'pre_event_active_task_mismatch'})
            break
        # The same prior memory is used by CONTINUE and ADVANCE; no future task leak.
        context=task_context(STAGES[index],[e['task'] for e in completed])
        memory=dict(mission_id=ep.manifest['episode_uuid'],plan_version=0,
            active_task_epoch=previous['active_task_epoch'], active_task_id=previous['active_task_id'],
            task_context=context)
        memory_id=hashlib.sha256(json.dumps(memory,sort_keys=True).encode()).hexdigest()
        for label,q in [('CONTINUE',before[-1]),('ADVANCE',after[0])]:
            obs,raw=rgb_binding(ep,q[2])
            rows.append(dict(schema='offline-rgb-transition-v1',episode_uuid=ep.manifest['episode_uuid'],
                task_family_id=ep.manifest['task_family_id'],observation_id=q[2],query_time_s=q[0],
                parent_memory_id=memory_id,memory_provenance='offline_teacher_forced_prior_memory_not_model_output',
                parent_memory_identity=memory,
                input=dict(original_instruction=ep.task['original_instruction'],task_context=context,
                    images=list(obs.images),image_times_s=list(obs.image_times_s)),
                label=dict(operation=label,next_task=STAGES[index+1] if label=='ADVANCE' else STAGES[index]),
                label_evidence=dict(source='offline_assisted_teacher_success_event',event_name=name,
                    event_time_s=t,event_wall_monotonic_ns=wall,query_frozen_wall_monotonic_ns=q[1],
                    source_events_sha256=sha(ep.root/'task_events.jsonl')),
                transition_training_eligible=True,deployment_feedback_available=False,ordinary_planner_eligibility=False))
        completed.append({'task':STAGES[index],'time_s':t})
    # Terminal events have no later exact-current four-RGB query in this recording.
    missing.extend({'event':e.get('name'),'reason':'finish_not_generated_requires_post_success_rgb_contract'}
                   for e in events if e.get('name')=='episode_success')
    return rows,missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--episodes',nargs='+',type=Path,required=True)
    p.add_argument('--split-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-release',type=Path,required=True,help='Existing qualified release; frozen train-only normalizer is copied, never refit')
    p.add_argument('--query-stride-ticks',type=int,default=20)
    a=p.parse_args();sys.path.insert(0,str(a.repo/'src'))
    from conveyor_bench.conveyorvla.collection_importer import CollectionEpisode
    from conveyor_bench.conveyorvla.staged_data import validate_family_splits
    if a.output.exists():raise FileExistsError(a.output)
    splits=json.loads(a.split_manifest.read_text());families={};actions=[];tasks=[];quarantine=[];queries=[];source_files={}
    base_manifest=json.loads((a.base_release/'manifest.json').read_text())
    normalizer=json.loads((a.base_release/'normalization.json').read_text())
    if base_manifest['split_manifest_sha256']!=sha(a.split_manifest):raise ValueError('base_family_split_changed')
    if base_manifest['normalizer_id']!=normalizer['normalizer_id']:raise ValueError('base_normalizer_identity_mismatch')
    for path in a.episodes:
        ep=CollectionEpisode(path,modalities='rgb')
        source_files.update({str(path.resolve()/name):value for name,value in ep.import_audit['source_file_hashes'].items()})
        family=ep.manifest['task_family_id'];split=splits[ep.manifest['episode_uuid']]
        validate_family_splits([ep],splits)
        if family in families and families[family]!=split:raise ValueError('cross_split_family')
        families[family]=split
        source_obs={o['observation_id']:o for o in map(json.loads,(ep.root/'observations.jsonl').open())}
        events=[json.loads(line) for line in (ep.root/'task_events.jsonl').open()]
        for oid,(obs,raw) in ep.observations.items():
            if not oid.endswith(':pre') or round(obs.time_s/.02)%a.query_stride_ticks:continue
            identity=dict(episode_uuid=ep.manifest['episode_uuid'],observation_id=oid,query_time_s=obs.time_s,split=split)
            try:
                if not raw.get('action_query'):raise ValueError('no_active_action_primitive')
                row=partial_action(ep,oid);row.update(split=split,episode_root=str(path.resolve()));actions.append(row)
                committed=[STAGES[i] for i,name in enumerate(EVENTS[:STAGES.index(row['route'])])
                    if any(e.get('name')==name and e.get('sim_time_s',float('inf'))<=obs.time_s
                        and e.get('wall_monotonic_ns',float('inf'))<=source_obs[oid]['frozen_wall_monotonic_ns'] for e in events)]
                row['task_context']=task_context(row['route'],committed)
                row['original_instruction']=ep.task['original_instruction']
                row['memory_provenance']='offline_committed_teacher_history_not_model_output'
                row['parent_memory_identity']=dict(mission_id=ep.manifest['episode_uuid'],plan_version=0,
                    active_task_id=row['active_task_id'],active_task_epoch=row['active_task_epoch'],task_context=row['task_context'])
                row['parent_memory_id']=hashlib.sha256(json.dumps(row['parent_memory_identity'],sort_keys=True).encode()).hexdigest()
                queries.append(identity|dict(action_eligible=True,valid_points=sum(row['action_valid_mask'])))
            except ValueError as error:
                quarantine.append(identity|dict(reason=str(error)));queries.append(identity|dict(action_eligible=False,reason=str(error)))
        labels,missing=transition_rows(ep,a.query_stride_ticks)
        for row in labels:row.update(split=split,episode_root=str(path.resolve()))
        tasks.extend(labels)
        quarantine.extend(dict(episode_uuid=ep.manifest['episode_uuid'],view='transition',**m) for m in missing)
    a.output.mkdir()
    for name,rows in [('actions.jsonl',actions),('transitions.jsonl',tasks),('queries.jsonl',queries),('quarantine.jsonl',quarantine)]:
        (a.output/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (a.output/'normalization.json').write_bytes((a.base_release/'normalization.json').read_bytes())
    manifest=dict(schema='full-episode-rgb-derived-v1',time_profile='causal_command_5hz',input_modalities='rgb',
        action_rows=len(actions),partial_rows=sum(not all(r['action_valid_mask']) for r in actions),
        transition_rows=len(tasks),transition_counts=dict(collections.Counter(r['label']['operation'] for r in tasks)),
        action_split_route=dict(collections.Counter(r['split']+':'+r['route'] for r in actions)),
        source_split_sha256=sha(a.split_manifest),files={f.name:sha(f) for f in a.output.iterdir()},
        normalizer_id=normalizer['normalizer_id'],normalizer_source=str((a.base_release/'normalization.json').resolve()),
        base_release_sha256=sha(a.base_release/'manifest.json'),normalizer_refit=False,
        source_files=source_files,ordinary_planner_eligibility=False,
        transition_training_eligible=True,prompt_module_sha256=sha(inspect.getsourcefile(transition_prompt)),
        dataset_builder_sha256=sha(__file__),teacher_forced_transition_labels_not_deployment_feedback=True,finish_rows=0)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
