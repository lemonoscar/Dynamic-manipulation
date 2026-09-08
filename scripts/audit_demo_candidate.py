"""Read-only closed-episode qualification; never feeds evaluator truth to policy."""
import collections
import hashlib
import json
from pathlib import Path


def audit(run):
    run=Path(run); ep=run/'runtime/episode_000000'
    summary=json.loads((ep/'summary.json').read_text())
    attempt=json.loads((run/'attempt.json').read_text())
    counts=collections.Counter(); errors=collections.Counter(); routes=collections.Counter()
    claims=[]; key_events=[]; tick_count=0; ticks_contiguous=True; tensor_ok=True; request_binding_ok=True
    requests={}; last_t=0.; stable_samples=0; stable_max=0; lift_peak=0.; closest=None
    initial=summary['initialization']['object_tensor_proof']['pose_wxyz']
    trace=ep/'trace.jsonl'
    for line in trace.open():
        x=json.loads(line); kind=x['event']; counts[kind]+=1
        if kind=='model_request':requests[x['request']['request_id']]=x
        elif kind=='model_response':
            v=x['response']; source=requests.get(v['request_id']); r=None if source is None else source['request']
            request_binding_ok &= bool(r is not None and v['identity']==r['identity'] and v['observation_id']==r['observation']['observation_id'] and v['query_time_s']==r['observation']['time_s'] and v['weights_sha256']==summary['model_identity']['weights_sha256'])
        elif kind=='control_step':
            tick_count+=1; state=x['state_after']; last_t=state['timestamp']; routes[x.get('route')]+=1
            ticks_contiguous &= x['control_step']==tick_count and state['step_index']==tick_count and abs(last_t-tick_count*.02)<1e-7
            obj=state['object_pose']; tcp=state['tcp_pose']; lift=obj[2]-initial[2]; lift_peak=max(lift_peak,lift)
            distance=sum((a-b)**2 for a,b in zip(obj[:3],tcp[:3]))**.5
            if x.get('route')=='PICK':closest=distance if closest is None else min(closest,distance)
            names=state['joint_names']; measured=state['joint_positions'][names.index('arm_joint7')]
            stable=bool(x.get('route') in ('PICK','NAV_TO_TARGET') and lift>=.04 and distance<=.08 and measured<=.02)
            stable_samples=stable_samples+1 if stable else 0; stable_max=max(stable_max,stable_samples)
        elif kind=='object_live_tensor':
            tensor_ok &= x.get('physics_handle_valid') is True and x.get('readback_error') is not None and x['readback_error']<=1e-6
        elif kind=='model_claim_transition':claims.append({'time_s':last_t,'active_task':x['context']['active_task']})
        if kind=='diagnostic_advisory':errors[x.get('reason','unknown')]+=1
        if kind in ('model_claim_transition','physical_pick_verified','geometry_hold_proxy_v2','physical_carry_verified','simulation_clock_started','simulation_clock_expired','contact_grasp_verified_v2'):
            key_events.append(x)
    evidence=summary.get('physics_evidence') or {}; ev2=evidence.get('evaluation_v2') or {}
    guards={
        'closed_normally_at_60s':attempt.get('status')=='exited' and attempt.get('returncode')==0 and summary.get('status')=='complete' and abs(summary.get('clock_elapsed_s',-1)-60)<1e-7,
        'continuous_3000_physics_ticks':tick_count==3000 and ticks_contiguous,
        'live_object_tensor_every_tick':counts['object_live_tensor']==3000 and tensor_ok,
        'model_response_identity_bound':counts['model_response']>0 and request_binding_ok,
        'scoring_valid':summary.get('scoring_valid') is True,
        'model_switched_into_pick':any(x['active_task']=='PICK' for x in claims),
        'legacy_pick_observed':evidence.get('pick_verified') is True,
        'stable_relative_grasp_geometry_1s':ev2.get('ever_geometry_hold_proxy') is True,
        'independent_lift_close_measured_finger_1s':stable_max>=51,
        'no_object_attachment_or_reset':evidence.get('profile')=='no_grasp_assist' and evidence.get('grasp_constraint_created') is False and evidence.get('mid_episode_object_resets')==0,
        'video_closed_300_frames':summary.get('video',{}).get('success') is True and summary['video'].get('frame_count')==300,
    }
    result={'schema':'train-demo-candidate-audit-v1','run':str(run),'seed':attempt.get('seed'),
        'candidate_pending_video_review':all(guards.values()),'gates':guards,
        'clock_s':last_t,'counts':dict(counts),'routes':dict(routes),'errors':dict(errors),
        'model_claims':claims,'peak_lift_m':lift_peak,'pick_closest_tcp_m':closest,
        'independent_stable_sample_count':stable_max,'independent_stable_duration_s':max(0,stable_max-1)*.02,
        'full_task_success':summary.get('full_task_success'),'strict_full_success':summary.get('strict_full_success'),
        'strict_contact':ev2.get('ever_contact_grasp_verified'),'declared_base_support_locks':evidence.get('manipulation_base_support_locks'),
        'video':summary.get('video'),'key_events':key_events,
        'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'summary_sha256':hashlib.sha256((ep/'summary.json').read_bytes()).hexdigest()}
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(audit(sys.argv[1]),indent=2))
