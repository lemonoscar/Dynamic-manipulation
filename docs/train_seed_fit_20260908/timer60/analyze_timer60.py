#!/usr/bin/env python3
"""Read-only CPU audit of the closed timer-only run. Writes derived reports only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finite(values, n):
    return isinstance(values, (list, tuple)) and len(values) >= n and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in values[:n])


def zero(command):
    return finite(command, 3) and all(abs(x) < 1e-12 for x in command[:3])


def movement(points, dimensions):
    valid = [tuple(p[:dimensions]) for p in points if finite(p, dimensions)]
    return {'valid_samples': len(valid), 'missing_or_invalid_samples': len(points)-len(valid),
            'first': valid[0] if valid else None, 'last': valid[-1] if valid else None,
            'net_displacement_m': math.dist(valid[0], valid[-1]) if valid else None,
            'sampled_path_length_m': sum(math.dist(a, b) for a, b in zip(valid, valid[1:])) if valid else None,
            'path_bridges_missing_samples': len(valid) != len(points)}


def pick_audit(ticks, summary):
    initial = summary.get('initialization', {}).get('object_tensor_proof', {}).get('pose_wxyz')
    selected = [r for r in ticks if r.get('route') == 'PICK']
    samples = []
    for r in selected:
        state=r.get('state_after', {}); obj=state.get('object_pose'); tcp=state.get('tcp_pose')
        names=state.get('joint_names', []); measured=state.get('joint_positions', [])
        meta=r.get('action', {}).get('metadata', {})
        joint=lambda name: measured[names.index(name)] if name in names and names.index(name)<len(measured) else None
        positions=meta.get('gripper_joint_positions')
        samples.append({'time_s':state.get('timestamp'),'step':state.get('step_index'),
            'tcp_object_distance_m':math.dist(tcp[:3],obj[:3]) if finite(tcp,3) and finite(obj,3) else None,
            'object_displacement_from_initial_m':math.dist(obj[:3],initial[:3]) if finite(obj,3) and finite(initial,3) else None,
            'object_height_delta_m':obj[2]-initial[2] if finite(obj,3) and finite(initial,3) else None,
            'gripper_requested_open_fraction':meta.get('gripper_open_fraction_requested'),
            'command_joint7_target_m':positions[0] if finite(positions,1) else None,
            'measured_joint7_m':joint('arm_joint7'),'measured_joint8_m':joint('arm_joint8'),
            'action_source':r.get('action',{}).get('source')})
    def range_for(key):
        valid=[v[key] for v in samples if isinstance(v.get(key),(int,float)) and math.isfinite(v[key])]
        return {'min':min(valid) if valid else None,'max':max(valid) if valid else None,'valid_count':len(valid)}
    close=[v for v in samples if isinstance(v['tcp_object_distance_m'],(int,float))]
    claims=summary.get('model_claim_events', [])
    changes=[]
    for event in claims:
        stamp=event.get('time_s'); before=[r for r in ticks if r.get('state_after',{}).get('timestamp',math.inf)<=stamp]
        prior=before[-1] if before else None
        obj=prior['state_after'].get('object_pose') if prior else None
        changes.append({'model_claim_event':event,'last_observed_state_time_s':prior['state_after']['timestamp'] if prior else None,
            'last_observed_route':prior.get('route') if prior else None,
            'object_height_delta_m':obj[2]-initial[2] if finite(obj,3) and finite(initial,3) else None})
    all_obj=[r['state_after'].get('object_pose') for r in ticks]
    displacement=[math.dist(x[:3],initial[:3]) for x in all_obj if finite(x,3) and finite(initial,3)]
    height=[x[2]-initial[2] for x in all_obj if finite(x,3) and finite(initial,3)]
    return {'phase_scope':'actual physical ticks whose route is PICK; model query ADVANCE does not establish completion',
        'tick_count':len(samples),'first_time_s':samples[0]['time_s'] if samples else None,'last_time_s':samples[-1]['time_s'] if samples else None,
        'closest_tcp_sample':min(close,key=lambda v:v['tcp_object_distance_m']) if close else None,
        'ranges':{k:range_for(k) for k in ['tcp_object_distance_m','object_displacement_from_initial_m','object_height_delta_m','gripper_requested_open_fraction','command_joint7_target_m','measured_joint7_m','measured_joint8_m']},
        'last_pick_sample':samples[-1] if samples else None,
        'model_manipulation_command_count':sum(v['action_source']=='joint_trajectory_pick' for v in samples),
        'model_manipulation_joint7_target_histogram_m':dict(Counter(str(round(v['command_joint7_target_m'],8)) for v in samples if v['action_source']=='joint_trajectory_pick' and isinstance(v['command_joint7_target_m'],(int,float)))),
        'last_model_command_at_or_below_half_open':next((v for v in reversed(samples) if v['action_source']=='joint_trajectory_pick' and isinstance(v['gripper_requested_open_fraction'],(int,float)) and v['gripper_requested_open_fraction']<=.5),None),
        'model_claim_boundaries':changes,
        'whole_episode_max_object_displacement_m':max(displacement) if displacement else None,
        'whole_episode_max_object_height_delta_m':max(height) if height else None,
        'positive_peak_lift_including_initialized_pose_m':max([0.]+height) if height else None,
        'joint_semantics':'joint7 active command/state in metres; joint8 reported separately; no target averaging',
        'interpretation':'If object never rises and pick/carry scores stay false before model-claimed ADVANCE, the trace supports transition into NAV_TO_TARGET without lifting. It does not establish contact absence; strict contact remains unknown.'}


def audit(run):
    episode = run/'runtime/episode_000000'
    paths = {'trace': episode/'trace.jsonl', 'summary': episode/'summary.json', 'attempt': run/'attempt.json'}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit('Required closed-run files missing: ' + ', '.join(missing))
    summary, attempt = (json.loads(paths[k].read_text()) for k in ('summary', 'attempt'))
    rows, malformed = [], []
    with paths['trace'].open() as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
                row['_line'] = line_no
                # Policy RGB payloads remain in source trace; audit does not duplicate images.
                row.pop('head_images', None); row.pop('wrist_images', None)
                rows.append(row)
            except (ValueError, TypeError) as e:
                malformed.append({'line': line_no, 'error': str(e)})
    events = Counter(r.get('event', 'UNKNOWN') for r in rows)
    ticks = [r for r in rows if r.get('event') == 'control_step']
    states = [r.get('state_after', {}) for r in ticks]
    times = [s.get('timestamp') for s in states]
    bad_ticks = []
    for i, (tick, state) in enumerate(zip(ticks, states)):
        if tick.get('control_step') != i+1 or state.get('step_index') != tick.get('state_before_step', -2)+1:
            bad_ticks.append({'line': tick['_line'], 'reason': 'tick_counter_or_before_after_discontinuity'})
        if i and (state.get('step_index') != states[i-1].get('step_index', -2)+1 or
                  not isinstance(times[i], (int,float)) or not isinstance(times[i-1], (int,float)) or
                  not math.isclose(times[i]-times[i-1], .02, abs_tol=1e-8)):
            bad_ticks.append({'line': tick['_line'], 'reason': 'state_or_timestamp_gap'})
    clocks = [r for r in rows if r.get('event') == 'simulation_clock_started']
    start = clocks[0].get('start_s') if clocks else summary.get('clock_start_s')
    deadline = clocks[0].get('deadline_s') if clocks else summary.get('clock_deadline_s')
    last = times[-1] if times else None
    expected = round((deadline-start)/.02) if isinstance(start,(int,float)) and isinstance(deadline,(int,float)) else None
    clock_pass = (isinstance(last,(int,float)) and isinstance(start,(int,float)) and isinstance(deadline,(int,float))
                  and math.isclose(deadline-start,60.,abs_tol=1e-8)
                  and math.isclose(last,deadline,abs_tol=1e-8) and len(ticks)==expected
                  and not bad_ticks and not malformed
                  and math.isclose(times[0],start+.02,abs_tol=1e-8))
    requests = [r for r in rows if r.get('event') == 'model_request']
    responses = [r for r in rows if r.get('event') == 'model_response']
    request_by_id = {r.get('request',{}).get('request_id'): r['request'] for r in requests}
    joins, transitions = [], Counter()
    for r in responses:
        response = r.get('response', {})
        rid = response.get('request_id'); request = request_by_id.get(rid)
        obs = request.get('observation', {}) if request else {}
        valid = bool(request and response.get('identity') == request.get('identity')
                     and response.get('observation_id') == obs.get('observation_id')
                     and response.get('query_time_s') == obs.get('time_s')
                     and response.get('time_profile') == request.get('time_profile'))
        proposal = (response.get('transition') or {}).get('proposal', {})
        transitions[proposal.get('operation', 'NONE')] += 1
        joins.append({'line':r['_line'],'request_id':rid,'binding_matches':valid,
                      'query_time_s':response.get('query_time_s'),'route':request.get('task',{}).get('primitive') if request else None,
                      'transition_proposal':proposal,'rtc_applied':response.get('rtc_applied'),
                      'client_roundtrip_s':r.get('client_roundtrip_s'),'server_wall_s':response.get('server_wall_s')})
    controls = [r for r in rows if r.get('event') == 'navigation_control']
    rejects = [r for r in rows if r.get('event') == 'navigation_control_rejected']
    zero_details = []
    for r in controls:
        if zero(r.get('command')):
            preceding = next((x for x in reversed(rejects) if x['_line'] < r['_line'] and x.get('reason') == r.get('reason')), None)
            trace = r.get('trace', {})
            raw = trace.get('raw_dwa_command')
            debug = None
            # Rejection is emitted immediately before its navigation_control.
            if preceding and preceding['_line'] == r['_line']-1:
                debug = preceding.get('dwa_debug')
                if raw is None and isinstance(debug,dict): raw=debug.get('command')
            introduced = None if not finite(raw,3) else not zero(raw)
            zero_details.append({'line':r['_line'],'reason':r.get('reason'),
                'guarded_command':r.get('command'),'raw_dwa_command':raw,
                'guard_changed_nonzero_to_zero':introduced,'trace':trace,'dwa_debug':debug})
    advisories = [r for r in rows if r.get('event') == 'diagnostic_advisory']
    invalid_evidence = [r for r in advisories if r.get('scoring_valid') is False]
    proofs = [r for r in rows if r.get('event') == 'object_live_tensor']
    init = summary.get('initialization',{})
    initial_object = init.get('object_tensor_proof',{}).get('pose_wxyz')
    object_points = [s.get('object_pose') for s in states]
    object_motion = movement(([initial_object] if finite(initial_object,3) else [])+object_points,3)
    object_motion['origin'] = 'initialized_live_tensor' if finite(initial_object,3) else 'first_logged_post_step_pose'
    scoring_valid = False if invalid_evidence else summary.get('scoring_valid')
    # Never infer missing positive success or turn timed completion into task success.
    full_success = summary.get('full_task_success') if scoring_valid is not False else None
    only_clock = (summary.get('termination_reason')=='simulation_clock_60s'
                  and summary.get('status')=='complete' and summary.get('failure_reason') is None
                  and clock_pass and attempt.get('returncode')==0)
    return {'schema':'timer60-independent-cpu-audit-v1','generated_at':datetime.now(timezone.utc).isoformat(),
        'run_dir':str(run.resolve()),'input_sha256':{k:digest(p) for k,p in paths.items()},
        'event_counts':dict(events),'malformed_lines':malformed,
        'clock':{'start_s':start,'deadline_s':deadline,'last_logged_physics_s':last,
                 'elapsed_s':last-start if isinstance(last,(int,float)) and isinstance(start,(int,float)) else None,
                 'expected_ticks':expected,'recorded_ticks':len(ticks),'tick_errors':bad_ticks,
                 'sixty_seconds_complete_and_continuous':clock_pass,'normal_exit_only_clock_verified':only_clock},
        'process':attempt,'summary_status':{k:summary.get(k) for k in ('status','failure_reason','termination_reason','clock_elapsed_s','control_steps','model_queries','timer_only','runner_sha256')},
        'model':{'requests':len(requests),'responses':len(responses),'duplicate_request_ids':len(requests)-len(request_by_id),
                 'unanswered_request_ids':sorted(set(request_by_id)-{j['request_id'] for j in joins},key=str),
                 'binding_mismatches':[j for j in joins if not j['binding_matches']],
                 'request_routes':dict(Counter(r['request'].get('task',{}).get('primitive','UNKNOWN') for r in requests)),
                 'transition_operation_counts':dict(transitions),'query_details':joins,
                 'applied_model_claim_transitions':[r for r in rows if r.get('event')=='model_claim_transition']},
        'navigation':{'control_count':len(controls),'zero_control_count':len(zero_details),
                      'zero_reasons':dict(Counter(str(z['reason']) for z in zero_details)),
                      'zero_raw_and_debug':zero_details,'all_rejection_events':rejects},
        'physical':{'physical_tick_routes':dict(Counter(str(t.get('route')) for t in ticks)),
                    'base_xy':{**movement([s.get('robot_root_pose') for s in states],2),'origin':'first_logged_post_step_pose; initialization-to-first-tick movement excluded'},
                    'object_xyz':object_motion,'object_live_tensor_samples':len(proofs),
                    'object_live_tensor_coverage_complete':len(proofs)==len(ticks) and all(p.get('physics_handle_valid') is True and isinstance(p.get('readback_error'),(int,float)) and p['readback_error']<=1e-5 for p in proofs),
                    'unique_object_velocities':len({tuple(s['object_velocity']) for s in states if finite(s.get('object_velocity'),6)}),
                    'initialization_object_proof':init.get('object_tensor_proof'),'scoring_valid':scoring_valid,
                    'evidence_invalid_events':invalid_evidence,'advisory_reasons':dict(Counter(r.get('reason','UNKNOWN') for r in advisories))},
        'pick_phase_audit':pick_audit(ticks,summary),
        'task_outcome':{'attempt_scope':'canonical full episode, bounded to 60 simulation seconds; visited phases reported separately',
                        'initial_plan_context':requests[0]['request'].get('plan_context') if requests else None,
                        'final_active_task':summary.get('final_active_task'),'visited_model_routes':summary.get('state_trace'),
                        'full_task_success':full_success,'summary_full_task_success_raw':summary.get('full_task_success'),
                        'strict_full_success':summary.get('strict_full_success'),'planner_FINISH':summary.get('planner_FINISH'),
                        'deployment_gate_passed':summary.get('deployment_gate_passed'),
                        'physics_evidence':summary.get('physics_evidence'),
                        'diagnostic_success_raw':summary.get('success'),'diagnostic_success_semantics':summary.get('success_semantics'),
                        'interpretation':'A clock-complete run is not task success; unvisited PICK/PLACE cannot be scored as attempted subtask failures; missing strict contact evidence stays unknown.'}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=Path(__file__).parent/'results/physical_timer60')
    p.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    args=p.parse_args(); result=audit(args.run_dir);args.output_dir.mkdir(parents=True,exist_ok=True)
    out=args.output_dir/'TIMER60_AUDIT.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    c=result['clock'];m=result['model'];t=result['task_outcome'];n=result['navigation'];physical=result['physical']
    md=(f"# 60 秒诊断独立审计\n\n计时区间：{c['start_s']} → {c['last_logged_physics_s']} s；记录 {c['recorded_ticks']} 个物理 tick。"
        f"连续且完整 60 秒：{c['sixty_seconds_complete_and_continuous']}；仅由计时正常退出：{c['normal_exit_only_clock_verified']}。\n\n"
        f"模型请求/响应：{m['requests']}/{m['responses']}；任务分布：{m['request_routes']}；转换：{m['transition_operation_counts']}。"
        f"最后任务：{t['final_active_task']}。这是完整 episode 起点的有界尝试；实际进入的阶段单独列出。\n\n"
        f"导航零控制 {n['zero_control_count']} 次：{n['zero_reasons']}。原始 DWA 命令与 debug 逐条保存在 JSON；缺失值不推断。\n\n"
        f"底盘观测净位移/路径长度：{physical['base_xy']['net_displacement_m']} / {physical['base_xy']['sampled_path_length_m']} m；"
        f"物体观测净位移：{physical['object_xyz']['net_displacement_m']} m；评分证据有效性：{physical['scoring_valid']}。\n\n"
        f"完整任务评分：{t['full_task_success']}；严格成功：{t['strict_full_success']}。计时完成不代表搬运成功；"
        "未进入阶段不视为已执行失败，未知接触证据保持未知。详细输入哈希、异常、控制与任务证据见 TIMER60_AUDIT.json。\n")
    pa=result['pick_phase_audit']
    md += (f"\nPICK 实际物理区间：{pa['first_time_s']}–{pa['last_time_s']} s，共 {pa['tick_count']} ticks；"
           f"最近 TCP–物体距离：{pa['ranges']['tcp_object_distance_m']['min']} m。"
           f"全程物体最大位移：{pa['whole_episode_max_object_displacement_m']} m，正向峰值抬升：{pa['positive_peak_lift_including_initialized_pose_m']} m。"
           f"PICK 主夹爪目标范围：{pa['ranges']['command_joint7_target_m']}；实测 joint7 范围：{pa['ranges']['measured_joint7_m']}。\n")
    (args.output_dir/'TIMER60_AUDIT.md').write_text(md)
    print(json.dumps({'audit':str(out),'clock':c,'requests':m['requests'],'responses':m['responses'],
        'routes':m['request_routes'],'final_task':t['final_active_task'],'full_task_success':t['full_task_success']},ensure_ascii=False))


if __name__=='__main__':main()
