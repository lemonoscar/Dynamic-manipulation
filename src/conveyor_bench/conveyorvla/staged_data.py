"""Read proposed raw-control-v2 episodes without guessing effective commands.

This reader is an explicit first implementation of docs/new_plan's wire format.
It never reads evaluator_truth.jsonl. Missing/unknown control quarantines chunks.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .contracts.action import TIME_PROFILES, encode_mani, finite_array
from .contracts.observation import CurrentObservation

ARM_NAMES = tuple(f'arm_joint{i}' for i in range(1, 7))


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


class RawEpisode:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if m.get('schema') != 'raw-control-v2' or m.get('action_contract') != 'joint-command-v2':
            raise ValueError('explicit raw-control-v2/joint-command-v2 required; 10D pose exports rejected')
        for field in ('episode_uuid', 'task_family_id', 'source_commit', 'resolved_config_sha256',
                      'assets_sha256', 'assistance_profile', 'observation_contract'):
            if not m.get(field):
                raise ValueError('missing release identity: '+field)
        if m['observation_contract'] != 'named-q6-dq6-gripper-rgb-v2':
            raise ValueError('unsupported observation contract')
        if m.get('joint_unit') != 'rad' or m.get('gripper_unit') not in {'open_fraction', 'rad'}:
            raise ValueError('joint/gripper units must be explicit')
        self.gripper_size = 1 if m['gripper_unit'] == 'open_fraction' else 2
        if self.gripper_size == 2:
            calibration = m['gripper_calibration']
            self.gripper_closed = finite_array(calibration['closed_joint_positions'], (2,), 'closed gripper')
            self.gripper_open = finite_array(calibration['open_joint_positions'], (2,), 'open gripper')
            self.gripper_pair_tolerance = float(calibration['pair_fraction_tolerance'])
            if ((self.gripper_open-self.gripper_closed) == 0).any() or not 0 < self.gripper_pair_tolerance <= .05:
                raise ValueError('invalid paired-gripper calibration')
        names = m.get('joint_names', [])
        if len(names) != 6 or set(names) != set(ARM_NAMES):
            raise ValueError('named arm joint mapping must be bijective')
        self.order = [names.index(n) for n in ARM_NAMES]
        self.observations = {}
        for raw in jsonl(self.root/'observations.jsonl'):
            raw['q'] = tuple(finite_array(raw['q'], (6,), 'q')[self.order])
            raw['dq'] = tuple(finite_array(raw['dq'], (6,), 'dq')[self.order])
            obs = CurrentObservation.from_record(raw)
            if obs.mission_id != m['episode_uuid'] or obs.observation_id in self.observations:
                raise ValueError('cross-episode/duplicate observation')
            self.observations[obs.observation_id] = (obs, raw)
        records = jsonl(self.root/'control_effective_50hz.jsonl')
        self.controls = self._controls(records)
        if not self.controls:
            raise ValueError('episode has no physical control ticks')
        self.times = np.array([r['clock']['sim_time_s'] for r in self.controls])
        self.task = json.loads((self.root/'task.json').read_text())

    def _controls(self, records):
        grouped = {}
        seen_keys = set()
        for r in records:
            if r.get('schema') != 'raw-control-v2':
                raise ValueError('mixed control schema')
            identity, clock = r['identity'], r['clock']
            if (identity['episode_uuid'] != self.manifest['episode_uuid'] or
                    identity['task_family_id'] != self.manifest['task_family_id']):
                raise ValueError('control identity mismatch')
            if identity['source_commit'] != self.manifest['source_commit'] or identity['resolved_config_sha256'] != self.manifest['resolved_config_sha256']:
                raise ValueError('source/config identity mismatch')
            key = (identity['reset_generation'], clock['control_tick'])
            full_key = (*key, clock['apply_sequence'])
            if full_key in seen_keys:
                raise ValueError('duplicate apply identity')
            seen_keys.add(full_key)
            if any(type(x) is not int or x < 0 for x in (*full_key, clock['wall_monotonic_ns'])) or not math.isfinite(clock['sim_time_s']):
                raise ValueError('invalid raw clock')
            grouped.setdefault(key, []).append(r)
        if len({key[0] for key in grouped}) > 1:
            raise ValueError('continuous episode cannot contain a mid-episode reset')
        result, commands, last_tick, last_time, last_wall = [], {}, None, None, None
        for key, applies in sorted(grouped.items()):
            if last_tick is not None and key[0] != last_tick[0]:
                raise ValueError('continuous episode cannot contain a mid-episode reset')
            ordered = sorted(applies, key=lambda r: r['clock']['apply_sequence'])
            advanced = [r for r in ordered if r['clock']['physics_advanced'] is True]
            if not advanced:
                continue  # event-only transition; never a control sample
            if len(advanced) != 1 or advanced[0] is not ordered[-1]:
                raise ValueError('only final apply can be consumed by a physical tick')
            r = advanced[0]
            t, wall = r['clock']['sim_time_s'], r['clock']['wall_monotonic_ns']
            if last_tick is not None and (key[1] != last_tick[1]+1 or not math.isclose(t-last_time, .02, abs_tol=1e-7)):
                raise ValueError('missing 50Hz control tick or false simulation clock')
            if last_wall is not None and wall <= last_wall:
                raise ValueError('nonmonotonic control wall clock')
            before, after = r['observation_ref'], r['post_observation_ref']
            if before not in self.observations or after not in self.observations:
                raise ValueError('missing pre/post observation reference')
            pre, post = self.observations[before][0], self.observations[after][0]
            if not math.isclose(pre.time_s, t, abs_tol=1e-7) or not math.isclose(post.time_s, t+.02, abs_tol=1e-7):
                raise ValueError('observation/apply/post timing mismatch')
            resolved = r['resolved']
            if not resolved.get('actuator_mode') or 'limiting_report' not in resolved:
                raise ValueError('effective controller mode/limiting report required')
            command_id = resolved['command_id']
            if (key[0], command_id) in commands:
                raise ValueError('effective command IDs must identify distinct applies')
            valid = {}
            for channel, size in (('arm_target', 6), ('gripper_target', self.gripper_size), ('base_twist', 3)):
                source = resolved['source_class'].get(channel, 'unknown')
                value = resolved.get(channel)
                valid[channel] = source in {'issued', 'held_valid'} and value is not None
                if source not in {'issued', 'held_valid', 'stale', 'unknown'}:
                    raise ValueError('unknown provenance class')
                if value is not None:
                    finite_array(value, (size,), channel)
                if source == 'held_valid':
                    parent = commands.get((key[0], resolved.get('parent_command_ids', {}).get(channel, resolved.get('parent_command_id'))))
                    valid[channel] = bool(valid[channel] and parent is not None and parent['_valid'][channel]
                        and np.array_equal(value, parent['resolved'][channel]))
            r['_valid'] = valid
            commands[(key[0], command_id)] = r
            result.append(r)
            last_tick, last_time, last_wall = key, t, wall
        return result

    def action_view(self, observation_id, *, profile='legacy_future_5hz'):
        obs, raw = self.observations[observation_id]
        # The 25Hz experiment changes Mani only; NAV retains ten 5Hz goals.
        selected_profile = ('causal_command_5hz' if profile == 'causal_command_25hz'
            and raw.get('primitive') in {'NAV_TO_SOURCE', 'NAV_TO_TARGET'} else profile)
        p = TIME_PROFILES[selected_profile]
        if raw.get('primitive') not in {'NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE'}:
            raise ValueError('action query needs an explicit active primitive')
        if not raw.get('active_task_id') or 'active_task_epoch' not in raw:
            raise ValueError('missing task/epoch action label identity')
        targets, sources = [], []
        # Quarantine the entire real interval, not just decimated label points.
        apply_end = obs.time_s + p.horizon*p.sample_period_s
        if self.times[-1]+p.control_period_s < apply_end-1e-7:
            raise ValueError('missing actual application tail; no invented terminal hold')
        end_time = max(p.target_times(obs.time_s)[-1],apply_end-p.control_period_s)
        channels = ('arm_target', 'gripper_target') if raw['primitive'] in {'PICK', 'PLACE'} else ('base_twist',)
        for control in self.controls:
            if obs.time_s-1e-8 <= control['clock']['sim_time_s'] <= end_time+1e-8:
                craw = self.observations[control['observation_ref']][1]
                if (craw.get('active_task_id'), craw.get('active_task_epoch')) != (raw['active_task_id'], raw['active_task_epoch']):
                    raise ValueError('future interval crosses task identity/epoch boundary')
                if 'gripper_target' in channels and control.get('_action_exclusion_reason'):
                    raise ValueError(control['_action_exclusion_reason']+' inside real interval')
                if not all(control['_valid'][c] for c in channels):
                    raise ValueError('quarantined stale/unknown effective command inside real interval')
                if control.get('intervention_refs'):
                    raise ValueError('intervened control interval is not action-only')
        for time_s in p.target_times(obs.time_s):
            idx = int(np.searchsorted(self.times, time_s+1e-8, side='right')-1)
            if idx < 0 or time_s >= self.times[idx]+.02-1e-8:
                raise ValueError('missing future control; no invented hold-tail')
            control = self.controls[idx]
            cobs, craw = self.observations[control['observation_ref']]
            if (craw.get('active_task_id'), craw.get('active_task_epoch')) != (raw['active_task_id'], raw['active_task_epoch']):
                raise ValueError('future crosses task identity/epoch boundary')
            if control.get('intervention_refs'):
                raise ValueError('intervened control interval is not an action-only transition')
            if raw['primitive'] in {'PICK', 'PLACE'}:
                if not all(control['_valid'][c] for c in ('arm_target', 'gripper_target')):
                    raise ValueError('quarantined stale/unknown effective Mani command')
                q = np.asarray(control['resolved']['arm_target'])[self.order]
                gripper = np.asarray(control['resolved']['gripper_target'])
                if self.gripper_size == 2:
                    fractions = (gripper-self.gripper_closed)/(self.gripper_open-self.gripper_closed)
                    if abs(fractions[0]-fractions[1]) > self.gripper_pair_tolerance:
                        raise ValueError('paired gripper commands disagree with calibration')
                    grip = float(fractions.mean())
                else:
                    grip = float(gripper[0])
                if not 0 <= grip <= 1:
                    raise ValueError('effective gripper outside calibrated range')
                targets.append([*q, grip])
            else:
                # NAV labels are local desired states, never inferred from base_twist.
                reference = self.manifest.get('nav_label_semantics') == 'future_measured_reference_not_goal_command'
                goal = craw.get('nav_reference_world_xyyaw') if reference else craw.get('nav_goal_world_xyyaw')
                if goal is None or (not reference and craw.get('nav_goal_source_class') not in {'issued', 'held_valid'}):
                    raise ValueError('NAV needs verified desired goal state labels')
                gx, gy, gyaw = finite_array(goal, (3,), 'world NAV goal')
                x, y, yaw = obs.base_xyyaw
                targets.append([math.cos(yaw)*(gx-x)+math.sin(yaw)*(gy-y),
                    -math.sin(yaw)*(gx-x)+math.cos(yaw)*(gy-y), (gyaw-yaw+math.pi)%(2*math.pi)-math.pi])
            sources.append(control['resolved']['command_id'])
        if not self.manifest.get('synthetic', False):
            for name in obs.images:
                path = (self.root/name).resolve()
                if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                    raise ValueError('missing or escaping RGB asset')
        # Required history: exact old/current capture identity, never future/nearest-neighbor.
        expected = (obs.time_s-p.history_span_s, obs.time_s)*2
        if len(obs.images) != 4 or not np.allclose(obs.image_times_s, expected, atol=1e-7):
            raise ValueError('missing or misaligned causal visual history')
        mani = raw['primitive'] in {'PICK', 'PLACE'}
        action = encode_mani(targets, obs.q) if mani else np.asarray(targets)
        depth = raw.get('depth')
        if depth is not None:
          for depth_item in (depth if isinstance(depth, list) else [depth]):
            if (depth_item.get('definition') not in {'z_depth', 'ray_range'} or not depth_item.get('calibration_id')
                    or not math.isfinite(depth_item['capture_time_s']) or depth_item['capture_time_s'] > obs.time_s
                    or abs(depth_item['capture_time_s']-obs.time_s) > 1e-7):
                raise ValueError('depth requires same-time capture and calibrated definition')
            for key in ('path', 'intrinsics', 'camera_to_base', 'unit_scale'):
                if key not in depth_item: raise ValueError('missing depth contract field: '+key)
        return {'schema': 'action-view-v2', 'depth': depth,
            'calibration_id': None if depth is None else ([d['calibration_id'] for d in depth] if isinstance(depth, list) else depth['calibration_id']), 'episode_uuid': self.manifest['episode_uuid'],
            'task_family_id': self.manifest['task_family_id'], 'observation_id': observation_id,
            'query_time_s': obs.time_s, 'time_profile': p.name,
            'first_target_offset_s': p.first_target_offset_s, 'target_times_s': p.target_times(obs.time_s).tolist(),
            'first_apply_time_s': obs.time_s, 'sample_period_s': p.sample_period_s,
            'active_task_id': raw['active_task_id'], 'active_task_epoch': raw['active_task_epoch'],
            'route': raw['primitive'], 'actions': action.tolist(), 'mani_state': list(obs.mani_state),
            'images': list(obs.images), 'image_times_s': list(obs.image_times_s),
            'source_command_ids': sources, 'target_kind': ['real_future']*p.horizon,
            'nav_label_semantics': self.manifest.get('nav_label_semantics', 'verified_desired_goal'),
            'training_eligible': True, 'synthetic': self.manifest.get('synthetic', False)}

    def task_view(self):
        from .task_memory import Evidence, PlannerEdit, Task, TaskMemory
        path = self.root/'task_events.jsonl'
        rows = []
        for item in jsonl(path):
            obs = self.observations[item['observation_id']][0]
            # All input state is replayed from earlier deployable observations;
            # future teacher/evaluator metadata is not included in the context.
            memory = TaskMemory(obs.mission_id, self.task['original_instruction'],
                tuple(Task(**t) for t in self.task['initial_plan']))
            for past in item.get('history', []):
                past_obs = self.observations[past['observation_id']][0]
                if past_obs.time_s >= obs.time_s:
                    raise ValueError('task history contains present/future information')
                feedback = tuple(Evidence(**e) for e in past.get('feedback', []))
                memory.observe(past_obs, feedback)
                if past.get('edit') is not None:
                    memory.apply(PlannerEdit.from_dict(past['edit']))
            feedback = tuple(Evidence(**e) for e in item.get('feedback', []))
            memory.observe(obs, feedback)
            context = memory.planner_context(obs, feedback)
            label = PlannerEdit.from_dict(item['label'])
            memory.apply(label)  # Validate schema and causal evidence before allowing CE supervision.
            rows.append({'schema': 'task-view-v2', 'observation_id': obs.observation_id,
                'task_family_id': self.manifest['task_family_id'], 'input': context,
                'label': item['label'], 'synthetic': self.manifest.get('synthetic', False)})
        return rows


def validate_family_splits(episodes, split_manifest):
    seen = {}
    for episode in episodes:
        family = episode.manifest['task_family_id']
        split = split_manifest[episode.manifest['episode_uuid']]
        if split not in {'train', 'validation', 'test'} or (family in seen and seen[family] != split):
            raise ValueError('task family crosses train/validation/test')
        seen[family] = split
    return seen
