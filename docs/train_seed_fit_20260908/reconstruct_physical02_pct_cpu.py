"""Read-only CPU reconstruction of physical02 PCT outputs; no model or evaluator inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RUN = Path('/diff/wallx_workspace/dzb/integration_runs/train_seed_fit_20260908_v1/physical_02_full_off_live')
argv = json.loads((RUN / 'command.json').read_text())['argv']
arg = lambda flag: argv[argv.index(flag) + 1]
reference = Path(arg('--reference-root')).resolve()
sys.path.insert(0, str(reference))
from source.navigation.pct_adapter import PCTPlannerClient, PCTPlannerConfig

config = PCTPlannerConfig(enabled=True,
    tomogram_path=Path(arg('--pct-tomogram-path')),
    walkable_path=Path(arg('--pct-walkable-path')),
    collision_ply_path=Path(arg('--pct-collision-ply-path')),
    coord_mode='identity', fallback_to_astar=False)
os.environ.update(PCTPlannerClient(config)._server_env())
spec = importlib.util.spec_from_file_location('pct_grid_reconstruct', arg('--pct-server-script'))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
state = module._load_state()
rows = list(map(json.loads, (RUN / 'runtime/episode_000000/trace.jsonl').open()))
result = {'evidence_kind': 'cpu_reconstruction_not_original_failure_trace',
    'run': str(RUN), 'source_head': '691a14444d22d18ef2c3c742ebbfab2062732b23',
    'reference_head': '388b6818f4c605a707d13c519fbb58b1d07acd92',
    'inputs': 'logged navigation_proposal goal_world and preceding measured robot_root_pose; bound PCT assets',
    'configuration': 'Actual CLI fixes identity coordinates and three asset paths; other PCT fields match pinned CLI/NavigationSettings/PCTPlannerConfig defaults.',
    'map': {'resolution': state.resolution, 'center': state.center.tolist(),
        'shape': list(state.traversability.shape), 'hard_obstacle_count': int(state.hard_obstacle_mask.sum())},
    'queries': [], 'sha256': {}}
for i, row in enumerate(rows):
    if row.get('event') != 'navigation_proposal':
        continue
    previous = next(x for x in rows[:i][::-1] if x.get('event') == 'control_step')
    pose = previous['state_after']['robot_root_pose']
    goal = row['goal_world']
    response = module._handle_request(state, json.dumps({'start': pose[:3], 'end': [*goal[:2], pose[2]]}))
    result['queries'].append({'query_time_s': row['query_time_s'], 'logged_A_xyyaw': goal,
        'logged_query_root_xyz': pose[:3], 'reconstructed_PCT_response': response,
        'reconstructed_B_raw_xyyaw': [*response['traj'][-1][:2], goal[2]],
        'historical_snap_0_10_pass': response['snap_end_distance_m'] <= .10})
for flag in ('--pct-tomogram-path', '--pct-walkable-path', '--pct-collision-ply-path', '--pct-server-script'):
    path = Path(arg(flag)); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    result['sha256'][flag] = digest.hexdigest()
result['verification'] = {
    'query_0_4_logged_snap': .048897484176705044,
    'query_0_8_logged_snap': .07875451507636096,
    'first_two_match_logged_exactly': all(q['reconstructed_PCT_response']['snap_end_distance_m'] == v
        and q['reconstructed_B_raw_xyyaw'][:2] == [-.7152450561523436, 6.4248662948608395]
        for q, v in zip(result['queries'][:2], [.048897484176705044, .07875451507636096])),
    'third_B_and_snap_directly_logged': False,
    'third_failure_interpretation': 'PCT status ok; same_floor_direct; snap_end_dist zero means rounded endpoint already walkable. 0.2 m grid quantization exceeds historical 0.10 m threshold.'}
assert result['verification']['first_two_match_logged_exactly']
print(json.dumps(result, indent=2))
