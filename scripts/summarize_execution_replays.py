#!/usr/bin/env python3
"""Summarize completed paired sampled PICK replays, separately from policy scores."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, write_json
from conveyor_bench.conveyorvla.formal_metrics import cluster_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-root', type=Path, required=True)
    parser.add_argument('--prepared-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.prepared_manifest.read_text())
    episodes = manifest['episode_ids'][:3]
    expected = {(e, p, o) for e in episodes for p in ['source_assisted','no_grasp_assist'] for o in [0,1]}
    rows, curves = {}, {}
    for path in sorted(args.matrix_root.glob('*/runtime/episode_*/summary.json')):
        summary = json.loads(path.read_text())
        key = (summary['source_episode'], summary['physics_profile'], summary['offset_samples'])
        if key in rows or key not in expected:
            raise ValueError('unexpected or duplicate replay condition')
        if summary['status'] != 'complete':
            raise ValueError(f'incomplete physical replay: {path}')
        process = json.loads((path.parents[2]/'process.json').read_text())
        if process['returncode'] != 0:
            raise ValueError(f'replay framework did not exit cleanly: {path}')
        physics = summary['physics_evidence']; latest = physics['latest']
        rows[key] = {'episode_id': key[0], 'physics_profile': key[1], 'offset_samples': key[2],
                     'commands_completed': summary['success'], 'formal_pick_verified': physics['pick_verified'],
                     'diagnostic_lift_hold': summary['diagnostic_lift_hold']['passed'],
                     'grasp_constraint_created': physics['grasp_constraint_created'],
                     'peak_lift_m': latest['peak_lift_m'], 'final_lift_m': latest['lift_m'],
                     'final_measured_gripper_fraction': latest['measured_gripper_fraction'],
                     'joint_tracking_mae_rad': summary['command_end_joint_tracking_mae_rad'],
                     'min_tcp_distance_m': summary['min_tcp_object_distance_m'],
                     'summary_path': str(path), 'summary_sha256': sha256(path)}
        initial_z = summary['initialization']['source_observation']['object_pose'][2]
        curve = []
        for line in (path.parent/'replay_trace.jsonl').open():
            tick = json.loads(line)
            if tick['event'] == 'control_step' and tick['route'] == 'PICK':
                curve.append(tick['state_after']['object_pose'][2]-initial_z)
        curves[key] = curve
    if set(rows) != expected:
        raise ValueError(f'missing {len(expected-set(rows))} of 12 predefined replays')
    groups = {}
    for profile in ['source_assisted','no_grasp_assist']:
        for offset in [0,1]:
            subset = [v for k,v in rows.items() if k[1:] == (profile, offset)]
            groups[f'{profile}/offset{offset}'] = {name: cluster_mean(
                [r[name] for r in subset], [r['episode_id'] for r in subset])
                for name in ['formal_pick_verified','diagnostic_lift_hold','peak_lift_m','final_lift_m']}
    differences = {profile: cluster_mean(
        [rows[(e,profile,1)]['final_lift_m']-rows[(e,profile,0)]['final_lift_m'] for e in episodes], episodes)
        for profile in ['source_assisted','no_grasp_assist']}
    report = {'schema': 'paired-sampled-pick-replay-summary-v1', 'status': 'complete', 'split': 'val',
              'policy_score': False, 'source_Sim6_reexecuted': False, 'full_task_success': None,
              'episodes': episodes, 'rollouts': list(rows.values()), 'groups': groups,
              'paired_offset1_minus_offset0_final_lift_m': differences,
              'prepared_manifest_sha256': sha256(args.prepared_manifest),
              'interpretation': 'Three prespecified episodes only; intervals are exploratory, not model capability or full-task success intervals.'}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir/'report.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,2,figsize=(11,10),sharex=True,sharey=True)
    for i, episode in enumerate(episodes):
        for j, profile in enumerate(['source_assisted','no_grasp_assist']):
            ax = axes[i,j]
            for offset in [0,1]:
                y = curves[(episode,profile,offset)]
                ax.plot([.02*(n+1) for n in range(len(y))], y, label=f'offset {offset}',linewidth=1.5)
            ax.axhline(.04,color='gray',linestyle=':',label='4 cm lift threshold')
            ax.set_title(f'{episode}\n{profile}');ax.grid(alpha=.2)
            ax.set_xlabel('Replay time (s)');ax.set_ylabel('Object lift (m)')
    axes[0,0].legend()
    fig.suptitle('Sampled PICK replay in Isaac 5.1 — no model, not full-task success')
    fig.tight_layout();fig.savefig(args.output_dir/'lift_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'rollouts':len(rows),'groups':groups,'paired_lift_difference':differences},indent=2))


if __name__ == '__main__':
    main()
