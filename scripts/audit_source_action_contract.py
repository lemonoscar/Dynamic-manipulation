#!/usr/bin/env python3
"""Run perfect validation labels through the deployed joint decoder/limits, offline."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, write_json, source_identity
from conveyor_bench.conveyorvla.formal_metrics import LIMITS, cluster_mean
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointTrajectoryExecutor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-records', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    decoder = DirectJointTrajectoryExecutor(LIMITS)
    rows, axes, horizons = [], Counter(), Counter()
    with (args.output_dir/'rows.jsonl').open('x') as output:
        for line in args.validation_records.open():
            raw = json.loads(line)
            if raw['split'] != 'val':
                raise ValueError('only validation labels are allowed')
            if raw['mani_state'] is None:
                continue
            q = np.asarray(raw['mani_state'][:6])
            targets = np.asarray(raw['mani_delta_q_gripper'])
            chunk = decoder.prepare(q, targets)
            absolute = targets.copy(); absolute[:, :6] += q
            applied = np.array([[*c.joint_position, c.gripper_open_fraction] for c in chunk.commands])
            changed = np.abs(applied - absolute) > 1e-12
            for h, axis in zip(*np.where(changed)):
                axes[str(int(axis))] += 1; horizons[str(int(h)+1)] += 1
            row = {'episode_id': raw['episode_id'], 'sample_id': raw['sample_id'], 'route': raw['route'],
                   'position_events': chunk.position_saturation_count, 'rate_events': chunk.rate_saturation_count,
                   'gripper_events': chunk.gripper_saturation_count, 'saturation_rate': chunk.saturation_rate,
                   'max_joint_change_rad': float(np.abs(applied[:, :6]-absolute[:, :6]).max()),
                   'changed_elements': int(changed.sum()), 'transition_window': raw['transition_window']}
            output.write(json.dumps(row)+'\n'); rows.append(row)
    report = {'schema': 'perfect-source-label-decoder-audit-v1', 'split': 'val', 'robot_motion': False,
              'input_sha256': sha256(args.validation_records), 'source_identity': source_identity(ROOT),
              'rows': len(rows), 'saturation_threshold': .005,
              'saturation_rate': cluster_mean([r['saturation_rate'] for r in rows], [r['episode_id'] for r in rows]),
              'events': {k: sum(r[k] for r in rows) for k in ['position_events','rate_events','gripper_events']},
              'changed_elements_by_axis_0based': dict(axes), 'changed_elements_by_horizon_1based': dict(horizons),
              'max_joint_change_rad': max(r['max_joint_change_rad'] for r in rows),
              'interpretation': 'Perfect sampled future labels through deployed limits; overlapping chunks are not independent physical trials.'}
    report['gate_passed'] = report['saturation_rate']['episode_mean'] <= .005
    write_json(args.output_dir/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_identity'}, indent=2))


if __name__ == '__main__':
    main()
