"""Evaluator-only sampled-command replay and navigation error decomposition.

These helpers never select a learned action or supply simulator truth to a model.
Source-time replay is a diagnostic, not a learned-policy capability score.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .joint_trajectory_data import _align_sampled_5hz_rows, _sampled_5hz_gripper_target
from .waypoint import wrap_to_pi
from .waypoint_planner_adapters import validate_dwa_inputs


def sampled_phase(samples, frames, phase="exec_pick"):
    aligned = _align_sampled_5hz_rows(samples, frames)
    phases = {"plan_pick", "exec_pick"} if phase == "pick_with_planning" else {phase}
    indices = [i for i, (sample, _, _) in enumerate(aligned) if sample["pipeline_state"] in phases]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("replay phase must be present and contiguous")
    return [aligned[i] for i in indices]


def replay_schedule(aligned_phase, offset: int):
    """Issue u(t) or u(t+0.2), holding each for ten ticks, equal duration.

    A future command outside the phase is never borrowed; the final source
    command is repeated once for offset=1. The first pre-action state is shared.
    """
    if offset not in (0, 1):
        raise ValueError("only contemporaneous and one-sample-ahead replay are supported")
    result = []
    for i, (sample, _, _) in enumerate(aligned_phase):
        j = min(i + offset, len(aligned_phase) - 1)
        target = aligned_phase[j][0]
        action = np.asarray(target["action"], dtype=float)
        if action.shape != (11,) or not np.isfinite(action).all():
            raise ValueError("invalid sampled control target")
        if np.any(action[:3] != 0):
            raise ValueError("isolated manipulation replay requires zero source base command")
        result.append({"source_query_timestamp_s": sample["timestamp"],
                       "source_target_timestamp_s": target["timestamp"],
                       "source_frame_index": target["frame_index"],
                       "end_hold": i + offset >= len(aligned_phase),
                       "absolute_joint_target": action[3:9].tolist(),
                       "gripper_fraction": _sampled_5hz_gripper_target(action[9:11].tolist()),
                       "control_ticks": 10})
    return result


def planar_pose(value: Sequence[float]):
    result = tuple(float(x) for x in value)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError("pose must be finite (x, y, yaw), not a mixed-unit distance")
    return result


def deploy_source_chunk(items, query_joint_position, normalizer, limits):
    """Encode perfect absolute goals at the live query anchor, roundtrip, limit.

    A short phase tail is padded only to satisfy the ten-point decoder contract;
    callers execute only the original number of items.
    """
    from .joint_trajectory_runtime import DirectJointTrajectoryExecutor
    if not 1 <= len(items) <= 10:
        raise ValueError('source chunk must contain 1..10 targets')
    q = np.asarray(query_joint_position, dtype=float)
    absolute = np.array([[*v['absolute_joint_target'], v['gripper_fraction']] for v in items])
    absolute = np.concatenate([absolute, np.repeat(absolute[-1:],10-len(items),axis=0)])
    relative = absolute.copy(); relative[:,:6] -= q
    normalized = normalizer.normalize_action('PICK', relative)
    recovered = normalizer.denormalize_action('PICK', normalized)
    decoded = DirectJointTrajectoryExecutor(limits).prepare(q, recovered)
    return decoded, {'contract':'query-relative-normalizer-decoder-limits-v2',
                     'query_joint_position':q.tolist(), 'raw_absolute_targets':absolute.tolist(),
                     'normalizer_roundtrip_max_abs_error':float(np.max(np.abs(np.array(recovered)-relative))),
                     'original_points':len(items), 'padded_points':10-len(items),
                     'position_events':decoded.position_saturation_count,
                     'rate_events':decoded.rate_saturation_count,
                     'gripper_events':decoded.gripper_saturation_count}


def navigation_decomposition(*, nominal, requested, planned, measured=None, reached=False):
    """G/A/B/C XY and wrapped-yaw errors; C is absent for planner-only probes."""
    poses = {"G": planar_pose(nominal), "A": planar_pose(requested), "B": planar_pose(planned),
             "C": None if measured is None else planar_pose(measured)}
    result = {"poses_xyyaw": poses, "G_semantics": "source_nominal_operation_pose_not_unique_feasible_pose",
              "local_goal_reached": bool(reached), "errors": {}}
    for a, b in [("A", "G"), ("B", "A"), ("C", "B"), ("C", "G"), ("C", "A")]:
        if poses[a] is None or poses[b] is None:
            result["errors"][a + "_minus_" + b] = None
            continue
        u, v = poses[a], poses[b]
        result["errors"][a + "_minus_" + b] = {
            "xy_m": math.hypot(u[0] - v[0], u[1] - v[1]),
            "yaw_rad": abs(wrap_to_pi(u[2] - v[2]))}
    result["nominal_tolerance_bound_C_minus_A_m"] = .22 if reached else None
    return result
