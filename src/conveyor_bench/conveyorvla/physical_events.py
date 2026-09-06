"""Read-only grasp evidence, with relative stability and explicit contact coverage.

No evaluator in this module has a simulation handle or can apply an intervention.
The proxy is not a contact-verified grasp; missing contact data stays unknown.
"""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class GraspEvaluationConfig:
    version: str = 'relative-grasp-evidence-v2'
    window_s: float = 1.
    max_sample_gap_s: float = .05
    lift_min_m: float = .04
    tcp_distance_max_m: float = .08
    relative_position_drift_max_m: float = .01
    relative_rotation_drift_max_rad: float = .15
    closed_command_max: float = .5
    safety_world_speed_max_mps: float = .30


def rotation_wxyz(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('invalid pose quaternion')
    w,x,y,z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class RelativeGraspEvaluator:
    def __init__(self, initial_object_z, record, config=GraspEvaluationConfig()):
        self.initial_object_z = float(initial_object_z)
        self.record, self.config = record, config
        self.window = deque()
        self.ever_proxy_hold = self.ever_contact_grasp = False
        self.observed_samples = self.unknown_contact_samples = 0
        self.latest = {}

    def observe(self, state, command_fraction, *, contacts=None):
        t = float(state.timestamp)
        op, ep = np.asarray(state.object_pose), np.asarray(state.tcp_pose)
        velocity = np.asarray(state.object_velocity[:3], dtype=float)
        if not np.isfinite([t, command_fraction, *op, *ep, *velocity]).all():
            raise ValueError('nonfinite grasp evidence')
        re = rotation_wxyz(ep[3:]); ro = rotation_wxyz(op[3:])
        relative = re.T @ (op[:3]-ep[:3])
        relative_rotation = re.T @ ro
        if self.window:
            gap = t-self.window[-1]['time']
            if gap <= 0:
                raise ValueError('grasp evidence timestamps must increase')
            if gap > self.config.max_sample_gap_s + 1e-9:
                self.window.clear()
        c = self.config
        lift = float(op[2])-self.initial_object_z
        distance = float(np.linalg.norm(relative))
        geometry = command_fraction <= c.closed_command_max and lift >= c.lift_min_m and distance <= c.tcp_distance_max_m
        self.window.append({'time':t, 'p':relative, 'r':relative_rotation,
                            'geometry':geometry, 'contacts':contacts})
        while self.window and self.window[0]['time'] < t-c.window_s-1e-9:
            self.window.popleft()
        first = self.window[0]
        full = t-first['time'] >= c.window_s-1e-9
        position_drift = max(float(np.linalg.norm(v['p']-first['p'])) for v in self.window)
        rotation_drift = max(math.acos(float(np.clip((np.trace(first['r'].T @ v['r'])-1)/2, -1, 1))) for v in self.window)
        proxy = bool(full and all(v['geometry'] for v in self.window)
                     and position_drift <= c.relative_position_drift_max_m
                     and rotation_drift <= c.relative_rotation_drift_max_rad)
        known = all(v['contacts'] is not None and
                    isinstance(v['contacts'].get('bilateral_finger_contact'), bool) and
                    isinstance(v['contacts'].get('opposing_finger_normals'), bool) and
                    isinstance(v['contacts'].get('external_support'), bool) for v in self.window)
        contact_grasp = (bool(proxy and all(v['contacts']['bilateral_finger_contact'] and
                                          v['contacts']['opposing_finger_normals'] and
                                          not v['contacts']['external_support'] for v in self.window)) if known else None)
        self.observed_samples += 1
        self.unknown_contact_samples += not known
        self.latest = {'schema':c.version, 'time_s':t, 'window_complete':full,
                       'geometry_hold_proxy':proxy, 'contact_grasp_verified':contact_grasp,
                       'contact_coverage_complete':known,
                       'relative_position_E_m':relative.tolist(), 'relative_position_drift_m':position_drift,
                       'relative_rotation_drift_rad':rotation_drift,
                       'lift_m':lift, 'tcp_distance_m':distance,
                       'world_speed_mps':float(np.linalg.norm(velocity)),
                       'world_speed_safe':bool(np.linalg.norm(velocity) <= c.safety_world_speed_max_mps),
                       'command_fraction':float(command_fraction)}
        if proxy and not self.ever_proxy_hold:
            self.record('geometry_hold_proxy_v2', self.latest)
        if contact_grasp is True and not self.ever_contact_grasp:
            self.record('contact_grasp_verified_v2', self.latest)
        self.ever_proxy_hold |= proxy
        self.ever_contact_grasp |= contact_grasp is True
        return self.latest

    def evidence(self):
        return {'config':asdict(self.config), 'latest':self.latest,
                'ever_geometry_hold_proxy':self.ever_proxy_hold,
                'ever_contact_grasp_verified':(True if self.ever_contact_grasp else
                    None if self.unknown_contact_samples or not self.observed_samples else False),
                'observed_samples':self.observed_samples,
                'unknown_contact_window_samples':self.unknown_contact_samples,
                'evaluation_changes_simulation':False,
                'missing_contacts_are_unknown':True}


class LegacyPhysicalEventEvaluator:
    """Frozen v1 event semantics, retained alongside v2 for comparable rescoring."""
    def __init__(self, record):
        self.record = record
        self.pick = self.carry = self.drop = self.release = False
        self.peak_lift = 0.
        self.latest = {}

    def observe(self, *, lift, distance, speed, fraction, closed_command_seen, displacement, attached):
        self.peak_lift = max(self.peak_lift, lift)
        verified = closed_command_seen and fraction <= .5 and lift >= .04 and distance <= .08 and speed <= .30
        evidence = {'lift_m':lift, 'peak_lift_m':self.peak_lift, 'object_tcp_distance_m':distance,
                    'object_speed_mps':speed, 'measured_gripper_fraction':fraction,
                    'closed_command_seen':closed_command_seen, 'contact_evidence':'geometry_proxy_not_contact_sensor'}
        if verified and not self.pick:
            self.pick = True; self.record('physical_pick_verified', evidence)
        if self.pick and displacement >= .20 and distance <= .12 and not self.carry:
            self.carry = True; self.record('physical_carry_verified', {**evidence, 'displacement_m':displacement})
        released = fraction >= .8 and distance >= .06 and not attached
        if self.pick and distance > .20 and not released and not self.drop:
            self.drop = True; self.record('physical_drop_proxy', evidence)
        if self.pick and released and not self.release:
            self.release = True; self.record('physical_release_observed', evidence)
        self.latest = evidence
