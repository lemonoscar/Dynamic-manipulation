"""Conservative continuous endpoint candidate validation (opt-in, offline first).

The supplied mask must certify obstacle-free volume AND traversable support for
each whole cell. A bool occupancy image alone cannot supply that evidence.
The circumscribed radius must cover the robot's full navigation/turning envelope.
This module certifies geometry, not locomotion tracking or task success.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .waypoint_execution import PCTPlan


@dataclass(frozen=True)
class SweptDiskEvidence:
    certified_cells: np.ndarray
    resolution_m: float
    origin_xyyaw: tuple[float, float, float]
    robot_radius_m: float
    geometry_sha256: str
    footprint_sha256: str

    def __post_init__(self):
        cells = np.asarray(self.certified_cells)
        if cells.ndim != 2 or not cells.size or cells.dtype != np.bool_:
            raise ValueError('whole-cell geometry/support certificate must be a nonempty boolean raster')
        if not np.isfinite([self.resolution_m, self.robot_radius_m, *self.origin_xyyaw]).all() or min(self.resolution_m, self.robot_radius_m) <= 0:
            raise ValueError('invalid swept disk geometry')
        for digest in (self.geometry_sha256, self.footprint_sha256):
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('geometry and footprint evidence require SHA256 identities')
        cells = cells.copy(); cells.flags.writeable = False
        object.__setattr__(self, 'certified_cells', cells)

    def check(self, start_xy, end_xy):
        """Cover every touched cell, including turning, without point sampling gaps.

        Cell circumcircles over-approximate squares. Testing distance of their
        centers to the entire segment therefore conservatively covers a capsule.
        """
        a, b = np.asarray(start_xy, float), np.asarray(end_xy, float)
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite([a, b]).all():
            raise ValueError('invalid continuous segment')
        ox, oy, yaw = self.origin_xyyaw
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, s], [-s, c]])
        a, b = rot @ (a-(ox, oy)), rot @ (b-(ox, oy))
        r, h = self.resolution_m, self.robot_radius_m
        height, width = self.certified_cells.shape
        if np.any(np.minimum(a,b)-h < 0) or np.any(np.maximum(a,b)+h > (width*r,height*r)):
            return {'valid':False, 'reason':'sweep_outside_geometry_coverage'}
        radius = h+r/math.sqrt(2)
        lo = np.maximum(0, np.floor((np.minimum(a,b)-radius)/r).astype(int))
        hi = np.minimum((width-1,height-1), np.floor((np.maximum(a,b)+radius)/r).astype(int))
        cols, bottoms = np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
        centers = np.stack(((cols+.5)*r,(bottoms+.5)*r),axis=-1)
        delta = b-a; length2 = float(delta @ delta)
        t = np.clip(np.sum((centers-a)*delta,axis=-1)/length2,0,1) if length2 else np.zeros(cols.shape)
        touched = np.linalg.norm(centers-(a+t[...,None]*delta),axis=-1) <= radius+1e-12
        valid = bool(np.all(self.certified_cells[height-1-bottoms,cols][touched]))
        return {'valid':valid, 'reason':'certified_sweep' if valid else 'uncertified_obstacle_or_support_cell',
                'tested_cells':int(touched.sum()), 'radius_m':h, 'resolution_m':r,
                'geometry_sha256':self.geometry_sha256, 'footprint_sha256':self.footprint_sha256,
                'covers_all_yaw_angles':True}


def continuous_endpoint_candidate(plan: PCTPlan, requested_xyzyaw, evidence: SweptDiskEvidence):
    """Append a real checked final segment while preserving the coarse endpoint.

    This is a candidate for controller validation, never an automatically
    deployable PCTPlan. No original snap threshold or evidence is overwritten.
    """
    target = tuple(float(x) for x in requested_xyzyaw)
    if len(target) != 4 or not np.isfinite(target).all() or not plan.path_world:
        raise ValueError('invalid continuous endpoint inputs')
    path = tuple(tuple(float(x) for x in p) for p in plan.path_world)
    coarse = plan.snapped_goal_world
    if math.dist(path[-1], coarse[:2]) > 1e-6:
        raise ValueError('coarse path endpoint disagrees with reported B')
    certificate = evidence.check(coarse[:2], target[:2])
    if abs(coarse[2]-target[2]) > 1e-6:
        certificate = {**certificate, 'valid':False, 'reason':'vertical_connector_not_certified'}
    if certificate['valid'] and math.dist(path[-1],target[:2]) > 1e-9:
        path += (target[:2],)
    return {'schema':'continuous-endpoint-candidate-v1',
            'coarse_endpoint_B':list(coarse), 'requested_goal_A':list(target),
            'coarse_snap_distance_m':plan.snap_distance_m,
            'candidate_path_world':[list(p) for p in path],
            'geometry_certificate':certificate,
            'controller_feasibility_verified':False, 'deployment_approved':False}


def classify_degenerate_path(current_xyyaw, goal_xyyaw, *, position_tolerance=.12, yaw_tolerance=.14):
    from .waypoint import wrap_to_pi
    if not np.isfinite([*current_xyyaw, *goal_xyyaw]).all():
        raise ValueError('nonfinite degenerate path poses')
    if math.dist(current_xyyaw[:2], goal_xyyaw[:2]) > position_tolerance:
        return 'reconnect_from_measured_pose_required'
    if abs(wrap_to_pi(goal_xyyaw[2]-current_xyyaw[2])) > yaw_tolerance:
        return 'validated_in_place_turn_required'
    return 'local_goal_reached'
