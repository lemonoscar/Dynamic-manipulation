"""One reach contract and conservative online geometry/stop coverage checks."""
from dataclasses import dataclass, replace
import math
import numpy as np


@dataclass(frozen=True)
class ReachConfig:
    position_tolerance_m: float = .12
    yaw_tolerance_rad: float = .14
    linear_stable_m_s: float = .05
    angular_stable_rad_s: float = .10
    stable_duration_s: float = .20

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('reach thresholds must be finite and positive')


class ReachMonitor:
    def __init__(self, config):
        self.config = config
        self.stable_since = None

    def update(self, distance, yaw_error, velocity, now):
        c = self.config
        if distance > c.position_tolerance_m:
            self.stable_since = None
            return 'TRANSLATE'
        if yaw_error > c.yaw_tolerance_rad:
            self.stable_since = None
            return 'VALIDATED_TURN_REQUIRED'
        if math.hypot(*velocity[:2]) > c.linear_stable_m_s or abs(velocity[2]) > c.angular_stable_rad_s:
            self.stable_since = None
            return 'SETTLE'
        if self.stable_since is None:
            self.stable_since = now
        return 'REACHED' if now-self.stable_since >= c.stable_duration_s-1e-9 else 'SETTLE'


@dataclass(frozen=True)
class OnlineNavigationSafety:
    evidence: object  # SweptDiskEvidence for full robot AND current carried envelope
    context_id: str
    valid_until_s: float
    reaction_time_s: float
    braking_deceleration_m_s2: float
    certified_speed_bound_m_s: float

    def __post_init__(self):
        if not self.context_id or any(not math.isfinite(v) or v <= 0 for v in (
                self.valid_until_s, self.reaction_time_s, self.braking_deceleration_m_s2,
                self.certified_speed_bound_m_s)):
            raise ValueError('online monitor needs validated clock, speed and braking bounds')

    def check(self, pose, measured_velocity, command, now_s):
        if now_s > self.valid_until_s:
            return {'valid': False, 'reason': 'geometry_context_expired'}
        if not np.isfinite([*pose, *measured_velocity, *command, now_s]).all():
            return {'valid': False, 'reason': 'untrusted_pose_or_command'}
        speed = max(math.hypot(*measured_velocity[:2]), math.hypot(*command[:2]))
        if speed > self.certified_speed_bound_m_s:
            return {'valid': False, 'reason': 'speed_outside_stop_model'}
        # A disk bounds any planar heading during reaction+braking. Full-yaw
        # robot radius is inherited, so pure turns also require free support.
        bound = self.certified_speed_bound_m_s
        stop_distance = bound*self.reaction_time_s + bound*bound/(2*self.braking_deceleration_m_s2)
        certificate = replace(self.evidence, robot_radius_m=self.evidence.robot_radius_m+stop_distance)
        result = certificate.check(pose[:2], pose[:2])
        return {**result, 'safety_context_id': self.context_id, 'stop_distance_bound_m': stop_distance}


def build_b0_navigation(pct_planner, dwa_controller, *, online_safety, reach=ReachConfig()):
    """B0-runtime entry: shared reach/stability and mandatory online certificate.

    Legacy executor construction remains available for B-old reproduction.
    Continuous endpoint candidates still do not bypass the original snap gate.
    """
    from .joint_trajectory_system import PCTDWAJointNavigationExecutor,JointNavigationConfig
    return PCTDWAJointNavigationExecutor(pct_planner,dwa_controller,
        JointNavigationConfig(goal_tolerance_m=reach.position_tolerance_m,
                              yaw_tolerance_rad=reach.yaw_tolerance_rad),
        reach_config=reach,online_safety=online_safety,require_online_safety=True)
