"""Seconds, query-relative codecs and immutable action identity."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TimeProfile:
    name: str
    horizon: int
    sample_period_s: float
    first_target_offset_s: float
    history_span_s: float = 0.2
    control_period_s: float = 0.02

    def target_times(self, query_time_s):
        return query_time_s + self.first_target_offset_s + np.arange(self.horizon) * self.sample_period_s

    def apply_times(self, first_apply_time_s):
        return first_apply_time_s + np.arange(self.horizon) * self.sample_period_s


TIME_PROFILES = {
    'legacy_future_5hz': TimeProfile('legacy_future_5hz', 10, .2, .2),
    'causal_command_5hz': TimeProfile('causal_command_5hz', 10, .2, 0.),
    'causal_command_25hz': TimeProfile('causal_command_25hz', 50, .04, 0.),
}


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name}: expected finite {shape}')
    return array


def decode_mani(delta_gripper, anchor_q):
    action = np.asarray(delta_gripper, dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 7 or not np.isfinite(action).all():
        raise ValueError('joint-command-v2 requires finite H x 7, never a 10D pose action')
    result = action.copy()
    result[:, :6] += finite_array(anchor_q, (6,), 'anchor_q')
    return result


def encode_mani(absolute, anchor_q):
    return decode_mani(absolute, -finite_array(anchor_q, (6,), 'anchor_q'))


def reanchor_mani(action, old_anchor, new_anchor):
    return encode_mani(decode_mani(action, old_anchor), new_anchor)


def nav_goal_world(goal, query_xyyaw):
    dx, dy, dyaw = finite_array(goal, (3,), 'query-body goal')
    x, y, yaw = finite_array(query_xyyaw, (3,), 'query pose')
    return (x + math.cos(yaw)*dx - math.sin(yaw)*dy,
            y + math.sin(yaw)*dx + math.cos(yaw)*dy,
            (yaw + dyaw + math.pi) % (2*math.pi) - math.pi)


@dataclass(frozen=True)
class ActionIdentity:
    mission_id: str
    instruction_version: int
    active_task_epoch: int
    active_task_id: str
    model_id: str
    normalizer_id: str
    safety_context_id: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.mission_id, self.active_task_id,
                self.model_id, self.normalizer_id, self.safety_context_id)):
            raise ValueError('action identity fields must be nonempty')
        if min(self.instruction_version, self.active_task_epoch) < 0:
            raise ValueError('negative identity version')
