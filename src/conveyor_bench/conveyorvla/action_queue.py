"""Absolute target queue on the application clock, independent of model latency."""
from dataclasses import dataclass
import math
import numpy as np
from .contracts.action import ActionIdentity, TIME_PROFILES, finite_array, encode_mani


@dataclass(frozen=True)
class ActionPlan:
    request_id: int
    identity: ActionIdentity
    source_plan_version: int
    observation_id: str
    query_time_s: float
    query_anchor_q: tuple
    time_profile: str
    first_apply_time_s: float
    raw_targets: tuple
    effective_absolute_targets: tuple
    valid_until_s: float

    def __post_init__(self):
        profile = TIME_PROFILES[self.time_profile]
        finite_array(self.query_anchor_q, (6,), 'query anchor')
        finite_array(self.raw_targets, (profile.horizon, 7), 'raw targets')
        effective = finite_array(self.effective_absolute_targets, (profile.horizon, 7), 'effective targets')
        object.__setattr__(self, 'effective_absolute_targets', tuple(map(tuple, effective)))
        object.__setattr__(self, 'raw_targets', tuple(map(tuple, self.raw_targets)))
        object.__setattr__(self, 'query_anchor_q', tuple(self.query_anchor_q))
        if ((effective[:, 6] < 0) | (effective[:, 6] > 1)).any():
            raise ValueError('queue requires bounded effective gripper')
        if not all(math.isfinite(x) for x in (self.query_time_s, self.first_apply_time_s, self.valid_until_s)):
            raise ValueError('nonfinite action clock')
        if self.query_time_s < 0 or self.first_apply_time_s < self.query_time_s or self.valid_until_s <= self.first_apply_time_s:
            raise ValueError('invalid action clock ordering')
        if self.request_id < 0 or not self.observation_id:
            raise ValueError('invalid action request identity')

    @property
    def profile(self):
        return TIME_PROFILES[self.time_profile]


@dataclass(frozen=True)
class QueueCommand:
    target: tuple | None
    reason: str
    request_id: int | None = None
    action_index: int | None = None


class AbsoluteActionQueue:
    def __init__(self, identity):
        self.identity = identity
        self._segments = []  # start, end, target, request, index, valid_until
        self.last_request_id = -1
        self.committed_until_s = -math.inf
        self.last_control_time_s = -math.inf
        self.history = []
        self.reason = 'initialization_hold'

    def cancel(self, identity=None, reason='task_epoch_cancelled'):
        self._segments.clear()
        self.committed_until_s = -math.inf
        if identity is not None:
            self.identity = identity
        self.reason = reason

    def commit_through(self, now, expected_delay_s, margin_s=0.):
        if any(not math.isfinite(x) or x < 0 for x in (now, expected_delay_s, margin_s)):
            raise ValueError('invalid latency commitment')
        until = now + expected_delay_s + margin_s
        cursor = now
        for start, end, _, _, _, valid in self._segments:
            if start > cursor + 1e-9:
                break
            if end > cursor:
                cursor = min(end, valid)
            if cursor >= until - 1e-9:
                self.committed_until_s = max(self.committed_until_s, until)
                return until
        raise ValueError('insufficient_queue_coverage')

    def accept(self, plan, *, now_s, plan_version):
        if not math.isfinite(now_s) or now_s < plan.query_time_s:
            raise ValueError('response clock precedes query')
        if plan.identity != self.identity:
            raise ValueError('stale task/model/normalizer/safety identity')
        if plan.source_plan_version > plan_version:
            raise ValueError('unknown future plan version')
        if plan.request_id <= self.last_request_id:
            raise ValueError('duplicate_or_out_of_order_response')
        if now_s >= plan.valid_until_s:
            raise ValueError('expired_response')
        boundary = max(now_s, self.committed_until_s)
        kept = []
        for start, end, target, request, index, valid in self._segments:
            if end > now_s and start < boundary:
                kept.append((start, min(end, boundary), target, request, index, valid))
        added = []
        for i, (start, target) in enumerate(zip(plan.profile.apply_times(plan.first_apply_time_s), plan.effective_absolute_targets)):
            end = min(start + plan.profile.sample_period_s, plan.valid_until_s)
            if end <= boundary + 1e-9:
                continue
            added.append((max(float(start), boundary), float(end), tuple(target), plan.request_id, i, plan.valid_until_s))
        if not added:
            raise ValueError('no_unexpired_suffix')
        self._segments = kept + added
        self.last_request_id = plan.request_id
        self.reason = 'active'

    def overlap(self, apply_times, new_anchor_q):
        values = np.zeros((len(apply_times), 7))
        valid = np.zeros(len(apply_times), dtype=bool)
        committed = np.zeros(len(apply_times), dtype=bool)
        for i, time_s in enumerate(apply_times):
            for start, end, target, _, _, expiry in self._segments:
                if start-1e-9 <= time_s < min(end, expiry)-1e-9:
                    values[i] = target
                    valid[i] = True
                    committed[i] = time_s < self.committed_until_s-1e-9
                    break
        relative = encode_mani(values, new_anchor_q)
        relative[~valid] = 0
        return relative, valid, committed

    def next_for_control_time(self, now_s):
        if not math.isfinite(now_s) or now_s <= self.last_control_time_s:
            raise ValueError('control clock must strictly advance; no replay')
        self.last_control_time_s = now_s
        self._segments = [s for s in self._segments if min(s[1], s[5]) > now_s+1e-9]
        for start, end, target, request, index, expiry in self._segments:
            if start-1e-9 <= now_s < min(end, expiry)-1e-9:
                return QueueCommand(target, 'action', request, index)
        return QueueCommand(None, self.reason if self.reason != 'active' else 'queue_exhausted_or_gap')

    def record_applied(self, time_s, command, effective_target):
        # Called only after controller.apply succeeds; requests are not execution evidence.
        target = finite_array(effective_target, (7,), 'applied target')
        self.history.append({'time_s': time_s, 'request_id': command.request_id,
            'action_index': command.action_index, 'effective_target': target.tolist(),
            'active_task_epoch': self.identity.active_task_epoch})
