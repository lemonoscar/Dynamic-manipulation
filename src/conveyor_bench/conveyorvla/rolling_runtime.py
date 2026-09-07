"""Nonblocking orchestration boundary for a frozen action service and rolling planner.

prepare_request / complete_request can run on opposite sides of a worker. tick
never invokes a model. The caller owns wall/simulation clocks and controller I/O.
"""
from dataclasses import dataclass, replace
import numpy as np
from .action_queue import AbsoluteActionQueue, ActionPlan, QueueCommand
from .contracts.action import ActionIdentity, TIME_PROFILES, decode_mani
from .joint_trajectory import JointTrajectoryRoute, canonical_solution, ROUTE_SUBTASKS, joint_trajectory_prompt


@dataclass(frozen=True)
class ActionRequest:
    request_id: int
    identity: ActionIdentity
    plan_version: int
    observation: object
    task: object
    rtc_context: dict | None


class FrozenActionBackend:
    """Original Qwen/canonical active-subtask path; no new JSON sent to old weights."""
    def __init__(self, policy, normalizer):
        self.policy = policy
        self.normalizer = normalizer
        policy.eval()

    def __call__(self, request, original_instruction):
        from .joint_trajectory_model import JointTrajectoryRouteDecision
        obs = request.observation
        if len(obs.images) != 4:
            raise ValueError('frozen baseline requires head[t-.2,t], wrist[t-.2,t]')
        expected = (obs.time_s-.2, obs.time_s, obs.time_s-.2, obs.time_s)
        if not np.allclose(obs.image_times_s, expected, atol=1e-6):
            raise ValueError('baseline visual history time mismatch')
        # The legacy canonical contract can only represent this object/destination.
        if request.task.target_ref != 'cola' or request.task.destination_ref != 'destination':
            raise ValueError('legacy checkpoint cannot represent a changed target identity')
        route = JointTrajectoryRoute(request.task.primitive)
        decision = JointTrajectoryRouteDecision(route, canonical_solution(route), ROUTE_SUBTASKS[route],
            1., {r.value: float(r == route) for r in JointTrajectoryRoute}, True)
        example = {'video': (obs.images[:2], obs.images[2:]),
            'lang': joint_trajectory_prompt(original_instruction),
            'mani_state': self.normalizer.normalize_mani_state(obs.mani_state)}
        kwargs = {} if request.rtc_context is None else {'rtc_contexts': [request.rtc_context]}
        value = self.policy.predict_actions([example], [decision], **kwargs)[0]
        return self.normalizer.denormalize_action(route, value)


class RollingRuntime:
    def __init__(self, memory, *, model_id, normalizer, safety_context_id,
                 limits, rtc=False, time_profile='legacy_future_5hz', max_guidance_weight=5.):
        self.memory = memory
        self.normalizer = normalizer
        self.model_id = model_id
        self.safety_context_id = safety_context_id
        self.limits = limits
        self.rtc = rtc
        self.profile = TIME_PROFILES[time_profile]
        self.max_guidance_weight = max_guidance_weight
        self.queue = AbsoluteActionQueue(self.identity())
        self.sequence = 0
        self.pending = {}
        self.events = []
        self._invalidated_task_id = None

    def identity(self):
        task = self.memory.active_task
        return ActionIdentity(self.memory.mission_id, self.memory.instruction_version,
            self.memory.active_task_epoch, task.task_id if task else 'FINISHED',
            self.model_id, self.normalizer.payload['normalizer_id'], self.safety_context_id)

    def synchronize_task(self):
        task = self.memory.active_task
        now = max(self.memory.observations.values(), default=0.)
        if task is not None and any(self.memory.fact_value(p, now) is not True for p in task.preconditions):
            if self._invalidated_task_id != task.task_id:
                self.memory.active_task_epoch += 1
                self.memory.plan_version += 1
                self._invalidated_task_id = task.task_id
            self.queue.cancel(self.identity(), reason='current_task_precondition_invalid')
        identity = self.identity()
        if identity != self.queue.identity:
            self.queue.cancel(identity)

    def prepare_request(self, observation, *, expected_delay_s=0., margin_s=0.):
        self.synchronize_task()
        if self.memory.finished or self.memory.active_task.primitive == 'VERIFY':
            raise ValueError('no continuous action for FINISH/VERIFY')
        if observation.mission_id != self.memory.mission_id:
            raise ValueError('cross-mission action observation')
        if (self.memory.observations.get(observation.observation_id) != observation.time_s
                or observation.time_s != max(self.memory.observations.values(), default=-1.)):
            raise ValueError('action observation must be the current registered snapshot')
        if any(self.memory.fact_value(p, observation.time_s) is not True for p in self.memory.active_task.preconditions):
            raise ValueError('current_task_precondition_invalid')
        context = None
        mani = self.memory.active_task.primitive in {'PICK', 'PLACE'}
        if self.rtc and mani and self.queue.last_request_id >= 0:
            try:
                self.queue.commit_through(observation.time_s, expected_delay_s, margin_s)
            except ValueError as error:
                self.queue.cancel(reason=str(error))
                self.events.append({'reason': str(error), 'time_s': observation.time_s})
            previous, valid, committed = self.queue.overlap(
                self.profile.apply_times(observation.time_s), observation.q)
            if valid.any():
                from .rtc_sampling import prefix_weights
                # A hole cannot become a fabricated conditioning prefix.
                overlap = next((i for i, v in enumerate(valid) if not v), len(valid))
                count = int(committed[:overlap].sum())
                weights = prefix_weights(count, overlap, self.profile.horizon).tolist()
                normalized = self.normalizer.normalize_action(self.memory.active_task.primitive, previous)
                context = {'previous': normalized, 'weights': weights,
                    'max_guidance_weight': self.max_guidance_weight}
        request = ActionRequest(self.sequence, self.identity(), self.memory.plan_version,
            observation, self.memory.active_task, context)
        self.pending[self.sequence] = request
        self.sequence += 1
        return request

    def complete_request(self, request_id, physical_actions, *, now_s, max_age_s=2.):
        request = self.pending.pop(request_id)
        self.synchronize_task()
        if any(self.memory.fact_value(p, now_s) is not True for p in request.task.preconditions):
            raise ValueError('current_task_precondition_invalid')
        if request.identity != self.identity():
            raise ValueError('action task invalidated during inference')
        if request.task.primitive not in {'PICK', 'PLACE'}:
            # NAV is a goal proposal, never inserted into the Mani queue.
            from .contracts.action import finite_array, nav_goal_world
            points = np.asarray(physical_actions, float)
            if points.shape != (10, 3) or not np.isfinite(points).all():
                raise ValueError('legacy NAV requires 10 query-body points')
            if now_s < request.observation.time_s or now_s-request.observation.time_s >= max_age_s:
                raise ValueError('expired NAV response')
            return {'identity': request.identity, 'query_time_s': request.observation.time_s,
                'query_base_xyyaw': request.observation.base_xyyaw,
                'valid_until_s': request.observation.time_s+max_age_s,
                'goal_world': nav_goal_world(points[-1], request.observation.base_xyyaw),
                'reference_query_body': points.tolist(), 'consumed_index': 9}
        absolute = decode_mani(physical_actions, request.observation.q)
        previous = np.asarray(request.observation.q)
        effective = absolute.copy()
        counts = {'position': 0, 'rate': 0, 'gripper': 0}
        for row in effective:
            bounded = np.clip(row[:6], self.limits.lower, self.limits.upper)
            counts['position'] += int(np.count_nonzero(np.abs(bounded-row[:6]) > 1e-12))
            delta = np.asarray(self.limits.max_rate_rad_s)*self.profile.sample_period_s
            limited = np.clip(bounded, previous-delta, previous+delta)
            counts['rate'] += int(np.count_nonzero(np.abs(limited-bounded) > 1e-12))
            counts['gripper'] += int(not 0 <= row[6] <= 1)
            row[:6] = limited; row[6] = np.clip(row[6], 0, 1)
            previous = row[:6].copy()
        plan = ActionPlan(request.request_id, request.identity, request.plan_version,
            request.observation.observation_id, request.observation.time_s, request.observation.q,
            self.profile.name, request.observation.time_s,
            tuple(map(tuple, absolute)), tuple(map(tuple, effective)),
            request.observation.time_s+max_age_s)
        self.queue.accept(plan, now_s=now_s, plan_version=self.memory.plan_version)
        self.events.append({'request_id': request_id, 'saturation_events': counts,
            'denominator': self.profile.horizon*7, 'time_profile': self.profile.name})
        return plan

    def tick(self, observation, *, safety, controller):
        self.synchronize_task()
        command = self.queue.next_for_control_time(observation.time_s)
        if command.target is None:
            controller.hold(observation, reason=command.reason)
            return command
        # Online safety must validate against the currently measured state, not
        # just the query anchor or old predicted path. No callback means no motion.
        safe, reason = safety(command.target, observation)
        if not safe:
            self.queue.cancel(reason='safety_stop_'+reason)
            controller.hold(observation, reason=reason)
            return QueueCommand(None, 'safety_stop_'+reason)
        applied = controller.apply(command.target, observation)
        self.queue.record_applied(observation.time_s, command, applied)
        return command
