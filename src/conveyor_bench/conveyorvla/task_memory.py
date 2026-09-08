"""Task-level rolling plan with causal evidence, append-only events and CAS edits."""
from dataclasses import dataclass, asdict, replace
from copy import deepcopy
import math

PRIMITIVES = {'NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE', 'VERIFY'}
OPERATIONS = {'CONTINUE', 'ADVANCE', 'REPAIR_SUFFIX', 'FINISH', 'UNRESOLVED'}


@dataclass(frozen=True)
class Task:
    task_id: str
    attempt_id: str
    primitive: str
    target_ref: str
    destination_ref: str | None = None
    preconditions: tuple = ()
    completion_conditions: tuple = ()
    invariants: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, 'preconditions', tuple(self.preconditions))
        object.__setattr__(self, 'completion_conditions', tuple(self.completion_conditions))
        # Older task records used preconditions for entry and continuous checks.
        object.__setattr__(self, 'invariants', tuple(self.preconditions if self.invariants is None else self.invariants))
        if self.primitive not in PRIMITIVES or not all((self.task_id, self.attempt_id, self.target_ref)):
            raise ValueError('invalid task identity/primitive')


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    mission_id: str
    observation_id: str
    observed_at_s: float
    source: str
    predicate: str
    value: bool | None
    validity_until_s: float | None = None

    def __post_init__(self):
        if self.source not in {'sensor_estimator', 'model_estimator', 'controller_feedback'}:
            raise ValueError('ordinary planner evidence cannot use evaluator/teacher truth')
        if self.value is not None and type(self.value) is not bool:
            raise ValueError('fact value must be bool or unknown (None)')
        if not all((self.evidence_id, self.mission_id, self.observation_id, self.predicate)):
            raise ValueError('missing evidence identity')
        if not math.isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError('invalid evidence time')
        if self.validity_until_s is not None and (not math.isfinite(self.validity_until_s)
                or self.validity_until_s < self.observed_at_s):
            raise ValueError('invalid fact expiry')


@dataclass(frozen=True)
class PlannerEdit:
    parent_plan_version: int
    observed_active_task_epoch: int
    observation_id: str
    operation: str
    evidence_refs: tuple = ()
    new_suffix: tuple | None = None
    changes_active_task_semantics: bool = False
    reason_code: str = 'unspecified'

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if values.get('new_suffix') is not None:
            values['new_suffix'] = tuple(Task(**t) for t in values['new_suffix'])
        return cls(**values)


class TaskMemory:
    def __init__(self, mission_id, original_instruction, tasks, *, mode='H2'):
        if mode not in {'H0', 'H1', 'H2'} or not mission_id or not original_instruction:
            raise ValueError('invalid planner mode/mission')
        self.mission_id = mission_id
        self.original_instruction = original_instruction
        self.instruction_version = 0
        self.mode = mode
        self.tasks = tuple(tasks)
        self._validate_tasks(self.tasks)
        self.plan_version = 0
        self.active_task_epoch = 0
        self.cursor = 0
        self.events = []
        self.current_facts = {}
        self.evidence = {}
        self.observations = {}
        self.last_model_output = None
        self.last_completed_task_id = None
        self.finished = False
        self._used_tasks = {t.task_id for t in tasks}
        self._used_attempts = {t.attempt_id for t in tasks}

    @staticmethod
    def _validate_tasks(tasks):
        if not tasks or len({t.task_id for t in tasks}) != len(tasks) or len({t.attempt_id for t in tasks}) != len(tasks):
            raise ValueError('tasks and attempts must be nonempty and unique')

    @property
    def active_task(self):
        return None if self.finished else self.tasks[self.cursor]

    def observe(self, observation, feedback=()):
        if observation.mission_id != self.mission_id:
            raise ValueError('cross-mission observation')
        if self.observations and observation.time_s < max(self.observations.values()):
            raise ValueError('out-of-order observation')
        if observation.observation_id in self.observations:
            raise ValueError('observation ID cannot be reused')
        if len({e.evidence_id for e in feedback}) != len(feedback):
            raise ValueError('duplicate feedback ID')
        for item in feedback:
            if (item.mission_id != self.mission_id or item.observation_id != observation.observation_id
                    or item.observed_at_s > observation.time_s or item.evidence_id in self.evidence):
                raise ValueError('invalid, future, cross-mission or duplicate feedback')
        self.observations[observation.observation_id] = observation.time_s
        for item in feedback:
            self.evidence[item.evidence_id] = item
            previous = self.current_facts.get(item.predicate)
            if previous is None or item.observed_at_s >= previous.observed_at_s:
                self.current_facts[item.predicate] = item

    def fact_value(self, predicate, now):
        fact = self.current_facts.get(predicate)
        if fact is None or fact.observed_at_s > now or (fact.validity_until_s is not None and now > fact.validity_until_s):
            return None
        return fact.value

    def _supported(self, predicates, refs, now):
        return bool(predicates) and all(self.fact_value(p, now) is True and
            self.current_facts[p].evidence_id in refs for p in predicates)

    def apply(self, edit):
        if self.finished:
            raise ValueError('mission already finished')
        if (edit.parent_plan_version != self.plan_version or
                edit.observed_active_task_epoch != self.active_task_epoch):
            raise ValueError('stale planner compare-and-swap')
        if edit.observation_id not in self.observations or edit.operation not in OPERATIONS:
            raise ValueError('unknown observation/operation')
        now = max(self.observations.values())
        query_time = self.observations[edit.observation_id]
        for ref in edit.evidence_refs:
            if ref not in self.evidence or self.evidence[ref].observed_at_s > query_time:
                raise ValueError('nonexistent/future evidence reference')
        old = self.active_task
        suffix = edit.new_suffix
        if edit.operation != 'REPAIR_SUFFIX' and suffix is not None:
            raise ValueError('suffix only legal for REPAIR_SUFFIX')
        if edit.operation in {'ADVANCE', 'FINISH'}:
            if not self._supported(old.completion_conditions, edit.evidence_refs, now):
                raise ValueError('completion unsupported by current causal facts')
            if edit.operation == 'FINISH' and (self.cursor != len(self.tasks)-1 or
                    not self._supported(('placed',), edit.evidence_refs, now)):
                raise ValueError('FINISH requires final task and current placed fact')
            if edit.operation == 'ADVANCE' and self.cursor == len(self.tasks)-1:
                raise ValueError('last task requires FINISH')
            if edit.operation == 'ADVANCE' and any(self.fact_value(p, now) is not True
                    for p in self.tasks[self.cursor+1].preconditions):
                raise ValueError('next task preconditions unsupported')
        elif edit.operation == 'REPAIR_SUFFIX':
            if self.mode != 'H2' or not suffix or not edit.evidence_refs:
                raise ValueError('repair requires H2, causal evidence and nonempty suffix')
            self._validate_tasks(suffix)
            start = self.cursor if edit.changes_active_task_semantics else self.cursor+1
            replaced = self.tasks[start:]
            for task in suffix:
                if task in replaced:
                    continue
                if task.task_id in self._used_tasks or task.attempt_id in self._used_attempts:
                    raise ValueError('retry/semantic edit must create fresh task and attempt IDs')
            candidate_tasks = self.tasks[:start] + tuple(suffix)
            self._validate_tasks(candidate_tasks)
            if edit.changes_active_task_semantics and suffix[0] == old:
                raise ValueError('active semantic change must replace current attempt')
        elif edit.changes_active_task_semantics:
            raise ValueError('CONTINUE/UNRESOLVED cannot change task semantics')
        # All validation is complete before mutation.
        self.last_model_output = asdict(edit)
        if edit.operation in {'ADVANCE', 'FINISH'}:
            self.events.append({'kind': 'TASK_COMPLETED', 'task': asdict(old),
                'time_s': now, 'evidence_refs': list(edit.evidence_refs)})
            self.last_completed_task_id = old.task_id
            if edit.operation == 'FINISH':
                self.finished = True
            else:
                self.cursor += 1
            self.active_task_epoch += 1
            self.plan_version += 1
        elif edit.operation == 'REPAIR_SUFFIX':
            self.events.append({'kind': 'TASKS_SUPERSEDED', 'tasks': [asdict(t) for t in replaced],
                'time_s': now, 'reason_code': edit.reason_code})
            self.tasks = candidate_tasks
            self._used_tasks.update(t.task_id for t in suffix)
            self._used_attempts.update(t.attempt_id for t in suffix)
            self.plan_version += 1
            if edit.changes_active_task_semantics:
                self.active_task_epoch += 1
        return self.active_task != old

    def planner_context(self, observation, feedback=()):
        current = {'original_instruction': self.original_instruction,
            'observation_id': observation.observation_id, 'time_s': observation.time_s,
            'feedback': [asdict(f) for f in feedback]}
        if self.mode == 'H0':
            return current
        current.update(plan_version=self.plan_version, active_task_epoch=self.active_task_epoch,
            active_task=asdict(self.active_task) if self.active_task else None,
            remaining_plan=[asdict(t) for t in self.tasks[self.cursor+1:]],
            event_history=deepcopy(self.events), last_model_output=deepcopy(self.last_model_output),
            last_completed_task_id=self.last_completed_task_id,
            current_facts={k: {**asdict(v), 'value': self.fact_value(k, observation.time_s)}
                for k, v in self.current_facts.items()})
        return current


def transfer_skeleton(target='cola', destination='destination'):
    names = ('NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE')
    conditions = ('source_reached', 'carrying', 'target_reached', 'placed')
    return tuple(Task(f'task-{i}', f'attempt-{i}', name, target, destination,
        ('carrying',) if i >= 2 else (), (conditions[i],),
        invariants=() if name=='PLACE' else None) for i, name in enumerate(names))
