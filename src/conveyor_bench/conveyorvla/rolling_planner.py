"""Common planner adapter; H0 is never shown server-side plan history."""
from dataclasses import asdict
from .task_memory import Task, PlannerEdit

PLANNER_SCHEMA = 'task-plan-v2'


class RollingPlanner:
    def __init__(self, memory, backend, *, model_id):
        self.memory = memory
        self.backend = backend
        self.model_id = model_id

    def prepare(self, observation, feedback=()):
        context = self.memory.planner_context(observation, feedback)
        request = {'schema': PLANNER_SCHEMA, 'mode': self.memory.mode,
            'context': context, 'images': observation.images,
            'allowed_operations': ['current_task_candidate'] if self.memory.mode == 'H0' else
                ['CONTINUE', 'ADVANCE', 'FINISH', 'UNRESOLVED'] +
                (['REPAIR_SUFFIX'] if self.memory.mode == 'H2' else [])}
        version = (self.memory.plan_version, self.memory.active_task_epoch)
        return request, version, observation.observation_id

    def request(self, observation, feedback=()):
        prepared = self.prepare(observation, feedback)
        return self.commit_response(prepared, self.backend(prepared[0]))

    def commit_response(self, prepared, response):
        request, version, observation_id = prepared
        if self.memory.mode != 'H0':
            edit = PlannerEdit.from_dict(response)
            if edit.observation_id != observation_id:
                raise ValueError('planner response observation mismatch')
            return self.memory.apply(edit)
        if version != (self.memory.plan_version, self.memory.active_task_epoch):
            raise ValueError('stale H0 response')
        if set(response) != {'current_task_candidate'}:
            raise ValueError('H0 must output only current_task_candidate')
        candidate = Task(**response['current_task_candidate'])
        old = self.memory.active_task
        if (candidate.primitive, candidate.target_ref, candidate.destination_ref,
                candidate.preconditions, candidate.completion_conditions, candidate.invariants) == (
                old.primitive, old.target_ref, old.destination_ref, old.preconditions, old.completion_conditions, old.invariants):
            self.memory.last_model_output = response
            return False
        if candidate.task_id in self.memory._used_tasks or candidate.attempt_id in self.memory._used_attempts:
            raise ValueError('H0 changed task requires fresh attempt identity')
        self.memory.last_model_output = response
        self.memory.tasks = (candidate,)
        self.memory.cursor = 0
        self.memory._used_tasks.add(candidate.task_id)
        self.memory._used_attempts.add(candidate.attempt_id)
        self.memory.plan_version += 1
        self.memory.active_task_epoch += 1
        return True


def planner_prompt(request):
    import json
    context=request['context']
    schema = {'current_task_candidate': {'task_id':'fresh ID','attempt_id':'fresh ID',
        'primitive':'PICK','target_ref':'cola','destination_ref':'destination'}} if request['mode']=='H0' else {
        'parent_plan_version':context['plan_version'],
        'observed_active_task_epoch':context['active_task_epoch'],
        'observation_id':context['observation_id'],'operation':'CONTINUE',
        'evidence_refs':[],'new_suffix':None,'changes_active_task_semantics':False,'reason_code':'observing'}
    prompt=('Return exactly one JSON object, no Markdown. Use only current observable evidence. '
        'A previous model answer is not completion evidence. Do not invent evidence IDs. '
        'Use UNRESOLVED if evidence is insufficient. Allowed operations: '+
        json.dumps(request['allowed_operations'])+'\nResponse shape example: '+json.dumps(schema)+
        '\nContext: '+json.dumps(context))
    return prompt


class QwenPlannerBackend:
    """Prompt-only/planner-adapted service, separate from frozen low-level weights.

    A JSON parse success is only a schema check. The old route model is not
    declared to understand this protocol until task-level adaptation is tested.
    """
    def __init__(self, qwen, *, max_new_tokens=768):
        self.qwen=qwen
        self.max_new_tokens=max_new_tokens

    def __call__(self, request):
        import json
        import torch
        prompt=planner_prompt(request)
        images=request['images']
        if len(images)!=4:raise ValueError('planner needs the same four legal RGB observations')
        inputs=dict(self.qwen.build_joint_trajectory_inputs([
            {'video':(images[:2],images[2:]),'lang':prompt}],supervise_solutions=False))
        inputs.pop('labels',None)
        with torch.inference_mode():
            output=self.qwen.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        text=self.qwen.processor.tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        return json.loads(text)
