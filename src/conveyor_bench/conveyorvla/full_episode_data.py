"""Shared RGB task context and transition prompt for offline training and deployment."""
import json

STAGES = ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE')


def task_context(active, completed):
    return dict(completed_tasks=list(completed),active_task=active,
        remaining_tasks=list(STAGES[STAGES.index(active):]),current_facts={})


def transition_prompt(row):
    """Serialize only public causal inputs; never labels or evaluator evidence."""
    value=row['input']
    return ('Use the four RGB observations and the committed task memory to decide whether to '
        'CONTINUE the active task or ADVANCE to the next task. Completed tasks are historical '
        'events, not evidence that an object is currently held. Unknown current facts remain '
        'unknown. Return only JSON with operation and next_task.\n'+json.dumps(
        {'original_instruction':value['original_instruction'],'task_context':value['task_context']},sort_keys=True))


def transition_response(row):
    return json.dumps(row['label'],sort_keys=True)
