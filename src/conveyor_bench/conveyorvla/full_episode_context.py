"""Public, bounded plan history shared by training and the online action backend."""
import json

ROUTES = ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE')

def public_context_text(context):
    if not isinstance(context,dict) or set(context) != {'completed_tasks','active_task','remaining_tasks','current_facts'}:
        raise ValueError('full-episode context requires explicit public memory fields')
    if context['current_facts'] != {}:
        raise ValueError('this context version has no calibrated current-fact input')
    if context['active_task'] not in ROUTES:
        raise ValueError('unknown active task')
    for key in ('completed_tasks','remaining_tasks'):
        values=context[key]
        if not isinstance(values,list) or len(values)>64 or any(v not in ROUTES for v in values):
            raise ValueError('invalid task history or remaining plan')
    if not context['remaining_tasks'] or context['remaining_tasks'][0]!=context['active_task']:
        raise ValueError('remaining suffix must start at the active task')
    return json.dumps(context,sort_keys=True,separators=(',',':'),ensure_ascii=False)
