"""One explicit bounded episode driver around the rolling runtime.

Environment-owned observer/feedback/safety/controller adapters supply legal
sensors and control. Evaluators are an independent result sink. The driver has
no access to simulator object truth and does not infer task completion from time.
"""
from concurrent.futures import ThreadPoolExecutor
import time


def run_rolling_episode(runtime, planner, action_backend, *, observer, feedback_estimator,
                        controller, safety, max_control_ticks, update_ticks=20,
                        planner_ticks=20, expected_delay_s=0., realtime=False,
                        navigation=None):
    """Run a bounded control loop; callbacks own the physical step/clock.

    realtime=False waits for action generation without advancing the environment:
    this is explicitly paused-simulation overlap diagnosis. realtime=True keeps
    ticking while one action worker runs; missing coverage goes through hold.
    NAV callback accepts versioned goal proposals and must enforce ReachConfig
    and online coverage. No NAV callback means an explicit safe hold.
    """
    if max_control_ticks<=0 or update_ticks<=0 or planner_ticks<=0:
        raise ValueError('episode budgets must be positive')
    pending=None;pending_request=None;planner_pending=None;planner_prepared=None
    latencies=[];failures=[];nav_goal=None
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rolling-action')
    planner_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='rolling-planner')
    try:
        for tick in range(max_control_ticks):
            observation=observer()
            feedback=tuple(feedback_estimator(observation))
            runtime.memory.observe(observation,feedback)
            if planner_pending is not None and (planner_pending.done() or not realtime):
                try:
                    changed=planner.commit_response(planner_prepared,planner_pending.result())
                    if changed:nav_goal=None
                except Exception as error:
                    failures.append({'tick':tick,'stage':'planner','reason':str(error)})
                planner_pending=None
            if tick%planner_ticks==0 and planner_pending is None:
                planner_prepared=planner.prepare(observation,feedback)
                def plan(request):
                    started=time.monotonic()
                    result=planner.backend(request)
                    latencies.append({'stage':'planner','wall_s':time.monotonic()-started})
                    return result
                planner_pending=planner_executor.submit(plan,planner_prepared[0])
                if not realtime:
                    try:
                        if planner.commit_response(planner_prepared,planner_pending.result()):nav_goal=None
                    except Exception as error:
                        failures.append({'tick':tick,'stage':'planner','reason':str(error)})
                    planner_pending=None
            runtime.synchronize_task()
            if runtime.memory.finished:
                controller.hold(observation,reason='planner_FINISH_not_evaluator_success')
                break
            if pending is not None and (pending.done() or not realtime):
                try:
                    result=pending.result()
                    proposal=runtime.complete_request(pending_request.request_id,result,now_s=observation.time_s)
                    if isinstance(proposal,dict):nav_goal=proposal
                except Exception as error:
                    # Worker failures are preserved, and cannot turn into success.
                    failures.append({'tick':tick,'stage':'action','reason':type(error).__name__+': '+str(error)})
                    runtime.pending.pop(pending_request.request_id,None)
                    runtime.queue.cancel(reason='action_service_failure')
                pending=None
            if tick%update_ticks==0 and pending is None and runtime.memory.active_task.primitive!='VERIFY':
                try:
                    pending_request=runtime.prepare_request(observation,expected_delay_s=expected_delay_s)
                    def generate(request):
                        started=time.monotonic()
                        result=action_backend(request,runtime.memory.original_instruction)
                        latencies.append({'stage':'action','wall_s':time.monotonic()-started})
                        return result
                    pending=executor.submit(generate,pending_request)
                    if not realtime:
                        proposal=runtime.complete_request(pending_request.request_id,pending.result(),now_s=observation.time_s)
                        if isinstance(proposal,dict):nav_goal=proposal
                        pending=None
                except Exception as error:
                    failures.append({'tick':tick,'stage':'action','reason':type(error).__name__+': '+str(error)})
                    if pending_request is not None:runtime.pending.pop(pending_request.request_id,None)
                    runtime.queue.cancel(reason='action_service_failure');pending=None
            if runtime.memory.active_task.primitive.startswith('NAV_'):
                if nav_goal is not None and nav_goal['identity']==runtime.identity() and navigation is not None:
                    navigation(nav_goal,observation)
                else:controller.hold(observation,reason='navigation_goal_or_safe_adapter_unavailable')
            elif runtime.memory.active_task.primitive=='VERIFY':
                controller.hold(observation,reason='verification_pending')
            else:
                runtime.tick(observation,safety=safety,controller=controller)
                if runtime.safety_stop_reason is not None:
                    break
        runtime.queue.cancel(reason='episode_budget_or_finish')
        return {'control_ticks':tick+1,'model_latencies':latencies,'failures':failures,
            'planner_finished':runtime.memory.finished,'geometry_transfer_success':None,
            'strict_full_success':None,'deployment_gate_passed':False,
            'safety_stop_reason':runtime.safety_stop_reason,
            'inference_pauses_simulation':not realtime,'planner_calls_synchronous':not realtime}
    finally:
        # Own worker only. Never terminate other processes/services.
        if pending is not None:pending.cancel()
        if planner_pending is not None:planner_pending.cancel()
        if runtime.safety_stop_reason is not None:
            controller.stop(observer(),reason=runtime.safety_stop_reason)
        else:
            controller.hold(observer(),reason='episode_final_hold')
        executor.shutdown(wait=True,cancel_futures=True)
        planner_executor.shutdown(wait=True,cancel_futures=True)
