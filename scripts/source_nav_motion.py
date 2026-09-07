"""Runtime geometry bounds and opt-in certified source navigation diagnostic."""
import math
import numpy as np
from dataclasses import asdict
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.waypoint import yaw_from_quaternion,wrap_to_pi
from conveyor_bench.conveyorvla.joint_trajectory_system import measured_named_joint_state
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointCommand

def live_full_reach(stage,cache,root,colliders):
    from pxr import UsdPhysics
    edges={};bodies=set();joint_evidence=[]
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(root+'/') or not prim.IsA(UsdPhysics.Joint):continue
        j=UsdPhysics.Joint(prim);a=j.GetBody0Rel().GetTargets();b=j.GetBody1Rel().GetTargets()
        if not a or not b:continue
        a,b=str(a[0]),str(b[0]);bodies.update((a,b));p0=np.asarray(j.GetLocalPos0Attr().Get());p1=np.asarray(j.GetLocalPos1Attr().Get());travel=0.
        if prim.IsA(UsdPhysics.PrismaticJoint):
            pj=UsdPhysics.PrismaticJoint(prim);travel=max(abs(float(pj.GetLowerLimitAttr().Get())),abs(float(pj.GetUpperLimitAttr().Get())))
        bound=float(np.linalg.norm(p0)+np.linalg.norm(p1)+travel);edges[b]=(a,bound);joint_evidence.append({'path':str(prim.GetPath()),'parent':a,'child':b,'translation_bound_m':bound})
    roots=bodies-set(edges)
    if roots!={root+'/base'}:raise ValueError(f'unverified runtime root relation: {roots}')
    def reach(link,seen=()):
        if link in roots:return 0.
        if link in seen or link not in edges:raise ValueError('incomplete joint tree')
        a,d=edges[link];return d+reach(a,(*seen,link))
    values=[]
    for c in colliders:
        if not c['robot']:continue
        if not c['bounds_valid']:raise ValueError('runtime collider has invalid bounds')
        link=max((b for b in bodies if c['path'].startswith(b+'/')),key=len,default=None)
        if link is None:raise ValueError('collider not bound to body')
        bb=cache.ComputeRelativeBound(stage.GetPrimAtPath(c['path']),stage.GetPrimAtPath(link)).ComputeAlignedRange();mn=np.array(bb.GetMin());mx=np.array(bb.GetMax())
        if np.any(mn>mx):raise ValueError('invalid body-relative collision bounds')
        radius=float(np.linalg.norm(np.maximum(np.abs(mn),np.abs(mx))));values.append({'collider':c['path'],'link':link,'radius_bound_m':reach(link)+radius})
    if not values:raise ValueError('no certified robot collision geometry')
    return {'radius_m':max(v['radius_bound_m'] for v in values)+.02,'contact_margin_m':.02,'root':root,'colliders':values,'joints':joint_evidence,'method':'live_body_graph_translation_triangle_inequality_all_joint_rotations_and_collision_AABB_corners'}

def execute_certified_motion(pipeline,path,goal,radius,arm_tracking_tolerance=None):
    p=pipeline;route=JointTrajectoryRoute.NAV_TO_SOURCE;state=p.simulation.read();initialz=state.robot_root_pose[2];initial_step=state.step_index;q=measured_named_joint_state(state);hold=DirectJointCommand(index=0,joint_position=q.joint_position,gripper_open_fraction=q.gripper_open_fraction)
    clean=[]
    for xy in path:
        if not clean or math.dist(clean[-1],xy)>1e-8:clean.append(tuple(xy))
    local=p._local_map(route);controller=p.system.navigation_executor.dwa_controller;reason='timeout';ticks=0;commands=[]
    try:
        for tick in range(500):
            state=p.simulation.read();pose=state.robot_root_pose;current=(pose[0],pose[1],yaw_from_quaternion(pose[3:]));distance=math.dist(current[:2],goal[:2]);yaw=wrap_to_pi(goal[2]-current[2])
            if distance<=.12 and abs(yaw)<=.14:reason='arrived';break
            if arm_tracking_tolerance is not None and np.max(np.abs(np.asarray(measured_named_joint_state(state).joint_position)-q.joint_position))>arm_tracking_tolerance:reason='arm_posture_tracking_guard';break
            if abs(pose[2]-initialz)>.10:reason='base_height_guard';break
            if float(state.metadata.get('nonfoot_contact_force_max',0))>1.:reason='nonfoot_contact_guard';break
            # Circle coverage was certified for every yaw, including any in-place turn.
            if distance<=.12:velocity=(0.,0.,max(-.4,min(.4,1.5*yaw)))
            elif len(clean)<2:reason='invalid_path';break
            else:velocity=controller.command(clean,current,state.metadata.get('body_velocity',(0.,0.,0.)),local)
            action=p.action_adapter.navigation(velocity,hold,route=route,sequence_id=tick)
            p._physical_step(action,route=route,command_index=None);ticks+=1
            after=p.simulation.read();commands.append({'tick':tick,'velocity':list(velocity),'actual_pose':list(after.robot_root_pose)})
    except Exception as error:reason=f'controller_error:{type(error).__name__}:{error}'
    final=p.simulation.read();pose=final.robot_root_pose;c=[pose[0],pose[1],yaw_from_quaternion(pose[3:])]
    p.simulation.apply(p.action_adapter.navigation((0.,0.,0.),hold,route=route,sequence_id=500))
    return {'status':'motion_complete','robot_motion_executed':ticks>0,'actual_arrival_C':c,'arrival_verified':reason=='arrived' and ticks>0,'already_within_tolerance':reason=='arrived' and ticks==0,'operation_handoff_verified':None,'termination_reason':reason,'position_error_m':math.dist(c[:2],goal[:2]),'yaw_error_rad':abs(wrap_to_pi(c[2]-goal[2])),'actual_control_ticks':ticks,'elapsed_simulation_s':ticks*.02,'actual_commands_and_path':commands,'actual_final_state':{'robot_root_pose':list(pose),'joint_positions':list(final.joint_positions),'joint_velocities':list(final.joint_velocities),'object_pose':list(final.object_pose),'object_velocity':list(final.object_velocity),'camera_history_available':list(p.frames.step_indices)},'initial_control_step':initial_step,'full_geometry_certificate_used':True,'controller_feasibility_verified':reason=='arrived' and ticks>0,'new_source_nav_protocol_not_formal_score':True}

def live_navigation_posture_bound(full,stage,cache,robot,current):
    """Full leg workspace plus held-arm pose with an explicit tracking-error condition."""
    from conveyor_bench.conveyorvla.physical_events import rotation_wxyz
    from pxr import UsdPhysics
    body_names=list(robot.body_names);body=robot.data.body_state_w[0].detach().cpu().numpy();rootpos=np.asarray(current.robot_root_pose[:3]);arm=[]
    for c in full['colliders']:
        if '/arm_' not in c['link']:continue
        name=c['link'].rsplit('/',1)[1];i=body_names.index(name);bb=cache.ComputeRelativeBound(stage.GetPrimAtPath(c['collider']),stage.GetPrimAtPath(c['link'])).ComputeAlignedRange();mn=np.asarray(bb.GetMin());mx=np.asarray(bb.GetMax());corners=np.array([[x,y,z] for x in [mn[0],mx[0]] for y in [mn[1],mx[1]] for z in [mn[2],mx[2]]]);world=(rotation_wxyz(body[i,3:7])@corners.T).T+body[i,:3];arm.append(float(np.linalg.norm(world-rootpos,axis=1).max()))
    travel=[]
    for j in full['joints']:
        prim=stage.GetPrimAtPath(j['path'])
        if prim.IsA(UsdPhysics.PrismaticJoint):
            pj=UsdPhysics.PrismaticJoint(prim);travel.append(abs(float(pj.GetUpperLimitAttr().Get())-float(pj.GetLowerLimitAttr().Get())))
    tolerance=.03;angular_margin=6*full['radius_m']*tolerance;gripper_margin=max(travel,default=0.)
    leg=max(c['radius_bound_m'] for c in full['colliders'] if '/arm_' not in c['link']);radius=max(leg,max(arm)+angular_margin+gripper_margin)+full['contact_margin_m']
    return {'radius_m':radius,'leg_all_rotation_bound_m':leg,'current_arm_collision_radius_m':max(arm),'arm_angular_error_bound_rad':tolerance,'arm_angular_geometry_margin_m':angular_margin,'gripper_full_travel_margin_m':gripper_margin,'contact_margin_m':full['contact_margin_m'],'condition':'hold source arm target; monitor all six joint errors every control tick; abort on violation','conditional_geometry_not_unconditional_physics_guarantee':True}
