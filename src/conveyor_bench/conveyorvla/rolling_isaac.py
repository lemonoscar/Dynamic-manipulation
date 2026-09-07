"""Isaac I/O binding for the rolling episode driver; no evaluator-driven policy.

Use the existing initialized simulation and IsaacJointActionAdapter. Sensor
capture timestamps and online safety certificates remain explicit caller inputs.
"""
from collections import deque
import math
from .contracts.observation import CurrentObservation
from .joint_trajectory import JointTrajectoryRoute
from .joint_trajectory_runtime import DirectJointCommand,NavigationReference
from .joint_trajectory_system import measured_named_joint_state,measured_body_velocity
from .waypoint import yaw_from_quaternion


class IsaacRollingAdapter:
    def __init__(self,simulation,action_adapter,memory,*,camera_reader,
                 navigation_executor,local_map,record,render=True):
        self.simulation=simulation;self.action_adapter=action_adapter;self.memory=memory
        self.camera_reader=camera_reader
        self.navigation_executor=navigation_executor;self.local_map=local_map
        self.record=record;self.render=render;self.frames=deque(maxlen=32)
        self._nav_identity=None;self._nav_query_time=None;self._nav_hold=None
        self._last_observation=None

    def observe(self):
        state=self.simulation.read();joints=measured_named_joint_state(state)
        time_s=float(state.timestamp)
        # camera_reader returns (capture_time, head RGB, wrist RGB), or None.
        # It must use sensor capture evidence, never relabel an old image as now.
        capture=self.camera_reader(state)
        if capture is not None:
            stamp,head,wrist=capture
            if not math.isfinite(stamp) or stamp>time_s+1e-8:
                raise ValueError('invalid/future sensor capture')
            if not self.frames or stamp>self.frames[-1][0]:
                self.frames.append((stamp,head.copy(),wrist.copy()))
        previous=next((f for f in self.frames if math.isclose(f[0],time_s-.2,abs_tol=1e-7)),None)
        current=next((f for f in self.frames if math.isclose(f[0],time_s,abs_tol=1e-7)),None)
        images=() if previous is None or current is None else (previous[1],current[1],previous[2],current[2])
        times=() if not images else (previous[0],current[0],previous[0],current[0])
        root=state.robot_root_pose
        observation=CurrentObservation(self.memory.mission_id,f'{self.memory.mission_id}:tick-{state.step_index}',
            time_s,joints.joint_position,joints.joint_velocity,joints.gripper_open_fraction,
            images,times,(root[0],root[1],yaw_from_quaternion(root[3:])))
        self._last_observation=observation
        return observation

    def _route(self):
        task=self.memory.active_task
        return None if task is None or task.primitive=='VERIFY' else JointTrajectoryRoute(task.primitive)

    def _step(self,action,reason):
        before=self.simulation.read()
        self.simulation.apply(action)
        self.simulation.step(render=self.render)
        after=self.simulation.read()
        if not math.isclose(after.timestamp-before.timestamp,.02,abs_tol=1e-7):
            raise ValueError('Isaac rolling control must advance exactly 0.02s')
        self.record({'kind':'control_apply','reason':reason,'time_s':before.timestamp,
            'post_time_s':after.timestamp,'task_epoch':self.memory.active_task_epoch,
            'source':action.source,'base_velocity':action.base_velocity,
            'arm_joint_positions':action.arm_joint_positions,'gripper_command':action.gripper_command,
            'provenance':'controller_target_applied_not_motor_torque'})

    def apply(self,target,observation):
        if self._route() not in {JointTrajectoryRoute.PICK,JointTrajectoryRoute.PLACE}:
            raise ValueError('Mani queue cannot execute under NAV/VERIFY/FINISH')
        command=DirectJointCommand(index=0,joint_position=tuple(target[:6]),gripper_open_fraction=target[6])
        action=self.action_adapter.manipulation(command,route=self._route(),sequence_id=self.memory.active_task_epoch)
        self._step(action,'rolling_mani')
        return tuple(target)

    def hold(self,observation,reason):
        # A measured hold belongs to this episode, never to a previous target cache.
        state=measured_named_joint_state(self.simulation.read())
        command=DirectJointCommand(index=0,joint_position=state.joint_position,
            gripper_open_fraction=state.gripper_open_fraction)
        action=self.action_adapter.hold(command,route=self._route(),sequence_id=self.memory.active_task_epoch,
            source='rolling_hold_'+reason)
        self._step(action,reason)

    def navigation(self,proposal,observation):
        identity=proposal['identity']
        if (identity.mission_id!=self.memory.mission_id or identity.active_task_epoch!=self.memory.active_task_epoch
                or identity.active_task_id!=self.memory.active_task.task_id or observation.time_s>=proposal['valid_until_s']):
            self.hold(observation,'stale_navigation_proposal');return
        executor=self.navigation_executor
        if executor is None or not executor.require_online_safety:
            self.hold(observation,'B0_navigation_safety_not_configured');return
        state=self.simulation.read()
        try:
            if (self._nav_identity,self._nav_query_time)!=(identity,proposal['query_time_s']):
                x,y,yaw=proposal['query_base_xyyaw']
                # This adapter is explicitly for planar navigation. Current z is
                # only an input to the map's floor selection, never a changed XY anchor.
                root=(x,y,state.robot_root_pose[2],math.cos(yaw/2),0.,0.,math.sin(yaw/2))
                points=tuple(map(tuple,proposal['reference_query_body']))
                executor.begin(NavigationReference(points,points[-1],.2),root,timestamp_s=observation.time_s)
                self._nav_identity=identity;self._nav_query_time=proposal['query_time_s']
                measured=measured_named_joint_state(state)
                self._nav_hold=DirectJointCommand(0,measured.joint_position,measured.gripper_open_fraction)
            control=executor.command(state.robot_root_pose,measured_body_velocity(state),self.local_map(self._route()),
                timestamp_s=observation.time_s)
            self.record({'kind':'navigation_feedback','reason':control.reason,'trace':control.trace,
                'reached_local_goal':control.reached_local_goal,'time_s':observation.time_s})
            if control.requires_requery:
                self.hold(observation,control.reason or 'navigation_requery');return
            action=self.action_adapter.navigation(control.base_velocity,self._nav_hold,
                route=self._route(),sequence_id=self.memory.active_task_epoch)
            self._step(action,'rolling_nav')
        except (ValueError,RuntimeError) as error:
            executor.reset();self._nav_identity=None
            self.record({'kind':'navigation_failure','reason':str(error),'time_s':observation.time_s})
            self.hold(observation,'navigation_failure')
