"""Runtime proxy composing read-only event evaluation and explicit assistance."""
from __future__ import annotations
from dataclasses import replace
import math
from .joint_trajectory_system import measured_named_joint_state
from .physical_events import LegacyPhysicalEventEvaluator, RelativeGraspEvaluator
from .grasp_assistance import GraspAssistanceController


class FormalPhysics:
    """Keep frozen v1 scores/assistance while adding independent v2 evidence.

    Both profiles retain declared manipulation base/support locks. No evaluator
    applies constraints; only the separately versioned assistance controller can.
    """
    def __init__(self, simulation, profile, record):
        self.simulation, self.profile, self.record = simulation, profile, record
        self.assistance = GraspAssistanceController(simulation, profile, record)
        self.evaluator = LegacyPhysicalEventEvaluator(record)
        self.relative_evaluator = None
        self.armed = False
        self.initial_pose = None
        self.command_fraction = self.previous_fraction = 1.
        self.closed_command_seen = False

    def __getattr__(self, name):
        if name in {'pick','carry','drop','release','latest','peak_lift'}:
            return getattr(self.evaluator, name)
        if name in {'attached','attached_once'}:
            return self.assistance.active if name == 'attached' else self.assistance.created
        return getattr(self.simulation, name)

    def arm(self):
        state = self.simulation.read()
        if state.object_pose is None:
            raise ValueError('formal physical evidence requires initial object pose')
        self.initial_pose = tuple(state.object_pose)
        self.relative_evaluator = RelativeGraspEvaluator(self.initial_pose[2], self.record)
        self.armed = True

    def prepare_object_for_pick(self, *_args, **_kwargs):
        raise RuntimeError('formal episodes forbid mid-episode object reset')

    def apply(self, action):
        metadata = dict(action.metadata)
        if metadata.get('joint_trajectory_policy'):
            manipulation = metadata.get('model_route') in {'PICK','PLACE'}
            metadata['manipulation_base_lock'] = manipulation
            metadata['manipulation_support_joint_lock'] = manipulation
            metadata['formal_physics_profile'] = self.profile
            fraction = float(metadata['gripper_open_fraction_requested'])
            if self.armed:
                self.assistance.before_command(opening=fraction > self.previous_fraction+1e-4)
            self.command_fraction = self.previous_fraction = fraction
            self.closed_command_seen |= self.armed and fraction <= .5
            action = replace(action, metadata=metadata)
        self.simulation.apply(action)

    def step(self, *args, **kwargs):
        result = self.simulation.step(*args, **kwargs)
        if self.armed:
            self.observe(self.simulation.read())
        return result

    def observe(self, state):
        if state.object_pose is None or state.tcp_pose is None or state.object_velocity is None:
            raise ValueError('physical event evidence lacks object/TCP/velocity')
        if not all(math.isfinite(float(v)) for v in [*state.object_pose,*state.tcp_pose,*state.object_velocity]):
            raise ValueError('nonfinite physical event evidence')
        measurement = dict(lift=float(state.object_pose[2])-self.initial_pose[2],
                           distance=math.dist(state.object_pose[:3],state.tcp_pose[:3]),
                           speed=math.sqrt(sum(float(v)**2 for v in state.object_velocity[:3])),
                           fraction=measured_named_joint_state(state).gripper_open_fraction,
                           closed_command_seen=self.closed_command_seen)
        self.evaluator.observe(**measurement, displacement=math.dist(state.object_pose[:2],self.initial_pose[:2]),
                               attached=self.assistance.active)
        if getattr(state, 'timestamp', None) is not None:
            # A contact provider must explicitly report finger/target pairs and
            # external support. Absence is unknown, never inferred from distance.
            provider = getattr(self.simulation, 'read_grasp_contacts', None)
            contacts = provider() if callable(provider) else None
            self.relative_evaluator.observe(state, self.command_fraction, contacts=contacts)
        self.assistance.observe(**measurement)

    def evidence(self):
        return {'profile':self.profile, 'pick_verified':self.pick, 'carry_verified':self.carry,
                'drop_detected':self.drop, 'release_observed':self.release,
                'grasp_constraint_created':self.assistance.created, 'grasp_constraint_active':self.assistance.active,
                'assistance_rule_id':self.assistance.rule_id,
                'mid_episode_object_resets':0, 'manipulation_base_support_locks':True,
                'pure_physics':False, 'latest':self.latest,
                'thresholds':{'lift_m':.04,'tcp_distance_m':.08,'speed_mps':.30,
                              'closed_fraction':.5,'carry_displacement_m':.20,'drop_separation_m':.20},
                'evaluation_v2':None if self.relative_evaluator is None else self.relative_evaluator.evidence(),
                'note':'Legacy scores and assistance preserved independently; v2 proxy does not establish contact or activate assistance.'}
