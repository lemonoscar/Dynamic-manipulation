"""Explicit simulation assistance and evaluator-only physical event evidence."""
from __future__ import annotations

from dataclasses import replace
import math

from .joint_trajectory_system import measured_named_joint_state


class FormalPhysics:
    """Proxy the runtime without changing model routes or continuous targets.

    Both profiles preserve declared manipulation base/support locks. Only
    source_assisted adds a constraint after measured lift/close/proximity/stability.
    No profile permits resetting the object after episode initialization.
    """

    def __init__(self, simulation, profile, record):
        if profile not in {"source_assisted", "no_grasp_assist"}:
            raise ValueError("unknown formal physics profile")
        self.simulation, self.profile, self.record = simulation, profile, record
        self.armed = False
        self.initial_pose = None
        self.command_fraction = 1.
        self.previous_fraction = 1.
        self.closed_command_seen = False
        self.attached = False
        self.attached_once = False
        self.pick = False
        self.carry = False
        self.drop = False
        self.release = False
        self.latest = {}
        self.peak_lift = 0.

    def __getattr__(self, name):
        return getattr(self.simulation, name)

    def arm(self):
        state = self.simulation.read()
        if state.object_pose is None:
            raise ValueError("formal physical evidence requires initial object pose")
        self.initial_pose = tuple(state.object_pose)
        self.armed = True

    def prepare_object_for_pick(self, *_args, **_kwargs):
        raise RuntimeError("formal episodes forbid mid-episode object reset")

    def apply(self, action):
        metadata = dict(action.metadata)
        if metadata.get("joint_trajectory_policy"):
            manipulation = metadata.get("model_route") in {"PICK", "PLACE"}
            metadata["manipulation_base_lock"] = manipulation
            metadata["manipulation_support_joint_lock"] = manipulation
            metadata["formal_physics_profile"] = self.profile
            fraction = float(metadata["gripper_open_fraction_requested"])
            opening = fraction > self.previous_fraction + 1e-4
            if self.armed and self.attached and opening:
                report = self.simulation.release_grasp_constraint(reason="formal_continuous_opening_target")
                if report.get("active") is not False:
                    raise RuntimeError("grasp constraint release not confirmed")
                self.record("grasp_assistance_release", report)
                self.attached = False
            self.command_fraction = fraction
            self.previous_fraction = fraction
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
            raise ValueError("physical event evidence lacks object/TCP/velocity")
        values = [*state.object_pose, *state.tcp_pose, *state.object_velocity]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("nonfinite physical event evidence")
        lift = float(state.object_pose[2]) - self.initial_pose[2]
        self.peak_lift = max(self.peak_lift, lift)
        distance = math.dist(state.object_pose[:3], state.tcp_pose[:3])
        speed = math.sqrt(sum(float(v)**2 for v in state.object_velocity[:3]))
        fraction = measured_named_joint_state(state).gripper_open_fraction
        verified = self.closed_command_seen and fraction <= .5 and lift >= .04 and distance <= .08 and speed <= .30
        evidence = {"lift_m": lift, "peak_lift_m": self.peak_lift, "object_tcp_distance_m": distance,
                    "object_speed_mps": speed, "measured_gripper_fraction": fraction,
                    "closed_command_seen": self.closed_command_seen,
                    "contact_evidence": "geometry_proxy_not_contact_sensor"}
        if verified and not self.pick:
            self.pick = True
            self.record("physical_pick_verified", evidence)
        if verified and self.profile == "source_assisted" and not self.attached_once:
            report = self.simulation.create_verified_grasp_constraint()
            if report.get("active") is not True:
                raise RuntimeError("source grasp assistance did not activate")
            self.attached = self.attached_once = True
            self.record("grasp_assistance_attach", {"verification": evidence, "runtime": report})
        displacement = math.dist(state.object_pose[:2], self.initial_pose[:2])
        if self.pick and displacement >= .20 and distance <= .12 and not self.carry:
            self.carry = True
            self.record("physical_carry_verified", {**evidence, "displacement_m": displacement})
        released = fraction >= .8 and distance >= .06 and not self.attached
        if self.pick and distance > .20 and not released and not self.drop:
            self.drop = True
            self.record("physical_drop_proxy", evidence)
        if self.pick and released and not self.release:
            self.release = True
            self.record("physical_release_observed", evidence)
        self.latest = evidence

    def evidence(self):
        return {"profile": self.profile, "pick_verified": self.pick, "carry_verified": self.carry,
                "drop_detected": self.drop, "release_observed": self.release,
                "grasp_constraint_created": self.attached_once, "grasp_constraint_active": self.attached,
                "mid_episode_object_resets": 0, "manipulation_base_support_locks": True,
                "pure_physics": False, "latest": self.latest,
                "thresholds": {"lift_m": .04, "tcp_distance_m": .08, "speed_mps": .30,
                               "closed_fraction": .5, "carry_displacement_m": .20,
                               "drop_separation_m": .20},
                "note": "Source-inspired assistance migrated to 5.1; continuous-close admission differs from legacy binary counter."}
