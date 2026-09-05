from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from conveyor_bench.conveyorvla.formal_physics import FormalPhysics


@dataclass(frozen=True)
class Action:
    metadata: dict
    arm_joint_positions: tuple = (0.,) * 6


class Simulation:
    def __init__(self):
        self.state = SimpleNamespace(object_pose=(0., 0., 0., 1., 0., 0., 0.),
                                     tcp_pose=(0., 0., 0., 1., 0., 0., 0.), object_velocity=(0.,) * 6)
        self.attaches = 0
        self.releases = 0

    def read(self):
        return self.state

    def apply(self, action):
        self.last_action = action

    def create_verified_grasp_constraint(self):
        self.attaches += 1
        return {"active": True}

    def release_grasp_constraint(self, *, reason):
        self.releases += 1
        return {"active": False, "reason": reason}


@pytest.mark.parametrize("profile,expected_attach", [("source_assisted", 1), ("no_grasp_assist", 0)])
def test_assistance_requires_measured_lift_and_releases_on_continuous_open(monkeypatch, profile, expected_attach):
    monkeypatch.setattr("conveyor_bench.conveyorvla.formal_physics.measured_named_joint_state",
                        lambda _: SimpleNamespace(gripper_open_fraction=.2))
    simulation = Simulation()
    events = []
    proxy = FormalPhysics(simulation, profile, lambda *e: events.append(e))
    proxy.arm()
    action = Action(dict(joint_trajectory_policy=True, model_route="PICK", gripper_open_fraction_requested=.1))
    proxy.apply(action)
    proxy.observe(simulation.read())
    assert simulation.attaches == 0  # A route label and a closed target do not prove a grasp.
    simulation.state.object_pose = (0., 0., .05, 1., 0., 0., 0.)
    proxy.observe(simulation.read())
    assert simulation.attaches == expected_attach
    assert proxy.pick
    assert "manipulation_base_lock" not in action.metadata
    assert simulation.last_action.arm_joint_positions == action.arm_joint_positions
    proxy.apply(Action({**action.metadata, "gripper_open_fraction_requested": .3}))
    assert simulation.releases == expected_attach
    assert proxy.evidence()["mid_episode_object_resets"] == 0
    with pytest.raises(RuntimeError, match="forbid"):
        proxy.prepare_object_for_pick(None)
