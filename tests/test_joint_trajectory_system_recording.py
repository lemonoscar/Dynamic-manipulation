from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from conveyor_bench.conveyorvla.joint_trajectory import (
    JointTrajectoryDomain,
    JointTrajectoryRoute,
)
from conveyor_bench.conveyorvla.joint_trajectory_recording import (
    FreshJointTrajectoryEpisodeRecorder,
    applied_control_sample_from_isaac,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (
    DirectJointChunk,
    DirectJointCommand,
    JointTrajectoryRuntimeStep,
    navigation_reference,
)
from conveyor_bench.conveyorvla.joint_trajectory_system import (
    IsaacJointActionAdapter,
    IsaacJointTrajectorySystemExecutor,
    IsaacTransferTruthAdapter,
    JointControlTick,
    PCTDWAJointNavigationExecutor,
    PlacementValidArea,
)
from conveyor_bench.conveyorvla.waypoint_execution import PCTPlan


@dataclass(frozen=True)
class _RobotAction:
    base_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    arm_joint_positions: tuple[float, ...] | None = None
    gripper_command: str | None = None
    source: str = "idle"
    metadata: dict = field(default_factory=dict)


class _PCT:
    def __init__(self):
        self.calls = []

    def plan(self, current, goal):
        self.calls.append((tuple(current), tuple(goal)))
        return PCTPlan(
            path_world=((current[0], current[1]), (goal[0], goal[1])),
            snapped_goal_world=tuple(goal),
            snap_distance_m=0.0,
            metadata={"planner": "pct"},
        )


class _DWA:
    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def command(self, path, pose, velocity, local_map):
        self.calls.append((tuple(path), tuple(pose), tuple(velocity), local_map))
        return (0.50, 0.0, 0.60)

    def reset(self):
        self.reset_count += 1


class _FailingDWA(_DWA):
    def command(self, path, pose, velocity, local_map):
        raise RuntimeError("controller unavailable")


class _Simulation:
    names = tuple(f"arm_joint{index}" for index in range(1, 9))

    def __init__(self):
        self.actions = []
        self._pending = None
        self.render_flags = []
        self.state = self._state(
            step=0,
            timestamp=0.0,
            positions=(0.0,) * 8,
            velocities=(0.0,) * 8,
            metadata={
                "joint_names": self.names,
                "body_velocity": (0.0, 0.0, 0.0),
                "arm_joint_position_target_apply_count": 0,
                "gripper_joint_position_target_apply_count": 0,
                "command_seen_vx": 0.0,
                "command_seen_vy": 0.0,
                "command_seen_wz": 0.0,
            },
        )

    def _state(self, *, step, timestamp, positions, velocities, metadata):
        return SimpleNamespace(
            step_index=step,
            timestamp=timestamp,
            robot_root_pose=(0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0),
            robot_root_velocity=(0.0,) * 6,
            joint_positions=tuple(positions),
            joint_velocities=tuple(velocities),
            object_pose=None,
            tcp_pose=None,
            metadata=dict(metadata),
        )

    def read(self):
        return self.state

    def apply(self, action):
        self._pending = action

    def step(self, *, render):
        action = self._pending
        assert action is not None
        self.actions.append(action)
        self.render_flags.append(render)
        previous = self.state
        positions = list(previous.joint_positions)
        velocities = [0.0] * len(positions)
        if action.arm_joint_positions is not None:
            for index, value in enumerate(action.arm_joint_positions):
                velocities[index] = (float(value) - positions[index]) / 0.02
                positions[index] = float(value)
        gripper_positions = tuple(action.metadata["gripper_joint_positions"])
        for offset, value in enumerate(gripper_positions, start=6):
            positions[offset] = float(value)
        metadata = dict(previous.metadata)
        arm_count = int(metadata["arm_joint_position_target_apply_count"]) + 1
        grip_count = int(metadata["gripper_joint_position_target_apply_count"]) + 1
        metadata.update(
            {
                "joint_names": self.names,
                "body_velocity": tuple(action.base_velocity),
                "command_seen_vx": float(action.base_velocity[0]),
                "command_seen_vy": float(action.base_velocity[1]),
                "command_seen_wz": float(action.base_velocity[2]),
                "arm_joint_position_target_apply_count": arm_count,
                "gripper_joint_position_target_apply_count": grip_count,
                "last_arm_joint_position_target_report": {
                    "applied": True,
                    "target_positions": list(action.arm_joint_positions),
                    "apply_count": arm_count,
                },
                "last_gripper_joint_position_target_report": {
                    "applied": True,
                    "target_positions": [gripper_positions[0]],
                    "apply_count": grip_count,
                },
            }
        )
        self.state = self._state(
            step=previous.step_index + 1,
            timestamp=previous.timestamp + 0.02,
            positions=positions,
            velocities=velocities,
            metadata=metadata,
        )
        self._pending = None


def _runtime_step(*, route, navigation=None, manipulation=None, hold=None):
    domain = (
        JointTrajectoryDomain.NAVIGATION
        if route in {JointTrajectoryRoute.NAV_TO_SOURCE, JointTrajectoryRoute.NAV_TO_TARGET}
        else JointTrajectoryDomain.MANIPULATION
    )
    return JointTrajectoryRuntimeStep(
        request_id="request-2",
        sequence_id=2,
        predicted_route=route,
        committed_route=route,
        commit_status=None,
        route_probs={candidate.value: 0.7 if candidate is route else 0.1 for candidate in JointTrajectoryRoute},
        subtask="test",
        action_domain=domain,
        navigation=navigation,
        manipulation=manipulation,
        hold=hold,
        pass2_executed=True,
        checkpoint_id="checkpoint",
        normalization_sha256="normalizer",
        elapsed_ms=1.0,
    )


@pytest.mark.parametrize('current,reason,reached',[
    ((0.,0.,0.,1.,0.,0.,0.),'local_goal_reached',True),
    ((0.,0.,0.,math.cos(.5),0.,0.,math.sin(.5)),'validated_in_place_turn_required',False),
    ((.2,0.,0.,1.,0.,0.,0.),'reconnect_from_measured_pose_required',False),
])
def test_collapsed_path_never_reaches_empty_segment_projection(current,reason,reached):
    class Collapsed:
        def plan(self,start,goal):
            return PCTPlan(((0.,0.),(0.,0.)),(0.,0.,0.,0.),0.)
    dwa=_DWA(); executor=PCTDWAJointNavigationExecutor(Collapsed(),dwa)
    executor.begin(navigation_reference([[0.,0.,0.]]*10),(0.,0.,0.,1.,0.,0.,0.),timestamp_s=0.)
    command=executor.command(current,(0.,0.,0.),None,timestamp_s=.02)
    assert command.requires_requery and command.reached_local_goal is reached
    assert command.reason==reason and command.base_velocity==(0.,0.,0.)
    assert not dwa.calls


def test_nav_preserves_ten_points_uses_point_ten_and_runs_exact_two_seconds():
    pct, dwa, simulation = _PCT(), _DWA(), _Simulation()
    action_adapter = IsaacJointActionAdapter(_RobotAction)
    navigation_executor = PCTDWAJointNavigationExecutor(pct, dwa)
    executor = IsaacJointTrajectorySystemExecutor(
        simulation, action_adapter, navigation_executor, render=False
    )
    reference = navigation_reference(
        [[0.20 * (index + 1), 0.01 * index, 0.0] for index in range(10)]
    )
    hold = DirectJointCommand(
        index=0,
        joint_position=(0.0,) * 6,
        gripper_open_fraction=0.37,
    )
    result = executor.execute(
        _runtime_step(
            route=JointTrajectoryRoute.NAV_TO_SOURCE,
            navigation=reference,
            hold=hold,
        ),
        local_map="map",
    )
    assert not result.failed and result.requires_requery
    assert result.control_ticks == 100
    assert result.trace["local_goal_reached"] is False
    assert result.trace["requested_goal_A_xyyaw"][:2] == pytest.approx([2., .09])
    assert len(result.trace["final_measured_pose_C_xyyaw"]) == 3
    assert len(result.trace["reference_world"]) == 10
    assert result.trace["pct_input_mode"] == "endpoint_only_approved_api"
    assert result.trace["pct_endpoint_reference_index"] == 9
    assert pct.calls[0][1][0] == pytest.approx(2.0)
    assert pct.calls[0][1][1] == pytest.approx(0.09)
    assert len(dwa.calls) == len(simulation.actions) == 100
    assert all(action.base_velocity == (0.30, 0.0, 0.35) for action in simulation.actions)
    assert all(action.gripper_command == "hold" for action in simulation.actions)
    assert all(action.metadata["segment_type"] == "post_motion_hold" for action in simulation.actions)
    assert all(
        action.metadata["gripper_joint_positions"] == pytest.approx((0.0148, 0.0148))
        for action in simulation.actions
    )
    assert all(action.metadata["uses_prefix_selector"] is False for action in simulation.actions)
    assert dwa.reset_count == 1


def test_mani_executes_all_ten_targets_for_ten_ticks_with_zero_base_and_no_ik():
    simulation = _Simulation()
    ticks = []
    executor = IsaacJointTrajectorySystemExecutor(
        simulation,
        IsaacJointActionAdapter(_RobotAction),
        PCTDWAJointNavigationExecutor(_PCT(), _DWA()),
        on_control_tick=ticks.append,
    )
    commands = tuple(
        DirectJointCommand(
            index=index,
            joint_position=tuple(0.01 * (index + 1) for _ in range(6)),
            gripper_open_fraction=index / 9.0,
        )
        for index in range(10)
    )
    chunk = DirectJointChunk(commands, 0, 0, 0)
    result = executor.execute(
        _runtime_step(route=JointTrajectoryRoute.PICK, manipulation=chunk)
    )
    assert not result.failed and result.control_ticks == 100
    assert len(ticks) == len(simulation.actions) == 100
    assert [tick.command_index for tick in ticks] == [index for index in range(10) for _ in range(10)]
    for index in range(10):
        actions = simulation.actions[10 * index : 10 * index + 10]
        assert all(action.arm_joint_positions == commands[index].joint_position for action in actions)
    assert all(action.base_velocity == (0.0, 0.0, 0.0) for action in simulation.actions)
    assert all(action.metadata["manipulation_base_lock"] is False for action in simulation.actions)
    assert all(action.metadata["segment_type"] == "direct_joint_motion" for action in simulation.actions)
    assert all(action.metadata["uses_ik"] is False for action in simulation.actions)
    assert all(action.metadata["uses_curobo"] is False for action in simulation.actions)


def test_dwa_failure_applies_one_zero_hold_then_stops_fail_closed():
    simulation = _Simulation()
    executor = IsaacJointTrajectorySystemExecutor(
        simulation,
        IsaacJointActionAdapter(_RobotAction),
        PCTDWAJointNavigationExecutor(_PCT(), _FailingDWA()),
    )
    hold = DirectJointCommand(0, (0.0,) * 6, 1.0)
    result = executor.execute(
        _runtime_step(
            route=JointTrajectoryRoute.NAV_TO_SOURCE,
            navigation=navigation_reference([[0.1 * (index + 1), 0.0, 0.0] for index in range(10)]),
            hold=hold,
        ),
        local_map="map",
    )
    assert result.failed and not result.requires_requery
    assert result.control_ticks == 1
    assert result.reason.startswith("dwa_control_failed:RuntimeError")
    assert simulation.actions[-1].base_velocity == (0.0, 0.0, 0.0)
    assert simulation.actions[-1].source == "joint_trajectory_dwa_control_failed_hold"
    assert "requested_goal_A_xyyaw" in result.trace
    assert "planned_endpoint_B_xyyaw" in result.trace


def test_rejected_pct_endpoint_preserves_requested_and_planned_pose():
    class MovedEndpoint(_PCT):
        def plan(self, current, goal):
            moved = (goal[0] + .1253, goal[1], goal[2], goal[3])
            return PCTPlan(path_world=((current[0], current[1]), moved[:2]),
                           snapped_goal_world=moved, snap_distance_m=.1253, metadata={})
    simulation, dwa = _Simulation(), _DWA()
    executor = IsaacJointTrajectorySystemExecutor(
        simulation, IsaacJointActionAdapter(_RobotAction),
        PCTDWAJointNavigationExecutor(MovedEndpoint(), dwa))
    result = executor.execute(_runtime_step(
        route=JointTrajectoryRoute.NAV_TO_SOURCE,
        navigation=navigation_reference([[.1*(i+1), 0., 0.] for i in range(10)]),
        hold=DirectJointCommand(0, (0.,)*6, 1.)), local_map="map")
    assert result.failed and "endpoint snap exceeds" in result.reason
    assert result.trace["endpoint_change_B_minus_A_m"] == pytest.approx(.1253)
    assert result.trace["planned_endpoint_B_xyyaw"][0] - result.trace["requested_goal_A_xyyaw"][0] == pytest.approx(.1253)
    assert not dwa.calls


def test_applied_control_row_requires_fresh_isaac_reports():
    simulation = _Simulation()
    events = []
    executor = IsaacJointTrajectorySystemExecutor(
        simulation,
        IsaacJointActionAdapter(_RobotAction),
        PCTDWAJointNavigationExecutor(_PCT(), _DWA()),
        on_control_tick=events.append,
    )
    command = DirectJointCommand(0, (0.01,) * 6, 0.5)
    executor.execute(
        _runtime_step(
            route=JointTrajectoryRoute.PLACE,
            manipulation=DirectJointChunk((command,) * 10, 0, 0, 0),
        )
    )
    row = applied_control_sample_from_isaac(
        events[0], control_tick_id=0, model_tick_id=7
    )
    assert row["q_command_applied"] == pytest.approx([0.01] * 6)
    assert row["gripper_command_applied"] == pytest.approx(0.5)
    assert row["base_command_requested"] == row["base_command_applied"] == [0.0, 0.0, 0.0]
    stale = JointControlTick(
        0,
        0,
        JointTrajectoryRoute.PLACE,
        events[0].action,
        events[0].state_after,
        events[0].state_after,
    )
    with pytest.raises(ValueError, match="did not advance"):
        applied_control_sample_from_isaac(stale, control_tick_id=1, model_tick_id=7)


def _truth_state(timestamp):
    return SimpleNamespace(
        timestamp=timestamp,
        object_pose=(0.5, 0.5, 0.15, 1.0, 0.0, 0.0, 0.0),
        tcp_pose=(0.5, 0.5, 0.30, 1.0, 0.0, 0.0, 0.0),
        joint_positions=(0.0,) * 6 + (0.04, 0.04),
        joint_velocities=(0.0,) * 8,
        metadata={
            "joint_names": tuple(f"arm_joint{index}" for index in range(1, 9)),
            "grasp_fixed_joint_report": {"active": False, "released": False},
        },
    )


def test_truth_adapter_needs_release_inside_and_one_second_but_stays_evaluator_only():
    adapter = IsaacTransferTruthAdapter(
        PlacementValidArea(0.0, 1.0, 0.0, 1.0, 0.10)
    )
    first = adapter.update(_truth_state(0.0))
    middle = adapter.update(_truth_state(0.5))
    final = adapter.update(_truth_state(1.0))
    assert first.released and first.inside_target_valid_area
    assert not first.success.success and not middle.success.success
    assert final.success.success
    assert final.release_source == "measured_gripper_and_object_tcp_separation"


def _control_row(tick):
    return {
        "tick_id": tick,
        "sim_step": tick,
        "model_tick": tick // 10,
        "timestamp_s": 0.02 * tick,
        "q_measured": [0.0] * 6,
        "dq_measured": [0.0] * 6,
        "gripper_measured": 1.0,
        "q_command_requested": [0.001 * tick] * 6,
        "q_command_applied": [0.001 * tick] * 6,
        "gripper_command_requested": 1.0,
        "gripper_command_applied": 1.0,
        "base_command_requested": [0.0, 0.0, 0.0],
        "base_command_applied": [0.0, 0.0, 0.0],
        "base_pose_world": [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        "base_twist_world": [0.0] * 6,
        "route": "PICK",
        "q_command_source": "controller_applied_after_saturation",
    }


def test_raw_recorder_publishes_atomic_episode_and_keeps_overview_out_of_model_assets(tmp_path):
    recorder = FreshJointTrajectoryEpisodeRecorder(
        tmp_path,
        episode_id="episode-001",
        split="train",
        episode_metadata={"teacher": "fresh-controller-v1"},
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    head_old = recorder.save_camera_frame("head", frame_id="h0", timestamp_s=0.0, image=image)
    wrist_old = recorder.save_camera_frame("wrist", frame_id="w0", timestamp_s=0.0, image=image)
    head_now = recorder.save_camera_frame("head", frame_id="h10", timestamp_s=0.2, image=image)
    wrist_now = recorder.save_camera_frame("wrist", frame_id="w10", timestamp_s=0.2, image=image)
    overview = recorder.save_camera_frame("overview", frame_id="o10", timestamp_s=0.2, image=image)
    for tick in range(11):
        recorder.record_control(_control_row(tick))
    recorder.record_query(
        {
            "sample_id": "sample-001",
            "episode_id": "episode-001",
            "split": "train",
            "control_tick_id": 10,
            "history_timestamps_s": [0.0, 0.2],
            "global_instruction": "Move the Coke from box1 to box2.",
            "head_images": [head_old.relative_path, head_now.relative_path],
            "wrist_images": [wrist_old.relative_path, wrist_now.relative_path],
            "overview_images": [overview.relative_path],
            "route": "PICK",
            "physical_progress_valid": True,
            "physical_progress": 0.4,
            "physical_progress_provenance": "pick_reach_alignment_grasp_lift",
            "transition_window": False,
        }
    )
    final = recorder.finalize(
        success=True,
        outcome_metadata={"released_inside_target_dwell_s": 1.0},
    )
    assert final == tmp_path / "episode-001"
    assert not recorder.staging_path.exists() and recorder.finalized
    summary = json.loads((final / "summary.json").read_text())
    assert summary["success"] is True
    assert summary["control_row_count"] == 11 and summary["query_row_count"] == 1
    query = json.loads((final / "joint_queries_5hz.jsonl").read_text())
    assert all("overview" not in path for path in query["head_images"] + query["wrist_images"])
    assert (final / query["overview_images"][0]).is_file()


def test_raw_recorder_rejects_non_50hz_control_and_missing_camera_asset(tmp_path):
    recorder = FreshJointTrajectoryEpisodeRecorder(
        tmp_path,
        episode_id="episode-bad",
        split="val",
        episode_metadata={},
    )
    recorder.record_control(_control_row(0))
    bad = _control_row(1)
    bad["timestamp_s"] = 0.03
    with pytest.raises(ValueError, match="50 Hz"):
        recorder.record_control(bad)
    with pytest.raises(ValueError, match="missing"):
        recorder.record_query(
            {
                "sample_id": "sample-bad",
                "episode_id": "episode-bad",
                "split": "val",
                "control_tick_id": 0,
                "history_timestamps_s": [-0.2, 0.0],
                "global_instruction": "move",
                "head_images": ["images/head/a.jpg", "images/head/b.jpg"],
                "wrist_images": ["images/wrist/a.jpg", "images/wrist/b.jpg"],
                "route": "PICK",
                "physical_progress_valid": False,
                "physical_progress": None,
            }
        )
