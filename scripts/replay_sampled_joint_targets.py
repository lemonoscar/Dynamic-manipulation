#!/usr/bin/env python3
"""Isolated PICK replay from a validation demonstration, with no VLM/DiT calls.

The source pre-action state is installed once before evaluation starts. This is
a phase-start interface diagnostic, never full-task or policy success evidence.
Use the same approved runtime arguments as run_joint_trajectory_rollout after --.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_joint_trajectory_rollout as runner
from conveyor_bench.conveyorvla.execution_consistency import sampled_phase, replay_schedule, deploy_source_chunk
from conveyor_bench.conveyorvla.formal_checkpoint import sha256, source_identity, write_json
from conveyor_bench.conveyorvla.formal_physics import FormalPhysics
from conveyor_bench.conveyorvla.formal_metrics import LIMITS
from conveyor_bench.conveyorvla.joint_trajectory_data import JointTrajectoryNormalizer
from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.joint_trajectory_runtime import DirectJointCommand
from conveyor_bench.conveyorvla.joint_trajectory_system import measured_named_joint_state


def initialize_source_state(simulation, observation):
    """Diagnostic-only initialization: never called after replay/physics arming."""
    import torch
    import numpy as np

    robot = simulation._adapter.robot
    device = robot.device
    tensor = lambda value: torch.tensor([value], dtype=torch.float32, device=device)
    names = list(robot.joint_names)
    source_names = observation["metadata"]["joint_names"]
    if set(names) != set(source_names):
        raise ValueError("source and runtime articulation joint sets differ")
    q = [observation["joint_positions"][source_names.index(n)] for n in names]
    dq = [observation["joint_velocities"][source_names.index(n)] for n in names]
    simulation._adapter.set_base_pose_lock(False)
    simulation._adapter.set_support_joint_lock(False)
    robot.write_root_pose_to_sim(tensor(observation["robot_root_pose"]))
    robot.write_root_velocity_to_sim(tensor(observation["robot_root_velocity"]))
    robot.write_joint_state_to_sim(tensor(q), tensor(dq))
    robot.set_joint_position_target(tensor(q))
    op = observation["object_pose"]
    report = simulation._write_object_physics_state(
        position_xyz=tuple(op[:3]), quaternion_wxyz=tuple(op[3:]),
        velocity_xyz_rpy=tuple(observation["object_velocity"]))
    if report.get("applied") is not True:
        raise ValueError("cannot restore source object initial state")
    simulation._runtime.scene.write_data_to_sim()
    actual = simulation.read()
    checks = {"base_pose_max_abs_error": float(np.max(np.abs(np.array(actual.robot_root_pose)-observation["robot_root_pose"]))),
              "joint_position_max_abs_error": float(np.max(np.abs(np.array(actual.joint_positions)-q))),
              "joint_velocity_max_abs_error": float(np.max(np.abs(np.array(actual.joint_velocities)-dq))),
              "object_xyz_error_m": math.dist(actual.object_pose[:3], op[:3]),
              "object_quaternion_abs_dot": abs(float(np.dot(actual.object_pose[3:],op[3:]))),
              "source_tcp_xyz_error_m": math.dist(actual.tcp_pose[:3],observation["tcp_pose"][:3])}
    if max(checks[k] for k in ("base_pose_max_abs_error", "joint_position_max_abs_error", "joint_velocity_max_abs_error", "object_xyz_error_m")) > 1e-5:
        raise ValueError(f"source initialization mismatch: {checks}")
    return {"checks": checks, "source_observation": observation,
            "actual_observation": runner.waypoint_runner._state_snapshot(actual),
            "initialization_only": True, "object_write": report}


def pipeline_type(options):
    conditions = iter([(options.offset, c) for c in ("absolute", "deployed")] if options.paired_contracts else
                      [(o, options.action_contract) for o in ((0, 1) if options.paired_offsets else (options.offset,))])
    class SampledReplayPipeline(runner.JointTrajectoryRolloutPipeline):
        def run_episode(self):
            offset, action_contract = next(conditions)
            self.episode_dir.mkdir(parents=True, exist_ok=True)
            self._trace_stream = (self.episode_dir / "replay_trace.jsonl").open("x")
            started = time.time()
            summary = {"schema": "sampled-pick-replay-v1", "status": "running",
                       "execution_mode": "validation_isolated_sampled_pick_replay",
                       "policy_score": False, "full_task_success": None, "model_queries": 0,
                       "pure_physics_success": False, "state_trace": ["sampled_exec_pick"],
                       "source_environment_reexecuted": False,
                       "environment": "IsaacSim5.1_migration_from_Sim6",
                       "offset_samples": offset, "physics_profile": options.physics_profile,
                       "action_contract": action_contract,
                       "source_phase": options.source_phase,
                       "normalization_sha256": None if options.normalization is None else sha256(options.normalization),
                       "action_period_s": .2, "control_dt_s": .02,
                       "source_sha256": source_identity(ROOT)["sha256"],
                       "input_hashes": {n: sha256(options.source_episode / n) for n in ("samples.jsonl", "frames.jsonl", "task.json", "summary.json")}}
            errors, distances, physical_ticks = [], [], []
            probe = None
            base_simulation = self.simulation
            summary['negative_control'] = options.negative_control
            try:
                source = json.loads((options.source_episode / "summary.json").read_text())
                if source.get("success") is not True:
                    raise ValueError("source demonstration is not successful")
                samples = [json.loads(x) for x in (options.source_episode / "samples.jsonl").open()]
                frames = [json.loads(x) for x in (options.source_episode / "frames.jsonl").open()]
                phase = sampled_phase(samples, frames, phase=options.source_phase)
                schedule = replay_schedule(phase, offset)
                summary.update(source_episode=options.source_episode.name, source_seed=source["seed"],
                               sampled_commands=len(schedule), replay_seconds=.2*len(schedule),
                               source_success_semantics=source.get("success_semantics"))
                self.config = replace(self.config, video=replace(self.config.video, fps=5.))
                self.episode_seed = source["seed"]
                write_json(self.episode_dir / "resolved_config.json", runner.waypoint_runner._jsonable(asdict(self.config)))
                self._start_video()
                self._prepare_episode()
                if options.record_contacts:
                    if options.contact_backend in ('finger-tensors','object-tensors'):
                        from conveyor_bench.isaac.finger_contact_tensor_probe import IsaacFingerContactTensorProbe
                        probe = IsaacFingerContactTensorProbe(base_simulation, self._record,
                            object_support_probe=options.contact_backend == 'object-tensors')
                    else:
                        from conveyor_bench.isaac.grasp_contact_probe import IsaacGraspContactProbe
                        probe = IsaacGraspContactProbe(base_simulation, self._record)
                    base_simulation.read_grasp_contacts = probe.read
                    if options.initialize_contact_reports:
                        summary['contact_report_initialization'] = base_simulation._hard_reset_stage_reuse_physics(self.episode_spec)
                # No time advances between restoring the state and issuing u(t)/u(t+.2).
                summary["initialization"] = initialize_source_state(self.simulation, phase[0][2])
                self._record("source_state_initialized", summary["initialization"])
                self.physics = FormalPhysics(self.simulation, options.physics_profile, self._record)
                self.simulation = self.physics
                self.physics.arm()
                self.physics.previous_fraction = measured_named_joint_state(self.simulation.read()).gripper_open_fraction
                self.physics.command_fraction = self.physics.previous_fraction
                normalizer = None if action_contract == "absolute" else JointTrajectoryNormalizer.from_path(options.normalization)
                for block_start in range(0, len(schedule), 10):
                    block = schedule[block_start:block_start+10]
                    if action_contract == "deployed":
                        query_q = measured_named_joint_state(self.simulation.read()).joint_position
                        chunk, diagnostic = deploy_source_chunk(block, query_q, normalizer, LIMITS)
                        self._record("deployment_decoder_chunk", {"block_start":block_start, **diagnostic})
                        commands = chunk.commands[:len(block)]
                    else:
                        commands = [DirectJointCommand(index=i, joint_position=tuple(item["absolute_joint_target"]),
                                                       gripper_open_fraction=item["gripper_fraction"])
                                    for i, item in enumerate(block)]
                    for local_index, (item, command) in enumerate(zip(block, commands)):
                        index = block_start+local_index
                        command = replace(command, index=index)
                        if options.negative_control == 'open_gripper':
                            command = replace(command, gripper_open_fraction=1.)
                        action = self.action_adapter.manipulation(command, route=JointTrajectoryRoute.PICK, sequence_id=index)
                        self._record("source_command", item)
                        for tick in range(item["control_ticks"]):
                            state = self._physical_step(action, route=JointTrajectoryRoute.PICK, command_index=index)
                            distance = math.dist(state.object_pose[:3], state.tcp_pose[:3])
                            distances.append(distance)
                            physical_ticks.append({**self.physics.latest,
                                                   "command_gripper_fraction": self.physics.command_fraction})
                            if tick == 9:
                                actual = measured_named_joint_state(state)
                                errors.append(sum(abs(a-b) for a,b in zip(actual.joint_position, command.joint_position))/6)
                summary.update(status="complete", failure_reason=None)
            except Exception as error:
                summary.update(status="failed", failure_reason=f"{type(error).__name__}:{error}",
                               traceback=traceback.format_exc())
            finally:
                if probe is not None:
                    probe.close()
                    del base_simulation.read_grasp_contacts
                # Framework compatibility: success here is execution completion,
                # never a grasp, full-task, or learned-policy success score.
                summary["success"] = summary["status"] == "complete"
                summary["success_semantics"] = "sampled_commands_executed_without_runtime_error"
                tail = physical_ticks[-50:]
                summary["diagnostic_lift_hold"] = {
                    "passed": len(tail) == 50 and all(
                        row["lift_m"] >= .04 and row["object_tcp_distance_m"] <= .08
                        and row["object_speed_mps"] <= .30 and row["command_gripper_fraction"] <= .5
                        for row in tail),
                    "window_s": 1., "observed_ticks": len(tail),
                    "thresholds": {"lift_min_m": .04, "distance_max_m": .08,
                                   "speed_max_mps": .30, "command_fraction_max": .5},
                    "evidence_type": "geometry_proxy_not_contact_sensor",
                    "replaces_formal_pick_verified": False,
                    "ticks": tail,
                }
                summary.update(wall_s=time.time()-started, control_steps=self._control_steps,
                               physics_evidence=None if self.physics is None else self.physics.evidence(),
                               min_tcp_object_distance_m=min(distances) if distances else None,
                               command_end_joint_tracking_mae_rad=sum(errors)/len(errors) if errors else None,
                               target_application=("absolute source targets" if action_contract == "absolute" else
                                                   "live query-relative encoding, normalizer roundtrip, deployment decoder and limits"),
                               note="Equal-duration phase replay; final command is repeated for one-ahead. Initial state copied once; no subsequent resets.")
                try:
                    summary["video"] = self._close_video(summary["status"])
                except Exception as error:
                    summary["video_error"] = str(error)
                write_json(self.episode_dir / "summary.json", runner.waypoint_runner._jsonable(summary))
                self._trace_stream.close()
                self._trace_stream = None
                if self.close_simulation_on_exit:
                    self.simulation.close()
            return summary
    return SampledReplayPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode", type=Path, required=True)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--offset", type=int, choices=(0, 1))
    timing.add_argument("--paired-offsets", action="store_true", help="run offsets 0 then 1; requires --num-episodes 2")
    parser.add_argument("--physics-profile", choices=("source_assisted", "no_grasp_assist"), required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--paired-contracts", action="store_true")
    parser.add_argument("--action-contract", choices=("absolute","deployed"), default="absolute")
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--record-contacts", action="store_true")
    parser.add_argument("--contact-backend", choices=('callbacks','finger-tensors','object-tensors'), default='callbacks')
    parser.add_argument("--initialize-contact-reports", action="store_true",
                        help="rebuild PhysX views after installing report API, before restoring source initial state")
    parser.add_argument("--negative-control", choices=("none", "open_gripper"), default="none")
    parser.add_argument("--source-phase", choices=("exec_pick", "pick_with_planning"), default="exec_pick",
                        help="planning prefix includes recorder-cached targets, not only explicit commands")
    options, runtime_args = parser.parse_known_args()
    if options.initialize_contact_reports and not options.record_contacts:
        raise ValueError('contact report initialization requires --record-contacts')
    if options.initialize_contact_reports and options.contact_backend != 'callbacks':
        raise ValueError('tensor reporters use existing robot APIs and must not be invalidated by reset')
    if options.paired_contracts and options.paired_offsets:
        raise ValueError("pair contracts or offsets in one process, not both")
    if (options.paired_contracts or options.action_contract == "deployed") and options.normalization is None:
        raise ValueError("deployed replay requires --normalization")
    expected_episodes = "2" if options.paired_offsets or options.paired_contracts else "1"
    if "--num-episodes" not in runtime_args or runtime_args[runtime_args.index("--num-episodes") + 1] != expected_episodes:
        raise ValueError(f"replay mode requires --num-episodes {expected_episodes}")
    manifest = json.loads(options.validation_manifest.read_text())
    if manifest.get("split") != "val" or options.source_episode.name not in manifest["episode_ids"]:
        raise ValueError("replay is restricted to the prepared validation episodes")
    for name in ("samples.jsonl", "frames.jsonl", "task.json", "summary.json", "migration_task.json"):
        if sha256(options.source_episode / name) != manifest["files"][f"source/{options.source_episode.name}/{name}"]:
            raise ValueError(f"prepared source evidence changed: {name}")
    if "--expected-identity" in runtime_args:
        raise ValueError("sampled replay must not impersonate a formal model service")
    reference = Path(runtime_args[runtime_args.index("--reference-root") + 1]).resolve()
    runner.waypoint_runner._reference_identity(reference)
    sys.path.insert(0, str(reference))
    import source.simulation as reference_simulation
    original_config = reference_simulation.IsaacLabNavigationRuntimeConfig
    def replay_runtime_config(*args, **kwargs):
        kwargs["camera_render_interval_control_steps"] = 10
        kwargs["enable_verified_grasp_fixed_joint"] = options.physics_profile == "source_assisted"
        return original_config(*args, **kwargs)
    reference_simulation.IsaacLabNavigationRuntimeConfig = replay_runtime_config
    runner.JointTrajectoryRolloutPipeline = pipeline_type(options)
    result = runner.main(runtime_args)
    output_dir = Path(runtime_args[runtime_args.index('--output-dir')+1])
    summaries = list(output_dir.glob('episode_*/summary.json'))
    complete = len(summaries) == int(expected_episodes) and all(
        json.loads(path.read_text()).get('status') == 'complete' for path in summaries)
    return result or (0 if complete else 1)


if __name__ == "__main__":
    raise SystemExit(main())
