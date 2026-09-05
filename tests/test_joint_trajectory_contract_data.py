import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    DATASET_PROFILE,
    DATASET_SCHEMA_VERSION,
    MANIPULATION_STRIDE_S,
    NAVIGATION_STRIDE_S,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    canonical_solution,
    direct_joint_targets,
    joint_trajectory_prompt,
    terminal_hold,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import (
    ConveyorVLAJointTrajectoryDataset,
    JointTrajectoryNormalizer,
    NoLegalFutureTargetError,
    derive_fresh_joint_trajectory_record,
    mani_action_from_applied_commands,
    validate_joint_trajectory_record,
)
from conveyor_bench.conveyorvla import joint_trajectory_data


def _record(route: JointTrajectoryRoute, sample: int = 0) -> dict:
    domain = (
        JointTrajectoryDomain.NAVIGATION
        if route in {JointTrajectoryRoute.NAV_TO_SOURCE, JointTrajectoryRoute.NAV_TO_TARGET}
        else JointTrajectoryDomain.MANIPULATION
    )
    progress_provenance = {
        JointTrajectoryRoute.NAV_TO_SOURCE: "source_distance_and_settle",
        JointTrajectoryRoute.PICK: "pick_reach_alignment_grasp_lift",
        JointTrajectoryRoute.NAV_TO_TARGET: "target_distance_carry_and_settle",
        JointTrajectoryRoute.PLACE: "place_alignment_release_and_separation",
    }[route]
    return {
        "sample_id": f"sample-{sample}-{route.value}",
        "episode_id": f"episode-{sample}",
        "split": "train",
        "query_timestamp_s": 1.0,
        "history_timestamps_s": [0.8, 1.0],
        "global_instruction": "Move the Coke can from box one to box two.",
        "head_images": ["head0.png", "head1.png"],
        "wrist_images": ["wrist0.png", "wrist1.png"],
        "route": route.value,
        "route_token": {
            JointTrajectoryRoute.NAV_TO_SOURCE: "<|route_nav_to_source|>",
            JointTrajectoryRoute.PICK: "<|route_pick|>",
            JointTrajectoryRoute.NAV_TO_TARGET: "<|route_nav_to_target|>",
            JointTrajectoryRoute.PLACE: "<|route_place|>",
        }[route],
        "assistant_solution": canonical_solution(route),
        "action_domain": domain.value,
        "nav_trajectory_body": (
            [[0.01 * (index + 1), 0.0, 0.0] for index in range(ACTION_HORIZON)]
            if domain is JointTrajectoryDomain.NAVIGATION
            else None
        ),
        "mani_delta_q_gripper": (
            [[0.001 * (index + 1)] * 6 + [0.8] for index in range(ACTION_HORIZON)]
            if domain is JointTrajectoryDomain.MANIPULATION
            else None
        ),
        "mani_state": (
            [0.1] * 6 + [0.01] * 6 + [0.8]
            if domain is JointTrajectoryDomain.MANIPULATION
            else None
        ),
        "action_provenance": (
            "controller_applied_after_saturation"
            if domain is JointTrajectoryDomain.MANIPULATION
            else "teacher_base_reference"
        ),
        "action_valid_mask": [True] * ACTION_HORIZON,
        "terminal_hold_start_index": ACTION_HORIZON,
        "terminal_hold_reason": None,
        "transition_window": False,
        "boundary_transition": None,
        "transition_id": None,
        "boundary_signed_time_s": None,
        "physical_progress": 0.5,
        "physical_progress_valid": True,
        "physical_progress_provenance": progress_provenance,
        "progress_bucket": "middle",
        "gripper_transition": False,
    }


def test_contract_has_four_routes_ten_full_targets_and_independent_clocks():
    assert [route.value for route in JointTrajectoryRoute] == [
        "NAV_TO_SOURCE",
        "PICK",
        "NAV_TO_TARGET",
        "PLACE",
    ]
    assert ACTION_HORIZON == 10
    assert NAVIGATION_STRIDE_S == 0.20
    assert MANIPULATION_STRIDE_S == 0.20
    assert "<|pred_done|>" not in joint_trajectory_prompt("move the Coke")
    for route in JointTrajectoryRoute:
        assert canonical_solution(route).startswith("<|pred_action|><|route_")


def test_terminal_hold_supervises_all_ten_and_direct_joint_uses_one_query_anchor():
    held = terminal_hold([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], "NAVIGATION")
    assert len(held) == ACTION_HORIZON
    assert held[:2] == ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0))
    assert held[2:] == ((0.2, 0.0, 0.0),) * 8
    action = [[0.1 * (index + 1)] * 6 + [0.7] for index in range(ACTION_HORIZON)]
    targets = direct_joint_targets([1.0] * 6, action)
    assert targets[0][:6] == pytest.approx([1.1] * 6)
    assert targets[1][:6] == pytest.approx([1.2] * 6)
    assert targets[1][0] != pytest.approx(targets[0][0] + 0.2)


def test_record_rejects_suffix_masks_state_leakage_and_elapsed_progress():
    record = _record(JointTrajectoryRoute.PICK)
    validate_joint_trajectory_record(record, expected_split="train")
    masked = dict(record, action_valid_mask=[True] * 5 + [False] * 5)
    with pytest.raises(ValueError, match="all true"):
        validate_joint_trajectory_record(masked)
    leaked = dict(record, elapsed_phase_fraction=0.5)
    with pytest.raises(ValueError, match="forbidden"):
        validate_joint_trajectory_record(leaked)
    wrong = dict(record, physical_progress_provenance="source_distance_and_settle")
    with pytest.raises(ValueError, match="does not match route"):
        validate_joint_trajectory_record(wrong)


def test_mani_label_requires_applied_controller_commands():
    samples = []
    for index in range(ACTION_HORIZON):
        samples.append(
            {
                "tick_id": index * 10 + 10,
                "sim_step": index * 10 + 10,
                "model_tick": 0,
                "timestamp_s": 1.20 + index * 0.20,
                "q_measured": [0.0] * 6,
                "dq_measured": [0.0] * 6,
                "gripper_measured": 1.0,
                "q_command_requested": [0.01 * (index + 1)] * 6,
                "q_command_applied": [0.009 * (index + 1)] * 6,
                "gripper_command_requested": 0.9,
                "gripper_command_applied": 0.85,
                "base_command_requested": [0.0, 0.0, 0.0],
                "base_command_applied": [0.0, 0.0, 0.0],
                "base_pose_world": [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                "base_twist_world": [0.0] * 6,
                "route": "PICK",
                "q_command_source": "controller_applied_after_saturation",
            }
        )
    action = mani_action_from_applied_commands([0.1] * 6 + [0.0] * 6 + [1.0], samples)
    assert action[0][:6] == pytest.approx([-0.091] * 6)
    missing = [dict(sample) for sample in samples]
    missing[0].pop("q_command_applied")
    with pytest.raises(ValueError, match="q_command_applied"):
        mani_action_from_applied_commands([0.0] * 12 + [1.0], missing)
    off_clock = [dict(sample) for sample in samples]
    off_clock[1]["timestamp_s"] += 0.01
    with pytest.raises(ValueError, match="5 Hz"):
        mani_action_from_applied_commands([0.0] * 12 + [1.0], off_clock)
    moving_base = [dict(sample) for sample in samples]
    moving_base[0]["base_command_applied"] = [0.001, 0.0, 0.0]
    with pytest.raises(ValueError, match="base command"):
        mani_action_from_applied_commands([0.0] * 12 + [1.0], moving_base)
    requested_base = [dict(sample) for sample in samples]
    requested_base[0]["base_command_requested"] = [0.001, 0.0, 0.0]
    with pytest.raises(ValueError, match="base command"):
        mani_action_from_applied_commands([0.0] * 12 + [1.0], requested_base)


def test_fresh_derivation_uses_applied_targets_and_terminal_holds_at_route_boundary():
    controls = {}
    for tick in range(0, 61):
        route = "PICK" if tick <= 49 else "NAV_TO_TARGET"
        controls[tick] = {
            "tick_id": tick,
            "sim_step": tick,
            "model_tick": tick // 10,
            "timestamp_s": 1.0 + tick * 0.02,
            "q_measured": [0.1] * 6,
            "dq_measured": [0.0] * 6,
            "gripper_measured": 1.0,
            "q_command_requested": [0.2 + 0.001 * tick] * 6,
            "q_command_applied": [0.15 + 0.001 * tick] * 6,
            "gripper_command_requested": 1.0 if tick < 4 else 0.0,
            "gripper_command_applied": 1.0 if tick < 4 else 0.2,
            "base_command_requested": [0.0, 0.0, 0.0],
            "base_command_applied": [0.0, 0.0, 0.0],
            "base_pose_world": [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            "base_twist_world": [0.0] * 6,
            "route": route,
            "q_command_source": "controller_applied_after_saturation",
        }
    query = {
        "control_tick_id": 0,
        "sample_id": "sample",
        "episode_id": "episode",
        "split": "train",
        "route": "PICK",
        "global_instruction": "move the Coke",
        "head_images": ["head0.png", "head1.png"],
        "wrist_images": ["wrist0.png", "wrist1.png"],
        "physical_progress": 0.5,
        "physical_progress_valid": True,
        "physical_progress_provenance": "pick_reach_alignment_grasp_lift",
        "transition_window": True,
        "boundary_transition": "PICK->NAV_TO_TARGET",
        "transition_id": "episode:PICK->NAV_TO_TARGET",
        "boundary_signed_time_s": -0.1,
    }
    record = derive_fresh_joint_trajectory_record(query, controls)
    assert record["terminal_hold_start_index"] == 4
    assert record["terminal_hold_reason"] == "boundary"
    assert record["action_valid_mask"] == [True] * 10
    assert record["mani_delta_q_gripper"][0][0] == pytest.approx(0.06)
    assert record["mani_delta_q_gripper"][4:] == [record["mani_delta_q_gripper"][3]] * 6

    no_future = dict(query, control_tick_id=40, sample_id="boundary-last-query")
    with pytest.raises(NoLegalFutureTargetError, match="no legal future target"):
        derive_fresh_joint_trajectory_record(no_future, controls)


def test_sampled_5hz_derivation_uses_exact_observation_and_next_saved_control():
    pipeline_states = (
        ["reset_episode"]
        + ["plan_nav_to_pick"] * 12
        + ["plan_pick"] * 12
        + ["plan_nav_to_place"] * 12
        + ["plan_place"] * 12
    )
    names = [f"arm_joint{index}" for index in range(1, 9)]
    samples = []
    frames = []
    for index, pipeline_state in enumerate(pipeline_states):
        timestamp = index * 0.2
        q = [0.01 * index + axis * 0.001 for axis in range(6)]
        gripper_m = 0.04 if pipeline_state != "plan_place" else 0.02
        action = [0.0, 0.0, 0.0] + [value + 0.005 for value in q] + [gripper_m] * 2
        base_pose = [0.01 * index, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
        camera_frames = {
            camera: {
                "frame_index": index,
                "timestamp": timestamp,
                "state_step": index * 10,
                "camera_capture_step": index * 10,
                "raw_image_path": f"images/{camera}/{index:05d}.jpg",
            }
            for camera in ("front", "wrist")
        }
        samples.append(
            {
                "frame_index": index,
                "timestamp": timestamp,
                "simulation_step": index * 10,
                "pipeline_state": pipeline_state,
                "base_pose": base_pose,
                "gripper_position": gripper_m,
                "camera_state_synchronized": True,
                "camera_frames": camera_frames,
                "action": action,
            }
        )
        frames.append(
            {
                "dataset_frame_index": index,
                "pipeline_state": pipeline_state,
                "action": {
                    "base_velocity": action[:3],
                    "arm_joint_positions": action[3:9],
                    "metadata": {"gripper_joint_positions": action[9:11]},
                },
                "observation": {
                    "step_index": index * 10,
                    "timestamp": timestamp,
                    "robot_root_pose": base_pose,
                    "joint_positions": q + [gripper_m, gripper_m],
                    "joint_velocities": [0.1] * 6 + [0.0, 0.0],
                    "metadata": {"joint_names": names},
                },
                # A deliberately different post-step state proves it is not used.
                "post_step_observation": {"joint_positions": [99.0] * 8},
            }
        )
    records, dropped = joint_trajectory_data._derive_sampled_5hz_episode_records(
        {"instruction": "move the Coke", "samples": samples, "frames": frames},
        episode_id="sampled-episode",
        split="train",
    )
    joint_trajectory_data._validate_complete_episode_records(records)
    first_pick = next(record for record in records if record["route"] == "PICK")
    query_index = int(first_pick["sample_id"].rsplit("-", maxsplit=1)[1])
    expected_first_target = [
        samples[query_index + 1]["action"][axis + 3]
        - frames[query_index]["observation"]["joint_positions"][axis]
        for axis in range(6)
    ]
    assert first_pick["mani_delta_q_gripper"][0][:6] == pytest.approx(
        expected_first_target
    )
    assert first_pick["mani_state"][:6] == pytest.approx(
        frames[query_index]["observation"]["joint_positions"][:6]
    )
    assert first_pick["action_provenance"] == "sampled_control_target_5hz"
    assert not first_pick["physical_progress_valid"]
    assert dropped == 4


def test_train_only_normalizer_roundtrip_and_lazy_batch_state_boundary(tmp_path: Path):
    records = [
        _record(JointTrajectoryRoute.NAV_TO_SOURCE, 0),
        _record(JointTrajectoryRoute.NAV_TO_TARGET, 1),
        _record(JointTrajectoryRoute.PICK, 2),
        _record(JointTrajectoryRoute.PLACE, 3),
    ]
    normalizer = JointTrajectoryNormalizer.fit(records)
    nav = records[0]["nav_trajectory_body"]
    assert np.allclose(
        normalizer.denormalize_action(
            records[0]["route"], normalizer.normalize_action(records[0]["route"], nav)
        ),
        nav,
    )
    mani = records[2]["mani_delta_q_gripper"]
    denormalized = normalizer.denormalize_action(
        records[2]["route"], normalizer.normalize_action(records[2]["route"], mani)
    )
    assert np.allclose(denormalized, mani)

    image = pytest.importorskip("PIL.Image")
    for name in ("head0.png", "head1.png", "wrist0.png", "wrist1.png"):
        image.new("RGB", (4, 4), (10, 20, 30)).save(tmp_path / name)
    record_path = tmp_path / "train.jsonl"
    record_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    normalization_path = tmp_path / "normalization.json"
    normalization_path.write_text(
        json.dumps(normalizer.payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "profile": DATASET_PROFILE,
                "records": {
                    "train": {
                        "relative_path": "train.jsonl",
                        "sha256": sha(record_path),
                    }
                },
                "normalization_relative_path": "normalization.json",
                "normalization_sha256": sha(normalization_path),
            }
        ),
        encoding="utf-8",
    )
    dataset = ConveyorVLAJointTrajectoryDataset(tmp_path, split="train")
    nav_example = dataset[0]
    mani_example = dataset[2]
    assert nav_example["mani_state"] is None
    assert len(mani_example["mani_state"]) == 13
    assert nav_example["action_valid_mask"] == (True,) * ACTION_HORIZON
    assert "state" not in nav_example and "phase" not in nav_example
