"""Adapt PCT Liangzhu raw episodes to the ConveyorVLA LeRobot contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from bisect import bisect_left, bisect_right
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.lerobot_v3 import (
    ACTION_DIM,
    ACTION_HORIZON,
    CONTROL_HZ,
    DEFAULT_LEROBOT_V3_CONFIG_PATH,
    MODEL_HZ,
    STATE_DIM,
    VIDEO_FEATURE_KEYS,
    lerobot_features,
    lerobot_frame_from_record,
    load_lerobot_v3_config,
)
from conveyor_bench.conveyorvla.online import build_live_state28
from conveyor_bench.conveyorvla.subtasks import (
    FULL_INSTRUCTION,
    PCT_PHASES,
    Phase,
    action_domain,
    phase_from_pct,
    phase_instruction,
)
from conveyor_bench.conveyorvla.temporal import (
    ACTION_DIMENSION_MASK,
    CAMERA_IDS,
    GRIPPER_ACTION_SOURCE,
    HISTORY_OFFSETS_MODEL_TICKS,
    POLICY_TASK_SCOPE,
    TEMPORAL_PROFILE,
    TEMPORAL_SCHEMA_VERSION,
    _conjugate,
    _multiply,
    _rotate,
    _unit_quaternion,
    relative_tcp_target,
)


PCT_VISUAL_HISTORY_SPAN_S = 0.2
PCT_VISUAL_HISTORY_MODEL_TICKS = (-5, 0)
PCT_CONTROL_STEPS_PER_MODEL_TICK = CONTROL_HZ // MODEL_HZ
PCT_QUERY_CONTROL_STEPS = CONTROL_HZ // 5
PCT_TRAINING_STATES = frozenset(PCT_PHASES)
PCT_REQUIRED_JOINTS = tuple(f"arm_joint{index}" for index in range(1, 9))


def discover_pct_episodes(source_roots: Iterable[str | Path]) -> tuple[Path, ...]:
    """Return all raw PCT episode directories below the supplied roots."""

    episodes: list[Path] = []
    for value in source_roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise M0MobileError(f"PCT source root is not a directory: {root}")
        episodes.extend(
            child
            for child in sorted(root.iterdir())
            if child.is_dir()
            and child.name.startswith("episode_")
            and (child / "frames.jsonl").is_file()
            and (child / "samples.jsonl").is_file()
        )
    if not episodes:
        raise M0MobileError("no PCT episodes were found")
    return tuple(episodes)


def audit_pct_episode(episode_root: str | Path) -> dict[str, Any]:
    """Fail closed on task, success, camera, and clock mismatches."""

    root = Path(episode_root).expanduser().resolve()
    try:
        task = _read_json(root / "task.json")
        summary = _read_json(root / "summary.json")
        manifest = _read_json(root / "lerobot_manifest.json")
    except M0MobileError as error:
        return {
            "episode_root": str(root),
            "eligible": False,
            "problems": [str(error)],
            "instruction": None,
            "source_success": None,
            "source_vla_training_action_available": None,
            "source_vla_training_ineligibility_reason": None,
        }
    problems: list[str] = []
    if summary.get("success") is not True or summary.get("failure_reason"):
        problems.append("episode is not a successful physical execution")
    if summary.get("execution_provenance_verified") is not True:
        problems.append("execution provenance is not verified")
    if summary.get("training_quality_gate_passed") is not True:
        problems.append("source training quality gate did not pass")
    if summary.get("training_visual_source_verified") is not True:
        problems.append("source training visuals are not verified")
    if manifest.get("raw_episode_ready") is not True:
        problems.append("raw episode is not ready")
    if manifest.get("camera_keys") != ["front", "wrist"]:
        problems.append("camera keys must be front then wrist")
    if manifest.get("missing_camera_keys") != []:
        problems.append("episode has missing camera keys")
    synchronization = manifest.get("camera_state_synchronization")
    if not isinstance(synchronization, Mapping) or synchronization.get("verified") is not True:
        problems.append("camera/state synchronization is not verified")
    frequency = manifest.get("frequency_report")
    if not isinstance(frequency, Mapping):
        problems.append("frequency report is missing")
    elif frequency.get("control_hz") != CONTROL_HZ or frequency.get("dataset_fps") != 5.0:
        problems.append("episode must use 50 Hz control and 5 Hz cameras")
    training_action = task.get("training_action")
    if not isinstance(training_action, Mapping) or training_action.get("enabled") is not True:
        problems.append("task does not request a VLA training action")
    elif training_action.get("source_gripper_joint_range_m") != [0.0, 0.04]:
        problems.append("unsupported gripper source range")
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        problems.append("task instruction is missing")
    return {
        "episode_root": str(root),
        "eligible": not problems,
        "problems": problems,
        "instruction": instruction,
        "source_success": summary.get("success"),
        "source_vla_training_action_available": manifest.get(
            "vla_training_action_available"
        ),
        "source_vla_training_ineligibility_reason": manifest.get(
            "vla_training_ineligibility_reason"
        ),
    }


def iter_pct_temporal_records(episode_root: str | Path) -> Iterator[dict[str, Any]]:
    """Yield 5 Hz observations with 25 Hz, 0.8 second future actions."""

    root = Path(episode_root).expanduser().resolve()
    audit = audit_pct_episode(root)
    if not audit["eligible"]:
        raise M0MobileError(
            f"PCT episode is not eligible: {root}: " + "; ".join(audit["problems"])
        )
    task = _read_json(root / "task.json")
    instruction = str(task["instruction"]).strip()
    controls = _control_frames(root / "frames.jsonl")
    control_steps = tuple(controls)
    action_starts = tuple(step - 1 for step in control_steps)
    action_records = tuple(controls[step] for step in control_steps)
    samples = _sample_frames(root / "samples.jsonl")
    collection = next(
        (
            parent.name
            for parent in root.parents
            if parent.name.startswith("liangzhu_") and "_n" in parent.name
        ),
        "liangzhu_pct",
    )
    episode_id = f"{collection}:{root.name}"
    for sample_index in range(1, len(samples)):
        history_sample = samples[sample_index - 1]
        source_sample = samples[sample_index]
        source_step = _integer(source_sample.get("simulation_step"), "simulation_step")
        history_step = _integer(history_sample.get("simulation_step"), "simulation_step")
        if source_step - history_step != PCT_QUERY_CONTROL_STEPS:
            continue
        if source_sample.get("pipeline_state") not in PCT_TRAINING_STATES:
            continue
        if not control_steps[0] <= source_step <= control_steps[-1]:
            continue
        target_steps = tuple(
            source_step + offset * PCT_CONTROL_STEPS_PER_MODEL_TICK
            for offset in range(1, ACTION_HORIZON + 1)
        )
        if target_steps[-1] > control_steps[-1]:
            continue

        raw_phase = str(source_sample["pipeline_state"])
        phase = phase_from_pct(raw_phase)
        phase_pure = _phase_pure_window(
            samples,
            sample_index,
            target_steps[-1],
            controls,
            control_steps,
            phase,
        )
        transition = _transition_metadata(
            samples,
            sample_index,
            source_step,
            target_steps,
            controls,
            control_steps,
            phase,
        )

        source = _observation_at(controls, control_steps, source_step)
        source_root_xyz, source_root_wxyz, source_tcp_xyz, source_tcp_wxyz = (
            _root_and_tcp_base(source)
        )
        actions = []
        for target_step in target_steps:
            target = _observation_at(controls, control_steps, target_step)
            target_root_xyz, target_root_wxyz, target_tcp_xyz, target_tcp_wxyz = (
                _root_and_tcp_base(target)
            )
            base_action = tuple(
                sum(
                    _base_action_at(
                        action_starts,
                        action_records,
                        step,
                    )[axis]
                    for step in (target_step - 1, target_step)
                )
                / PCT_CONTROL_STEPS_PER_MODEL_TICK
                for axis in range(3)
            )
            actions.append(
                base_action
                + relative_tcp_target(
                    source_root_xyz,
                    source_root_wxyz,
                    source_tcp_xyz,
                    source_tcp_wxyz,
                    target_root_xyz,
                    target_root_wxyz,
                    target_tcp_xyz,
                    target_tcp_wxyz,
                )
                + (_gripper_fraction(target),)
            )

        camera_clips = []
        for camera_id, pct_key in zip(CAMERA_IDS, ("front", "wrist"), strict=True):
            camera_clips.append(
                {
                    "camera_id": camera_id,
                    # Compatibility key consumed by the existing temporal loader.
                    # The exact 0.20 s source interval is recorded separately below.
                    "history_offsets_model_ticks": HISTORY_OFFSETS_MODEL_TICKS,
                    "frames": [
                        {
                            "camera_id": camera_id,
                            "relative_path": _camera_path(root, frame, pct_key),
                        }
                        for frame in (history_sample, source_sample)
                    ],
                }
            )
        model_tick = source_step // PCT_CONTROL_STEPS_PER_MODEL_TICK
        yield {
            "schema_version": TEMPORAL_SCHEMA_VERSION,
            "profile": TEMPORAL_PROFILE,
            "source_episode_id": episode_id,
            "source_task_outcome": "success",
            "source_assisted": False,
            "sample_id": f"{episode_id}:control-step-{source_step}",
            "instruction": instruction,
            "policy_task_scope": POLICY_TASK_SCOPE,
            "phase": raw_phase,
            "source_pipeline_state": raw_phase,
            "phase_id": int(phase),
            "phase_name": phase.name,
            "action_domain_id": int(action_domain(phase)),
            "action_domain_name": action_domain(phase).name,
            "full_instruction": FULL_INSTRUCTION,
            "phase_instruction": phase_instruction(phase),
            "phase_pure_action_horizon": phase_pure,
            **transition,
            "observation_model_tick": model_tick,
            "observation_control_tick": source_step,
            "camera_clips": camera_clips,
            "history_offsets_model_ticks": HISTORY_OFFSETS_MODEL_TICKS,
            "source_visual_history_model_ticks": PCT_VISUAL_HISTORY_MODEL_TICKS,
            "source_visual_history_span_s": PCT_VISUAL_HISTORY_SPAN_S,
            "source_state_resampling": "linear_pose_nlerp_quaternion",
            "source_command_resampling": "zero_order_hold",
            "state28": _state28(source),
            "state_layout": _state_layout(),
            "model_action10_chunk": actions,
            "action_rate_hz": MODEL_HZ,
            "control_rate_hz": CONTROL_HZ,
            "action_horizon": ACTION_HORIZON,
            "future_offsets_model_ticks": list(range(1, ACTION_HORIZON + 1)),
            "future_model_ticks": [
                model_tick + offset for offset in range(1, ACTION_HORIZON + 1)
            ],
            "gripper_action_source": GRIPPER_ACTION_SOURCE,
            "action_dimension_mask": ACTION_DIMENSION_MASK,
            "object_state_is_model_input": False,
        }


def _transition_metadata(
    samples: Sequence[Mapping[str, Any]],
    sample_index: int,
    source_step: int,
    target_steps: Sequence[int],
    controls: Mapping[int, Mapping[str, Any]],
    control_steps: Sequence[int],
    phase: Phase,
) -> dict[str, Any]:
    """Describe the nearest phase boundary and the current-expert action prefix."""

    current = phase_from_pct(samples[sample_index].get("pipeline_state"))
    if current is not phase:
        raise M0MobileError("PCT phase metadata changed while building a record")
    previous_phase = None
    phase_start_step = source_step
    for sample in reversed(samples[:sample_index]):
        candidate = PCT_PHASES.get(str(sample.get("pipeline_state", "")))
        if candidate is current:
            phase_start_step = _integer(sample.get("simulation_step"), "simulation_step")
        elif candidate is not None:
            previous_phase = candidate
            break

    next_phase = None
    boundary_step = None
    for sample in samples[sample_index + 1 :]:
        candidate = PCT_PHASES.get(str(sample.get("pipeline_state", "")))
        if candidate is not None and candidate is not current:
            next_phase = candidate
            boundary_step = _integer(sample.get("simulation_step"), "simulation_step")
            break
    if boundary_step is None:
        boundary_step = _integer(samples[-1].get("simulation_step"), "simulation_step")

    seconds_since_previous = (source_step - phase_start_step) / CONTROL_HZ
    seconds_to_next = max(0.0, (boundary_step - source_step) / CONTROL_HZ)
    upcoming = next_phase is not None and seconds_to_next <= seconds_since_previous
    seconds_to_boundary = -seconds_to_next if upcoming else seconds_since_previous
    previous_transition = (
        None
        if previous_phase is None
        else f"{previous_phase.name}->{current.name}"
    )
    next_name = next_phase.name if next_phase is not None else Phase.DONE.name
    next_transition = f"{current.name}->{next_name}"
    boundary_transition = next_transition if upcoming else previous_transition
    boundary_window = bool(
        (next_phase is not None and seconds_to_next <= 1.0)
        or (previous_phase is not None and seconds_since_previous <= 1.0)
    )

    source_domain = action_domain(current)
    valid_mask: list[bool] = []
    crossed_domain = False
    interval_start = source_step
    for target_step in target_steps:
        lower = bisect_right(control_steps, interval_start)
        upper = bisect_right(control_steps, target_step)
        interval = control_steps[lower:upper]
        if not interval:
            index = bisect_right(control_steps, target_step) - 1
            interval = () if index < 0 else (control_steps[index],)
        for step in interval:
            candidate = PCT_PHASES.get(str(controls[step].get("pipeline_state", "")))
            if candidate is None or action_domain(candidate) is not source_domain:
                crossed_domain = True
                break
        valid_mask.append(not crossed_domain)
        interval_start = target_step

    reasons = {
        "NAV_TO_SOURCE->PICK": "base_stopped_source_in_reach",
        "PICK->NAV_TO_TARGET": "grasp_lifted_carry_ready",
        "NAV_TO_TARGET->PLACE": "base_stopped_target_in_reach",
        "PLACE->DONE": "released_in_target",
    }
    return {
        "previous_subtask_label": (
            None if previous_phase is None else previous_phase.name
        ),
        "next_subtask_label": next_name,
        "seconds_to_boundary": seconds_to_boundary,
        "seconds_to_next_boundary_s": seconds_to_next,
        "seconds_since_previous_boundary_s": seconds_since_previous,
        "is_boundary_window": boundary_window,
        "boundary_transition": boundary_transition,
        "transition_reason": (
            reasons.get(boundary_transition, "phase_interior")
            if boundary_window
            else "phase_interior"
        ),
        "action_valid_mask": valid_mask,
    }


def _phase_pure_window(
    samples: Sequence[Mapping[str, Any]],
    sample_index: int,
    target_step: int,
    controls: Mapping[int, Mapping[str, Any]],
    control_steps: Sequence[int],
    phase: Phase,
) -> bool:
    """Require history and future controls to retain one semantic subtask."""

    if (
        sample_index <= 0
        or PCT_PHASES.get(str(samples[sample_index - 1].get("pipeline_state", "")))
        is not phase
    ):
        return False
    future_samples = []
    for sample in samples[sample_index:]:
        step = _integer(sample.get("simulation_step"), "simulation_step")
        if step > target_step:
            break
        future_samples.append(sample)
    if (
        not future_samples
        or _integer(future_samples[-1].get("simulation_step"), "simulation_step")
        != target_step
        or any(
            PCT_PHASES.get(str(sample.get("pipeline_state", ""))) is not phase
            for sample in future_samples
        )
    ):
        return False
    lower = bisect_right(control_steps, _integer(samples[sample_index].get("simulation_step"), "simulation_step"))
    upper = bisect_right(control_steps, target_step)
    future_control_steps = control_steps[lower:upper]
    return bool(future_control_steps) and all(
        PCT_PHASES.get(str(controls[step].get("pipeline_state", ""))) is phase
        for step in future_control_steps
    )


def materialize_pct_lerobot_v3(
    episode_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_LEROBOT_V3_CONFIG_PATH,
    repo_id: str = "local/conveyorvla-liangzhu-pct",
) -> dict[str, Any]:
    """Create one immutable LeRobot v3 dataset from eligible PCT episodes."""

    roots = tuple(Path(value).expanduser().resolve() for value in episode_roots)
    if not roots or len(roots) != len(set(roots)):
        raise M0MobileError("PCT episode roots must be non-empty and unique")
    config_source = Path(config_path).expanduser().resolve()
    config = load_lerobot_v3_config(config_source)
    expected_version = str(config["format"]["lerobot_package_version"])
    try:
        installed_version = version("lerobot")
    except PackageNotFoundError as error:
        raise M0MobileError(f"lerobot=={expected_version} is required") from error
    if installed_version != expected_version:
        raise M0MobileError(
            f"PCT conversion requires lerobot=={expected_version}, got {installed_version}"
        )
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise M0MobileError("cannot import LeRobotDataset") from error

    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise M0MobileError(f"LeRobot output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    encoding = config["encoding"]
    reports = []
    total_frames = 0
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=staging,
            robot_type=str(config["format"]["robot_type"]),
            fps=5,
            features=lerobot_features(config),
            use_videos=True,
            vcodec=str(encoding["vcodec"]),
            image_writer_threads=int(encoding["image_writer_threads"]),
            metadata_buffer_size=int(encoding["metadata_buffer_size"]),
            batch_encoding_size=int(encoding["batch_encoding_size"]),
            encoder_threads=int(encoding["encoder_threads"]),
            streaming_encoding=bool(encoding["streaming_encoding"]),
        )
        for root in roots:
            count = 0
            source_id = None
            for record in iter_pct_temporal_records(root):
                source_id = record["source_episode_id"]
                dataset.add_frame(lerobot_frame_from_record(record, root, config))
                count += 1
            if count == 0:
                raise M0MobileError(f"PCT episode produced no training frames: {root}")
            dataset.save_episode()
            total_frames += count
            reports.append(
                {
                    "source_episode_id": source_id,
                    "source_episode_root": str(root),
                    "query_frames": count,
                }
            )
        dataset.finalize()
        loaded = LeRobotDataset(repo_id=repo_id, root=staging, video_backend="pyav")
        if int(loaded.meta.total_episodes) != len(reports) or len(loaded) != total_frames:
            raise M0MobileError("official LeRobot reload count mismatch")
        first = loaded[0]
        if tuple(first["observation.state"].shape) != (STATE_DIM,):
            raise M0MobileError("decoded PCT state shape mismatch")
        if tuple(first["action"].shape) != (ACTION_HORIZON * ACTION_DIM,):
            raise M0MobileError("decoded PCT action shape mismatch")
        for key in VIDEO_FEATURE_KEYS:
            if tuple(first[key].shape) != (3, 224, 224):
                raise M0MobileError(f"decoded PCT video shape mismatch: {key}")
        manifest = {
            "schema_version": (
                "conveyor-vla-al0-liangzhu-0815-lerobot-v3-dense-transition-manifest-5"
            ),
            "dataset_version": "v3.0",
            "repo_id": repo_id,
            "robot_type": config["format"]["robot_type"],
            "lerobot_package_version": installed_version,
            "config_sha256": _sha256(config_source),
            "source_format": "pct_full_physics_raw",
            "source_adapter": "conveyorvla_pct_dense_transition_v5",
            "source_phase_aliases": {
                raw: phase.name for raw, phase in sorted(PCT_PHASES.items())
            },
            "transition_observations_included": True,
            "source_collections": sorted(
                {
                    str(report["source_episode_id"]).split(":", 1)[0]
                    for report in reports
                }
            ),
            "source_license": "Apache-2.0",
            "source_state_resampling": "linear_pose_nlerp_quaternion",
            "source_command_resampling": "zero_order_hold",
            "query_fps": 5,
            "action_rate_hz": MODEL_HZ,
            "control_hz": CONTROL_HZ,
            "history_offsets_model_ticks": list(PCT_VISUAL_HISTORY_MODEL_TICKS),
            "history_span_s": PCT_VISUAL_HISTORY_SPAN_S,
            "compatibility_video_feature_names": list(VIDEO_FEATURE_KEYS),
            "video_feature_keys": list(VIDEO_FEATURE_KEYS),
            "state_shape": [STATE_DIM],
            "action_storage_shape": [ACTION_HORIZON * ACTION_DIM],
            "action_logical_shape": [ACTION_HORIZON, ACTION_DIM],
            "episode_count": len(reports),
            "frame_count": total_frames,
            "episodes": reports,
        }
        manifest_path = staging / "meta" / "conveyorvla_al0_conversion.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return {**manifest, "dataset_root": str(output)}
    except Exception:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                # Video encoders can still be releasing files after a failed
                # save.  Cleanup must never replace the conversion error that
                # explains why the immutable output was not published.
                pass
        raise


def _control_frames(path: Path) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for record in _read_jsonl(path):
        action = _mapping(record.get("action"), "action")
        metadata = action.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("skip_physics_step") is True:
            continue
        post = _mapping(record.get("post_step_observation"), "post_step_observation")
        step = _integer(post.get("step_index"), "post_step_observation.step_index")
        if step in result:
            raise M0MobileError(f"duplicate physical control step {step} in {path}")
        _finite_vector(post.get("robot_root_pose"), 7, "robot_root_pose")
        _finite_vector(post.get("robot_root_velocity"), 6, "robot_root_velocity")
        _finite_vector(post.get("tcp_pose"), 7, "tcp_pose")
        result[step] = record
    if not result:
        raise M0MobileError(f"no physical control frames in {path}")
    return result


def _sample_frames(path: Path) -> tuple[Mapping[str, Any], ...]:
    samples = tuple(_read_jsonl(path))
    if not samples:
        raise M0MobileError(f"no sampled camera frames in {path}")
    previous = None
    for sample in samples:
        step = _integer(sample.get("simulation_step"), "simulation_step")
        if previous is not None and step <= previous:
            raise M0MobileError("PCT sampled simulation steps must increase")
        previous = step
    return samples


def _observation_at(
    controls: Mapping[int, Mapping[str, Any]],
    control_steps: Sequence[int],
    step: int,
) -> Mapping[str, Any]:
    """Interpolate the sparse PCT observations onto the 50 Hz control clock."""

    if step in controls:
        return _mapping(
            controls[step].get("post_step_observation"),
            "post_step_observation",
        )
    upper_index = bisect_left(control_steps, step)
    if upper_index == 0 or upper_index == len(control_steps):
        raise M0MobileError(f"cannot interpolate PCT control step {step}")
    lower_step = control_steps[upper_index - 1]
    upper_step = control_steps[upper_index]
    lower = _mapping(
        controls[lower_step].get("post_step_observation"),
        "post_step_observation",
    )
    upper = _mapping(
        controls[upper_step].get("post_step_observation"),
        "post_step_observation",
    )
    fraction = (step - lower_step) / (upper_step - lower_step)
    lower_root = _finite_vector(lower.get("robot_root_pose"), 7, "robot_root_pose")
    upper_root = _finite_vector(upper.get("robot_root_pose"), 7, "robot_root_pose")
    lower_tcp = _finite_vector(lower.get("tcp_pose"), 7, "tcp_pose")
    upper_tcp = _finite_vector(upper.get("tcp_pose"), 7, "tcp_pose")
    lower_names = _joint_names(lower)
    if _joint_names(upper) != lower_names:
        raise M0MobileError("PCT joint order changes within an episode")
    return {
        "robot_root_pose": _lerp(lower_root[:3], upper_root[:3], fraction)
        + _nlerp_quaternion(lower_root[3:], upper_root[3:], fraction),
        "robot_root_velocity": _lerp(
            _finite_vector(lower.get("robot_root_velocity"), 6, "robot_root_velocity"),
            _finite_vector(upper.get("robot_root_velocity"), 6, "robot_root_velocity"),
            fraction,
        ),
        "joint_positions": _lerp(
            _finite_vector(lower.get("joint_positions"), len(lower_names), "joint_positions"),
            _finite_vector(upper.get("joint_positions"), len(lower_names), "joint_positions"),
            fraction,
        ),
        "joint_velocities": _lerp(
            _finite_vector(lower.get("joint_velocities"), len(lower_names), "joint_velocities"),
            _finite_vector(upper.get("joint_velocities"), len(lower_names), "joint_velocities"),
            fraction,
        ),
        "tcp_pose": _lerp(lower_tcp[:3], upper_tcp[:3], fraction)
        + _nlerp_quaternion(lower_tcp[3:], upper_tcp[3:], fraction),
        "metadata": {"joint_names": lower_names},
    }


def _state28(post: Mapping[str, Any]) -> tuple[float, ...]:
    root_pose = _finite_vector(post.get("robot_root_pose"), 7, "robot_root_pose")
    root_q = _unit_quaternion(root_pose[3:], "robot root quaternion")
    inverse = _conjugate(root_q)
    root_velocity = _finite_vector(
        post.get("robot_root_velocity"), 6, "robot_root_velocity"
    )
    positions, velocities = _joint_state(post)
    _, _, tcp_xyz, tcp_q = _root_and_tcp_base(post)
    return build_live_state28(
        _rotate(inverse, root_velocity[:3]),
        _rotate(inverse, root_velocity[3:]),
        _rotate(inverse, (0.0, 0.0, -1.0)),
        positions[:6],
        velocities[:6],
        tcp_xyz,
        tcp_q,
        _gripper_fraction(post),
    )


def _root_and_tcp_base(
    post: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    root_pose = _finite_vector(post.get("robot_root_pose"), 7, "robot_root_pose")
    tcp_world = _finite_vector(post.get("tcp_pose"), 7, "tcp_pose")
    root_xyz = root_pose[:3]
    root_q = _unit_quaternion(root_pose[3:], "robot root quaternion")
    inverse = _conjugate(root_q)
    tcp_xyz = _rotate(
        inverse,
        tuple(tcp_world[index] - root_xyz[index] for index in range(3)),
    )
    tcp_q = _unit_quaternion(
        _multiply(inverse, _unit_quaternion(tcp_world[3:], "TCP world quaternion")),
        "TCP base quaternion",
    )
    return root_xyz, root_q, tcp_xyz, tcp_q


def _joint_state(
    post: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    names = _joint_names(post)
    positions = _finite_vector(post.get("joint_positions"), len(names), "joint_positions")
    velocities = _finite_vector(post.get("joint_velocities"), len(names), "joint_velocities")
    index = {str(name): offset for offset, name in enumerate(names)}
    missing = [name for name in PCT_REQUIRED_JOINTS if name not in index]
    if missing:
        raise M0MobileError("PCT frame is missing joints: " + ", ".join(missing))
    return (
        tuple(positions[index[name]] for name in PCT_REQUIRED_JOINTS),
        tuple(velocities[index[name]] for name in PCT_REQUIRED_JOINTS),
    )


def _joint_names(post: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = _mapping(post.get("metadata"), "post_step_observation.metadata")
    names = metadata.get("joint_names")
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise M0MobileError("PCT frame has no joint_names sequence")
    return tuple(str(name) for name in names)


def _gripper_fraction(post: Mapping[str, Any]) -> float:
    positions, _ = _joint_state(post)
    return min(1.0, max(0.0, ((positions[6] + positions[7]) / 2.0) / 0.04))


def _base_action_at(
    action_starts: Sequence[int],
    action_records: Sequence[Mapping[str, Any]],
    step: int,
) -> tuple[float, ...]:
    index = bisect_right(action_starts, step) - 1
    if index < 0:
        raise M0MobileError(f"no PCT command is available at control step {step}")
    record = action_records[index]
    action = _mapping(record.get("action"), "action")
    return _finite_vector(action.get("base_velocity"), 3, "action.base_velocity")


def _lerp(
    lower: Sequence[float],
    upper: Sequence[float],
    fraction: float,
) -> tuple[float, ...]:
    return tuple(
        float(start) + fraction * (float(end) - float(start))
        for start, end in zip(lower, upper, strict=True)
    )


def _nlerp_quaternion(
    lower: Sequence[float],
    upper: Sequence[float],
    fraction: float,
) -> tuple[float, float, float, float]:
    start = _unit_quaternion(lower, "interpolation start quaternion")
    end = _unit_quaternion(upper, "interpolation end quaternion")
    if sum(a * b for a, b in zip(start, end, strict=True)) < 0.0:
        end = tuple(-value for value in end)
    return _unit_quaternion(
        _lerp(start, end, fraction),
        "interpolated quaternion",
    )


def _camera_path(root: Path, sample: Mapping[str, Any], key: str) -> str:
    frames = _mapping(sample.get("camera_frames"), "camera_frames")
    frame = _mapping(frames.get(key), f"camera_frames.{key}")
    relative = Path(str(frame.get("raw_image_path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise M0MobileError(f"invalid {key} image path")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise M0MobileError(f"{key} image escapes the episode") from error
    if not resolved.is_file():
        raise M0MobileError(f"missing {key} image: {resolved}")
    return relative.as_posix()


@lru_cache(maxsize=1)
def _state_layout() -> tuple[str, ...]:
    config = load_lerobot_v3_config(DEFAULT_LEROBOT_V3_CONFIG_PATH)
    return tuple(config["features"]["state"]["names"])


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read PCT JSON {path}: {error}") from error
    return _mapping(value, str(path))


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise M0MobileError(f"{path}:{line_number}: {error}") from error
                yield _mapping(value, f"{path}:{line_number}")
    except OSError as error:
        raise M0MobileError(f"cannot read PCT JSONL {path}: {error}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise M0MobileError(f"{name} must be a non-negative integer")
    return value


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != length:
        raise M0MobileError(f"{name} must contain {length} values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise M0MobileError(f"{name} must contain finite values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "PCT_VISUAL_HISTORY_MODEL_TICKS",
    "PCT_VISUAL_HISTORY_SPAN_S",
    "audit_pct_episode",
    "discover_pct_episodes",
    "iter_pct_temporal_records",
    "materialize_pct_lerobot_v3",
]
