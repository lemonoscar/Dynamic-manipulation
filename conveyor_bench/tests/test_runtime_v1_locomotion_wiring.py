from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conveyor_bench.isaac.arm_kinematics import CalibratedArmKinematics
from conveyor_bench.v1.protocol import FailureReason, Pose
from conveyor_bench.v1.stationary import (
    STATIONARY_DESTINATION_ZONE_ID,
    STATIONARY_SCENARIOS,
    STATIONARY_SPAWN_ORIGIN_XY_M,
    STATIONARY_TARGET_ASSET_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "runtime_v1.py"
)
RUNTIME_V2_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "runtime_v2.py"
)
RUN_M0_PATH = PROJECT_ROOT / "scripts" / "run_m0_closed_loop.py"
V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "v1.json"


def _runtime_tree() -> ast.Module:
    return ast.parse(RUNTIME_PATH.read_text(encoding="utf-8"))


def _runtime_class(tree: ast.Module) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConveyorRuntimeV1"
    )


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    runtime = _runtime_class(tree)
    return next(
        node
        for node in runtime.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _constants(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id.startswith("_LOCOMOTION_"):
            values[target.id] = ast.literal_eval(node.value)
    return values


def _literal_constant(tree: ast.Module, name: str) -> object:
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    )
    return ast.literal_eval(node.value)


def _load_mobile_preoracle_contract():
    tree = _runtime_tree()
    exception_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_MobilePreconditionFailure"
    )
    method_node = copy.deepcopy(_method(tree, "_mobile_preoracle_command"))
    method_node.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(
            body=[copy.deepcopy(exception_node), method_node],
            type_ignores=[],
        )
    )
    namespace = {
        "FailureReason": FailureReason,
        **_constants(tree),
    }
    exec(compile(module, str(RUNTIME_PATH), "exec"), namespace)
    return (
        namespace["_mobile_preoracle_command"],
        namespace["_MobilePreconditionFailure"],
        namespace,
    )


def test_runtime_locks_checkpoint_timing_and_cpu_fabric() -> None:
    tree = _runtime_tree()
    constants = _constants(tree)
    assert constants["_LOCOMOTION_PHYSICS_HZ"] == 400
    assert constants["_LOCOMOTION_POLICY_HZ"] == 50
    assert constants["_LOCOMOTION_DECIMATION"] == 8
    assert constants["_LOCOMOTION_WARMUP_POLICY_STEPS"] == 50

    init_source = ast.unparse(_method(tree, "__init__"))
    assert "use_fabric=True" in init_source
    assert "make_go2_x5_policy_cfg()" in init_source
    assert "make_go2_x5_cfg(fix_base=True)" in init_source
    assert "unexpected locomotion timing" in init_source

    resolve_source = ast.unparse(_method(tree, "_resolve_entities"))
    assert "orientation_tolerance=0.08" in resolve_source
    assert "self.place_position_kinematics = kinematics_factory" in (
        resolve_source
    )
    assert "orientation_weight=0.002" in resolve_source
    assert "orientation_tolerance=4.0" in resolve_source


def test_loaded_arm_slew_preserves_the_audited_payload_envelope() -> None:
    source = ast.unparse(_method(_runtime_tree(), "_slew_arm_target"))

    assert "(0.008, 0.01, 0.01, 0.01, 0.008, 0.01)" in source
    assert "if carrying_object" in source
    assert "else (0.02,) * len(ARM_JOINT_NAMES)" in source


def test_mobile_place_orientation_uses_the_live_tcp_state() -> None:
    source = ast.unparse(_method(_runtime_tree(), "_mobile_place_target"))

    assert "self.place_position_kinematics.solve" in source
    assert "state['tcp_base'].wxyz" in source
    assert "seed=self._arm_ik_seed" in source


def test_mobile_release_is_a_reachable_drop_above_the_tray_rim() -> None:
    source = ast.unparse(_method(_runtime_tree(), "_make_oracle"))

    assert "asset.half_extents_xyz[2] + 0.09" in source
    assert "reachable_release_x = zone_x + 0.03" in source
    assert "reachable_release_y = zone_y - math.copysign(0.1, zone_y)" in source


def test_v1_carry_turn_uses_the_audited_bidirectional_timeout() -> None:
    source = ast.unparse(_method(_runtime_tree(), "_mobile_carry_stage_timeout_s"))

    assert "'turn': 10.0" in source


def test_v1_compact_carry_tcp_is_frozen_to_policy_usd_kinematics() -> None:
    tree = _runtime_tree()
    arm = _literal_constant(tree, "_MOBILE_COMPACT_ARM")
    frozen_tcp = _literal_constant(tree, "_MOBILE_COMPACT_TCP_BASE")
    computed_tcp, _ = CalibratedArmKinematics.in_policy_usd_root_frame().forward(
        arm
    )

    assert arm == pytest.approx((0.0, 0.3, 0.5, 0.0, 0.0, 0.0))
    assert frozen_tcp == pytest.approx(computed_tcp, abs=1.0e-12)


def test_v1_carry_places_from_the_measured_loaded_turn_endpoint() -> None:
    runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    source = ast.unparse(_method(_runtime_tree(), "_mobile_post_turn_stage"))

    assert "return 'settle'" in source
    assert "_MOBILE_NAVIGATE_HEADING_TOLERANCE_RAD = 0.21" in runtime_source

    forward = ast.unparse(
        _method(_runtime_tree(), "_mobile_navigate_forward_speed_mps")
    )
    lateral = ast.unparse(
        _method(_runtime_tree(), "_mobile_navigate_lateral_speed_mps")
    )
    bearing = ast.unparse(
        _method(_runtime_tree(), "_mobile_navigation_yaw_error")
    )
    yaw = ast.unparse(_method(_runtime_tree(), "_mobile_navigation_yaw_command"))
    drive_tolerance = ast.unparse(
        _method(
            _runtime_tree(),
            "_mobile_navigation_drive_heading_tolerance_rad",
        )
    )
    carry = ast.unparse(_method(_runtime_tree(), "_mobile_carry_command"))
    assert "return 0.2" in forward
    assert "return 0.0" in lateral
    assert "math.atan2(delta_y, delta_x)" in bearing
    assert "_MOBILE_NAVIGATE_HEADING_TOLERANCE_RAD" in drive_tolerance
    assert "drive_heading_tolerance" in carry
    assert "else 0.0" in carry
    assert "if abs(yaw_error_rad) <= _MOBILE_NAVIGATE_HEADING_TOLERANCE_RAD" in yaw
    assert "1.5 * yaw_error_rad" in yaw
    assert "min(0.35" in yaw


def test_runtime_writes_explicit_actuators_every_physics_substep() -> None:
    tree = _runtime_tree()
    run_episode = _method(tree, "_run_episode")
    decimation_loops = [
        node
        for node in ast.walk(run_episode)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Attribute)
        and node.iter.args[0].attr == "physics_decimation"
    ]
    assert len(decimation_loops) == 1
    body_source = [ast.unparse(node) for node in decimation_loops[0].body]
    assert body_source[:2] == [
        "self.scene.write_data_to_sim()",
        "self._step_physics()",
    ]


def test_runtime_checks_fixed_base_and_resets_policy_warmup() -> None:
    tree = _runtime_tree()
    resolve_source = ast.unparse(_method(tree, "_resolve_entities"))
    reset_source = ast.unparse(_method(tree, "_reset_episode"))
    apply_source = ast.unparse(_method(tree, "_apply_base_command"))

    assert "self.robot.is_fixed_base" in resolve_source
    assert "spawned robot fixed-base state" in resolve_source
    assert "self._locomotion_policy_step_count = 0" in reset_source
    assert (
        "(self._locomotion_policy_step_count + 1) / "
        "_LOCOMOTION_WARMUP_POLICY_STEPS"
    ) in apply_source
    assert "self._locomotion_policy_step_count += 1" in apply_source


def test_runtime_consumes_split_local_tasking_contract() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    make_task_source = ast.unparse(_method(tree, "_make_task"))
    assert "curriculum_split" in options_source
    assert "task_family" in options_source
    assert "instruction_language" in options_source
    assert "split_object_ids()[self.curriculum_split]" in options_source
    assert "split_object_ids()[self.options.curriculum_split]" in (
        make_task_source
    )


def test_stationary_belt_is_an_explicit_seeded_diagnostic() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    make_task_source = ast.unparse(_method(tree, "_make_task"))
    summary_source = ast.unparse(_method(tree, "_summary_task_type"))
    intercept_source = ast.unparse(_method(tree, "_intercept_y_world"))

    assert "belt_speed_mps must be finite and non-negative" in options_source
    assert "stationary-belt diagnostic requires active_object_count=1" in (
        options_source
    )
    assert "stationary-belt diagnostic requires single_target" in options_source
    assert "stationary-belt diagnostic requires part_red_block" in options_source
    assert "stationary-belt diagnostic requires sort_bin_blue" in options_source
    assert "TaskType.STATIONARY_SORT" in summary_source
    assert "TaskType.STATIONARY_SORT" in make_task_source
    assert "stationary_belt_diagnostic" in make_task_source
    assert "stationary_spawn_offset_y_m" in make_task_source
    assert "task_split = stationary_scenario.split" in make_task_source
    assert "Pick the stationary" in make_task_source
    assert "resolved.spawn_y_by_id" in intercept_source
    assert "'tasking_schema_version': TASKING_SCHEMA_VERSION" in (
        make_task_source
    )
    assert "'curriculum_split': self.options.curriculum_split.value" in (
        make_task_source
    )
    assert "InstructionLanguage.ENGLISH" in make_task_source


def test_stationary_runtime_registry_matches_machine_readable_config() -> None:
    config = json.loads(V1_CONFIG_PATH.read_text(encoding="utf-8"))[
        "stationary_diagnostic"
    ]
    registry_scenarios = {
        str(seed): {
            "split": scenario.split,
            "object_xy_offset_m": list(scenario.object_xy_offset_m),
            "root_xy_offset_m": list(scenario.root_xy_offset_m),
            "root_yaw_rad": scenario.root_yaw_rad,
        }
        for seed, scenario in STATIONARY_SCENARIOS.items()
    }
    registry_splits = {
        split: sorted(
            seed
            for seed, scenario in STATIONARY_SCENARIOS.items()
            if scenario.split == split
        )
        for split in {scenario.split for scenario in STATIONARY_SCENARIOS.values()}
    }

    assert registry_scenarios == config["scenarios"]
    assert registry_splits == config["scenario_splits"]
    configured_seeds = [
        seed for seeds in config["scenario_splits"].values() for seed in seeds
    ]
    assert len(configured_seeds) == len(set(configured_seeds))
    assert sorted(configured_seeds) == sorted(STATIONARY_SCENARIOS)
    assert STATIONARY_TARGET_ASSET_ID == config["target_asset_id"]
    assert STATIONARY_DESTINATION_ZONE_ID == config["destination_zone_id"]
    assert list(STATIONARY_SPAWN_ORIGIN_XY_M) == config["spawn_origin_xy_m"]


def test_m0_pregrasp_workspace_guard_is_explicit_and_diagnostic_only() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    episode_source = ast.unparse(_method(tree, "_run_episode"))
    apply_source = ast.unparse(_method(tree, "_apply_m0_mobile_action"))
    v2_source = RUNTIME_V2_PATH.read_text(encoding="utf-8")
    cli_source = RUN_M0_PATH.read_text(encoding="utf-8")

    assert "m0_pregrasp_workspace_guard: bool = False" in options_source
    assert "requires online M0" in options_source
    assert "self.options.m0_pregrasp_workspace_guard and phase == 'pregrasp'" in (
        episode_source
    )
    assert "guard_pregrasp_workspace: bool=False" in apply_source
    assert "m0_full_with_workspace_guard" in episode_source
    assert "diagnostic_only" in episode_source
    assert "m0_pregrasp_workspace_guard: bool = False" in v2_source
    assert '"--pregrasp-workspace-guard"' in cli_source
    assert "action=\"store_true\"" in cli_source


def test_m0_mobile_approach_assist_is_audited_and_default_off() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    episode_source = ast.unparse(_method(tree, "_run_episode"))
    v2_source = RUNTIME_V2_PATH.read_text(encoding="utf-8")
    cli_source = RUN_M0_PATH.read_text(encoding="utf-8")

    assert "m0_mobile_approach_assist: bool = False" in options_source
    assert "m0_mobile_approach_assist requires online M0" in options_source
    assert "diagnostic_mobile_approach_assist" in episode_source
    assert "approach_assist_active" in episode_source
    assert "policy_requests_before_object_spawn" in episode_source
    assert "arm_max_joint_error_rad" in episode_source
    assert "policy_request_suppressed" in episode_source
    assert "m0_mobile_approach_assist and oracle is None" in episode_source
    assert "policy_proposed_action_applied" in episode_source
    assert "m0_mobile_approach_assist" in v2_source
    assert '"--mobile-approach-assist"' in cli_source


def test_m0_pregrasp_staging_assist_is_static_audited_and_default_off() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    episode_source = ast.unparse(_method(tree, "_run_episode"))
    pose_source = ast.unparse(
        _method(tree, "_m0_pregrasp_staging_pose")
    )
    command_source = ast.unparse(
        _method(tree, "_command_m0_pregrasp_staging_assist")
    )
    v2_source = RUNTIME_V2_PATH.read_text(encoding="utf-8")
    cli_source = RUN_M0_PATH.read_text(encoding="utf-8")

    assert "m0_pregrasp_staging_assist: bool = False" in options_source
    assert "mutually exclusive diagnostics" in options_source
    assert "m0_pregrasp_staging_assist requires online M0" in options_source
    assert "previous_phase == 'pregrasp'" in episode_source
    assert "phase in _M0_TRANSITION_CHUNK_PHASES" in episode_source
    assert "diagnostic_pregrasp_staging_assist" in episode_source
    assert "policy_proposed_action_applied" in episode_source
    assert episode_source.count(
        "self._command_m0_pregrasp_staging_assist"
    ) == 2
    assert "'observation_phase': phase" in episode_source
    assert "resolved.spawn_x_by_id" in pose_source
    assert "BELT_TOP_Z_M" in pose_source
    assert "_M0_DIAGNOSTIC_PREGRASP_CLEARANCE_M" in pose_source
    assert "object_state" not in pose_source
    assert "oracle" not in pose_source
    assert "target_uses_realtime_object_state" in command_source
    assert "activation_uses_shadow_oracle_phase" in command_source
    assert "handoff_uses_shadow_oracle_phase" in command_source
    assert "m0_action_applied" in command_source
    assert "m0_pregrasp_staging_assist: bool = False" in v2_source
    assert '"--pregrasp-staging-assist"' in cli_source
    assert "add_mutually_exclusive_group" in cli_source


def test_m0_pregrasp_staging_pose_uses_only_registered_scene_geometry() -> None:
    tree = _runtime_tree()
    method_node = copy.deepcopy(
        _method(tree, "_m0_pregrasp_staging_pose")
    )
    method_node.decorator_list = []
    module = ast.fix_missing_locations(
        ast.Module(body=[method_node], type_ignores=[])
    )
    namespace = {
        "_ResolvedTask": object,
        "Pose": Pose,
        "OBJECT_LANE_X_M": 0.65,
        "_MOBILE_INTERCEPT_Y_WORLD_M": 0.10,
        "BELT_TOP_Z_M": 0.50,
        "_M0_DIAGNOSTIC_PREGRASP_CLEARANCE_M": 0.10,
    }
    exec(compile(module, str(RUNTIME_PATH), "exec"), namespace)
    asset = SimpleNamespace(
        object_id="part_red_block",
        half_extents_xyz=(0.024, 0.024, 0.032),
        grasp_affordances=(
            SimpleNamespace(tcp_offset_xyz=(0.0, 0.0, 0.004)),
        ),
    )
    resolved = SimpleNamespace(
        target_asset=asset,
        spawn_x_by_id={"part_red_block": 0.65},
    )

    runtime = SimpleNamespace(
        _intercept_y_world=lambda _resolved: 0.10,
    )
    pose = namespace["_m0_pregrasp_staging_pose"](runtime, resolved)

    assert pose.xyz == pytest.approx((0.65, 0.10, 0.636))
    assert pose.wxyz == pytest.approx((-1.0, 0.0, 0.0, 0.0))


def test_m0_carry_teacher_uses_the_cartesian_policy_executor() -> None:
    tree = _runtime_tree()
    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "RuntimeOptionsV1"
    )
    options_source = ast.unparse(options)
    episode_source = ast.unparse(_method(tree, "_run_episode"))
    v2_source = RUNTIME_V2_PATH.read_text(encoding="utf-8")
    cli_source = RUN_M0_PATH.read_text(encoding="utf-8")

    assert (
        "m0_carry_retract_teacher_executor: bool = False"
        in options_source
    )
    assert (
        "m0_carry_retract_teacher_executor requires online M0"
        in options_source
    )
    assert "phase == 'carry_retract'" in episode_source
    assert (
        "self._apply_m0_mobile_action(shadow_teacher_physical_action10, "
        "state_before)"
    ) in episode_source
    assert "diagnostic_teacher_via_m0_executor" in episode_source
    assert "shadow_oracle_projected_m0_physical10" in episode_source
    assert "'direct_joint_target_write': False" in episode_source
    assert (
        "m0_carry_retract_teacher_executor: bool = False" in v2_source
    )
    assert '"--carry-retract-teacher-executor"' in cli_source


def test_forbidden_belt_intrusion_is_spatially_scoped() -> None:
    tree = _runtime_tree()
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_tcp_intrudes_belt"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(helper)], type_ignores=[])
    )
    namespace = {
        "Sequence": list,
        "BELT_CENTER_X_M": 0.70,
        "BELT_CENTER_Y_M": 0.0,
        "BELT_WIDTH_M": 0.42,
        "BELT_LENGTH_M": 1.20,
        "BELT_TOP_Z_M": 0.50,
    }
    exec(compile(module, str(RUNTIME_PATH), "exec"), namespace)
    intrudes = namespace["_tcp_intrudes_belt"]
    assert intrudes((0.70, 0.0, 0.47))
    assert not intrudes((0.20, 0.0, 0.47))
    assert not intrudes((0.70, 0.80, 0.47))
    assert not intrudes((0.70, 0.0, 0.49))


def test_mobile_preoracle_timeout_and_fall_are_task_failures() -> None:
    run_source = ast.unparse(_method(_runtime_tree(), "_run_episode"))
    assert "failure_reason = error.reason" in run_source
    assert (
        "abort_metadata['precondition_failure'] = error.code"
        in run_source
    )

    transition, failure_type, constants = _load_mobile_preoracle_contract()
    runtime = SimpleNamespace(_mobile_stable_since_s=None)

    with pytest.raises(failure_type) as fallen:
        transition(
            runtime,
            stage="mobile_approach",
            stage_started_at=1.0,
            sim_time_s=1.2,
            root_x=-0.20,
            root_planar_speed_mps=0.0,
            robot_fallen=True,
        )
    assert fallen.value.reason is FailureReason.ROBOT_FALLEN
    assert fallen.value.code == "robot_fallen"

    with pytest.raises(failure_type) as timeout:
        transition(
            runtime,
            stage="mobile_approach",
            stage_started_at=1.0,
            sim_time_s=1.0 + constants["_LOCOMOTION_APPROACH_TIMEOUT_S"],
            root_x=-0.20,
            root_planar_speed_mps=0.0,
            robot_fallen=False,
        )
    assert timeout.value.reason is FailureReason.TIMEOUT
    assert timeout.value.code == "approach_timeout"


def test_mobile_preoracle_requires_continuous_position_and_speed_dwell() -> None:
    transition, _, constants = _load_mobile_preoracle_contract()
    runtime = SimpleNamespace(_mobile_stable_since_s=None)
    target_x = constants["_LOCOMOTION_APPROACH_TARGET_X_M"]
    dwell = constants["_LOCOMOTION_STABLE_DWELL_S"]

    stage, stage_started, command, ready = transition(
        runtime,
        stage="mobile_stabilize",
        stage_started_at=2.0,
        sim_time_s=2.1,
        root_x=target_x,
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert (stage, stage_started, command, ready) == (
        "mobile_stabilize",
        2.0,
        (0.0, 0.0, 0.0),
        False,
    )
    assert runtime._mobile_stable_since_s == pytest.approx(2.1)

    # Breaking either condition resets the continuous dwell.
    transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.2,
        root_x=target_x,
        root_planar_speed_mps=0.09,
        robot_fallen=False,
    )
    assert runtime._mobile_stable_since_s is None
    transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.3,
        root_x=target_x,
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert runtime._mobile_stable_since_s == pytest.approx(2.3)

    transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.4,
        root_x=(
            target_x
            + constants[
                "_LOCOMOTION_APPROACH_POSITION_TOLERANCE_M"
            ]
            + 0.01
        ),
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert runtime._mobile_stable_since_s is None
    transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.5,
        root_x=target_x,
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert runtime._mobile_stable_since_s == pytest.approx(2.5)

    stage, stage_started, command, ready = transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.5 + dwell - 0.01,
        root_x=target_x,
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert stage == "mobile_stabilize"
    assert ready is False

    stage, _, command, ready = transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=2.5 + dwell,
        root_x=target_x,
        root_planar_speed_mps=0.05,
        robot_fallen=False,
    )
    assert stage == "arm_preposition"
    assert command == (0.0, 0.0, 0.0)
    assert ready is False


def test_mobile_preoracle_warmup_and_position_gate() -> None:
    transition, _, constants = _load_mobile_preoracle_contract()
    runtime = SimpleNamespace(_mobile_stable_since_s=None)
    warmup_seconds = (
        constants["_LOCOMOTION_WARMUP_POLICY_STEPS"]
        / constants["_LOCOMOTION_POLICY_HZ"]
    )

    stage, _, command, _ = transition(
        runtime,
        stage="mobile_settle",
        stage_started_at=0.0,
        sim_time_s=warmup_seconds - 0.02,
        root_x=-0.22,
        root_planar_speed_mps=0.0,
        robot_fallen=False,
    )
    assert stage == "mobile_settle"
    assert command == (0.0, 0.0, 0.0)

    stage, stage_started, command, _ = transition(
        runtime,
        stage="mobile_settle",
        stage_started_at=0.0,
        sim_time_s=warmup_seconds,
        root_x=-0.22,
        root_planar_speed_mps=0.0,
        robot_fallen=False,
    )
    assert stage == "mobile_approach"
    assert command == (0.20, 0.0, 0.0)

    stage, stage_started, command, _ = transition(
        runtime,
        stage=stage,
        stage_started_at=stage_started,
        sim_time_s=warmup_seconds + 0.5,
        root_x=constants["_LOCOMOTION_APPROACH_TARGET_X_M"],
        root_planar_speed_mps=0.20,
        robot_fallen=False,
    )
    assert stage == "mobile_stabilize"
    assert command == (0.0, 0.0, 0.0)
