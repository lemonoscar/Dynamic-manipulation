from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from conveyor_bench.v1.protocol import FailureReason


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = (
    PROJECT_ROOT / "src" / "conveyor_bench" / "isaac" / "runtime_v1.py"
)


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
    assert "'tasking_schema_version': TASKING_SCHEMA_VERSION" in (
        make_task_source
    )
    assert "'curriculum_split': self.options.curriculum_split.value" in (
        make_task_source
    )
    assert "InstructionLanguage.ENGLISH" in make_task_source


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
