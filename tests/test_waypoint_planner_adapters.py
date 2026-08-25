from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.serve_waypoint_curobo import (
    DirectPoseCuroboService,
    build_parser as build_curobo_parser,
)

from conveyor_bench.conveyorvla.waypoint_execution import ArmPlan, ArmPlanUnavailableError
from conveyor_bench.conveyorvla.waypoint_planner_adapters import (
    APPROVED_ARM_VLA_COMMIT,
    CUROBO_REQUEST_SCHEMA,
    CUROBO_RESPONSE_SCHEMA,
    ArmVLADWAControllerAdapter,
    ArmVLAPCTPlannerAdapter,
    JointPathController,
    WaypointCuRoboPlannerAdapter,
)


@dataclass
class _State:
    step_index: int
    timestamp: float
    robot_root_pose: tuple[float, ...]
    robot_root_velocity: tuple[float, ...]
    metadata: dict = field(default_factory=dict)


@dataclass
class _Goal:
    x: float
    y: float
    yaw: float
    z: float


def _planner_scene(
    *,
    position=(0.0, 0.0, 0.0),
    quaternion=(1.0, 0.0, 0.0, 0.0),
    cuboids=(),
):
    return {
        "frame": "curobo-planner-base",
        "planner_base_from_query_base": {
            "position_xyz": list(position),
            "quaternion_wxyz": list(quaternion),
        },
        "cuboids_base": list(cuboids),
    }


def test_curobo_service_accepts_separate_runtime_workspace():
    args = build_curobo_parser().parse_args(
        [
            "--reference-root",
            "/clean/reference",
            "--workspace-root",
            "/runtime/assets",
        ]
    )
    assert args.reference_root.as_posix() == "/clean/reference"
    assert args.workspace_root.as_posix() == "/runtime/assets"


@dataclass
class _ReferencePlan:
    waypoints: tuple[tuple[float, float], ...]
    metadata: dict


class _ReferencePCT:
    def __init__(self, *, fallback=False):
        self.config = SimpleNamespace(enabled=True, fallback_to_astar=fallback)
        self.fallback_planner = None
        self.calls = []

    def plan(self, state, goal):
        self.calls.append((state, goal))
        return _ReferencePlan(
            waypoints=((state.robot_root_pose[0], state.robot_root_pose[1]), (goal.x, goal.y)),
            metadata={
                "planner": "pct",
                "path_3d": (
                    (state.robot_root_pose[0], state.robot_root_pose[1], state.robot_root_pose[2]),
                    (goal.x, goal.y, goal.z),
                ),
                "snap_end_distance_m": 0.0,
            },
        )


def test_pct_adapter_preserves_floor_height_and_disables_fallback():
    planner = _ReferencePCT()
    adapter = ArmVLAPCTPlannerAdapter(
        planner,
        simulation_state_factory=_State,
        nav_goal_factory=_Goal,
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    result = adapter.plan((1.0, 2.0, 1.25, 0.3), (1.2, 2.1, 1.25, 0.4))
    state, goal = planner.calls[0]
    assert state.robot_root_pose[2] == 1.25
    assert goal.z == 1.25
    assert state.metadata == {"consumer": "waypoint_executor_only", "model_input": False}
    assert result.snapped_goal_world == pytest.approx((1.2, 2.1, 1.25, 0.4))
    assert result.snap_distance_m == pytest.approx(0.0)
    assert result.metadata["current_world_pose"] == [1.0, 2.0, 1.25, 0.3]
    assert result.metadata["fallback_allowed"] is False
    with pytest.raises(ValueError, match=r"disable A\* fallback"):
        ArmVLAPCTPlannerAdapter(
            _ReferencePCT(fallback=True),
            simulation_state_factory=_State,
            nav_goal_factory=_Goal,
            reference_commit=APPROVED_ARM_VLA_COMMIT,
        )


class _ReferenceDWA:
    instances = []

    def __init__(self, path, grid, config, raw_grid_map=None):
        self.path = path
        self.grid = grid
        self.config = config
        self.raw = raw_grid_map
        self.calls = []
        self.instances.append(self)

    def compute_command(self, pose, velocity):
        self.calls.append((pose, velocity))
        return (0.2, 0.0, 0.1), {"feasible_candidates": 4}


def test_dwa_adapter_reuses_controller_for_one_pct_path_and_traces_debug():
    _ReferenceDWA.instances.clear()
    adapter = ArmVLADWAControllerAdapter(
        _ReferenceDWA,
        SimpleNamespace(name="strict"),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    local_map = {"grid_map": object(), "raw_grid_map": object()}
    path = ((0.0, 0.0), (0.2, 0.0))
    assert adapter.command(path, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), local_map) == (
        0.2,
        0.0,
        0.1,
    )
    adapter.command(path, (0.1, 0.0, 0.0), (0.1, 0.0, 0.0), local_map)
    assert len(_ReferenceDWA.instances) == 1
    assert adapter.last_trace["debug"]["feasible_candidates"] == 4
    assert adapter.last_trace["measured_body_velocity"] == [0.1, 0.0, 0.0]
    adapter.command(
        ((0.0, 0.0), (0.3, 0.0)),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        local_map,
    )
    assert len(_ReferenceDWA.instances) == 2


def test_direct_tcp_curobo_adapter_uses_executor_state_only_and_fails_closed():
    requests = []

    def transport(request):
        requests.append(request)
        return {
            "schema_version": CUROBO_RESPONSE_SCHEMA,
            "arm_vla_reference_commit": APPROVED_ARM_VLA_COMMIT,
            "ok": True,
            "reachable": True,
            "collision_free": True,
            "joint_path": [[0.0, 0.0], [0.05, -0.05]],
            "target_position_error_m": 0.001,
            "target_orientation_error_rad": 0.01,
            "metadata": {"planner": "curobo.MotionPlanner.plan_pose"},
        }

    adapter = WaypointCuRoboPlannerAdapter(
        transport,
        deployment="simulation",
        safety_gate=lambda request, response: (
            request["target_frame"] == "query-base-B_t" and response["collision_free"]
        ),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    plan = adapter.plan(
        (0.0, 0.0),
        (0.3, 0.0, 0.2, 0.0, 0.0, 0.0),
        _planner_scene(),
    )
    assert plan.planner == "curobo" and plan.reachable and plan.collision_free
    assert set(requests[0]) == {
        "schema_version",
        "command",
        "deployment",
        "target_frame",
        "target_units",
        "current_joints",
        "target_tcp_base",
        "scene_collision",
    }
    assert "phase" not in requests[0] and "target_pose_world" not in requests[0]
    assert plan.metadata["current_joints"] == [0.0, 0.0]
    assert plan.metadata["target_tcp_base"] == [0.3, 0.0, 0.2, 0.0, 0.0, 0.0]
    assert plan.metadata["planner_base_from_query_base"] == {
        "position_xyz": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    assert plan.metadata["scene_collision_cuboid_count"] == 0

    rejecting = WaypointCuRoboPlannerAdapter(
        transport,
        deployment="real",
        safety_gate=lambda _request, _response: False,
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    with pytest.raises(RuntimeError, match="real cuRobo safety gate rejected"):
        rejecting.plan(
            (0.0, 0.0),
            (0.3, 0.0, 0.2, 0.0, 0.0, 0.0),
            _planner_scene(),
        )

    unavailable = WaypointCuRoboPlannerAdapter(
        lambda _request: {
            "schema_version": CUROBO_RESPONSE_SCHEMA,
            "arm_vla_reference_commit": APPROVED_ARM_VLA_COMMIT,
            "ok": False,
            "error_kind": "plan_pose_unavailable",
            "error": "no collision-free direct-pose plan",
        },
        deployment="simulation",
        safety_gate=lambda _request, _response: True,
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    with pytest.raises(ArmPlanUnavailableError, match="no collision-free"):
        unavailable.plan(
            (0.0, 0.0),
            (0.3, 0.0, 0.2, 0.0, 0.0, 0.0),
            _planner_scene(),
        )

    broken = WaypointCuRoboPlannerAdapter(
        lambda _request: {
            "schema_version": CUROBO_RESPONSE_SCHEMA,
            "arm_vla_reference_commit": APPROVED_ARM_VLA_COMMIT,
            "ok": False,
            "error_kind": "service_error",
            "error": "planner process crashed",
        },
        deployment="simulation",
        safety_gate=lambda _request, _response: True,
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    with pytest.raises(RuntimeError, match="planner process crashed"):
        broken.plan(
            (0.0, 0.0),
            (0.3, 0.0, 0.2, 0.0, 0.0, 0.0),
            _planner_scene(),
        )


def test_joint_path_controller_stops_after_first_tcp_plan_finishes():
    controller = JointPathController()
    controller.reset(
        ArmPlan(
            joint_path=((0.0, 0.0), (0.05, -0.05)),
            planner="curobo",
            reachable=True,
            collision_free=True,
            target_position_error_m=0.0,
            target_orientation_error_rad=0.0,
        ),
        0.25,
    )
    target, done = controller.command((0.0, 0.0))
    assert target == (0.05, -0.05) and not done
    target, done = controller.command((0.05, -0.05))
    assert target == (0.05, -0.05) and done


def test_joint_path_controller_does_not_deadlock_on_intermediate_tracking_error():
    controller = JointPathController()
    controller.reset(
        ArmPlan(
            joint_path=((0.0,), (0.04,), (0.08,)),
            planner="curobo",
            reachable=True,
            collision_free=True,
            target_position_error_m=0.0,
            target_orientation_error_rad=0.0,
        ),
        0.25,
    )
    assert controller.command((0.0,)) == ((0.04,), False)
    assert controller.command((0.0,)) == ((0.08,), False)
    assert controller.command((0.051,)) == ((0.08,), True)


class _Profiler:
    def summary(self):
        return {"plan_pose": 0.01}


class _CuroboModule:
    EXPECTED_JOINT_NAMES = ("j1", "j2")
    ROBOT_URDF = "robot.urdf"
    Profiler = _Profiler
    PROFILER = _Profiler()

    def __init__(self):
        self.target_position = None
        self.target_quaternion = None
        self.collision = None

    def load_joint_limits_from_urdf(self, _path):
        return {"j1": (-1.0, 1.0), "j2": (-1.0, 1.0)}

    def clip_q_to_joint_limits(self, q, _limits):
        return q

    def make_world_collision_scene(self, payload):
        self.collision = payload
        return payload

    def update_planner_world(self, _planner, _scene):
        return None

    def plan_pose_path(
        self,
        *,
        planner,
        q_start,
        target_position,
        target_quaternion,
        segment_name,
    ):
        del planner, segment_name
        self.target_position = target_position
        self.target_quaternion = target_quaternion
        return np.stack((q_start, q_start + 0.05)), {"planner_success": True}

    def run_fk(self, _planner, _joints):
        return self.target_position, self.target_quaternion


def test_direct_pose_service_plans_the_exact_rpy_target_without_fallback():
    module = _CuroboModule()
    service = DirectPoseCuroboService(
        module,
        object(),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    response = service.handle(
        {
            "schema_version": CUROBO_REQUEST_SCHEMA,
            "command": "plan_tcp_target",
            "deployment": "simulation",
            "target_frame": "query-base-B_t",
            "target_units": ["m", "m", "m", "rad", "rad", "rad"],
            "current_joints": [0.0, 0.0],
            "target_tcp_base": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0],
            "scene_collision": _planner_scene(),
        }
    )
    assert response["ok"] is True
    np.testing.assert_allclose(response["joint_path"], [[0.0, 0.0], [0.05, 0.05]])
    assert module.target_position.tolist() == pytest.approx([0.3, 0.0, 0.2])
    assert module.target_quaternion.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert response["metadata"]["orientation_fallback_used"] is False
    assert module.collision == {"world_collision": {"cuboids_base": []}}


def test_direct_pose_service_marks_only_plan_pose_none_as_requeryable():
    class UnavailableModule(_CuroboModule):
        def plan_pose_path(self, **_kwargs):
            raise RuntimeError("approach_to_grasp: cuRobo plan_pose 返回 None。")

    service = DirectPoseCuroboService(
        UnavailableModule(),
        object(),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    response = service.handle(
        {
            "schema_version": CUROBO_REQUEST_SCHEMA,
            "command": "plan_tcp_target",
            "deployment": "simulation",
            "target_frame": "query-base-B_t",
            "target_units": ["m", "m", "m", "rad", "rad", "rad"],
            "current_joints": [0.0, 0.0],
            "target_tcp_base": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0],
            "scene_collision": _planner_scene(),
        }
    )
    assert response["ok"] is False
    assert response["error_kind"] == "plan_pose_unavailable"


def test_direct_pose_service_transforms_query_base_target_into_planner_base():
    module = _CuroboModule()
    service = DirectPoseCuroboService(
        module,
        object(),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    half_sqrt = 2.0**-0.5
    response = service.handle(
        {
            "schema_version": CUROBO_REQUEST_SCHEMA,
            "command": "plan_tcp_target",
            "deployment": "simulation",
            "target_frame": "query-base-B_t",
            "target_units": ["m", "m", "m", "rad", "rad", "rad"],
            "current_joints": [0.0, 0.0],
            "target_tcp_base": [0.3, 0.1, 0.2, 0.0, 0.0, 0.0],
            "scene_collision": _planner_scene(
                position=(1.0, 2.0, 3.0),
                quaternion=(half_sqrt, 0.0, 0.0, half_sqrt),
            ),
        }
    )
    assert response["ok"] is True
    assert module.target_position.tolist() == pytest.approx([0.9, 2.3, 3.2])
    assert module.target_quaternion.tolist() == pytest.approx(
        [half_sqrt, 0.0, 0.0, half_sqrt]
    )
    assert response["metadata"]["input_target_frame"] == "query-base-B_t"
    assert response["metadata"]["planner_target_frame"] == "curobo-planner-base"


def test_direct_pose_service_rejects_malformed_collision_instead_of_skipping_it():
    module = _CuroboModule()
    service = DirectPoseCuroboService(
        module,
        object(),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    request = {
        "schema_version": CUROBO_REQUEST_SCHEMA,
        "command": "plan_tcp_target",
        "deployment": "simulation",
        "target_frame": "query-base-B_t",
        "target_units": ["m", "m", "m", "rad", "rad", "rad"],
        "current_joints": [0.0, 0.0],
        "target_tcp_base": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0],
        "scene_collision": _planner_scene(
            cuboids=[
                {
                    "name": "table",
                    "pose_base": {
                        "position_xyz": [0.4, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "dims_xyz": [0.0, 0.5, 0.1],
                }
            ]
        ),
    }
    with pytest.raises(ValueError, match="dimensions must be positive"):
        service.handle(request)


def test_curobo_adapter_and_service_reject_missing_query_to_planner_transform():
    adapter = WaypointCuRoboPlannerAdapter(
        lambda _request: {},
        deployment="simulation",
        safety_gate=lambda _request, _response: True,
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    with pytest.raises(ValueError, match="collision frame"):
        adapter.plan((0.0, 0.0), (0.3, 0.0, 0.2, 0.0, 0.0, 0.0), {"cuboids_base": []})

    module = _CuroboModule()
    service = DirectPoseCuroboService(
        module,
        object(),
        reference_commit=APPROVED_ARM_VLA_COMMIT,
    )
    request = {
        "schema_version": CUROBO_REQUEST_SCHEMA,
        "command": "plan_tcp_target",
        "deployment": "simulation",
        "target_frame": "query-base-B_t",
        "target_units": ["m", "m", "m", "rad", "rad", "rad"],
        "current_joints": [0.0, 0.0],
        "target_tcp_base": [0.3, 0.0, 0.2, 0.0, 0.0, 0.0],
        "scene_collision": {
            "frame": "curobo-planner-base",
            "cuboids_base": [],
        },
    }
    with pytest.raises(ValueError, match="planner_base_from_query_base"):
        service.handle(request)
