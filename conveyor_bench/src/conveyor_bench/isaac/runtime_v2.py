"""Isaac Sim adapter for the ConveyorBench V2 benchmark suite."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from conveyor_bench.v1.assets import (
    load_object_registry,
    load_receptacles,
)
from conveyor_bench.v1.protocol import Pose, RobotMode, TaskType
from conveyor_bench.v1.tasking import (
    CurriculumSplit,
    InstructionLanguage,
    TaskFamily,
)
from conveyor_bench.v2.camera_contracts import camera_contract_for_scene
from conveyor_bench.v2.config import DEFAULT_SUITE_CONFIG, SceneId
from conveyor_bench.v2.tasking import (
    build_task_context,
    validate_task_combination,
)

from .runtime_v1 import (
    _ResolvedTarget,
    _ResolvedTask,
    ConveyorRuntimeV1,
)
from .scene_remote_delivery import (
    REMOTE_RECEPTACLE_MANIFEST_PATH,
    ConveyorRemoteDeliverySceneCfg,
)
from .scene_v1 import OBJECT_LANE_X_M, OBJECT_SPAWN_Y_M, ConveyorSceneV1Cfg
from .scene_v2 import ConveyorNearSortV2SceneCfg


V2_ASSET_LOCK_PATH = (
    Path(__file__).resolve().parents[3] / "assets" / "asset_lock_v2.json"
)


@dataclass(frozen=True)
class RuntimeOptionsV2:
    """Validated V2 collection options resolved before simulator startup."""

    output_root: Path
    scene_id: SceneId = SceneId.TRANSVERSE_NEAR_SORT_V2
    task_family: TaskFamily = TaskFamily.SINGLE_TARGET
    robot_mode: RobotMode = RobotMode.FIXED_BASE
    episodes: int = 1
    seed: int = 0
    belt_speed_mps: float = 0.06
    max_duration_s: float | None = None
    device: str = "cpu"
    enable_cameras: bool = False
    save_camera_frames: bool = False
    curriculum_split: CurriculumSplit = CurriculumSplit.TRAIN
    instruction_language: InstructionLanguage = (
        InstructionLanguage.BILINGUAL
    )
    m0_policy_endpoint: str | None = None
    m0_state_statistics: Path | None = None
    m0_policy_timeout_s: float = 30.0
    m0_policy_seed: int = 20260803
    m0_actions_per_replan: int = 2
    m0_transition_actions_per_replan: int = 12
    m0_mobile_approach_assist: bool = False
    m0_pregrasp_workspace_guard: bool = False
    m0_pregrasp_staging_assist: bool = False
    m0_carry_retract_teacher_executor: bool = False

    def __post_init__(self) -> None:
        try:
            scene_id = (
                self.scene_id
                if isinstance(self.scene_id, SceneId)
                else SceneId(self.scene_id)
            )
            family = (
                self.task_family
                if isinstance(self.task_family, TaskFamily)
                else TaskFamily(self.task_family)
            )
            mode = (
                self.robot_mode
                if isinstance(self.robot_mode, RobotMode)
                else RobotMode(self.robot_mode)
            )
            split = (
                self.curriculum_split
                if isinstance(self.curriculum_split, CurriculumSplit)
                else CurriculumSplit(self.curriculum_split)
            )
            language = (
                self.instruction_language
                if isinstance(
                    self.instruction_language, InstructionLanguage
                )
                else InstructionLanguage(self.instruction_language)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid V2 runtime option: {error}") from error

        validate_task_combination(scene_id, family, mode)
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "task_family", family)
        object.__setattr__(self, "robot_mode", mode)
        object.__setattr__(self, "curriculum_split", split)
        object.__setattr__(self, "instruction_language", language)
        object.__setattr__(self, "output_root", Path(self.output_root))

        if (
            isinstance(self.episodes, bool)
            or not isinstance(self.episodes, int)
            or self.episodes <= 0
        ):
            raise ValueError("episodes must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.belt_speed_mps, bool)
            or not math.isfinite(self.belt_speed_mps)
            or not any(
                math.isclose(
                    self.belt_speed_mps,
                    candidate,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for candidate in DEFAULT_SUITE_CONFIG.belt_speeds_mps
            )
        ):
            raise ValueError(
                "belt_speed_mps must be one of the frozen V2 belt speeds"
            )
        if self.max_duration_s is None:
            object.__setattr__(
                self,
                "max_duration_s",
                DEFAULT_SUITE_CONFIG.scene(scene_id).default_max_duration_s,
            )
        elif (
            isinstance(self.max_duration_s, bool)
            or not math.isfinite(self.max_duration_s)
            or self.max_duration_s <= 0.0
        ):
            raise ValueError("max_duration_s must be finite and positive")
        preflight_context = build_task_context(
            seed=self.seed,
            scene_id=scene_id,
            family=family,
            mode=mode,
            split=split,
            instruction_language=language,
        )
        initialization_end_s = max(
            float(entry["initialization_end_s"])
            for entry in preflight_context.task.metadata["spawn_schedule"]
        )
        assert self.max_duration_s is not None
        if self.max_duration_s <= initialization_end_s:
            raise ValueError(
                "max_duration_s must exceed the task initialization horizon "
                f"({initialization_end_s:g} s)"
            )
        if self.device != "cpu":
            raise ValueError(
                "V2 currently requires CPU PhysX for the audited conveyor "
                "surface-velocity contact path"
            )
        if self.save_camera_frames and not self.enable_cameras:
            raise ValueError("save_camera_frames requires enable_cameras")
        if (
            isinstance(self.m0_policy_timeout_s, bool)
            or not math.isfinite(self.m0_policy_timeout_s)
            or self.m0_policy_timeout_s <= 0.0
        ):
            raise ValueError("m0_policy_timeout_s must be finite and positive")
        if (
            isinstance(self.m0_policy_seed, bool)
            or not isinstance(self.m0_policy_seed, int)
            or self.m0_policy_seed < 0
        ):
            raise ValueError("m0_policy_seed must be a non-negative integer")
        if (
            isinstance(self.m0_actions_per_replan, bool)
            or not isinstance(self.m0_actions_per_replan, int)
            or not 1 <= self.m0_actions_per_replan <= 16
        ):
            raise ValueError("m0_actions_per_replan must be within [1, 16]")
        if (
            isinstance(self.m0_transition_actions_per_replan, bool)
            or not isinstance(self.m0_transition_actions_per_replan, int)
            or not 1 <= self.m0_transition_actions_per_replan <= 16
        ):
            raise ValueError(
                "m0_transition_actions_per_replan must be within [1, 16]"
            )
        if not isinstance(self.m0_pregrasp_workspace_guard, bool):
            raise TypeError("m0_pregrasp_workspace_guard must be a bool")
        if not isinstance(self.m0_mobile_approach_assist, bool):
            raise TypeError("m0_mobile_approach_assist must be a bool")
        if not isinstance(self.m0_pregrasp_staging_assist, bool):
            raise TypeError("m0_pregrasp_staging_assist must be a bool")
        if not isinstance(self.m0_carry_retract_teacher_executor, bool):
            raise TypeError(
                "m0_carry_retract_teacher_executor must be a bool"
            )
        if (
            self.m0_pregrasp_workspace_guard
            and self.m0_pregrasp_staging_assist
        ):
            raise ValueError(
                "the pregrasp workspace guard and staging assist are "
                "mutually exclusive diagnostics"
            )
        if self.m0_policy_endpoint is not None:
            if not isinstance(self.m0_policy_endpoint, str):
                raise TypeError("m0_policy_endpoint must be a string")
            if scene_id is not SceneId.TRANSVERSE_NEAR_SORT_V2:
                raise ValueError("the first online M0 gate requires near-sort")
            if family is not TaskFamily.SINGLE_TARGET:
                raise ValueError("the first online M0 gate requires single_target")
            if mode is not RobotMode.WHOLE_BODY_POLICY:
                raise ValueError("online M0 requires whole_body_policy")
            if not self.enable_cameras:
                raise ValueError("online M0 requires enable_cameras")
            if self.m0_state_statistics is None:
                raise ValueError("online M0 requires m0_state_statistics")
            object.__setattr__(
                self, "m0_state_statistics", Path(self.m0_state_statistics)
            )
        elif self.m0_mobile_approach_assist:
            raise ValueError(
                "m0_mobile_approach_assist requires online M0"
            )
        elif self.m0_pregrasp_workspace_guard:
            raise ValueError(
                "m0_pregrasp_workspace_guard requires online M0"
            )
        elif self.m0_pregrasp_staging_assist:
            raise ValueError(
                "m0_pregrasp_staging_assist requires online M0"
            )
        elif self.m0_carry_retract_teacher_executor:
            raise ValueError(
                "m0_carry_retract_teacher_executor requires online M0"
            )


class ConveyorRuntimeV2(ConveyorRuntimeV1):
    """V2 scenes and task contracts over the canonical V1 collector."""

    options: RuntimeOptionsV2

    def _make_scene_cfg(self) -> ConveyorSceneV1Cfg:
        scene_class = (
            ConveyorRemoteDeliverySceneCfg
            if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2
            else ConveyorNearSortV2SceneCfg
        )
        return scene_class(
            num_envs=1,
            env_spacing=3.0,
            replicate_physics=True,
            clone_in_fabric=False,
            lazy_sensor_update=True,
        )

    def _viewer_camera_view(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return (-2.80, -2.60, 3.20), (0.25, 0.0, 0.45)
        return super()._viewer_camera_view()

    def _asset_lock_path(self) -> Path:
        return V2_ASSET_LOCK_PATH

    def _layout_id(self) -> str:
        return DEFAULT_SUITE_CONFIG.scene(self.options.scene_id).layout_id

    def _camera_contract(self) -> dict[str, Any]:
        return camera_contract_for_scene(self.options.scene_id)

    def _guard_locomotion_command(
        self, command: Sequence[float]
    ) -> tuple[float, float, float]:
        if self.options.scene_id is not SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return super()._guard_locomotion_command(command)
        vx, vy, wz = (float(value) for value in command)
        if 0.0 < abs(vx) < 0.16:
            vx = 0.0
        return (
            min(0.30, max(-0.30, vx)),
            min(0.20, max(-0.20, vy)),
            min(0.35, max(-0.35, wz)),
        )

    def _summary_task_type(self) -> TaskType:
        if self.options.task_family is TaskFamily.CONTINUOUS_MULTI_TARGET:
            return TaskType.CONTINUOUS_SORT
        return TaskType.DYNAMIC_SORT

    def _extra_episode_metadata(
        self, resolved: _ResolvedTask
    ) -> dict[str, Any]:
        suite = resolved.manifest.metadata["benchmark_suite"]
        return {
            "benchmark_suite": suite,
            "benchmark_profile": "conveyor-bench-v2",
        }

    def _extra_step_metadata(
        self,
        resolved: _ResolvedTask,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        del state
        return {
            "benchmark_suite_version": "conveyor-bench-v2",
            "scene_id": self.options.scene_id.value,
            "task_family": self.options.task_family.value,
            "current_target_id": resolved.target_instance_id,
            "current_subtask_index": resolved.current_target_index,
        }

    def _make_task(self, seed: int) -> _ResolvedTask:
        context = build_task_context(
            seed=seed,
            scene_id=self.options.scene_id,
            family=self.options.task_family,
            mode=self.options.robot_mode,
            split=self.options.curriculum_split,
            instruction_language=self.options.instruction_language,
        )
        manifest = replace(
            context.task,
            belt_speed_mps=self.options.belt_speed_mps,
            max_duration_s=float(self.options.max_duration_s),
            metadata={
                **dict(context.task.metadata),
                "belt_speed_mps": self.options.belt_speed_mps,
            },
        )
        asset_by_id = {
            asset.object_id: asset for asset in load_object_registry()
        }
        assets = tuple(
            asset_by_id[obj.asset_id] for obj in manifest.objects
        )
        receptacle_assets = (
            load_receptacles(REMOTE_RECEPTACLE_MANIFEST_PATH)
            if self.options.scene_id
            is SceneId.MOBILE_REMOTE_DELIVERY_V2
            else load_receptacles()
        )
        receptacles = {item.zone_id: item for item in receptacle_assets}
        targets = tuple(
            _ResolvedTarget(
                instance_id=target_id,
                asset=asset_by_id[context.instance_asset_map[target_id]],
                zone=receptacles[
                    context.destination_zone_by_target[target_id]
                ],
            )
            for target_id in context.target_sequence_ids
        )
        if len(targets) > 1:
            spawn_y_by_id = {
                asset.object_id: OBJECT_SPAWN_Y_M for asset in assets
            }
        else:
            spawn_slots = (
                0.34,
                0.48,
                0.10,
                -0.14,
                -0.34,
                0.25,
                -0.02,
                0.42,
            )
            spawn_y_by_id = {
                asset.object_id: spawn_slots[index]
                for index, asset in enumerate(assets)
            }
        return _ResolvedTask(
            manifest=manifest,
            assets=assets,
            targets=targets,
            spawn_x_by_id={
                asset.object_id: OBJECT_LANE_X_M for asset in assets
            },
            spawn_y_by_id=spawn_y_by_id,
            service_gated_spawn=len(targets) > 1,
        )

    def _plan_mobile_carry_goal(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> tuple[float, tuple[float, float]]:
        if self.options.scene_id is not SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return super()._plan_mobile_carry_goal(resolved, root_pose)
        contract = self._destination_contract(resolved)
        goal_xy = contract.get("delivery_root_goal_xy_m")
        goal_yaw = contract.get("delivery_goal_yaw_rad")
        if (
            not isinstance(goal_xy, (list, tuple))
            or len(goal_xy) != 2
            or not isinstance(goal_yaw, (int, float))
        ):
            raise RuntimeError("remote destination lacks navigation contract")
        return float(goal_yaw), (float(goal_xy[0]), float(goal_xy[1]))

    def _destination_contract(
        self, resolved: _ResolvedTask
    ) -> Mapping[str, Any]:
        suite = resolved.manifest.metadata["benchmark_suite"]
        contracts = suite["destination_zone_contracts"]
        contract = contracts[resolved.target_zone.zone_id]
        if not isinstance(contract, Mapping):
            raise RuntimeError("destination contract must be a mapping")
        return contract

    def _mobile_post_turn_stage(self, resolved: _ResolvedTask) -> str:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return "navigate"
        return super()._mobile_post_turn_stage(resolved)

    def _mobile_continue_carry_before_place(
        self, resolved: _ResolvedTask, oracle_phase: str
    ) -> bool:
        del resolved
        return (
            self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2
            and self._mobile_carry_stage != "place"
            and oracle_phase
            in {
                "preplace",
                "place_descend",
                "open",
                "retreat",
                "verify_place",
            }
        )

    def _mobile_place_descend_step_m(self, resolved: _ResolvedTask) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            # The loaded arm has roughly 3 mm of vertical static deflection;
            # an 8 mm incremental target retains IK margin while overcoming
            # that sag for the taller remote parts.
            return 0.008
        return super()._mobile_place_descend_step_m(resolved)

    def _object_crossed_task_exit(
        self,
        resolved: _ResolvedTask,
        asset: Any,
        position_world: Sequence[float],
        *,
        active: bool,
    ) -> bool:
        if (
            self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2
            and asset.object_id == resolved.target_asset.object_id
            and self._ever_held_target
        ):
            return False
        return super()._object_crossed_task_exit(
            resolved,
            asset,
            position_world,
            active=active,
        )

    def _mobile_turn_forward_speed_mps(
        self, resolved: _ResolvedTask
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return 0.0
        return super()._mobile_turn_forward_speed_mps(resolved)

    def _mobile_navigation_yaw_error(self, root_pose: Pose) -> float:
        if self.options.scene_id is not SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return super()._mobile_navigation_yaw_error(root_pose)
        current_yaw = math.atan2(
            2.0
            * (
                root_pose.wxyz[0] * root_pose.wxyz[3]
                + root_pose.wxyz[1] * root_pose.wxyz[2]
            ),
            1.0
            - 2.0
            * (
                root_pose.wxyz[2] * root_pose.wxyz[2]
                + root_pose.wxyz[3] * root_pose.wxyz[3]
            ),
        )
        goal_yaw = self._mobile_goal_yaw_rad
        assert goal_yaw is not None
        return math.atan2(
            math.sin(goal_yaw - current_yaw),
            math.cos(goal_yaw - current_yaw),
        )

    def _mobile_navigate_forward_speed_mps(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            along_track, _ = self._remote_navigation_errors(root_pose)
            if abs(along_track) <= 0.02:
                return 0.0
            return math.copysign(0.30, along_track)
        return super()._mobile_navigate_forward_speed_mps(
            resolved, root_pose
        )

    def _mobile_navigate_lateral_speed_mps(
        self, resolved: _ResolvedTask, root_pose: Pose
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            _, cross_track = self._remote_navigation_errors(root_pose)
            return max(-0.20, min(0.20, 1.5 * cross_track))
        return super()._mobile_navigate_lateral_speed_mps(
            resolved, root_pose
        )

    def _mobile_navigation_drive_heading_tolerance_rad(
        self, resolved: _ResolvedTask
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return 0.18
        return super()._mobile_navigation_drive_heading_tolerance_rad(
            resolved
        )

    def _mobile_navigation_yaw_command(
        self,
        resolved: _ResolvedTask,
        yaw_error_rad: float,
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return max(-0.35, min(0.35, 1.5 * yaw_error_rad))
        return super()._mobile_navigation_yaw_command(
            resolved, yaw_error_rad
        )

    def _remote_navigation_errors(
        self, root_pose: Pose
    ) -> tuple[float, float]:
        """Project world goal error into the frozen final-heading frame."""

        assert self._mobile_goal_root_xy is not None
        assert self._mobile_goal_yaw_rad is not None
        delta_x = self._mobile_goal_root_xy[0] - root_pose.xyz[0]
        delta_y = self._mobile_goal_root_xy[1] - root_pose.xyz[1]
        yaw = self._mobile_goal_yaw_rad
        return (
            delta_x * math.cos(yaw) + delta_y * math.sin(yaw),
            delta_x * -math.sin(yaw) + delta_y * math.cos(yaw),
        )

    def _mobile_navigation_yaw_command(
        self,
        resolved: _ResolvedTask,
        yaw_error_rad: float,
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            # The loaded policy has a yaw deadband near 0.15 rad/s.  Use the
            # same audited 0.35 rad/s envelope as the in-place carry turn so
            # cross-track recovery works in both signed delivery corridors.
            return max(-0.35, min(0.35, 1.5 * yaw_error_rad))
        return super()._mobile_navigation_yaw_command(
            resolved, yaw_error_rad
        )

    def _mobile_navigation_position_tolerance_m(
        self, resolved: _ResolvedTask
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            # The X5 can reach the tray across this residual corridor error.
            # With the calibrated x=-0.16 route, a 0.09 m goal ball retains
            # margin above the mandatory 0.65 m loaded displacement.
            return 0.09
        return super()._mobile_navigation_position_tolerance_m(resolved)

    def _mobile_turn_angular_speed_tolerance_radps(
        self, resolved: _ResolvedTask
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            # A loaded zero-command stance exhibits a bounded ~0.20 rad/s
            # yaw-rate ripple while translating less than 7 mm/s.  Accept the
            # measured balance ripple once heading is inside the 0.08 rad gate.
            return 0.21
        return super()._mobile_turn_angular_speed_tolerance_radps(resolved)

    def _mobile_settle_angular_speed_tolerance_radps(
        self, resolved: _ResolvedTask
    ) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return 0.21
        return super()._mobile_settle_angular_speed_tolerance_radps(resolved)

    def _mobile_settle_position_tolerance_m(
        self, resolved: _ResolvedTask
    ) -> float | None:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return 0.12
        return super()._mobile_settle_position_tolerance_m(resolved)

    def _mobile_carry_stage_timeout_s(self, stage: str) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return {
                "retract": 6.0,
                # The negative-yaw delivery reaches its final heading near
                # 8 s but needs roughly one extra second for angular settling.
                "turn": 10.0,
                "navigate": 14.0,
                "settle": 5.0,
                "place": 20.0,
            }[stage]
        return super()._mobile_carry_stage_timeout_s(stage)

    def _oracle_phase_timeout_s(self, resolved: _ResolvedTask) -> float:
        if self.options.scene_id is SceneId.MOBILE_REMOTE_DELIVERY_V2:
            return 50.0
        return super()._oracle_phase_timeout_s(resolved)


def run_collection_v2(options: RuntimeOptionsV2) -> dict[str, Any]:
    """Run V2 collection and always release the singleton simulator owner."""

    runtime = ConveyorRuntimeV2(options)
    try:
        return runtime.run()
    finally:
        runtime.close()


__all__ = [
    "ConveyorRuntimeV2",
    "RuntimeOptionsV2",
    "V2_ASSET_LOCK_PATH",
    "run_collection_v2",
]
