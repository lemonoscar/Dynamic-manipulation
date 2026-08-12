"""Current collector using native NuRec registered compositing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conveyor_bench.schema.assets import sha256_file
from conveyor_bench.sidecar.assets import (
    AssetBundle,
    LIANGZHU_SCENE_TRANSLATION_XYZ_M,
    resolve_asset_root,
    validate_asset_bundle,
)
from conveyor_bench.sidecar.objects import (
    OBJECT_ASSETS,
    OBJECT_SPLITS,
    STATIONARY_TARGET_ASSET_ID,
)

from .runtime_core import _ConveyorRuntimeCore, _RuntimeOptions
from .scene import (
    SCENE_ID,
    TASK_AREA_GROUND_XYZ_M,
    disable_liangzhu_background_collision,
    make_conveyor_scene_cfg,
    place_workcell_in_liangzhu_task_area,
    validate_liangzhu_stage,
    validate_object_fixtures,
)


@dataclass(frozen=True)
class RuntimeOptions(_RuntimeOptions):
    """Collection options plus one immutable SSH asset bundle."""

    asset_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_assets", OBJECT_ASSETS)
        object.__setattr__(self, "object_split_ids", OBJECT_SPLITS)
        object.__setattr__(
            self,
            "stationary_target_asset_id",
            STATIONARY_TARGET_ASSET_ID,
        )
        super().__post_init__()
        object.__setattr__(self, "asset_root", resolve_asset_root(self.asset_root))


class ConveyorRuntime(_ConveyorRuntimeCore):
    """Run the canonical task inside the registered NuRec scene."""

    options: RuntimeOptions

    def __init__(self, options: RuntimeOptions):
        assert options.asset_root is not None
        print("verifying full sidecar asset hashes", flush=True)
        self.asset_bundle: AssetBundle = validate_asset_bundle(
            options.asset_root,
            verify_all_hashes=True,
        )
        runtime_directory = (
            Path(options.output_root).expanduser().resolve() / "_runtime"
        )
        self.runtime_layer = self.asset_bundle.write_runtime_layer(
            runtime_directory / "liangzhu_conveyorvla.usda"
        )
        print(f"runtime layer ready: {self.runtime_layer}", flush=True)
        super().__init__(options)

    def _make_scene_cfg(self):
        return make_conveyor_scene_cfg(
            self.runtime_layer,
            object_assets=self.object_assets,
            object_usd_paths={
                asset.object_id: self.asset_bundle.object_usd(asset.object_id)
                for asset in self.object_assets
            },
        )

    def _post_scene_creation(self, stage: Any) -> None:
        self.workcell_placement = place_workcell_in_liangzhu_task_area(
            self.scene, stage
        )
        self.scene_stage_contract = validate_liangzhu_stage(stage)
        self.background_collision_contract = (
            disable_liangzhu_background_collision(
                stage,
                self.scene_stage_contract["collision_mesh_prims"],
            )
        )
        self.object_fixture_contract = validate_object_fixtures(
            stage,
            self.object_assets,
            {
                asset.object_id: self.asset_bundle.object_usd(asset.object_id)
                for asset in self.object_assets
            },
        )

    def _layout_id(self) -> str:
        return SCENE_ID

    def _task_world_origin_xyz(self) -> tuple[float, float, float]:
        return TASK_AREA_GROUND_XYZ_M

    def _episode_asset_hashes(self, resolved) -> dict[str, str]:
        return {
            asset.object_id: sha256_file(
                self.asset_bundle.object_usd(asset.object_id)
            )
            for asset in resolved.assets
        }

    def _scene_metadata(self) -> dict[str, Any]:
        return {
            "backend": "isaac_rtx_native_nurec",
            "asset_delivery": "ssh_sidecar_bundle",
            "asset_bundle": self.asset_bundle.report.to_dict(),
            "runtime_layer": str(self.runtime_layer),
            "scene_translation_xyz_m": list(
                LIANGZHU_SCENE_TRANSLATION_XYZ_M
            ),
            "workcell_placement": self.workcell_placement,
            "stage_contract": self.scene_stage_contract,
            "background_collision_contract": (
                self.background_collision_contract
            ),
            "object_fixture_contract": self.object_fixture_contract,
        }


def run_collection(options: RuntimeOptions) -> dict[str, Any]:
    runtime = ConveyorRuntime(options)
    try:
        return runtime.run()
    finally:
        runtime.close()


__all__ = [
    "ConveyorRuntime",
    "RuntimeOptions",
    "run_collection",
]
