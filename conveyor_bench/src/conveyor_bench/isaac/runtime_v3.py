"""ConveyorBench V3 collector using native NuRec registered compositing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conveyor_bench.v1.assets import sha256_file
from conveyor_bench.v3.assets import (
    LIANGZHU_SCENE_TRANSLATION_XYZ_M,
    V3AssetBundle,
    resolve_asset_root,
    validate_asset_bundle,
)
from conveyor_bench.v3.objects import (
    V3_OBJECT_ASSETS,
    V3_OBJECT_SPLITS,
    V3_STATIONARY_TARGET_ASSET_ID,
)

from .runtime_v1 import ConveyorRuntimeV1, RuntimeOptionsV1
from .scene_v3 import (
    SCENE_ID,
    make_conveyor_scene_v3_cfg,
    place_workcell_in_liangzhu_task_area,
    validate_liangzhu_stage,
    validate_v3_object_fixtures,
)


@dataclass(frozen=True)
class RuntimeOptionsV3(RuntimeOptionsV1):
    """V1 collection options plus one immutable SSH asset bundle."""

    asset_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_assets", V3_OBJECT_ASSETS)
        object.__setattr__(self, "object_split_ids", V3_OBJECT_SPLITS)
        object.__setattr__(
            self,
            "stationary_target_asset_id",
            V3_STATIONARY_TARGET_ASSET_ID,
        )
        super().__post_init__()
        object.__setattr__(self, "asset_root", resolve_asset_root(self.asset_root))


class ConveyorRuntimeV3(ConveyorRuntimeV1):
    """Keep V1 control/data semantics while replacing the static backdrop."""

    options: RuntimeOptionsV3

    def __init__(self, options: RuntimeOptionsV3):
        assert options.asset_root is not None
        self.asset_bundle: V3AssetBundle = validate_asset_bundle(
            options.asset_root,
            verify_all_hashes=True,
        )
        runtime_directory = (
            Path(options.output_root).expanduser().resolve() / "_v3_runtime"
        )
        self.runtime_layer = self.asset_bundle.write_runtime_layer(
            runtime_directory / "liangzhu_conveyorvla_v3.usda"
        )
        super().__init__(options)

    def _make_scene_cfg(self):
        return make_conveyor_scene_v3_cfg(
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
        self.object_fixture_contract = validate_v3_object_fixtures(
            stage,
            self.object_assets,
            {
                asset.object_id: self.asset_bundle.object_usd(asset.object_id)
                for asset in self.object_assets
            },
        )

    def _layout_id(self) -> str:
        return SCENE_ID

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
            "object_fixture_contract": self.object_fixture_contract,
        }


def run_collection_v3(options: RuntimeOptionsV3) -> dict[str, Any]:
    runtime = ConveyorRuntimeV3(options)
    try:
        return runtime.run()
    finally:
        runtime.close()


__all__ = [
    "ConveyorRuntimeV3",
    "RuntimeOptionsV3",
    "run_collection_v3",
]
