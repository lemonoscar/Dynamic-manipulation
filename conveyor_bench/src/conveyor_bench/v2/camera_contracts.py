"""Frozen, simulator-independent camera contracts for ConveyorBench V2."""

from __future__ import annotations

from typing import Any

from .config import SceneId


_HORIZONTAL_APERTURE_MM = 20.955


def _pinhole_contract(
    *,
    width: int,
    height: int,
    focal_length_mm: float,
    role: str,
    parent: str,
    xyz_m: tuple[float, float, float],
    wxyz: tuple[float, float, float, float],
    orientation_convention: str,
) -> dict[str, Any]:
    focal_px = focal_length_mm / _HORIZONTAL_APERTURE_MM * width
    return {
        "resolution": [width, height],
        "fps": 25,
        "role": role,
        "model": "pinhole",
        "intrinsics": [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        "mount": {
            "parent": parent,
            "xyz_m": list(xyz_m),
            "wxyz": list(wxyz),
            "orientation_convention": orientation_convention,
        },
    }


def camera_contract_for_scene(scene_id: SceneId | str) -> dict[str, Any]:
    """Return a fresh copy of the exact three-camera contract for one scene."""

    scene = scene_id if isinstance(scene_id, SceneId) else SceneId(scene_id)
    overview_focal_length_mm = 18.0
    overview_xyz_m = (-2.10, -1.60, 2.40)
    overview_wxyz = (
        0.92554193,
        -0.07441319,
        0.26721596,
        0.25774106,
    )
    if scene is SceneId.MOBILE_REMOTE_DELIVERY_V2:
        overview_focal_length_mm = 16.0
        overview_xyz_m = (-2.80, -2.60, 3.20)
        overview_wxyz = (
            0.89625224,
            -0.10238193,
            0.27791988,
            0.33016722,
        )

    return {
        "head_rgb": _pinhole_contract(
            width=224,
            height=224,
            focal_length_mm=24.0,
            role="policy_observation",
            parent="base",
            xyz_m=(0.28, 0.0, 0.07),
            wxyz=(0.5, -0.5, 0.5, -0.5),
            orientation_convention="ros",
        ),
        "wrist_rgb": _pinhole_contract(
            width=224,
            height=224,
            focal_length_mm=18.0,
            role="policy_observation",
            parent="arm_link6",
            xyz_m=(0.0, 0.0, 0.10),
            wxyz=(0.353553, -0.612372, 0.612372, -0.353553),
            orientation_convention="ros",
        ),
        "overview_rgb": _pinhole_contract(
            width=480,
            height=320,
            focal_length_mm=overview_focal_length_mm,
            role="observer_only",
            parent="environment_origin",
            xyz_m=overview_xyz_m,
            wxyz=overview_wxyz,
            orientation_convention="world",
        ),
    }


__all__ = ["camera_contract_for_scene"]
