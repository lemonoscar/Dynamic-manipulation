"""Frozen, simulator-independent camera contracts for ConveyorBench V2."""

from __future__ import annotations

from typing import Any

from .config import SceneId


_HORIZONTAL_APERTURE_MM = 20.955
_D436_RESOLUTION = (640, 480)
_D436_INTRINSICS = (
    (383.44608095, 0.0, 324.33479864),
    (0.0, 383.52724198, 238.90275478),
    (0.0, 0.0, 1.0),
)
_D436_DISTORTION = (0.0,) * 12
_D436_FALLBACK_OPTICS = {
    "focal_length_mm": 18.0,
    "horizontal_aperture_mm": 30.040158257372415,
    "vertical_aperture_mm": 22.530118693029312,
}


def _d436_contract(
    *,
    role: str,
    parent: str,
    prim_path: str,
    xyz_m: tuple[float, float, float],
    wxyz: tuple[float, float, float, float],
    calibration_source: str,
    clipping_range_m: tuple[float, float],
) -> dict[str, Any]:
    width, height = _D436_RESOLUTION
    return {
        "resolution": [width, height],
        "fps": 25,
        "role": role,
        "model": "opencv_pinhole",
        "intrinsics": [list(row) for row in _D436_INTRINSICS],
        "distortion_coefficients": list(_D436_DISTORTION),
        "calibration_source": calibration_source,
        "fallback_optics": dict(_D436_FALLBACK_OPTICS),
        "clipping_range_m": list(clipping_range_m),
        "mount": {
            "parent": parent,
            "prim_path": prim_path,
            "xyz_m": list(xyz_m),
            "wxyz": list(wxyz),
            "orientation_convention": "ros",
        },
    }


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

    head = _d436_contract(
        role="policy_observation",
        parent="base",
        prim_path="{ENV_REGEX_NS}/Robot/base/head_cam",
        xyz_m=(0.28, 0.0, 0.07),
        wxyz=(0.5, -0.5, 0.5, -0.5),
        calibration_source="dwa_play_nav_cs",
        clipping_range_m=(0.1, 1.0e5),
    )
    wrist = _d436_contract(
        role="policy_observation",
        parent="arm_link6",
        prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera",
        xyz_m=(0.0666580792, 0.0028071889, 0.0935779972),
        wxyz=(
            0.3377891849,
            -0.6214992221,
            0.6185057335,
            -0.3421810063,
        ),
        calibration_source="hand_eye_calibration_with_visual_alignment_v3",
        clipping_range_m=(0.03, 5.0),
    )
    wrist["calibration_frame"] = "arm_link6_T_camera_color_optical"
    wrist["raw_hand_eye_position_xyz_m"] = [
        0.0559054476,
        0.0026732239,
        0.0767149320,
    ]
    wrist["visual_alignment_offset_camera_xyz_m"] = [0.0, -0.02, 0.0]

    return {
        "head_rgb": head,
        "wrist_rgb": wrist,
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
