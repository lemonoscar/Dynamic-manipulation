#!/usr/bin/env python3
"""Lift a prompted SAM2 coke mask into one synchronized LiDAR scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEAD_CAMERA_OFFSET_XYZ_M = np.asarray((0.28, 0.0, 0.07), dtype=np.float64)
HEAD_CAMERA_OFFSET_ROS_WXYZ = np.asarray((0.5, -0.5, 0.5, -0.5), dtype=np.float64)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "artifacts/models/base/sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt"
)
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "artifacts/sources/checkouts/sam2"
DEFAULT_IOPATH_ROOT = PROJECT_ROOT / "artifacts/sources/checkouts/iopath"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    return None if not len(x) else [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _overlay(
    image: np.ndarray,
    mask: np.ndarray,
    pixels_xy: np.ndarray,
    projected_valid: np.ndarray,
    selected: np.ndarray,
    object_ids: np.ndarray,
) -> Image.Image:
    canvas = image.copy()
    tint = np.asarray([20, 220, 255], dtype=np.float32)
    canvas[mask] = np.clip(0.50 * canvas[mask] + 0.50 * tint, 0, 255).astype(np.uint8)
    rendered = Image.fromarray(canvas)
    draw = ImageDraw.Draw(rendered)
    for index in np.flatnonzero(projected_valid):
        x, y = pixels_xy[index]
        if selected[index] and object_ids[index] == 4:
            color, radius = (45, 255, 85), 4
        elif selected[index]:
            color, radius = (255, 55, 55), 4
        elif object_ids[index] == 4:
            color, radius = (255, 210, 20), 3
        else:
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--iopath-root", type=Path, default=DEFAULT_IOPATH_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--point", type=int, nargs=2, default=(321, 278))
    parser.add_argument("--box", type=int, nargs=4, default=(310, 258, 334, 300))
    parser.add_argument("--depth-tolerance-m", type=float, default=0.05)
    args = parser.parse_args()

    evidence_root = args.evidence_root.expanduser().resolve(strict=True)
    capture_dir = evidence_root / "raw" / "sync_capture"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    camera = json.loads((capture_dir / "camera.json").read_text(encoding="utf-8"))
    image = np.asarray(Image.open(capture_dir / "head_rgb.png").convert("RGB")).copy()
    with np.load(capture_dir / "head_depth.npz") as payload:
        depth = np.asarray(payload["depth"], dtype=np.float64).squeeze()
    raw_path = (capture_dir / camera["lidar_raw_path"]).resolve(strict=True)
    audit_path = (capture_dir / camera["lidar_object_id_audit_path"]).resolve(strict=True)
    if not raw_path.is_relative_to(evidence_root) or not audit_path.is_relative_to(evidence_root):
        parser.error("capture paths escape evidence root")
    with np.load(raw_path) as payload:
        xyz_sensor = np.asarray(payload["xyz"], dtype=np.float64)
    with np.load(audit_path) as payload:
        object_ids = np.asarray(payload["object_id"], dtype=np.uint32)

    tf_rows = [
        json.loads(line)
        for line in (evidence_root / "raw" / "tf.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scan_index = int(camera["scan_index"])
    tf = next(row for row in tf_rows if int(row["scan_index"]) == scan_index)
    sensor_to_world = np.asarray(tf["matrix_row_major"], dtype=np.float64).reshape(4, 4)
    xyz_world = xyz_sensor @ sensor_to_world[:3, :3].T + sensor_to_world[:3, 3]

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    sam2_root = args.sam2_root.expanduser().resolve(strict=True)
    iopath_root = args.iopath_root.expanduser().resolve(strict=True)
    sys.path[:0] = [str(sam2_root), str(iopath_root)]
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device=args.device
    )
    predictor = SAM2ImagePredictor(model)
    inference_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with inference_context:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=np.asarray([args.point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            box=np.asarray(args.box, dtype=np.float32),
            multimask_output=False,
        )
    mask = masks[0].astype(bool)

    if "robot_root_orientation_world_wxyz" in camera:
        root_rotation = _quaternion_wxyz_to_matrix(
            np.asarray(camera["robot_root_orientation_world_wxyz"], dtype=np.float64)
        )
        camera_position = np.asarray(
            camera["robot_root_position_world_xyz_m"], dtype=np.float64
        ) + root_rotation @ HEAD_CAMERA_OFFSET_XYZ_M
        camera_rotation = root_rotation @ _quaternion_wxyz_to_matrix(
            HEAD_CAMERA_OFFSET_ROS_WXYZ
        )
        camera_extrinsic_source = "robot root pose composed with fixed head-camera mount"
    else:
        camera_position = np.asarray(
            camera["camera_position_world_xyz_m"], dtype=np.float64
        )
        camera_rotation = _quaternion_wxyz_to_matrix(
            np.asarray(
                camera["camera_orientation_world_from_ros_wxyz"], dtype=np.float64
            )
        )
        camera_extrinsic_source = "CameraData direct pose"
    xyz_camera = (xyz_world - camera_position) @ camera_rotation
    intrinsics = np.asarray(camera["intrinsics_row_major"], dtype=np.float64).reshape(3, 3)
    z = xyz_camera[:, 2]
    projected = xyz_camera @ intrinsics.T
    uv = np.full((len(xyz_camera), 2), np.nan, dtype=np.float64)
    forward = z > 0.0
    uv[forward] = projected[forward, :2] / projected[forward, 2:3]
    pixels_xy = np.zeros((len(xyz_camera), 2), dtype=np.int32)
    finite_uv = np.isfinite(uv).all(axis=1)
    pixels_xy[finite_uv] = np.rint(uv[finite_uv]).astype(np.int32)
    height, width = image.shape[:2]
    in_image = (
        finite_uv
        & forward
        & (pixels_xy[:, 0] >= 0)
        & (pixels_xy[:, 0] < width)
        & (pixels_xy[:, 1] >= 0)
        & (pixels_xy[:, 1] < height)
    )
    sampled_depth = np.full(len(xyz_sensor), np.nan, dtype=np.float64)
    sampled_depth[in_image] = depth[pixels_xy[in_image, 1], pixels_xy[in_image, 0]]
    depth_consistent = (
        in_image
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (np.abs(z - sampled_depth) <= args.depth_tolerance_m)
    )
    inside_mask = np.zeros(len(xyz_sensor), dtype=bool)
    inside_mask[in_image] = mask[pixels_xy[in_image, 1], pixels_xy[in_image, 0]]
    selected = inside_mask & depth_consistent
    coke = object_ids == 4
    visible_coke = coke & depth_consistent
    tp = int(np.count_nonzero(selected & coke))
    fp = int(np.count_nonzero(selected & ~coke))
    fn_visible = int(np.count_nonzero(visible_coke & ~selected))

    np.savez_compressed(
        output_dir / f"scan_{scan_index:06d}_sam3d.npz",
        selected=selected,
        xyz_sensor=xyz_sensor.astype(np.float32),
        xyz_world=xyz_world.astype(np.float32),
        projected_pixel_xy=pixels_xy,
        depth_consistent=depth_consistent,
    )
    Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / "sam_mask.png")
    overlay = _overlay(image, mask, pixels_xy, depth_consistent, selected, object_ids)
    overlay.save(output_dir / "projection_overlay.png")
    selected_cloud = xyz_sensor[selected]
    metrics = {
        "method": "prompted SAM2 RGB mask lifted to synchronized LiDAR by calibrated projection",
        "camera_extrinsic_source": camera_extrinsic_source,
        "camera_position_world_xyz_m": camera_position.tolist(),
        "scan_index": scan_index,
        "sim_time_s": float(camera["sim_time_s"]),
        "prompt_point_xy": list(args.point),
        "prompt_box_xyxy": list(args.box),
        "sam_predicted_iou": float(scores[0]),
        "sam_mask_area_px": int(mask.sum()),
        "sam_mask_bounds_xyxy": _mask_bounds(mask),
        "depth_tolerance_m": args.depth_tolerance_m,
        "raw_lidar_point_count": int(len(xyz_sensor)),
        "projected_depth_consistent_point_count": int(depth_consistent.sum()),
        "sam_selected_3d_point_count": int(selected.sum()),
        "audit_coke_point_count_raw_scan": int(coke.sum()),
        "audit_coke_point_count_camera_depth_visible": int(visible_coke.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative_camera_visible": fn_visible,
        "precision": _safe_ratio(tp, tp + fp),
        "recall_camera_visible": _safe_ratio(tp, tp + fn_visible),
        "recall_all_raw_coke_returns": _safe_ratio(tp, int(coke.sum())),
        "selected_xyz_sensor_bounds_m": (
            None
            if not len(selected_cloud)
            else {
                "min": selected_cloud.min(axis=0).tolist(),
                "max": selected_cloud.max(axis=0).tolist(),
            }
        ),
        "checkpoint_sha256": _sha256(checkpoint),
        "audit_note": "simulator object IDs were used only after inference for metrics",
        "limitation": "prompted instance segmentation, not automatic coke detection and not a native 3D foundation model",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
