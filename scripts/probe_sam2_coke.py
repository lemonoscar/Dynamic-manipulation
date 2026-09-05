#!/usr/bin/env python3
"""Run a single-frame, prompted SAM2 coke-can segmentation probe."""

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
DEFAULT_SOURCE = (
    PROJECT_ROOT / "artifacts/runs/liangzhu-sam2-coke-poc-20260830/head_rgb_01s.png"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/runs/liangzhu-sam2-coke-poc-20260830"
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


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    y, x = np.nonzero(mask)
    if not len(x):
        return None
    return [int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def _panel(
    image: np.ndarray,
    mask: np.ndarray | None,
    title: str,
    *,
    box: tuple[int, int, int, int],
    point: tuple[int, int],
) -> Image.Image:
    canvas = image.copy()
    if mask is not None:
        color = np.asarray([20, 220, 255], dtype=np.float32)
        canvas[mask] = np.clip(0.42 * canvas[mask] + 0.58 * color, 0, 255).astype(np.uint8)
    rendered = Image.fromarray(canvas)
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(box, outline=(255, 205, 25), width=2)
    radius = 4
    draw.ellipse(
        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
        fill=(255, 35, 35),
        outline=(255, 255, 255),
    )
    titled = Image.new("RGB", (rendered.width, rendered.height + 28), (15, 18, 22))
    titled.paste(rendered, (0, 28))
    ImageDraw.Draw(titled).text((8, 7), title, fill=(240, 244, 248))
    return titled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--iopath-root", type=Path, default=DEFAULT_IOPATH_ROOT)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--point", type=int, nargs=2, default=(321, 278), metavar=("X", "Y"))
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=(310, 258, 334, 300),
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    sam2_root = args.sam2_root.expanduser().resolve(strict=True)
    iopath_root = args.iopath_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(sam2_root), str(iopath_root)]

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    image = np.asarray(Image.open(source).convert("RGB")).copy()
    point = tuple(args.point)
    box = tuple(args.box)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        parser.error("CUDA was requested but is unavailable")
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint),
        device=args.device,
    )
    predictor = SAM2ImagePredictor(model)
    inference_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.device.startswith("cuda")
        else nullcontext()
    )
    with inference_context:
        predictor.set_image(image)
        point_masks, point_scores, _ = predictor.predict(
            point_coords=np.asarray([point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=True,
        )
        combined_masks, combined_scores, _ = predictor.predict(
            point_coords=np.asarray([point], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            box=np.asarray(box, dtype=np.float32),
            multimask_output=False,
        )
    point_choice = int(np.argmax(point_scores))
    point_mask = point_masks[point_choice].astype(bool)
    combined_mask = combined_masks[0].astype(bool)
    Image.fromarray(point_mask.astype(np.uint8) * 255).save(output_dir / "point_mask.png")
    Image.fromarray(combined_mask.astype(np.uint8) * 255).save(
        output_dir / "box_point_mask.png"
    )
    panels = [
        _panel(image, None, "INPUT + MANUAL PROMPTS", box=box, point=point),
        _panel(
            image,
            point_mask,
            f"POINT ONLY | score={float(point_scores[point_choice]):.3f}",
            box=box,
            point=point,
        ),
        _panel(
            image,
            combined_mask,
            f"BOX + POINT | score={float(combined_scores[0]):.3f}",
            box=box,
            point=point,
        ),
    ]
    comparison = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    x = 0
    for panel in panels:
        comparison.paste(panel, (x, 0))
        x += panel.width
    comparison.save(output_dir / "comparison.png")
    margin = 28
    zoom_bounds = (
        max(0, box[0] - margin),
        max(0, box[1] + 28 - margin),
        min(panels[0].width, box[2] + margin),
        min(panels[0].height, box[3] + 28 + margin),
    )
    zoom_panels = [
        panel.crop(zoom_bounds).resize(
            ((zoom_bounds[2] - zoom_bounds[0]) * 4, (zoom_bounds[3] - zoom_bounds[1]) * 4)
        )
        for panel in panels
    ]
    zoom_comparison = Image.new(
        "RGB", (sum(panel.width for panel in zoom_panels), zoom_panels[0].height)
    )
    x = 0
    for panel in zoom_panels:
        zoom_comparison.paste(panel, (x, 0))
        x += panel.width
    zoom_comparison.save(output_dir / "zoom_comparison.png")
    metrics = {
        "source": str(source),
        "source_kind": "head RGB crop from rendered three-panel diagnostic",
        "prompt_point_xy": point,
        "prompt_box_xyxy": box,
        "point_only": {
            "selected_candidate": point_choice,
            "predicted_iou": float(point_scores[point_choice]),
            "area_px": int(point_mask.sum()),
            "bounds_xyxy": _mask_bounds(point_mask),
        },
        "box_plus_point": {
            "predicted_iou": float(combined_scores[0]),
            "area_px": int(combined_mask.sum()),
            "bounds_xyxy": _mask_bounds(combined_mask),
        },
        "model": "SAM 2.1 Hiera Tiny",
        "checkpoint_sha256": _sha256(checkpoint),
        "device": args.device,
        "limitation": "RGB-only feasibility probe; no camera-to-LiDAR projection performed",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
