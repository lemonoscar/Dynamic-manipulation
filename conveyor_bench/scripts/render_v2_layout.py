#!/usr/bin/env python3
"""Render a deterministic, dependency-free top view of ConveyorBench V2."""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets"
V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "v1.json"
V2_CONFIG_PATH = PROJECT_ROOT / "configs" / "v2.json"
OBJECT_REGISTRY_PATH = ASSET_ROOT / "objects" / "registry.json"
NEAR_RECEPTACLE_PATH = ASSET_ROOT / "receptacles" / "ASSET_MANIFEST.json"
REMOTE_RECEPTACLE_PATH = (
    ASSET_ROOT / "receptacles" / "remote_delivery_v2.json"
)
V1_WORKCELL_PATH = (
    ASSET_ROOT / "workcells" / "conveyor_station_v1" / "ASSET_MANIFEST.json"
)
REMOTE_WORKCELL_PATH = (
    ASSET_ROOT
    / "workcells"
    / "remote_delivery_v2"
    / "ASSET_MANIFEST.json"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "docs" / "images" / "conveyorbench_v2_layout.svg"
)

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 820
ORIGIN_X = 600.0
ORIGIN_Y = 690.0
SCALE_Y_TO_SCREEN_X = 340.0
SCALE_X_TO_SCREEN_Y = 390.0


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        return _load_json(path)
    except FileNotFoundError:
        return None


def _screen(world_x: float, world_y: float) -> tuple[float, float]:
    """Map world +Y to screen-left so belt -Y motion reads left-to-right."""

    return (
        ORIGIN_X - world_y * SCALE_Y_TO_SCREEN_X,
        ORIGIN_Y - world_x * SCALE_X_TO_SCREEN_Y,
    )


def _world_rect(
    center_xyz: list[float], size_xyz: list[float]
) -> tuple[float, float, float, float]:
    center_x, center_y = float(center_xyz[0]), float(center_xyz[1])
    size_x, size_y = float(size_xyz[0]), float(size_xyz[1])
    center_screen_x, center_screen_y = _screen(center_x, center_y)
    width = size_y * SCALE_Y_TO_SCREEN_X
    height = size_x * SCALE_X_TO_SCREEN_Y
    return (
        center_screen_x - width / 2.0,
        center_screen_y - height / 2.0,
        width,
        height,
    )


def _zone_rect(zone: dict[str, Any]) -> tuple[float, float, float, float]:
    half = zone["goal_half_extents_xyz_m"]
    size = [2.0 * float(half[0]), 2.0 * float(half[1]), 0.0]
    return _world_rect(zone["center_xyz_m"], size)


def _fmt(value: float) -> str:
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _rgb_hex(values: list[float]) -> str:
    channels = [max(0, min(255, round(float(value) * 255))) for value in values]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def _attrs(**values: object) -> str:
    rendered: list[str] = []
    for name, value in values.items():
        xml_name = name.rstrip("_").replace("_", "-")
        if isinstance(value, float):
            value = _fmt(value)
        rendered.append(f'{xml_name}="{escape(str(value), quote=True)}"')
    return " ".join(rendered)


def _validate_config_zones(
    config: dict[str, Any] | None,
    scene_id: str,
    manifest_zones: list[dict[str, Any]],
) -> None:
    """Reject a completed config that disagrees with manifest geometry."""

    if config is None:
        return
    scene = config.get("scenes", {}).get(scene_id)
    if not isinstance(scene, dict) or "goal_zones" not in scene:
        return
    configured = {
        zone["zone_id"]: zone
        for zone in scene["goal_zones"]
        if isinstance(zone, dict) and "zone_id" in zone
    }
    for manifest_zone in manifest_zones:
        zone_id = manifest_zone["zone_id"]
        config_zone = configured.get(zone_id)
        if config_zone is None:
            continue
        for field in ("center_xyz_m", "goal_half_extents_xyz_m"):
            if field not in config_zone:
                continue
            manifest_values = [float(value) for value in manifest_zone[field]]
            config_values = [float(value) for value in config_zone[field]]
            if manifest_values != config_values:
                raise ValueError(
                    f"V2 config disagrees with {zone_id}.{field} manifest geometry"
                )


def _zone_svg(zone: dict[str, Any], *, profile: str) -> list[str]:
    x, y, width, height = _zone_rect(zone)
    zone_id = escape(str(zone["zone_id"]), quote=True)
    color = _rgb_hex(zone["color_rgb"])
    profile_class = "near-bin" if profile == "near" else "remote-bin"
    label_y = y + height / 2.0 + 5 if profile == "near" else y + height + 22
    return [
        f'    <g id="{profile_class}-{zone_id}" class="bin {profile_class}">',
        f"      <rect {_attrs(x=x, y=y, width=width, height=height, rx=10.0, fill=color)}/>",
        f"      <rect {_attrs(x=x + 7, y=y + 7, width=width - 14, height=height - 14, rx=6.0, fill='#101923')}/>",
        f"      <text {_attrs(x=x + width / 2, y=label_y, text_anchor='middle')} class=\"bin-label\">{escape(str(zone['zone_id']))}</text>",
        "    </g>",
    ]


def _object_svg(
    part: dict[str, Any], x: float, y: float, index: int
) -> list[str]:
    color_name = str(part["attributes"]["color"])
    color = {
        "silver": "#b8c2cc",
        "yellow": "#f0b014",
        "green": "#31a66a",
        "blue": "#2968d8",
        "red": "#dc4437",
        "orange": "#ed8434",
        "purple": "#9563d8",
        "cyan": "#30c6cf",
    }.get(color_name, color_name)
    geometry_kind = str(part["geometry"]["kind"])
    common = _attrs(
        id=f"part-{index:02d}",
        data_object_id=part["object_id"],
        fill=color,
        stroke="#f8fafc",
        stroke_width=2.0,
    )
    if geometry_kind == "box":
        shape = f"      <rect {common} {_attrs(x=x - 13, y=y - 11, width=26.0, height=22.0, rx=4.0)}/>"
    elif geometry_kind == "cylinder":
        shape = f"      <circle {common} {_attrs(cx=x, cy=y, r=12.0)}/>"
    else:
        points = " ".join(
            f"{_fmt(px)},{_fmt(py)}"
            for px, py in (
                (x, y - 14),
                (x + 13, y - 5),
                (x + 8, y + 13),
                (x - 8, y + 13),
                (x - 13, y - 5),
            )
        )
        shape = f"      <polygon {common} points=\"{points}\"/>"
    return [
        f'    <g class="part" data-geometry-kind="{escape(geometry_kind)}">',
        shape,
        "    </g>",
    ]


def build_svg() -> str:
    """Build the SVG from repository-local JSON sources only."""

    v1_config = _load_json(V1_CONFIG_PATH)
    v2_config = _load_optional_json(V2_CONFIG_PATH)
    object_registry = _load_json(OBJECT_REGISTRY_PATH)
    near_manifest = _load_json(NEAR_RECEPTACLE_PATH)
    remote_manifest = _load_json(REMOTE_RECEPTACLE_PATH)
    v1_workcell = _load_json(V1_WORKCELL_PATH)
    remote_workcell = _load_json(REMOTE_WORKCELL_PATH)

    near_zones = [
        zone
        for zone in near_manifest["receptacles"]
        if str(zone["zone_id"]).startswith("sort_bin_")
    ]
    remote_zones = list(remote_manifest["receptacles"])
    _validate_config_zones(
        v2_config, "transverse_near_sort_v2", near_zones
    )
    _validate_config_zones(
        v2_config, "mobile_remote_delivery_v2", remote_zones
    )

    design = v1_workcell["design"]
    remote_design = remote_workcell["design"]
    expected_receptacle = str(
        REMOTE_RECEPTACLE_PATH.relative_to(PROJECT_ROOT)
    )
    if remote_workcell["receptacle_manifest"] != expected_receptacle:
        raise ValueError("Remote workcell points at an unexpected receptacle manifest")

    belt_center = design["belt_center_xyz_m"]
    belt_size = design["belt_size_xyz_m"]
    transport = design["transport_axis_world"]
    belt_x, belt_y, belt_width, belt_height = _world_rect(
        belt_center, belt_size
    )
    arrow_span = float(belt_size[1]) * 0.32
    arrow_start_world = (
        float(belt_center[0]) - float(transport[0]) * arrow_span,
        float(belt_center[1]) - float(transport[1]) * arrow_span,
    )
    arrow_end_world = (
        float(belt_center[0]) + float(transport[0]) * arrow_span,
        float(belt_center[1]) + float(transport[1]) * arrow_span,
    )
    arrow_start = _screen(*arrow_start_world)
    arrow_end = _screen(*arrow_end_world)
    if arrow_end[0] <= arrow_start[0]:
        raise ValueError("Transport manifest no longer produces left-to-right motion")

    corridor = remote_design["flat_navigation_corridor"]
    corridor_x, corridor_y, corridor_width, corridor_height = _world_rect(
        corridor["center_xyz_m"], corridor["size_xyz_m"]
    )
    root_goals = remote_design["delivery_root_goals"]
    ordered_goals = [root_goals[zone["zone_id"]] for zone in remote_zones]
    robot_world = (
        sum(float(goal[0]) for goal in ordered_goals) / len(ordered_goals),
        sum(float(goal[1]) for goal in ordered_goals) / len(ordered_goals),
    )
    robot_screen = _screen(*robot_world)
    overview_xyz = remote_design["overview_camera"]["position_xyz_m"]
    cameras = v1_config["cameras"]
    camera_roles = {
        name: spec["role"]
        for name, spec in cameras.items()
        if isinstance(spec, dict) and name.endswith("_rgb") and "role" in spec
    }
    required_cameras = {"head_rgb", "wrist_rgb", "overview_rgb"}
    if set(camera_roles) != required_cameras:
        raise ValueError("V1 camera contract must expose exactly three RGB cameras")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" {_attrs(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, viewBox=f"0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}", role="img", aria_labelledby="title desc")}>',
        "  <title id=\"title\">ConveyorBench V2 top-view layout</title>",
        "  <desc id=\"desc\">横向传送带、近端分拣框、远端投放框、机器狗移动路线与三相机示意。</desc>",
        "  <defs>",
        "    <linearGradient id=\"belt-gradient\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">",
        "      <stop offset=\"0\" stop-color=\"#263340\"/>",
        "      <stop offset=\"0.5\" stop-color=\"#101820\"/>",
        "      <stop offset=\"1\" stop-color=\"#263340\"/>",
        "    </linearGradient>",
        "    <marker id=\"arrow-cyan\" markerWidth=\"12\" markerHeight=\"12\" refX=\"10\" refY=\"6\" orient=\"auto\" markerUnits=\"strokeWidth\">",
        "      <path d=\"M 0 0 L 12 6 L 0 12 z\" fill=\"#38d8e8\"/>",
        "    </marker>",
        "    <marker id=\"arrow-green\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"5\" orient=\"auto\" markerUnits=\"strokeWidth\">",
        "      <path d=\"M 0 0 L 10 5 L 0 10 z\" fill=\"#55d68b\"/>",
        "    </marker>",
        "    <filter id=\"soft-shadow\" x=\"-20%\" y=\"-20%\" width=\"140%\" height=\"140%\">",
        "      <feDropShadow dx=\"0\" dy=\"4\" stdDeviation=\"5\" flood-opacity=\"0.28\"/>",
        "    </filter>",
        "  </defs>",
        "  <style>",
        "    text { font-family: Inter, 'Noto Sans SC', sans-serif; fill: #e8eef5; }",
        "    .muted { fill: #9eb0c2; font-size: 14px; }",
        "    .bin-label { fill: #dce7f1; font-size: 13px; font-weight: 650; }",
        "    .route { stroke: #55d68b; stroke-width: 5; stroke-dasharray: 12 10; fill: none; marker-end: url(#arrow-green); }",
        "    .camera-label { font-size: 13px; font-weight: 650; }",
        "  </style>",
        "  <rect width=\"1200\" height=\"820\" fill=\"#09121b\"/>",
        "  <rect x=\"28\" y=\"24\" width=\"1144\" height=\"772\" rx=\"24\" fill=\"#101d29\" stroke=\"#274052\" stroke-width=\"2\"/>",
        "  <text x=\"62\" y=\"72\" fill=\"#e8eef5\" font-size=\"28\" font-weight=\"750\">ConveyorBench V2 · Top-view collection layout</text>",
        "  <text x=\"62\" y=\"101\" class=\"muted\">Near + remote profiles overlaid · near bins are disabled in the remote scene · geometry comes from local manifests.</text>",
        f"  <g id=\"navigation-corridor\"><rect {_attrs(x=corridor_x, y=corridor_y, width=corridor_width, height=corridor_height, rx=28.0, fill='#143329', stroke='#285b49', stroke_width=2.0, stroke_dasharray='9 9')}/></g>",
        "  <g id=\"conveyor-belt\" filter=\"url(#soft-shadow)\">",
        f"    <rect {_attrs(x=belt_x - 18, y=belt_y - 15, width=belt_width + 36, height=belt_height + 30, rx=20.0, fill='#536270')}/>",
        f"    <rect {_attrs(x=belt_x, y=belt_y, width=belt_width, height=belt_height, rx=12.0, fill='url(#belt-gradient)', stroke='#6d7c89', stroke_width=2.0)}/>",
    ]

    seam_count = 9
    for index in range(1, seam_count):
        seam_x = belt_x + belt_width * index / seam_count
        lines.append(
            f"    <line {_attrs(x1=seam_x, y1=belt_y + 8, x2=seam_x, y2=belt_y + belt_height - 8, stroke='#344553', stroke_width=2.0)}/>"
        )
    lines.extend(
        [
            f"    <line id=\"direction-left-to-right\" {_attrs(x1=arrow_start[0], y1=arrow_start[1], x2=arrow_end[0], y2=arrow_end[1], stroke='#38d8e8', stroke_width=7.0, stroke_linecap='round', marker_end='url(#arrow-cyan)')}/>",
            f"    <text {_attrs(x=(arrow_start[0] + arrow_end[0]) / 2, y=belt_y - 24, text_anchor='middle')} fill=\"#e8eef5\" font-size=\"19\" font-weight=\"750\">PART FLOW: LEFT → RIGHT / 左 → 右</text>",
            f"    <text {_attrs(x=(arrow_start[0] + arrow_end[0]) / 2, y=belt_y + belt_height - 12, text_anchor='middle')} class=\"muted\">belt top z = {_fmt(design['belt_top_z_m'])} m · world axis [{_fmt(transport[0])}, {_fmt(transport[1])}, {_fmt(transport[2])}]</text>",
            "  </g>",
            "  <g id=\"representative-parts\">",
        ]
    )
    parts = list(object_registry["objects"])
    part_count = min(4, len(parts))
    for index, part in enumerate(parts[:part_count]):
        fraction = (index + 1) / (part_count + 1)
        part_x = belt_x + fraction * belt_width
        part_y = belt_y + belt_height * (0.43 if index % 2 == 0 else 0.62)
        lines.extend(_object_svg(part, part_x, part_y, index))
    lines.extend(
        [
            f"    <text x=\"817\" y=\"285\" class=\"muted\" text-anchor=\"end\">{len(parts)} local procedural part classes / 本地零件</text>",
            "  </g>",
            "  <g id=\"near-sort-profile\">",
        ]
    )
    for zone in near_zones:
        lines.extend(_zone_svg(zone, profile="near"))
    lines.extend(
        [
            "  </g>",
            "  <g id=\"remote-delivery-profile\">",
        ]
    )
    for zone in remote_zones:
        lines.extend(_zone_svg(zone, profile="remote"))
    for zone, goal in zip(remote_zones, ordered_goals, strict=True):
        goal_screen = _screen(float(goal[0]), float(goal[1]))
        lines.extend(
            [
                f"    <path id=\"route-{escape(str(zone['zone_id']), quote=True)}\" class=\"route\" d=\"M {_fmt(robot_screen[0])} {_fmt(robot_screen[1])} L {_fmt(goal_screen[0])} {_fmt(goal_screen[1])}\"/>",
                f"    <circle {_attrs(cx=goal_screen[0], cy=goal_screen[1], r=12.0, fill='#10261e', stroke='#55d68b', stroke_width=3.0)}/>",
            ]
        )
    lines.extend(
        [
            f"    <text x=\"600\" y=\"782\" text-anchor=\"middle\" class=\"muted\">loaded base route · minimum displacement {_fmt(remote_design['minimum_loaded_base_displacement_m'])} m · standoff {_fmt(remote_design['delivery_standoff_m'])} m</text>",
            "  </g>",
            f"  <g id=\"robot-start\" transform=\"translate({_fmt(robot_screen[0])} {_fmt(robot_screen[1])})\" filter=\"url(#soft-shadow)\">",
            "    <rect x=\"-39\" y=\"-30\" width=\"78\" height=\"60\" rx=\"20\" fill=\"#e4e8eb\" stroke=\"#ffffff\" stroke-width=\"3\"/>",
            "    <rect x=\"-29\" y=\"-42\" width=\"58\" height=\"18\" rx=\"7\" fill=\"#5e7181\"/>",
            "    <circle cx=\"-48\" cy=\"-19\" r=\"8\" fill=\"#8ea1b1\"/>",
            "    <circle cx=\"48\" cy=\"-19\" r=\"8\" fill=\"#8ea1b1\"/>",
            "    <circle cx=\"-48\" cy=\"19\" r=\"8\" fill=\"#8ea1b1\"/>",
            "    <circle cx=\"48\" cy=\"19\" r=\"8\" fill=\"#8ea1b1\"/>",
            "    <path d=\"M 0 -58 L -9 -46 L 9 -46 z\" fill=\"#38d8e8\"/>",
            "    <text x=\"0\" y=\"53\" text-anchor=\"middle\" fill=\"#e8eef5\" font-size=\"14\" font-weight=\"750\">GO2-X5 START / 起点</text>",
            "  </g>",
        ]
    )

    arm_endpoint = _screen(
        float(belt_center[0]) - float(belt_size[0]) / 2.0,
        float(belt_center[1]),
    )
    lines.extend(
        [
            f"  <g id=\"camera-head\" data-role=\"{escape(camera_roles['head_rgb'], quote=True)}\">",
            f"    <line {_attrs(x1=robot_screen[0], y1=robot_screen[1] - 43, x2=robot_screen[0], y2=robot_screen[1] - 92, stroke='#f3c866', stroke_width=3.0, stroke_dasharray='5 5')}/>",
            f"    <path d=\"M {_fmt(robot_screen[0] - 12)} {_fmt(robot_screen[1] - 98)} h 24 l 8 7 h -8 v 13 h -24 z\" fill=\"#f3c866\"/>",
            f"    <text {_attrs(x=robot_screen[0] + 17, y=robot_screen[1] - 82)} class=\"camera-label\" fill=\"#f3c866\">C1</text>",
            "  </g>",
            f"  <g id=\"camera-wrist\" data-role=\"{escape(camera_roles['wrist_rgb'], quote=True)}\">",
            f"    <line {_attrs(x1=robot_screen[0], y1=robot_screen[1] - 31, x2=arm_endpoint[0], y2=arm_endpoint[1], stroke='#bcc8d2', stroke_width=9.0, stroke_linecap='round')}/>",
            f"    <circle {_attrs(cx=arm_endpoint[0], cy=arm_endpoint[1], r=11.0, fill='#f3c866', stroke='#fff3bf', stroke_width=3.0)}/>",
            f"    <path d=\"M {_fmt(arm_endpoint[0] - 10)} {_fmt(arm_endpoint[1] + 18)} L {_fmt(arm_endpoint[0])} {_fmt(arm_endpoint[1] + 34)} L {_fmt(arm_endpoint[0] + 10)} {_fmt(arm_endpoint[1] + 18)}\" fill=\"none\" stroke=\"#f3c866\" stroke-width=\"3\"/>",
            f"    <text {_attrs(x=arm_endpoint[0] + 16, y=arm_endpoint[1] + 5)} class=\"camera-label\" fill=\"#f3c866\">C2</text>",
            "  </g>",
            f"  <g id=\"camera-overview\" data-role=\"{escape(camera_roles['overview_rgb'], quote=True)}\" transform=\"translate(846 128)\">",
            "    <rect x=\"0\" y=\"0\" width=\"293\" height=\"132\" rx=\"14\" fill=\"#162737\" stroke=\"#47647b\"/>",
            "    <path d=\"M 18 17 h 24 l 9 8 v 22 h -33 z M 51 27 l 14 -8 v 27 l -14 -8 z\" fill=\"#f3c866\"/>",
            "    <text x=\"78\" y=\"29\" class=\"camera-label\" fill=\"#e8eef5\">C1 HEAD · dog forward / 狗头前视</text>",
            "    <text x=\"18\" y=\"61\" class=\"camera-label\" fill=\"#e8eef5\">C2 WRIST · above gripper, slight-down</text>",
            "    <text x=\"18\" y=\"91\" class=\"camera-label\" fill=\"#e8eef5\">C3 OVERVIEW · pulled-back observer view</text>",
            f"    <text x=\"18\" y=\"115\" class=\"muted\">C3 xyz [{', '.join(_fmt(value) for value in overview_xyz)}] m · not to scale</text>",
            "  </g>",
            "  <g id=\"world-axes\" transform=\"translate(87 730)\">",
            "    <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"-52\" stroke=\"#ff7f7f\" stroke-width=\"3\"/>",
            "    <line x1=\"0\" y1=\"0\" x2=\"-52\" y2=\"0\" stroke=\"#7fd5ff\" stroke-width=\"3\"/>",
            "    <text x=\"8\" y=\"-42\" class=\"muted\">+X / dog forward</text>",
            "    <text x=\"-58\" y=\"22\" class=\"muted\">+Y</text>",
            "  </g>",
            "  <g id=\"profile-key\" transform=\"translate(62 128)\">",
            "    <rect x=\"0\" y=\"0\" width=\"204\" height=\"76\" rx=\"13\" fill=\"#162737\" stroke=\"#47647b\"/>",
            "    <rect x=\"16\" y=\"17\" width=\"28\" height=\"15\" rx=\"4\" fill=\"#2b6bdc\" stroke=\"#8fb2cf\" stroke-dasharray=\"4 3\"/>",
            "    <text x=\"55\" y=\"30\" class=\"muted\">near bins · local place</text>",
            "    <path d=\"M 17 55 h 28\" class=\"route\"/>",
            "    <text x=\"55\" y=\"61\" class=\"muted\">remote · loaded motion</text>",
            "  </g>",
            f"  <metadata>sources: {escape(str(V1_WORKCELL_PATH.relative_to(PROJECT_ROOT)))}, {escape(str(NEAR_RECEPTACLE_PATH.relative_to(PROJECT_ROOT)))}, {escape(str(REMOTE_WORKCELL_PATH.relative_to(PROJECT_ROOT)))}, {escape(str(REMOTE_RECEPTACLE_PATH.relative_to(PROJECT_ROOT)))}</metadata>",
            "</svg>",
            "",
        ]
    )
    return "\n".join(lines)


def render_layout(output_path: Path = DEFAULT_OUTPUT_PATH) -> Path:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(), encoding="utf-8", newline="\n")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"SVG output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(render_layout(args.output))


if __name__ == "__main__":
    main()
