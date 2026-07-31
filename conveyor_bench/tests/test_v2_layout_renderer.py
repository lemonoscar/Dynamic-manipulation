from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from xml.etree import ElementTree

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "render_v2_layout.py"
CHECKED_IN_SVG = (
    PROJECT_ROOT / "docs" / "images" / "conveyorbench_v2_layout.svg"
)
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def _load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_v2_layout", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _svg_ids(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    return {
        element_id: element
        for element in root.iter()
        if (element_id := element.attrib.get("id")) is not None
    }


def test_renderer_is_deterministic_and_checked_in_svg_is_current(
    tmp_path: Path,
) -> None:
    renderer = _load_renderer()
    first = renderer.render_layout(tmp_path / "first.svg")
    second = renderer.render_layout(tmp_path / "second.svg")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == CHECKED_IN_SVG.read_bytes()


def test_renderer_cli_creates_a_valid_standalone_svg(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "layout.svg"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == str(output.resolve())

    root = ElementTree.parse(output).getroot()
    assert root.tag == f"{SVG_NAMESPACE}svg"
    assert root.attrib["viewBox"] == "0 0 1200 820"
    assert not list(root.iter(f"{SVG_NAMESPACE}image"))
    for element in root.iter():
        for attribute in ("href", "{http://www.w3.org/1999/xlink}href"):
            assert attribute not in element.attrib


def test_svg_contains_manifest_driven_layout_and_three_cameras() -> None:
    root = ElementTree.parse(CHECKED_IN_SVG).getroot()
    ids = _svg_ids(root)
    required_ids = {
        "conveyor-belt",
        "direction-left-to-right",
        "robot-start",
        "near-bin-sort_bin_blue",
        "near-bin-sort_bin_yellow",
        "remote-bin-delivery_bin_blue",
        "remote-bin-delivery_bin_yellow",
        "route-delivery_bin_blue",
        "route-delivery_bin_yellow",
        "camera-head",
        "camera-wrist",
        "camera-overview",
    }
    assert required_ids <= set(ids)

    direction = ids["direction-left-to-right"]
    assert float(direction.attrib["x1"]) < float(direction.attrib["x2"])
    assert direction.attrib["marker-end"] == "url(#arrow-cyan)"
    assert ids["camera-head"].attrib["data-role"] == "policy_observation"
    assert ids["camera-wrist"].attrib["data-role"] == "policy_observation"
    assert ids["camera-overview"].attrib["data-role"] == "observer_only"

    visible_text = " ".join(
        text.strip() for text in root.itertext() if text.strip()
    )
    assert "LEFT → RIGHT" in visible_text
    assert "8 local procedural part classes" in visible_text
    assert "minimum displacement 0.65 m" in visible_text
    assert "C1 HEAD" in visible_text
    assert "C2 WRIST" in visible_text
    assert "C3 OVERVIEW" in visible_text


def test_bin_centres_and_routes_match_manifests() -> None:
    renderer = _load_renderer()
    root = ElementTree.parse(CHECKED_IN_SVG).getroot()
    ids = _svg_ids(root)
    manifests = (
        (
            "near-bin-",
            PROJECT_ROOT / "assets" / "receptacles" / "ASSET_MANIFEST.json",
        ),
        (
            "remote-bin-",
            PROJECT_ROOT
            / "assets"
            / "receptacles"
            / "remote_delivery_v2.json",
        ),
    )

    for prefix, path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for zone in payload["receptacles"]:
            element_id = f"{prefix}{zone['zone_id']}"
            if element_id not in ids:
                continue
            outer_rect = next(ids[element_id].iter(f"{SVG_NAMESPACE}rect"))
            center_x = float(outer_rect.attrib["x"]) + float(
                outer_rect.attrib["width"]
            ) / 2.0
            center_y = float(outer_rect.attrib["y"]) + float(
                outer_rect.attrib["height"]
            ) / 2.0
            expected_x, expected_y = renderer._screen(
                float(zone["center_xyz_m"][0]),
                float(zone["center_xyz_m"][1]),
            )
            assert center_x == pytest.approx(expected_x, abs=0.02)
            assert center_y == pytest.approx(expected_y, abs=0.02)

    remote_workcell = json.loads(
        (
            PROJECT_ROOT
            / "assets"
            / "workcells"
            / "remote_delivery_v2"
            / "ASSET_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    for zone_id, goal in remote_workcell["design"][
        "delivery_root_goals"
    ].items():
        path_data = ids[f"route-{zone_id}"].attrib["d"].split()
        goal_screen = renderer._screen(float(goal[0]), float(goal[1]))
        assert float(path_data[-2]) == pytest.approx(goal_screen[0], abs=0.02)
        assert float(path_data[-1]) == pytest.approx(goal_screen[1], abs=0.02)


def test_renderer_tolerates_absent_v2_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    renderer = _load_renderer()
    monkeypatch.setattr(renderer, "V2_CONFIG_PATH", tmp_path / "missing.json")
    svg = renderer.build_svg()
    assert "remote-bin-delivery_bin_blue" in svg
    assert "near-bin-sort_bin_blue" in svg
