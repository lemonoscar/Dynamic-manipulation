#!/usr/bin/env python3
"""Audit ConveyorBench assets and Python dependencies without launching Isaac Sim."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ASSET_RELATIVE_PATH = Path("assets") / "robots" / "go2_x5"
URDF_NAME = "go2_x5.urdf"
REQUIRED_MODULES = {
    "isaacsim": "Isaac Sim",
    "isaaclab": "Isaac Lab",
    "torch": "PyTorch",
    "numpy": "NumPy",
    "cv2": "OpenCV",
}
URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _element_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _collect_urdf_references(root: ET.Element) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    for element in root.iter():
        kind = _element_name(element)
        if kind not in {"mesh", "texture"}:
            continue
        filename = element.attrib.get("filename")
        if filename is not None:
            references.append((kind, filename.strip()))
    return references


def _issue(code: str, reference: str | None, message: str) -> dict[str, Any]:
    return {
        "code": code,
        "reference": reference,
        "message": message,
    }


def audit_robot_assets(project_root: Path) -> dict[str, Any]:
    """Check that every URDF asset reference is local, contained, and present."""

    asset_root = (project_root / ROBOT_ASSET_RELATIVE_PATH).resolve()
    urdf_path = asset_root / URDF_NAME
    report: dict[str, Any] = {
        "ok": False,
        "asset_root": str(asset_root),
        "urdf_path": str(urdf_path),
        "urdf_exists": urdf_path.is_file(),
        "reference_occurrences": 0,
        "unique_reference_count": 0,
        "references": [],
        "issues": [],
    }

    if not urdf_path.is_file():
        report["issues"].append(
            _issue("missing_urdf", None, f"URDF does not exist: {urdf_path}")
        )
        return report

    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as error:
        report["issues"].append(
            _issue("invalid_urdf", None, f"URDF could not be parsed: {error}")
        )
        return report

    occurrences = _collect_urdf_references(root)
    report["reference_occurrences"] = len(occurrences)
    unique_references = list(dict.fromkeys(occurrences))
    report["unique_reference_count"] = len(unique_references)

    if not unique_references:
        report["issues"].append(
            _issue("no_asset_references", None, "URDF contains no mesh or texture references")
        )

    for kind, value in unique_references:
        entry: dict[str, Any] = {
            "kind": kind,
            "value": value,
            "resolved_path": None,
            "exists": False,
            "within_asset_root": False,
            "ok": False,
        }

        if not value:
            report["issues"].append(
                _issue("empty_reference", value, f"{kind} filename is empty")
            )
            report["references"].append(entry)
            continue

        if URI_PATTERN.match(value):
            report["issues"].append(
                _issue(
                    "external_uri_reference",
                    value,
                    f"{kind} reference uses an external URI",
                )
            )
            report["references"].append(entry)
            continue

        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
            report["issues"].append(
                _issue(
                    "absolute_reference",
                    value,
                    f"{kind} reference is an absolute path",
                )
            )
            report["references"].append(entry)
            continue

        resolved_path = (urdf_path.parent / value).resolve()
        within_asset_root = resolved_path.is_relative_to(asset_root)
        exists = resolved_path.is_file()
        entry.update(
            {
                "resolved_path": str(resolved_path),
                "exists": exists,
                "within_asset_root": within_asset_root,
                "ok": within_asset_root and exists,
            }
        )

        if not within_asset_root:
            report["issues"].append(
                _issue(
                    "reference_outside_asset_root",
                    value,
                    f"{kind} reference resolves outside the robot asset directory",
                )
            )
        elif not exists:
            report["issues"].append(
                _issue(
                    "missing_local_reference",
                    value,
                    f"{kind} reference does not exist: {resolved_path}",
                )
            )
        report["references"].append(entry)

    report["ok"] = not report["issues"]
    return report


def _distribution_versions(module_name: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution_name in importlib.metadata.packages_distributions().get(
        module_name, []
    ):
        try:
            versions[distribution_name] = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def audit_python_modules() -> dict[str, Any]:
    """Discover required modules without importing them or creating a SimulationApp."""

    modules: dict[str, Any] = {}
    missing: list[str] = []
    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        available = spec is not None
        modules[module_name] = {
            "display_name": display_name,
            "available": available,
            "origin": None if spec is None else spec.origin,
            "distribution_versions": (
                _distribution_versions(module_name) if available else {}
            ),
        }
        if not available:
            missing.append(module_name)

    return {
        "ok": not missing,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "modules": modules,
        "missing_modules": missing,
        "simulation_started": False,
    }


def audit_environment(project_root: Path) -> dict[str, Any]:
    asset_report = audit_robot_assets(project_root)
    module_report = audit_python_modules()
    failed_checks: list[str] = []
    if not asset_report["ok"]:
        failed_checks.append("robot_assets")
    failed_checks.extend(
        f"python_modules.{name}" for name in module_report["missing_modules"]
    )
    return {
        "schema_version": "conveyor-bench-environment-check-v1",
        "ok": not failed_checks,
        "project_root": str(project_root.resolve()),
        "checks": {
            "robot_assets": asset_report,
            "python_modules": module_report,
        },
        "failed_checks": failed_checks,
    }


def main() -> int:
    report = audit_environment(PROJECT_ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
