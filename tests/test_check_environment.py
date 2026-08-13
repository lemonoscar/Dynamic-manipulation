import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"
SPEC = importlib.util.spec_from_file_location("check_environment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_environment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_environment
SPEC.loader.exec_module(check_environment)


def _write_urdf(project_root: Path, references: list[str]) -> Path:
    asset_root = project_root / "assets" / "robots" / "go2_x5"
    asset_root.mkdir(parents=True)
    mesh_nodes = "\n".join(
        f'<mesh filename="{reference}" />' for reference in references
    )
    urdf_path = asset_root / "go2_x5.urdf"
    urdf_path.write_text(
        f"<robot name=\"test\"><link name=\"base\">{mesh_nodes}</link></robot>",
        encoding="utf-8",
    )
    return asset_root


def test_asset_audit_accepts_existing_project_local_references(tmp_path: Path) -> None:
    asset_root = _write_urdf(
        tmp_path,
        ["./meshes/body.dae", "./meshes/finger.STL"],
    )
    (asset_root / "meshes").mkdir()
    (asset_root / "meshes" / "body.dae").write_text("mesh", encoding="utf-8")
    (asset_root / "meshes" / "finger.STL").write_text("mesh", encoding="utf-8")

    report = check_environment.audit_robot_assets(tmp_path)

    assert report["ok"] is True
    assert report["reference_occurrences"] == 2
    assert report["unique_reference_count"] == 2
    assert report["issues"] == []


def test_asset_audit_rejects_missing_and_external_references(tmp_path: Path) -> None:
    external_uri = "https:" + "//example.invalid/body.dae"
    escaping_reference = ".." + "/" + ".." + "/outside/escape.STL"
    _write_urdf(
        tmp_path,
        [
            "./meshes/missing.dae",
            external_uri,
            "/opt/external/finger.STL",
            escaping_reference,
        ],
    )

    report = check_environment.audit_robot_assets(tmp_path)

    assert report["ok"] is False
    issue_codes = {issue["code"] for issue in report["issues"]}
    assert issue_codes == {
        "missing_local_reference",
        "external_uri_reference",
        "absolute_reference",
        "reference_outside_asset_root",
    }


def test_asset_audit_reports_invalid_urdf(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets" / "robots" / "go2_x5"
    asset_root.mkdir(parents=True)
    (asset_root / "go2_x5.urdf").write_text("<robot>", encoding="utf-8")

    report = check_environment.audit_robot_assets(tmp_path)

    assert report["ok"] is False
    assert report["issues"][0]["code"] == "invalid_urdf"


def test_main_prints_json_and_returns_failure_exit_code(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        check_environment,
        "audit_environment",
        lambda project_root: {
            "schema_version": "conveyor-bench-environment-check-v1",
            "ok": False,
            "project_root": str(project_root),
            "checks": {},
            "failed_checks": ["python_modules.cv2"],
        },
    )

    exit_code = check_environment.main()
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert output["ok"] is False
