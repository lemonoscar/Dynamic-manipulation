from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_m0_window_booster.py"


def test_builds_hardlinked_window_subset_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    export = source / "exports" / "m0_mobile.jsonl"
    camera = source / "cameras" / "head_rgb" / "000000.png"
    export.parent.mkdir(parents=True)
    camera.parent.mkdir(parents=True)
    rows = [
        {"observation_time_s": 0.0, "sample_id": "a"},
        {"observation_time_s": 1.0, "sample_id": "b"},
        {"observation_time_s": 2.0, "sample_id": "c"},
    ]
    export.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    camera.write_bytes(b"png")
    output = tmp_path / "booster"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--episode-root",
            str(source),
            "--output-root",
            str(output),
            "--window",
            "0.75:1.25",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert export.read_text(encoding="utf-8").count("\n") == 3
    selected = [
        json.loads(line)
        for line in (output / "exports" / "m0_mobile.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["sample_id"] for row in selected] == ["b"]
    assert (source / "cameras" / "head_rgb" / "000000.png").stat().st_ino == (
        output / "cameras" / "head_rgb" / "000000.png"
    ).stat().st_ino
    manifest = json.loads(
        (output / "m0_window_booster.json").read_text(encoding="utf-8")
    )
    assert manifest["record_count"] == 1
    assert manifest["windows_s"] == [[0.75, 1.25]]
